//===- remote_client_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Builds `runtime/vx_worker_main.cpp` against the CPU backend, starts it on a
//! port, and drives it from `tests/runtime/remote_client_test.cpp` over TCP
//! (#348).
//!
//! This is the first test in which a dispatch leaves the process entirely: two
//! operands staged onto a separate program, a matmul run there, the result
//! named by a handle, fetched back, and checked. What makes it possible on a
//! laptop is that the worker holds no vendor code -- what it *is* depends only
//! on what it was linked against, so a CPU worker exercises the same path a GPU
//! one will.
//!
//! Ports are chosen from the process id so two copies of the suite do not
//! collide, and the worker is killed on the way out whether the test passed or
//! not.

use std::path::PathBuf;
use std::process::{Child, Command};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Kills the worker when the test ends, including on a panic.
struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn compile(cxx: &str, sources: &[PathBuf], out: &PathBuf, extra: &[String]) {
    let mut cmd = Command::new(cxx);
    cmd.args(["-std=c++17", "-O1", "-Wall", "-Werror"]);
    for s in sources {
        cmd.arg(s);
    }
    for e in extra {
        cmd.arg(e);
    }
    cmd.arg("-o").arg(out);

    let built = cmd
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));
    assert!(
        built.status.success(),
        "compiling {:?} failed:\n{}",
        sources,
        String::from_utf8_lossy(&built.stderr)
    );
}

/// libffi is not in the default include path on macOS, and the CPU backend
/// needs it for the kernels it does not route.
fn ffi_flags() -> Vec<String> {
    let mut flags = vec!["-lffi".to_string()];
    for prefix in ["/opt/homebrew/opt/libffi", "/usr/local/opt/libffi"] {
        if PathBuf::from(prefix).join("include").is_dir() {
            flags.push(format!("-I{prefix}/include"));
            flags.push(format!("-L{prefix}/lib"));
        }
    }
    flags
}

#[test]
fn a_dispatch_reaches_a_worker_process_over_tcp() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let worker_bin = tmp.join("vx-worker-test");
    compile(
        &cxx,
        &[
            root.join("runtime/vx_worker_main.cpp"),
            root.join("runtime/host_dispatch.cpp"),
        ],
        &worker_bin,
        &ffi_flags(),
    );

    let client_bin = tmp.join("remote_client_test");
    compile(
        &cxx,
        &[root.join("tests/runtime/remote_client_test.cpp")],
        &client_bin,
        &[],
    );

    // A port nobody else in this suite is using. Topology 0 because the CPU
    // backend is the device here; worker id 2 so a handle it mints is visibly
    // not the default.
    let port = 20000 + (std::process::id() % 10000) as u16;
    let worker = Worker(
        Command::new(&worker_bin)
            .args([
                "--port",
                &port.to_string(),
                "--topology",
                "0",
                "--worker-id",
                "2",
            ])
            .spawn()
            .expect("failed to start the worker"),
    );

    // The listen call happens before the first accept, but the process still
    // has to get there.
    std::thread::sleep(std::time::Duration::from_millis(300));

    let run = Command::new(&client_bin)
        .args(["127.0.0.1", &port.to_string()])
        .output()
        .expect("failed to run the client");

    drop(worker);

    assert!(
        run.status.success(),
        "the dispatch did not survive the trip:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}

/// A Vx program placing its data on a worker, computing there, and reading the
/// answer home -- checked against the same program run with no manifest.
///
/// The test above drives the client library directly, so it proves the protocol
/// and nothing about what the compiler emits. This one compiles and runs a
/// program, which is the only way to catch the failure it was written for: a
/// `transfer` home was lowered to an allocation and a `memref.copy`, and the
/// copy read the source address directly. On one machine that address is
/// ordinary memory and the program is correct, so every local run agreed. On a
/// fleet it is a handle -- a non-canonical address naming memory in the worker's
/// process -- and the program died on a signal with no output, having never sent
/// a FETCH (#321, #348).
///
/// Loopback, so the run needs no second machine: the wire is the same, only the
/// latency is missing, and what is being tested is which messages are sent.
#[test]
fn a_placed_tensor_can_be_read_home_from_a_worker() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let worker_bin = tmp.join("vx-worker-readback");
    compile(
        &cxx,
        &[
            root.join("runtime/vx_worker_main.cpp"),
            root.join("runtime/host_dispatch.cpp"),
        ],
        &worker_bin,
        &ffi_flags(),
    );

    // Placed operands, a dispatch that leaves the result where it computed it,
    // and then the way home. `matmul_into` rather than `c = a @ b` because the
    // second publishes into a slot the worker cannot fill, and the dispatch
    // would decline to route -- making this a test of a program that stayed
    // here.
    let prog = tmp.join("readback.vx");
    std::fs::write(
        &prog,
        r#"Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}

fn main() -> i32 {
  let mut a_h = Tensor<f32>([ 8, 8 ]);
  let mut b_h = Tensor<f32>([ 8, 8 ]);
  let mut c_h = Tensor<f32>([ 8, 8 ]);
  for i in 0..8 {
    for j in 0..8 {
      a_h[i][j] = ((i + j) as f32) * 0.25;
      b_h[i][j] = ((i - j) as f32) * 0.5;
    }
  }
  let a = transfer(a_h, Memory::GPU_HBM);
  let b = transfer(b_h, Memory::GPU_HBM);
  let mut c = transfer(c_h, Memory::GPU_HBM);
  spawn on(Topology::GPU) {
    matmul_into(&mut c, &a, &b);
  }
  let home = transfer(c, Memory::CPU_DRAM);
  print(home[0][0]);
  return 0;
}
"#,
    )
    .expect("failed to write the program");

    let vxc = env!("CARGO_BIN_EXE_vxc");
    let local = Command::new(vxc)
        .args([prog.to_str().unwrap(), "--run"])
        .output()
        .expect("failed to run the program locally");
    assert!(
        local.status.success(),
        "the local run failed:\n{}",
        String::from_utf8_lossy(&local.stderr)
    );
    let expected = String::from_utf8_lossy(&local.stdout)
        .lines()
        .last()
        .unwrap_or_default()
        .trim()
        .to_string();
    assert!(!expected.is_empty(), "the local run printed nothing");

    let port = 20000 + ((std::process::id() + 1) % 10000) as u16;
    let manifest = tmp.join("readback-manifest");
    std::fs::write(&manifest, format!("GPU[0]  127.0.0.1  {port}\n"))
        .expect("failed to write the manifest");

    let worker = Worker(
        Command::new(&worker_bin)
            .args([
                "--port",
                &port.to_string(),
                "--topology",
                "0",
                "--worker-id",
                "2",
            ])
            .spawn()
            .expect("failed to start the worker"),
    );
    std::thread::sleep(std::time::Duration::from_millis(300));

    // VX_FLEET_STRICT, so a placement the manifest does not name is an error
    // rather than a quiet local run -- without it this test would still pass
    // with the worker switched off, and prove nothing.
    let remote = Command::new(vxc)
        .args([prog.to_str().unwrap(), "--run"])
        .env("VX_FLEET_MANIFEST", &manifest)
        .env("VX_FLEET_STRICT", "1")
        .output()
        .expect("failed to run the program against the worker");
    drop(worker);

    assert!(
        remote.status.success(),
        "the program died reading its result home:\n{}{}",
        String::from_utf8_lossy(&remote.stdout),
        String::from_utf8_lossy(&remote.stderr)
    );
    let got = String::from_utf8_lossy(&remote.stdout)
        .lines()
        .last()
        .unwrap_or_default()
        .trim()
        .to_string();
    assert_eq!(
        got,
        expected,
        "the worker's answer is not the local one:\n{}{}",
        String::from_utf8_lossy(&remote.stdout),
        String::from_utf8_lossy(&remote.stderr)
    );
}

/// A backend handed an address on another machine says so, rather than reading
/// it.
///
/// The guard behind this exists because of how the read-back failure presented:
/// SIGSEGV, empty output, a backtrace naming the signal handler and nothing
/// else. Everything needed to explain it -- that the address is a handle, which
/// worker minted it, which topology was asked -- is known at the point of the
/// fault and was not being said.
///
/// Both directions are checked. A guard that refused ordinary pointers too
/// would abort every local run, since these are the same entry points a
/// single-machine program uses for every allocation it frees.
#[test]
fn a_backend_refuses_an_address_that_belongs_to_another_machine() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let bin = tmp.join("routing_refusal_test");
    compile(
        &cxx,
        &[root.join("tests/runtime/routing_refusal_test.cpp")],
        &bin,
        &[format!("-I{}", root.join("runtime").display())],
    );

    let ok = Command::new(&bin)
        .arg("pointer")
        .output()
        .expect("failed to run the refusal test");
    assert!(
        ok.status.success(),
        "an ordinary pointer was refused:\n{}",
        String::from_utf8_lossy(&ok.stderr)
    );

    let refused = Command::new(&bin)
        .arg("handle")
        .output()
        .expect("failed to run the refusal test");
    assert!(
        !refused.status.success(),
        "a handle was accepted by a local path"
    );
    let said = String::from_utf8_lossy(&refused.stderr);
    assert!(
        said.contains("worker 3") && said.contains("a read-back"),
        "the refusal did not name the worker and the operation:\n{said}"
    );
}
