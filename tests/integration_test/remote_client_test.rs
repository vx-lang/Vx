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
