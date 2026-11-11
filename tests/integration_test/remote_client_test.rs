//===- remote_client_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command};

use crate::integration_test::runtime_test_build;

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
    let mut cmd = runtime_test_build::command(cxx, "c++17");
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

/// Run a command, killing it if it outlives `secs`.
///
/// A fleet program that cannot make progress does not fail, it waits: the
/// connection pool's deadlock left the program blocked on a socket the worker
/// would never accept, with nothing on either side to time out. `output()` would
/// then hang the whole suite rather than failing one test, which is the
/// difference between a red run and a run nobody can interpret.
fn run_bounded(cmd: &mut Command, secs: u64) -> std::process::Output {
    use std::process::Stdio;
    let child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to start the command");
    let pid = child.id();
    let finished = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let watching = finished.clone();
    std::thread::spawn(move || {
        for _ in 0..(secs * 10) {
            if watching.load(std::sync::atomic::Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
    });
    let out = child.wait_with_output().expect("failed to collect output");
    finished.store(true, std::sync::atomic::Ordering::Relaxed);
    out
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
    let cxx = runtime_test_build::cxx();

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
    let mut worker = Worker(
        Command::new(&worker_bin)
            .args([
                "--port",
                &port.to_string(),
                "--topology",
                "0",
                "--worker-id",
                "2",
            ])
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to start the worker"),
    );

    // The listen call happens before the first accept, but the process still
    // has to get there.
    std::thread::sleep(std::time::Duration::from_millis(300));

    // The client does not free its two input buffers. The worker must free them
    // when the first client disconnects, before it accepts the second client.
    for attempt in ["first", "second"] {
        let run = Command::new(&client_bin)
            .args(["127.0.0.1", &port.to_string()])
            .output()
            .expect("failed to run the client");
        assert!(
            run.status.success(),
            "the dispatch of the {attempt} client did not survive the trip:\n{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
    }

    let mut stderr = worker.0.stderr.take().expect("the worker has no stderr");
    drop(worker);
    let mut log = String::new();
    stderr
        .read_to_string(&mut log)
        .expect("failed to read the worker's log");
    assert!(
        log.contains("freed 2 buffer(s) that the host left"),
        "the worker did not free the buffers of the first client:\n{log}"
    );
}

/// Two hosts use two workers in opposite order, and both finish. Host A holds
/// worker 1 and then asks worker 2, while host B holds worker 2 and then asks
/// worker 1. A worker that served one host until it disconnected left both
/// hosts waiting forever.
#[test]
fn two_hosts_that_use_two_workers_in_opposite_order_both_finish() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = runtime_test_build::cxx();

    let worker_bin = tmp.join("vx-worker-two-hosts");
    compile(
        &cxx,
        &[
            root.join("runtime/vx_worker_main.cpp"),
            root.join("runtime/host_dispatch.cpp"),
        ],
        &worker_bin,
        &ffi_flags(),
    );
    let client_bin = tmp.join("two_workers_client");
    compile(
        &cxx,
        &[root.join("tests/runtime/two_workers_client.cpp")],
        &client_bin,
        &[],
    );

    let port_1 = 20000 + ((std::process::id() + 4) % 10000) as u16;
    let port_2 = 20000 + ((std::process::id() + 5) % 10000) as u16;
    let mut workers = Vec::new();
    for (port, id) in [(port_1, "1"), (port_2, "2")] {
        workers.push(Worker(
            Command::new(&worker_bin)
                .args([
                    "--port",
                    &port.to_string(),
                    "--topology",
                    "0",
                    "--worker-id",
                    id,
                ])
                .spawn()
                .expect("failed to start a worker"),
        ));
    }
    std::thread::sleep(std::time::Duration::from_millis(400));

    // Each host makes its file when it holds its first worker.
    let file_a = tmp.join("two-hosts-a-ready");
    let file_b = tmp.join("two-hosts-b-ready");
    let _ = std::fs::remove_file(&file_a);
    let _ = std::fs::remove_file(&file_b);
    let start = |first: u16, second: u16, mine: &PathBuf, other: &PathBuf| {
        Command::new(&client_bin)
            .args([
                "127.0.0.1",
                &first.to_string(),
                &second.to_string(),
                mine.to_str().unwrap(),
                other.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to start a host")
    };
    let mut hosts = [
        ("A", start(port_1, port_2, &file_a, &file_b)),
        ("B", start(port_2, port_1, &file_b, &file_a)),
    ];

    // A deadlock does not fail, it waits. So stop the hosts after 30 s.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while hosts
        .iter_mut()
        .any(|(_, h)| h.try_wait().expect("failed to poll a host").is_none())
    {
        if std::time::Instant::now() > deadline {
            for (_, h) in hosts.iter_mut() {
                let _ = h.kill();
            }
            panic!("the two hosts did not finish in 30 s: a deadlock across the two workers");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(workers);
    for (name, h) in hosts {
        let out = h.wait_with_output().expect("failed to collect a host");
        assert!(
            out.status.success(),
            "host {name} failed:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// One host gets stuck in the middle of a message in `mode`. The worker must
/// drop it after its time limit of 2 s, and then serve a second host.
fn a_stuck_host_does_not_stop_the_worker(mode: &str, logged: &str) {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = runtime_test_build::cxx();

    let worker_bin = tmp.join(format!("vx-worker-stuck-{mode}"));
    compile(
        &cxx,
        &[
            root.join("runtime/vx_worker_main.cpp"),
            root.join("runtime/host_dispatch.cpp"),
        ],
        &worker_bin,
        &ffi_flags(),
    );
    let stuck_bin = tmp.join(format!("stuck_host-{mode}"));
    compile(
        &cxx,
        &[root.join("tests/runtime/stuck_host.cpp")],
        &stuck_bin,
        &[],
    );
    let client_bin = tmp.join(format!("remote_client_after_{mode}"));
    compile(
        &cxx,
        &[root.join("tests/runtime/remote_client_test.cpp")],
        &client_bin,
        &[],
    );

    let offset = if mode == "half" { 6 } else { 7 };
    let port = 20000 + ((std::process::id() + offset) % 10000) as u16;
    let mut worker = Worker(
        Command::new(&worker_bin)
            .args([
                "--port",
                &port.to_string(),
                "--topology",
                "0",
                "--io-timeout",
                "2",
            ])
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to start the worker"),
    );
    std::thread::sleep(std::time::Duration::from_millis(300));

    let mut stuck = Worker(
        Command::new(&stuck_bin)
            .args(["127.0.0.1", &port.to_string(), mode])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("failed to start the stuck host"),
    );
    let mut said = [0u8; 6];
    stuck
        .0
        .stdout
        .take()
        .expect("the stuck host has no stdout")
        .read_exact(&mut said)
        .expect("the stuck host did not get stuck");
    assert_eq!(&said, b"stuck\n");

    let run = run_bounded(
        Command::new(&client_bin).args(["127.0.0.1", &port.to_string()]),
        20,
    );
    assert!(
        run.status.success(),
        "the second host did not finish while the first host was stuck:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    drop(stuck);
    let mut stderr = worker.0.stderr.take().expect("the worker has no stderr");
    drop(worker);
    let mut log = String::new();
    stderr
        .read_to_string(&mut log)
        .expect("failed to read the worker's log");
    assert!(
        log.contains(logged),
        "the worker did not drop the stuck host for its time limit:\n{log}"
    );
}

#[test]
fn a_host_that_sends_half_a_request_does_not_stop_the_worker() {
    a_stuck_host_does_not_stop_the_worker(
        "half",
        "the host sent part of a request and then nothing for 2 s; dropping the host",
    );
}

#[test]
fn a_host_that_does_not_read_its_reply_does_not_stop_the_worker() {
    a_stuck_host_does_not_stop_the_worker(
        "no-read",
        "the host did not read its reply for 2 s; dropping the host",
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
    let cxx = runtime_test_build::cxx();

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

/// Data produced on one device, handed to a second, and read home -- the
/// disaggregated shape, with each device a separate worker process.
///
/// Four separate defects sat on this path, and each one is why the next was
/// invisible:
///
///   the handoff       `transfer(x, Memory::B)` where x is already on device A
///                     lowered to `alloc_and_transfer`, whose source argument is
///                     read as host memory. `vx_plugin_transfer_peer` exists in
///                     every backend, is wired to the routing, and had no caller
///                     in generated code (#347).
///   the pool          keyed connections by worker *name*. One machine serving
///                     two names -- a memory space and a topology, which is what
///                     a device is -- opened two sockets to a worker that serves
///                     one connection at a time, and the second blocked forever.
///   the staging       the fetch from A and the transfer to B shared one buffer,
///                     so the second built its header over the bytes the first
///                     had just landed. Message counts stayed exactly right and
///                     every byte of the result was zero.
///   the build         the fleet headers were listed for cargo on one platform
///                     only, so on the other a fix to them rebuilt nothing.
///
/// The third is why this test compares the answer rather than counting
/// messages: the counts were correct while the data was gone.
#[test]
fn a_handoff_between_two_workers_moves_the_data() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = runtime_test_build::cxx();

    let worker_bin = tmp.join("vx-worker-handoff");
    compile(
        &cxx,
        &[
            root.join("runtime/vx_worker_main.cpp"),
            root.join("runtime/host_dispatch.cpp"),
        ],
        &worker_bin,
        &ffi_flags(),
    );

    // Two declared memories and two topologies, because a device is both: data
    // lands in a memory space and work is placed on a topology, and a manifest
    // has to name each. `Memory::GPU_HBM` cannot express this -- it maps to
    // topology 500 whatever the program meant -- so a second device's memory
    // has to be declared to be named.
    let prog = tmp.join("handoff.vx");
    std::fs::write(
        &prog,
        r#"Memory CPU_DRAM {}
Memory HBM_A {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Memory HBM_B {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Topology DevA { memory: Memory::HBM_A, visible: [ Memory::HBM_A ] }
Topology DevB { memory: Memory::HBM_B, visible: [ Memory::HBM_B ] }

fn main() -> i32 {
  let mut a_h = Tensor<f32>([ 16, 16 ]);
  let mut b_h = Tensor<f32>([ 16, 16 ]);
  let mut c_h = Tensor<f32>([ 16, 16 ]);
  for i in 0..16 {
    for j in 0..16 {
      a_h[i][j] = ((i + j) as f32) * 0.125;
      b_h[i][j] = ((i - j) as f32) * 0.25;
    }
  }
  let a = transfer(a_h, Memory::HBM_A);
  let b = transfer(b_h, Memory::HBM_A);
  let mut c = transfer(c_h, Memory::HBM_A);
  spawn on(Topology::DevA) {
    matmul_into(&mut c, &a, &b);
  }
  let c_b = transfer(c, Memory::HBM_B);
  let home = transfer(c_b, Memory::CPU_DRAM);
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

    let port_a = 20000 + ((std::process::id() + 2) % 10000) as u16;
    let port_b = 20000 + ((std::process::id() + 3) % 10000) as u16;
    let manifest = tmp.join("handoff-manifest");
    std::fs::write(
        &manifest,
        format!(
            "HBM_A  127.0.0.1  {port_a}\n\
             DevA   127.0.0.1  {port_a}\n\
             HBM_B  127.0.0.1  {port_b}\n\
             DevB   127.0.0.1  {port_b}\n"
        ),
    )
    .expect("failed to write the manifest");

    let mut workers = Vec::new();
    for (port, id) in [(port_a, "1"), (port_b, "2")] {
        workers.push(Worker(
            Command::new(&worker_bin)
                .args([
                    "--port",
                    &port.to_string(),
                    "--topology",
                    "0",
                    "--worker-id",
                    id,
                ])
                .spawn()
                .expect("failed to start a worker"),
        ));
    }
    std::thread::sleep(std::time::Duration::from_millis(400));

    let remote = run_bounded(
        Command::new(vxc)
            .args([prog.to_str().unwrap(), "--run"])
            .env("VX_FLEET_MANIFEST", &manifest)
            .env("VX_FLEET_STRICT", "1"),
        90,
    );
    drop(workers);

    assert!(
        remote.status.success(),
        "the handoff did not complete:\n{}{}",
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
        "the data did not survive the handoff:\n{}{}",
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
    let cxx = runtime_test_build::cxx();

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

/// The machine description and the kernel of a GPU that shares host memory, with its
/// `offload:` field. Without the mark, CPU_DRAM is `managed: cached`, so the kernel reads the
/// host tensors with no `transfer`. With it, the program moves them with `transfer`.
fn shared_gpu_program(offload: bool) -> String {
    let machine = format!(
        "Memory CPU_DRAM {{\n  capacity: 480 GiB, bandwidth: 500 GB/s, managed: cached\n}}\n\
         Memory HBM {{\n  capacity: 96 GiB, bandwidth: 4 TB/s, managed: explicit, scope: device\n}}\n\
         Topology SharedGpu {{\n  arch: nvptx64,\n  levels: [blocks(max_lanes: 1024, memory: cta), \
         subgroups(lanes: 32), lanes(memory: thread)],\n  chip: sm_90,\n  offload: {offload},\n  \
         memory: Memory::HBM,\n  visible: [Memory::HBM],\n  \
         transfer Memory::CPU_DRAM -> Memory::HBM : 450 GB/s,\n  \
         transfer Memory::HBM -> Memory::CPU_DRAM : 450 GB/s\n}}\n"
    );
    let kernel = if offload {
        "fn main() -> i32 {\n  let x = transfer(Tensor<f32, [4]>::fill(3.0), Memory::HBM);\n  \
         let mut out = transfer(Tensor<f32, [4]>::fill(0.0), Memory::HBM);\n  \
         spawn on(Topology::SharedGpu) {\n    for i in 0..4 across blocks {\n      \
         out[i] = 2.0 * x[i];\n    }\n  }\n  let home = transfer(out, Memory::CPU_DRAM);\n  \
         println!(\"{}\", home[0]);\n  return 0;\n}\n"
    } else {
        "fn double(out : &mut Tensor<f32, [?]>, x : &Tensor<f32, [?]>, n : i32) -> void {\n  \
         spawn on(Topology::SharedGpu) {\n    for i in 0..n across blocks {\n      \
         out[i] = 2.0 * x[i];\n    }\n  }\n  return;\n}\n\
         fn main() -> i32 {\n  let x = Tensor<f32, [4]>::fill(3.0);\n  \
         let mut out = Tensor<f32, [4]>::fill(0.0);\n  double(&mut out, &x, 4);\n  \
         println!(\"{}\", out[0]);\n  return 0;\n}\n"
    };
    machine + kernel
}

/// A manifest that sends the topology to a worker of another node. Without `offload: true`, the
/// runtime refuses: the kernel reads host memory that such a worker does not share. With it, the
/// kernel goes to the worker, and the run stops only because no worker listens there. Port 1 is
/// a port where nothing listens, and the refusal comes before any connection.
#[test]
fn a_remote_worker_takes_only_a_topology_marked_offload() {
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let manifest = tmp.join("offload-manifest");
    std::fs::write(&manifest, "SharedGpu  127.0.0.1  1\n").expect("failed to write the manifest");
    let vxc = env!("CARGO_BIN_EXE_vxc");
    let refusal = "does not say `offload: true`";
    // Each code generator carries the mark to the payload of the kernel.
    for (offload, codegen) in [
        (false, None),
        (true, None),
        (false, Some("--legacy-codegen")),
    ] {
        let prog = tmp.join(format!("offload-{offload}.vx"));
        std::fs::write(&prog, shared_gpu_program(offload)).expect("failed to write the program");
        let run = run_bounded(
            Command::new(vxc)
                .args([prog.to_str().unwrap(), "--run"])
                .args(codegen)
                .env("VX_FLEET_MANIFEST", &manifest),
            120,
        );
        let stderr = String::from_utf8_lossy(&run.stderr);
        assert!(
            !run.status.success(),
            "the run with offload: {offload} worked:\n{stderr}"
        );
        if offload {
            assert!(
                !stderr.contains(refusal) && stderr.contains("cannot reach SharedGpu"),
                "the marked topology did not go to the worker:\n{stderr}"
            );
        } else {
            assert!(
                stderr.contains(refusal),
                "the runtime sent a topology with no mark to the worker:\n{stderr}"
            );
        }
    }
}

/// A lost machine or link sends no EOF, so a host connection turns on TCP
/// keepalive with VX_REMOTE_LINK_TIMEOUT as the limit. Losing a link needs
/// root, so the test checks the options on the socket.
#[test]
fn a_host_connection_watches_its_link() {
    let root = repo_root();
    let tmp = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let cxx = runtime_test_build::cxx();
    let bin = tmp.join("link_watch_test");
    compile(
        &cxx,
        &[root.join("tests/runtime/link_watch_test.cpp")],
        &bin,
        &[],
    );
    let run = run_bounded(&mut Command::new(&bin), 20);
    assert!(
        run.status.success(),
        "the host connection does not watch its link:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
