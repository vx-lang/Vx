//===- ane_device_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Drives scripts/tools/ane_device_check.py, which asks CoreML where it will place
// each of a set of graphs and asserts the answer.
//
// This exists because no `.vx` test can make the claim. A backend test sees the
// numbers a kernel produced, and the same numbers come back whether the work ran
// on the Neural Engine, on a CoreML CPU path, or on the host shim -- which is
// how three earlier `ane_*.vx` tests passed for months while the ANE path was
// broken. The device evidence has to come from `MLComputePlan`, and the script
// is where it does.
//
// Skips rather than fails when the toolchain is absent: the check needs macOS,
// a coremltools that can build a model, and `coremlc` from a full Xcode. A
// Linux CI box has none of those and should not report a failure for it.

use std::path::PathBuf;
use std::process::Command;

/// The interpreters that might have a working coremltools, in preference order.
/// `venv-ane` first: the checkout keeps coremltools there, and the bare
/// `python3` on PATH generally cannot build a model.
fn python_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("VX_PYTHON") {
        v.push(p);
    }
    for p in [
        "venv-ane/bin/python3",
        "venv/bin/python3",
        "python3",
        "python3.13",
        "python3.12",
        "python3.11",
    ] {
        v.push(p.to_string());
    }
    v
}

/// An interpreter counts only if it can import the *compiled* half of
/// coremltools. The pure-Python half installs on an interpreter that ships no
/// wheel for it, so `import coremltools` succeeds and the model build dies much
/// later with `BlobWriter not loaded`.
fn usable_python(root: &PathBuf) -> Option<String> {
    python_candidates().into_iter().find(|p| {
        Command::new(p)
            .current_dir(root)
            .args(["-c", "import coremltools, coremltools.libmilstoragepython"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

#[test]
fn coreml_places_these_graphs_where_we_measured() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script = root.join("scripts/tools/ane_device_check.py");
    assert!(script.is_file(), "missing {}", script.display());

    let Some(python) = usable_python(&root) else {
        eprintln!("skipping: no interpreter here can build a CoreML model");
        return;
    };

    let out = Command::new(&python)
        .current_dir(&root)
        .arg(&script)
        .output()
        .expect("failed to run the ANE device check");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    // 77 is the script's "toolchain absent" code, which is a skip and not a
    // failure -- a machine without Xcode cannot answer the question.
    if out.status.code() == Some(77) {
        eprintln!("skipping: {}", stdout.trim());
        return;
    }

    // Print the verdict, so a run with --nocapture shows what was checked rather
    // than only that something passed.
    if let Some(last) = stdout.lines().rfind(|l| !l.trim().is_empty()) {
        eprintln!("ANE device check: {last}");
    }

    assert!(
        out.status.success(),
        "CoreML no longer places these graphs where they were measured.\n\
         This is a real signal rather than a flaky test: it means the device \
         assignment changed under a new OS, Xcode or coremltools, and anything \
         claiming Neural Engine execution needs re-checking.\n\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
}

/// The dispatcher must not claim hardware it has not asked about.
///
/// `--- EXECUTING ON APPLE NEURAL ENGINE ---` printed whenever the dispatcher
/// took the CoreML route, which says nothing about what CoreML then did with the
/// work. For the primitives shipped at the time it did the work on the CPU, so
/// the one line in the system that appeared to attest Neural Engine execution
/// was asserting hardware nobody had checked -- and a paper or a README citing
/// it would have been citing that.
///
/// The line is now derived from `MLComputePlan`, so it reports the device CoreML
/// picked rather than the route the dispatcher took. This pins both halves: the
/// measured line appears, and the unmeasured claim does not come back.
// The binary under test comes from CARGO_BIN_EXE_vxc, which cargo points at the
// profile the suite is running in. A hardcoded `target/debug/vxc` is absent under
// `cargo test --release`, and these tests skipped there rather than failing -- so
// they read as passing in a job that never ran them.
#[test]
fn the_dispatcher_reports_the_device_coreml_chose() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vxc = PathBuf::from(env!("CARGO_BIN_EXE_vxc"));
    let program = root.join("tests/backend/pass/ane_matmul_f16.vx");
    assert!(program.is_file(), "missing {}", program.display());

    let out = Command::new(&vxc)
        .current_dir(&root)
        .arg(&program)
        .env("VX_DISPATCH_VERBOSE", "1")
        .output()
        .expect("failed to run vxc");
    let log = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // The rule is about every route, and the log of one program can only speak
    // for the routes that program took. This one drives the f16 GEMM path and
    // never reaches the affine path, so asserting the absence of the old banner
    // here would pass whether or not that path still printed it -- which is
    // exactly what it did when this test was first written. The source is where
    // the claim can be checked for all of them at once.
    let dispatcher = std::fs::read_to_string(root.join("runtime/npu_dispatch.mm"))
        .expect("failed to read runtime/npu_dispatch.mm");
    assert!(
        !dispatcher.contains("EXECUTING ON APPLE NEURAL ENGINE"),
        "a dispatch route announces the Neural Engine without asking CoreML \
         which device it chose; the banner has to come from the compute plan"
    );

    // The positive half needs the route to have fired, and "it did not fire" has two
    // very different causes. With no compiled primitive the program falls back to the
    // CPU shim and there is nothing to report, which is a missing fixture. With the
    // primitive present, a route that does not fire is the regression this test is
    // for -- so the model on disk decides which of the two this is, rather than the
    // absence of a log line skipping the check either way.
    let primitive = root.join("ane-primitives/matmul_512x512_fp16.mlmodelc");
    let fired = log.contains("Recognised GEMM 512x512x512 f16");
    if !fired && !primitive.is_dir() {
        eprintln!(
            "skipping the placement check: no {} was built",
            primitive.display()
        );
        return;
    }
    assert!(
        fired,
        "{} exists, so the f16 GEMM route should have taken it; the dispatcher \
         fell back instead:\n{log}",
        primitive.display()
    );
    assert!(
        log.contains("CoreML plans matmul_512x512_fp16.mlmodelc on the Neural Engine"),
        "the f16 512 primitive is the one CoreML puts on the Neural Engine, and \
         the dispatcher must say so from the plan rather than from the route:\n{log}"
    );
}

/// A kernel placed on the Neural Engine says when it runs on the CPU, and
/// `VX_ENABLE_CPU_FALLBACK=0` stops the program instead.
///
/// The program runs from an empty directory, so the CoreML model of its f16
/// matmul is not on the search path. The matmul runs twice. A region on
/// `Topology::CpuNeon` runs first. The program places it on the CPU, so the
/// switch must not stop it.
#[cfg(target_os = "macos")]
#[test]
fn a_neural_engine_kernel_with_no_model_says_it_runs_on_the_cpu() {
    const PROGRAM: &str = r#"
fn main() -> i32 {
  let mut d = Tensor<f32, [4]>::new();
  spawn on(Topology::CpuNeon) {
    for i in 0..4 {
      d[i] = 1.5;
    }
  }
  print(d[0]);
  print(" ");
  let mut c = Tensor<f16, [512, 512], Memory::NPU_HBM>::new();
  for _k in 0..2 {
    let mut a = Tensor<f16, [512, 512], Memory::NPU_HBM>::new();
    let mut b = Tensor<f16, [512, 512], Memory::NPU_HBM>::new();
    for i in 0..512 {
      for j in 0..512 {
        a[i][j] = 2.0;
        b[i][j] = 1.0;
      }
    }
    spawn on(Topology::NPU[0]) {
      c = a @ b;
    }
  }
  print(c[0][0]);
  return 0;
}
"#;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let vxc = PathBuf::from(env!("CARGO_BIN_EXE_vxc"));
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("a_neural_engine_kernel_with_no_model");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to make the test directory");
    let program = dir.join("two_matmuls.vx");
    std::fs::write(&program, PROGRAM).expect("failed to write the program");

    // Runs the program in `cwd`, and gives whether it succeeded, its stdout and its stderr.
    let run = |cwd: &PathBuf, fallback: Option<&str>| {
        let mut cmd = Command::new(&vxc);
        cmd.current_dir(cwd)
            .arg(&program)
            .env_remove("VX_ANE_MODEL_DIR")
            .env_remove("VX_DISPATCH_VERBOSE")
            .env_remove("VX_ENABLE_CPU_FALLBACK");
        if let Some(value) = fallback {
            cmd.env("VX_ENABLE_CPU_FALLBACK", value);
        }
        let out = cmd.output().expect("failed to run vxc");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };

    let (ok, stdout, stderr) = run(&dir, None);
    assert!(
        ok && stdout.contains("1.5 1024"),
        "the program failed:\n{stdout}{stderr}"
    );
    let notes: Vec<&str> = stderr
        .lines()
        .filter(|l| {
            l.starts_with("[Vx Dispatcher] the program places vx_npu_kernel_")
                && l.contains(
                    " on Topology::NPU[0], but it runs on the CPU: \
                     no matmul_512x512_fp16.mlmodelc in ",
                )
        })
        .collect();
    assert_eq!(
        notes.len(),
        1,
        "the kernel ran twice, and the dispatcher must say once that it ran on the CPU:\n{stderr}"
    );
    assert!(
        notes[0].contains(&format!("{}/ane-primitives", dir.display())),
        "the line must name the directories that the dispatcher searched:\n{stderr}"
    );

    let (ok, stdout, stderr) = run(&dir, Some("0"));
    assert!(
        !ok,
        "VX_ENABLE_CPU_FALLBACK=0 did not stop the program:\n{stdout}{stderr}"
    );
    assert!(
        stdout.contains("1.5") && !stdout.contains("1024"),
        "the region on Topology::CpuNeon must run, and the matmul must not:\n{stdout}{stderr}"
    );
    assert!(
        stderr.contains("FATAL: the program places vx_npu_kernel_")
            && stderr.contains(
                " on Topology::NPU[0], but only the CPU can run it here: \
                 no matmul_512x512_fp16.mlmodelc in "
            ),
        "the dispatcher must name the kernel and the reason when it stops:\n{stderr}"
    );

    // From the checkout, the model is on the search path, so CoreML runs the
    // matmul and the dispatcher says nothing.
    if root
        .join("ane-primitives/matmul_512x512_fp16.mlmodelc")
        .is_dir()
    {
        let (ok, stdout, stderr) = run(&root, None);
        assert!(
            ok && stdout.contains("1.5 1024"),
            "the program failed:\n{stdout}{stderr}"
        );
        assert!(
            !stderr.contains("runs on the CPU"),
            "CoreML ran the matmul, but the dispatcher said it ran on the CPU:\n{stderr}"
        );
    }
}
