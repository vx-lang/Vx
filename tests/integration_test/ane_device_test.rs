//===- ane_device_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Drives scripts/ane_device_check.py, which asks CoreML where it will place
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
    let script = root.join("scripts/ane_device_check.py");
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
