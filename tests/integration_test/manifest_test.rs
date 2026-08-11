//===- manifest_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/manifest_test.cpp`, which turns the
//! `toponame=DecodeWorker` a dispatch carries into a machine (#348).
//!
//! The rule under test is what the location-transparency claim rests on: a name
//! that is absent means *local*. A program with no manifest runs entirely on
//! this machine, and adding an entry is what distributes it, which is why
//! `llama2.vx` is the same text either way.
//!
//! That rule is also why a malformed line is refused rather than skipped.
//! Skipping one leaves a worker silently local -- the program runs, the numbers
//! are right, and the distribution it was meant to demonstrate did not happen,
//! with no error anywhere to notice afterwards.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_worker_name_resolves_to_a_machine() {
    let root = repo_root();
    let source = root.join("tests/runtime/manifest_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("manifest_test");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let build = Command::new(&cxx)
        .args(["-std=c++17", "-O1", "-Wall", "-Werror"])
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {cxx}: {e}"));

    assert!(
        build.status.success(),
        "compiling {} failed:\n{}",
        source.display(),
        String::from_utf8_lossy(&build.stderr)
    );

    let run = Command::new(&binary)
        .output()
        .expect("failed to run the compiled region test");

    assert!(
        run.status.success(),
        "remote region checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
