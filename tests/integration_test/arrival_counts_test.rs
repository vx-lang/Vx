//===- arrival_counts_test.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/arrival_counts_test.cpp`, which checks the
//! records of the counts that a program keeps between launches of a kernel
//! with code after a nest of blocks.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

#[test]
fn counts_are_kept_for_the_next_launch_without_being_shared() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/runtime/arrival_counts_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("arrival_counts_test");
    let cxx = runtime_test_build::cxx();

    let build = runtime_test_build::command(&cxx, "c++17")
        .arg("-pthread")
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
        .expect("failed to run the compiled arrival count test");

    assert!(
        run.status.success(),
        "arrival count checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
