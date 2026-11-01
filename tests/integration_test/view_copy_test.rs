//===- view_copy_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/view_copy_test.cpp`. It checks how the CUDA
//! plugin copies tensor views to the host and back, with no GPU.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn views_that_share_memory_share_one_copy() {
    let root = repo_root();
    let source = root.join("tests/runtime/view_copy_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("view_copy_test");
    let cxx = runtime_test_build::cxx();

    let build = runtime_test_build::command(&cxx, "c++17")
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
        .expect("failed to run the compiled view copy test");

    assert!(
        run.status.success(),
        "view copy checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
