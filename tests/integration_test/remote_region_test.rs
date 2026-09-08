//===- remote_region_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/remote_region_test.cpp`, which checks how a
//! plugin names memory that lives on another machine (#348).
//!
//! The naming lives in a header rather than in a backend for the same reason
//! the GEMM decode does: every failure it guards against is silent. A handle
//! that resolves into the wrong region runs a dispatch against the wrong tensor
//! and returns plausible numbers; a stale one reads whatever was allocated
//! after it; one mistaken for a host pointer sends the plugin off to stage
//! memory that is not there. None of them fault and none appear in a dispatch
//! trace, so they have to be caught on a machine with no GPU and no network.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_remote_handle_resolves_to_the_buffer_it_names() {
    let root = repo_root();
    let source = root.join("tests/runtime/remote_region_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("remote_region_test");
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
