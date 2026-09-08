//===- loopback_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/loopback_test.cpp`, which puts a whole remote
//! dispatch through its paces in one process (#348): stage a buffer, encode a
//! dispatch, decode it on the far side, resolve the handles, run the GEMM the
//! compiler described, publish the result, encode the reply, decode it back,
//! and check the numbers.
//!
//! This is everything that would otherwise first run on two rented machines,
//! and every failure it looks for is silent. A handle resolved to the wrong
//! backing pointer computes the wrong product and returns it. A slot published
//! but never collected leaves the caller with a descriptor of zeroes. Swapping
//! the operands -- which is what `roles=` exists to prevent -- yields a
//! plausible matrix rather than an error.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_dispatch_runs_on_the_far_side_and_comes_back() {
    let root = repo_root();
    let source = root.join("tests/runtime/loopback_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("loopback_test");
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
