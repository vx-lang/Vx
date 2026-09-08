//===- gemm_plan_test.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/gemm_plan_test.cpp`, which checks how a
//! plugin decodes a dispatch into the matrix multiply to run (#325).
//!
//! The decode lives in a header rather than in the CUDA backend precisely so it
//! can be tested here, with no GPU: it decides which operand a vendor GEMM
//! reads and which it writes, and a mistake there produces a plausible matrix
//! of wrong numbers rather than a failure. Discovering that on rented hardware
//! is the expensive way to find out.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn dispatch_decodes_to_the_gemm_the_compiler_described() {
    let root = repo_root();
    let source = root.join("tests/runtime/gemm_plan_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("gemm_plan_test");
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
        .expect("failed to run the compiled plan test");

    assert!(
        run.status.success(),
        "plan decode checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
