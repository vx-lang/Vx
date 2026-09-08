//===- wire_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/wire_test.cpp`, which checks the bytes a
//! dispatch becomes on its way to another machine (#348).
//!
//! Half of it is round trips, pinning the format against the ABI tags that
//! decide it. The other half is input that is wrong, which on a socket is the
//! normal case rather than the exceptional one: a truncated message, a length
//! field longer than the bytes present, a rank that would index past a fixed
//! array. Removing the rank bound turns the last of those into a
//! stack-buffer-overflow that AddressSanitizer catches, which is why the check
//! is there and why this runs where a sanitizer can reach it.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_dispatch_survives_the_wire() {
    let root = repo_root();
    let source = root.join("tests/runtime/wire_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("wire_test");
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
