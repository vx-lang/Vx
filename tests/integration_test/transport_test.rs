//===- transport_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/transport_test.cpp`, which runs the same
//! dispatch across a socket between two real processes (#348): `fork`, a
//! `socketpair`, and a child that owns the worker's memory and never sees the
//! parent's.
//!
//! Two things only a real descriptor can exercise. Short reads, because a
//! stream socket delivers what has arrived rather than what was asked for --
//! removing the loop makes the child fail on the first 512 KiB message and the
//! parent die of SIGPIPE. And separate address spaces: the child resolves
//! handles in its own table against its own memory, so a pointer that leaked
//! across instead of a handle would have worked in the in-process test and
//! cannot work here.
//!
//! It does not test the artifact assumption. Parent and child are the same
//! binary, so the outlined kernel is trivially present on both sides; on two
//! real machines that has to be arranged.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_dispatch_crosses_a_socket_between_two_processes() {
    let root = repo_root();
    let source = root.join("tests/runtime/transport_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("transport_test");
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
