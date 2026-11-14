//===- remote_kernel_output_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/remote_kernel_output_test.cpp`. It checks that
//! a kernel run on a worker writes its results into the host's buffers. The
//! worker is a fake one in a child process, so the test needs no GPU.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn a_remote_kernel_writes_reach_the_host_buffers() {
    let root = repo_root();
    let source = root.join("tests/runtime/remote_kernel_output_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("remote_kernel_output_test");
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
        .expect("failed to run the compiled kernel output test");

    assert!(
        run.status.success(),
        "a remote kernel's output did not reach the host:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
