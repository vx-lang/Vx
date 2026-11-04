//===- kernel_launch_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/kernel_launch_test.cpp`, which checks how a
//! dispatch's arguments become a device kernel's parameter list (#251).
//!
//! The marshalling lives in a header rather than in the CUDA backend so it can
//! be checked here, with no GPU. It has to be: a parameter list of the wrong
//! length or the wrong order still launches -- the driver cannot check one
//! against the kernel's signature -- so the kernel reads its arguments from the
//! wrong offsets and returns a plausible wrong answer. Finding that on rented
//! hardware is the expensive way, and finding it in a large fused kernel is
//! worse, because there is nothing to compare the output against.

use std::path::PathBuf;
use std::process::Command;

use crate::integration_test::runtime_test_build;

#[test]
fn dispatch_arguments_become_the_parameter_list_the_kernel_declares() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/runtime/kernel_launch_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_launch_test");
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
        .expect("failed to run the compiled marshalling test");

    assert!(
        run.status.success(),
        "kernel launch marshalling checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
