//===- kernel_launch_test.rs - Vx Compiler ----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
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
//! hardware is the expensive way, and finding it in a fused attention kernel is
//! worse, because there is nothing to compare the output against.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn dispatch_arguments_become_the_parameter_list_the_kernel_declares() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/runtime/kernel_launch_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("kernel_launch_test");
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
        .expect("failed to run the compiled marshalling test");

    assert!(
        run.status.success(),
        "kernel launch marshalling checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
