//===- device_pool_test.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! Builds and runs `tests/runtime/device_pool_test.cpp`, which checks the free
//! list a CUDA dispatch takes its staging buffers from (#321).
//!
//! The bookkeeping lives in a header with the allocator as a template
//! parameter, so it runs on a machine with no GPU -- which matters because
//! every way it can be wrong is silent on one. Handing the same buffer to two
//! live dispatches gives each the other's operands; dropping one on the
//! capacity path leaks device memory until an unrelated allocation fails hours
//! later; freeing one that is still in use is a use-after-free inside the
//! driver. None fault where the mistake is, and none show up in a dispatch
//! trace.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn staging_buffers_are_reused_without_being_confused() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/runtime/device_pool_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("device_pool_test");
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "clang++".to_string());

    let build = Command::new(&cxx)
        .args(["-std=c++17", "-O1", "-Wall", "-Werror", "-pthread"])
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
        .expect("failed to run the compiled device pool test");

    assert!(
        run.status.success(),
        "device pool checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
