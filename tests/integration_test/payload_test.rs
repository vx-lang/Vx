//===- payload_test.rs - the dispatch payload's own format ------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! The two payload entries that are about the format rather than about a kernel
//! (#1137).
//!
//! `abi=` is the version a dispatch library checks before it trusts anything
//! else: a key a consumer does not know is harmless, a value it misreads is not,
//! and the `vx_plugin_*` interface is not frozen. A binary device image -- SPIR-V
//! -- rides in a length-prefixed section after the entries, because its first
//! word holds NUL bytes and a NUL-terminated entry cannot carry it.
//!
//! The C++ half (`tests/runtime/payload_test.cpp`) pins that layout against a
//! payload built by hand, including the length that does not match the blob.
//! The Rust half pins the other end of the contract: that the compiler stamps
//! the version the header defines. Those are two different failures -- a decoder
//! that is wrong, and a producer that writes a number its own header does not
//! -- and neither test can see the other's.

use std::path::PathBuf;
use std::process::Command;

/// A placed kernel with no library equivalent, compiled the way
/// `device_image_test.rs` compiles it: through the binary, because the payload
/// is text in the emitted LLVM IR and both codegen paths have to agree on it.
fn emitted_ir(program: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("tests/backend/pass").join(program);
    let out = Command::new(env!("CARGO_BIN_EXE_vxc"))
        .arg(&path)
        .arg("--emit-llvm")
        .output()
        .unwrap_or_else(|e| panic!("could not run vxc: {e}"));
    assert!(
        out.status.success(),
        "vxc --emit-llvm failed on {}:\n{}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The compiler stamps the version, and the version is the one the header
/// defines.
///
/// The number is written literally here on purpose. `VX_PAYLOAD_ABI` is what the
/// compiler emits and what every dispatch library reads; if it moves, this
/// assertion is what fails, and the fix is to move both this line and the
/// `VX_PAYLOAD_ABI` check in the C++ test together.
#[test]
fn the_compiler_stamps_the_payload_version() {
    let ir = emitted_ir("placed_kernel_four_operands.vx");

    // In the IR the payload is a string constant, so the NUL that ends the entry
    // is escaped rather than literal.
    assert!(
        ir.contains("abi=1\\00"),
        "the dispatch payload carries no `abi=1` entry, so a dispatch library \
         has nothing to check the format against"
    );
    assert!(
        ir.contains("image="),
        "the payload this checks has no device image in it, so it is not \
         exercising the shape the version key exists for"
    );
}

/// Builds and runs `tests/runtime/payload_test.cpp`.
#[test]
fn the_payload_helpers_are_checked_on_the_host() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = root.join("tests/runtime/payload_test.cpp");
    let binary = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("payload_test");
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
        .expect("failed to run the compiled payload test");

    assert!(
        run.status.success(),
        "dispatch payload checks failed:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
}
