//===- runtime_test_build.rs - Vx Compiler ---------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! The compilers and flags for the C and C++ test programs in `tests/runtime/`.
//!
//! With `VX_TEST_SANITIZERS=1`, the programs also get AddressSanitizer and
//! UndefinedBehaviorSanitizer. CI's own LLVM has no sanitizer runtimes, so CI
//! sets `VX_TEST_CXX=g++` and `VX_TEST_CC=gcc` with it. These are not `CXX` and
//! `CC`, because a change to those makes cargo rebuild the C code of other crates.

use std::process::Command;

/// The C++ compiler: `$VX_TEST_CXX`, then `$CXX`, then `clang++`.
pub(crate) fn cxx() -> String {
    std::env::var("VX_TEST_CXX")
        .or_else(|_| std::env::var("CXX"))
        .unwrap_or_else(|_| "clang++".to_string())
}

/// The C compiler: `$VX_TEST_CC`, then `$CC`, then `clang`.
pub(crate) fn cc() -> String {
    std::env::var("VX_TEST_CC")
        .or_else(|_| std::env::var("CC"))
        .unwrap_or_else(|_| "clang".to_string())
}

/// A command that runs `compiler` with the flags of every test program.
/// `standard` is a `-std=` value, such as `c++17` or `c11`.
pub(crate) fn command(compiler: &str, standard: &str) -> Command {
    let mut cmd = Command::new(compiler);
    cmd.arg(format!("-std={standard}"))
        .args(["-O1", "-Wall", "-Werror"]);
    if std::env::var("VX_TEST_SANITIZERS").is_ok_and(|v| v == "1") {
        cmd.args([
            "-g",
            "-fsanitize=address,undefined",
            "-fno-sanitize-recover=all",
        ]);
    }
    cmd
}
