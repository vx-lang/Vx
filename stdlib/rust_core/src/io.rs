//===- io.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the standard library I/O primitives.
// It contains the Rust-side FFI implementations for file system operations,
// console printing, and basic input/output functionality exposed to Vx programs.
//
//===----------------------------------------------------------------------===//
//! FFI bindings for `std::fs::File`.

use crate::{instantiate_file_ffi, instantiate_stdio_ffi};

instantiate_file_ffi!();
instantiate_stdio_ffi!();

#[no_mangle]
pub extern "C" fn print_i32(v: i32) -> i32 {
    println!("{}", v);
    0
}
