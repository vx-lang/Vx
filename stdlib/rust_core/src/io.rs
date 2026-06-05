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
pub extern "C" fn print_i32(val: i32) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_f32(val: f32) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_f64(val: f64) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_str(piece: *const libc::c_char) -> i32 {
    if piece.is_null() { return 0; }
    if let Ok(s) = unsafe { std::ffi::CStr::from_ptr(piece) }.to_str() {
        print!("{}", s);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    0
}

#[no_mangle]
pub extern "C" fn println() -> i32 {
    println!();
    0
}
