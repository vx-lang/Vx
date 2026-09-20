//===- io.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
pub extern "C" fn print_i64(val: i64) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

/// The unsigned counterparts of `print_i32`/`print_i64`. MLIR's integer types are signless,
/// so a Vx `u64` and `i64` reach codegen as the same `i64` and only the element type says
/// which is which; without these every value above the signed range printed as a negative
/// number. One per width, so the element type alone decides the call.
#[no_mangle]
pub extern "C" fn print_u8(val: u8) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_u16(val: u16) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_u32(val: u32) -> i32 {
    print!("{}", val);
    let _ = std::io::Write::flush(&mut std::io::stdout());
    0
}

#[no_mangle]
pub extern "C" fn print_u64(val: u64) -> i32 {
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
    if piece.is_null() {
        return 0;
    }
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
