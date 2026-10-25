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

/// While a task runs, each thread keeps what it prints until a newline, and then writes the
/// whole line at once. So the lines of tasks that print at the same time do not mix. With no
/// task running, each piece is written at once, as before, so its order with output from C is
/// the order of the calls.
pub(crate) mod lines {
    use std::cell::RefCell;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static RUNNING_TASKS: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        static LINE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
        /// Set between the two `vx_print_to_stderr` calls around an `eprint!` or a `panic`.
        static TO_STDERR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    pub(crate) fn set_stderr(on: bool) {
        TO_STDERR.with(|flag| flag.set(on));
    }

    pub(crate) fn task_started() {
        RUNNING_TASKS.fetch_add(1, Ordering::SeqCst);
    }

    /// At the end of a task on its own thread: writes what is left of its line.
    pub(crate) fn task_ended() {
        flush();
        RUNNING_TASKS.fetch_sub(1, Ordering::SeqCst);
    }

    /// Writes what is left of this thread's line.
    pub(crate) fn flush() {
        LINE.with(|line| {
            let mut line = line.borrow_mut();
            if !line.is_empty() {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(&line);
                let _ = out.flush();
                line.clear();
            }
        });
    }

    pub(crate) fn print(text: &str) {
        // Standard error is not buffered, and a task's line on it is not kept apart.
        if TO_STDERR.with(|flag| flag.get()) {
            let _ = std::io::stderr().lock().write_all(text.as_bytes());
            return;
        }
        LINE.with(|line| {
            let mut line = line.borrow_mut();
            if RUNNING_TASKS.load(Ordering::SeqCst) == 0 && line.is_empty() {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
                return;
            }
            line.extend_from_slice(text.as_bytes());
            if let Some(end) = line.iter().rposition(|&b| b == b'\n') {
                let mut out = std::io::stdout().lock();
                let _ = out.write_all(&line[..=end]);
                let _ = out.flush();
                line.drain(..=end);
            }
        });
    }
}

#[no_mangle]
pub extern "C" fn print_i32(val: i32) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_i64(val: i64) -> i32 {
    lines::print(&format!("{}", val));
    0
}

/// The unsigned counterparts of `print_i32`/`print_i64`. MLIR's integer types are signless,
/// so a Vx `u64` and `i64` reach codegen as the same `i64` and only the element type says
/// which is which; without these every value above the signed range printed as a negative
/// number. One per width, so the element type alone decides the call.
#[no_mangle]
pub extern "C" fn print_u8(val: u8) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_u16(val: u16) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_u32(val: u32) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_u64(val: u64) -> i32 {
    lines::print(&format!("{}", val));
    0
}

/// A `bool`, widened to an `i32` by the caller: prints `true` or `false`, as Rust does.
#[no_mangle]
pub extern "C" fn print_bool(val: i32) -> i32 {
    lines::print(&format!("{}", val != 0));
    0
}

#[no_mangle]
pub extern "C" fn print_f32(val: f32) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_f64(val: f64) -> i32 {
    lines::print(&format!("{}", val));
    0
}

#[no_mangle]
pub extern "C" fn print_str(piece: *const libc::c_char) -> i32 {
    if piece.is_null() {
        return 0;
    }
    if let Ok(s) = unsafe { std::ffi::CStr::from_ptr(piece) }.to_str() {
        lines::print(s);
    }
    0
}

#[no_mangle]
pub extern "C" fn println() -> i32 {
    lines::print("\n");
    0
}

/// With `on` non-zero, the `print_*` functions write to standard error on this thread until the
/// next call with zero. The code generators put an `eprint!` and a `panic` between two calls.
#[no_mangle]
pub extern "C" fn vx_print_to_stderr(on: i32) -> i32 {
    lines::set_stderr(on != 0);
    0
}
