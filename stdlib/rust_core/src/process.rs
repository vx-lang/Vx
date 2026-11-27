//===- process.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Rust core behind `Processes::run` in `std::process`: start another program and wait until
// it ends.
//
//===----------------------------------------------------------------------===//

use std::ffi::c_void;
use std::process::Command;

/// A new command that starts the program that the Vx string `program` names. `vx_process_run`
/// frees it.
#[no_mangle]
pub extern "C" fn vx_process_new(program: *const c_void) -> *mut c_void {
    assert!(!program.is_null());
    let program = unsafe { &*(program as *const String) };
    Box::into_raw(Box::new(Command::new(program))) as *mut c_void
}

/// Adds the Vx string `arg` to the arguments of `command`.
#[no_mangle]
pub extern "C" fn vx_process_arg(command: *mut c_void, arg: *const c_void) {
    assert!(!command.is_null() && !arg.is_null());
    let command = unsafe { &mut *(command as *mut Command) };
    command.arg(unsafe { &*(arg as *const String) });
}

/// Starts `command`, waits until the program ends, and frees `command`. 0 and the exit code in
/// `*code`, 1 when the program did not start, 2 and the number of the signal in `*code` when a
/// signal stopped the program.
#[no_mangle]
pub extern "C" fn vx_process_run(command: *mut c_void, code: *mut i32) -> i32 {
    assert!(!command.is_null() && !code.is_null());
    let mut command = unsafe { Box::from_raw(command as *mut Command) };
    // The program writes to the same standard output, so write what this program printed first.
    crate::io::lines::flush();
    unsafe { libc::fflush(std::ptr::null_mut()) };
    let Ok(status) = command.status() else {
        return 1;
    };
    if let Some(exit) = status.code() {
        unsafe { *code = exit };
        return 0;
    }
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status).unwrap_or(0);
    #[cfg(not(unix))]
    let signal = 0;
    unsafe { *code = signal };
    2
}
