//===- vx-opt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Standalone Vx optimizer tool that interfaces with the MLIR backend.
//
//===----------------------------------------------------------------------===//

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

// Force Cargo to pull in melior's transitive dependencies (MLIR libraries)
#[allow(unused_imports, clippy::single_component_path_imports)]
use melior;

#[allow(clippy::duplicated_attributes)]
#[link(name = "vx_dialect", kind = "static")]
#[link(name = "plugin_loader", kind = "static")]
extern "C" {
    fn run_vx_opt(argc: c_int, argv: *const *const c_char) -> c_int;
}

use std::os::unix::ffi::OsStrExt;

fn main() {
    let c_args: Vec<CString> = std::env::args_os()
        .map(|arg| {
            CString::new(arg.as_bytes()).unwrap_or_else(|_| CString::new("invalid_arg").unwrap())
        })
        .collect();

    let c_ptrs: Vec<*const c_char> = c_args.iter().map(|a| a.as_ptr()).collect();

    let status = unsafe { run_vx_opt(c_ptrs.len() as c_int, c_ptrs.as_ptr()) };

    std::process::exit(status);
}
