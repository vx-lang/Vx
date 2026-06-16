pub mod break_utils;
pub mod generator;
pub mod lower;
pub use break_utils::*;
pub use generator::*;
pub use lower::*;

//===- melior_codegen.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the primary MLIR lowering pipeline using the Melior crate.
// It translates the type-checked Vx Abstract Syntax Tree into specific MLIR dialects
// (such as arith, scf, func, and linalg), performing the heavy lifting required
// for optimization and hardware targeting.
//
//===----------------------------------------------------------------------===//
use std::collections::HashMap;

use melior::{
    dialect::DialectRegistry,
    ir::{Block, BlockLike, Location, Module, Region, RegionLike, Type, Value, ValueLike},
    Context,
};

use crate::ast::*;

extern "C" {
    fn loadMlirPassPlugin(path: *const std::os::raw::c_char) -> bool;
    fn registerVxDialect(ctx: mlir_sys::MlirContext);
    fn registerVxPassesC();
    pub fn addVxLoweringPass(pm: mlir_sys::MlirPassManager);
    pub fn addVxToLLVMPass(pm: mlir_sys::MlirPassManager);
    fn parseCommandLineOptions(argc: std::ffi::c_int, argv: *const *const std::ffi::c_char);
    fn mlirEnableOptimizationRemarks(ctx: mlir_sys::MlirContext);
}

pub fn register_vx_dialect(context: &Context) {
    unsafe {
        registerVxDialect(context.to_raw());
    }
}

pub fn enable_optimization_remarks(context: &Context) {
    unsafe {
        mlirEnableOptimizationRemarks(context.to_raw());
    }
}

pub fn parse_command_line_options(args: &[String]) -> Result<(), String> {
    let mut c_args = Vec::new();
    for arg in args {
        let c_str = std::ffi::CString::new(arg.as_str())
            .map_err(|_| format!("Invalid CLI argument (contains null byte): {}", arg))?;
        c_args.push(c_str);
    }
    let c_args_ptrs: Vec<*const std::ffi::c_char> = c_args.iter().map(|s| s.as_ptr()).collect();
    unsafe {
        parseCommandLineOptions(c_args_ptrs.len() as std::ffi::c_int, c_args_ptrs.as_ptr());
    }
    Ok(())
}

pub fn register_vx_passes() {
    unsafe {
        registerVxPassesC();
    }
}

pub fn lower_to_llvm<'c>(context: &'c Context, module: &mut Module<'c>) -> Result<bool, String> {
    // Register the custom `vx` dialect before loading dialects
    register_vx_dialect(context);

    let pass_manager = melior::pass::PassManager::new(context);

    // Add custom Vx lowering passes
    unsafe {
        addVxLoweringPass(pass_manager.to_raw());
        addVxToLLVMPass(pass_manager.to_raw());
    }

    // Register all built-in passes
    melior::utility::register_all_passes();

    // Check if an external plugin is specified via ENZYME_LIB (for MLIR Enzyme)
    let mut has_enzyme = false;
    if let Ok(enzyme_lib) = std::env::var("ENZYME_LIB") {
        match std::ffi::CString::new(enzyme_lib.clone()) {
            Ok(c_path) => {
                let loaded = unsafe { loadMlirPassPlugin(c_path.as_ptr()) };
                if loaded {
                    println!("[CodeGen] Loaded MLIR Pass Plugin: {}", enzyme_lib);
                    has_enzyme = true;
                } else {
                    eprintln!(
                        "[CodeGen] Failed to load MLIR Pass Plugin (may not export MLIR plugin hooks): {}",
                        enzyme_lib
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "[CodeGen] Invalid ENZYME_LIB path (contains null byte): {}",
                    e
                );
            }
        }
    }

    // Architectural Note on PassManagers:
    // We use a separate `vx_pm` PassManager for our custom lowering passes first,
    // before running the standard string-based pass pipeline.
    // This is because `melior::utility::parse_pass_pipeline` directly overwrites
    // or clears the manager it is applied to. If we appended standard passes to
    // the same manager using the string API, it could conflict or drop the custom
    // passes we added via the raw C API `addVxLoweringPass`.
    // Running them in two separate sequences guarantees safety.
    let vx_pm = melior::pass::PassManager::new(context);
    unsafe {
        addVxLoweringPass(vx_pm.to_raw());
        addVxToLLVMPass(vx_pm.to_raw());
    }
    vx_pm
        .run(module)
        .map_err(|e| format!("Failed to lower Vx dialect: {}", e))?;

    // Now run standard pipeline
    let mut pipeline = "builtin.module(".to_string();
    if has_enzyme {
        pipeline.push_str("enzyme,");
    }
    pipeline.push_str("convert-linalg-to-loops,lower-affine,convert-scf-to-cf,expand-strided-metadata,convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-math-to-llvm,convert-math-to-libm,convert-cf-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts)");

    melior::utility::parse_pass_pipeline(pass_manager.as_operation_pass_manager(), &pipeline)
        .map_err(|e| format!("Failed to parse pass pipeline: {}", e))?;

    pass_manager
        .run(module)
        .map_err(|e| format!("Failed to lower MLIR module to LLVM: {}", e))?;

    Ok(has_enzyme)
}
