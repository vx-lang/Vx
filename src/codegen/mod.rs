pub mod generator;
pub mod lower;
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
    pub fn addVxLoweringPass(pm: mlir_sys::MlirPassManager);
    pub fn addVxToLLVMPass(pm: mlir_sys::MlirPassManager);
}

pub fn register_vx_dialect(context: &Context) {
    unsafe {
        registerVxDialect(context.to_raw());
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
        let c_path = std::ffi::CString::new(enzyme_lib.clone()).unwrap();
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

    // Instead of overwriting with parse_pass_pipeline, we append passes manually
    // or we parse a pipeline into an empty manager and nest it?
    // Let's just use pass_manager.add_pass() for standard passes!
    // But `parse_pass_pipeline` is easier. So we can just parse the rest of the pipeline
    // by appending our pass name to the string!
    // Note: the pass name is not registered as a string! ConvertVxToStandardPass has no String name unless we give it one!
    // We can just add the passes one by one using the string API, or `pass_manager.add_pass`.
    // Actually, `parse_pass_pipeline` adds to the pass manager, it doesn't necessarily clear it?
    // Wait! `melior::utility::parse_pass_pipeline` DOES clear or overwrite if it's top level!
    // Note: add the C++ pass AFTER parse_pass_pipeline.
    // NO, VxLowering must happen FIRST because it removes custom `vx` ops.
    // So let's parse the standard pipeline, but wait, `addVxLoweringPass` is a C API.
    // If we call `addVxLoweringPass` BEFORE, and `parse_pass_pipeline` clears it, that's bad.
    // Let's just use a separate PassManager for VxLowering!
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
    pipeline.push_str("convert-linalg-to-loops,lower-affine,convert-scf-to-cf,expand-strided-metadata,convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-cf-to-llvm,convert-arith-to-llvm,reconcile-unrealized-casts)");

    melior::utility::parse_pass_pipeline(pass_manager.as_operation_pass_manager(), &pipeline)
        .map_err(|e| format!("Failed to parse pass pipeline: {}", e))?;

    pass_manager
        .run(module)
        .map_err(|e| format!("Failed to lower MLIR module to LLVM: {}", e))?;

    Ok(has_enzyme)
}
