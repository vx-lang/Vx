pub mod flat;
pub mod generator;
pub mod lower;
pub use generator::*;
pub use lower::*;

//===- melior_codegen.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

use crate::syntax::*;

/// The name of the Enzyme wrapper a differentiated call goes through.
///
/// Enzyme recognizes the call by this name, so both backends have to spell it the same way:
/// `__enzyme_fwddiff_jvp_f` for forward mode, `__enzyme_autodiff_grad_f` for reverse.
pub fn enzyme_wrapper_name(forward: bool, target: &str) -> String {
    if forward {
        format!("__enzyme_fwddiff_jvp_{target}")
    } else {
        format!("__enzyme_autodiff_grad_{target}")
    }
}

/// `core::libm` functions whose derivative Enzyme already knows under their C name. They read
/// a float's bits, which Enzyme cannot differentiate, so each is marked to be differentiated as
/// the C function instead.
const ENZYME_KNOWN_LIBM: &[(&str, &str)] = &[
    ("libm_sinf", "sinf"),
    ("libm_cosf", "cosf"),
    ("libm_tanf", "tanf"),
    ("libm_asinf", "asinf"),
    ("libm_acosf", "acosf"),
    ("libm_atanf", "atanf"),
    ("libm_atan2f", "atan2f"),
    ("libm_sinhf", "sinhf"),
    ("libm_coshf", "coshf"),
    ("libm_tanhf", "tanhf"),
    ("libm_asinhf", "asinhf"),
    ("libm_acoshf", "acoshf"),
    ("libm_atanhf", "atanhf"),
    ("libm_expf", "expf"),
    ("libm_exp2f", "exp2f"),
    ("libm_exp10f", "exp10f"),
    ("libm_expm1f", "expm1f"),
    ("libm_logf", "logf"),
    ("libm_log2f", "log2f"),
    ("libm_log10f", "log10f"),
    ("libm_log1pf", "log1pf"),
    ("libm_log", "log"),
    ("libm_log2", "log2"),
    ("libm_log10", "log10"),
    ("libm_log1p", "log1p"),
    ("libm_exp", "exp"),
    ("libm_exp2", "exp2"),
    ("libm_expm1", "expm1"),
    ("libm_sin", "sin"),
    ("libm_cos", "cos"),
    ("libm_tan", "tan"),
    ("libm_asin", "asin"),
    ("libm_acos", "acos"),
    ("libm_atan", "atan"),
    ("libm_atan2", "atan2"),
    ("libm_hypot", "hypot"),
    ("libm_pow", "pow"),
    ("libm_sinh", "sinh"),
    ("libm_cosh", "cosh"),
    ("libm_tanh", "tanh"),
    ("libm_asinh", "asinh"),
    ("libm_acosh", "acosh"),
    ("libm_atanh", "atanh"),
    ("libm_cbrt", "cbrt"),
    ("libm_cbrtf", "cbrtf"),
    ("libm_erff", "erff"),
    ("libm_hypotf", "hypotf"),
    ("libm_powf", "powf"),
];

/// Gives each function in `ENZYME_KNOWN_LIBM` the `enzyme_math` attribute naming its C function.
pub fn mark_libm_for_enzyme<'c>(context: &'c Context, module: &mut Module<'c>) {
    use melior::ir::attribute::{Attribute, StringAttribute};
    use melior::ir::operation::{OperationLike, OperationMutLike};
    let mut next = module.body().first_operation_mut();
    while let Some(mut op) = next {
        next = op.next_in_block_mut();
        if op.name().as_string_ref().as_str() != Ok("func.func") {
            continue;
        }
        let Some(name) = op
            .attribute("sym_name")
            .ok()
            .and_then(|a| StringAttribute::try_from(a).ok())
        else {
            continue;
        };
        let Some((_, c_name)) = ENZYME_KNOWN_LIBM.iter().find(|(n, _)| *n == name.value()) else {
            continue;
        };
        assert!(
            op.attribute("passthrough").is_err(),
            "{} already has LLVM attributes",
            name.value()
        );
        let attr = format!("[[\"enzyme_math\", \"{c_name}\"]]");
        op.set_attribute("passthrough", Attribute::parse(context, &attr).unwrap());
    }
}

extern "C" {
    fn loadMlirPassPlugin(path: *const std::os::raw::c_char) -> bool;
    fn registerVxDialect(ctx: mlir_sys::MlirContext);
    fn registerVxPassesC();
    pub fn addVxLoweringPass(pm: mlir_sys::MlirPassManager);
    pub fn addVxToLLVMPass(pm: mlir_sys::MlirPassManager);
    fn parseCommandLineOptions(argc: std::ffi::c_int, argv: *const *const std::ffi::c_char);
    fn mlirEnableOptimizationRemarks(ctx: mlir_sys::MlirContext);
}

use std::sync::Once;

static INIT: Once = Once::new();

pub fn init_codegen_globals() {
    INIT.call_once(|| {
        melior::utility::register_all_passes();
        register_vx_passes();
    });
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
    let c_args: Vec<std::ffi::CString> = args
        .iter()
        .map(|s| {
            std::ffi::CString::new(s.as_str())
                .map_err(|_| format!("Invalid CLI argument (contains null byte): {}", s))
        })
        .collect::<Result<Vec<_>, _>>()?;

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

    // Ensure all passes (built-in and custom) are registered exactly once globally
    init_codegen_globals();
    mark_libm_for_enzyme(context, module);

    let pass_manager = melior::pass::PassManager::new(context);
    // Verify after each pass. NOTE: this only checks IR validity; it does not
    // catch the NPU dispatch ABI limitation (by-value floats), which is a
    // runtime concern. See docs/lang/abi.md and runtime/npu_dispatch.mm.
    pass_manager.enable_verifier(true);

    // Check if an external plugin is specified via ENZYME_LIB (for MLIR Enzyme)
    let mut has_enzyme = false;
    if let Ok(enzyme_lib) = std::env::var("ENZYME_LIB") {
        match std::ffi::CString::new(enzyme_lib.as_str()) {
            Ok(c_path) => {
                let loaded = unsafe { loadMlirPassPlugin(c_path.as_ptr()) };
                if loaded {
                    println!("[CodeGen] Loaded MLIR Pass Plugin from {}", enzyme_lib);
                    has_enzyme = true;
                } else {
                    eprintln!(
                        "[CodeGen] Failed to load MLIR Pass Plugin at {}",
                        enzyme_lib
                    );
                }
            }
            Err(_) => {
                eprintln!("[CodeGen] Invalid ENZYME_LIB path (contains null byte)");
            }
        }
    }

    // Run unified pipeline. Custom Vx lowering passes run first, followed by standard lowering.
    let mut pipeline =
        "builtin.module(convert-vx-to-standard,vx-reorderable-reductions,vx-to-llvm,".to_string();
    if has_enzyme {
        pipeline.push_str("enzyme,");
    }
    // symbol-dce must precede finalize-memref-to-llvm. An unused `extern`
    // declaration lands as `func.func private @malloc`, which memref
    // finalization cannot reuse (it looks for an llvm.func), so it creates its
    // own `@malloc` and the symbol table uniques the name -- one `@malloc_N`
    // per allocation site, none of which resolves at link time. Importing
    // std::vec was enough to trigger it. Dropping dead declarations first lets
    // the lowering define `@malloc` under its own name.
    // `vx-promote-buffers-to-stack` + `vx-normalize-stack-buffers`: a small buffer that does not escape
    // belongs on the stack rather than in a `malloc` nothing frees, and the hoist is what makes
    // the promoted allocation a frame slot instead of per-iteration stack growth (#641). What
    // stays on the heap is freed at its drop, by convert-vx-to-standard.
    // See the fuller note on the copy of this pipeline in src/driver.rs.
    pipeline.push_str("symbol-dce,func.func(convert-linalg-to-loops,lower-affine),convert-scf-to-cf,expand-strided-metadata,func.func(vx-promote-buffers-to-stack),func.func(vx-normalize-stack-buffers),func.func(lower-affine),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-index-to-llvm,convert-math-to-llvm,convert-math-to-libm,convert-func-to-llvm,convert-cf-to-llvm,convert-arith-to-llvm,convert-ub-to-llvm,reconcile-unrealized-casts)");

    melior::utility::parse_pass_pipeline(pass_manager.as_operation_pass_manager(), &pipeline)
        .map_err(|e| format!("Failed to parse pass pipeline: {}", e))?;

    pass_manager
        .run(module)
        .map_err(|e| format!("Failed to lower MLIR module to LLVM: {}", e))?;

    Ok(has_enzyme)
}
