//===- melior_test2.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Binary for testing Melior MLIR bindings and lowering.
//
//===----------------------------------------------------------------------===//

use anyhow::{Context, Result};

fn main() -> Result<()> {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    let f = melior::ir::Type::parse(&context, "!llvm.func<i32 (!llvm.ptr, ...)>")
        .context("Failed to parse function type")?;
    println!("func type: {:?}", f);
    Ok(())
}
