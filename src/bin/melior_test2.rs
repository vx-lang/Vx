//===- melior_test2.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Binary for testing Melior MLIR bindings and lowering.
//
//===----------------------------------------------------------------------===//

fn main() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    let f = melior::ir::Type::parse(&context, "!llvm.func<i32 (!llvm.ptr, ...)>");
    println!("func type: {:?}", f);
}
