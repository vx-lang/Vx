//===- scratch.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Scratchpad for compiler experiments and isolated tests.
//
//===----------------------------------------------------------------------===//

#[allow(unused_imports)]
use melior::ir::BlockLike;

#[test]
pub fn test_operation_cloning_with_nested_regions() {
    // This experiment verifies that cloning a "vx.macro_wrapper"
    // correctly handles nested regions and yields.
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);
    let source =
        r#"module { "vx.macro_wrapper"() ({ ^bb0: "vx.yield"() : () -> () }) : () -> () }"#;
    let module = melior::ir::Module::parse(&context, source)
        .expect("Failed to parse MLIR source in scratchpad");
    let op = module
        .body()
        .first_operation()
        .expect("Failed to get first operation");
    // try cloning explicitly
    let cloned_op = melior::ir::operation::Operation::clone(&op);
    let dest_module = melior::ir::Module::new(melior::ir::Location::unknown(&context));
    dest_module.body().append_operation(cloned_op);
}
