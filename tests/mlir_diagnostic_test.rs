use melior::{
    ir::{operation::OperationBuilder, operation::OperationLike, BlockLike, Location, Module},
    Context,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_mlir_diagnostic_suppressed_by_default() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let captured = Arc::new(Mutex::new(String::new()));
    let captured_clone = captured.clone();

    // Default behavior is to suppress (our compiler driver logic)
    let emit_diagnostics = false;
    context.attach_diagnostic_handler(move |diagnostic| {
        if emit_diagnostics {
            *captured_clone.lock().unwrap() = format!("{}", diagnostic);
        }
        true
    });

    let location = Location::unknown(&context);
    let module = Module::new(location);

    // Create an invalid operation: arith.addi with no operands
    let invalid_op = OperationBuilder::new("arith.addi", location)
        .build()
        .expect("Failed to build op");
    module.body().append_operation(invalid_op);

    let is_valid = module.as_operation().verify();
    assert!(
        !is_valid,
        "Module should be invalid due to malformed operation"
    );
    assert!(
        captured.lock().unwrap().is_empty(),
        "Diagnostic should have been suppressed"
    );
}

#[test]
fn test_mlir_diagnostic_emitted() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let captured = Arc::new(Mutex::new(String::new()));
    let captured_clone = captured.clone();

    // Behavior when --emit-backend-diagnostics is passed
    let emit_diagnostics = true;
    context.attach_diagnostic_handler(move |diagnostic| {
        if emit_diagnostics {
            *captured_clone.lock().unwrap() = format!("{}", diagnostic);
        }
        true
    });

    let location = Location::unknown(&context);
    let module = Module::new(location);

    // Create an invalid operation: arith.addi with no operands
    let invalid_op = OperationBuilder::new("arith.addi", location)
        .build()
        .expect("Failed to build op");
    module.body().append_operation(invalid_op);

    let is_valid = module.as_operation().verify();
    assert!(
        !is_valid,
        "Module should be invalid due to malformed operation"
    );
    assert!(
        captured.lock().unwrap().contains("requires one result"),
        "Diagnostic was not captured properly"
    );
}

#[test]
fn test_mlir_diagnostic_parse_error() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    melior::utility::register_all_passes();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    let captured = Arc::new(Mutex::new(String::new()));
    let captured_clone = captured.clone();

    // We want to capture the diagnostic
    context.attach_diagnostic_handler(move |diagnostic| {
        captured_clone
            .lock()
            .unwrap()
            .push_str(&format!("{}", diagnostic));
        true
    });

    // 1. Try to parse an invalid MLIR string to trigger a diagnostic
    let mlir_str = "module { func.invalid_op() }";
    let _ = melior::ir::Module::parse(&context, mlir_str);

    let diag = captured.lock().unwrap();
    assert!(
        diag.contains("custom op 'func.invalid_op' is unknown"),
        "Expected parse diagnostic, got: {}",
        diag
    );
}

#[test]
fn test_vx_valid_program_emits_remark() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    melior::utility::register_all_passes();
    vxc::codegen::register_vx_passes();

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    // Enable optimization remarks directly on the context via our C++ wrapper
    vxc::codegen::enable_optimization_remarks_for_testing(&context);

    let captured = Arc::new(Mutex::new(String::new()));
    let captured_clone = captured.clone();

    // We want to capture the diagnostic
    context.attach_diagnostic_handler(move |diagnostic| {
        captured_clone
            .lock()
            .unwrap()
            .push_str(&format!("{}", diagnostic));
        true
    });

    // 1. Load an MLIR string containing a vx operation (or just an empty module since our pass runs on ModuleOp)
    let mlir_str = r#"
module {
  func.func @test() {
    return
  }
}
"#;
    let mut module = melior::ir::Module::parse(&context, mlir_str).unwrap();

    // 2. Trigger an optimization remark using a pass that analyzes the IR and emits remarks
    let pass_manager = melior::pass::PassManager::new(&context);
    melior::utility::parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(convert-vx-to-standard)",
    )
    .unwrap();
    let _ = pass_manager.run(&mut module);

    let diag = captured.lock().unwrap();
    assert!(
        diag.contains("Lowering Vx to Standard dialects"),
        "Expected Vx lowering remark, got: {}",
        diag
    );
}
