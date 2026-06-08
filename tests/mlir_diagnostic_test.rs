use melior::{
    ir::{operation::OperationBuilder, operation::OperationLike, BlockLike, Location, Module},
    Context,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_mlir_diagnostic_suppressed_by_default() -> Result<(), String> {
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
    if is_valid {
        return Err("Module should be invalid due to malformed operation".to_string());
    }
    if !(captured.lock().unwrap().is_empty()) {
        return Err("Diagnostic should have been suppressed".to_string());
    }

    Ok(())
}

#[test]
fn test_mlir_diagnostic_emitted() -> Result<(), String> {
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
    if is_valid {
        return Err("Module should be invalid due to malformed operation".to_string());
    }
    if !(captured.lock().unwrap().contains("requires one result")) {
        return Err("Diagnostic was not captured properly".to_string());
    }

    Ok(())
}

#[test]
fn test_mlir_diagnostic_parse_error() -> Result<(), String> {
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
    if !(diag.contains("custom op 'func.invalid_op' is unknown")) {
        return Err(format!("Expected parse diagnostic, got: {}", diag));
    }

    Ok(())
}

#[test]
fn test_vx_valid_program_emits_remark() -> Result<(), String> {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    melior::utility::register_all_passes();
    vxc::codegen::register_vx_passes();

    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    // Enable optimization remarks directly on the context via our C++ wrapper
    vxc::codegen::enable_optimization_remarks(&context);

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

    // 1. Compile a valid Vx program
    let input = r#"
fn test_remark() -> f32 {
    return 1.0;
}
"#;
    let mut lexer = vxc::lexer::Lexer::new(input);
    let tokens = lexer.tokenize();
    let mut parser = vxc::parser::Parser::new(tokens, input);
    let mut ast = parser.parse().unwrap();

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [ast.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = vxc::sema::TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        checker.check_function(f);
    }
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err("Assertion failed: !checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)"
            .to_string());
    }

    // Generate MLIR
    let module_asts = std::collections::HashMap::new();
    let mut codegen = vxc::codegen::MeliorGenerator::new(&context);
    let _ = codegen.generate(&ast, &module_asts);
    let mut module = codegen.into_module();

    // 2. Trigger an optimization remark using a pass that analyzes the IR and emits remarks
    let pass_manager = melior::pass::PassManager::new(&context);
    melior::utility::parse_pass_pipeline(
        pass_manager.as_operation_pass_manager(),
        "builtin.module(convert-vx-to-standard)",
    )
    .unwrap();
    let _ = pass_manager.run(&mut module);

    let diag = captured.lock().unwrap();
    if !(diag.contains("Lowering Vx to Standard dialects")) {
        return Err(format!("Expected Vx lowering remark, got: {}", diag));
    }

    Ok(())
}
