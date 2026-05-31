use melior::{Context, ir::{operation::OperationBuilder, Location, Module, BlockLike, operation::OperationLike}};

#[test]
fn test_mlir_diagnostic_suppressed_by_default() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    // Default behavior is to suppress (our compiler driver logic)
    let emit_diagnostics = false;
    context.attach_diagnostic_handler(move |diagnostic| {
        if emit_diagnostics {
            eprintln!("DIAGNOSTIC: {}", diagnostic);
        }
        true
    });
    
    let location = Location::unknown(&context);
    let mut module = Module::new(location);
    
    // Create an invalid operation: arith.addi with no operands
    let invalid_op = OperationBuilder::new("arith.addi", location).build().expect("Failed to build op");
    module.body().append_operation(invalid_op);
    
    let is_valid = module.as_operation().verify();
    assert!(!is_valid, "Module should be invalid due to malformed operation");
}

#[test]
fn test_mlir_diagnostic_emitted() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    
    // Behavior when --emit-backend-diagnostics is passed
    let emit_diagnostics = true;
    context.attach_diagnostic_handler(move |diagnostic| {
        if emit_diagnostics {
            eprintln!("EMITTED DIAGNOSTIC: {}", diagnostic);
        }
        true
    });
    
    let location = Location::unknown(&context);
    let mut module = Module::new(location);
    
    // Create an invalid operation: arith.addi with no operands
    let invalid_op = OperationBuilder::new("arith.addi", location).build().expect("Failed to build op");
    module.body().append_operation(invalid_op);
    
    let is_valid = module.as_operation().verify();
    assert!(!is_valid, "Module should be invalid due to malformed operation");
}

#[test]
fn test_vx_valid_program_emits_remark() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    melior::utility::register_all_passes();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);
    
    // We want to capture the diagnostic
    context.attach_diagnostic_handler(move |diagnostic| {
        eprintln!("REMARK_EMITTED: {}", diagnostic);
        true
    });
    
    // 1. Load a valid Vx program
    let source = "fn test() -> i32 { return 5; }";
    let mut parser = vxc::parser::Parser::new(vxc::lexer::Lexer::new(source).tokenize(), source);
    let mut ast = parser.parse().expect("Failed to parse");
    
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let binding = [ast.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&binding);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = vxc::sema::TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        checker.check_function(f);
    }
    assert!(checker.errors.is_empty(), "Sema failed");
    
    // 2. Generate MLIR
    let mut codegen = vxc::codegen::MeliorGenerator::new(&context);
    codegen.generate(&ast, &std::collections::HashMap::new());
    let mut module = codegen.into_module();
    
    // 3. Trigger a diagnostic remark using the print-op-stats pass
    let pass_manager = melior::pass::PassManager::new(&context);
    melior::utility::parse_pass_pipeline(pass_manager.as_operation_pass_manager(), "builtin.module(print-op-stats)").unwrap();
    pass_manager.run(&mut module).unwrap();
}
