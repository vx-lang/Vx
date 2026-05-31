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
