#[allow(unused_imports)]
use melior::ir::BlockLike;

#[test]
pub fn test_clone4() {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    context.set_allow_unregistered_dialects(true);
    let source =
        r#"module { "vx.macro_wrapper"() ({ ^bb0: "vx.yield"() : () -> () }) : () -> () }"#;
    let module = melior::ir::Module::parse(&context, source).unwrap();
    let op = module.body().first_operation().unwrap();
    // try cloning explicitly
    let cloned_op = melior::ir::operation::Operation::clone(&op);
    let dest_module = melior::ir::Module::new(melior::ir::Location::unknown(&context));
    dest_module.body().append_operation(cloned_op);
}
