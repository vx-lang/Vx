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
