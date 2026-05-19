use melior::{
    dialect::DialectRegistry,
    ir::{Location, Module},
    Context,
};

fn main() {
    let registry = DialectRegistry::new();
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let location = Location::unknown(&context);
    let module = Module::new(location);

    let op = module.as_operation();
    println!("{}", op);
}
