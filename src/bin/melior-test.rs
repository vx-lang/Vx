use melior::Context;
use std::collections::HashMap;
use vxc::ast::*;
use vxc::melior_codegen::MeliorGenerator;

fn main() {
    let context = Context::new();
    let mut generator = MeliorGenerator::new(&context);

    // Create a dummy AST
    // fn test_func() -> i32 { return 42; }
    let func = Function {
        name: "test_func".to_string(),
        generics: vec![],
        params: vec![],
        return_type: Type::Scalar(ElementType::I32),
        body: vec![Statement::Return(
            Expr::Number("42".to_string(), None, Span::default()),
            Span::default(),
        )],
    };

    let program = Program {
        module_path: "test.vx".to_string(),
        imports: vec![],
        externs: vec![],
        structs: vec![],
        enums: vec![],
        traits: vec![],
        impls: vec![],
        functions: vec![func],
    };

    let modules = HashMap::new();
    let mlir_str = generator.generate(&program, &modules);

    println!("{}", mlir_str);
}
