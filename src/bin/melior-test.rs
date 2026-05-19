use melior::Context;
use std::collections::HashMap;
use vxc::ast::*;
use vxc::melior_codegen::MeliorGenerator;

fn main() {
    let context = Context::new();
    let mut generator = MeliorGenerator::new(&context);

    // Create a dummy AST
    // fn test_func() -> i32 {
    //     let a = 20;
    //     let b = 22;
    //     return a + b;
    // }
    let func = Function {
        name: "test_func".to_string(),
        generics: vec![],
        params: vec![],
        return_type: Type::Scalar(ElementType::I32),
        body: vec![
            Statement::LetDecl(
                "a".to_string(),
                false,
                None,
                Expr::Number("20".to_string(), None, Span::default()),
                Span::default(),
            ),
            Statement::LetDecl(
                "b".to_string(),
                false,
                None,
                Expr::Number("22".to_string(), None, Span::default()),
                Span::default(),
            ),
            Statement::Return(
                Expr::BinaryOp(
                    Box::new(Expr::Identifier("a".to_string(), Span::default())),
                    BinaryOp::Add,
                    Box::new(Expr::Identifier("b".to_string(), Span::default())),
                    Span::default(),
                ),
                Span::default(),
            ),
        ],
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
