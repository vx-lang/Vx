use melior::Context;
use std::collections::HashMap;
use vxc::ast::*;
use vxc::melior_codegen::MeliorGenerator;

fn main() {
    let context = Context::new();
    let mut generator = MeliorGenerator::new(&context);

    // Create a dummy AST
    // fn test_func() -> i32 {
    //     let mut sum = 0;
    //     for i in 0..10 {
    //         if i > 5 {
    //             sum = sum + i;
    //         }
    //     }
    //     return sum;
    // }
    let func = Function {
        name: "test_func".to_string(),
        generics: vec![],
        params: vec![],
        return_type: Type::Scalar(ElementType::I32),
        body: vec![
            Statement::LetDecl(
                "sum".to_string(),
                true,
                None,
                Expr::Number("0".to_string(), None, Span::default()),
                Span::default(),
            ),
            Statement::ForLoop(
                "i".to_string(),
                Box::new(Expr::Number("0".to_string(), None, Span::default())),
                Box::new(Expr::Number("10".to_string(), None, Span::default())),
                vec![Statement::ExprStmt(
                    Expr::If(
                        Box::new(Expr::BinaryOp(
                            Box::new(Expr::Identifier("i".to_string(), Span::default())),
                            BinaryOp::Gt,
                            Box::new(Expr::Number("5".to_string(), None, Span::default())),
                            Span::default(),
                        )),
                        vec![Statement::Assign(
                            Expr::Identifier("sum".to_string(), Span::default()),
                            Expr::BinaryOp(
                                Box::new(Expr::Identifier("sum".to_string(), Span::default())),
                                BinaryOp::Add,
                                Box::new(Expr::Identifier("i".to_string(), Span::default())),
                                Span::default(),
                            ),
                            Span::default(),
                        )],
                        None,
                        Span::default(),
                    ),
                    false,
                    Span::default(),
                )],
                Span::default(),
            ),
            Statement::Return(
                Expr::Identifier("sum".to_string(), Span::default()),
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
