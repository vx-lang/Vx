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
            Statement::LetDecl(LetDeclStmt {
                name: "sum".to_string(),
                is_mut: true,
                ty_ann: None,
                expr: Expr::Number(NumberExpr { value: "0".to_string(), ty: None, span: Span::default() }),
                span: Span::default(),
            }),
            Statement::ForLoop(ForLoopStmt {
                iter: "i".to_string(),
                start: Box::new(Expr::Number(NumberExpr { value: "0".to_string(), ty: None, span: Span::default() })),
                end: Box::new(Expr::Number(NumberExpr { value: "10".to_string(), ty: None, span: Span::default() })),
                body: vec![Statement::ExprStmt(ExprStmtStmt {
                    expr: Expr::If(IfExpr {
                        cond: Box::new(Expr::BinaryOp(BinaryOpExpr {
                            lhs: Box::new(Expr::Identifier(IdentifierExpr { name: "i".to_string(), span: Span::default() })),
                            op: BinaryOp::Gt,
                            rhs: Box::new(Expr::Number(NumberExpr { value: "5".to_string(), ty: None, span: Span::default() })),
                            span: Span::default(),
                        })),
                        then_block: vec![Statement::Assign(AssignStmt {
                            lhs: Expr::Identifier(IdentifierExpr { name: "sum".to_string(), span: Span::default() }),
                            rhs: Expr::BinaryOp(BinaryOpExpr {
                                lhs: Box::new(Expr::Identifier(IdentifierExpr { name: "sum".to_string(), span: Span::default() })),
                                op: BinaryOp::Add,
                                rhs: Box::new(Expr::Identifier(IdentifierExpr { name: "i".to_string(), span: Span::default() })),
                                span: Span::default(),
                            }),
                            span: Span::default(),
                        })],
                        else_block: None,
                        span: Span::default(),
                    }),
                    has_semi: false,
                    span: Span::default(),
                })],
                span: Span::default(),
            }),
            Statement::Return(ReturnStmt {
                expr: Expr::Identifier(IdentifierExpr { name: "sum".to_string(), span: Span::default() }),
                span: Span::default(),
            }),
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
