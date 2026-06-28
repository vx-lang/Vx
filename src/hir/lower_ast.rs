//===- lower_ast.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Converts the tree-based syntax AST into the flat, arena-based HIR.
//
//===----------------------------------------------------------------------===//

use crate::syntax::{Expr, Statement};
use crate::hir::arena::*;

impl HirArena {
    pub fn lower_expr(&mut self, expr: &Expr) -> ExprId {
        let hir_expr = match expr {
            Expr::Number(e) => HirExpr::Number(HirNumberExpr {
                value: e.value.clone(),
                ty: e.ty.clone(),
                span: e.span,
            }),
            Expr::Identifier(e) => HirExpr::Identifier(HirIdentifierExpr {
                name: e.name.clone(),
                span: e.span,
            }),
            Expr::BinaryOp(e) => HirExpr::BinaryOp(HirBinaryOpExpr {
                lhs: self.lower_expr(&e.lhs),
                op: e.op.clone(),
                rhs: self.lower_expr(&e.rhs),
                span: e.span,
            }),
            _ => todo!("lower_expr: {:?}", expr),
        };
        self.alloc_expr(hir_expr)
    }
    
    pub fn lower_stmt(&mut self, stmt: &Statement) -> StmtId {
        let hir_stmt = match stmt {
            Statement::LetDecl(s) => HirStmt::LetDecl(HirLetDeclStmt {
                name: s.name.clone(),
                ty: s.ty_ann.clone(),
                expr: self.lower_expr(&s.expr),
                span: s.span,
            }),
            Statement::Return(s) => HirStmt::Return(HirReturnStmt {
                expr: self.lower_expr(&s.expr),
                span: s.span,
            }),
            _ => todo!("lower_stmt: {:?}", stmt),
        };
        self.alloc_stmt(hir_stmt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::{Expr, BinaryOpExpr, NumberExpr, IdentifierExpr, Statement, LetDeclStmt, ReturnStmt, Span, ElementType};
    use crate::symbol::Symbol;

    #[test]
    fn test_lower_number_expr() {
        let mut arena = HirArena::new();
        let expr = Expr::Number(NumberExpr {
            value: "42".into(),
            ty: Some(ElementType::I32),
            span: Span::default(),
        });
        let id = arena.lower_expr(&expr);
        assert_eq!(arena.exprs.len(), 1);
        match arena.exprs.get(id).unwrap() {
            HirExpr::Number(n) => assert_eq!(n.value.as_ref(), "42"),
            _ => panic!("Expected Number"),
        }
    }

    #[test]
    fn test_lower_binary_op_deep_nesting() {
        let mut arena = HirArena::new();
        
        let mut current_expr = Expr::Number(NumberExpr {
            value: "1".into(),
            ty: Some(ElementType::I32),
            span: Span::default(),
        });
        
        // Deep nest 100 times to stress test arena allocation
        for _ in 0..100 {
            current_expr = Expr::BinaryOp(BinaryOpExpr {
                lhs: Box::new(current_expr),
                op: crate::syntax::BinaryOp::Add,
                rhs: Box::new(Expr::Number(NumberExpr {
                    value: "1".into(),
                    ty: Some(ElementType::I32),
                    span: Span::default(),
                })),
                span: Span::default(),
            });
        }

        let expr_id = arena.lower_expr(&current_expr);
        
        // 1 initial number + (1 binary op + 1 rhs number) * 100 = 201 nodes
        assert_eq!(arena.exprs.len(), 201);
    }
    
    #[test]
    fn test_lower_let_decl_stmt() {
        let mut arena = HirArena::new();
        
        let stmt = Statement::LetDecl(LetDeclStmt {
            name: Symbol::intern("x"),
            is_mut: false,
            ty_ann: None,
            expr: Box::new(Expr::Identifier(IdentifierExpr {
                name: Symbol::intern("y"),
                span: Span::default(),
            })),
            span: Span::default(),
        });
        
        let stmt_id = arena.lower_stmt(&stmt);
        assert_eq!(arena.stmts.len(), 1);
        assert_eq!(arena.exprs.len(), 1); // The identifier 'y'
        
        if let HirStmt::LetDecl(l) = arena.stmts.get(stmt_id).unwrap() {
            assert_eq!(l.name.as_ref(), "x");
        } else {
            panic!("Expected LetDecl");
        }
    }
    
    #[test]
    fn test_lower_return_stmt() {
        let mut arena = HirArena::new();
        
        let stmt = Statement::Return(ReturnStmt {
            expr: Box::new(Expr::Number(NumberExpr {
                value: "42".into(),
                ty: None,
                span: Span::default(),
            })),
            span: Span::default(),
        });
        
        let stmt_id = arena.lower_stmt(&stmt);
        assert_eq!(arena.stmts.len(), 1);
        assert_eq!(arena.exprs.len(), 1);
        
        if let HirStmt::Return(_) = arena.stmts.get(stmt_id).unwrap() {
        } else {
            panic!("Expected Return");
        }
    }
}
