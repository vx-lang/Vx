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
    use crate::syntax::{Expr, BinaryOpExpr, NumberExpr, Span, ElementType};
    use crate::symbol::Symbol;

    #[test]
    fn test_lower_ast_to_arena() {
        let mut arena = HirArena::new();
        
        let expr = Expr::BinaryOp(BinaryOpExpr {
            lhs: Box::new(Expr::Number(NumberExpr {
                value: "10".into(),
                ty: Some(ElementType::I32),
                span: Span::default(),
            })),
            op: crate::syntax::BinaryOp::Add,
            rhs: Box::new(Expr::Number(NumberExpr {
                value: "20".into(),
                ty: Some(ElementType::I32),
                span: Span::default(),
            })),
            span: Span::default(),
        });

        let expr_id = arena.lower_expr(&expr);
        
        // Ensure 3 expressions were allocated: 2 numbers, 1 binary op
        assert_eq!(arena.exprs.len(), 3);
        
        if let HirExpr::BinaryOp(b) = arena.exprs.get(expr_id.0).unwrap() {
            assert_eq!(b.op, crate::syntax::BinaryOp::Add);
            assert!(arena.exprs.get(b.lhs.0).is_some());
            assert!(arena.exprs.get(b.rhs.0).is_some());
        } else {
            panic!("Expected BinaryOp");
        }
    }
}
