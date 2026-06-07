//===- break_utils.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Utilities for handling control flow lowering (break/continue) in the MLIR backend.
//
//===----------------------------------------------------------------------===//

use crate::ast::expr::*;
use crate::ast::stmt::*;

pub fn contains_break(stmt: &Statement) -> bool {
    match stmt {
        Statement::Break(_) => true,
        Statement::Continue(_) => true,
        Statement::LetDecl(s) => expr_contains_break(&s.expr),
        Statement::Return(s) => expr_contains_break(&s.expr),
        Statement::ExprStmt(s) => expr_contains_break(&s.expr),
        Statement::Assign(s) => expr_contains_break(&s.lhs) || expr_contains_break(&s.rhs),
        Statement::CompoundAssign(s) => expr_contains_break(&s.lhs) || expr_contains_break(&s.rhs),
        Statement::Assert(s) => expr_contains_break(&s.expr),
        Statement::Loop(_) => false, // inner loop handles its own breaks
        Statement::ForLoop(_) => false, // inner loop
        Statement::MacroCall(_) => panic!("Macros should be expanded before codegen"),
    }
}

pub fn expr_contains_break(expr: &Expr) -> bool {
    match expr {
        Expr::If(e) => {
            expr_contains_break(&e.cond)
                || e.then_block.iter().any(contains_break)
                || e.else_block
                    .as_ref()
                    .is_some_and(|b| b.iter().any(contains_break))
        }
        Expr::Match(e) => {
            expr_contains_break(&e.expr) || e.arms.iter().any(|a| a.body.iter().any(contains_break))
        }
        Expr::BinaryOp(e) => expr_contains_break(&e.lhs) || expr_contains_break(&e.rhs),
        Expr::RelationalOp(e) => expr_contains_break(&e.lhs) || expr_contains_break(&e.rhs),
        Expr::LogicalOp(e) => expr_contains_break(&e.lhs) || expr_contains_break(&e.rhs),
        Expr::UnaryOp(e) => expr_contains_break(&e.expr),
        Expr::StructInit(e) => e.fields.iter().any(|f| expr_contains_break(&f.1)),
        Expr::MemberAccess(e) => expr_contains_break(&e.base),
        Expr::IndexAccess(e) => expr_contains_break(&e.base) || expr_contains_break(&e.index),
        Expr::FunctionCall(e) => e.args.iter().any(expr_contains_break),
        Expr::MethodCall(e) => {
            expr_contains_break(&e.base) || e.args.iter().any(expr_contains_break)
        }
        Expr::Array(e) => e.elements.iter().any(expr_contains_break),
        Expr::UnsafeBlock(e) => e.stmts.iter().any(contains_break),
        Expr::ComptimeBlock(e) => e.stmts.iter().any(contains_break),
        Expr::Borrow(e) => expr_contains_break(&e.expr),
        Expr::Dereference(e) => expr_contains_break(&e.expr),
        Expr::SpawnOn(e) => {
            e.stmts.iter().any(contains_break)
                || e.ret.as_ref().is_some_and(|r| expr_contains_break(r))
        }
        Expr::Transfer(e) => expr_contains_break(&e.expr),
        _ => false,
    }
}
