//===- arena.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Flat, arena-allocated High-Level Intermediate Representation (HIR)
//
//===----------------------------------------------------------------------===//

use crate::syntax::*;
use crate::symbol::Symbol;
use id_arena::{Arena, Id};

pub type ExprId = Id<HirExpr>;
pub type StmtId = Id<HirStmt>;

#[derive(Debug, Clone)]
pub struct HirArena {
    pub exprs: Arena<HirExpr>,
    pub stmts: Arena<HirStmt>,
}

impl HirArena {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            exprs: Arena::new(),
            stmts: Arena::new(),
        }
    }
    
    pub fn alloc_expr(&mut self, expr: HirExpr) -> ExprId {
        self.exprs.alloc(expr)
    }
    
    pub fn alloc_stmt(&mut self, stmt: HirStmt) -> StmtId {
        self.stmts.alloc(stmt)
    }
}

// -----------------------------------------------------------------------------
// HIR AST Nodes
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct HirNumberExpr {
    pub value: Symbol,
    pub ty: Option<ElementType>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirIdentifierExpr {
    pub name: Symbol,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirBinaryOpExpr {
    pub lhs: ExprId,
    pub op: BinaryOp,
    pub rhs: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirExpr {
    Number(HirNumberExpr),
    Identifier(HirIdentifierExpr),
    BinaryOp(HirBinaryOpExpr),
    // TODO: Add remaining variants
}

#[derive(Debug, Clone)]
pub struct HirLetDeclStmt {
    pub name: Symbol,
    pub ty: Option<Type>,
    pub expr: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct HirReturnStmt {
    pub expr: ExprId,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum HirStmt {
    LetDecl(HirLetDeclStmt),
    Return(HirReturnStmt),
    // TODO: Add remaining variants
}
