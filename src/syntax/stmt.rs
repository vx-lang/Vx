//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Abstract Syntax Tree nodes for Vx statements.
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::symbol::Symbol;

#[derive(Debug, PartialEq, Clone)]
pub struct LetDeclStmt {
    pub name: Symbol,
    pub is_mut: bool,
    pub ty_ann: Option<Type>,
    pub expr: Expr,
    pub span: Span,
}
impl LetDeclStmt {
    pub fn new(name: String, is_mut: bool, ty_ann: Option<Type>, expr: Expr, span: Span) -> Self {
        Self {
            name: name.into(),
            is_mut,
            ty_ann,
            expr,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ReturnStmt {
    pub expr: Expr,
    pub span: Span,
}
impl ReturnStmt {
    pub fn new(expr: Expr, span: Span) -> Self {
        Self { expr, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ExprStmtStmt {
    pub expr: Expr,
    pub has_semi: bool,
    pub span: Span,
}
impl ExprStmtStmt {
    pub fn new(expr: Expr, has_semi: bool, span: Span) -> Self {
        Self {
            expr,
            has_semi,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ForLoopStmt {
    pub iter: String,
    pub iterable: Box<Expr>,
    pub invariants: Vec<Expr>,
    pub body: Vec<Statement>,
    pub span: Span,
}
impl ForLoopStmt {
    pub fn new(
        iter: String,
        iterable: Box<Expr>,
        invariants: Vec<Expr>,
        body: Vec<Statement>,
        span: Span,
    ) -> Self {
        Self {
            iter,
            iterable,
            invariants,
            body,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AssignStmt {
    pub lhs: Expr,
    pub rhs: Expr,
    pub span: Span,
}
impl AssignStmt {
    pub fn new(lhs: Expr, rhs: Expr, span: Span) -> Self {
        Self { lhs, rhs, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct CompoundAssignStmt {
    pub lhs: Expr,
    pub op: BinaryOp,
    pub rhs: Expr,
    pub span: Span,
}
impl CompoundAssignStmt {
    pub fn new(lhs: Expr, op: BinaryOp, rhs: Expr, span: Span) -> Self {
        Self { lhs, op, rhs, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct AssertStmt {
    pub expr: Box<Expr>,
    pub msg: Option<String>,
    pub span: Span,
}
impl AssertStmt {
    pub fn new(expr: Box<Expr>, msg: Option<String>, span: Span) -> Self {
        Self { expr, msg, span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct LoopStmt {
    pub invariants: Vec<Expr>,
    pub body: Vec<Statement>,
    pub span: Span,
}
impl LoopStmt {
    pub fn new(invariants: Vec<Expr>, body: Vec<Statement>, span: Span) -> Self {
        Self {
            invariants,
            body,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct BreakStmt {
    pub span: Span,
}
impl BreakStmt {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ContinueStmt {
    pub span: Span,
}
impl ContinueStmt {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct MacroCallStmt {
    pub name: Symbol,
    pub token_tree: TokenTree,
    pub block_tree: Option<TokenTree>,
    pub has_semi: bool,
    pub span: Span,
}
impl MacroCallStmt {
    pub fn new(
        name: String,
        token_tree: TokenTree,
        block_tree: Option<TokenTree>,
        has_semi: bool,
        span: Span,
    ) -> Self {
        Self {
            name: name.into(),
            token_tree,
            block_tree,
            has_semi,
            span,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Statement {
    LetDecl(LetDeclStmt),
    Return(ReturnStmt),
    ExprStmt(ExprStmtStmt),
    ForLoop(ForLoopStmt),
    Assign(AssignStmt),
    CompoundAssign(CompoundAssignStmt),
    Assert(AssertStmt),
    Loop(LoopStmt),
    Break(BreakStmt),
    Continue(ContinueStmt),
    MacroCall(MacroCallStmt),
    Error(Span),
}

macro_rules! delegate_stmt {
    ($self:ident, $inner:ident => $expr:expr) => {
        match $self {
            Statement::LetDecl($inner) => $expr,
            Statement::Return($inner) => $expr,
            Statement::ExprStmt($inner) => $expr,
            Statement::ForLoop($inner) => $expr,
            Statement::Assign($inner) => $expr,
            Statement::CompoundAssign($inner) => $expr,
            Statement::Assert($inner) => $expr,
            Statement::Loop($inner) => $expr,
            Statement::Break($inner) => $expr,
            Statement::Continue($inner) => $expr,
            Statement::MacroCall($inner) => $expr,
            Statement::Error(_) => unreachable!(),
        }
    };
}

impl Statement {
    pub fn substitute(
        &self,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> Statement {
        match self {
            Statement::LetDecl(e) => Statement::LetDecl(LetDeclStmt {
                name: e.name.clone(),
                is_mut: e.is_mut,
                ty_ann: e.ty_ann.as_ref().map(|t| t.substitute(mapping)),
                expr: e.expr.substitute(mapping),
                span: e.span,
            }),
            Statement::Return(e) => Statement::Return(ReturnStmt {
                expr: e.expr.substitute(mapping),
                span: e.span,
            }),
            Statement::ExprStmt(e) => Statement::ExprStmt(ExprStmtStmt {
                expr: e.expr.substitute(mapping),
                has_semi: e.has_semi,
                span: e.span,
            }),
            Statement::ForLoop(e) => Statement::ForLoop(ForLoopStmt {
                iter: e.iter.clone(),
                iterable: Box::new(e.iterable.substitute(mapping)),
                invariants: e
                    .invariants
                    .iter()
                    .map(|expr| expr.substitute(mapping))
                    .collect(),
                body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
                span: e.span,
            }),
            Statement::Assign(e) => Statement::Assign(AssignStmt {
                lhs: e.lhs.substitute(mapping),
                rhs: e.rhs.substitute(mapping),
                span: e.span,
            }),
            Statement::CompoundAssign(e) => Statement::CompoundAssign(CompoundAssignStmt {
                lhs: e.lhs.substitute(mapping),
                op: e.op.clone(),
                rhs: e.rhs.substitute(mapping),
                span: e.span,
            }),
            Statement::Assert(e) => Statement::Assert(AssertStmt {
                expr: Box::new(e.expr.substitute(mapping)),
                msg: e.msg.clone(),
                span: e.span,
            }),
            Statement::Loop(e) => Statement::Loop(LoopStmt {
                invariants: e
                    .invariants
                    .iter()
                    .map(|expr| expr.substitute(mapping))
                    .collect(),
                body: e.body.iter().map(|s| s.substitute(mapping)).collect(),
                span: e.span,
            }),
            Statement::Break(e) => Statement::Break(e.clone()),
            Statement::Continue(e) => Statement::Continue(e.clone()),
            Statement::MacroCall(e) => Statement::MacroCall(MacroCallStmt {
                name: e.name.clone(),
                token_tree: e.token_tree.clone(),
                block_tree: e.block_tree.clone(),
                has_semi: e.has_semi,
                span: e.span,
            }),
            Statement::Error(s) => Statement::Error(*s),
        }
    }

    pub fn span(&self) -> Span {
        delegate_stmt!(self, s => s.span)
    }
}
