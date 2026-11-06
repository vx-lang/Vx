//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
    /// `None` for a bare `return;`, which only a `void` function may write. Modelled as an
    /// absence rather than a synthesized zero: a `void` function has no value to return, and
    /// inventing one would have to be typed as something.
    pub expr: Option<Expr>,
    pub span: Span,
}
impl ReturnStmt {
    pub fn new(expr: Option<Expr>, span: Span) -> Self {
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
    /// The `next` the checker resolved for this loop's iterator, by its instance name. Two
    /// instances of one generic iterator have two `next`s, and only the checker knows which.
    pub next_fn: Option<crate::symbol::Symbol>,
    /// The bounds of a loop that the code generator writes out once for each pass, set by the
    /// checker when the loop variable indexes a tensor that must stay in registers.
    pub unroll: Option<(i64, i64)>,
    /// `across <level>` or `across <level>(n)`: the loop is one axis of a GPU launch.
    pub across: Option<AcrossClause>,
}

/// `across lanes` or `across lanes(n)` on a `for`. `level` is a level of the region's
/// topology, and `share` is the `n` that the passes are shared among.
#[derive(Debug, PartialEq, Clone)]
pub struct AcrossClause {
    pub level: Symbol,
    pub share: Option<Box<Expr>>,
    pub span: Span,
    /// What the checker found: the kind of the level, and the lanes of a subgroup of the
    /// topology. The code generator reads them. `None` before the check.
    pub kind: Option<crate::arch::LevelKind>,
    pub subgroup_lanes: Option<u64>,
    /// Set by the checker under `-fauto-schedule-levels`, when the code generator of the
    /// region's arch cannot launch the levels. The loop runs as a plain `for` that carries its
    /// level as metadata.
    pub auto_schedule: bool,
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
            next_fn: None,
            unroll: None,
            across: None,
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
    /// The accumulator's element type, filled in by the checker, for the same reason
    /// `BinaryOpExpr` carries one: `/=`, `%=` and `>>=` need the sign.
    pub operand_ty: Option<crate::syntax::ElementType>,
}
impl CompoundAssignStmt {
    pub fn new(lhs: Expr, op: BinaryOp, rhs: Expr, span: Span) -> Self {
        Self {
            lhs,
            op,
            rhs,
            span,
            operand_ty: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssertStmt {
    pub expr: Box<Expr>,
    pub msg: Option<String>,
    pub span: Span,
    /// `<file>:<line>` of the assert, which a failed assert prints. The checker sets it.
    pub at: Option<String>,
}

// `at` comes from `span`, so two asserts are equal without it. The checker can walk a function
// twice, and it finds the asserts that it added the first time by equality.
impl PartialEq for AssertStmt {
    fn eq(&self, other: &Self) -> bool {
        self.expr == other.expr && self.msg == other.msg && self.span == other.span
    }
}
impl AssertStmt {
    pub fn new(expr: Box<Expr>, msg: Option<String>, span: Span) -> Self {
        Self {
            expr,
            msg,
            span,
            at: None,
        }
    }

    /// What a failed assert prints: its message, and `  at <file>:<line>` on the next line.
    pub fn failure_message(&self) -> String {
        let msg = self.msg.as_deref().unwrap_or("assertion failed");
        match &self.at {
            Some(at) => format!("{msg}\n  at {at}"),
            None => msg.to_string(),
        }
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
    /// Set by the checker for the `join()` of a task that failed: then this leaves the innermost
    /// `tasks` block, not a loop, and this `Err` is the value of the block. No program writes it.
    pub tasks_value: Option<Box<super::Expr>>,
}
impl BreakStmt {
    pub fn new(span: Span) -> Self {
        Self {
            span,
            tasks_value: None,
        }
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

/// The call that turns cancellation off before a `drop`, with the drop's time limit in seconds.
pub const CANCEL_OFF: &str = "vx::cancel_off";
/// The call that turns cancellation back on after a `drop`.
pub const CANCEL_ON: &str = "vx::cancel_on";
/// `assume e;`: the checker adds the fact `e` without a proof. Only an `unsafe` block can hold
/// it, and it generates no code.
pub const ASSUME: &str = "vx::assume";

/// `Statement::Drop`: the tensor named here is freed at this point.
#[derive(Debug, PartialEq, Clone)]
pub struct DropStmt {
    pub name: Symbol,
    /// The fields to follow from `name` to the tensor freed: `["inner", "t"]` for
    /// `name.inner.t`. Empty when `name` is the tensor.
    pub path: Vec<Symbol>,
    /// The `drop` of a type that implements `Drop`, by its mangled name: then this drop calls
    /// it with `&mut name.path` instead of freeing a tensor.
    pub call: Option<Symbol>,
    /// What the code generators lower, built by the checker: the call `call(&mut name.path)`,
    /// or the place `name.path` of the tensor to free. `None` when `name` is the tensor.
    pub expr: Option<Box<super::Expr>>,
    /// A `bool` local that is true once the tensor was moved: then nothing is freed.
    pub flag: Option<Symbol>,
    /// Freed only once the next statement has computed its value: a `return`'s, a block's last
    /// expression, or an assignment's right-hand side, which may still read the tensor.
    pub after_value: bool,
    /// Frees nothing: the struct or enum `name` is no longer used, so its stack slot can hold
    /// another value. It comes after the other drops of `name`.
    pub ends_lifetime: bool,
    pub span: Span,
    /// With `call`: how long the waits in the `drop` can take, in nanoseconds. The call
    /// `wait_limit_nanos(&name.path)`, or the default when the impl has no copy of it.
    pub limit: Option<Box<super::Expr>>,
}

impl DropStmt {
    /// For a drop that calls a `Drop` impl, or drops an enum's payload: the statements that do
    /// it, inside `if !moved { .. }` when the value may have been moved.
    ///
    /// A `Drop` impl's `drop` runs with cancellation off (F6), so that a wait in it can finish,
    /// until its time limit (F6a): `vx::cancel_off(wait_limit_nanos(&x)); drop(&mut x);
    /// vx::cancel_on();`. The code generators lower the two `vx::` calls to the runtime.
    pub fn call_statements(&self) -> Option<Vec<Statement>> {
        let stmt = |e: Expr| Statement::ExprStmt(ExprStmtStmt::new(e, true, Span::default()));
        let runtime = |name: &str, args: Vec<Expr>| {
            stmt(Expr::FunctionCall(FunctionCallExpr::new(
                name.into(),
                None,
                args,
                Span::default(),
            )))
        };
        let payload = matches!(self.expr.as_deref(), Some(Expr::Match(_)));
        if self.call.is_none() && !payload {
            return None;
        }
        let call = self
            .expr
            .as_deref()
            .expect("a drop that runs code has it")
            .clone();
        let mut body = vec![stmt(call)];
        if self.call.is_some() {
            let limit = self
                .limit
                .as_deref()
                .expect("a drop that calls `Drop::drop` asks `wait_limit`")
                .clone();
            body.insert(0, runtime(CANCEL_OFF, vec![limit]));
            body.push(runtime(CANCEL_ON, Vec::new()));
        }
        let Some(flag) = &self.flag else {
            return Some(body);
        };
        let not_moved = Expr::UnaryOp(UnaryOpExpr::new(
            UnaryOp::Not,
            Box::new(Expr::Identifier(IdentifierExpr::new(
                flag.clone(),
                Span::default(),
            ))),
            Span::default(),
        ));
        Some(vec![Statement::ExprStmt(ExprStmtStmt::new(
            Expr::If(IfExpr::new(
                false,
                Box::new(not_moved),
                body,
                None,
                Span::default(),
            )),
            true,
            Span::default(),
        ))])
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
    /// Free what the tensor `name` owns. Never written by a programmer: the checker puts it
    /// where the tensor's drop point is.
    Drop(DropStmt),
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
            Statement::Drop($inner) => $expr,
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
                expr: e.expr.as_ref().map(|x| x.substitute(mapping)),
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
                next_fn: e.next_fn.clone(),
                unroll: e.unroll,
                across: e.across.as_ref().map(|a| AcrossClause {
                    level: a.level.clone(),
                    share: a.share.as_ref().map(|n| Box::new(n.substitute(mapping))),
                    span: a.span,
                    kind: a.kind,
                    subgroup_lanes: a.subgroup_lanes,
                    auto_schedule: a.auto_schedule,
                }),
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
                operand_ty: e.operand_ty.clone(),
            }),
            Statement::Assert(e) => Statement::Assert(AssertStmt {
                expr: Box::new(e.expr.substitute(mapping)),
                msg: e.msg.clone(),
                span: e.span,
                at: e.at.clone(),
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
            Statement::Drop(d) => Statement::Drop(d.clone()),
            Statement::Error(s) => Statement::Error(*s),
        }
    }

    pub fn span(&self) -> Span {
        delegate_stmt!(self, s => s.span)
    }
}
