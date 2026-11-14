//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A wait in a cancelled task gives `Cancelled` at once, every time. So a `loop` or `while` that
// matches `Cancelled` and waits again never ends. An arm that changes the condition of its
// `while` can end it, and so does a `while` that changes its condition on every pass. W1035 warns about such an arm (F4 in
// docs/discussions/waiting_tasks_channels_and_task_errors.md). It is a warning, not an error,
// because the check does not look into the functions that the arm calls.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::syntax::{MatchArm, Pattern};

impl TypeChecker<'_> {
    /// W1035 for each arm of a `match` on `ty` that can take `Cancelled` and stays in the loop.
    pub(crate) fn warn_wait_again_after_cancel(&mut self, ty: &Type, arms: &[MatchArm]) {
        let Some(Some(exits)) = self.cancel_loops.last() else {
            return;
        };
        if self.speculating || !self.can_be_cancelled(ty) {
            return;
        }
        let exits = exits.clone();
        for arm in arms {
            if !takes_cancelled(&arm.pattern)
                || leaves_the_loop(&arm.body)
                || assigns(&arm.body, &exits)
            {
                continue;
            }
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1035,
                "this arm can take 'Cancelled' and stays in the loop. In a cancelled task each \
                 wait gives 'Cancelled' at once, so the loop never ends. Leave the loop in this \
                 arm, or give the error a name and check it for 'Cancelled'",
                Some(crate::diagnostic::SourceSpan::from_ast_span(&arm.span)),
            );
        }
    }

    /// Whether `ty` is a `Result` whose error can say that the task was cancelled: an enum with
    /// a `Cancelled` case, such as `RecvError`, or a struct with a `cancelled` field, such as
    /// `SendError<T>`.
    fn can_be_cancelled(&self, ty: &Type) -> bool {
        let Some((_, err)) = super::tasks::result_parts(ty) else {
            return false;
        };
        let name = match &err {
            Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
            Type::GenericInstance(base, _) => match base.as_ref() {
                Type::Enum(n, _) | Type::Struct(n, _) => n.clone(),
                _ => return false,
            },
            _ => return false,
        };
        let base = name.split('<').next().unwrap_or(&name);
        if let Some(e) = self.env.enums.get(base) {
            return e.variants.iter().any(|(v, _)| v.as_ref() == "Cancelled");
        }
        if let Some(s) = self.env.structs.get(base) {
            return s.fields.iter().any(|(f, _)| f.as_ref() == "cancelled");
        }
        false
    }
}

/// What W1035 needs to know about the loop with body `body`. A `while c { .. }` is parsed as
/// `loop { if c { .. } else { break; } }`. When its body changes a variable of `c` in a
/// statement of its own, each pass changes the condition, so the loop ends: `None`. Otherwise the
/// variables of `c`, and none for a plain `loop`.
pub(crate) fn loop_exits(body: &[Statement]) -> Option<Vec<String>> {
    let [Statement::ExprStmt(e)] = body else {
        return Some(Vec::new());
    };
    let Expr::If(i) = &e.expr else {
        return Some(Vec::new());
    };
    if !matches!(i.else_block.as_deref(), Some([Statement::Break(_)])) {
        return Some(Vec::new());
    }
    let mut vars = Vec::new();
    crate::syntax::visit_expr(&i.cond, &mut |e| {
        if let Expr::Identifier(id) = e {
            vars.push(id.name.to_string());
        }
    });
    let changes =
        |lhs: &Expr| matches!(lhs, Expr::Identifier(id) if vars.contains(&id.name.to_string()));
    let every_pass = i.then_block.iter().any(|s| match s {
        Statement::Assign(a) => changes(&a.lhs),
        Statement::CompoundAssign(a) => changes(&a.lhs),
        _ => false,
    });
    (!every_pass).then_some(vars)
}

/// Whether `stmts` assign one of `vars`, at any depth.
fn assigns(stmts: &[Statement], vars: &[String]) -> bool {
    if vars.is_empty() {
        return false;
    }
    let changes =
        |lhs: &Expr| matches!(lhs, Expr::Identifier(id) if vars.contains(&id.name.to_string()));
    let mut found = stmts.iter().any(|s| match s {
        Statement::Assign(a) => changes(&a.lhs),
        Statement::CompoundAssign(a) => changes(&a.lhs),
        Statement::Loop(l) => assigns(&l.body, vars),
        Statement::ForLoop(l) => assigns(&l.body, vars),
        _ => false,
    });
    crate::syntax::visit_exprs(stmts, &mut |e| match e {
        Expr::If(i) => {
            found |= assigns(&i.then_block, vars)
                || i.else_block.as_deref().is_some_and(|b| assigns(b, vars));
        }
        Expr::Match(m) => found |= m.arms.iter().any(|a| assigns(&a.body, vars)),
        _ => {}
    });
    found
}

/// Whether an arm with this pattern takes `Cancelled`: `_`, `Err(_)`, or a name that starts with
/// `_`. An arm that binds the error to a used name can look at it, so it gets no warning.
fn takes_cancelled(p: &Pattern) -> bool {
    let ignored = |p: &Pattern| match p {
        Pattern::Wildcard => true,
        Pattern::Identifier(n) => n.starts_with('_'),
        _ => false,
    };
    match p {
        Pattern::EnumVariant(_, variant, Some(payload)) if variant.as_ref() == "Err" => {
            matches!(payload.as_slice(), [one] if ignored(one))
        }
        other => ignored(other),
    }
}

/// Whether `stmts` can leave the loop around them: a `break` that is not in an inner loop, a
/// `return`, or a `panic` or `abort`. A `break` inside an `if` or a `match` counts.
fn leaves_the_loop(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Break(_) | Statement::Return(_) => true,
        // A `return` in an inner loop also leaves this one.
        Statement::Loop(l) => returns(&l.body),
        Statement::ForLoop(l) => returns(&l.body),
        _ => {
            let mut found = false;
            crate::syntax::visit_exprs(std::slice::from_ref(s), &mut |e| match e {
                Expr::If(i) => {
                    found |= leaves_the_loop(&i.then_block)
                        || i.else_block.as_deref().is_some_and(leaves_the_loop);
                }
                Expr::Match(m) => found |= m.arms.iter().any(|a| leaves_the_loop(&a.body)),
                Expr::FunctionCall(c) => found |= crate::syntax::is_abort_or_panic(c),
                _ => {}
            });
            found
        }
    })
}

/// Whether `stmts` hold a `return`, at any depth.
fn returns(stmts: &[Statement]) -> bool {
    let mut found = stmts.iter().any(|s| matches!(s, Statement::Return(_)));
    crate::syntax::visit_exprs(stmts, &mut |e| match e {
        Expr::If(i) => {
            found |= returns(&i.then_block) || i.else_block.as_deref().is_some_and(returns);
        }
        Expr::Match(m) => found |= m.arms.iter().any(|a| returns(&a.body)),
        _ => {}
    });
    found
}
