//===- across_lanes.rs - Vx Compiler --------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What the lanes of a block share, for the rules of #252 that keep every lane at each barrier:
// which code only lane 0 runs (L5), and which values are the same for every lane (L8).
//
//===----------------------------------------------------------------------===//

use crate::hir::flatten::is_tensor_alloc_call;
use crate::syntax::stmt::Statement;
use crate::syntax::{Expr, Span};
use std::collections::HashSet;

/// Whether `s` holds an `across` loop or a lane reduction at any depth. Every lane of the block
/// or the subgroup must get to each of them.
pub(crate) fn holds_across(s: &Statement) -> bool {
    let mut found = false;
    visit_stmts(std::slice::from_ref(s), &mut |s| {
        found |= match s {
            Statement::ForLoop(f) => f.across.is_some(),
            Statement::LetDecl(l) => crate::hir::check::lanes::bare_lanes_call(&l.expr).is_some(),
            _ => false,
        };
    });
    found
}

/// Calls `f` on each statement in `stmts`, at any depth.
pub(crate) fn visit_stmts(stmts: &[Statement], f: &mut dyn FnMut(&Statement)) {
    for s in stmts {
        f(s);
        match s {
            Statement::ForLoop(l) => visit_stmts(&l.body, f),
            Statement::Loop(l) => visit_stmts(&l.body, f),
            _ => {}
        }
        let mut blocks: Vec<Vec<Statement>> = Vec::new();
        let mut each = |e: &Expr| match e {
            Expr::If(i) => {
                blocks.push(i.then_block.clone());
                if let Some(b) = &i.else_block {
                    blocks.push(b.clone());
                }
            }
            Expr::Match(m) => m.arms.iter().for_each(|a| blocks.push(a.body.clone())),
            Expr::UnsafeBlock(b) => blocks.push(b.stmts.clone()),
            _ => {}
        };
        match s {
            Statement::LetDecl(l) => crate::syntax::visit_expr(&l.expr, &mut each),
            Statement::ExprStmt(e) => crate::syntax::visit_expr(&e.expr, &mut each),
            _ => {}
        }
        for b in &blocks {
            visit_stmts(b, f);
        }
    }
}

/// Every expression in `stmts`, at any depth.
fn visit_all_exprs(stmts: &[Statement], f: &mut dyn FnMut(&Expr)) {
    crate::syntax::visit_exprs(stmts, f);
}

/// The first name in `stmts` that is in `names`, with where it is read.
pub(crate) fn reads_any(stmts: &[Statement], names: &HashSet<String>) -> Option<(String, Span)> {
    let mut found = None;
    visit_all_exprs(stmts, &mut |e| {
        if let Expr::Identifier(id) = e {
            if found.is_none() && names.contains(id.name.as_ref()) {
                found = Some((id.name.as_ref().to_string(), id.span));
            }
        }
    });
    found
}

/// The first name in `e` that is in `names`.
pub(crate) fn expr_reads_any(e: &Expr, names: &HashSet<String>) -> Option<(String, Span)> {
    let mut found = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if let Expr::Identifier(id) = e {
            if found.is_none() && names.contains(id.name.as_ref()) {
                found = Some((id.name.as_ref().to_string(), id.span));
            }
        }
    });
    found
}

/// The root name of an assignment's target: `a` of `a`, `a[i]` and `a.f`.
fn root(e: &Expr) -> Option<&str> {
    match e {
        Expr::Identifier(id) => Some(id.name.as_ref()),
        Expr::IndexAccess(ix) => root(&ix.base),
        Expr::MemberAccess(m) => root(&m.base),
        _ => None,
    }
}

/// What a stretch of code between lane loops does (L5).
pub(crate) struct Stretch {
    /// It writes memory, or a name that it does not declare, so lane 0 runs it.
    pub writes: bool,
    /// The names that it declares, but not its tensors, which are memory of the block.
    pub values: HashSet<String>,
    /// The names that it assigns.
    pub assigned: HashSet<String>,
}

pub(crate) fn stretch(stmts: &[&Statement]) -> Stretch {
    let owned: Vec<Statement> = stmts.iter().map(|s| (*s).clone()).collect();
    let mut declared = HashSet::new();
    let mut values = HashSet::new();
    let mut tensors = HashSet::new();
    let mut writes = false;
    let mut assigned = HashSet::new();
    visit_stmts(&owned, &mut |s| match s {
        Statement::LetDecl(l) => {
            let name = l.name.as_ref().to_string();
            match crate::syntax::bare_unsafe_value(&l.expr) {
                Expr::FunctionCall(fc) if is_tensor_alloc_call(fc) => {
                    // A tensor with values written into it; `uninit` writes none.
                    writes |= fc.name.as_ref() != "Tensor::uninit";
                    tensors.insert(name.clone());
                }
                _ => {
                    values.insert(name.clone());
                }
            }
            declared.insert(name);
        }
        Statement::ForLoop(f) => {
            declared.insert(f.iter.clone());
        }
        Statement::Assign(a) => note_target(&a.lhs, &mut writes, &mut assigned),
        Statement::CompoundAssign(a) => note_target(&a.lhs, &mut writes, &mut assigned),
        _ => {}
    });
    // Only the statements at the top of the stretch declare names that later code can read.
    let top: HashSet<String> = stmts
        .iter()
        .filter_map(|s| match s {
            Statement::LetDecl(l) => Some(l.name.as_ref().to_string()),
            _ => None,
        })
        .collect();
    values.retain(|v| top.contains(v));
    writes |= assigned.iter().any(|a| !declared.contains(a));
    // A call that is not pure, a print or an `mlir!` block may write, so lane 0 runs it once.
    visit_all_exprs(&owned, &mut |e| match e {
        Expr::FunctionCall(fc) => {
            writes |= !crate::hir::check::parallel::parallel_pure_call(&fc.name)
                && !is_tensor_alloc_call(fc)
                && !fc.name.as_ref().starts_with("lanes::")
        }
        Expr::Print(_)
        | Expr::Println(_)
        | Expr::InlineMlir(_)
        | Expr::MethodCall(_)
        | Expr::IndirectCall(_) => writes = true,
        Expr::Borrow(b) => writes |= b.is_mut,
        _ => {}
    });
    Stretch {
        writes,
        values,
        assigned,
    }
}

/// Whether lane 0 alone runs the stretch `part` (L5): it writes memory, or it reads a name in
/// `lane0`, which only lane 0 has. Then `lane0` gains the names that the stretch sets.
pub(crate) fn runs_on_lane0(part: &[Statement], lane0: &mut HashSet<String>) -> bool {
    let refs: Vec<&Statement> = part.iter().collect();
    let st = stretch(&refs);
    let on_lane0 = st.writes || reads_any(part, lane0).is_some();
    if on_lane0 {
        lane0.extend(st.values);
        lane0.extend(st.assigned);
    }
    on_lane0
}

fn note_target(lhs: &Expr, writes: &mut bool, assigned: &mut HashSet<String>) {
    match lhs {
        Expr::Identifier(id) => {
            assigned.insert(id.name.as_ref().to_string());
        }
        // An element or a field is memory, which every lane sees after the barrier.
        _ => *writes = true,
    }
}

/// Why `e` may not be the same for every lane, or `None`. `uniform` says which names are, and
/// `fixed` says which tensors no plain write changes while the region runs. An element of such a
/// tensor is the same for every lane when its index is.
pub(crate) fn not_uniform(
    e: &Expr,
    uniform: &dyn Fn(&str) -> bool,
    fixed: &dyn Fn(&str) -> bool,
) -> Option<(String, Span)> {
    // Every lane gets the same result from a lane reduction.
    if crate::hir::check::lanes::bare_lanes_call(e).is_some() {
        return None;
    }
    let mut why = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if why.is_some() {
            return;
        }
        match e {
            Expr::Identifier(id) if !uniform(id.name.as_ref()) => {
                why = Some((format!("`{}` is not one of these", id.name), id.span));
            }
            Expr::IndexAccess(ix) if !element_of(ix).is_some_and(fixed) => {
                why = Some((
                    "it reads memory, which a lane may see at another time".to_string(),
                    ix.span,
                ));
            }
            Expr::FunctionCall(fc)
                if !crate::hir::check::parallel::parallel_pure_call(&fc.name) =>
            {
                why = Some((format!("it calls `{}`", fc.name), fc.span));
            }
            Expr::MethodCall(_) | Expr::IndirectCall(_) | Expr::InlineMlir(_) => {
                why = Some(("it makes a call".to_string(), e.span()));
            }
            _ => {}
        }
    });
    why
}

/// The tensor that `ix` reads an element of, when its base is a name or an element of one.
fn element_of(ix: &crate::syntax::IndexAccessExpr) -> Option<&str> {
    match &*ix.base {
        Expr::Identifier(id) => Some(id.name.as_ref()),
        Expr::IndexAccess(inner) => element_of(inner),
        _ => None,
    }
}

/// The names of the tensors that `stmts` write an element of.
pub(crate) fn written_tensors(stmts: &[Statement]) -> HashSet<String> {
    let mut names = HashSet::new();
    visit_stmts(stmts, &mut |s| match s {
        Statement::Assign(a) if !matches!(a.lhs, Expr::Identifier(_)) => {
            names.extend(root(&a.lhs).map(str::to_string));
        }
        Statement::CompoundAssign(a) if !matches!(a.lhs, Expr::Identifier(_)) => {
            names.extend(root(&a.lhs).map(str::to_string));
        }
        _ => {}
    });
    names
}

/// Why the initial value of a `let` before the nest of blocks may write memory, or `None`.
pub(crate) fn may_write(e: &Expr) -> Option<Span> {
    let mut at = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if at.is_some() {
            return;
        }
        match e {
            Expr::FunctionCall(fc)
                if !crate::hir::check::parallel::parallel_pure_call(&fc.name) =>
            {
                at = Some(fc.span)
            }
            Expr::Borrow(b) if b.is_mut => at = Some(b.span),
            Expr::MethodCall(_)
            | Expr::IndirectCall(_)
            | Expr::InlineMlir(_)
            | Expr::Print(_)
            | Expr::Println(_)
            | Expr::If(_)
            | Expr::Match(_)
            | Expr::UnsafeBlock(_)
            | Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::Closure(_) => at = Some(e.span()),
            _ => {}
        }
    });
    at
}
