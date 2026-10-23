//===- across.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Where an `across` loop may be, and what its size may be (E6032). The design is in #252: a
// region with `across` has one `blocks` nest at its top level, the levels nest blocks, then
// subgroups, then lanes, and a block's size is known when the kernel is compiled.
//
//===----------------------------------------------------------------------===//

use crate::arch::{Level, LevelKind};
use crate::syntax::stmt::{ForLoopStmt, Statement};
use crate::syntax::{Expr, Span};
use std::collections::HashSet;

/// The `spawn on` region that a statement is in.
#[derive(Clone)]
struct Region {
    /// The name of its topology, for messages.
    name: String,
    levels: Option<Vec<Level>>,
    /// The names that the region declares. A `blocks` bound may not read them: the host
    /// computes it before the region starts.
    locals: HashSet<String>,
}

/// One enclosing `across` loop.
#[derive(Clone)]
struct Axis {
    kind: LevelKind,
    /// The lanes of a `lanes` loop, or the trip of a `subgroups` loop.
    size: Option<u64>,
}

#[derive(Clone, Default)]
struct Ctx {
    region: Option<Region>,
    axes: Vec<Axis>,
    /// Plain loops opened since the innermost `across` loop. A `break` or a `continue` with
    /// none would leave the `across` loop.
    plain_loops: usize,
    /// Whether the statement is directly in the region, not in an `if`, a loop or a block.
    top: bool,
    /// Inside a closure, where `return` leaves the closure and not the loop.
    in_closure: bool,
}

struct Walk<'g> {
    graph: &'g crate::arch::TransferCostGraph,
    errors: Vec<(String, Span)>,
}

impl crate::hir::TypeChecker<'_> {
    /// E6032 for each `across` loop in `body` that its place or its size does not allow.
    pub(crate) fn check_across_loops(&mut self, body: &[Statement]) {
        let mut walk = Walk {
            graph: self.transfer_cost_graph,
            errors: Vec::new(),
        };
        walk.stmts(body, &Ctx::default());
        for (msg, at) in walk.errors {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6032,
                msg,
                Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
            );
        }
    }
}

/// The integer that `e` is when it is a literal.
fn literal(e: &Expr) -> Option<u64> {
    match e {
        Expr::Number(n) => n.value.parse().ok(),
        _ => None,
    }
}

fn kind_word(kind: LevelKind) -> &'static str {
    match kind {
        LevelKind::Blocks => "block",
        LevelKind::Subgroups => "subgroup",
        LevelKind::Lanes => "lane",
    }
}

impl Walk<'_> {
    fn error(&mut self, msg: String, at: Span) {
        self.errors.push((msg, at));
    }

    fn stmts(&mut self, stmts: &[Statement], ctx: &Ctx) {
        let mut ctx = ctx.clone();
        for s in stmts {
            self.stmt(s, &ctx);
            // A later `blocks` bound may not read a name that the region declares.
            if let (Statement::LetDecl(l), Some(r)) = (s, ctx.region.as_mut()) {
                r.locals.insert(l.name.as_ref().to_string());
            }
        }
    }

    /// `ctx` for the statements of a block that is not the region itself.
    fn inner(ctx: &Ctx) -> Ctx {
        Ctx {
            top: false,
            ..ctx.clone()
        }
    }

    fn stmt(&mut self, s: &Statement, ctx: &Ctx) {
        match s {
            Statement::ForLoop(f) => {
                self.expr(&f.iterable, ctx);
                if f.across.is_some() {
                    self.across(f, ctx);
                } else {
                    let mut body = Self::inner(ctx);
                    body.plain_loops += 1;
                    self.stmts(&f.body, &body);
                }
            }
            Statement::Loop(l) => {
                let mut body = Self::inner(ctx);
                body.plain_loops += 1;
                self.stmts(&l.body, &body);
            }
            Statement::Break(b) if !ctx.axes.is_empty() && ctx.plain_loops == 0 => self.error(
                "`break` may not leave an `across` loop: every lane must reach the end of the loop"
                    .to_string(),
                b.span,
            ),
            Statement::Continue(c) if !ctx.axes.is_empty() && ctx.plain_loops == 0 => self.error(
                "`continue` may not leave a pass of an `across` loop: every lane must reach the \
                 end of the loop"
                    .to_string(),
                c.span,
            ),
            Statement::Return(r) => {
                if !ctx.axes.is_empty() && !ctx.in_closure {
                    self.error(
                        "`return` may not leave an `across` loop: every lane must reach the end \
                         of the loop"
                            .to_string(),
                        r.span,
                    );
                }
                if let Some(e) = &r.expr {
                    self.expr(e, ctx);
                }
            }
            Statement::LetDecl(l) => self.expr(&l.expr, ctx),
            Statement::ExprStmt(e) => self.expr(&e.expr, ctx),
            Statement::Assign(a) => {
                self.expr(&a.lhs, ctx);
                self.expr(&a.rhs, ctx);
            }
            Statement::CompoundAssign(a) => {
                self.expr(&a.lhs, ctx);
                self.expr(&a.rhs, ctx);
            }
            Statement::Assert(a) => self.expr(&a.expr, ctx),
            Statement::Break(_)
            | Statement::Continue(_)
            | Statement::MacroCall(_)
            | Statement::Drop(_)
            | Statement::Error(_) => {}
        }
    }

    /// The expressions in `e` that hold statements, with the context of each.
    fn expr(&mut self, e: &Expr, ctx: &Ctx) {
        match e {
            Expr::SpawnOn(sp) => {
                let kind = sp.top.kind();
                let levels = self
                    .graph
                    .descriptor(&kind)
                    .and_then(|d| d.levels_of(&kind));
                let region = Ctx {
                    region: Some(Region {
                        name: sp.top.display_name(),
                        levels,
                        locals: HashSet::new(),
                    }),
                    axes: Vec::new(),
                    plain_loops: 0,
                    top: true,
                    in_closure: false,
                };
                self.region_nests(&sp.stmts, &region);
                self.stmts(&sp.stmts, &region);
                if let Some(r) = &sp.ret {
                    self.expr(r, &Self::inner(&region));
                }
            }
            // A task runs on a CPU thread of its own, not in the region around it.
            Expr::TaskBlock(b) => {
                let ctx = Ctx::default();
                self.stmts(&b.stmts, &ctx);
                if let Some(r) = &b.ret {
                    self.expr(r, &ctx);
                }
            }
            Expr::If(i) => {
                self.expr(&i.cond, ctx);
                let inner = Self::inner(ctx);
                self.stmts(&i.then_block, &inner);
                if let Some(b) = &i.else_block {
                    self.stmts(b, &inner);
                }
            }
            Expr::Match(m) => {
                self.expr(&m.expr, ctx);
                let inner = Self::inner(ctx);
                for arm in &m.arms {
                    self.stmts(&arm.body, &inner);
                }
            }
            Expr::UnsafeBlock(b) => {
                let inner = Self::inner(ctx);
                self.stmts(&b.stmts, &inner);
                if let Some(r) = &b.ret {
                    self.expr(r, &inner);
                }
            }
            Expr::ComptimeBlock(b) => {
                let inner = Self::inner(ctx);
                self.stmts(&b.stmts, &inner);
                if let Some(r) = &b.ret {
                    self.expr(r, &inner);
                }
            }
            Expr::Closure(c) => {
                let mut inner = Self::inner(ctx);
                inner.in_closure = true;
                self.expr(&c.body, &inner);
            }
            // The other expressions hold no statements, but they can hold an expression that
            // does, such as `1 + if c { .. } else { .. }`.
            _ => {
                let mut subs: Vec<Expr> = Vec::new();
                crate::syntax::visit_expr(e, &mut |s| {
                    if !std::ptr::eq(s, e) && holds_statements(s) {
                        subs.push(s.clone());
                    }
                });
                // `visit_expr` also visits what a found expression holds; keep the outermost.
                let mut seen: Vec<Expr> = Vec::new();
                for s in subs {
                    if !seen.iter().any(|o| contains(o, &s)) {
                        seen.push(s);
                    }
                }
                for s in &seen {
                    self.expr(s, ctx);
                }
            }
        }
    }

    /// L10: a region with `across` has one `blocks` nest, directly in it.
    fn region_nests(&mut self, stmts: &[Statement], region: &Ctx) {
        let blocks = region
            .region
            .as_ref()
            .and_then(|r| r.levels.as_ref())
            .and_then(|l| l.first())
            .map(|l| l.name.clone());
        let nests: Vec<&ForLoopStmt> = stmts
            .iter()
            .filter_map(|s| match s {
                Statement::ForLoop(f) if f.across.as_ref().map(|a| &a.level) == blocks.as_ref() => {
                    Some(f)
                }
                _ => None,
            })
            .collect();
        for extra in nests.iter().skip(1) {
            self.error(
                "a region has one `across` nest of blocks. A second one would be a second launch, \
                 which Vx does not have yet"
                    .to_string(),
                extra.across.as_ref().map_or(extra.span, |a| a.span),
            );
        }
    }

    fn across(&mut self, f: &ForLoopStmt, ctx: &Ctx) {
        let a = f.across.as_ref().expect("an across loop");
        let Some(region) = &ctx.region else {
            self.error(
                "an `across` loop is valid only in a `spawn on` region".to_string(),
                a.span,
            );
            return;
        };
        let Some(levels) = &region.levels else {
            self.error(
                format!(
                    "`across {}`: the topology `{}` has no levels, so its regions use plain `for` \
                     loops. A GPU topology has `blocks`, `subgroups` and `lanes`",
                    a.level, region.name
                ),
                a.span,
            );
            return;
        };
        let Some(level) = levels.iter().find(|l| l.name == a.level) else {
            let names: Vec<String> = levels.iter().map(|l| format!("`{}`", l.name)).collect();
            self.error(
                format!(
                    "`across {}`: the topology `{}` has the levels {}",
                    a.level,
                    region.name,
                    names.join(", ")
                ),
                a.span,
            );
            return;
        };
        let Expr::Range(range) = &*f.iterable else {
            self.error(
                format!(
                    "an `across` loop goes over a range such as `0..n`, so that its bound is the \
                     size of the {} level",
                    kind_word(level.kind)
                ),
                f.iterable.span(),
            );
            return;
        };
        let max_lanes = levels.iter().find_map(|l| l.max_lanes);
        let subgroup = levels.iter().find_map(|l| l.lanes);
        let outer = ctx.axes.last();

        // N1: blocks, then subgroups, then lanes.
        if let Some(o) = outer {
            let rank = |k: LevelKind| match k {
                LevelKind::Blocks => 0,
                LevelKind::Subgroups => 1,
                LevelKind::Lanes => 2,
            };
            if rank(level.kind) < rank(o.kind) {
                self.error(
                    format!(
                        "a {} loop may not be inside a {} loop: the levels nest blocks, then \
                         subgroups, then lanes",
                        kind_word(level.kind),
                        kind_word(o.kind)
                    ),
                    a.span,
                );
                return;
            }
        }
        let same = ctx.axes.iter().filter(|x| x.kind == level.kind).count();
        match level.kind {
            LevelKind::Blocks => {
                if !ctx.top && ctx.axes.is_empty() {
                    self.error(
                        "the `across` loop of blocks is directly in the `spawn on` region, not in \
                         an `if`, a loop or a block"
                            .to_string(),
                        a.span,
                    );
                    return;
                }
                if same >= 3 {
                    self.error(
                        "a region has at most three axes of blocks".to_string(),
                        a.span,
                    );
                    return;
                }
                // L2: the host computes the size before the region starts.
                let mut bounds = vec![&*range.start, &*range.end];
                bounds.extend(a.share.as_deref());
                for b in bounds {
                    let mut read = None;
                    crate::syntax::visit_expr(b, &mut |e| {
                        if let Expr::Identifier(id) = e {
                            if read.is_none() && region.locals.contains(id.name.as_ref()) {
                                read = Some(id.name.as_ref().to_string());
                            }
                        }
                    });
                    if let Some(name) = read {
                        self.error(
                            format!(
                                "the size of an `across` loop of blocks reads `{name}`, which the \
                                 region declares. The host computes the size before the region \
                                 starts, so it may read only values from outside the region"
                            ),
                            b.span(),
                        );
                    }
                }
            }
            LevelKind::Subgroups | LevelKind::Lanes => {
                if !ctx.axes.iter().any(|x| x.kind == LevelKind::Blocks) {
                    self.error(
                        format!(
                            "a {} loop is valid only inside the `across` loop of blocks",
                            kind_word(level.kind)
                        ),
                        a.span,
                    );
                    return;
                }
            }
        }
        // L2: the size of a subgroup or lane loop is known when the kernel is compiled.
        let size = match level.kind {
            LevelKind::Blocks => None,
            _ => {
                let size = match a.share.as_deref() {
                    Some(n) => literal(n),
                    None => match (literal(&range.start), literal(&range.end)) {
                        (Some(lo), Some(hi)) if hi > lo => Some(hi - lo),
                        _ => None,
                    },
                };
                let Some(size) = size else {
                    let what = if a.share.is_some() {
                        format!("the `n` of `across {}(n)`", a.level)
                    } else {
                        "the bound of the loop".to_string()
                    };
                    self.error(
                        format!(
                            "{what} is a number such as 256, because it is the size of a block \
                             when the kernel is compiled. To go over a size that the program \
                             computes, write `across {}(n)` with a number `n`",
                            a.level
                        ),
                        a.share.as_ref().map_or(f.iterable.span(), |n| n.span()),
                    );
                    return;
                };
                Some(size)
            }
        };
        let axis = Axis {
            kind: level.kind,
            size,
        };
        match level.kind {
            LevelKind::Blocks => {}
            LevelKind::Subgroups => {
                if same >= 1 {
                    self.error(
                        "a subgroup loop may not be inside another one".to_string(),
                        a.span,
                    );
                    return;
                }
                if let (Some(n), Some(width), Some(max)) = (size, subgroup, max_lanes) {
                    if n * width > max {
                        self.error(
                            format!(
                                "{n} subgroups of {width} lanes are {} lanes, but a block of `{}` \
                                 has at most {max}",
                                n * width,
                                region.name
                            ),
                            a.span,
                        );
                        return;
                    }
                }
            }
            LevelKind::Lanes => {
                let in_subgroup = ctx.axes.iter().any(|x| x.kind == LevelKind::Subgroups);
                if in_subgroup {
                    if same >= 1 {
                        self.error(
                            "in a subgroup loop, a lane loop may not be inside another one"
                                .to_string(),
                            a.span,
                        );
                        return;
                    }
                    if let (Some(n), Some(width)) = (size, subgroup) {
                        if n != width {
                            self.error(
                                format!(
                                    "a lane loop in a subgroup loop runs once for each lane of \
                                     the subgroup, so it has {width} passes or is \
                                     `across {}({width})`, not {n}",
                                    a.level
                                ),
                                a.span,
                            );
                            return;
                        }
                    }
                } else {
                    if same >= 3 {
                        self.error(
                            "a block has at most three axes of lanes".to_string(),
                            a.span,
                        );
                        return;
                    }
                    let lanes: u64 = ctx
                        .axes
                        .iter()
                        .filter(|x| x.kind == LevelKind::Lanes)
                        .filter_map(|x| x.size)
                        .product::<u64>()
                        * size.unwrap_or(1);
                    if let Some(max) = max_lanes {
                        if lanes > max {
                            let hint = if same == 0 {
                                format!(
                                    ". Write `across {}(n)` to share the passes among `n` lanes",
                                    a.level
                                )
                            } else {
                                String::new()
                            };
                            self.error(
                                format!(
                                    "this loop asks for {lanes} lanes, but a block of `{}` has at \
                                     most {max}{hint}",
                                    region.name
                                ),
                                a.span,
                            );
                            return;
                        }
                    }
                }
            }
        }
        // N2: an inner axis of the same level is the whole body of the outer one.
        if level.kind != LevelKind::Subgroups {
            for s in &f.body {
                if let Statement::ForLoop(inner) = s {
                    if inner.across.as_ref().map(|x| &x.level) == Some(&a.level)
                        && f.body.len() != 1
                    {
                        self.error(
                            format!(
                                "a {} loop inside another one is the whole body of the outer loop",
                                kind_word(level.kind)
                            ),
                            inner.across.as_ref().map_or(inner.span, |x| x.span),
                        );
                    }
                }
            }
        }
        let mut body = Self::inner(ctx);
        body.axes.push(axis);
        body.plain_loops = 0;
        if let Some(r) = body.region.as_mut() {
            r.locals.insert(f.iter.clone());
        }
        self.stmts(&f.body, &body);
    }
}

/// Whether `e` holds statements of its own.
fn holds_statements(e: &Expr) -> bool {
    matches!(
        e,
        Expr::SpawnOn(_)
            | Expr::TaskBlock(_)
            | Expr::If(_)
            | Expr::Match(_)
            | Expr::UnsafeBlock(_)
            | Expr::ComptimeBlock(_)
            | Expr::Closure(_)
    )
}

/// Whether `inner` is inside `outer`.
fn contains(outer: &Expr, inner: &Expr) -> bool {
    let mut found = false;
    crate::syntax::visit_expr(outer, &mut |e| found |= e == inner);
    found
}
