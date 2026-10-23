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
use crate::hir::check::{across_lanes, lanes};
use crate::syntax::stmt::{ForLoopStmt, Statement};
use crate::syntax::{Expr, Span};
use std::collections::{HashMap, HashSet};

/// The `spawn on` region that a statement is in.
#[derive(Clone)]
struct Region {
    /// The name of its topology, for messages.
    name: String,
    levels: Option<Vec<Level>>,
    /// The `arch:` of its topology, and whether its code generator can launch the levels.
    arch: Option<String>,
    launches: bool,
    /// The names that the region declares. A `blocks` bound may not read them: the host
    /// computes it before the region starts.
    locals: HashSet<String>,
    /// The position of the region, which names it.
    span: Span,
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
    /// Names that are the same for every lane: indices of blocks and subgroups, and values
    /// computed only from them and from the host (L8). A name that the region does not declare
    /// is a value of the host.
    uniform: HashSet<String>,
    /// Names that code set which only lane 0 runs (L5). No lane loop may read them.
    lane0: HashSet<String>,
    /// In a plain loop around a lane loop, where an `if` may `break` before a barrier.
    strict_ifs: bool,
}

struct Walk<'g> {
    graph: &'g crate::arch::TransferCostGraph,
    errors: Vec<(String, Span)>,
    /// The level of each `across` loop, by the position of its clause, with the lanes of a
    /// subgroup of its topology, and whether `-fauto-schedule-levels` makes it a plain `for`.
    found: HashMap<Span, (LevelKind, Option<u64>, bool)>,
    /// `-fauto-schedule-levels`.
    auto_schedule: bool,
    /// The memories with `scope: cta` or `scope: sm`, whose tensors the lanes of a block share.
    shared: HashSet<String>,
    /// The parameters with an `atomic` clause, for each function that has one.
    contracts: HashMap<String, Vec<usize>>,
    /// The size of the tensor of each lane reduction, from the type checker.
    lane_sizes: HashMap<Span, u64>,
    /// The lane reductions in a place that allows them.
    lane_calls: Vec<LaneCall>,
}

/// A lane reduction in the code of a level.
struct LaneCall {
    at: Span,
    name: String,
    /// `Blocks` in the code of a block, `Subgroups` in the code of a subgroup.
    scope: LevelKind,
    region: Region,
}

impl crate::hir::TypeChecker<'_> {
    /// E6032 for each `across` loop in `body` that its place or its size does not allow. Each
    /// loop gets the kind of its level, for the code generator.
    pub(crate) fn check_across_loops(&mut self, body: &mut [Statement]) {
        let shared = self
            .env
            .memories
            .values()
            .filter(|d| {
                matches!(
                    d.scope,
                    Some(crate::syntax::Scope::Cta | crate::syntax::Scope::Sm)
                )
            })
            .map(|d| d.name.as_ref().to_string())
            .collect();
        let mut walk = Walk {
            graph: self.transfer_cost_graph,
            errors: Vec::new(),
            found: HashMap::new(),
            auto_schedule: self.worker.global.auto_schedule_levels,
            shared,
            contracts: self.atomic_contracts(),
            lane_sizes: self.lane_reductions.clone(),
            lane_calls: Vec::new(),
        };
        walk.stmts(body, &Ctx::default());
        mark_stmts(body, &walk.found);
        walk.lane_reductions(body);
        // The proof of an outer loop also walks the inner loops, so one place can come twice.
        let mut seen = HashSet::new();
        walk.errors
            .retain(|(_, at)| seen.insert((at.line, at.column)));
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
        // L5: the code of a blocks or subgroups body outside its lane loops falls into
        // stretches, and lane 0 runs each stretch that writes memory.
        let lane0_after = if Self::stretch_level(&ctx) {
            Self::lane0_names(stmts, &ctx.lane0)
        } else {
            vec![ctx.lane0.clone(); stmts.len()]
        };
        for (i, s) in stmts.iter().enumerate() {
            self.stmt(s, &ctx);
            ctx.lane0 = lane0_after[i].clone();
            if let Statement::LetDecl(l) = s {
                let name = l.name.as_ref().to_string();
                // L8: an immutable value from uniform names is uniform too.
                let at_top = ctx.top && ctx.axes.is_empty();
                let uniform = !l.is_mut
                    && !ctx.lane0.contains(&name)
                    && (Self::stretch_level(&ctx) || at_top)
                    && if at_top {
                        across_lanes::may_write(&l.expr).is_none()
                    } else {
                        across_lanes::not_uniform(&l.expr, &|n| Self::is_uniform(&ctx, n)).is_none()
                    };
                if uniform {
                    ctx.uniform.insert(name.clone());
                } else {
                    ctx.uniform.remove(&name);
                }
                // A later `blocks` bound may not read a name that the region declares.
                if let Some(r) = ctx.region.as_mut() {
                    r.locals.insert(name);
                }
            }
        }
    }

    /// Whether code here is in a blocks or subgroups body, outside every lane loop.
    fn stretch_level(ctx: &Ctx) -> bool {
        ctx.region.is_some()
            && matches!(
                ctx.axes.last().map(|a| a.kind),
                Some(LevelKind::Blocks | LevelKind::Subgroups)
            )
    }

    fn is_uniform(ctx: &Ctx, name: &str) -> bool {
        if ctx.lane0.contains(name) {
            return false;
        }
        ctx.uniform.contains(name)
            || ctx
                .region
                .as_ref()
                .is_some_and(|r| !r.locals.contains(name))
    }

    /// The names that only lane 0 has after each statement of `stmts`.
    fn lane0_names(stmts: &[Statement], before: &HashSet<String>) -> Vec<HashSet<String>> {
        let mut lane0 = before.clone();
        let mut after = Vec::with_capacity(stmts.len());
        let mut i = 0;
        while i < stmts.len() {
            if across_lanes::holds_across(&stmts[i]) {
                after.push(lane0.clone());
                i += 1;
                continue;
            }
            let start = i;
            while i < stmts.len() && !across_lanes::holds_across(&stmts[i]) {
                i += 1;
            }
            across_lanes::runs_on_lane0(&stmts[start..i], &mut lane0);
            after.extend(std::iter::repeat_n(lane0.clone(), i - start));
        }
        after
    }

    /// A matmul of whole tensors in a region with `across` loops. The code generator sends a
    /// whole matmul to cuBLAS, which a kernel cannot call, so the region would get no GPU kernel.
    /// Its loops ask for a launch, so that is an error here, with a position, and not a region
    /// that runs on the CPU in silence.
    fn whole_matmuls(&mut self, stmts: &[Statement]) {
        let mut found = Vec::new();
        crate::syntax::visit_exprs(stmts, &mut |e| {
            if let Expr::BinaryOp(b) = e {
                if matches!(b.op, crate::syntax::BinaryOp::MatMul) {
                    found.push(b.span);
                }
            }
        });
        for at in found {
            self.error(
                "a region with `across` loops may not have a matmul of whole tensors: the code \
                 generator sends it to cuBLAS, which a GPU kernel cannot call. Write its loops in \
                 the region, or do the matmul before the region"
                    .to_string(),
                at,
            );
        }
    }

    /// A tensor that the code of a block makes outside its lane loops is one for the block, which
    /// its lanes share. So it is in a memory of the block. A subgroup has no memory of its own.
    fn block_tensor(&mut self, l: &crate::syntax::stmt::LetDeclStmt, ctx: &Ctx) {
        let Expr::FunctionCall(fc) = crate::syntax::bare_unsafe_value(&l.expr) else {
            return;
        };
        if !crate::hir::flatten::is_tensor_alloc_call(fc) {
            return;
        }
        if ctx.axes.last().map(|a| a.kind) == Some(LevelKind::Subgroups) {
            self.error(
                "a subgroup has no memory of its own, so the code of a subgroup loop outside its \
                 lane loop may not make a tensor. Make it in the code of the block, in a memory \
                 such as `Memory::SMEM`"
                    .to_string(),
                l.span,
            );
            return;
        }
        let space = crate::hir::flatten::tensor_alloc_placement(fc).map(|p| p.space.name());
        if !space.as_ref().is_some_and(|s| self.shared.contains(s)) {
            self.error(
                "the lanes of a block share a tensor that the code of the block makes outside its \
                 lane loops, so it goes in a memory with `scope: cta` or `scope: sm`, such as \
                 `Memory::SMEM`. A tensor for each lane is made inside a lane loop"
                    .to_string(),
                l.span,
            );
        }
    }

    /// L8: a condition or a bound around a lane or subgroup loop is the same for every lane.
    fn uniform_around(&mut self, e: &Expr, what: &str, ctx: &Ctx) {
        if let Some((why, at)) = across_lanes::not_uniform(e, &|n| Self::is_uniform(ctx, n)) {
            let scope = if ctx.axes.last().map(|a| a.kind) == Some(LevelKind::Subgroups) {
                "subgroup"
            } else {
                "block"
            };
            self.error(
                format!(
                    "{what} decides whether the lanes reach the end of a lane loop or a lane \
                     reduction, so it must be the same for every lane of the {scope}. It may read \
                     only the indices of blocks and subgroups, values from the host, and names \
                     computed from them, but {why}"
                ),
                at,
            );
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
                    if Self::stretch_level(ctx) && across_lanes::holds_across(s) {
                        self.uniform_around(&f.iterable, "the bound of this loop", ctx);
                        body.strict_ifs = true;
                        if across_lanes::not_uniform(&f.iterable, &|n| Self::is_uniform(ctx, n))
                            .is_none()
                        {
                            body.uniform.insert(f.iter.clone());
                        }
                    }
                    if let Some(r) = body.region.as_mut() {
                        r.locals.insert(f.iter.clone());
                    }
                    self.stmts(&f.body, &body);
                }
            }
            Statement::Loop(l) => {
                let mut body = Self::inner(ctx);
                body.plain_loops += 1;
                if Self::stretch_level(ctx) && across_lanes::holds_across(s) {
                    body.strict_ifs = true;
                }
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
            Statement::LetDecl(l) => {
                if Self::stretch_level(ctx) {
                    self.block_tensor(l, ctx);
                }
                if let Some(fc) = lanes::bare_lanes_call(&l.expr) {
                    if let (true, false, Some(region), Some(axis)) = (
                        Self::stretch_level(ctx),
                        ctx.in_closure,
                        ctx.region.clone(),
                        ctx.axes.last(),
                    ) {
                        self.lane_calls.push(LaneCall {
                            at: fc.span,
                            name: fc.name.as_ref().to_string(),
                            scope: axis.kind,
                            region,
                        });
                    }
                }
                self.expr(&l.expr, ctx)
            }
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
                let desc = self.graph.descriptor(&kind);
                let region = Ctx {
                    region: Some(Region {
                        name: sp.top.display_name(),
                        levels: desc.and_then(|d| d.levels.clone()),
                        arch: desc.and_then(|d| d.arch.as_ref()).map(|a| a.to_string()),
                        launches: desc.is_some_and(|d| d.launches_levels(&kind)),
                        locals: HashSet::new(),
                        span: sp.span,
                    }),
                    top: true,
                    ..Ctx::default()
                };
                self.region_nests(&sp.stmts, &region);
                // A region that `-fauto-schedule-levels` runs as plain loops has no launch, so its
                // matmul goes to cuBLAS as in any other region.
                let launches = region.region.as_ref().is_some_and(|r| r.launches);
                if launches
                    && sp
                        .stmts
                        .iter()
                        .any(|s| matches!(s, Statement::ForLoop(f) if f.across.is_some()))
                {
                    self.whole_matmuls(&sp.stmts);
                }
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
                let branches = || i.then_block.iter().chain(i.else_block.iter().flatten());
                let around = branches().any(across_lanes::holds_across);
                // An `if` that cannot jump lets every lane get to the barrier.
                let mut jumps = false;
                let stmts: Vec<Statement> = branches().cloned().collect();
                across_lanes::visit_stmts(&stmts, &mut |s| {
                    jumps |= matches!(
                        s,
                        Statement::Break(_) | Statement::Continue(_) | Statement::Return(_)
                    );
                });
                if Self::stretch_level(ctx) && (around || (ctx.strict_ifs && jumps)) {
                    self.uniform_around(&i.cond, "this condition", ctx);
                }
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

    /// The checks of each lane reduction, after the loops have their levels. A reduction in the
    /// code of a block combines at most 32 groups of 32, and one in the code of a subgroup one
    /// group. To launch it, the lanes of a block are a multiple of 32, and they hold the tensor.
    fn lane_reductions(&mut self, body: &[Statement]) {
        let mut block_lanes: HashMap<Span, Option<u64>> = HashMap::new();
        crate::syntax::visit_exprs(body, &mut |e| {
            if let Expr::SpawnOn(sp) = e {
                block_lanes.insert(
                    sp.span,
                    crate::hir::flatten::across::across_block_lanes(&sp.stmts),
                );
            }
        });
        let legal: HashSet<Span> = self.lane_calls.iter().map(|c| c.at).collect();
        let mut elsewhere = Vec::new();
        crate::syntax::visit_exprs(body, &mut |e| {
            if let Expr::FunctionCall(fc) = e {
                if lanes::lane_routine(&fc.name).is_some() && !legal.contains(&fc.span) {
                    elsewhere.push((fc.name.as_ref().to_string(), fc.span));
                }
            }
        });
        for (name, at) in elsewhere {
            self.error(
                format!(
                    "`{name}` must be the whole value of a `let` in the code of a loop of blocks \
                     or subgroups, outside its lane loops. Every lane of the block or the \
                     subgroup must get to it"
                ),
                at,
            );
        }
        let calls = std::mem::take(&mut self.lane_calls);
        for c in &calls {
            let Some(&n) = self.lane_sizes.get(&c.at) else {
                continue;
            };
            let (most, scope) = match c.scope {
                LevelKind::Subgroups => (32, "subgroup"),
                _ => (lanes::MAX_BLOCK_ELEMENTS, "block"),
            };
            if n > most {
                self.error(
                    format!(
                        "`{}` in the code of a {scope} combines at most {most} elements, but its \
                         tensor has {n}",
                        c.name
                    ),
                    c.at,
                );
                continue;
            }
            if !c.region.launches {
                continue;
            }
            match c.scope {
                LevelKind::Subgroups => {
                    let sub = c
                        .region
                        .levels
                        .iter()
                        .flatten()
                        .find(|l| l.kind == LevelKind::Subgroups)
                        .and_then(|l| l.lanes);
                    if sub != Some(32) {
                        self.error(
                            format!(
                                "`{}` in the code of a subgroup needs subgroups of 32 lanes, but \
                                 the topology `{}` gives {}",
                                c.name,
                                c.region.name,
                                sub.map_or("no size".to_string(), |s| s.to_string())
                            ),
                            c.at,
                        );
                    }
                }
                _ => {
                    let lanes = block_lanes
                        .get(&c.region.span)
                        .copied()
                        .flatten()
                        .unwrap_or(1);
                    if lanes % 32 != 0 || n > lanes {
                        self.error(
                            format!(
                                "`{}` in the code of a block needs a block whose lanes are a \
                                 multiple of 32 and at least the {n} elements of its tensor, but \
                                 the lane loops of this region give {lanes} lanes",
                                c.name
                            ),
                            c.at,
                        );
                    }
                }
            }
        }
    }

    /// L10: a region with `across` has one `blocks` nest, directly in it.
    fn region_nests(&mut self, stmts: &[Statement], region: &Ctx) {
        // A topology with no levels has no nest of blocks.
        let Some(blocks) = region
            .region
            .as_ref()
            .and_then(|r| r.levels.as_ref())
            .and_then(|l| l.first())
            .map(|l| l.name.clone())
        else {
            return;
        };
        let nests: Vec<&ForLoopStmt> = stmts
            .iter()
            .filter_map(|s| match s {
                Statement::ForLoop(f) if f.across.as_ref().map(|a| &a.level) == Some(&blocks) => {
                    Some(f)
                }
                _ => None,
            })
            .collect();
        // L10: before the nest, every lane runs the code, so it may only declare values, and
        // it may not read what the blocks write. Nothing comes after the nest.
        if let Some(first) = nests.first() {
            let at = stmts
                .iter()
                .position(|s| matches!(s, Statement::ForLoop(f) if std::ptr::eq(f, *first)))
                .expect("the nest is in the region");
            let written = across_lanes::written_tensors(std::slice::from_ref(&stmts[at]));
            for s in &stmts[..at] {
                match s {
                    Statement::Drop(_) => {}
                    Statement::LetDecl(l) => {
                        if let Some(w) = across_lanes::may_write(&l.expr) {
                            self.error(
                                "before the `across` nest of blocks, every lane runs the code, \
                                 so a `let` there may only compute a value: no calls, blocks or \
                                 `&mut`"
                                    .to_string(),
                                w,
                            );
                        } else if let Some((name, w)) =
                            across_lanes::expr_reads_any(&l.expr, &written)
                        {
                            self.error(
                                format!(
                                    "before the `across` nest of blocks, this reads `{name}`, \
                                     which the blocks write, so the read races with them"
                                ),
                                w,
                            );
                        }
                    }
                    other => self.error(
                        "before the `across` nest of blocks, a region may only declare values \
                         with `let`: every lane runs that code"
                            .to_string(),
                        other.span(),
                    ),
                }
            }
            for s in &stmts[at + 1..] {
                let nest = matches!(s, Statement::ForLoop(f) if f.across.is_some());
                if !nest && !matches!(s, Statement::Drop(_)) {
                    self.error(
                        "a region with `across` loops does nothing after its nest of blocks"
                            .to_string(),
                        s.span(),
                    );
                }
            }
        }
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
                    "`across {}`: the topology `{}` has no levels. Give its fleet file `levels:`, \
                     such as `levels: [blocks(max_lanes: 1024), lanes]`",
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
        let auto = !region.launches;
        if auto && !self.auto_schedule {
            let arch = match &region.arch {
                Some(arch) => format!("whose arch is `{arch}`"),
                None => "which has no `arch:`".to_string(),
            };
            self.error(
                format!(
                    "`across {}`: no code generator can launch the levels of the topology `{}`, \
                     {arch}. Pass `-fauto-schedule-levels` to run the loop as a plain `for` that \
                     carries its level as metadata",
                    a.level, region.name
                ),
                a.span,
            );
            return;
        }
        self.found.insert(a.span, (level.kind, subgroup, auto));
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
                    } else if let Some(at) = across_lanes::may_write(b) {
                        self.error(
                            "the host computes the size of an `across` loop of blocks before the \
                             region starts, and the kernel computes it again. So the size may \
                             not call a function"
                                .to_string(),
                            at,
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
            let same_level = |s: &Statement| match s {
                Statement::ForLoop(g) if g.across.as_ref().map(|x| &x.level) == Some(&a.level) => {
                    Some(g.across.as_ref().map_or(g.span, |x| x.span))
                }
                _ => None,
            };
            let whole_body = f.body.len() == 1 && same_level(&f.body[0]).is_some();
            for s in &f.body {
                if whole_body {
                    break;
                }
                // A lane loop at any depth, but a block loop only directly in the body: a block
                // loop in a lane loop gets the error of N1. The first one is the outermost, and
                // the loops in it get their own check.
                let mut found = same_level(s);
                if level.kind == LevelKind::Lanes {
                    across_lanes::visit_stmts(std::slice::from_ref(s), &mut |s| {
                        found = found.or_else(|| same_level(s));
                    });
                }
                if let Some(at) = found {
                    self.error(
                        format!(
                            "a {} loop inside another one is the whole body of the outer loop",
                            kind_word(level.kind)
                        ),
                        at,
                    );
                }
            }
        }
        // L4: no pass may touch a place that another pass writes.
        if let Some((why, at)) = crate::hir::check::across_proof::across_loop_race(
            f,
            kind_word(level.kind),
            &self.contracts,
        ) {
            self.error(why, at);
        }
        // L5: a lane loop may not read what only lane 0 has.
        if level.kind != LevelKind::Blocks {
            let read = across_lanes::expr_reads_any(&f.iterable, &ctx.lane0)
                .or_else(|| across_lanes::reads_any(&f.body, &ctx.lane0));
            if let Some((name, at)) = read {
                self.error(
                    format!(
                        "this {} loop reads `{name}`, which only lane 0 has: lane 0 alone runs \
                         the code that sets it, because that code writes memory. Pass the value \
                         through memory, such as an element of a tensor in shared memory",
                        kind_word(level.kind)
                    ),
                    at,
                );
            }
        }
        let mut body = Self::inner(ctx);
        if level.kind != LevelKind::Lanes {
            body.uniform.insert(f.iter.clone());
        }
        body.strict_ifs = false;
        body.axes.push(axis);
        body.plain_loops = 0;
        if let Some(r) = body.region.as_mut() {
            r.locals.insert(f.iter.clone());
        }
        self.stmts(&f.body, &body);
    }
}

/// Gives each `across` loop in `stmts` the level that the walk found for it. A loop in a place
/// that this does not visit keeps `None`, and the code generator runs it as a plain `for`.
fn mark_stmts(stmts: &mut [Statement], found: &HashMap<Span, (LevelKind, Option<u64>, bool)>) {
    for s in stmts {
        match s {
            Statement::ForLoop(f) => {
                if let Some(a) = f.across.as_mut() {
                    if let Some(&(kind, lanes, auto)) = found.get(&a.span) {
                        a.kind = Some(kind);
                        a.subgroup_lanes = lanes;
                        a.auto_schedule = auto;
                    }
                }
                mark_stmts(&mut f.body, found);
            }
            Statement::Loop(l) => mark_stmts(&mut l.body, found),
            Statement::LetDecl(l) => mark_expr(&mut l.expr, found),
            Statement::ExprStmt(e) => mark_expr(&mut e.expr, found),
            Statement::Return(r) => {
                if let Some(e) = r.expr.as_mut() {
                    mark_expr(e, found);
                }
            }
            _ => {}
        }
    }
}

fn mark_expr(e: &mut Expr, found: &HashMap<Span, (LevelKind, Option<u64>, bool)>) {
    match e {
        Expr::SpawnOn(sp) => {
            mark_stmts(&mut sp.stmts, found);
            if let Some(r) = sp.ret.as_mut() {
                mark_expr(r, found);
            }
        }
        Expr::If(i) => {
            mark_stmts(&mut i.then_block, found);
            if let Some(b) = i.else_block.as_mut() {
                mark_stmts(b, found);
            }
        }
        Expr::Match(m) => {
            for arm in &mut m.arms {
                mark_stmts(&mut arm.body, found);
            }
        }
        Expr::UnsafeBlock(b) => {
            mark_stmts(&mut b.stmts, found);
            if let Some(r) = b.ret.as_mut() {
                mark_expr(r, found);
            }
        }
        _ => {}
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
