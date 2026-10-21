//===- parallel.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Whether the loop of a `spawn on` region can run its passes in parallel, and if not, why.
// The lowering asks it which loops to split; the checker asks it which loops to warn about,
// so the warning and what the program does cannot disagree.
//
//===----------------------------------------------------------------------===//

use crate::hir::flatten::{is_tensor_alloc_call, tensor_alloc_placement, SPAWN_COOP};
use crate::syntax::stmt::Statement;
use crate::syntax::{BinaryOp, Expr, RelationalOp, Span};
use std::collections::{HashMap, HashSet};

/// What stops the passes of a `spawn` loop from running at the same time, and where it is.
#[derive(Clone, Debug)]
pub(crate) struct Blocker {
    /// Finishes the sentence "this `spawn` loop runs on one thread: ...".
    pub why: String,
    pub span: Span,
}

/// What `outer_for` found out about a region.
enum Verdict {
    /// The loop at this statement index runs in parallel, with this many passes.
    Parallel(usize, u64),
    /// The region is not one loop meant to be split: it has no single loop over a range, or
    /// its passes write memory another pass may write too.
    NotSplittable,
    /// Every pass writes only memory of its own, and something else stops the split.
    Blocked(Blocker),
}

impl crate::hir::TypeChecker<'_> {
    /// W1033 for each `spawn` region in `body` whose loop runs on one thread although every pass
    /// writes only memory of its own. Run on the checked body, which is what the lowering splits,
    /// so the warning names what keeps the loop serial there.
    pub(crate) fn warn_serial_spawn_loops(&mut self, body: &[Statement]) {
        let mut found = Vec::new();
        crate::syntax::visit_exprs(body, &mut |e| {
            // A region with `across` loops maps its work to lanes itself.
            if let Expr::SpawnOn(sp) = e {
                if sp
                    .stmts
                    .iter()
                    .any(crate::hir::check::across_lanes::holds_across)
                {
                    return;
                }
                if let Some(b) = serial_loop_blocker(&sp.stmts) {
                    // An expression the compiler made may have no position of its own.
                    let at = if b.span == Span::default() {
                        sp.span
                    } else {
                        b.span
                    };
                    found.push((b.why, at, contains_barrier(&sp.stmts)));
                }
            }
        });
        for (why, at, kernel) in found {
            let msg = if kernel {
                format!(
                    "this `spawn` region calls `barrier()`, but it runs on one thread: {why}. \
                     The compiler splits a region with `barrier()` into blocks and threads only \
                     when it can show that no two threads write the same memory"
                )
            } else {
                format!(
                    "this `spawn` loop runs on one thread: {why}. Each pass writes only its own \
                     rows, so without that its passes could run at the same time"
                )
            };
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1033,
                msg,
                Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
            );
        }
    }
}

/// Why the loop of a `spawn` region runs on one thread, when every pass writes only memory of
/// its own and something else stops the split. `None` when the loop runs in parallel, and when
/// it is not a loop meant to be split. A region with a `barrier()` is always meant to be split
/// into blocks and threads, so for one that runs on one thread this names the first thing that
/// stopped the split.
pub(crate) fn serial_loop_blocker(stmts: &[Statement]) -> Option<Blocker> {
    if contains_barrier(stmts) {
        if parallel_outer_for(stmts).is_some() {
            return None;
        }
        return two_level(stmts).err();
    }
    match outer_for(stmts, true) {
        Verdict::Blocked(b) if parallel_two_level(stmts).is_none() => Some(b),
        _ => None,
    }
}

/// Whether everything after a region's loop, at `idx`, may run per-thread after each thread's
/// stripe: nothing, or drops of scratch declared before the loop, which each thread has a copy
/// of.
fn only_scratch_drops_after(stmts: &[Statement], idx: usize) -> bool {
    stmts[idx + 1..].iter().all(|s| match s {
        Statement::Drop(d) => stmts[..idx]
            .iter()
            .any(|p| matches!(p, Statement::LetDecl(l) if l.name == d.name)),
        _ => false,
    })
}

/// Is the outermost `for` of this spawn region safe to grid-stride — run its iterations on
/// concurrent device threads instead of one (#251)?
///
/// Splitting a loop across threads is only sound when no iteration can observe another. The proof
/// obligation is about **captured memory** — everything a thread declares itself is
/// thread-private on a GPU by construction (an entry-block alloca is in the thread's own local
/// depot, so even the region's pre-loop scratch declarations replicate harmlessly). So the rule
/// this walker enforces, conservatively and syntactically, is:
///
/// - the region is `[zero or more local declarations] for iv in lo..hi { body }` and nothing else,
///   with literal integer bounds (the trip count sizes the launch);
/// - every write in the body lands either in a name declared inside the region, or in a captured
///   tensor indexed **first by `iv`** — each iteration then owns row `iv` and no other;
/// - a captured tensor the body writes is read only at the iteration's own row, since any other
///   row may be another iteration's, written at the same time;
/// - nothing in the body can write through a side door: no free calls, no prints, no transfers, no
///   nested spawns, no method calls rooted at a captured name (a `&mut self` method on captured
///   state is a write this walker cannot see);
/// - no `break` of the outer loop (a serial `break` stops every later iteration; a strided one
///   only stops the current thread's — different programs), and no shadowing that could make one
///   name mean "captured" at one point and "local" at another (a name referenced before a later
///   `let` of the same name rejects the region).
///
/// Returns the index of the `for` statement and the trip count, or `None` — and `None` costs
/// nothing: the region lowers exactly as it always did and runs serially.
///
/// The declared/used-captured distinction is tracked in program order, so the walker is immune to
/// shadowing tricks without needing scopes: a declaration is only accepted while its name has
/// never been read as captured.
pub(crate) fn parallel_outer_for(stmts: &[Statement]) -> Option<(usize, u64)> {
    match outer_for(stmts, false) {
        Verdict::Parallel(idx, trip) => Some((idx, trip)),
        _ => None,
    }
}

/// `parallel_outer_for`, and why not. With `explain` the walk goes on past the first thing
/// that stops the split, to tell a loop that writes shared memory from one stopped by
/// something else; without it, it stops there, as the lowering needs.
fn outer_for(stmts: &[Statement], explain: bool) -> Verdict {
    use crate::syntax::stmt::Statement as S;

    let mut for_idx: Option<usize> = None;
    for (i, s) in stmts.iter().enumerate() {
        if matches!(s, S::ForLoop(_)) {
            if for_idx.is_some() {
                return Verdict::NotSplittable; // two top-level loops: neither owns the region
            }
            for_idx = Some(i);
        }
    }
    let Some(idx) = for_idx else {
        return Verdict::NotSplittable;
    };
    let S::ForLoop(f) = &stmts[idx] else {
        return Verdict::NotSplittable;
    };
    let Expr::Range(r) = &*f.iterable else {
        return Verdict::NotSplittable;
    };
    let mut scan = ParallelScan::fresh(f.iter.clone());
    scan.explain = explain;
    // A statement the compiler added, such as a drop, is not the program's to change.
    if !only_scratch_drops_after(stmts, idx) {
        match stmts[idx + 1..].iter().find(|s| !matches!(s, S::Drop(_))) {
            Some(after) if scan.refuse("the region does more after the loop", after.span()) => {}
            _ => return Verdict::NotSplittable,
        }
    }
    let bounds = parallel_int_literal(&r.start).zip(parallel_int_literal(&r.end));
    if let Some((lo, hi)) = bounds {
        if hi <= lo {
            return Verdict::NotSplittable;
        }
    } else if !scan.refuse(
        "its bounds are not numbers known when compiling",
        f.iterable.span(),
    ) {
        return Verdict::NotSplittable;
    }

    for s in &stmts[..idx] {
        let S::LetDecl(d) = s else {
            if matches!(s, S::Drop(_))
                || !scan.refuse(
                    "the region does more than declare names before the loop",
                    s.span(),
                )
            {
                return Verdict::NotSplittable;
            }
            continue;
        };
        // A pre-loop declaration replicates per thread, so its initializer may read anything but
        // must not be able to write: `Tensor<..>(..)` scratch is the intended shape, also inside
        // `unsafe { .. }`, and a plain expression passes through the same no-side-door walk as
        // body reads.
        let init_ok = match crate::syntax::bare_unsafe_value(&d.expr) {
            Expr::FunctionCall(fc)
                if is_tensor_alloc_call(fc) && fc.args.iter().all(|a| scan.expr(a)) =>
            {
                true
            }
            _ => scan.expr(&d.expr),
        };
        if !init_ok || !scan.declare(&d.name, d.span) {
            return Verdict::NotSplittable;
        }
    }
    if !scan.declare(&f.iter, f.span) || !scan.stmts(&f.body, 0) {
        return Verdict::NotSplittable;
    }
    if scan.shared_write {
        return Verdict::NotSplittable;
    }
    if let Some(b) = scan.blocker {
        return Verdict::Blocked(b);
    }
    match (scan.shared_row(), bounds) {
        (Some((name, span)), _) => Verdict::Blocked(Blocker {
            why: format!("a pass reads a row of `{name}` that another pass writes"),
            span,
        }),
        (None, Some((lo, hi))) => Verdict::Parallel(idx, (hi - lo) as u64),
        (None, None) => Verdict::NotSplittable,
    }
}

/// What `parallel_two_level` proved: which loop is block-mapped and which are thread-mapped,
/// addressed by AST node identity (stable for the duration of lowering).
pub(crate) struct TwoLevelPlan {
    pub block: usize,
    pub threads: Vec<usize>,
}

/// Is this spawn region a sound two-level (block/thread) kernel (Vx#379)?
///
/// The accepted shape is
///
/// ```text
/// [local declarations]
/// for bq in 0..NB {            // block-mapped: iterations -> blocks
///     <items>
/// }
/// ```
///
/// where `<items>` is a sequence of: local declarations (each thread makes its own copy --
/// block-scope statements execute redundantly per thread, which is only sound because nothing at
/// block scope may write captured memory), `barrier()` calls, THREAD loops, sequential loops
/// whose bodies are again `<items>` (a tile loop around cooperative phases), and -- stage C --
/// COOPERATIVE thread loops: a thread loop with the barriers INSIDE it, so one thread's whole
/// walk (private accumulator, own-row fill, barrier, consume, barrier) is a single body and the
/// accumulator never leaves registers. A cooperative loop's trip must equal the launch width
/// exactly (each thread runs it once) and its barriers must sit in uniform positions; see
/// `coop_item`.
///
/// A THREAD loop is a literal-bound `for t in 0..K` whose body passes the single-level write
/// walk with one extension: a captured write's first index may be the loop's own IV, or the
/// affine form `bq * K + t` (either operand order) where `K` is EXACTLY this thread loop's trip
/// -- each (block, thread-iteration) pair then owns one row and no other. The multiplier
/// equaling the trip is what makes the partition exact; `bq * 8 + t` with `t in 0..4` would
/// leave rows unowned and `t in 0..16` would double-book them.
///
/// The degeneracy that keeps every existing guarantee: at a 1x1x1 launch, block and thread
/// strides are both 1 and offsets both 0, so the kernel IS the serial nest -- the host path and
/// the launcher's EXPECT check hold by construction, not by re-verification.
pub(crate) fn parallel_two_level(stmts: &[Statement]) -> Option<(u64, TwoLevelPlan)> {
    two_level(stmts).ok()
}

/// `parallel_two_level`, and why not: the first thing that stopped the split, with where it is.
/// A reason with no position of its own uses the default span.
fn two_level(stmts: &[Statement]) -> Result<(u64, TwoLevelPlan), Blocker> {
    use crate::syntax::stmt::Statement as S;
    let no = |why: &str, span: Span| {
        Err(Blocker {
            why: why.to_string(),
            span,
        })
    };

    // Same region prologue rule as the single-level prover: declarations only, then the loop,
    // nothing after it.
    let mut for_idx: Option<usize> = None;
    for (i, s) in stmts.iter().enumerate() {
        if matches!(s, S::ForLoop(_)) {
            if for_idx.is_some() {
                return no("the region has more than one loop", s.span());
            }
            for_idx = Some(i);
        }
    }
    let Some(idx) = for_idx else {
        return no("the region has no loop", Span::default());
    };
    if !only_scratch_drops_after(stmts, idx) {
        let after = stmts[idx + 1..].iter().find(|s| !matches!(s, S::Drop(_)));
        return no(
            "the region does more after the loop",
            after.map(|s| s.span()).unwrap_or_default(),
        );
    }
    let S::ForLoop(f) = &stmts[idx] else {
        return no("the region has no loop", Span::default());
    };
    let block_bounds = "the loop over blocks does not go over `0..N`, with `N` a number known \
                        when compiling";
    let Expr::Range(r) = &*f.iterable else {
        return no(block_bounds, f.iterable.span());
    };
    let (Some(lo), Some(hi)) = (parallel_int_literal(&r.start), parallel_int_literal(&r.end))
    else {
        return no(block_bounds, f.iterable.span());
    };
    if lo != 0 || hi <= lo {
        return no(block_bounds, f.iterable.span()); // the affine ownership argument assumes lb 0
    }

    // A region with no thread loop or no barrier is not a cooperative kernel; let the flat
    // grid-stride path have it (it maps the same work with less machinery).
    let mut scan = ParallelScan::fresh(f.iter.clone());
    let why_not = |scan: ParallelScan, why: &str, span: Span| {
        Err(scan.refusal.unwrap_or(Blocker {
            why: why.to_string(),
            span,
        }))
    };
    for s in &stmts[..idx] {
        let S::LetDecl(d) = s else {
            return no(
                "the region does more than declare names before the loop",
                s.span(),
            );
        };
        if !scan.let_decl_two_level(d) {
            return why_not(
                scan,
                "a name declared before the loop cannot be split",
                d.span,
            );
        }
    }
    if !scan.declare(&f.iter, f.span) {
        return why_not(scan, "the loop over blocks reuses a name", f.span);
    }

    let mut walker = TwoLevelWalk {
        scan,
        block_iv: f.iter.to_string(),
        threads: Vec::new(),
        saw_thread_loop: false,
        max_thread_trip: 0,
        coop_trips: Vec::new(),
    };
    // Twice: the block loop grid-strides, so this body's END wraps to its BEGINNING in the same
    // block -- one thread's next-bq fill against another's lagging consume. The ledgers
    // surviving pass one are exactly that seam, and only a second pass walks a write into them.
    if !walker.items(&f.body) || !walker.items(&f.body) {
        return why_not(
            walker.scan,
            "a statement in the loop over blocks cannot be split",
            f.span,
        );
    }
    if !walker.saw_thread_loop {
        return no(
            "the loop over blocks holds no loop that the threads of a block can share",
            f.span,
        );
    }
    if let Some((name, span)) = walker.scan.shared_row() {
        return no(
            &format!("a pass reads a row of `{name}` that another pass writes"),
            span,
        );
    }
    let threads = walker.max_thread_trip.clamp(1, 1024) as u64;
    // A cooperative loop's barriers live INSIDE it, so every thread of the block must enter it:
    // its trip must BE the block shape, exactly. A wider sibling loop would leave threads
    // outside the bar, and the kernel hangs.
    if let Some(t) = walker.coop_trips.iter().find(|t| **t as u64 != threads) {
        return no(
            &format!(
                "a loop that holds a `barrier()` has {t} passes, but the block has {threads} \
                 threads, one for each pass of its widest loop"
            ),
            f.span,
        );
    }
    // The walks above ran twice (sequences twice per pass), so the plan vector carries
    // duplicates; membership is all `lower_for` asks of it.
    walker.threads.sort_unstable();
    walker.threads.dedup();
    let coop = if walker.coop_trips.is_empty() {
        0
    } else {
        SPAWN_COOP
    };
    Ok((
        (hi - lo) as u64 | (threads << 32) | coop,
        TwoLevelPlan {
            // The address of the ForLoopStmt itself, because that is what `lower_for` is
            // handed -- the enum wrapper's address would never match.
            block: f as *const crate::syntax::ForLoopStmt as usize,
            threads: walker.threads,
        },
    ))
}

/// The block-body grammar walk behind `parallel_two_level`. Wraps the single-level
/// `ParallelScan` for expression legality and adds the item grammar around it.
struct TwoLevelWalk {
    scan: ParallelScan,
    block_iv: String,
    threads: Vec<usize>,
    saw_thread_loop: bool,
    /// The widest thread-loop trip: the block shape the launch should use, so a TQ=64 kernel
    /// gets 64 threads instead of 64 working and 64 idling (Vx#379).
    max_thread_trip: i64,
    /// Trips of the COOPERATIVE thread loops accepted (Vx#379 stage C). Each must equal the
    /// final launch width -- checked at plan end, because a wider sibling loop scanned later
    /// could widen the block after this loop was accepted.
    coop_trips: Vec<i64>,
}

impl TwoLevelWalk {
    fn items(&mut self, stmts: &[Statement]) -> bool {
        use crate::syntax::stmt::Statement as S;
        let is_barrier = |s: &Statement| {
            matches!(s, S::ExprStmt(e) if matches!(
                &e.expr,
                Expr::FunctionCall(fc) if &*fc.name == "barrier" && fc.args.is_empty()
            ))
        };
        // A thread loop that wrote block-shared storage must be followed by a barrier --
        // including at the END of a sequence, because both a sequential loop's next
        // iteration and the block loop's next block-stride iteration reuse the same
        // storage. "The next statement is the barrier" is the whole v1 discipline.
        let mut need_barrier = false;
        let no_barrier = |scan: &mut ParallelScan, span: Span| {
            let mut names: Vec<&String> = scan.smem_written.iter().collect();
            names.sort();
            let name = names.first().map(|n| n.to_string()).unwrap_or_default();
            scan.fail(
                &format!(
                    "a loop writes the block-shared tensor `{name}`, and the next statement is \
                     not `barrier()`"
                ),
                span,
            )
        };
        let mut last_span = Span::default();
        for s in stmts {
            if need_barrier && !is_barrier(s) {
                return no_barrier(&mut self.scan, last_span);
            }
            last_span = s.span();
            need_barrier = false;
            let ok = match s {
                // Block-scope declaration: every thread evaluates it into its own copy --
                // unless its type names a space, which makes it one block-shared allocation
                // (Vx#379 stage B) that joins the shared-write discipline.
                S::LetDecl(d) => self.scan.let_decl_two_level(d),
                // Block-scope scalar assignment to a thread-private LOCAL is
                // redundant-per-thread and harmless; a captured or block-shared write at
                // block scope is every thread racing.
                S::Assign(a) => {
                    (matches!(&a.lhs, Expr::Identifier(id)
                        if self.scan.declared.contains(&*id.name)
                            && !self.scan.smem.contains(&*id.name))
                        || self.scan.fail(
                            "outside the loops that the threads share, it writes memory that \
                             every thread of the block would write",
                            a.span,
                        ))
                        && self.scan.expr(&a.rhs)
                }
                S::ExprStmt(_) => {
                    let b = is_barrier(s);
                    if b {
                        // The bar retires the dirt: everything written before it is visible
                        // to everyone after it, and every read before it has completed.
                        self.scan.smem_written.clear();
                        self.scan.smem_nonown_read.clear();
                    }
                    b || self.scan.fail(
                        "outside the loops that the threads share, it does something other \
                         than `barrier()`, which every thread of the block would do",
                        s.span(),
                    )
                }
                S::ForLoop(inner) => {
                    let ok = self.for_item(inner, s);
                    if ok && !self.scan.smem_written.is_empty() {
                        need_barrier = true;
                    }
                    ok
                }
                _ => self.scan.fail(
                    "outside the loops that the threads share, it holds a statement that every \
                     thread of the block would run",
                    s.span(),
                ),
            };
            if !ok {
                return false;
            }
        }
        !need_barrier || no_barrier(&mut self.scan, last_span)
    }

    fn for_item(&mut self, inner: &crate::syntax::ForLoopStmt, node: &Statement) -> bool {
        let bounds = "a loop inside the loop over blocks does not go over `0..N`, with `N` a \
                      number known when compiling";
        let Expr::Range(r) = &*inner.iterable else {
            return self.scan.fail(bounds, inner.iterable.span());
        };
        let (Some(lo), Some(hi)) = (parallel_int_literal(&r.start), parallel_int_literal(&r.end))
        else {
            return self.scan.fail(bounds, inner.iterable.span());
        };
        if lo != 0 || hi <= lo {
            return self.scan.fail(bounds, inner.iterable.span());
        }
        if *inner.iter == *self.block_iv {
            return self.scan.fail(
                "an inner loop reuses the name of the loop over blocks",
                inner.span,
            );
        }
        if !self.scan.declare(&inner.iter, inner.span) {
            return false;
        }
        // Decide the loop's role by whether its subtree synchronizes: a barrier anywhere
        // below makes this a sequence container or a COOPERATIVE thread loop, tried in that
        // order of specificity reversed -- coop first, because it is the shape worth having
        // (the accumulator stays in registers). No barrier anywhere makes it a plain
        // THREAD-loop candidate. Merely containing a `for` must NOT make a loop sequential
        // -- a thread loop legitimately nests serial walks (the key loop inside a consume
        // phase), and classifying it sequential sent its inner serial loop to the thread
        // rule, which rejected the whole region.
        let is_sequence = contains_barrier(&inner.body);
        if is_sequence {
            let refusal = self.scan.refusal.clone();
            if self.coop_item(inner, node, hi - lo) {
                return true;
            }
            // Twice for the same reason the block body walks twice: a sequence's end wraps
            // to its own beginning on the next iteration, and a dangling nonown read meeting
            // the next pass's fill is visible only to a second walk. When this fails too, the
            // warning keeps the reason the cooperative loop failed, which names the cause.
            if self.items(&inner.body) && self.items(&inner.body) {
                self.scan.refusal = refusal;
                return true;
            }
            return false;
        }
        let saved_iv = std::mem::replace(&mut self.scan.iv, inner.iter.to_string());
        let saved_affine = self.scan.affine.take();
        self.scan.affine = Some((self.block_iv.clone(), hi - lo));
        // The ledgers deliberately survive from the previous item -- no clear here: they are
        // "dirt since the last barrier", and a thread loop's writes meeting a neighbour's
        // un-barriered reads from the PREVIOUS loop are the same race as within one loop. The
        // access-time seam checks in `target` and `expr` enforce both directions.
        let ok = self.scan.stmts(&inner.body, 0);
        self.scan.iv = saved_iv;
        self.scan.affine = saved_affine;
        if ok {
            self.saw_thread_loop = true;
            self.max_thread_trip = self.max_thread_trip.max(hi - lo);
            let addr = match node {
                Statement::ForLoop(fl) => fl as *const crate::syntax::ForLoopStmt as usize,
                _ => return false,
            };
            self.threads.push(addr);
        }
        ok
    }

    /// A COOPERATIVE thread loop (Vx#379 stage C): `for qi in 0..K` with the barriers INSIDE
    /// it. The loop itself is thread-mapped, so one thread's whole walk -- private
    /// accumulator, own-row tile fill, barrier, consume, barrier -- is a single loop body,
    /// and the accumulator lives in registers instead of a shared tensor with per-tile
    /// spills. Soundness needs one fact the plain thread rule does not: every thread must
    /// meet every barrier the same number of times. `parallel_two_level` pins the launch to
    /// `threads == trip` (each thread runs the body exactly once), and the scan keeps
    /// barriers out of divergent positions -- conditionals, variable-bound loops, loops a
    /// `break` could leave early.
    fn coop_item(
        &mut self,
        inner: &crate::syntax::ForLoopStmt,
        node: &Statement,
        trip: i64,
    ) -> bool {
        if trip > 1024 {
            // The launcher clamps blocks at 1024 threads; a wider coop loop would stride,
            // and a strided loop's barrier counts diverge.
            return false;
        }
        // The sequence interpretation this falls back to must start from the state it would
        // have seen had this attempt never run; the scan mutates in place, so snapshot.
        let saved_declared = self.scan.declared.clone();
        let saved_captured = self.scan.used_captured.clone();
        let saved_written = self.scan.smem_written.clone();
        let saved_read = self.scan.smem_nonown_read.clone();
        let saved_iv = std::mem::replace(&mut self.scan.iv, inner.iter.to_string());
        let saved_affine = self.scan.affine.take();
        self.scan.affine = Some((self.block_iv.clone(), trip));
        self.scan.coop = true;
        self.scan.barrier_ok = true;
        // Twice: the block loop grid-strides, so this body's end wraps to its own beginning
        // (next bq, same block) -- pass two walks that seam.
        let ok = self.scan.stmts(&inner.body, 0) && self.scan.stmts(&inner.body, 0);
        self.scan.coop = false;
        self.scan.barrier_ok = false;
        self.scan.iv = saved_iv;
        self.scan.affine = saved_affine;
        if !ok {
            self.scan.declared = saved_declared;
            self.scan.used_captured = saved_captured;
            self.scan.smem_written = saved_written;
            self.scan.smem_nonown_read = saved_read;
            return false;
        }
        self.saw_thread_loop = true;
        self.max_thread_trip = self.max_thread_trip.max(trip);
        self.coop_trips.push(trip);
        let addr = match node {
            Statement::ForLoop(fl) => fl as *const crate::syntax::ForLoopStmt as usize,
            _ => return false,
        };
        self.threads.push(addr);
        true
    }
}

/// Does a barrier statement appear anywhere below? This decides a loop's role in the
/// two-level walk (synchronizing loops are sequences or cooperative loops; barrier-free ones
/// are thread candidates) and which loops the coop scan must hold to uniform trips.
fn contains_barrier(stmts: &[Statement]) -> bool {
    use crate::syntax::stmt::Statement as S;
    stmts.iter().any(|s| match s {
        S::ExprStmt(e) => {
            matches!(&e.expr, Expr::FunctionCall(fc) if &*fc.name == "barrier")
        }
        S::ForLoop(f) => contains_barrier(&f.body),
        S::Loop(l) => contains_barrier(&l.body),
        _ => false,
    })
}

/// The indices of an access after its first index, which picks the row: `t[i][2]` gives
/// `[Some(2)]`. An index that is not a number is `None`.
fn columns(indices: &[&Expr]) -> Vec<Option<i64>> {
    // `indices` is outermost-last, so the row is the last one.
    let n = indices.len().saturating_sub(1);
    indices[..n]
        .iter()
        .map(|e| parallel_int_literal(e))
        .collect()
}

/// Whether two accesses to the same tensor, given by `columns`, always reach different
/// elements: they have the same number of indices, and in one position both are different
/// numbers.
fn columns_apart(a: &[Option<i64>], b: &[Option<i64>]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .any(|(x, y)| matches!((x, y), (Some(x), Some(y)) if x != y))
}

/// A non-negative integer literal, for `parallel_outer_for`'s bounds.
fn parallel_int_literal(e: &Expr) -> Option<i64> {
    let Expr::Number(n) = e else {
        return None;
    };
    n.value.parse::<i64>().ok()
}

/// Scalar math the walk may trust. By the time `parallel_outer_for` runs, the checker has
/// rewritten every method call into a mangled free call — `(x).exp()` arrives as
/// `f32$exp(x)` — so a syntactic walk meets no `MethodCall`, only calls whose names carry the
/// receiver type. These are the stdlib's value-receiver scalar intrinsics: value in, value out,
/// nothing written anywhere, which is exactly what the disjointness proof needs from a call.
/// Anything not on this list rejects the region — a name here is a purity claim.
pub(crate) fn parallel_pure_call(name: &str) -> bool {
    let Some((ty, method)) = name.split_once('$') else {
        return false;
    };
    matches!(ty, "f32" | "f64" | "i32" | "i64")
        && matches!(
            method,
            "exp"
                | "fast_exp"
                | "exp2"
                | "sqrt"
                | "abs"
                | "ln"
                | "log2"
                | "log10"
                | "sin"
                | "cos"
                | "tan"
                | "tanh"
                | "fast_tanh"
                | "floor"
                | "ceil"
                | "round"
                | "min"
                | "max"
                | "pow"
                | "powi"
                | "recip"
        )
}

/// An index expression written out without spans, with the operands of `+` and `*` in a fixed
/// order, so `lo + i` and `i + lo` are one key, and the loop's own variable written `$iv`, so two
/// loops of one kernel over the same rows agree. `None` for anything else, which `owns_row` never
/// accepts as a row.
fn index_key(e: &Expr, iv: &str) -> Option<String> {
    match e {
        Expr::Identifier(id) if *id.name == *iv => Some("$iv".to_string()),
        Expr::Identifier(id) => Some(id.name.to_string()),
        Expr::Number(n) => Some(n.value.to_string()),
        Expr::BinaryOp(b) => {
            let (l, r) = (index_key(&b.lhs, iv)?, index_key(&b.rhs, iv)?);
            let op = match b.op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
                _ => return None,
            };
            let ordered = matches!(op, "-" | "/");
            let (l, r) = if !ordered && r < l { (r, l) } else { (l, r) };
            Some(format!("({l}{op}{r})"))
        }
        _ => None,
    }
}

/// The disjointness walk behind `parallel_outer_for`: what each iteration writes, and what it reads
/// of what is written. `declared` and `used_captured` grow in program order; every `false` means
/// "reject the region", never an error.
struct ParallelScan {
    iv: String,
    declared: HashSet<String>,
    used_captured: HashSet<String>,
    /// Set while scanning a THREAD loop of a two-level region (Vx#379): `(block_iv, K)` where `K`
    /// is the thread loop's trip. Widens the captured-write rule to accept the affine first index
    /// `block_iv * K + iv` -- the (block, thread-iteration) pair's own row.
    affine: Option<(String, i64)>,
    /// Region-locals that are BLOCK-SHARED rather than thread-private (Vx#379 stage B): declared
    /// `Tensor<T, [..], Memory::X>::uninit()`. Any space is treated as shared -- stricter than
    /// necessary for a non-sm space, never looser. Shared changes the write rule: only a thread
    /// loop may write one, at its own IV's row.
    smem: HashSet<String>,
    /// Per-thread-loop race ledgers, reset by `TwoLevelWalk::for_item`: shared tensors this loop
    /// wrote, and shared tensors it read at a row other than its own. A tensor in both sets is a
    /// read-write race between two iterations of the same loop; a written tensor also demands a
    /// barrier as the next statement.
    smem_written: HashSet<String>,
    smem_nonown_read: HashSet<String>,
    /// Tensors from outside the region that it writes, and those it reads at a row the reading
    /// iteration may not own. A tensor in both is a race: one iteration reads a row another may
    /// be writing. The exception is in `columns_apart`.
    captured_written: HashSet<String>,
    /// For each such read, and for each write, the indices after the first (see `columns`).
    captured_nonown_read: HashMap<String, Vec<Vec<Option<i64>>>>,
    captured_write_columns: HashMap<String, Vec<Vec<Option<i64>>>>,
    /// For each tensor from outside the region, the first index of every access that reads or
    /// writes a row its pass owns, written out without spans. A written tensor reached through
    /// two different first indices, such as `o[i]` and `o[lo + i]`, is a race: one pass's
    /// `lo + i` is another pass's `i`.
    row_keys: HashMap<String, HashSet<String>>,
    /// Set while scanning a COOPERATIVE thread loop (Vx#379 stage C): admits `barrier()` as a
    /// statement (under the divergence guards below) and arms the seam checks.
    coop: bool,
    /// Whether a barrier statement is currently in a uniform position: false inside an
    /// if-expression's blocks, where threads may disagree about arriving at all.
    barrier_ok: bool,
    /// One frame per loop the scan is inside: does that loop contain a barrier? A `break` or
    /// `continue` whose innermost loop synchronizes would let one thread leave while the
    /// others wait at the bar forever.
    loop_barrier_stack: Vec<bool>,
    /// How many loops inside the region's loop the statement being walked is: 0 in its body.
    /// An `if` in that statement is at the same depth.
    depth: u32,
    /// Set by the checker, which asks why a loop stays serial: the walk goes on past a
    /// refusal, so it can see whether any write reaches memory another pass writes.
    explain: bool,
    /// The `c` of each `if iv % c == 0` the walk is inside, with `c` a number. Only every `c`th
    /// pass runs such a block, so row `iv / c` is that pass's own.
    multiple_of: Vec<i64>,
    /// The first thing other than a write that stopped the split.
    blocker: Option<Blocker>,
    /// The first thing of any kind that stopped the split. The warning for a block-and-thread
    /// kernel names it.
    refusal: Option<Blocker>,
    /// Whether a pass writes memory that another pass may write too.
    shared_write: bool,
    /// For each tensor from outside the region, where a pass first reaches a row another pass
    /// may own: a read of another row, or a second first index (see `row_keys`).
    other_row_at: HashMap<String, Span>,
}

impl ParallelScan {
    fn fresh(iv: String) -> Self {
        ParallelScan {
            iv,
            declared: HashSet::new(),
            used_captured: HashSet::new(),
            affine: None,
            smem: HashSet::new(),
            smem_written: HashSet::new(),
            smem_nonown_read: HashSet::new(),
            captured_written: HashSet::new(),
            captured_nonown_read: HashMap::new(),
            captured_write_columns: HashMap::new(),
            row_keys: HashMap::new(),
            coop: false,
            barrier_ok: false,
            loop_barrier_stack: Vec::new(),
            depth: 0,
            explain: false,
            multiple_of: Vec::new(),
            blocker: None,
            refusal: None,
            shared_write: false,
            other_row_at: HashMap::new(),
        }
    }

    /// Stop the split because of `why`, at `span`. Answers whether the walk goes on: only when
    /// explaining, so the lowering still stops at the first refusal.
    fn refuse(&mut self, why: &str, span: Span) -> bool {
        if self.blocker.is_none() {
            self.blocker = Some(Blocker {
                why: why.to_string(),
                span,
            });
        }
        self.fail(why, span);
        self.explain
    }

    /// Stop the split because a write may reach memory another pass writes. A loop without
    /// `barrier()` that does this is not one meant to run in parallel, so W1033 says nothing.
    fn refuse_write(&mut self, why: &str, span: Span) -> bool {
        self.shared_write = true;
        self.fail(why, span);
        self.explain
    }

    /// Note `why` as the reason the split stopped, unless an earlier reason is noted. Always
    /// answers false, so `return self.fail(..)` stops the walk.
    fn fail(&mut self, why: &str, span: Span) -> bool {
        if self.refusal.is_none() {
            self.refusal = Some(Blocker {
                why: why.to_string(),
                span,
            });
        }
        false
    }

    /// A `let` in a two-level region (prologue or block scope). Two shapes pass: a plain
    /// expression that cannot write (the ordinary rule), and a fresh `Tensor` -- thread-private
    /// scratch when unplaced, and BLOCK-SHARED storage when its type names a space, which joins
    /// the shared-write discipline via `smem` (Vx#379 stage B).
    fn let_decl_two_level(&mut self, d: &crate::syntax::stmt::LetDeclStmt) -> bool {
        let mut is_smem = false;
        let init_ok = match crate::syntax::bare_unsafe_value(&d.expr) {
            Expr::FunctionCall(fc)
                if is_tensor_alloc_call(fc) && fc.args.iter().all(|a| self.expr(a)) =>
            {
                is_smem = tensor_alloc_placement(fc).is_some();
                true
            }
            _ => self.expr(&d.expr),
        };
        if !init_ok || !self.declare(&d.name, d.span) {
            return false;
        }
        if is_smem {
            self.smem.insert(d.name.to_string());
        }
        true
    }

    /// Record a read of `name`. Reads of captured state are always fine — but they pin the name as
    /// captured, so a later `let` of the same name rejects the region (see `declare`).
    fn note(&mut self, name: &str) {
        if !self.declared.contains(name) {
            self.used_captured.insert(name.to_string());
        }
    }

    /// A `let` (or loop IV) introduces a local — unless the name was already read as captured, or
    /// shadows the outer IV, either of which would let one spelling mean two memories.
    fn declare(&mut self, name: &str, span: Span) -> bool {
        if self.used_captured.contains(name) {
            return self.refuse(
                &format!("`{name}` is used from outside the region before a `let` of that name"),
                span,
            );
        }
        if !self.declared.is_empty() && name == self.iv && self.declared.contains(name) {
            return self.refuse(
                &format!("a `let` reuses the loop's own name, `{name}`"),
                span,
            );
        }
        self.declared.insert(name.to_string());
        true
    }

    /// A block's statements. A name its `let`s declare is region-local only until the block
    /// ends; after it, the name is whatever it was before, often a captured variable.
    fn stmts(&mut self, stmts: &[Statement], depth: u32) -> bool {
        let outer = self.declared.clone();
        let ok = stmts.iter().all(|s| self.stmt(s, depth));
        self.declared = outer;
        ok
    }

    fn stmt(&mut self, s: &Statement, depth: u32) -> bool {
        use crate::syntax::stmt::Statement as S;
        self.depth = depth;
        match s {
            S::LetDecl(d) => self.expr(&d.expr) && self.declare(&d.name, d.span),
            S::Assign(a) => self.target(&a.lhs) && self.expr(&a.rhs),
            // `x op= e` both reads and writes x; `target` covers the write, `expr` the read.
            S::CompoundAssign(a) => self.target(&a.lhs) && self.expr(&a.lhs) && self.expr(&a.rhs),
            S::ForLoop(inner) => {
                let Expr::Range(r) = &*inner.iterable else {
                    return self.refuse(
                        "an inner `for` loops over something other than a range",
                        inner.iterable.span(),
                    );
                };
                // Shadowing the strided IV would defeat the first-index rule.
                if *inner.iter == *self.iv
                    && !self.refuse("an inner loop reuses the loop's own name", inner.span)
                {
                    return false;
                }
                // The loop variable, like a `let` in the body, ends with the loop.
                let outer = self.declared.clone();
                if !(self.expr(&r.start)
                    && self.expr(&r.end)
                    && self.declare(&inner.iter, inner.span))
                {
                    return false;
                }
                let synchronizes = self.coop && contains_barrier(&inner.body);
                // A barrier's arrival count must not depend on the thread: a loop holding one
                // needs literal bounds, or its trip -- and with it how many times each thread
                // meets the bar -- could vary per thread and hang the block.
                if synchronizes
                    && (parallel_int_literal(&r.start).is_none()
                        || parallel_int_literal(&r.end).is_none())
                {
                    return self.fail(
                        "a loop that holds a `barrier()` has bounds that are not numbers known \
                         when compiling",
                        inner.iterable.span(),
                    );
                }
                self.loop_barrier_stack.push(synchronizes);
                // A synchronizing loop wraps: its end meets its own beginning on the next
                // iteration, and the ledgers a mid-body barrier cleared hide that seam from a
                // single pass -- so walk the body twice.
                let ok = self.stmts(&inner.body, depth + 1)
                    && (!synchronizes || self.stmts(&inner.body, depth + 1));
                self.loop_barrier_stack.pop();
                self.declared = outer;
                ok
            }
            S::Loop(l) => {
                // An infinite loop's trip is break-decided, i.e. potentially per-thread:
                // there is no uniform position for a barrier in one.
                if self.coop && contains_barrier(&l.body) {
                    return self.fail("a `loop` holds a `barrier()`", l.span);
                }
                self.loop_barrier_stack.push(false);
                let ok = self.stmts(&l.body, depth + 1);
                self.loop_barrier_stack.pop();
                ok
            }
            // Leaving or short-circuiting a synchronizing loop early desynchronizes its
            // barriers: one thread stops arriving while the rest wait.
            S::Break(_) | S::Continue(_) if *self.loop_barrier_stack.last().unwrap_or(&false) => {
                self.fail(
                    "a `break` or `continue` leaves a loop that holds a `barrier()`, so the \
                     other threads would wait at it forever",
                    s.span(),
                )
            }
            // A `break` of the outer loop stops every later iteration when serial but only the
            // current thread's stripe when strided — two different programs.
            S::Break(b) => {
                depth > 0
                    || self.refuse(
                        "a `break` would stop only one thread's share of the passes",
                        b.span,
                    )
            }
            S::Continue(_) => true,
            S::ExprStmt(e) => {
                if self.coop {
                    if let Expr::FunctionCall(fc) = &e.expr {
                        if &*fc.name == "barrier" && fc.args.is_empty() {
                            // A statement-position barrier: legal only where every thread
                            // arrives, and it retires the dirt on both ledgers.
                            if !self.barrier_ok {
                                return self.fail(
                                    "a `barrier()` is inside an `if`, so some threads may not \
                                     reach it",
                                    fc.span,
                                );
                            }
                            self.smem_written.clear();
                            self.smem_nonown_read.clear();
                            return true;
                        }
                    }
                }
                self.expr(&e.expr)
            }
            // Return, assert, macro calls, parse errors: none of these have stridable semantics.
            // Frees nothing yet; a drop in a parallel region is decided with phase 5.
            S::Drop(_) => true,
            S::Return(r) => self.refuse("it can `return`", r.span),
            S::Assert(a) => self.refuse("it holds an `assert`", a.span),
            S::MacroCall(_) | S::Error(_) => false,
        }
    }

    /// The left of an assignment: where does the write land?
    fn target(&mut self, lhs: &Expr) -> bool {
        match lhs {
            // A bare name: fine only if it is region-local AND thread-private. A captured scalar
            // written by every thread is the definition of a race; so is a whole-tensor write to
            // block-shared storage.
            Expr::Identifier(id) => {
                if !self.declared.contains(&*id.name) {
                    return self.refuse_write(
                        &format!(
                            "every pass writes `{}`, a name from outside the region",
                            id.name
                        ),
                        id.span,
                    );
                }
                !self.smem.contains(&*id.name)
                    || self.fail(
                        &format!(
                            "it gives the whole block-shared tensor `{}` a new value",
                            id.name
                        ),
                        id.span,
                    )
            }
            // An index chain `a[e0][e1]..`: peel to the base. A local base is thread-private
            // regardless of indices; a captured base must be indexed first by the IV, giving each
            // iteration its own row.
            Expr::IndexAccess(_) => {
                let mut indices = Vec::new();
                let mut cur = lhs;
                while let Expr::IndexAccess(ix) = cur {
                    indices.push(&*ix.index);
                    cur = &ix.base;
                }
                let Expr::Identifier(id) = cur else {
                    return self.refuse_write(
                        "it writes an element of something that is not a named tensor",
                        lhs.span(),
                    );
                };
                // `indices` is outermost-last: `a[i][d]` pushed d then i.
                let first = match indices.last() {
                    Some(e) => *e,
                    None => return false,
                };
                if !indices.iter().all(|e| self.expr(e)) {
                    return false;
                }
                if self.declared.contains(&*id.name) {
                    if !self.smem.contains(&*id.name) {
                        return true; // thread-private: any indexing is this thread's own
                    }
                    // Block-shared (Vx#379 stage B): writable only from a thread loop
                    // (`affine` is Some exactly there), one row per iteration, the row being
                    // the loop's own IV. The ledger feeds the barrier-after-write rule at the
                    // items level and the seam checks here.
                    if self.affine.is_none() {
                        return self.fail(
                            &format!(
                                "it writes the block-shared tensor `{}` outside a loop that \
                                 the threads share",
                                id.name
                            ),
                            lhs.span(),
                        );
                    }
                    // The write-after-read seam: rows a neighbour read since the last barrier
                    // (the next tile's fill racing this tile's consume). Only a barrier
                    // between the two orders them.
                    if self.smem_nonown_read.contains(&*id.name) {
                        return self.fail(
                            &format!(
                                "it writes `{}` after a pass read another pass's row of it, \
                                 with no `barrier()` between",
                                id.name
                            ),
                            lhs.span(),
                        );
                    }
                    self.smem_written.insert(id.name.to_string());
                    return matches!(first, Expr::Identifier(fid) if *fid.name == *self.iv)
                        || self.fail(
                            &format!(
                                "a pass writes a row of the block-shared tensor `{}` that is \
                                 not the row its loop variable `{}` picks",
                                id.name, self.iv
                            ),
                            lhs.span(),
                        );
                }
                self.note(&id.name);
                self.captured_written.insert(id.name.to_string());
                if !self.owns_row(&indices) {
                    return self.refuse_write(
                        &format!(
                            "the compiler cannot show that each pass of the loop over `{}` \
                             writes a different element of `{}`",
                            self.iv, id.name
                        ),
                        lhs.span(),
                    );
                }
                self.record_row(&id.name, first, lhs.span());
                self.captured_write_columns
                    .entry(id.name.to_string())
                    .or_default()
                    .push(columns(&indices));
                true
            }
            // A write through a pointer, a field or anything else may reach any memory.
            _ => self.refuse_write("it writes through a pointer or a field", lhs.span()),
        }
    }

    /// Note which row of `name` an access the current pass owns goes to; see `row_keys`.
    fn record_row(&mut self, name: &str, first: &Expr, at: Span) {
        if let Some(key) = index_key(first, &self.iv) {
            let keys = self.row_keys.entry(name.to_string()).or_default();
            keys.insert(key);
            if keys.len() > 1 {
                self.other_row_at.entry(name.to_string()).or_insert(at);
            }
        }
    }

    /// `iv + e`, `e + iv`, `iv - e` or `e - iv`, where `e` does not change during the loop: it
    /// is made of numbers and names from outside the region, which the region cannot write
    /// (`target` refuses a write to one). Each pass then reaches a different row.
    fn offset_first(&self, e: &Expr) -> bool {
        let Expr::BinaryOp(b) = e else {
            return false;
        };
        if !matches!(b.op, BinaryOp::Add | BinaryOp::Sub) {
            return false;
        }
        let is_iv = |x: &Expr| matches!(x, Expr::Identifier(id) if *id.name == *self.iv);
        (is_iv(&b.lhs) && self.loop_invariant(&b.rhs))
            || (is_iv(&b.rhs) && self.loop_invariant(&b.lhs))
    }

    /// `iv / c` inside an `if iv % c == 0`: the passes that run the block are `c` apart, so each
    /// reaches a different row.
    fn group_first(&self, e: &Expr) -> bool {
        let Expr::BinaryOp(b) = e else {
            return false;
        };
        b.op == BinaryOp::Div
            && matches!(&*b.lhs, Expr::Identifier(id) if *id.name == *self.iv)
            && parallel_int_literal(&b.rhs).is_some_and(|c| self.multiple_of.contains(&c))
    }

    /// The `c` of `cond` when it is `iv % c == 0`, with `c` a positive number.
    fn multiple_required(&self, cond: &Expr) -> Option<i64> {
        let Expr::RelationalOp(r) = cond else {
            return None;
        };
        let Expr::BinaryOp(b) = &*r.lhs else {
            return None;
        };
        let iv_rem = b.op == BinaryOp::Rem
            && matches!(&*b.lhs, Expr::Identifier(id) if *id.name == *self.iv);
        (r.op == RelationalOp::Eq && iv_rem && parallel_int_literal(&r.rhs) == Some(0))
            .then(|| parallel_int_literal(&b.rhs))
            .flatten()
            .filter(|c| *c > 0)
    }

    /// Whether `e` has the same value on every pass: numbers, and names from outside the region.
    fn loop_invariant(&self, e: &Expr) -> bool {
        match e {
            Expr::Number(_) => true,
            Expr::Identifier(id) => *id.name != *self.iv && !self.declared.contains(&*id.name),
            Expr::BinaryOp(b) => {
                matches!(b.op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
                    && self.loop_invariant(&b.lhs)
                    && self.loop_invariant(&b.rhs)
            }
            _ => false,
        }
    }

    /// Whether the index chain `indices` (outermost-last, as `target` collects it) names a row
    /// the current iteration owns: first index the IV, or one of the two-level ownership forms.
    fn owns_row(&self, indices: &[&Expr]) -> bool {
        let Some(first) = indices.last() else {
            return false;
        };
        matches!(first, Expr::Identifier(fid) if *fid.name == *self.iv)
            || self.offset_first(first)
            || self.group_first(first)
            || self.affine_first(first)
            || self.block_row_write(indices)
    }

    /// A tensor from outside the region that the loop writes and that a pass reaches at a row
    /// another pass may own, with where it does: a pass reads a row it may not own, or reaches
    /// the tensor through two different first indices.
    fn shared_row(&self) -> Option<(String, Span)> {
        let mut names: Vec<&String> = self.captured_written.iter().collect();
        names.sort();
        names
            .into_iter()
            .find(|name| {
                self.reads_a_written_element(name)
                    || self.row_keys.get(*name).is_some_and(|keys| keys.len() > 1)
            })
            .map(|name| {
                let at = self.other_row_at.get(name).copied().unwrap_or_default();
                (name.clone(), at)
            })
    }

    /// Whether a read of a row that a pass may not own can reach an element that a pass of the
    /// loop writes. A read of `t[i][0]` cannot reach a write to `t[k][2]`.
    fn reads_a_written_element(&self, name: &str) -> bool {
        let Some(reads) = self.captured_nonown_read.get(name) else {
            return false;
        };
        let writes = self.captured_write_columns.get(name);
        reads
            .iter()
            .any(|r| writes.is_none_or(|ws| ws.iter().any(|w| !columns_apart(w, r))))
    }

    /// The third captured-write ownership form (Vx#379 R3): the BLOCK owns the row and the
    /// THREADS partition its columns by residue -- `s[bq][t + x * K]` with `bq` the block IV,
    /// `t` this thread loop's IV, and `K` exactly its trip. Distinct threads write distinct
    /// residues mod K; distinct blocks write distinct rows; and `x` may be ANY declared local
    /// (a serial loop IV, typically), because whatever it evaluates to, the residue -- the
    /// ownership class -- is still `t`. This is the shape a block-per-row reduction sweep
    /// writes: a coalesced softmax's exp pass, thread `t` touching columns t, t+K, t+2K...
    fn block_row_write(&self, indices: &[&Expr]) -> bool {
        let Some((bq, k)) = &self.affine else {
            return false;
        };
        // Exactly [second, first]: rank-2, row then column, nothing deeper.
        if indices.len() != 2 {
            return false;
        }
        let first = indices[1];
        let second = indices[0];
        if !matches!(first, Expr::Identifier(fid) if *fid.name == **bq) {
            return false;
        }
        let is_iv = |x: &Expr| matches!(x, Expr::Identifier(id) if *id.name == *self.iv);
        if is_iv(second) {
            return true; // bare `t`: one column per thread
        }
        let Expr::BinaryOp(b) = second else {
            return false;
        };
        if !matches!(b.op, BinaryOp::Add) {
            return false;
        }
        let is_scaled_local = |x: &Expr| -> bool {
            let Expr::BinaryOp(m) = x else {
                return false;
            };
            if !matches!(m.op, BinaryOp::Mul) {
                return false;
            }
            let stride_of = |e: &Expr| parallel_int_literal(e) == Some(*k);
            let local_of = |e: &Expr| {
                matches!(e, Expr::Identifier(id)
                    if *id.name != *self.iv && *id.name != **bq && self.declared.contains(&*id.name))
            };
            (local_of(&m.lhs) && stride_of(&m.rhs)) || (stride_of(&m.lhs) && local_of(&m.rhs))
        };
        (is_iv(&b.lhs) && is_scaled_local(&b.rhs)) || (is_scaled_local(&b.lhs) && is_iv(&b.rhs))
    }

    /// The two-level widening of the first-index rule (Vx#379): `block_iv * K + iv` in either
    /// operand order, with `K` exactly the thread loop's trip -- see `parallel_two_level` for why
    /// the multiplier must equal the trip.
    fn affine_first(&self, e: &Expr) -> bool {
        let Some((bq, k)) = &self.affine else {
            return false;
        };
        let Expr::BinaryOp(b) = e else {
            return false;
        };
        if !matches!(b.op, BinaryOp::Add) {
            return false;
        }
        let is_scaled_block = |x: &Expr| -> bool {
            let Expr::BinaryOp(m) = x else {
                return false;
            };
            matches!(m.op, BinaryOp::Mul)
                && ((matches!(&*m.lhs, Expr::Identifier(id) if *id.name == **bq)
                    && parallel_int_literal(&m.rhs) == Some(*k))
                    || (matches!(&*m.rhs, Expr::Identifier(id) if *id.name == **bq)
                        && parallel_int_literal(&m.lhs) == Some(*k)))
        };
        let is_iv = |x: &Expr| matches!(x, Expr::Identifier(id) if *id.name == *self.iv);
        (is_scaled_block(&b.lhs) && is_iv(&b.rhs)) || (is_iv(&b.lhs) && is_scaled_block(&b.rhs))
    }

    /// A read-position expression: allowed unless it could hide a write (a call, a print, a
    /// transfer) or is a construct this walk has no sound story for.
    fn expr(&mut self, e: &Expr) -> bool {
        match e {
            // A name read whole may read every row of it.
            Expr::Identifier(id) => {
                self.note(&id.name);
                if !self.declared.contains(&*id.name) {
                    self.captured_nonown_read
                        .entry(id.name.to_string())
                        .or_default()
                        .push(Vec::new());
                    self.other_row_at
                        .entry(id.name.to_string())
                        .or_insert(id.span);
                }
                true
            }
            Expr::Number(_) | Expr::StringLiteral(_) | Expr::SizeOf(_) | Expr::EnumVariant(_) => {
                true
            }
            Expr::BinaryOp(b) => self.expr(&b.lhs) && self.expr(&b.rhs),
            Expr::RelationalOp(b) => self.expr(&b.lhs) && self.expr(&b.rhs),
            Expr::LogicalOp(b) => self.expr(&b.lhs) && self.expr(&b.rhs),
            Expr::UnaryOp(u) => self.expr(&u.expr),
            Expr::AsCast(c) => self.expr(&c.expr),
            Expr::IndexAccess(_) => {
                let mut root = e;
                let mut indices = Vec::new();
                while let Expr::IndexAccess(step) = root {
                    indices.push(&*step.index);
                    root = &step.base;
                }
                let Expr::Identifier(id) = root else {
                    return self.expr(root) && indices.iter().all(|i| self.expr(i));
                };
                // Ledger for the same-loop shared read/write race check (Vx#379 stage B): a
                // read of a block-shared tensor at a row other than the thread loop's own IV.
                // Reading a neighbour's row is exactly what a consume phase does to a tile the
                // FILL phase wrote -- legal across a barrier, a race within one loop.
                if self.affine.is_some() && self.smem.contains(&*id.name) {
                    let own = matches!(
                        indices.last(),
                        Some(Expr::Identifier(fid)) if *fid.name == *self.iv
                    );
                    if !own {
                        // The read-after-write seam: a neighbour's row of a tensor
                        // written since the last barrier -- the fill may not be
                        // visible yet. Legal only across a barrier.
                        if self.smem_written.contains(&*id.name) {
                            return self.fail(
                                &format!(
                                    "a pass reads another pass's row of `{}` after it was \
                                     written, with no `barrier()` between",
                                    id.name
                                ),
                                e.span(),
                            );
                        }
                        self.smem_nonown_read.insert(id.name.to_string());
                    }
                }
                // A row of a tensor from outside the region that this iteration may not own.
                if !self.declared.contains(&*id.name) {
                    if self.owns_row(&indices) {
                        if let Some(first) = indices.last() {
                            self.record_row(&id.name, first, e.span());
                        }
                    } else {
                        self.captured_nonown_read
                            .entry(id.name.to_string())
                            .or_default()
                            .push(columns(&indices));
                        self.other_row_at
                            .entry(id.name.to_string())
                            .or_insert(e.span());
                    }
                }
                self.note(&id.name);
                indices.iter().all(|i| self.expr(i))
            }
            Expr::MemberAccess(m) => self.expr(&m.base),
            Expr::Array(a) => a.elements.iter().all(|el| self.expr(el)),
            Expr::Range(r) => self.expr(&r.start) && self.expr(&r.end),
            // The checker has already rewritten `.exp()` into `f32$exp(x)` by the time this walk
            // runs, so calls are the shape scalar math arrives in. Only the known-pure intrinsics
            // pass; any other call could write through an argument and rejects the region. A
            // `Borrow` shows up when such an intrinsic takes `&self` — harmless only immutably.
            Expr::FunctionCall(fc) => {
                let pure =
                    parallel_pure_call(&fc.name) || self.refuse(&call_blocker(&fc.name), fc.span);
                pure && fc.args.iter().all(|a| self.expr(a))
            }
            // A `&mut` borrow may be written through, anywhere.
            Expr::Borrow(b) => {
                (!b.is_mut || self.refuse_write("it takes a `&mut` borrow", e.span()))
                    && self.expr(&b.expr)
            }
            // Belt for the pre-rewrite spelling, should this walk ever run on an unchecked AST: a
            // method rooted at a captured name is refused (`&mut self` mutation is a write this
            // walk cannot see); on a local or a computed scalar it is thread-private either way.
            Expr::MethodCall(mc) => {
                let mut root = &*mc.base;
                while let Expr::IndexAccess(ix) = root {
                    root = &ix.base;
                }
                if let Expr::Identifier(id) = root {
                    if !self.declared.contains(&*id.name)
                        && !self.refuse_write(
                            &format!("it calls a method on `{}`, which may change it", id.name),
                            mc.span,
                        )
                    {
                        return false;
                    }
                }
                self.expr(&mc.base) && mc.args.iter().all(|a| self.expr(a))
            }
            Expr::If(i) => {
                // An `if` is no loop, so its blocks are at the loop depth of the statement it is
                // in, and a `break` in them leaves that loop. Threads may disagree about entering
                // a branch, so no barrier position inside one is uniform.
                let depth = self.depth;
                let saved = self.barrier_ok;
                self.barrier_ok = false;
                // `iv % c == 0` holds only in the `then` block.
                let outer = self.multiple_of.len();
                let multiple = self.multiple_required(&i.cond);
                self.multiple_of.extend(multiple);
                let then_ok = self.expr(&i.cond) && self.stmts(&i.then_block, depth);
                self.multiple_of.truncate(outer);
                let ok = then_ok && i.else_block.as_ref().is_none_or(|b| self.stmts(b, depth));
                self.barrier_ok = saved;
                ok
            }
            // Everything else — calls, prints, transfers, spawns, closures, borrows, raw pointers,
            // inline MLIR, autodiff, matches, struct construction — is either a possible write or
            // a construct with no sound striding story. Reject; the loop stays serial.
            other => self.refuse(&format!("it holds {}", construct(other)), other.span()),
        }
    }
}

/// Why a call stops the split. Only a call to a function known to write nothing can run on
/// every thread at once.
fn call_blocker(name: &str) -> String {
    // A method or a generic function's instance has a `$` in its name, which the program
    // never spelled; the warning's position shows which call it is.
    if name.contains('$') {
        "it calls a function that the compiler cannot show writes nothing".to_string()
    } else {
        format!("it calls `{name}`, which the compiler cannot show writes nothing")
    }
}

/// What an expression the walk does not split is, for a warning.
fn construct(e: &Expr) -> &'static str {
    match e {
        Expr::InlineMlir(_) => "inline MLIR, which the compiler cannot see into",
        Expr::Transfer(_) => "a `transfer`",
        Expr::SpawnOn(_) => "a `spawn` of its own",
        Expr::Closure(_) => "a closure",
        Expr::Match(_) => "a `match`",
        Expr::UnsafeBlock(_) => "an `unsafe` block",
        Expr::IndirectCall(_) => "a call through a function value",
        Expr::Dereference(_) => "a read through `*`",
        _ => "an expression that is not split across threads",
    }
}
