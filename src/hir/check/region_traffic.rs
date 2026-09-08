//===- check/region_traffic.rs - Vx Compiler --------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// What a `spawn on(...)` region moves, counted from its own code (Vx#353 A4 T4).
//
// The transfer-hop counts in `raw.rs` answer "what did it cost to get this tile to this
// space". This answers the question after it: what does the kernel then DO with the tile.
// That is where re-reading lives, and no declared edge cost can reach it -- the movement
// across the edge happened exactly once, so the edge is silent about a body that reads the
// arrived tile sixteen times.
//
// Attention is the shape that makes it concrete. A query loop over key blocks re-reads K and
// V on every query iteration, so K's reads are queries x keys x d elements against a
// footprint of keys x d: an amplification factor that is a property of the LOOP NEST, not of
// any edge, and that nobody declared anywhere.
//
// TOUCHES, NOT FOOTPRINT. The count is how many times an access executes, never how many
// distinct elements it reaches. `k[t * 2 + jj][d]` is not analysed as an index; it is counted
// once per execution of the statement containing it. Reading one element a thousand times is
// a thousand reads, which is the fact this stage exists to report.
//
// PER LAUNCH. One record describes one `spawn` site and what a single execution of it moves.
// A `spawn` inside a host loop still yields one record holding one launch's bytes: the host
// loop's trip count belongs to the caller and is not always static, so folding it in here
// would make the figure depend on something this walker cannot always see. Stated in the
// schema doc too, because a consumer totalling a program's traffic by summing records
// undercounts by exactly that factor.
//
// The honesty rules are `raw.rs`'s, deliberately: one thing this walker cannot count exactly
// makes the WHOLE region absent-with-a-reason, never partial and never zero. A silent
// undercount would be worse than no figure at all, because a consumer cannot tell it from a
// measurement of an efficient kernel.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::check::raw::overflowed;
use std::collections::BTreeMap;

/// Bytes read and written against one placed buffer, while a region is being counted.
#[derive(Clone, PartialEq, Eq, Default)]
struct BufferAcc {
    space: Option<MemorySpace>,
    read: u64,
    written: u64,
}

/// Per-buffer accumulator for one `spawn` region.
///
/// A `BTreeMap` rather than a `HashMap` because this feeds a published record, and this tree
/// tests that its output is byte-identical across thread counts. Iteration order of a
/// `HashMap` is not a nit here; it is a bug.
#[derive(Clone, PartialEq, Eq, Default)]
struct RegionAcc {
    /// Keyed by (spelled name, space name), never by the name alone: a body-local `let` can
    /// rebind a name to a DIFFERENT placed buffer in a different space, and a name-only key
    /// would fold the second buffer's bytes into the first buffer's row under the first
    /// buffer's space -- emptying a space that the kernel really touched.
    per_buffer: BTreeMap<(String, String), BufferAcc>,
}

impl RegionAcc {
    fn slot(&mut self, buffer: &str, space: &MemorySpace) -> &mut BufferAcc {
        let e = self
            .per_buffer
            .entry((buffer.to_string(), space.name()))
            .or_default();
        e.space = Some(space.clone());
        e
    }

    fn add_read(&mut self, buffer: &str, space: &MemorySpace, bytes: u64) -> Result<(), String> {
        let e = self.slot(buffer, space);
        e.read = e.read.checked_add(bytes).ok_or_else(overflowed)?;
        Ok(())
    }

    fn add_written(&mut self, buffer: &str, space: &MemorySpace, bytes: u64) -> Result<(), String> {
        let e = self.slot(buffer, space);
        e.written = e.written.checked_add(bytes).ok_or_else(overflowed)?;
        Ok(())
    }

    /// The per-buffer maximum of two branch outcomes, matching `TrafficAcc::max_with`: one
    /// execution takes one arm, so summing them would overstate what the hardware does. The
    /// caller marks the result inexact -- and only when the arms actually disagree.
    fn max_with(&self, other: &RegionAcc) -> RegionAcc {
        let mut out = self.clone();
        for (key, b) in &other.per_buffer {
            let e = out.per_buffer.entry(key.clone()).or_default();
            if e.space.is_none() {
                e.space = b.space.clone();
            }
            e.read = e.read.max(b.read);
            e.written = e.written.max(b.written);
        }
        out
    }
}

/// A placed value: its space and element size, or the reason it cannot be counted (a sub-byte
/// element type). The error is carried rather than dropped so that indexing such a value
/// refuses honestly instead of silently contributing nothing.
type Placed = Result<(MemorySpace, u64), String>;

/// The placed names visible to a region, captured from the enclosing scopes BEFORE the body is
/// checked. Only placed names appear.
///
/// A snapshot rather than live lookups, because checking the body MUTATES the scopes it reads:
/// a call that consumes a placed tensor moves it out, and a later live lookup then finds
/// nothing. That made a kernel handing a tile to a function report no traffic at all -- both
/// the call itself and every subsequent access to the moved name silently vanished, which is
/// precisely the undercount this stage must never produce. Found by probing, not by review.
type PlacedMap = std::collections::HashMap<String, Placed>;

/// Names bound inside the region body: `Some` when the binding is placed, `None` when the walk
/// has determined it is not. A body-local `let` shadows an outer name of the same spelling.
type Bindings = std::collections::HashMap<String, Option<Placed>>;

/// Does this subtree contain any index access at all?
///
/// The guard on the walker's catch-all, mirroring `traffic_expr`'s: an unrecognised construct
/// is refused only when it could HIDE a movement. A subtree with no indexing in it cannot
/// touch a placed buffer, so treating it as empty is sound rather than optimistic.
///
/// Via the `Debug` rendering rather than a hand-written traversal, deliberately. A traversal
/// would need an arm per `Expr` variant, and the failure mode of forgetting one is a subtree
/// silently reported as containing no accesses -- an UNDERCOUNT, the one error this stage
/// must never make. `Debug` derives recursively over every variant, including any added
/// later, so it cannot miss. Its error direction is a false positive, which costs an honest
/// refusal rather than a wrong number, and this runs only on constructs the walker did not
/// recognise in the first place.
fn contains_index(e: &Expr) -> bool {
    format!("{e:?}").contains("IndexAccess")
}

/// Does this subtree mention any of `names`?
///
/// The second half of the catch-all guard, and the one that closes a gap `contains_index`
/// alone leaves open: a construct the walker does not recognise can hand a placed tensor
/// somewhere opaque WITHOUT indexing it -- `f(ad)` through a function pointer, an `InlineMlir`
/// block, a `print(ad)`. No index appears, so an index-only guard waves it through and every
/// access the callee performs is missing from the total.
///
/// Same `Debug`-based reasoning as `contains_index`, and the same error direction: it can only
/// over-refuse. A name is matched as a quoted token so that `a` does not match `abc`.
///
/// Covered since the campaign's own #354/#355 fixes unblocked the programs that reach it
/// (`an_unrecognised_construct_that_hands_off_a_placed_tensor_is_refused`); an earlier note
/// here recorded it as uncoverable, which was true only until those landed.
fn mentions_any(e: &Expr, names: impl Iterator<Item = String>) -> bool {
    let rendered = format!("{e:?}");
    names
        .into_iter()
        .any(|n| rendered.contains(&format!("\"{n}\"")))
}

impl<'a> TypeChecker<'a> {
    /// The memory space a value of type `ty` is PLACED in, or `None` if it is not placed.
    ///
    /// Deliberately not `value_memory_space`, which falls back to the space of the ambient
    /// topology when a type carries no placement of its own. Inside a `spawn on(Topology::GPU)`
    /// that fallback reports GPU_HBM for an ordinary `Tensor<f32>([1, 2])` declared in the
    /// body -- a scratch tile that was never transferred anywhere. Counting it would put
    /// kernel-local scratch into the device-memory traffic figure, which is exactly the kind
    /// of plausible-but-wrong number this stage must not publish.
    ///
    /// So placement must be carried by the TYPE: a tensor's own `Placement` (what a
    /// `transfer` yields, and what `Memory::` on a declaration writes), the older
    /// `Pinned` wrapper, or a `Ref` annotated with a space. Anything else is unplaced
    /// and contributes nothing.
    fn placed_space(&self, ty: &Type) -> Option<MemorySpace> {
        // The placement names the space outright, which is the case this stage wants:
        // it is carried by the type rather than guessed from a topology, so kernel-local
        // scratch -- which carries none -- still contributes nothing.
        if let Some(p) = ty.placement() {
            return Some(p.space.clone());
        }
        match ty {
            Type::Ref(_, mem) => Some(mem.clone()),
            Type::Pinned(_, topo) if !matches!(topo, Topology::Current) => {
                Some(self.transfer_cost_graph.default_memory_for(topo))
            }
            _ => None,
        }
    }

    /// Element size in bytes of a placed tensor type, or an error naming why it cannot be
    /// counted. Sub-byte elements are refused rather than rounded up, for the reason the A4
    /// review established: rounding i4 to a byte made a faithful copy report the same 2.0
    /// ratio that is this stage's evidence of waste, and a figure indistinguishable from the
    /// thing it exists to detect is worse than no figure.
    fn region_elem_bytes(&self, ty: &Type) -> Result<u64, String> {
        // `as_tensor_operand` already peels `Pinned`/`Ref`/`Borrow`/`Pointer` down to the
        // tensor, so the placement wrapper needs no unwrapping here.
        let (elem, _, _) = Self::as_tensor_operand(ty)
            .ok_or_else(|| "a placed value indexed here is not a tensor".to_string())?;
        let bits = crate::hir::memory::element_bits(elem)
            .ok_or_else(|| "a placed tensor's element type has no known width".to_string())?;
        if bits % 8 != 0 {
            return Err(
                "a placed tensor has sub-byte elements, whose packed byte count this counter \
                 does not model"
                    .to_string(),
            );
        }
        Ok(bits / 8)
    }

    /// Capture the placed names the enclosing scopes hold, for use by a region walk.
    ///
    /// Called BEFORE the region body is checked. See [`PlacedMap`] for why a snapshot rather
    /// than live lookups.
    pub(crate) fn placed_names_snapshot(&self) -> PlacedMap {
        let mut out = PlacedMap::new();
        // Outermost first, so an inner scope's binding overwrites (shadows) an outer one.
        for scope in self.scopes.iter() {
            for (name, (ty, _)) in scope {
                let Some(space) = self.placed_space(ty) else {
                    // Shadowed by an UNPLACED binding of the same name: the outer placed
                    // entry must go, or an access here would be attributed to a space this
                    // name no longer denotes.
                    out.remove(name.as_ref());
                    continue;
                };
                let entry = match self.region_elem_bytes(ty) {
                    Ok(elem) => Ok((space, elem)),
                    Err(why) => Err(why),
                };
                out.insert(name.as_ref().to_string(), entry);
            }
        }
        out
    }

    /// What a name denotes during a region walk: a body-local binding first (it shadows),
    /// then the snapshot of the enclosing scopes.
    fn region_placed(&self, name: &str, outer: &PlacedMap, binds: &Bindings) -> Option<Placed> {
        if let Some(entry) = binds.get(name) {
            return entry.clone();
        }
        outer.get(name).cloned()
    }

    /// The placed buffer an index expression ultimately touches, as
    /// `(name, space, element bytes)`. `Ok(None)` when the base is not a placed buffer.
    ///
    /// `a[i][d]` nests: the outer `IndexAccess`'s base is another `IndexAccess`. Peeling to
    /// the identifier is what makes a rank-2 access ONE touch rather than two.
    fn indexed_buffer(
        &self,
        e: &Expr,
        outer: &PlacedMap,
        binds: &Bindings,
    ) -> Result<Option<(String, MemorySpace, u64)>, String> {
        let mut cur = e;
        loop {
            match cur {
                Expr::IndexAccess(ix) => cur = &ix.base,
                // `&ad` and `*r` denote the same buffer `ad` does: placement is transparent
                // through them, so they are peeled like a subscript is. Without this,
                // `(*r)[i][d]` looks like an index with no name under it.
                Expr::Borrow(b) => cur = &b.expr,
                Expr::Dereference(d) => cur = &d.expr,
                _ => break,
            }
        }
        let Expr::Identifier(id) = cur else {
            // NOT the same as "the base is provably unplaced". This is "I cannot tell", and
            // only the first is safe to report as zero: an unrecognised base that happens to
            // be a placed tile would contribute nothing, with no reason given, which is the
            // silent undercount this stage exists to prevent.
            return Err(
                "an indexed access in the region has a base this counter cannot resolve to a \
                 named buffer, so whether it touches a placed tensor is unknown"
                    .to_string(),
            );
        };
        let name = id.name.as_ref().to_string();
        match self.region_placed(&name, outer, binds) {
            Some(Ok((space, elem))) => Ok(Some((name, space, elem))),
            Some(Err(why)) => Err(why),
            None => Ok(None),
        }
    }

    /// A bound that folds to an integer, in exact `i64` arithmetic; `None` when it does not.
    ///
    /// `fold_raw_extent` rewrites `raw::extent(t)` to a literal but does no arithmetic, so
    /// `0..(1 + 1)` arrives as a `BinaryOp` and `-2..2` as a `UnaryOp` -- bounds the rest of
    /// the compiler resolves statically, refused here with a reason that says the bound is not
    /// statically known, which is untrue and silences a whole region.
    ///
    /// Deliberately not `eval_expr`: it computes in `f64` (a bound past 2^53 comes back
    /// rounded, and a rounded trip count is a WRONG byte figure, not a refusal) and it folds
    /// `Not` but not `Neg`. Checked integer arithmetic here, and anything not exactly foldable
    /// -- `Div`, whose truncation this does not model, included -- still returns `None`, which
    /// is the refusal direction.
    fn const_i64(&self, e: &Expr) -> Option<i64> {
        match e {
            Expr::Number(n) => n.value.as_ref().parse::<i64>().ok(),
            Expr::UnaryOp(u) if matches!(u.op, UnaryOp::Neg) => {
                self.const_i64(&u.expr)?.checked_neg()
            }
            Expr::BinaryOp(b) => {
                let (l, r) = (self.const_i64(&b.lhs)?, self.const_i64(&b.rhs)?);
                match b.op {
                    BinaryOp::Add => l.checked_add(r),
                    BinaryOp::Sub => l.checked_sub(r),
                    BinaryOp::Mul => l.checked_mul(r),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Trip count of `for _ in a..b` when both bounds fold to literals; `None` otherwise.
    ///
    /// A sibling of `static_trip_count` rather than a reuse of it: that one folds
    /// `raw::extent(src|dst)` against a lowering's two declared tile parameters, which do not
    /// exist here. An empty or reversed range runs zero times.
    fn region_trip_count(&self, iterable: &Expr) -> Option<u64> {
        let Expr::Range(r) = iterable else {
            return None;
        };
        let lit = |e: &Expr| -> Option<i64> { self.const_i64(&self.fold_raw_extent(e)) };
        let (a, b) = (lit(&r.start)?, lit(&r.end)?);
        Some(if b > a { (b - a) as u64 } else { 0 })
    }

    /// Count one `spawn` region, or say why it cannot be counted.
    ///
    /// Run while the region's scope is still live, so free names in the body resolve to the
    /// enclosing function's placed values.
    pub(crate) fn derive_spawn_traffic(
        &self,
        stmts: &[Statement],
        ret: Option<&Expr>,
        outer: &PlacedMap,
    ) -> Result<(crate::report::Traffic, Vec<crate::report::BufferTraffic>), String> {
        let mut acc = RegionAcc::default();
        let mut exact = true;
        let mut binds = Bindings::new();
        self.region_stmts(stmts, 1, &mut acc, &mut exact, outer, &mut binds, true)?;
        if let Some(r) = ret {
            self.region_expr(r, 1, &mut acc, &mut exact, outer, &mut binds)?;
        }

        // Per buffer, sorted by name (the BTreeMap already is), then folded into per-space
        // totals sorted by space name so the record is stable run to run.
        let mut by_buffer: Vec<crate::report::BufferTraffic> = Vec::new();
        for ((name, _), b) in &acc.per_buffer {
            let Some(space) = &b.space else { continue };
            by_buffer.push(crate::report::BufferTraffic {
                buffer: name.clone(),
                space: space.clone(),
                read_bytes: b.read,
                written_bytes: b.written,
            });
        }
        let mut per_space: BTreeMap<String, crate::report::SpaceTraffic> = BTreeMap::new();
        for b in &by_buffer {
            let e =
                per_space
                    .entry(b.space.name())
                    .or_insert_with(|| crate::report::SpaceTraffic {
                        space: b.space.clone(),
                        read_bytes: 0,
                        written_bytes: 0,
                    });
            e.read_bytes = e
                .read_bytes
                .checked_add(b.read_bytes)
                .ok_or_else(overflowed)?;
            e.written_bytes = e
                .written_bytes
                .checked_add(b.written_bytes)
                .ok_or_else(overflowed)?;
        }
        Ok((
            crate::report::Traffic {
                per_space: per_space.into_values().collect(),
                source: crate::report::TrafficSource::SpawnRegion,
                exact,
            },
            by_buffer,
        ))
    }

    // Eight parameters, one over the lint's threshold. This is a recursive walker threading four
    // pieces of mutable state (`acc`, `exact`, `binds`, and the `mult`/`top_level` position) down
    // an AST, and the alternative -- bundling them into a context struct -- is a refactor of the
    // traversal rather than a fix for a lint. Left as-is deliberately; see Vx#361, which is where
    // this became visible.
    #[allow(clippy::too_many_arguments)]
    fn region_stmts(
        &self,
        stmts: &[Statement],
        mult: u64,
        acc: &mut RegionAcc,
        exact: &mut bool,
        outer: &PlacedMap,
        binds: &mut Bindings,
        top_level: bool,
    ) -> Result<(), String> {
        for s in stmts {
            match s {
                Statement::LetDecl(l) => {
                    self.region_expr(&l.expr, mult, acc, exact, outer, binds)?;
                    // Record what this name now denotes, so a later index through it is
                    // attributed to the right space -- and so a scratch tile declared in the
                    // body is remembered as unplaced rather than resolved against an outer
                    // name that happens to match.
                    //
                    // An initializer the resolver does not recognise (a call, an if-, a
                    // match-expression) can still RETURN a placed tensor, and calling that
                    // "unplaced" published exact-zero traffic for every read through the
                    // binding (found by the codegen review). At the region's top level the
                    // checker has already typed the binding and its scope is still live, so
                    // its answer is authoritative. In a nested block that scope is gone by
                    // walk time and a name lookup could hit an OUTER same-name binding, so
                    // the honest verdict is Err -- uncountable, with a reason -- not a guess
                    // in either direction.
                    let placed = self
                        .region_binding_placed(&l.expr, outer, binds)
                        .or_else(|| {
                            if !Self::init_can_carry_placement(&l.expr) {
                                return None;
                            }
                            // A type annotation settles it wherever the binding lives: an
                            // annotated-unplaced `let m_new : f32 = if ...` is provably not a
                            // tensor, nested or not, and an annotated-placed one carries its
                            // space on its face. Without this, every scalar if-expression
                            // binding inside a loop was refused -- which took the flash
                            // fixture's count from exact to absent, the wrong direction of
                            // honest.
                            if let Some(ann) = &l.ty_ann {
                                return self.placed_space(ann).map(|space| {
                                    match self.region_elem_bytes(ann) {
                                        Ok(elem) => Ok((space, elem)),
                                        Err(why) => Err(why),
                                    }
                                });
                            }
                            if top_level {
                                match self.lookup(l.name.as_ref()) {
                                    Some((ty, _)) => match self.placed_space(ty) {
                                        Some(space) => {
                                            let ty = ty.clone();
                                            Some(match self.region_elem_bytes(&ty) {
                                                Ok(elem) => Ok((space, elem)),
                                                Err(why) => Err(why),
                                            })
                                        }
                                        None => None,
                                    },
                                    None => {
                                        Some(Err("a region binding's type could not be resolved"
                                            .to_string()))
                                    }
                                }
                            } else {
                                Some(Err(
                                    "a nested region binding is initialized by an expression \
                                     this counter cannot resolve"
                                        .to_string(),
                                ))
                            }
                        });
                    binds.insert(l.name.as_ref().to_string(), placed);
                }
                Statement::Return(r) => {
                    self.region_expr(&r.expr, mult, acc, exact, outer, binds)?
                }
                Statement::ExprStmt(e) => {
                    self.region_expr(&e.expr, mult, acc, exact, outer, binds)?
                }
                Statement::Assign(a) => {
                    // The right-hand side is evaluated first, and every placed index in it is
                    // a READ. Then the destination's own subscripts are read, and the
                    // destination itself is ONE write -- not a read: walking the lhs as an
                    // ordinary expression (which is what the lowering counter does, where an
                    // lhs is never an indexed tile) would book `o[i][d] = ...` as a read of
                    // `o` that the program never performs.
                    self.region_expr(&a.rhs, mult, acc, exact, outer, binds)?;
                    self.region_subscripts(&a.lhs, mult, acc, exact, outer, binds)?;
                    self.region_write(&a.lhs, mult, acc, outer, binds)?;
                }
                Statement::CompoundAssign(c) => {
                    // `x += y` reads the destination, then writes it. Missing the read is the
                    // undercount this arm exists to prevent.
                    //
                    // `region_expr` on the lhs already books the read AND walks its subscripts,
                    // so only the write is added here. Walking the lhs a second time would
                    // count its subscripts twice -- harmless for `o[i][d]`, where the indices
                    // are scalars, but a real over-count for `o[p[0]] += ...`, where the
                    // subscript is itself a read of a placed buffer.
                    self.region_expr(&c.rhs, mult, acc, exact, outer, binds)?;
                    self.region_expr(&c.lhs, mult, acc, exact, outer, binds)?;
                    self.region_write(&c.lhs, mult, acc, outer, binds)?;
                }
                Statement::Assert(a) => {
                    self.region_expr(&a.expr, mult, acc, exact, outer, binds)?
                }
                Statement::ForLoop(fl) => {
                    self.region_expr(&fl.iterable, mult, acc, exact, outer, binds)?;
                    let trips = self.region_trip_count(&fl.iterable).ok_or_else(|| {
                        "a loop bound in the region is not statically known, so the number of \
                         times its body runs cannot be counted"
                            .to_string()
                    })?;
                    let next = mult.checked_mul(trips).ok_or_else(|| {
                        "the region's loop nest trip count overflows a 64-bit counter".to_string()
                    })?;
                    // The induction variable is a fresh scalar, and the body's own bindings
                    // must not leak out of the loop.
                    let mut inner = binds.clone();
                    inner.insert(fl.iter.clone(), None);
                    self.region_stmts(&fl.body, next, acc, exact, outer, &mut inner, false)?;
                }
                Statement::Loop(_) => {
                    return Err(
                        "the region contains an unbounded `loop`, which runs an unknown number \
                         of times"
                            .to_string(),
                    )
                }
                Statement::Break(_) | Statement::Continue(_) => {
                    return Err(
                        "a `break`/`continue` in the region makes its loop trip counts a fiction"
                            .to_string(),
                    )
                }
                // Macros are expanded before type checking, so one reaching here is a
                // compiler bug rather than a program property -- and an unexpanded macro
                // could expand to any number of accesses. Refuse rather than count zero.
                Statement::MacroCall(_) => {
                    return Err(
                        "the region contains an unexpanded macro, whose accesses cannot be seen"
                            .to_string(),
                    )
                }
                Statement::Error(_) => {}
            }
        }
        Ok(())
    }

    /// Can this initializer form RETURN a placed tensor that `region_binding_placed` does not
    /// see? Calls and branch expressions can; literals and arithmetic over scalars cannot
    /// (a placed tensor inside them is caught by the walk of the expression itself).
    fn init_can_carry_placement(init: &Expr) -> bool {
        matches!(
            init,
            Expr::FunctionCall(_)
                | Expr::MethodCall(_)
                | Expr::IndirectCall(_)
                | Expr::If(_)
                | Expr::Match(_)
                | Expr::SpawnOn(_)
                | Expr::MemberAccess(_)
        )
    }

    /// What a `let` initializer places its name in, or `None` for unplaced. Only the forms
    /// that can actually carry a placement are recognised; everything else is unplaced, which
    /// is the safe answer because an unplaced name is simply not counted -- and a name wrongly
    /// believed placed would attribute a scratch tile's accesses to device memory.
    fn region_binding_placed(
        &self,
        init: &Expr,
        outer: &PlacedMap,
        binds: &Bindings,
    ) -> Option<Placed> {
        match init {
            // `let tile = transfer(ad, Memory::SMEM)` -- the placement this stage most wants
            // to see, because it is what makes a staged tile's re-reads attributable to SMEM
            // rather than to the space it came from. The element size comes from the value
            // being transferred: a transfer moves bytes, it does not convert them (C2).
            Expr::Transfer(t) => {
                let mut cur: &Expr = &t.expr;
                while let Expr::Borrow(b) = cur {
                    cur = &b.expr;
                }
                let Expr::Identifier(id) = cur else {
                    return None;
                };
                // The destination is placed in `t.space` NO MATTER where the source was:
                // staging a body-local scratch tile into SMEM yields an SMEM tile exactly as
                // staging a device tile does. Only the ELEMENT SIZE comes from the source, so
                // an unplaced source is sized from its checked type rather than making the
                // destination unplaced -- which would drop every access to it.
                let elem = match self.region_placed(id.name.as_ref(), outer, binds) {
                    Some(Ok((_, elem))) => Ok(elem),
                    // The source could not be sized, so neither can the destination.
                    Some(Err(why)) => Err(why),
                    None => match self.lookup(id.name.as_ref()) {
                        Some((ty, _)) => self.region_elem_bytes(ty),
                        None => Err("a tile staged inside the region comes from a value whose \
                                     type this counter cannot see, so its bytes cannot be sized"
                            .to_string()),
                    },
                };
                Some(elem.map(|e| (t.space.clone(), e)))
            }
            Expr::Identifier(id) => self.region_placed(id.name.as_ref(), outer, binds),
            // `let q = &mut ad[i][d]` binds a reference INTO a placed buffer. Recognising
            // it keeps the binding from reading as unplaced scratch; the write through it
            // is refused above, because following the reference is what this walk cannot do.
            Expr::Borrow(b) => self.region_binding_placed(&b.expr, outer, binds),
            Expr::IndexAccess(_) => match self.indexed_buffer(init, outer, binds) {
                Ok(Some((_, space, elem))) => Some(Ok((space, elem))),
                Ok(None) => None,
                Err(why) => Some(Err(why)),
            },
            _ => None,
        }
    }

    /// Walk the subscript expressions of an assignment destination, which are ordinary reads:
    /// `o[p[0]] = ...` reads `p` before it writes `o`.
    fn region_subscripts(
        &self,
        lhs: &Expr,
        mult: u64,
        acc: &mut RegionAcc,
        exact: &mut bool,
        outer: &PlacedMap,
        binds: &mut Bindings,
    ) -> Result<(), String> {
        let mut cur = lhs;
        while let Expr::IndexAccess(ix) = cur {
            self.region_expr(&ix.index, mult, acc, exact, outer, binds)?;
            cur = &ix.base;
        }
        Ok(())
    }

    /// Book a WRITE of the placed buffer an assignment destination names, and nothing else.
    /// The destination's subscripts are the caller's business, so that a caller which has
    /// already walked them does not walk them twice. A destination that is not a placed index
    /// (a scalar local, an unplaced scratch tile) moves nothing.
    fn region_write(
        &self,
        lhs: &Expr,
        mult: u64,
        acc: &mut RegionAcc,
        outer: &PlacedMap,
        binds: &Bindings,
    ) -> Result<(), String> {
        // A write THROUGH a dereference reaches whatever the reference points at, and this
        // walk cannot follow a reference to its referent. `let q = &mut ad[i][d]; *q = 3.0;`
        // wrote 16 bytes of device memory while the record said `written_bytes: 0,
        // exact: true` -- and booked the address computation's reads, so it was wrong in
        // both directions at once. Refuse: an unfollowable write is uncountable, and this
        // counter's rule is absent-with-a-reason over a confident zero.
        if matches!(lhs, Expr::Dereference(_)) {
            return Err(
                "the region writes through a dereference, and this counter cannot follow a                  reference to the buffer it points at"
                    .to_string(),
            );
        }
        if !matches!(lhs, Expr::IndexAccess(_)) {
            return Ok(());
        }
        if let Some((name, space, elem)) = self.indexed_buffer(lhs, outer, binds)? {
            let bytes = elem.checked_mul(mult).ok_or_else(overflowed)?;
            acc.add_written(&name, &space, bytes)?;
        }
        Ok(())
    }

    fn region_expr(
        &self,
        e: &Expr,
        mult: u64,
        acc: &mut RegionAcc,
        exact: &mut bool,
        outer: &PlacedMap,
        binds: &mut Bindings,
    ) -> Result<(), String> {
        match e {
            Expr::IndexAccess(ix) => {
                // Index sub-expressions are reads in their own right, and they are counted
                // whether or not the base turns out to be placed.
                self.region_expr(&ix.index, mult, acc, exact, outer, binds)?;
                let mut base: &Expr = &ix.base;
                while let Expr::IndexAccess(inner) = base {
                    self.region_expr(&inner.index, mult, acc, exact, outer, binds)?;
                    base = &inner.base;
                }
                if let Some((name, space, elem)) = self.indexed_buffer(e, outer, binds)? {
                    let bytes = elem.checked_mul(mult).ok_or_else(overflowed)?;
                    acc.add_read(&name, &space, bytes)?;
                } else {
                    // Not a placed buffer, but the base may still contain one
                    // (`scratch[q[i]]` reads q through an unplaced base).
                    self.region_expr(base, mult, acc, exact, outer, binds)?;
                }
                Ok(())
            }
            Expr::FunctionCall(fc) => {
                for a in &fc.args {
                    self.region_expr(a, mult, acc, exact, outer, binds)?;
                }
                self.region_call_opacity(&fc.args, None, outer, binds)
            }
            Expr::MethodCall(mc) => {
                self.region_expr(&mc.base, mult, acc, exact, outer, binds)?;
                for a in &mc.args {
                    self.region_expr(a, mult, acc, exact, outer, binds)?;
                }
                self.region_call_opacity(&mc.args, Some(&mc.base), outer, binds)
            }
            Expr::If(ifx) => {
                self.region_expr(&ifx.cond, mult, acc, exact, outer, binds)?;
                // `if comptime` is decided at compile time, so exactly one arm survives into
                // the emitted code and the count is exact. Weighing it as a runtime branch
                // would report the maximum of two arms for a body whose answer the compiler
                // already knows -- the same correction the lowering counter carries.
                if ifx.is_comptime {
                    if let Some(crate::hir::env::Value::Bool(taken)) =
                        self.eval_expr(&ifx.cond, &std::collections::HashMap::new())
                    {
                        let block = if taken {
                            Some(&ifx.then_block)
                        } else {
                            ifx.else_block.as_ref()
                        };
                        if let Some(b) = block {
                            let mut inner = binds.clone();
                            self.region_stmts(b, mult, acc, exact, outer, &mut inner, false)?;
                        }
                        return Ok(());
                    }
                }
                let mut then_acc = acc.clone();
                let mut then_binds = binds.clone();
                self.region_stmts(
                    &ifx.then_block,
                    mult,
                    &mut then_acc,
                    exact,
                    outer,
                    &mut then_binds,
                    false,
                )?;
                let mut else_acc = acc.clone();
                if let Some(eb) = &ifx.else_block {
                    let mut else_binds = binds.clone();
                    self.region_stmts(
                        eb,
                        mult,
                        &mut else_acc,
                        exact,
                        outer,
                        &mut else_binds,
                        false,
                    )?;
                }
                // Inexact only when the arms DISAGREE: a branch whose arms move identical
                // bytes has one answer, and flagging it would differ from the same program
                // written without the branch.
                if then_acc != else_acc {
                    *exact = false;
                }
                *acc = then_acc.max_with(&else_acc);
                Ok(())
            }
            Expr::Match(m) => {
                self.region_expr(&m.expr, mult, acc, exact, outer, binds)?;
                let entry = acc.clone();
                let mut outcomes: Vec<RegionAcc> = Vec::new();
                for arm in &m.arms {
                    let mut arm_acc = entry.clone();
                    let mut arm_binds = binds.clone();
                    self.region_stmts(
                        &arm.body,
                        mult,
                        &mut arm_acc,
                        exact,
                        outer,
                        &mut arm_binds,
                        false,
                    )?;
                    outcomes.push(arm_acc);
                }
                if outcomes.windows(2).any(|w| w[0] != w[1]) {
                    *exact = false;
                }
                let mut worst = entry;
                for o in &outcomes {
                    worst = worst.max_with(o);
                }
                *acc = worst;
                Ok(())
            }
            Expr::UnsafeBlock(u) => {
                let mut inner = binds.clone();
                self.region_stmts(&u.stmts, mult, acc, exact, outer, &mut inner, false)?;
                if let Some(r) = &u.ret {
                    self.region_expr(r, mult, acc, exact, outer, &mut inner)?;
                }
                Ok(())
            }
            Expr::ComptimeBlock(c) => {
                let mut inner = binds.clone();
                self.region_stmts(&c.stmts, mult, acc, exact, outer, &mut inner, false)?;
                if let Some(r) = &c.ret {
                    self.region_expr(r, mult, acc, exact, outer, &mut inner)?;
                }
                Ok(())
            }
            // A `transfer` inside the region stages a tile; the MOVEMENT it performs is
            // counted by that transfer's own route record, so counting it here too would
            // double it. Its operand's subscripts are still ordinary reads.
            Expr::Transfer(t) => self.region_expr(&t.expr, mult, acc, exact, outer, binds),
            // A nested `spawn` gets its own record; counting its body here as well would
            // report the same movement twice.
            Expr::SpawnOn(_) => Err("the region encloses a nested `spawn`, whose movement is \
                 reported on that region's own record and is not included here"
                .to_string()),
            Expr::BinaryOp(b) => {
                self.region_expr(&b.lhs, mult, acc, exact, outer, binds)?;
                self.region_expr(&b.rhs, mult, acc, exact, outer, binds)
            }
            Expr::RelationalOp(r) => {
                self.region_expr(&r.lhs, mult, acc, exact, outer, binds)?;
                self.region_expr(&r.rhs, mult, acc, exact, outer, binds)
            }
            Expr::LogicalOp(l) => {
                self.region_expr(&l.lhs, mult, acc, exact, outer, binds)?;
                self.region_expr(&l.rhs, mult, acc, exact, outer, binds)
            }
            Expr::UnaryOp(u) => self.region_expr(&u.expr, mult, acc, exact, outer, binds),
            Expr::Borrow(b) => self.region_expr(&b.expr, mult, acc, exact, outer, binds),
            Expr::Dereference(d) => self.region_expr(&d.expr, mult, acc, exact, outer, binds),
            Expr::AsCast(c) => self.region_expr(&c.expr, mult, acc, exact, outer, binds),
            Expr::Range(r) => {
                self.region_expr(&r.start, mult, acc, exact, outer, binds)?;
                self.region_expr(&r.end, mult, acc, exact, outer, binds)
            }
            Expr::MemberAccess(m) => self.region_expr(&m.base, mult, acc, exact, outer, binds),
            Expr::Identifier(_) | Expr::Number(_) | Expr::StringLiteral(_) => Ok(()),
            // Anything else is refused rather than assumed empty -- but only when it could
            // hide a movement, matching the lowering counter's catch-all. A subtree with no
            // indexing in it cannot touch a placed buffer.
            other => {
                if contains_index(other) {
                    return Err(
                        "the region contains a construct this counter does not know how to \
                         weigh, and it indexes something"
                            .to_string(),
                    );
                }
                // No indexing, but it may still hand a placed tensor somewhere opaque.
                let placed_names = outer.keys().cloned().chain(
                    binds
                        .iter()
                        .filter(|(_, v)| v.is_some())
                        .map(|(k, _)| k.clone()),
                );
                if mentions_any(other, placed_names) {
                    return Err(
                        "the region mentions a placed tensor inside a construct this counter \
                         does not know how to weigh"
                            .to_string(),
                    );
                }
                Ok(())
            }
        }
    }

    /// A call whose arguments (or receiver) include a placed buffer makes the region
    /// uncountable: the callee's body is not walked, so any accesses it performs would be
    /// silently missing from the total.
    ///
    /// Scoped to placed arguments on purpose. `(m - m_new).exp()` and `print(scalar)` move
    /// nothing a caller can see, and poisoning on every call would make almost every real
    /// kernel uncountable -- an honest `null` that is never anything else is not information.
    fn region_call_opacity(
        &self,
        args: &[Expr],
        receiver: Option<&Expr>,
        outer: &PlacedMap,
        binds: &Bindings,
    ) -> Result<(), String> {
        // Three-valued per argument: definitely placed (refuse, naming the tensor),
        // definitely not (fine), or unresolvable (refuse too, with its own reason -- an
        // argument this walk cannot classify could be a placed tensor, and "could be" is
        // enough, because the failure being prevented is a silent exact-zero).
        //
        // The peel loop covers every wrapper an argument can put around a name:
        // `f(ad)`, `f(ad[0])`, `f(&ad)`, and -- found by the codegen review publishing
        // exact-zero traffic -- `f(*r)` where `r = &ad`, and `f(h.t)` where the FIELD is
        // placed even though `h` is not. Members resolve through the struct environment;
        // a chain that resolves to a scalar or unplaced tensor is legitimately fine
        // (`f(cfg.max)` must not be refused as "a placed tensor").
        enum Arg {
            Placed,
            NotPlaced,
            Unresolvable,
        }
        let classify = |e: &Expr| -> Arg {
            let mut cur = e;
            loop {
                match cur {
                    Expr::IndexAccess(ix) => cur = &ix.base,
                    Expr::Borrow(b) => cur = &b.expr,
                    Expr::Dereference(d) => cur = &d.expr,
                    _ => break,
                }
            }
            match cur {
                Expr::Identifier(id) => match self.region_placed(id.name.as_ref(), outer, binds) {
                    // `Err` is a binding whose placement is known-uncountable (sub-byte,
                    // opaque initializer): not classifiable, so not passable.
                    Some(_) => Arg::Placed,
                    None => Arg::NotPlaced,
                },
                Expr::MemberAccess(_) => match self.member_field_type(cur, outer, binds) {
                    Some(ty) => {
                        if self.placed_space(&ty).is_some() {
                            Arg::Placed
                        } else {
                            Arg::NotPlaced
                        }
                    }
                    None => Arg::Unresolvable,
                },
                // Literals and arithmetic cannot be a placed tensor unless they CONTAIN a
                // mention of one, which the general mentions_any guard below covers.
                _ => Arg::NotPlaced,
            }
        };
        let mut unresolvable = false;
        for a in args.iter().chain(receiver) {
            match classify(a) {
                Arg::Placed => {
                    return Err(
                        "the region passes a placed tensor to a call, and the callee's own \
                         accesses are not counted here"
                            .to_string(),
                    )
                }
                Arg::Unresolvable => unresolvable = true,
                Arg::NotPlaced => {}
            }
        }
        if unresolvable {
            // Distinct reason on purpose: this is "cannot tell", not "saw a tensor".
            return Err(
                "the region passes an argument this counter cannot classify to a call; if it \
                 is a placed tensor, the callee's accesses would go uncounted"
                    .to_string(),
            );
        }
        // Backstop for argument shapes the classifier does not walk (nested calls, struct
        // inits, casts...): any mention of a placed name inside an argument subtree is a
        // handoff this counter cannot follow.
        let placed_names: Vec<String> = outer
            .keys()
            .cloned()
            .chain(
                binds
                    .iter()
                    .filter(|(_, v)| v.is_some())
                    .map(|(k, _)| k.clone()),
            )
            .collect();
        for a in args.iter().chain(receiver) {
            if matches!(
                a,
                Expr::Identifier(_) | Expr::IndexAccess(_) | Expr::Borrow(_)
            ) {
                continue; // already classified precisely above
            }
            if mentions_any(a, placed_names.iter().cloned()) {
                return Err(
                    "the region mentions a placed tensor inside a call argument this counter \
                     does not know how to weigh"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// The type of a (possibly nested) struct-field access rooted at an identifier, resolved
    /// through the checker's live scopes and the struct environment. `None` when any link of
    /// the chain cannot be resolved -- the caller must treat that as "could be anything",
    /// never as "not placed".
    fn member_field_type(&self, e: &Expr, outer: &PlacedMap, binds: &Bindings) -> Option<Type> {
        match e {
            Expr::MemberAccess(m) => {
                let base_ty = self.member_field_type(&m.base, outer, binds)?;
                let struct_name = match &base_ty {
                    Type::Struct(name, _) => name.clone(),
                    _ => return None,
                };
                let decl = self.env.structs.get(&struct_name)?;
                decl.fields
                    .iter()
                    .find(|(fname, _)| fname == &m.member)
                    .map(|(_, ty)| ty.clone())
            }
            Expr::Identifier(id) => {
                // A body-local rebinding shadows; its placement verdict is authoritative but
                // carries no full type, so resolution stops (=> conservative None upstream)
                // unless the name resolves through the live scopes.
                if binds.contains_key(id.name.as_ref()) {
                    return None;
                }
                let _ = outer; // placement of the ROOT is not the question; its type is
                self.lookup(id.name.as_ref()).map(|(t, _)| t.clone())
            }
            _ => None,
        }
    }
}
