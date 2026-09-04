//===- check submodule - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// One expression family's type checks, split out of the former 5000-line `hir/expr.rs` along the
// `check_expr_type_flag` dispatch seam (frontend_refactoring_borrow_checker.md R2, #279). An additional
// `impl TypeChecker` block; moves only, zero logic change. Reaches the shared types and helpers via
// `use super::super::*`, exactly as `expr.rs` uses `use super::*`.
//
// A transfer is checked in phases, one function each: resolve the edge, pick the lowering that
// runs on it, record the route and the traffic it moves, discharge the seam obligation, and give
// the result its new type. `check_transfer_expr` is the order they run in.
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// What resolving a transfer's edge produced: where the value is, where it is going, the route
/// the hardware takes between them, and what that route costs. Every later phase reads it.
pub(crate) struct ResolvedEdge {
    /// The type of the value being moved.
    pub inner_ty: Type,
    pub source_mem: MemorySpace,
    pub target_mem: MemorySpace,
    /// Every space the value passes through, source first. More than two means the transfer is
    /// staged, and gets rewritten into a chain of single hops.
    pub path: Vec<MemorySpace>,
    /// The cost graph's total for the route, which is the reachability cost rather than the
    /// bandwidth-derived one below.
    pub graph_cost: u32,
    /// Bytes this transfer moves, when the tile's shape is statically known.
    pub moved_bytes: Option<u64>,
    /// The bandwidth-derived (roofline) cost, with the unit it is in and which declaration it
    /// came from. Set when the declarations supply a rate for this edge; `graph_cost` is what
    /// the router used either way.
    pub derived_cost: Option<u64>,
    pub derived_unit: Option<crate::syntax::RatePer>,
    pub cost_source: Option<crate::report::CostSource>,
}

impl<'a> TypeChecker<'a> {
    /// Best-effort label for the buffer crossing a seam, taken from the transferred
    /// operand (an identifier, a borrow of one, or the base of a chained transfer).
    /// Used to instantiate the per-buffer obligation with the program's real name so
    /// distinct buffers produce distinct, localized diagnostics.
    pub(crate) fn buffer_label(e: &Expr) -> String {
        match e {
            Expr::Identifier(id) => id.name.as_ref().to_string(),
            Expr::Transfer(t) => Self::buffer_label(&t.expr),
            Expr::Borrow(b) => Self::buffer_label(&b.expr),
            _ => "buffer".to_string(),
        }
    }

    /// Source span of the transferred operand. The parser leaves transfer/method-call
    /// nodes with a default span, but the operand identifier carries a real one — using
    /// it lets each seam in a multi-stage pipeline be localized to its own statement.
    pub(crate) fn buffer_span(e: &Expr) -> Option<Span> {
        match e {
            Expr::Identifier(id) => Some(id.span),
            Expr::Transfer(t) => Self::buffer_span(&t.expr),
            Expr::Borrow(b) => Self::buffer_span(&b.expr),
            _ => None,
        }
    }

    /// Discharge the per-seam local-completeness / soundness obligation for one transfer
    /// hop `src -> dst` (see "Precision at the Boundary" and `crate::hir::seam`).
    ///
    /// The footprint is the actual `buffer` crossing this seam. Because tensor *contents*
    /// are opaque, the obligation tracks the coarsest abstraction — definite (`CONST`) vs
    /// possibly-stale (`TOP`): the producer establishes the buffer (definite), a
    /// synchronizing transfer is the identity (obligation `unsat` => ACCEPT), and a relaxed
    /// transfer sends the published buffer to `TOP`, so a consumer can read it stale
    /// (`sat` => REJECT, with a concrete counterexample naming the buffer). This is the
    /// per-buffer instance of the paper's `post /\ ~conclusion` schema; the value-contract
    /// form (`flag => data`) is the message-passing worked example (`seam::check_seam`).
    /// Pre-scan a statement block, recording `assert(var == const)` facts (the value a
    /// consumer requires of `var`). Recurses into nested blocks (`spawn`, `if`, loops),
    /// so a transfer seam checked *before* the consumer's `spawn` body can still consult
    /// the contract the consumer will impose on the transferred buffer.
    pub(crate) fn collect_assert_contracts(
        stmts: &[Statement],
        out: &mut std::collections::HashMap<String, u64>,
    ) {
        for s in stmts {
            match s {
                Statement::Assert(a) => Self::extract_eq_const(&a.expr, out),
                Statement::LetDecl(l) => Self::scan_expr_for_asserts(&l.expr, out),
                Statement::ExprStmt(e) => Self::scan_expr_for_asserts(&e.expr, out),
                Statement::Return(r) => Self::scan_expr_for_asserts(&r.expr, out),
                Statement::ForLoop(f) => Self::collect_assert_contracts(&f.body, out),
                _ => {}
            }
        }
    }

    /// Descend into the block-bearing expressions that can hold consumer asserts.
    pub(crate) fn scan_expr_for_asserts(
        e: &Expr,
        out: &mut std::collections::HashMap<String, u64>,
    ) {
        match e {
            Expr::SpawnOn(s) => {
                Self::collect_assert_contracts(&s.stmts, out);
                if let Some(r) = &s.ret {
                    Self::scan_expr_for_asserts(r, out);
                }
            }
            Expr::UnsafeBlock(b) => {
                Self::collect_assert_contracts(&b.stmts, out);
                if let Some(r) = &b.ret {
                    Self::scan_expr_for_asserts(r, out);
                }
            }
            Expr::ComptimeBlock(b) => {
                Self::collect_assert_contracts(&b.stmts, out);
                if let Some(r) = &b.ret {
                    Self::scan_expr_for_asserts(r, out);
                }
            }
            Expr::If(i) => {
                Self::collect_assert_contracts(&i.then_block, out);
                if let Some(eb) = &i.else_block {
                    Self::collect_assert_contracts(eb, out);
                }
            }
            _ => {}
        }
    }

    /// Recognize `ident == N` (or `N == ident`) with N a small non-negative integer,
    /// recording `ident -> N`. This is the conclusion of the boundary contract: the
    /// value the consumer asserts the buffer holds after the seam.
    pub(crate) fn extract_eq_const(e: &Expr, out: &mut std::collections::HashMap<String, u64>) {
        let Expr::RelationalOp(b) = e else { return };
        if b.op != RelationalOp::Eq {
            return;
        }
        let pair = match (&*b.lhs, &*b.rhs) {
            (Expr::Identifier(i), Expr::Number(n)) => Some((i, n)),
            (Expr::Number(n), Expr::Identifier(i)) => Some((i, n)),
            _ => None,
        };
        let Some((i, n)) = pair else { return };
        // Any non-negative integer that fits `u64` is a valid value-contract payload (the
        // seam field is `VAL_BITS = 64` wide). Parsing as u64 rejects negatives and floats
        // exactly, without the precision loss an f64 round-trip would introduce.
        if let Ok(v) = n.value.as_ref().parse::<u64>() {
            out.insert(i.name.as_ref().to_string(), v);
        }
    }

    /// If `name` holds a statically-known non-negative integer (tracked by the
    /// constant evaluator) that fits the seam obligation's value field, return it.
    /// This is what lets a seam be checked with a *value* contract rather than the
    /// coarser visibility one when the producer's payload is known at compile time.
    pub(crate) fn const_value_of(&self, name: &str) -> Option<u64> {
        let sym = crate::symbol::Symbol::from(name);
        for env in self.consteval.env.iter().rev() {
            if let Some(crate::hir::env::Value::Number(n)) = env.get(&sym) {
                // A non-negative integer the value field (seam::VAL_BITS = 64) can pin. The
                // source is an f64, so cap at 2^53 where every integer is still exact rather
                // than risk pinning a rounded value.
                const F64_EXACT_INT_MAX: f64 = (1u64 << 53) as f64;
                if n.fract() == 0.0 && *n >= 0.0 && *n <= F64_EXACT_INT_MAX {
                    return Some(*n as u64);
                }
                return None;
            }
        }
        None
    }

    /// Whether the hardware holding `space` can represent `elem` at all, per the `dtypes:` list
    /// on its topology. E6026.
    ///
    /// The capacity check asks whether a value fits; this asks whether the machine can hold that
    /// kind of value in the first place. Both read a declared property of a device and compare a
    /// placement against it, which is why they are called from the same four positions.
    ///
    /// Silent unless the topology declares `dtypes:`. Undeclared is permissive by construction --
    /// a machine file that says nothing about element types constrains nothing.
    pub(crate) fn check_element_type(
        &mut self,
        elem: &ElementType,
        space: &MemorySpace,
        context: &str,
    ) {
        // An un-substituted generic is not an element type yet; the instantiated body is checked.
        if matches!(elem, ElementType::Generic(_)) {
            return;
        }
        let Ok(topology) = self.transfer_cost_graph.owning_topology(space) else {
            return;
        };
        let Some(desc) = self.transfer_cost_graph.descriptor(&topology.kind()) else {
            return;
        };
        let Some(declared) = desc.dtypes.as_ref() else {
            return;
        };
        if declared.contains(elem) {
            return;
        }
        let listed = declared
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E6026,
            format!(
                "{context} has element type {elem}, which {} cannot represent; it declares [{}]",
                topology.display_name(),
                listed
            ),
            None,
        );
    }

    /// If a statically-shaped `Tensor<elem, dims>` placed in `space` exceeds that space's
    /// declared `capacity` (after granule rounding), emit E6009. No-op for a dynamic shape, an
    /// undeclared space, or a space with no `capacity`.
    pub(crate) fn check_capacity(
        &mut self,
        elem: &ElementType,
        dims: &[Expr],
        space: &MemorySpace,
        context: &str,
        span: &Span,
    ) {
        let sized = {
            let h = crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
            let Some(decl) = h.descriptor(space) else {
                return;
            };
            let Some(crate::syntax::ByteSize(cap)) = decl.capacity else {
                return;
            };
            let Some(raw) = crate::hir::memory::static_tensor_bytes(elem, dims) else {
                // A dynamic (non-literal) shape placed in a capacity-bounded space cannot be
                // capacity-checked (E6009/E6010) — warn rather than skip silently, so the user knows
                // the placement is unverified. We are past the `capacity` guard above, so there
                // genuinely was a bound to check against; and only monomorphized bodies reach here
                // (`check_function` skips generic templates), so a non-literal dim is a true runtime
                // value, not an un-substituted const generic. P0-4 / W1029.
                let dyn_note = dims
                    .iter()
                    .enumerate()
                    .find(|(_, d)| !matches!(d, Expr::Number(_)))
                    .map(|(i, d)| match d {
                        Expr::Identifier(id) => {
                            format!("dimension {i} is the runtime value '{}'", id.name)
                        }
                        _ => format!("dimension {i} is not a compile-time constant"),
                    })
                    .unwrap_or_else(|| "the shape is not statically known".to_string());
                self.errors
                    .warn(
                        crate::diagnostic::DiagnosticCode::W1029,
                        format!(
                            "capacity of '{}' not verified — {context} has a dynamic shape",
                            space.name()
                        ),
                        None,
                    )
                    .notes
                    .push(crate::diagnostic::Note {
                        message: dyn_note.into(),
                        span: None,
                    });
                return;
            };
            let rounded = crate::hir::memory::granule_round(raw, decl.granule.map(|g| g.0));
            (rounded, cap)
        };
        let (rounded, cap) = sized;
        // Precise per-tile check: a single tile larger than the whole space is always wrong.
        if rounded > cap {
            self.errors
                .error_with_code(
                    crate::diagnostic::DiagnosticCode::E6009,
                    format!(
                        "{context} needs {rounded} bytes but memory space '{}' has capacity {cap} bytes",
                        space.name()
                    ),
                    None,
                )
                // Machine-readable form of the same verdict, for `--diagnostics-json` (#282).
                .facts = Some(crate::diagnostic::DiagnosticFacts::Capacity {
                space: space.name(),
                required_bytes: rounded,
                available_bytes: cap,
                tiles: None,
            });
        }
        // Record for the cumulative (working-set) budget check at end of function, keyed by
        // *where the placement is written* rather than by the name it is bound to.
        //
        // The binding name collapsed two placements that shared one, which is a legal program:
        // shadowing is deliberate here, and nothing releases the tile the shadowed name held. So
        // `let s = ..; let s = ..;` recorded one tile of the later size, understating the working
        // set and reporting a resident-set figure that was the last placement's size rather than
        // the peak (Vx#443).
        //
        // A source position is the identity that survives the thing that goes wrong with a
        // counter: `check_capacity` has no `speculating` guard, unlike `record_staging_route`
        // beside it, so a node checked twice must land on the same key both times. A counter mints
        // a fresh one per visit and would inflate the sum; a span does not move.
        // Whatever the key is, it has to be the *same* on every visit to one placement. The
        // checker walks some nodes twice -- a method-call transfer is checked once as written and
        // again after it is rewritten into a call, both times with `speculating` false -- and
        // `check_capacity` has no guard against that, unlike `record_staging_route` beside it. A
        // counter mints a fresh key per visit and turns one 3 MiB tile into two.
        let key = if span.line != 0 || span.column != 0 {
            format!("@{}:{}:{}", span.line, span.column, span.length)
        } else if let Some(name) = &self.current_assignment_target {
            // A synthesized placement carrying no position: fall back to the binding, which is
            // what this keyed on before and is stable across the second visit. Two shadowed
            // placements that *both* lack a position still collapse, exactly as they did before.
            name.clone()
        } else {
            self.traffic.placement_site += 1;
            format!("@site{}", self.traffic.placement_site)
        };
        // Order is assigned once per placement. A node the checker visits twice keeps the
        // position it had the first time, so a re-check cannot make one tile look like two
        // that straddle a block boundary.
        let scope = self.traffic.scope_chain.clone();
        let tiles = self
            .traffic
            .memory_placements
            .entry(space.clone())
            .or_default();
        let order = match tiles.get(&key) {
            Some(existing) => existing.order,
            None => {
                self.traffic.placement_order += 1;
                self.traffic.placement_order
            }
        };
        tiles.insert(
            key,
            crate::hir::check_state::Placement {
                bytes: rounded,
                scope,
                order,
            },
        );
    }

    /// Cumulative budget check: for each memory space, the sum of the tiles a function places
    /// there (its working set) must fit `capacity`. This catches the collective overflow that
    /// the per-tile check (E6009) misses -- e.g. Q/K/V in SMEM or S/P/O in TMEM summing past the
    /// on-chip budget. Conservative (assumes all placed tiles coexist), so a space declared
    /// `overcommit` downgrades the error (E6010) to a warning (W1028). Clears the per-function
    /// placement map. Only fires when >1 tile shares a space (a lone tile is E6009's job).
    pub(crate) fn check_cumulative_capacity(&mut self) {
        let placements = std::mem::take(&mut self.traffic.memory_placements);
        self.traffic.placement_site = 0;
        let h = crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
        // (space, total, cap, tile_count, overcommit, granule)
        let mut violations: Vec<(MemorySpace, u64, u64, usize, bool, Option<u64>)> = Vec::new();
        // Sorted, because `placements` is a `HashMap` and Rust randomises its hasher per process.
        // Iterating it directly made `resident_sets` -- and therefore `--diagnostics-json` --
        // come out in a different order on every run: the same compiler on the same input emitted
        // `HBM` before `L2` once and after it the next time. That is a user-visible
        // nondeterminism in the machine-readable artifact downstream tools consume, and it broke
        // byte-reproducibility of the frozen predictions (vx-review#14), which is where it was
        // found. Diagnostics are also emitted in this order, so it decided their order too.
        let mut placements: Vec<_> = placements.iter().collect();
        placements.sort_by_key(|(space, _)| space.name());
        for (space, tiles) in placements {
            let Some(decl) = h.descriptor(space) else {
                continue;
            };
            let Some(crate::syntax::ByteSize(cap)) = decl.capacity else {
                continue;
            };
            // A granule'd sub-space consumes the *granule-rounded* size per tile (SS3): a 1-byte
            // tile still occupies a whole granule, so the true working set rounds each tile up.
            let granule = decl.granule.as_ref().map(|g| g.0).filter(|g| *g > 0);
            // The peak, not the sum. Walk the placements in program order; at each one the
            // tiles still occupying the space are those placed earlier in a block that is still
            // open -- that is, whose scope chain is a prefix of this one. A tile in a sibling
            // block was released when that block closed and is not among them.
            //
            // A function whose placements all sit in one block is unaffected: every chain is a
            // prefix of every other, so the peak is the sum, which is what those tiles really do.
            let mut placed: Vec<(u64, &Vec<u32>, usize)> = tiles
                .values()
                .map(|p| {
                    (
                        crate::hir::memory::granule_round(p.bytes, granule),
                        &p.scope,
                        p.order,
                    )
                })
                .collect();
            placed.sort_by_key(|&(_, _, order)| order);
            let mut total: u64 = 0;
            let mut peak_tiles = 0usize;
            for (i, (_, scope, _)) in placed.iter().enumerate() {
                let live: u64 = placed[..=i]
                    .iter()
                    .filter(|(_, earlier, _)| {
                        earlier.len() <= scope.len() && scope.starts_with(earlier)
                    })
                    .map(|&(b, _, _)| b)
                    .sum();
                if live > total {
                    total = live;
                    peak_tiles = placed[..=i]
                        .iter()
                        .filter(|(_, earlier, _)| {
                            earlier.len() <= scope.len() && scope.starts_with(earlier)
                        })
                        .count();
                }
            }
            let tile_count = peak_tiles.max(1);
            // Record the working set whether or not it violates. An *admitted* program emits no
            // capacity diagnostic, so without this its resident set would be absent from the JSON
            // record -- and the resident total is what a downstream consumer needs to compute the
            // utilization an engine must be given (#285). A verdict alone does not carry it.
            self.traffic.resident_sets.push(crate::report::ResidentSet {
                space: space.clone(),
                total_bytes: total,
                capacity_bytes: cap,
                tiles: tile_count,
                overcommit: decl.overcommit,
            });
            // The cumulative *diagnostic* stays gated on >1 tile: a lone oversized tile is
            // E6009's job, and reporting it twice would double-count in the record.
            if total > cap && tile_count > 1 {
                violations.push((
                    space.clone(),
                    total,
                    cap,
                    tile_count,
                    decl.overcommit,
                    granule,
                ));
            }
        }
        for (space, total, cap, count, overcommit, granule) in violations {
            let rounded_note = match granule {
                Some(g) => format!(" (each rounded up to the {g}-byte granule)"),
                None => String::new(),
            };
            let msg = format!(
                "the working set placed in memory space '{}' ({} tiles){} sums to {} bytes, over \
                 its {} byte capacity",
                space.name(),
                count,
                rounded_note,
                total,
                cap
            );
            // The same working-set numbers in machine-readable form, on whichever of the two
            // codes this space's `overcommit` selects (#282).
            let facts = crate::diagnostic::DiagnosticFacts::Capacity {
                space: space.name(),
                required_bytes: total,
                available_bytes: cap,
                tiles: Some(count),
            };
            if overcommit {
                self.errors
                    .warn(
                        crate::diagnostic::DiagnosticCode::W1028,
                        format!("{msg}; allowed because '{}' is `overcommit`", space.name()),
                        None,
                    )
                    .facts = Some(facts);
            } else {
                self.errors
                    .error_with_code(
                        crate::diagnostic::DiagnosticCode::E6010,
                        format!("{msg}; place fewer/smaller tiles or declare it `overcommit`"),
                        None,
                    )
                    .facts = Some(facts);
            }
        }
    }

    /// Whether a memory space is declared `managed: cached` (hardware-coherent), so an implicit
    /// cross-space use of a value there — or a relaxed transfer into it — is safe. Undeclared
    /// spaces are treated as `explicit` (the strict default), preserving the pre-M5 behavior.
    pub(crate) fn space_is_cached(&self, space: &MemorySpace) -> bool {
        self.env.memories.values().any(|d| {
            d.managed == crate::syntax::Management::Cached
                && MemorySpace::from_name(d.name.as_ref()) == *space
        })
    }

    /// The memory space a value of type `ty` actually lives in: a `Ref`'s space, a `Pinned`'s
    /// topology default space, else the space of `fallback_top` (the binding's topology). Used
    /// to find the space whose `managed` policy governs an implicit cross-space use.
    pub(crate) fn value_memory_space(&self, ty: &Type, fallback_top: &Topology) -> MemorySpace {
        // A placed tensor names its space outright, which is the whole point of
        // carrying a placement -- read it before falling back to deriving one from
        // a topology. Without this arm a transferred tensor reported the binding's
        // space instead of its own, and the visibility diagnostic that depends on
        // this stopped firing.
        if let Some(p) = ty.placement() {
            return p.space.clone();
        }
        match ty {
            Type::Ref(_, mem) => mem.clone(),
            Type::Pinned(_, topo) if !matches!(topo, Topology::Current) => {
                self.transfer_cost_graph.default_memory_for(topo)
            }
            _ if !matches!(fallback_top, Topology::Current) => {
                self.transfer_cost_graph.default_memory_for(fallback_top)
            }
            _ => MemorySpace::CPUDRAM,
        }
    }

    /// Walk a declared type for a `Ref`/`Pinned` tensor bound to a capacity-bearing space and
    /// check it fits. Used on `let` annotations.
    ///
    /// A `let` annotation is the one placement position not reachable from the declaration tables,
    /// which is why the check that a placement names a real place is also called from here.
    pub(crate) fn check_type_placement(&mut self, ty: &Type, context: &str, span: &Span) {
        self.report_unheld_placements(ty, context);
        // A placed tensor is checked against the space it names, the same as the
        // two wrapper spellings below.
        if let Some(p) = ty.placement() {
            if let Some((e, d)) = Self::tensor_of(ty) {
                let space = p.space.clone();
                self.check_capacity(e, d, &space, context, span);
                self.check_element_type(e, &space, context);
            }
            return;
        }
        match ty {
            Type::Ref(inner, mem) => {
                if let Some((e, d)) = Self::tensor_of(inner) {
                    self.check_capacity(e, d, mem, context, span);
                    self.check_element_type(e, mem, context);
                }
            }
            Type::Pinned(inner, top) if !matches!(top, Topology::Current) => {
                if let Some((e, d)) = Self::tensor_of(inner) {
                    let space = self.transfer_cost_graph.default_memory_for(top);
                    self.check_capacity(e, d, &space, context, span);
                    self.check_element_type(e, &space, context);
                }
            }
            _ => {}
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_seam_hop(
        &mut self,
        src: &MemorySpace,
        dst: &MemorySpace,
        relaxed: bool,
        buffer: &str,
        known_val: Option<u64>,
        span: Span,
    ) {
        use crate::hir::seam::{AbsState, Cell, Solver, Transfer, Verdict};

        // M5: a relaxed transfer into a `managed: cached` space is safe — hardware coherence
        // keeps the buffer visible, so there is no seam obligation to discharge.
        if relaxed && self.space_is_cached(dst) {
            return;
        }

        // Reached state at the producer side: the transferred buffer is established.
        // If the producer value is statically known we pin it (a value contract);
        // otherwise the concrete value is immaterial to the coarser visibility obligation.
        let reached = AbsState {
            cells: vec![(buffer.to_string(), Cell::constant(known_val.unwrap_or(1)))],
        };
        let transfer = if relaxed {
            // Relaxed escape hatch: the published buffer loses its visibility guarantee.
            Transfer::Relaxed {
                published: vec![buffer.to_string()],
            }
        } else {
            Transfer::Sync
        };
        let consumed = [buffer.to_string()];

        // Lazily spawn the persistent solver on the first seam (one-time cost), then
        // reuse it for every seam so the per-seam timing is solving, not process startup.
        if self.seam.solver.is_none() {
            let init = std::time::Instant::now();
            self.seam.solver = Some(Solver::new());
            self.seam.init_time += init.elapsed();
        }
        let solver = self.seam.solver.as_mut().unwrap();

        let start = std::time::Instant::now();
        // A buffer with a statically-known value gets the stronger value contract (pin
        // the value); an opaque buffer falls back to the visibility obligation.
        let verdict = match known_val {
            Some(v) => solver.check_seam_value(&reached, &transfer, buffer, v),
            None => solver.check_seam_buffers(&reached, &transfer, &consumed),
        };
        self.seam.check_time += start.elapsed();
        self.seam.checks += 1;

        match verdict {
            Ok(Verdict::Accept) => {}
            Ok(Verdict::Reject { counterexample }) => {
                if !self.speculating {
                    self.errors
                        .error_with_code(
                            crate::diagnostic::DiagnosticCode::E6004,
                            format!(
                                "relaxed transfer of '{}' across the {:?} -> {:?} seam violates \
                                 the boundary contract{}: the buffer carries no synchronizing \
                                 release, so a consumer may read it stale",
                                buffer,
                                src,
                                dst,
                                match known_val {
                                    Some(v) => format!(" ('{buffer}' == {v})"),
                                    None => String::new(),
                                }
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        )
                        .notes
                        .push(crate::diagnostic::Note {
                            message: format!(
                                "seam obligation is satisfiable; z3 counterexample: {}",
                                counterexample
                            )
                            .into(),
                            span: None,
                        });
                }
            }
            Err(e) => {
                // An obligation that could not be discharged is not a discharged one. This site
                // used to warn and carry on, which meant a machine with no solver *accepted* the
                // programs a machine with one rejects -- the same fail-open Vx#374 closed at the
                // declaration site, left open at the transfer site that `--verify-seams` drives.
                // E6004 was also the wrong code to say it with: that code means the contract was
                // shown to be violated, and nothing here was shown at all.
                if !self.speculating {
                    let msg = format!(
                        "the seam obligation for '{buffer}' across the {} -> {} hop was NOT \
                         verified: {e}",
                        src.name(),
                        dst.name()
                    );
                    if crate::hir::solver::unverified_allowed() {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1031,
                            msg,
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        );
                    } else {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6024,
                            msg,
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                        );
                    }
                }
            }
        }
    }

    /// Resolve where this transfer starts, where it ends, and how the hardware gets from one to
    /// the other. Reports the edge's own diagnostics on the way: capacity, and a host memory
    /// nothing declared. `None` when no route exists at all.
    fn resolve_transfer_edge(&mut self, t: &mut TransferExpr) -> Option<ResolvedEdge> {
        let prev = self.allow_cross_topology;
        self.allow_cross_topology = true;
        let inner_ty = self.check_expr_type_flag(&mut t.expr, false);
        self.allow_cross_topology = prev;

        // Extract source memory space, preferring exact space from an inner transfer if present
        let source_mem = if let Expr::Transfer(inner_t) = &*t.expr {
            inner_t.space.clone()
        } else if let Some(space) = inner_ty.placement().map(|p| p.space.clone()) {
            // A placed tensor names the space it is in. While `transfer` produced
            // `Pinned`, a transferred value matched an arm below and the fallback saw
            // only genuinely host-resident values; with the placement unread it called
            // this one host memory and computed the edge from CPU_DRAM, inserting a
            // redundant staging hop into the region.
            space
        } else {
            match &inner_ty {
                Type::Ref(_, mem) => mem.clone(),
                // A value pinned on a *declared* custom topology lives in the memory its
                // descriptor names (`Topology RubinCPX { memory: Memory::GDDR7 }`) -- resolving
                // through the descriptor, or the declared GDDR7->HBM4 edge is missed and the
                // handoff reports "no hardware path" (#253). Only an undeclared custom topology
                // falls back to the like-named space (`Memory::Foo` <-> `Topology::Foo`), the
                // sub-space placement case (SMEM/TMEM), which has no descriptor.
                Type::Pinned(_, Topology::Custom(name)) => {
                    let top = Topology::Custom(name.clone());
                    if self.transfer_cost_graph.descriptor(&top.kind()).is_some() {
                        self.transfer_cost_graph.default_memory_for(&top)
                    } else {
                        MemorySpace::from_name(name.as_ref())
                    }
                }
                Type::Pinned(_, top) => self.transfer_cost_graph.default_memory_for(top),
                _ => MemorySpace::CPUDRAM,
            }
        };
        let target_mem = t.space.clone();

        // Capacity: a statically-shaped tensor transferred into a declared space must fit.
        if let Some((e, d)) = Self::tensor_of(&inner_ty) {
            // `TransferExpr` carries no position of its own, so the buffer it moves supplies
            // one: distinct per `transfer(..)` written in the source, and the same on a
            // second visit to the same node.
            let placement_span = Self::buffer_span(&t.expr).unwrap_or(t.span);
            self.check_capacity(e, d, &target_mem, "transferred tensor", &placement_span);
            self.check_element_type(e, &target_mem, "transferred tensor");
        }

        // Bandwidth-derived roofline cost (bytes / bandwidth along the hierarchy). When the
        // memory declarations make it computable it becomes the transfer's cost (emitted on
        // `vx.transfer`); otherwise the fixed cost-graph value is kept. This does not touch
        // reachability/seam, which still use `transfer_path` below.
        // Two ways a hop's cost can be known, and exactly one applies per edge (E6013): a
        // bandwidth declared on the *link* (`transfer A -> B : 64 GB/s`), or the roofline
        // derived from the endpoints' own bandwidths along the containment tree. The link rate
        // is checked first because it exists precisely where the containment walk cannot help
        // -- a host<->device hop whose endpoints do not nest.
        // How many bytes this transfer moves. Recorded, not just consumed: a harvested cost is
        // uninterpretable without the size it is a cost *of*, and S4's freeze artifact is this
        // record (vx-review#12).
        let moved_bytes: Option<u64> = Self::tensor_of(&inner_ty)
            .and_then(|(e, d)| crate::hir::memory::static_tensor_bytes(e, d));

        // Which of the two cost sources applied. They are mutually exclusive by construction
        // (E6013), so this names the one that fired rather than a precedence winner.
        let link_bw = self.transfer_cost_graph.link_rate(&source_mem, &target_mem);
        let derived: Option<crate::hir::memory::DerivedCost> = moved_bytes.and_then(|bytes| {
            link_bw
                .and_then(|bw| {
                    crate::hir::memory::hop_cost(bytes, bw)
                        .map(|value| crate::hir::memory::DerivedCost { value, per: bw.per })
                })
                .or_else(|| {
                    crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied())
                        .derived_transfer_cost(&source_mem, &target_mem, bytes)
                })
        });
        let derived_cost = derived.map(|d| d.value);
        let derived_unit = derived.map(|d| d.per);
        let cost_source = derived.map(|_| {
            if link_bw.is_some() {
                crate::report::CostSource::LinkRate
            } else {
                crate::report::CostSource::Containment
            }
        });

        let mut path_result = self
            .transfer_cost_graph
            .transfer_path(&source_mem, &target_mem);

        // Sub-spaces have no transfer edges of their own (SMEM/TMEM); a transfer into or
        // between them is reachable via their enclosing device spaces. If there is no direct
        // path, resolve each endpoint to itself-or-a-`within:`-ancestor and take the first
        // reachable pairing -- a sibling->sibling move meets at their common parent. See
        // subspace_scheduling.md §2.3.
        if path_result.is_none() {
            let hierarchy =
                crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
            let sources: Vec<MemorySpace> = std::iter::once(source_mem.clone())
                .chain(hierarchy.ancestors(&source_mem))
                .collect();
            let targets: Vec<MemorySpace> = std::iter::once(target_mem.clone())
                .chain(hierarchy.ancestors(&target_mem))
                .collect();
            'outer: for s in &sources {
                for t in &targets {
                    if let Some(p) = self.transfer_cost_graph.transfer_path(s, t) {
                        path_result = Some(p);
                        break 'outer;
                    }
                }
            }
        }

        if path_result.is_none() {
            if !self.speculating {
                // E6002 is this diagnostic's code, and it used to carry none: the
                // code was declared and emitted nowhere, so the compiler could not
                // produce it while the behaviour it names was already refused here.
                // Spaces print by name rather than by `{:?}`, which rendered a
                // declared one as `Custom("Island")`.
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6002,
                    format!(
                        "Cannot transfer from {} to {}: no hardware path exists",
                        source_mem.name(),
                        target_mem.name()
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&t.span)),
                );
            }
            return None;
        }

        // A seam with the host on one end is reasoning about a host, so
        // there had better be one. `--machine` describes an accelerator and
        // says nothing about the machine it hangs off, and `Memory::CPU_DRAM`
        // otherwise arrives from the built-ins with no declaration at all --
        // so this edge was being costed against a source nobody described.
        //
        // Only when a machine model is present: without one this is an
        // ordinary native build, not a fleet-level question, and demanding
        // a host file would be noise.
        if !self.speculating {
            let touches_host =
                source_mem == MemorySpace::CPUDRAM || target_mem == MemorySpace::CPUDRAM;
            let host_declared = self
                .env
                .memories
                .values()
                .any(|m| m.name.as_ref() == "CPU_DRAM");
            let machine_declared = self
                .env
                .memories
                .values()
                .any(|m| matches!(m.scope, Some(crate::syntax::Scope::Device)));
            if touches_host && machine_declared && !host_declared {
                self.errors
                    .error_with_code(
                        crate::diagnostic::DiagnosticCode::E6014,
                        format!(
                            "this program stages through host memory, but no host was declared: \
                             the transfer {:?} -> {:?} has an end nothing describes",
                            source_mem, target_mem
                        ),
                        None,
                    )
                    .notes
                    .push(crate::diagnostic::Note {
                        message: "pass --host <file> to name the host, or --host default for \
                                  the machine compiling this. `--machine` describes the \
                                  accelerator only."
                            .into(),
                        span: None,
                    });
            }
        }
        let (graph_cost, path) = path_result.unwrap();
        Some(ResolvedEdge {
            inner_ty,
            source_mem,
            target_mem,
            path,
            graph_cost,
            moved_bytes,
            derived_cost,
            derived_unit,
            cost_source,
        })
    }

    /// The tile shape a type declares, when every extent is a literal. `None` for anything
    /// whose shape is only known at run time, which is what makes a lowering unemittable.
    fn static_shape(ty: &Type) -> Option<Vec<u64>> {
        let (_, dims, _) = Self::as_tensor_operand(ty)?;
        let mut out = Vec::new();
        for d in dims {
            let crate::syntax::Expr::Number(n) = d else {
                return None;
            };
            out.push(n.value.as_ref().parse::<u64>().ok()?);
        }
        Some(out)
    }

    /// The tile a lowering declares must be the tile actually being transferred -- same shape,
    /// same element type. A lowering is chosen by edge, so nothing else ties the two together,
    /// and the primitives read their extents from the declaration: a smaller declared tile copies
    /// part of the site's tile and leaves the rest uninitialised, a larger one stores past the end.
    ///
    /// Records the lowering on the site when they agree, which is what makes codegen inline it.
    fn record_emittable_lowering(
        &mut self,
        t: &mut TransferExpr,
        edge: &ResolvedEdge,
        decl_src: &Type,
        decl_dst: &Type,
        topology: &str,
    ) {
        let source_mem = edge.source_mem.clone();
        let target_mem = edge.target_mem.clone();
        let inner_ty = &edge.inner_ty;
        // The declared tile shape must be the shape actually being
        // transferred. A lowering is chosen per EDGE, so nothing
        // else ties the two together -- and the primitives read
        // their extents from the declaration. Declaring a smaller
        // tile than the site's copies part of it and reads the rest
        // back uninitialised; declaring a larger one stores past
        // the end. Both were reproduced, the second as a SIGSEGV.
        let site_dims = Self::as_tensor_operand(inner_ty)
            .map(|(_, dims, _)| dims.clone())
            .map(|dims| {
                dims.iter()
                    .map(|d| match d {
                        crate::syntax::Expr::Number(n) => n.value.as_ref().parse::<u64>().ok(),
                        _ => None,
                    })
                    .collect::<Option<Vec<u64>>>()
            });
        // BOTH tiles, not just the source. The destination's declared
        // shape is what `raw::extent(dst)` folds to and what the
        // primitives delinearize against, so a lowering declaring an
        // [8,8] destination for a [2,2] site emitted 64 stores into a
        // 4-element buffer -- admitted, with `written_bytes: 256`
        // published beside `bytes: 16`.
        // Both are statically shaped: that is one of the conditions that made this lowering
        // emittable in the first place.
        let src_shape = Self::static_shape(decl_src).expect("an emittable source tile has a shape");
        let dst_shape =
            Self::static_shape(decl_dst).expect("an emittable destination tile has a shape");
        let declared = (src_shape == dst_shape).then(|| src_shape.clone());
        if src_shape != dst_shape {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6023,
                format!(
                    "impl transfer {} -> {} declares a {:?} source and a {:?} destination; a \
                     transfer moves a tile, so both sides are the same shape",
                    source_mem.name(),
                    target_mem.name(),
                    src_shape,
                    dst_shape
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&t.span)),
            );
        }
        // Element types too: edge selection does not look at them,
        // and a mismatch reaches MLIR as a bare verifier failure
        // ("result type matches element type of 'memref'") on a
        // program the checker accepted.
        let site_elem = Self::as_tensor_operand(inner_ty).map(|(e, _, _)| e.clone());
        let decl_elem = Self::as_tensor_operand(decl_src).map(|(e, _, _)| e.clone());
        if let (Some(se), Some(de)) = (&site_elem, &decl_elem) {
            if se != de {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6023,
                    format!(
                        "impl transfer {} -> {} declares {:?} tiles \
                             but this transfer moves a {:?} tile; a \
                             lowering is chosen by edge, so its declared \
                             element type must be the one it is given",
                        source_mem.name(),
                        target_mem.name(),
                        de,
                        se
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&t.span)),
                );
            }
        }
        match (site_dims, declared) {
            (Some(Some(site)), Some(decl)) if site != decl => {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6023,
                    format!(
                        "impl transfer {} -> {} declares {:?} tiles \
                             but this transfer moves a {:?} tile; a \
                             lowering is chosen by edge, so its declared \
                             shape must be the shape it is given",
                        source_mem.name(),
                        target_mem.name(),
                        decl,
                        site
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&t.span)),
                );
            }
            // The site's tile has no statically known shape, so nothing
            // relates it to the lowering's declared one. Emission would
            // splice the DECLARED extents and strides against a runtime
            // ?x? buffer: a [2,2] lowering on a 3x3 tile copies 4 of 9
            // elements and scatters them to (i/2)*3 + i%2 instead of i.
            // E6023 exists to prevent exactly that mismatch; falling
            // through here let the unknown-shape case past it.
            //
            // Refusing the LOWERING, not the program: the builtin copy
            // moves a dynamically shaped tile correctly, so declining to
            // emit is a silent, correct fallback rather than an error.
            (Some(None), Some(decl)) => {
                if !self.speculating {
                    self.errors.push_warning(format!(
                        "impl transfer {} -> {} declares {:?} tiles but \
                             this transfer's tile has no statically known \
                             shape, so nothing relates the two; the builtin \
                             copy is used instead. Emitting the body would \
                             splice the declared extents and strides against \
                             a runtime-shaped buffer",
                        source_mem.name(),
                        target_mem.name(),
                        decl
                    ));
                }
            }
            _ => {
                t.lowering = Some((source_mem.clone(), target_mem.clone(), topology.to_string()));
            }
        }
    }

    /// Pick the user-written lowering that runs on this edge, if one both exists and can be
    /// emitted, and record it on the site. Codegen then inlines that body in place of the
    /// builtin copy; the builtin, with its barrier, stays the fallback for everything else.
    ///
    /// Emittable means exactly one method whose two parameters are the source and destination
    /// tiles. The body's own obligations -- E6015, E6019, E6021 and E6022 -- were discharged
    /// when the impl block itself was checked.
    ///
    /// Only single hops reach here. Each hop of a staged route re-enters the check and is
    /// matched on its own.
    fn select_transfer_lowering(&mut self, t: &mut TransferExpr, edge: &ResolvedEdge) {
        let source_mem = edge.source_mem.clone();
        let target_mem = edge.target_mem.clone();
        // Which machine's lowering runs here. Candidates are the lowerings for this
        // edge; the topology in force at the site picks among them.
        //
        // If the active topology declares this edge itself, its own declarations are
        // the whole answer: its lowering if it wrote one, the builtin if it did not.
        // A machine that declined to implement its edge did not delegate the choice
        // to whoever else implemented a like-named edge -- a peer's body may lean on
        // capabilities (a copy engine) the active machine never declared, so
        // borrowing it is not a default, it is a different machine's code. The
        // review reproduced exactly that: one machine's double-read body silently
        // counted (and would have been inlined) for a spawn on the machine next to
        // it (Vx#353).
        //
        // The fallback below exists for sites on NO machine that owns this edge --
        // a host edge is moved from the host, so `transfer(a, Memory::GPU_HBM)` in
        // `main` runs with the CPU active while implementing an edge that belongs
        // to the device. There a single candidate is unambiguous and is taken; only
        // several candidates with nothing to pick among them is refused.
        let candidates: Vec<&crate::syntax::TransferImplDecl> = self
            .env
            .transfer_impls
            .iter()
            .filter(|li| li.from == source_mem && li.to == target_mem)
            .copied()
            .collect();
        let active = self.active_topology.display_name();
        let active_declares_edge = self
            .env
            .topologies
            .values()
            .find(|d| d.name.as_ref() == active)
            .map(|d| {
                d.descriptor
                    .transfers
                    .iter()
                    .any(|e| e.from == source_mem && e.to == target_mem)
            })
            .unwrap_or(false);
        let chosen = candidates
            .iter()
            .find(|li| li.topology.display_name() == active)
            .or(if !active_declares_edge && candidates.len() == 1 {
                candidates.first()
            } else {
                None
            })
            .copied();
        if chosen.is_none() && !active_declares_edge && candidates.len() > 1 {
            let names: Vec<String> = candidates
                .iter()
                .map(|li| li.topology.display_name().to_string())
                .collect();
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6015,
                format!(
                    "the edge {} -> {} is implemented by {} ({}), and none of them is \
                     the topology in force here ({}); which lowering should run is \
                     ambiguous -- perform this transfer inside a `spawn on` for the \
                     machine that owns it",
                    source_mem.name(),
                    target_mem.name(),
                    names.len(),
                    names.join(", "),
                    active
                ),
                None,
            );
        }
        if let Some(li) = chosen {
            // EXACTLY two parameters, both tiles. A third parameter has
            // nothing to bind to at the site: it would resolve against
            // whatever the caller happens to have under that name and let
            // the lowering write into a buffer it never named (constraint
            // C9), or fail to resolve at all. Reproduced both ways before
            // this said `==`.
            // The destination must be SM-scoped. That is the one edge kind
            // emission handles: the site becomes a shared-memory allocation
            // the body fills in place. On any other edge the builtin still
            // runs -- the plugin's device copy, or the host alloc+copy --
            // and the body would run *beside* it: the same bytes moved
            // twice, and on a real GPU a host store through a device
            // pointer. Reproduced on three edges before this gate existed.
            let dst_is_sm = self
                .env
                .memories
                .values()
                .find(|d| crate::syntax::MemorySpace::from_name(d.name.as_ref()) == target_mem)
                .and_then(|d| d.scope)
                == Some(crate::syntax::Scope::Sm);
            // The destination is the one the body writes, so it is the one
            // declared `&mut`. Positional binding alone let a lowering
            // written `fn move_tile(dst, src)` read the empty tile and
            // clobber the source, silently.
            let ordered = li.methods.len() == 1
                && li.methods[0].params.len() == 2
                && !matches!(li.methods[0].params[0].1, Type::Borrow { is_mut: true, .. })
                && matches!(li.methods[0].params[1].1, Type::Borrow { is_mut: true, .. });
            // The body must be written against the `raw::` primitives. The
            // whole-body contract -- the trailing barrier (C3), the early-return
            // refusal, the async discipline -- is SKIPPED for a body with no
            // `raw::` call ("A1-era body", raw.rs), because such bodies predate
            // the primitives and are carried, not emitted. Emitting one anyway
            // took both halves of that bargain: the site is marked
            // `user_lowered`, so VxLowering skips the builtin copy AND its C3
            // barrier on the stated grounds that "its own trailing
            // raw::barrier() is the synchronization (checked, E6021)" -- while
            // E6021 never ran. Probed: a body filling `dst[i][d] = src[i][d]`
            // emitted a synchronizing sm transfer with no barrier from either
            // source, and a nested `return 7` in such a body spliced
            // `vx.return` into the middle of the kernel region.
            //
            // Carried-but-not-emitted is the A1-era contract; this restores it.
            let uses_raw = li
                .methods
                .iter()
                .any(|f| crate::hir::check::raw::body_uses_raw(&f.body));
            let emittable = li.methods.len() == 1
                && li.methods[0].params.len() == 2
                && li.methods[0]
                    .params
                    .iter()
                    .all(|(_, ty)| Self::static_shape(ty).is_some())
                && ordered
                && uses_raw
                && dst_is_sm;
            if !emittable && !self.speculating {
                self.errors.push_warning(format!(
                    "impl transfer {} -> {} exists but is not emitted here: \
                     a lowering needs exactly one method taking exactly two \
                     statically-shaped tiles -- the source, then the \
                     destination held by `&mut` -- a body written against the \
                     `raw::` primitives, and a destination space \
                     declared `scope: sm`{}. The builtin copy is used instead",
                    source_mem.name(),
                    target_mem.name(),
                    if dst_is_sm {
                        ""
                    } else {
                        " (this one is not sm-scoped)"
                    }
                ));
            } else if emittable {
                // Copied out of the declaration first: it is borrowed from the environment, and
                // the checks below report against the checker.
                let decl_src = li.methods[0].params[0].1.clone();
                let decl_dst = li.methods[0].params[1].1.clone();
                let topology = li.topology.display_name().to_string();
                self.record_emittable_lowering(t, edge, &decl_src, &decl_dst, &topology);
            }
        }
    }

    /// Record the resolved hop for the accept side of `--diagnostics-json`: the route, what it
    /// costs, and how much traffic it moves. Only single hops are recorded -- a staged route is
    /// rewritten into a chain that each record themselves, so recording it whole would count the
    /// same movement twice.
    fn record_staging_route(&mut self, t: &TransferExpr, edge: &ResolvedEdge) {
        if self.speculating {
            return;
        }
        let source_mem = edge.source_mem.clone();
        let target_mem = edge.target_mem.clone();
        let inner_ty = &edge.inner_ty;
        let moved_bytes = edge.moved_bytes;
        let derived_cost = edge.derived_cost;
        let derived_unit = edge.derived_unit;
        let cost_source = edge.cost_source;
        // The figure the machine file *declared* for this edge, not the weight the
        // router happened to use. Since an edge may now decline to declare a cost
        // (leaving it to the endpoints' bandwidths), its routing weight is 1 — and
        // reporting that 1 here would put a number nobody wrote into the record a
        // measurement campaign harvests.
        let declared_cost = self
            .transfer_cost_graph
            .declared_edge_cost(&source_mem, &target_mem);
        // Derived traffic (#353 A4). The builtin copy reads the whole tile
        // from the source space and writes it into the target: one pass, no
        // amplification, exact by construction. A hop whose body a user
        // supplied is counted from that body instead (T2), and a hop whose
        // size is not statically known is not counted at all -- an
        // uncountable movement is reported as uncountable, never as zero.
        let (traffic, traffic_absent_reason) = match (moved_bytes, &t.lowering) {
            (Some(b), None) => (
                Some(crate::report::Traffic {
                    per_space: vec![
                        crate::report::SpaceTraffic {
                            space: source_mem.clone(),
                            read_bytes: b,
                            written_bytes: 0,
                        },
                        crate::report::SpaceTraffic {
                            space: target_mem.clone(),
                            read_bytes: 0,
                            written_bytes: b,
                        },
                    ],
                    source: crate::report::TrafficSource::BuiltinCopy,
                    exact: true,
                }),
                None,
            ),
            (None, _) => (
                None,
                Some("the transferred tile has no statically known size".to_string()),
            ),
            // A user lowering's traffic is counted from the body that will
            // actually run, not from the tile size the builtin would have
            // moved: that difference is the whole point -- a body that reads
            // the source twice reports twice the reads, with nobody declaring
            // anything (#353 A4).
            (Some(_), Some((_, _, topo))) => {
                // Look up the lowering sema CHOSE, topology included. Matching on
                // the edge alone would count the body of whichever machine's
                // lowering happened to be first in the list.
                let li = self.env.transfer_impls.iter().find(|li| {
                    li.from == source_mem
                        && li.to == target_mem
                        && li.topology.display_name() == *topo
                });
                match li {
                    Some(li) => {
                        // Sub-byte elements are refused, not rounded. Rounding
                        // i4 up to a byte made a FAITHFUL copy of a 2x2 i4
                        // tile report 4 bytes against a 2-byte tile -- a ratio
                        // of 2.0, numerically identical to the amplification
                        // factor that is this stage's evidence that a plan is
                        // wasteful. A figure indistinguishable from the thing
                        // it exists to detect is worse than no figure.
                        let elem_bytes = Self::as_tensor_operand(inner_ty)
                            .and_then(|(e, _, _)| crate::hir::memory::element_bits(e))
                            .filter(|bits| bits % 8 == 0)
                            .map(|bits| bits / 8);
                        match elem_bytes {
                            None => (
                                None,
                                Some(
                                    "the tile's element type is not a whole \
                                     number of bytes, so its movement cannot \
                                     be counted in bytes"
                                        .to_string(),
                                ),
                            ),
                            Some(elem_bytes) => {
                                let m = &li.methods[0];
                                match self.derive_lowering_traffic(
                                    &m.body,
                                    elem_bytes,
                                    &m.params,
                                    &source_mem,
                                    &target_mem,
                                ) {
                                    Ok(t) => (Some(t), None),
                                    Err(why) => (None, Some(why)),
                                }
                            }
                        }
                    }
                    None => (
                        None,
                        Some("the matched lowering could not be found".to_string()),
                    ),
                }
            }
        };
        self.traffic
            .staging_routes
            .push(crate::report::StagingRoute {
                path: edge.path.clone(),
                traffic,
                traffic_absent_reason,
                edge_costs: vec![declared_cost],
                total_cost: edge.graph_cost,
                bytes: moved_bytes,
                cost_source,
                derived_cost,
                derived_unit,
                // Only a containment route composes anything -- a link rate is one leg.
                // Read from the destination, which is where the fill mechanism lives.
                composition: (cost_source == Some(crate::report::CostSource::Containment)).then(
                    || {
                        self.env
                            .memories
                            .values()
                            .find(|d| {
                                crate::syntax::MemorySpace::from_name(d.name.as_ref()) == target_mem
                            })
                            .map(|d| d.crossing)
                            .unwrap_or_default()
                    },
                ),
            });
    }

    /// Discharge this hop's local-completeness and soundness obligation, so the buffer crossing
    /// the seam is one the program has said enough about.
    fn discharge_seam_obligation(&mut self, t: &TransferExpr, edge: &ResolvedEdge, relaxed: bool) {
        let source_mem = edge.source_mem.clone();
        let target_mem = edge.target_mem.clone();
        let span = Self::buffer_span(&t.expr).unwrap_or(t.span);
        let buffer = Self::buffer_label(&t.expr);
        // Prefer the value the consumer asserts of the buffer this transfer
        // produces (looked up by the let-binding target, e.g. `local_a`), which
        // is the boundary contract's conclusion; fall back to the producer's
        // own statically-known constant, else the coarse visibility check.
        let asserted_val = self
            .current_assignment_target
            .as_ref()
            .and_then(|tgt| self.seam.contracts.get(tgt).copied());
        let known_val = asserted_val.or_else(|| self.const_value_of(&buffer));
        // A hop over a *declared* `relaxed` edge is relaxed too, not just the caller's
        // `*_relaxed` intrinsic escape hatch: route it through the same seam obligation so a
        // declared relaxed transfer gets the per-buffer E6004 at the use site, not only the
        // blunt declaration-time W1027 (P0-3). A relaxed hop anywhere in a staged multi-hop
        // route taints that hop, since each single hop re-enters this check.
        let hop_relaxed = self
            .transfer_cost_graph
            .is_relaxed_edge(&source_mem, &target_mem);
        self.run_seam_hop(
            &source_mem,
            &target_mem,
            relaxed || hop_relaxed,
            &buffer,
            known_val,
            span,
        );
    }

    /// Rewrite a staged transfer into a chain of single-hop transfers, one per edge on the route,
    /// then check the chain. Each hop re-enters `check_transfer_expr` and is costed, lowered and
    /// discharged on its own.
    fn stage_multi_hop(
        &mut self,
        expr: &mut Expr,
        edge: ResolvedEdge,
        relaxed: bool,
        consume: bool,
    ) -> Type {
        let ResolvedEdge {
            path, target_mem, ..
        } = edge;
        let Expr::Transfer(t) = std::mem::replace(
            expr,
            Expr::Number(NumberExpr::new("0".to_string(), None, Span::default())),
        ) else {
            unreachable!()
        };
        let mut current_expr = *t.expr;
        let path_len = path.len();
        for intermediate_space in path.into_iter().skip(1).take(path_len - 2) {
            current_expr = Expr::Transfer(TransferExpr {
                expr: Box::new(current_expr),
                space: intermediate_space,
                cost: None,
                lowering: None,
                span: t.span,
            });
        }
        *expr = Expr::Transfer(TransferExpr {
            expr: Box::new(current_expr),
            space: target_mem,
            cost: None,
            lowering: None,
            span: t.span,
        });
        // Propagate the relaxed marker so each rewritten single-hop transfer is checked with the
        // right transfer function.
        self.seam.pending_relaxed = relaxed;
        self.check_transfer_expr(expr, consume)
    }

    pub(crate) fn check_transfer_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        // Whether this transfer is a relaxed escape hatch (set by the caller, e.g. the
        // `to_device_relaxed` method arm). Consumed here so it does not leak to siblings.
        let relaxed = std::mem::take(&mut self.seam.pending_relaxed);

        let Expr::Transfer(t) = expr else {
            unreachable!()
        };
        // No route means no transfer, so the type is poisoned rather than faked as a plausible
        // f32 tensor that later stages would reason about.
        let Some(edge) = self.resolve_transfer_edge(t) else {
            return Type::Unknown;
        };

        if edge.path.len() > 2 {
            return self.stage_multi_hop(expr, edge, relaxed, consume);
        }

        // A single hop is the unit everything below is stated for: which lowering runs on it,
        // what it costs, and the obligation crossing it.
        let single_hop = edge.path.len() == 2;
        if single_hop {
            self.select_transfer_lowering(t, &edge);
        }
        // Record the bandwidth-derived roofline cost when the hierarchy provides one; otherwise
        // leave it unset, so a transfer with no bandwidth behind it emits no `cost` attribute.
        t.cost = edge.derived_cost;
        self.record_staging_route(t, &edge);
        if single_hop && self.seam.verify {
            self.discharge_seam_obligation(t, &edge, relaxed);
        }

        self.transfer_result_type(expr, edge.inner_ty, &edge.target_mem, consume)
    }

    /// Which topology a value living in `space` is pinned on. Every memory space maps to
    /// one, so this is total.
    fn pinned_topology_for(space: &MemorySpace) -> Topology {
        match space {
            MemorySpace::NPUHBM => Topology::NPU(Box::new(Expr::Number(NumberExpr {
                value: "0".into(),
                ty: Some(ElementType::I32),
                span: Span::default(),
            }))),
            MemorySpace::LocalSRAM => Topology::AccCore(Box::new(Expr::Number(NumberExpr {
                value: "0".into(),
                ty: Some(ElementType::I32),
                span: Span::default(),
            }))),
            MemorySpace::NicRam | MemorySpace::RemoteHbm => {
                Topology::NPU(Box::new(Expr::Number(NumberExpr {
                    value: "0".into(),
                    ty: Some(ElementType::I32),
                    span: Span::default(),
                })))
            }
            MemorySpace::GpuHbm => Topology::gpu(0),
            MemorySpace::CPUDRAM => Topology::CPU,
            // A value in a user-defined memory space is pinned on the like-named
            // custom topology (naming convention: Memory::Foo <-> Topology::Foo).
            MemorySpace::Custom(name) => Topology::Custom(name.clone()),
        }
    }

    /// The type a transfer produces: the value's own type, restated in the space it now
    /// lives in. Called once the edge has been resolved and its obligations discharged.
    fn transfer_result_type(
        &mut self,
        expr: &mut Expr,
        inner_ty: Type,
        target_mem: &MemorySpace,
        consume: bool,
    ) -> Type {
        match inner_ty {
            Type::Ref(base_ty, _) => Type::Ref(base_ty, target_mem.clone()),
            // A dynamic tensor re-homes exactly as a statically shaped one does; only the
            // capacity check differs, and that is what W1029 reports (Vx#399).
            //
            // The result is a *placed tensor*, the same type a declaration produces. It used to
            // be `Pinned<Tensor, Topology>`, which is the older wrapper shape from before Vx#429
            // folded placement into the tensor type -- so `transfer(x, Memory::NPU_HBM)` and
            // `Tensor<f16, [n, n], Memory::NPU_HBM>` were two different types for the same fact
            // and would not assign to one another. `Pinned` also carries only a topology, so the
            // space the transfer named was not recorded on the value at all.
            Type::Tensor(el, dims, _) => Type::Tensor(
                el,
                dims,
                Some(Placement::in_space(
                    target_mem.clone(),
                    Self::pinned_topology_for(target_mem),
                )),
            ),
            Type::DynTensor(el, _) => Type::DynTensor(
                el,
                Some(Placement::in_space(
                    target_mem.clone(),
                    Self::pinned_topology_for(target_mem),
                )),
            ),
            Type::Verified(_inner) => {
                if let Expr::Transfer(t) = expr {
                    let inner_pinned = self.check_expr_type_flag(&mut t.expr, consume);
                    Type::Verified(Box::new(inner_pinned))
                } else {
                    unreachable!()
                }
            }
            Type::Pinned(base, _) => Type::Pinned(base, Self::pinned_topology_for(target_mem)),
            _ => {
                if !self.speculating {
                    self.errors.push(format!(
                        "Cannot transfer non-reference type: {:?}",
                        inner_ty
                    ));
                }
                Type::Unknown
            }
        }
    }

    pub(crate) fn check_spawnon_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::SpawnOn(SpawnOnExpr {
                top,
                stmts,
                ret,
                span,
            }) => {
                let spawn_span = *span;
                let mut actual_top = top.clone();
                if actual_top == Topology::Current {
                    actual_top = self.active_topology.clone();
                }

                // Typo-safety for the open topology set: a user-defined topology with no
                // registered descriptor (not declared via `Topology <Name> { ... }` and not
                // registered by a plugin) is often a misspelled built-in. Warn; it still
                // compiles with host-like placement.
                if let Topology::Custom(name) = &actual_top {
                    if self
                        .transfer_cost_graph
                        .descriptor(&crate::syntax::TopologyKind::Custom(name.clone()))
                        .is_none()
                    {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1025,
                            format!(
                                "unknown topology '{}': not declared (`Topology {} {{ ... }}`) \
                                 or registered by a plugin; defaulting to host-like placement",
                                name, name
                            ),
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&spawn_span)),
                        );
                    }
                }

                // A non-constant device index resolves to no instance, so dispatch falls back to
                // index 0 -- every `GPU[i]` spawn lands on device 0. That used to be silent, which
                // in a fleet program reads as "8 devices" while meaning "device 0, eight times".
                // Vx models one representative device per declared kind (#284), so this is a
                // modelling boundary the program has crossed, not a compiler shortcoming.
                if let Some(idx) = crate::arch::non_constant_index(&actual_top) {
                    let what = match idx {
                        Expr::Identifier(id) => format!("the runtime value '{}'", id.name),
                        _ => "a runtime expression".to_string(),
                    };
                    self.errors.warn(
                        crate::diagnostic::DiagnosticCode::W1030,
                        format!(
                            "device index of '{}' is {}, not a compile-time constant; it cannot \
                             select a device instance and falls back to index 0, so every such \
                             spawn targets the same device",
                            actual_top.display_name(),
                            what
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&spawn_span)),
                    );
                }

                // Validate topology index expressions BEFORE switching context,
                // since the index (e.g., NPU[i]) refers to variables in the
                // outer scope.
                match &mut actual_top {
                    Topology::NPU(expr) | Topology::AccCore(expr) => {
                        let _ty = self.check_expr_type(expr);
                    }
                    Topology::Slice(_, start, end) => {
                        let _t1 = self.check_expr_type(start);
                        let _t2 = self.check_expr_type(end);
                    }
                    // No index expression to validate (Custom carries only a name).
                    Topology::GPU(e) => {
                        let _t = self.check_expr_type(e);
                    }
                    // No index expression to validate (Custom carries only a name).
                    Topology::CPU
                    | Topology::AMX
                    | Topology::ANE
                    | Topology::CpuAvx512
                    | Topology::CpuNeon
                    | Topology::Custom(_)
                    | Topology::Current => {}
                }

                let prev_top = self.active_topology.clone();
                let prev_mem = self.active_memory.clone();
                self.active_topology = actual_top.clone();
                self.active_memory = self.transfer_cost_graph.default_memory_for(&actual_top);

                *top = actual_top;

                // The region's tiles are released at its end now that the free follows the
                // tile's lifetime rather than the block's dominance.
                self.push_releasing_scope();

                // The placed values this region can see, captured BEFORE its body is checked
                // (#353 A4 T4). Checking mutates the scopes it reads -- a call that consumes a
                // placed tensor moves it out -- so a live lookup afterwards finds nothing and
                // the region silently reports no traffic. Probed and reproduced.
                let placed_outer = self.placed_names_snapshot();

                self.check_expr_block(stmts, consume);

                let mut ret_ty = Type::Struct("void".into(), None); // default void-like type
                let has_ret = ret.is_some();
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type_flag(r, consume);
                }

                // What this region moves, counted from its own accesses (#353 A4 T4). Here,
                // BEFORE the scope pops, because the body's free names -- the placed tensors
                // it was given -- resolve through the enclosing function's live scopes.
                //
                // Not while speculating: a probe re-checks the same node, and a second record
                // for one `spawn` would double it in the published set.
                if !self.speculating {
                    let (traffic, by_buffer, reason) =
                        match self.derive_spawn_traffic(stmts, ret.as_deref(), &placed_outer) {
                            Ok((t, b)) => (Some(t), b, None),
                            Err(why) => (None, Vec::new(), Some(why)),
                        };
                    let function = self.current_function.clone();
                    self.traffic
                        .spawn_regions
                        .push(crate::report::SpawnRegionTraffic {
                            function,
                            topology: self.active_topology.display_name(),
                            traffic,
                            by_buffer,
                            traffic_absent_reason: reason,
                        });
                }

                self.pop_scope();

                self.active_topology = prev_top;
                self.active_memory = prev_mem;

                // The result of a device kernel is located ON that device: return
                // `Pinned<ret_ty, d>` so reading it on another topology re-triggers the USE
                // rule (visibility or an explicit transfer) instead of being silently treated
                // as host-local. A void spawn (no result) has nothing to locate; a result the
                // body already produced as `Pinned<..>` is already located, so don't re-wrap.
                if has_ret && !matches!(ret_ty, Type::Pinned(..)) {
                    Type::Pinned(Box::new(ret_ty), (*top).clone())
                } else {
                    ret_ty
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
