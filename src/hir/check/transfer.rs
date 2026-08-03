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
//===----------------------------------------------------------------------===//

use super::super::*;

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
        for env in self.eval_env.iter().rev() {
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

    /// Check the coherence obligations of every user-defined topology registered for this
    /// compilation, emitting diagnostics for incoherent declarations.
    ///
    /// Graph-decidable obligations (`default_space` visible, memory reachable from the host)
    /// come from `arch::descriptor_coherence`. The consistency obligation is discharged
    /// through the seam engine: a declared `relaxed` edge is modeled as a relaxed transfer of
    /// a published payload and handed to `seam::check_seam_buffers`; a `Reject` (the buffer
    /// can be read stale) means the edge does not preserve visibility.
    pub fn check_topology_coherence(&mut self, declared: &[crate::arch::TopologyDecl]) {
        for decl in declared {
            let name = &decl.name;
            for issue in
                crate::arch::descriptor_coherence(&decl.descriptor, &self.transfer_cost_graph)
            {
                match issue {
                    crate::arch::CoherenceIssue::DefaultNotVisible => {
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E6005,
                            format!(
                                "topology '{name}' is incoherent: it cannot see its own default \
                                 memory space (not in its visibility set)"
                            ),
                            None,
                        );
                    }
                    crate::arch::CoherenceIssue::MemoryUnreachableFromHost => {
                        self.errors.warn(
                            crate::diagnostic::DiagnosticCode::W1026,
                            format!(
                                "topology '{name}': its memory is unreachable from the host; \
                                 declare a `transfer` edge so data can reach it"
                            ),
                            None,
                        );
                    }
                }
            }

            // Consistency obligation, discharged via the seam engine: a relaxed edge that
            // carries a payload does not preserve the buffer's visibility.
            for edge in &decl.descriptor.transfers {
                if edge.sync {
                    continue;
                }
                use crate::hir::seam::{AbsState, Cell, Transfer, Verdict};
                let reached = AbsState {
                    cells: vec![("payload".into(), Cell::constant(1))],
                };
                let transfer = Transfer::Relaxed {
                    published: vec!["payload".into()],
                };
                if self.seam_solver.is_none() {
                    self.seam_solver = Some(crate::hir::seam::Solver::new());
                }
                let solver = self.seam_solver.as_mut().unwrap();
                if let Ok(Verdict::Reject { .. }) =
                    solver.check_seam_buffers(&reached, &transfer, &["payload".to_string()])
                {
                    self.errors.warn(
                        crate::diagnostic::DiagnosticCode::W1027,
                        format!(
                            "topology '{name}': declared relaxed transfer {:?} -> {:?} does not \
                             preserve visibility; a consumer may read stale data",
                            edge.from, edge.to
                        ),
                        None,
                    );
                }
            }
        }
    }

    /// Check coherence of the program's declared memory spaces (`Memory <Name> { ... }`):
    /// `within:` is acyclic, a sub-space's capacity does not exceed its parent's, and declared
    /// properties are positive. Reads descriptors from the per-compilation env (`self.env`),
    /// not a process-global registry, so it needs no scoping list (unlike topologies).
    pub fn check_memory_coherence(&mut self) {
        use crate::hir::memory::{MemoryCoherenceIssue, MemoryHierarchy};
        let issues = MemoryHierarchy::build(self.env.memories.values().copied()).coherence_issues();
        for issue in issues {
            match issue {
                MemoryCoherenceIssue::Cycle { space } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6006,
                        format!(
                            "memory space '{}' is in a `within:` cycle (a space cannot contain \
                             itself)",
                            space.name()
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::CapacityExceedsParent {
                    child,
                    parent,
                    child_bytes,
                    parent_bytes,
                } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6007,
                        format!(
                            "memory space '{}' ({} bytes) is larger than its parent '{}' ({} \
                             bytes); a sub-space cannot exceed what contains it",
                            child.name(),
                            child_bytes,
                            parent.name(),
                            parent_bytes
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::NonPositiveProperty { space, property } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6008,
                        format!(
                            "memory space '{}' has a non-positive `{}`",
                            space.name(),
                            property
                        ),
                        None,
                    );
                }
                MemoryCoherenceIssue::ScopeWidensInChild {
                    child,
                    parent,
                    child_scope,
                    parent_scope,
                } => {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E6011,
                        format!(
                            "memory space '{}' has scope {:?}, broader than its parent '{}' \
                             ({:?}); locality must narrow down the hierarchy, not widen",
                            child.name(),
                            child_scope,
                            parent.name(),
                            parent_scope
                        ),
                        None,
                    );
                }
            }
        }
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
            let rounded = match decl.granule {
                Some(crate::syntax::ByteSize(g)) if g > 0 => raw.div_ceil(g) * g,
                _ => raw,
            };
            (rounded, cap)
        };
        let (rounded, cap) = sized;
        // Precise per-tile check: a single tile larger than the whole space is always wrong.
        if rounded > cap {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E6009,
                format!(
                    "{context} needs {rounded} bytes but memory space '{}' has capacity {cap} bytes",
                    space.name()
                ),
                None,
            );
        }
        // Record for the cumulative (working-set) budget check at end of function.
        let key = match &self.current_assignment_target {
            Some(name) => name.clone(),
            None => {
                self.placement_site += 1;
                format!("@site{}", self.placement_site)
            }
        };
        self.memory_placements
            .entry(space.clone())
            .or_default()
            .insert(key, rounded);
    }

    /// Cumulative budget check: for each memory space, the sum of the tiles a function places
    /// there (its working set) must fit `capacity`. This catches the collective overflow that
    /// the per-tile check (E6009) misses -- e.g. Q/K/V in SMEM or S/P/O in TMEM summing past the
    /// on-chip budget. Conservative (assumes all placed tiles coexist), so a space declared
    /// `overcommit` downgrades the error (E6010) to a warning (W1028). Clears the per-function
    /// placement map. Only fires when >1 tile shares a space (a lone tile is E6009's job).
    pub(crate) fn check_cumulative_capacity(&mut self) {
        let placements = std::mem::take(&mut self.memory_placements);
        self.placement_site = 0;
        let h = crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
        // (space, total, cap, tile_count, overcommit, granule)
        let mut violations: Vec<(MemorySpace, u64, u64, usize, bool, Option<u64>)> = Vec::new();
        for (space, tiles) in &placements {
            if tiles.len() < 2 {
                continue;
            }
            let Some(decl) = h.descriptor(space) else {
                continue;
            };
            let Some(crate::syntax::ByteSize(cap)) = decl.capacity else {
                continue;
            };
            // A granule'd sub-space consumes the *granule-rounded* size per tile (SS3): a 1-byte
            // tile still occupies a whole granule, so the true working set rounds each tile up.
            let granule = decl.granule.as_ref().map(|g| g.0).filter(|g| *g > 0);
            let total: u64 = tiles
                .values()
                .map(|&b| match granule {
                    Some(g) => b.div_ceil(g) * g,
                    None => b,
                })
                .sum();
            if total > cap {
                violations.push((
                    space.clone(),
                    total,
                    cap,
                    tiles.len(),
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
            if overcommit {
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1028,
                    format!("{msg}; allowed because '{}' is `overcommit`", space.name()),
                    None,
                );
            } else {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E6010,
                    format!("{msg}; place fewer/smaller tiles or declare it `overcommit`"),
                    None,
                );
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
    pub(crate) fn check_type_placement(&mut self, ty: &Type, context: &str) {
        match ty {
            Type::Ref(inner, mem) => {
                if let Some((e, d)) = Self::tensor_of(inner) {
                    self.check_capacity(e, d, mem, context);
                }
            }
            Type::Pinned(inner, top) if !matches!(top, Topology::Current) => {
                if let Some((e, d)) = Self::tensor_of(inner) {
                    let space = self.transfer_cost_graph.default_memory_for(top);
                    self.check_capacity(e, d, &space, context);
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
        if self.seam_solver.is_none() {
            let init = std::time::Instant::now();
            self.seam_solver = Some(Solver::new());
            self.solver_init_time += init.elapsed();
        }
        let solver = self.seam_solver.as_mut().unwrap();

        let start = std::time::Instant::now();
        // A buffer with a statically-known value gets the stronger value contract (pin
        // the value); an opaque buffer falls back to the visibility obligation.
        let verdict = match known_val {
            Some(v) => solver.check_seam_value(&reached, &transfer, buffer, v),
            None => solver.check_seam_buffers(&reached, &transfer, &consumed),
        };
        self.seam_check_time += start.elapsed();
        self.seam_checks += 1;

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
                // Solver error: fail open (as prover.rs does) but record a warning.
                if !self.speculating {
                    self.errors.warn(
                        crate::diagnostic::DiagnosticCode::E6004,
                        format!("seam obligation could not be discharged: {}", e),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                    );
                }
            }
        }
    }

    pub(crate) fn check_transfer_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let mut do_rewrite = None;
        let target_mem;
        let inner_ty;

        // Whether this transfer is a relaxed escape hatch (set by the caller, e.g. the
        // `to_device_relaxed` method arm). Consumed here so it does not leak to siblings.
        let relaxed = std::mem::take(&mut self.pending_transfer_relaxed);

        if let Expr::Transfer(t) = expr {
            let prev = self.allow_cross_topology;
            self.allow_cross_topology = true;
            inner_ty = self.check_expr_type_flag(&mut t.expr, false);
            self.allow_cross_topology = prev;

            // Extract source memory space, preferring exact space from an inner transfer if present
            let source_mem = if let Expr::Transfer(inner_t) = &*t.expr {
                inner_t.space.clone()
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
            target_mem = t.space.clone();

            // Capacity: a statically-shaped tensor transferred into a declared space must fit.
            if let Some((e, d)) = Self::tensor_of(&inner_ty) {
                self.check_capacity(e, d, &target_mem, "transferred tensor");
            }

            // Bandwidth-derived roofline cost (bytes / bandwidth along the hierarchy). When the
            // memory declarations make it computable it becomes the transfer's cost (emitted on
            // `vx.transfer`); otherwise the fixed cost-graph value is kept. This does not touch
            // reachability/seam, which still use `transfer_path` below.
            let derived_cost: Option<u32> = Self::tensor_of(&inner_ty)
                .and_then(|(e, d)| crate::hir::memory::static_tensor_bytes(e, d))
                .and_then(|bytes| {
                    crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied())
                        .derived_transfer_cost(&source_mem, &target_mem, bytes)
                        .map(|dc| dc.value.min(u32::MAX as u64) as u32)
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
                    self.errors.push(format!(
                        "Cannot transfer from {:?} to {:?}: no hardware path exists",
                        source_mem, target_mem
                    ));
                }
                return Type::Unknown; // Poison: no valid transfer, don't fake an f32 tensor
            }

            let (_cost, path) = path_result.unwrap();
            if path.len() > 2 {
                do_rewrite = Some(path);
            } else {
                // Record the bandwidth-derived roofline cost when the hierarchy provides one;
                // otherwise leave it unset (the fixed reachability cost stays internal, so
                // bandwidth-less transfers emit no `cost` attribute — unchanged output).
                t.cost = derived_cost;
                // Single hop (`path == [source_mem, target_mem]`): discharge the
                // per-seam local-completeness / soundness obligation. Multi-hop paths
                // are rewritten into a chain of single-hop transfers below, each of
                // which re-enters here and is checked individually.
                if path.len() == 2 && self.verify_seams {
                    let span = Self::buffer_span(&t.expr).unwrap_or(t.span);
                    let buffer = Self::buffer_label(&t.expr);
                    // Prefer the value the consumer asserts of the buffer this transfer
                    // produces (looked up by the let-binding target, e.g. `local_a`), which
                    // is the boundary contract's conclusion; fall back to the producer's
                    // own statically-known constant, else the coarse visibility check.
                    let asserted_val = self
                        .current_assignment_target
                        .as_ref()
                        .and_then(|tgt| self.seam_contracts.get(tgt).copied());
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
            }
        } else {
            unreachable!()
        }

        if let Some(path) = do_rewrite {
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
                    span: t.span,
                });
            }
            *expr = Expr::Transfer(TransferExpr {
                expr: Box::new(current_expr),
                space: target_mem.clone(),
                cost: None,
                span: t.span,
            });
            // Recursively re-evaluate to ensure intermediate types and costs are resolved properly!
            // Propagate the relaxed marker so each rewritten single-hop transfer is checked
            // with the right transfer function.
            self.pending_transfer_relaxed = relaxed;
            return self.check_transfer_expr(expr, consume);
        }

        match inner_ty {
            Type::Ref(base_ty, _) => Type::Ref(base_ty, target_mem.clone()),
            Type::Tensor(_, _, _) => {
                let pinned_top = match &target_mem {
                    MemorySpace::NPUHBM => Topology::NPU(Box::new(Expr::Number(NumberExpr {
                        value: "0".into(),
                        ty: Some(ElementType::I32),
                        span: Span::default(),
                    }))),
                    MemorySpace::LocalSRAM => {
                        Topology::AccCore(Box::new(Expr::Number(NumberExpr {
                            value: "0".into(),
                            ty: Some(ElementType::I32),
                            span: Span::default(),
                        })))
                    }
                    MemorySpace::NicRam | MemorySpace::RemoteHbm => {
                        Topology::NPU(Box::new(Expr::Number(NumberExpr {
                            value: "0".into(),
                            ty: Some(ElementType::I32),
                            span: Span::default(),
                        })))
                    }
                    MemorySpace::GpuHbm => Topology::GPU,
                    MemorySpace::CPUDRAM => Topology::CPU,
                    // A value in a user-defined memory space is pinned on the like-named
                    // custom topology (naming convention: Memory::Foo <-> Topology::Foo).
                    MemorySpace::Custom(name) => Topology::Custom(name.clone()),
                };
                Type::Pinned(Box::new(inner_ty.clone()), pinned_top)
            }
            Type::Verified(_inner) => {
                if let Expr::Transfer(t) = expr {
                    let inner_pinned = self.check_expr_type_flag(&mut t.expr, consume);
                    Type::Verified(Box::new(inner_pinned))
                } else {
                    unreachable!()
                }
            }
            Type::Pinned(base, _) => {
                let pinned_top = match &target_mem {
                    MemorySpace::NPUHBM => Topology::NPU(Box::new(Expr::Number(NumberExpr {
                        value: "0".into(),
                        ty: Some(ElementType::I32),
                        span: Span::default(),
                    }))),
                    MemorySpace::LocalSRAM => {
                        Topology::AccCore(Box::new(Expr::Number(NumberExpr {
                            value: "0".into(),
                            ty: Some(ElementType::I32),
                            span: Span::default(),
                        })))
                    }
                    MemorySpace::NicRam | MemorySpace::RemoteHbm => {
                        Topology::NPU(Box::new(Expr::Number(NumberExpr {
                            value: "0".into(),
                            ty: Some(ElementType::I32),
                            span: Span::default(),
                        })))
                    }
                    MemorySpace::GpuHbm => Topology::GPU,
                    MemorySpace::CPUDRAM => Topology::CPU,
                    // A value in a user-defined memory space is pinned on the like-named
                    // custom topology (naming convention: Memory::Foo <-> Topology::Foo).
                    MemorySpace::Custom(name) => Topology::Custom(name.clone()),
                };
                Type::Pinned(base, pinned_top)
            }
            _ => {
                if !self.speculating {
                    self.errors.push(format!(
                        "Cannot transfer non-reference type: {:?}",
                        inner_ty
                    ));
                }
                Type::Tensor(ElementType::F32, vec![], None)
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
                    Topology::CPU
                    | Topology::AMX
                    | Topology::ANE
                    | Topology::GPU
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

                self.push_scope();

                self.check_expr_block(stmts, consume);

                let mut ret_ty = Type::Tensor(ElementType::F32, vec![], None); // default void-like type
                let has_ret = ret.is_some();
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type_flag(r, consume);
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
