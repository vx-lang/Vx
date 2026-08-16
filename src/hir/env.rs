//===- sema.rs - Vx Compiler -----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Semantic Analyzer for the Vx compiler.
// It is responsible for type checking, resolving operator overloading (e.g., tensor
// matrix multiplication), verifying memory topology constraints, and constructing
// the global AST environment for subsequent lowering phases.
//
// The Semantic Analyzer also includes the Lexical Borrow Checker, which handles local variable
// lifetimes and Strict Aliasing (Shared XOR Mutable).
// For a comprehensive overview of the Borrow Checker architecture (and how it interacts with
// the FastPath in borrow.rs), see: `docs/discussions/borrow_checker_architecture.md`.
//
// DESIGN NOTE: Speculative checking is carried by the `speculating` field (see its doc on
// `TypeChecker`), not a threaded `silent` parameter. It prevents duplicate compiler errors:
// an AST node is sometimes checked twice — once to probe a type (e.g. the method-call →
// function-call return-type probe), then again for real — so the probe runs with `speculating`
// on to suppress redundant emissions and side effects. Retired the former `silent: bool`
// parameter that threaded through ~36 checker signatures (#279 R3).
//
//===----------------------------------------------------------------------===//
use crate::syntax::*;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Bool(bool),
    Number(f64),
    Topology(Topology),
}

/// One memory space's working set for a function: the granule-rounded sum of the tiles placed
/// there, against the space's declared capacity. Recorded for *every* placed space, not only
/// those that overflow — an admitted program emits no capacity diagnostic, so this is the only
/// place its resident total appears. Downstream that total is what turns a capacity verdict into
/// an engine's required memory utilization (#285). See [`TypeChecker::resident_sets`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentSet {
    pub space: crate::syntax::MemorySpace,
    /// Granule-rounded sum of tiles placed in this space by one function.
    pub total_bytes: u64,
    pub capacity_bytes: u64,
    pub tiles: usize,
    /// Whether the space is declared `overcommit` (an overflow is W1028, not E6010).
    pub overcommit: bool,
}

/// A resolved `transfer`: the chain of memory spaces the value actually moves through and the
/// declared cost of each hop. A direct transfer has one hop; a multi-hop route (synthesized when
/// no direct edge exists) has one per staging step. `derived_cost` is the bandwidth-roofline cost
/// when the declarations make it computable, which is what the emitted `vx.transfer` carries.
/// See [`TypeChecker::staging_routes`] (#282).
#[derive(Debug, Clone, PartialEq)]
pub struct StagingRoute {
    /// The spaces traversed, source first: `[CPU_DRAM, HBM3e, SMEM]`.
    pub path: Vec<crate::syntax::MemorySpace>,
    /// Per-edge declared cost, one shorter than `path`; `None` for an edge with no declared cost.
    pub edge_costs: Vec<Option<u32>>,
    /// The cost graph's total for the whole route.
    pub total_cost: u32,
    /// Bandwidth-derived (roofline) cost, when the declarations supply bandwidths.
    ///
    /// `u64`, not `u32`: a picosecond-resolution time cost overflows `u32` at 4.3 ms, which a
    /// multi-gigabyte host transfer exceeds, and a saturated cost would read as a slow transfer
    /// rather than a missing one.
    pub derived_cost: Option<u64>,
    /// Bytes this transfer moves, when the shape is statically known.
    ///
    /// Without it a harvested cost cannot be interpreted: "260064 ps" is not a prediction unless
    /// the size it is a cost *of* travels with it, and comparing predicted against measured needs
    /// both. `None` for a dynamically-shaped tensor, where no cost is derivable either.
    pub bytes: Option<u64>,
    /// Which cost source produced `derived_cost`. The two are mutually exclusive per edge (E6013),
    /// so this names the one that applied rather than a precedence winner — and it is what tells a
    /// consumer whether a residual is attributable to a declared link figure or to an inferred
    /// containment roofline.
    pub cost_source: Option<CostSource>,
    /// The unit `derived_cost` is in — cycles, or picoseconds. Carried rather than dropped because
    /// one program's routes legitimately mix them: an on-die hop declared `B/cyc` and a host link
    /// declared `GB/s` produce costs of different *dimension*, and a harvested prediction that does
    /// not say which is not a prediction. (The fleet's `HBM->L2` and `L2->SMEM` are exactly this
    /// pair.)
    pub derived_unit: Option<crate::syntax::RatePer>,
    /// Which composition law priced a containment route: `sum` or `bottleneck` (vx-review#26).
    ///
    /// `None` for a link-rate hop, which has one leg and so composes nothing. Carried because the
    /// two laws differ by ~2x on a multi-hop walk and a harvested prediction that does not say
    /// which one applied cannot be re-scored later -- the same reason `derived_unit` is carried.
    pub composition: Option<crate::syntax::Crossing>,
}

/// Where a hop's predicted cost came from. Exactly one applies per edge (E6013).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostSource {
    /// A bandwidth declared on the link itself (`transfer A -> B : 64 GB/s`). Used where the
    /// endpoints do not nest, so containment can derive nothing — the host↔device seam.
    LinkRate,
    /// The roofline derived from the endpoints' own `bandwidth:` figures along the `within:` tree.
    Containment,
}

impl CostSource {
    pub fn as_str(self) -> &'static str {
        match self {
            CostSource::LinkRate => "link_rate",
            CostSource::Containment => "containment",
        }
    }
}

/// One `Memory`/`Topology` name declared by two modules with *different* declarations — see
/// [`GlobalAstEnv::duplicate_decls`].
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateDecl {
    /// What was re-declared, for the message: `"memory space"` or `"topology"`.
    pub kind: &'static str,
    pub name: crate::symbol::Symbol,
    /// The module whose declaration collided with one already indexed.
    pub module: crate::symbol::Symbol,
}

pub struct GlobalAstEnv<'a> {
    pub structs: HashMap<crate::symbol::Symbol, &'a StructDecl>,
    #[allow(clippy::type_complexity)]
    pub enums: HashMap<crate::symbol::Symbol, &'a EnumDecl>,
    pub traits: HashMap<crate::symbol::Symbol, &'a TraitDecl>,
    pub impls: HashMap<crate::symbol::Symbol, Vec<&'a ImplBlock>>,
    #[allow(clippy::type_complexity)]
    pub functions:
        HashMap<crate::symbol::Symbol, (Type, bool, Vec<Type>, Topology, Vec<Expr>, Vec<Expr>)>,
    pub syntax_functions: HashMap<crate::symbol::Symbol, &'a Function>,
    pub generic_functions: HashMap<crate::symbol::Symbol, (&'a Function, u64)>, // (func, origin_module_hash)
    /// User-defined memory spaces (`Memory <Name> { ... }`), indexed by name. Populated from
    /// `Program.memories` — the per-compilation home for memory descriptors (no global registry).
    pub memories: HashMap<crate::symbol::Symbol, &'a MemoryDecl>,
    /// Transfer lowerings (`impl transfer A -> B { ... }`) across every module of this
    /// compilation, in module order. A Vec rather than an edge-keyed map on purpose: the
    /// duplicate-edge check (E6015) needs to SEE both declarations to report one, and a map
    /// would have silently kept whichever was inserted last.
    pub transfer_impls: Vec<&'a crate::syntax::TransferImplDecl>,
    /// User-defined topologies (`Topology <Name> { ... }`), indexed by name. Populated from
    /// `Program.topologies`; the per-compilation home for topology descriptors, seeded into each
    /// `TransferCostGraph` — no global registry (see docs/parallel_compiler_architecture.md).
    pub topologies: HashMap<crate::symbol::Symbol, &'a crate::arch::TopologyDecl>,
    /// `Memory`/`Topology` names declared by more than one module in this compilation unit, with
    /// the modules that declared them. The tables above are name-keyed, so a duplicate would
    /// silently resolve to whichever module was indexed last -- and module order is a `HashMap`
    /// iteration, so the machine model actually in force would vary between runs. Recorded here
    /// and reported as E6012 by `check_declaration_conflicts` (#281).
    pub duplicate_decls: Vec<DuplicateDecl>,
    /// Per-function return-provenance summary (#243): which parameter slot(s) a reference-returning
    /// function's result roots in. A read-only, precomputed artifact of the immutable env, consulted
    /// at call sites to make the reborrow-persistence decision per argument. Filled from *present*
    /// bodies here (covers full-AST callers); production entry points that strip bodies before
    /// building refill it from the full modules via [`Self::annotate_return_provenances`]. A missing
    /// entry means `AnyParam` — today's conservative behaviour. See `crate::hir::provenance`.
    pub return_provenances:
        HashMap<crate::symbol::Symbol, crate::hir::provenance::ReturnProvenance>,
    /// This compilation's transfer-cost graph: the built-in memory-space edges plus every declared
    /// topology's, with the all-pairs shortest-path matrix precomputed.
    ///
    /// Built once here, at the end of `build_from_refs`, and shared by `&` with every
    /// `TypeChecker`. It used to be built inside `TypeChecker::new`, i.e. once per *function*,
    /// running the all-pairs precompute twice each time — which a profile put at the top of the
    /// whole compiler by self time. Every use in the checker is a `&self` read, so there is nothing
    /// per-function about it.
    pub transfer_cost_graph: crate::arch::TransferCostGraph,
}

impl<'a> GlobalAstEnv<'a> {
    pub fn build(modules: &'a [Program]) -> Self {
        let refs: Vec<&'a Program> = modules.iter().collect();
        Self::build_from_refs(&refs)
    }

    pub fn build_from_refs(modules: &[&'a Program]) -> Self {
        let mut env = Self {
            structs: HashMap::new(),
            enums: HashMap::new(),
            traits: HashMap::new(),
            impls: HashMap::new(),
            functions: HashMap::new(),
            syntax_functions: HashMap::new(),
            generic_functions: HashMap::new(),
            memories: HashMap::new(),
            topologies: HashMap::new(),
            duplicate_decls: Vec::new(),
            transfer_impls: Vec::new(),
            return_provenances: HashMap::new(),
            transfer_cost_graph: crate::arch::TransferCostGraph::default(),
        };

        for &module in modules {
            let module_hash = crate::hash::compute_module_hash(&module.module_path);
            for s in &module.structs {
                env.structs.insert(s.name.clone(), s);
            }
            for m in &module.memories {
                if let Some(prev) = env.memories.insert(m.name.clone(), m) {
                    // Re-declaring an *identical* space in two modules is harmless (the same
                    // machine file reached twice); only a genuine disagreement is ambiguous.
                    if prev != m {
                        env.duplicate_decls.push(DuplicateDecl {
                            kind: "memory space",
                            name: m.name.clone(),
                            module: module.module_path.clone(),
                        });
                    }
                }
            }
            for t in &module.transfer_impls {
                env.transfer_impls.push(t);
            }
            for t in &module.topologies {
                if let Some(prev) = env.topologies.insert(t.name.clone(), t) {
                    if prev != t {
                        env.duplicate_decls.push(DuplicateDecl {
                            kind: "topology",
                            name: t.name.clone(),
                            module: module.module_path.clone(),
                        });
                    }
                }
            }
            for e in &module.enums {
                env.enums.insert(e.name.clone(), e);
            }
            for t in &module.traits {
                env.traits.insert(t.name.clone(), t);
            }
            for i in &module.impls {
                let trait_name = match &i.trait_name {
                    Some(name) => name.clone(),
                    None => "_inherent".to_string().into(),
                };
                env.impls.entry(trait_name).or_default().push(i);
            }
            for ext in &module.externs {
                let param_types: Vec<Type> = ext.params.iter().map(|(_, t)| t.clone()).collect();
                env.functions.insert(
                    ext.name.clone(),
                    (
                        ext.return_type.clone(),
                        !ext.is_safe,
                        param_types,
                        Topology::CPU,
                        Vec::new(),
                        Vec::new(),
                    ),
                );
            }
            for func in &module.functions {
                if !func.generics.is_empty() {
                    env.generic_functions
                        .insert(func.name.clone(), (func, module_hash));
                } else {
                    let param_types: Vec<Type> =
                        func.params.iter().map(|(_, t)| t.clone()).collect();
                    env.functions.insert(
                        func.name.clone(),
                        (
                            func.return_type.clone(),
                            false, /* func.is_unsafe */
                            param_types,
                            func.topology.clone(),
                            func.requires.clone(),
                            func.ensures.clone(),
                        ),
                    );
                    env.syntax_functions.insert(func.name.clone(), func);
                    // Summarize the return provenance while the body is present. Signature-stripped
                    // callers (production) leave this empty and refill via
                    // `annotate_return_provenances` from the full modules.
                    if !func.body.is_empty() {
                        env.return_provenances.insert(
                            func.name.clone(),
                            crate::hir::provenance::compute_return_provenance(func),
                        );
                    }
                }
            }
        }
        // Build the transfer-cost graph once, now that `topologies` is populated. It used to be
        // built inside `TypeChecker::new` -- that is, once per *function* -- where it ran the
        // all-pairs shortest-path precompute twice (once in `default()`, once in
        // `seed_from_topologies`) and cloned every topology declaration in the compilation. A
        // profile of a 6,000-module corpus put `TransferCostGraph::transfer_path` at the top of the
        // whole compiler by self time, on a corpus that declares no topologies at all.
        //
        // The graph depends only on this compilation's topology declarations, and every one of its
        // ~31 uses in the checker is a `&self` read, so one per compilation is not merely an
        // optimisation -- it is what the comment on `seed_from_topologies` already claimed it was:
        // "the per-compilation, lock-free carrier the parallel pipeline shares by `&`".
        env.transfer_cost_graph = {
            let mut g = crate::arch::TransferCostGraph::default();
            let decls: Vec<crate::arch::TopologyDecl> =
                env.topologies.values().map(|&d| d.clone()).collect();
            g.seed_from_topologies(&decls);
            // Routing minimises predicted cost (vx-review#19), and a containment hop's cost lives
            // in the memory declarations rather than the topology's. Without this the on-die edges
            // -- which is every hop inside a device on every fleet SKU -- stay unpriced and the
            // router treats them as the last resort they are not.
            g.resolve_derived_route_costs(env.memories.values().copied());
            g
        };
        env
    }

    /// Refill `return_provenances` from modules that still carry function bodies (#243). Production
    /// pipelines build the env from *signature-stripped* modules, so `build` cannot summarize their
    /// free functions; the entry points call this with the full pre-strip modules to restore
    /// precision. Idempotent — overwrites any existing entry. A function whose summary cannot be
    /// computed (no body) is left absent, i.e. conservative `AnyParam` at lookup.
    pub fn annotate_return_provenances(&mut self, modules: &[Program]) {
        for module in modules {
            for func in &module.functions {
                if func.generics.is_empty() && !func.body.is_empty() {
                    self.return_provenances.insert(
                        func.name.clone(),
                        crate::hir::provenance::compute_return_provenance(func),
                    );
                }
            }
        }
    }

    /// The return-provenance summary for a callee, or the conservative default when unknown.
    pub fn return_provenance_of(&self, name: &str) -> crate::hir::provenance::ReturnProvenance {
        self.return_provenances
            .get(name)
            .copied()
            .unwrap_or(crate::hir::provenance::ReturnProvenance::AnyParam)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BorrowRecord {
    pub is_mut: bool,
    pub scope_depth: usize,
    pub borrower_name: Option<String>,
    pub path: Vec<String>,
}

/// Where a reference value ultimately points, for return-escape analysis (#243).
/// A reference may be returned iff its provenance is `External`; returning a `Local`
/// reference would leave it dangling once the function's stack frame is gone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefProvenance {
    /// Roots in caller-owned memory: a reference *parameter* (or `'static`). Safe to return.
    External,
    /// Roots in a function-local slot: a `let` binding, a by-value parameter, or a
    /// temporary. Returning it dangles.
    Local,
}

pub struct TypeChecker<'a> {
    pub worker: &'a mut crate::session::LocalWorkerState,
    pub env: &'a GlobalAstEnv<'a>,
    pub(crate) scopes: Vec<HashMap<crate::symbol::Symbol, (Type, Topology)>>,
    pub allow_cross_topology: bool,
    pub monomorphized_functions: Vec<(Function, u64)>,
    pub errors: crate::diagnostic::DiagnosticsVec,
    pub(crate) in_unsafe_block: bool,
    pub(crate) active_topology: Topology,
    pub(crate) active_memory: MemorySpace,
    pub transfer_cost_graph: &'a crate::arch::TransferCostGraph,
    /// Borrow-checking state (active records + NLL liveness), encapsulated so a conflict-check read
    /// cannot bypass the dead-borrow sweep (frontend_refactoring_borrow_checker.md R1; the #276 bug class). Replaces
    /// the former `active_borrows` / `block_liveness` / `current_stmt_idx` fields.
    pub(crate) borrow: crate::hir::borrow_cx::BorrowCx,
    /// Speculative-check mode. When set, expression checking is a *probe* — diagnostics are
    /// suppressed, borrow/move side effects and unreachable-code warnings are skipped, and
    /// `check_if_expr`/`check_match_expr` return a placeholder without descending into the block.
    /// Replaces the `silent: bool` parameter that used to thread through ~36 checker signatures
    /// (frontend_refactoring_borrow_checker.md R3, #279). The only site that turns it on is the
    /// method-call → function-call return-type probe in `check_methodcall_expr`; the two
    /// "fresh check" entry points (`check_expr_type`, `check_block`) force it back off, so the
    /// field's dynamic scope reproduces the old parameter's exactly.
    pub(crate) speculating: bool,
    pub constraints: Vec<Expr>,
    pub return_constraints: Vec<Expr>,
    pub(crate) next_id: u32,
    pub eval_env: Vec<HashMap<crate::symbol::Symbol, Value>>,
    pub current_return_type: Option<Type>,
    #[allow(dead_code)]
    pub(crate) closure_depths: Vec<usize>,
    #[allow(dead_code)]
    pub(crate) closure_captures_stack: Vec<HashMap<crate::symbol::Symbol, Type>>,
    pub generated_structs: Vec<StructDecl>,
    pub(crate) current_assignment_target: Option<String>,
    /// Tracks which variables have been read during the current function check.
    pub(crate) used_vars: std::collections::HashSet<crate::symbol::Symbol>,
    /// Tracks declared variables with their spans (for unused variable warnings).
    pub(crate) declared_vars: Vec<(crate::symbol::Symbol, crate::syntax::Span)>,
    /// Set immediately before a transfer is checked to mark it a *relaxed* (escape-hatch)
    /// transfer that does not carry a synchronizing release/DMA-completion. Consumed and
    /// reset by `check_transfer_expr`. See `crate::hir::seam`.
    pub(crate) pending_transfer_relaxed: bool,
    /// The edge of the `impl transfer` lowering whose body is being checked, when one is.
    /// `Some((from, to))` is what makes the eight `raw::` primitives resolve (Vx#353 A2)
    /// and gives their space and capability obligations an edge to check against. `None`
    /// everywhere else -- which is the floor property: outside a lowering, `raw::` names
    /// do not exist. Set by the driver and pipeline around lowering-body checks.
    pub transfer_lowering_edge: Option<(MemorySpace, MemorySpace)>,
    /// The current lowering method's parameter names: the only tiles `raw::` may touch.
    /// A local alias would escape the async-discipline walk (it is name-keyed), so the
    /// primitives are limited to the names the walk can see. Only read while
    /// `transfer_lowering_edge` is `Some`.
    pub(crate) transfer_lowering_params: std::collections::HashSet<crate::symbol::Symbol>,
    /// Number of per-seam obligations discharged (eval metric M1).
    pub seam_checks: usize,
    /// Every staging route this compilation resolved, in source order: the hops a `transfer`
    /// lowered to and their per-edge costs. An *admitted* program emits no diagnostics, so this
    /// is what `--diagnostics-json` reports for it -- the accept side of the admission verdict
    /// (#282).
    pub staging_routes: Vec<StagingRoute>,
    /// Per-function, per-space working sets — the resident sets an admitted program implies.
    /// See [`ResidentSet`]; reported by `--diagnostics-json` (#285).
    pub resident_sets: Vec<ResidentSet>,
    /// Total marginal solving time across all seams, excluding the one-time solver
    /// startup below (eval metric M1: per-seam proof cost).
    pub seam_check_time: std::time::Duration,
    /// One-time cost of spawning the persistent solver and installing its preamble,
    /// paid once per compilation regardless of program size (eval metric M1).
    pub solver_init_time: std::time::Duration,
    /// Persistent z3 process, lazily started on the first seam and reused for all of
    /// them so the marginal per-seam cost is solving time, not process startup.
    pub(crate) seam_solver: Option<crate::hir::seam::Solver>,
    /// Per-function pre-scan of `assert(var == const)` facts: the value a consumer
    /// requires of `var`. Populated before statements are checked so a transfer seam
    /// (checked before the consumer's `spawn` body) can consult the downstream
    /// contract on the buffer it produces. See `collect_assert_contracts`.
    pub(crate) seam_contracts: std::collections::HashMap<String, u64>,
    /// Per-function working set: `memory space -> {buffer key -> granule-rounded bytes}`. Every
    /// tile placed in a declared space (via `transfer`/`Ref` annotation) is recorded here so the
    /// *cumulative* budget check can sum them and flag a space whose total exceeds `capacity`.
    /// Reset per function. See `check_cumulative_capacity`.
    pub(crate) memory_placements: std::collections::HashMap<
        crate::syntax::MemorySpace,
        std::collections::HashMap<String, u64>,
    >,
    /// Monotonic id for placements with no binding name (so they still count toward the sum).
    pub(crate) placement_site: usize,
    /// Whether to discharge per-seam boundary obligations (the assert pre-scan and the
    /// z3 checks). Off by default so ordinary compilation pays nothing and needs no
    /// solver; enabled with `vxc --verify-seams`. See `crate::hir::seam`.
    pub verify_seams: bool,
    /// Topology generic parameters (`<D: Topology>`) of the function currently being
    /// instantiated. Set around the deduction unify at a generic call so `unify_types`
    /// can bind a `Pinned<_, D>` param's topology variable instead of demanding equality.
    pub(crate) pending_topo_vars: std::collections::HashSet<crate::symbol::Symbol>,
    /// Topology bindings (`D -> concrete topology`) deduced during that unify, consumed by
    /// `instantiate_function` to specialize `on D` and `Pinned<_, D>`.
    pub(crate) pending_topo_bindings: std::collections::HashMap<crate::symbol::Symbol, Topology>,
    /// Expected type of the expression currently being checked, from a `let x: T = …` or a
    /// `return` in a typed function. Lets a generic call deduce a *return-only* topology (or
    /// type) variable — e.g. `D` in `-> Pinned<T, D>` — from the call's context.
    pub(crate) expected_type: Option<Type>,
    /// Signature of each closure literal, keyed by its generated `Closure_N` struct name:
    /// `(param types, return type)`. A closure expression checks to `Struct("Closure_N")`, which
    /// erases the call signature, so unifying it against a `ClosureK<Args.., Ret>` parameter needs
    /// this side table to recover the args/ret and bind the method's generics (e.g. `NewItem` in
    /// `.map(|x| ...)`). See `check_closure_expr` and `unify_types_internal`.
    pub(crate) closure_signatures:
        std::collections::HashMap<crate::symbol::Symbol, (Vec<Type>, Type)>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(
        env: &'a GlobalAstEnv<'a>,
        worker: &'a mut crate::session::LocalWorkerState,
    ) -> Self {
        // Build the per-compilation cost graph first: fold in the topologies this compilation
        // declared (carried on the AST, indexed by `env.topologies`) so it holds both their
        // descriptors and transfer edges. No global registry -- the graph is the per-compilation,
        // lock-free carrier the parallel pipeline shares by `&`.
        let transfer_cost_graph = &env.transfer_cost_graph;
        let active_memory = transfer_cost_graph.default_memory_for(&Topology::CPU);
        Self {
            env,
            worker,
            scopes: vec![HashMap::new()],
            monomorphized_functions: Vec::new(),
            errors: crate::diagnostic::DiagnosticsVec::new(),
            in_unsafe_block: false,
            allow_cross_topology: false,
            active_topology: Topology::CPU,
            active_memory,
            transfer_cost_graph,
            borrow: crate::hir::borrow_cx::BorrowCx::default(),
            speculating: false,
            constraints: Vec::new(),
            return_constraints: Vec::new(),
            next_id: 1,
            eval_env: vec![HashMap::new()],
            current_return_type: None,
            closure_depths: Vec::new(),
            closure_captures_stack: Vec::new(),
            generated_structs: Vec::new(),
            current_assignment_target: None,
            used_vars: std::collections::HashSet::new(),
            declared_vars: Vec::new(),
            pending_transfer_relaxed: false,
            transfer_lowering_edge: None,
            transfer_lowering_params: std::collections::HashSet::new(),
            seam_checks: 0,
            staging_routes: Vec::new(),
            resident_sets: Vec::new(),
            seam_check_time: std::time::Duration::ZERO,
            solver_init_time: std::time::Duration::ZERO,
            seam_solver: None,
            seam_contracts: std::collections::HashMap::new(),
            memory_placements: std::collections::HashMap::new(),
            placement_site: 0,
            verify_seams: false,
            pending_topo_vars: std::collections::HashSet::new(),
            pending_topo_bindings: std::collections::HashMap::new(),
            expected_type: None,
            closure_signatures: std::collections::HashMap::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(std::collections::HashMap::new());
        self.borrow
            .moved_vars
            .push(std::collections::HashSet::new());
        self.eval_env.push(std::collections::HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        let depth = self.scopes.len();
        self.scopes.pop();
        self.borrow.moved_vars.pop();
        self.eval_env.pop();

        // Lexical Lifetime cleanup: Remove borrows originating in this scope
        self.borrow.retain_scope(depth);
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        let current_top = self.active_topology.clone();
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), (ty, current_top));
        }
    }

    pub fn extract_uses_stmt(stmt: &Statement, uses: &mut std::collections::HashSet<String>) {
        match stmt {
            Statement::ExprStmt(e) => Self::extract_uses_expr(&e.expr, uses),
            Statement::Return(r) => Self::extract_uses_expr(&r.expr, uses),
            Statement::Assign(a) => {
                Self::extract_uses_expr(&a.lhs, uses);
                Self::extract_uses_expr(&a.rhs, uses);
            }
            Statement::CompoundAssign(ca) => {
                Self::extract_uses_expr(&ca.lhs, uses);
                Self::extract_uses_expr(&ca.rhs, uses);
            }
            Statement::LetDecl(l) => Self::extract_uses_expr(&l.expr, uses),
            Statement::ForLoop(f) => {
                Self::extract_uses_expr(&f.iterable, uses);
                for inv in &f.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &f.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Loop(l) => {
                for inv in &l.invariants {
                    Self::extract_uses_expr(inv, uses);
                }
                for s in &l.body {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Statement::Assert(a) => Self::extract_uses_expr(&a.expr, uses),
            _ => {}
        }
    }

    pub fn extract_uses_expr(expr: &Expr, uses: &mut std::collections::HashSet<String>) {
        match expr {
            Expr::Identifier(id) => {
                uses.insert(id.name.to_string());
            }
            Expr::MemberAccess(m) => Self::extract_uses_expr(&m.base, uses),
            Expr::MethodCall(m) => {
                Self::extract_uses_expr(&m.base, uses);
                for a in &m.args {
                    Self::extract_uses_expr(a, uses);
                }
            }
            Expr::FunctionCall(f) => {
                uses.insert(f.name.to_string());
                for a in &f.args {
                    Self::extract_uses_expr(a, uses);
                }
            }
            Expr::BinaryOp(b) => {
                Self::extract_uses_expr(&b.lhs, uses);
                Self::extract_uses_expr(&b.rhs, uses);
            }
            Expr::RelationalOp(r) => {
                Self::extract_uses_expr(&r.lhs, uses);
                Self::extract_uses_expr(&r.rhs, uses);
            }
            Expr::LogicalOp(l) => {
                Self::extract_uses_expr(&l.lhs, uses);
                Self::extract_uses_expr(&l.rhs, uses);
            }
            Expr::UnaryOp(u) => Self::extract_uses_expr(&u.expr, uses),
            Expr::IndexAccess(i) => {
                Self::extract_uses_expr(&i.base, uses);
                Self::extract_uses_expr(&i.index, uses);
            }
            Expr::Borrow(b) => Self::extract_uses_expr(&b.expr, uses),
            Expr::Dereference(d) => Self::extract_uses_expr(&d.expr, uses),
            Expr::StructInit(s) => {
                for f in &s.fields {
                    Self::extract_uses_expr(&f.1, uses);
                }
            }
            Expr::Array(a) => {
                for e in &a.elements {
                    Self::extract_uses_expr(e, uses);
                }
            }
            Expr::If(i) => {
                Self::extract_uses_expr(&i.cond, uses);
                for s in &i.then_block {
                    Self::extract_uses_stmt(s, uses);
                }
                if let Some(eb) = &i.else_block {
                    for s in eb {
                        Self::extract_uses_stmt(s, uses);
                    }
                }
            }
            Expr::UnsafeBlock(u) => {
                for s in &u.stmts {
                    Self::extract_uses_stmt(s, uses);
                }
            }
            Expr::AsCast(c) => Self::extract_uses_expr(&c.expr, uses),
            Expr::Closure(c) => {
                Self::extract_uses_expr(&c.body, uses);
            }
            _ => {}
        }
    }

    pub fn consume(&mut self, name: &str) {
        // Find the most recent block where it's defined
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.remove(name);
                if let Some(last) = self.borrow.moved_vars.last_mut() {
                    last.insert(name.to_string());
                }
                return;
            }
        }
    }

    pub fn is_moved(&self, name: &str) -> bool {
        for moved in self.borrow.moved_vars.iter().rev() {
            if moved.contains(name) {
                return true;
            }
        }
        false
    }

    pub fn lookup(&self, name: &str) -> Option<&(Type, Topology)> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    pub fn lookup_with_depth(&self, name: &str) -> Option<(&Type, &Topology, usize)> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some((ty, top)) = scope.get(name) {
                return Some((ty, top, i));
            }
        }
        None
    }

    pub fn unify_types(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        let mut temp_mapping = mapping.clone();
        if self.unify_types_internal(generic_ty, concrete_ty, &mut temp_mapping) {
            *mapping = temp_mapping;
            true
        } else {
            false
        }
    }

    fn unify_types_internal(
        &mut self,
        generic_ty: &Type,
        concrete_ty: &Type,
        mapping: &mut std::collections::HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        match (generic_ty, concrete_ty) {
            (Type::Generic(name, _), _) => {
                if let Some(existing) = mapping.get(name) {
                    existing == concrete_ty
                } else {
                    mapping.insert(name.clone(), concrete_ty.clone());
                    true
                }
            }
            (Type::Tensor(e1, d1, t1), Type::Tensor(e2, d2, t2)) => {
                let e1_match = if let ElementType::Generic(ref name) = e1 {
                    if let Some(existing) = mapping.get(name) {
                        existing == &Type::Scalar(e2.clone())
                    } else {
                        mapping.insert(name.clone(), Type::Scalar(e2.clone()));
                        true
                    }
                } else {
                    e1 == e2
                };
                if !e1_match || d1.len() != d2.len() || t1 != t2 {
                    return false;
                }
                for (dim1, dim2) in d1.iter().zip(d2.iter()) {
                    if let Expr::Identifier(id) = dim1 {
                        if let Expr::Number(n) = dim2 {
                            mapping.insert(id.name.clone(), Type::Generic(n.value.clone(), None));
                        } else if let Expr::Identifier(id2) = dim2 {
                            mapping.insert(id.name.clone(), Type::Generic(id2.name.clone(), None));
                        } else if dim1 != dim2 {
                            return false;
                        }
                    } else if dim1 != dim2 {
                        return false;
                    }
                }
                true
            }
            (Type::Pointer(t1, m1, mut1), Type::Pointer(t2, m2, mut2)) => {
                m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping)
            }
            (
                Type::Borrow {
                    inner: t1,
                    mem_space: m1,
                    is_mut: mut1,
                    ..
                },
                Type::Borrow {
                    inner: t2,
                    mem_space: m2,
                    is_mut: mut2,
                    ..
                },
            ) => m1 == m2 && mut1 == mut2 && self.unify_types_internal(t1, t2, mapping),
            (Type::Ref(t1, m1), Type::Ref(t2, m2)) => {
                m1 == m2 && self.unify_types_internal(t1, t2, mapping)
            }
            (Type::GenericInstance(b1, args1), Type::GenericInstance(b2, args2)) => {
                if args1.len() != args2.len() {
                    return false;
                }
                if !self.unify_types_internal(b1, b2, mapping) {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !self.unify_types_internal(a1, a2, mapping) {
                        return false;
                    }
                }
                true
            }
            (Type::Function(p1, r1), Type::Function(p2, r2)) => {
                if p1.len() != p2.len() {
                    return false;
                }
                if !self.unify_types_internal(r1, r2, mapping) {
                    return false;
                }
                for (a1, a2) in p1.iter().zip(p2.iter()) {
                    if !self.unify_types_internal(a1, a2, mapping) {
                        return false;
                    }
                }
                true
            }
            // A closure literal checks to `Struct("Closure_N")`, erasing its call signature.
            // Matching it against a `ClosureK<Args.., Ret>` parameter (e.g. `.map`'s
            // `Closure1<T, NewItem>`) recovers the args/ret from `closure_signatures` and unifies
            // them into the mapping so return-type generics like `NewItem` get bound. The generic
            // instance carries `[Args.., Ret]`; the recorded signature carries `(Args.., Ret)`.
            (Type::GenericInstance(inner, gi_args), Type::Struct(cn, _))
                if cn.starts_with("Closure_")
                    && matches!(&**inner, Type::Struct(n, _) if n.starts_with("Closure")) =>
            {
                if let Some((params, ret)) = self.closure_signatures.get(cn).cloned() {
                    if gi_args.len() != params.len() + 1 {
                        return false;
                    }
                    for (a, p) in gi_args.iter().zip(params.iter()) {
                        if !self.unify_types_internal(a, p, mapping) {
                            return false;
                        }
                    }
                    self.unify_types_internal(&gi_args[gi_args.len() - 1], &ret, mapping)
                } else {
                    false
                }
            }
            (Type::Struct(n1, _), Type::Struct(n2, _)) => n1 == n2,
            (Type::Pinned(t1, top1), Type::Pinned(t2, top2)) => {
                // A topology variable (`Pinned<_, D>` with D a `<D: Topology>` param) binds
                // to the argument's concrete topology; a concrete topology must match. Then
                // unify the payload (which may itself carry type variables).
                let tops_ok = match top1 {
                    Topology::Custom(name) if self.pending_topo_vars.contains(name) => {
                        self.pending_topo_bindings
                            .insert(name.clone(), top2.clone());
                        true
                    }
                    _ => top1 == top2,
                };
                tops_ok && self.unify_types_internal(t1, t2, mapping)
            }
            (t1, t2) => t1 == t2,
        }
    }

    /// Replace a topology variable (`Custom(name)` with `name` in `topo_mapping`) by its
    /// bound concrete topology; leave everything else unchanged.
    fn substitute_topology(
        top: Topology,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Topology {
        if let Topology::Custom(name) = &top {
            if let Some(bound) = topo_mapping.get(name) {
                return bound.clone();
            }
        }
        top
    }

    /// Apply `substitute_topology` to the topology component of every located type
    /// (`Pinned`/`Ref`) inside `ty`.
    fn substitute_topology_in_type(
        ty: Type,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Type {
        match ty {
            Type::Pinned(inner, top) => Type::Pinned(
                Box::new(Self::substitute_topology_in_type(*inner, topo_mapping)),
                Self::substitute_topology(top, topo_mapping),
            ),
            Type::Ref(inner, mem) => Type::Ref(
                Box::new(Self::substitute_topology_in_type(*inner, topo_mapping)),
                mem,
            ),
            other => other,
        }
    }

    /// In-place topology-variable substitution over a body statement (and its sub-
    /// expressions): specializes an explicit `spawn on(D)` and `Pinned<_, D>` annotation
    /// *inside* a generic body. The signature is handled separately in `instantiate_function`.
    fn subst_topo_in_stmt(
        stmt: &mut crate::syntax::Statement,
        tm: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) {
        use crate::syntax::Statement as S;
        match stmt {
            S::LetDecl(l) => {
                if let Some(ty) = l.ty_ann.take() {
                    l.ty_ann = Some(Self::substitute_topology_in_type(ty, tm));
                }
                Self::subst_topo_in_expr(&mut l.expr, tm);
            }
            S::ExprStmt(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            S::Return(r) => Self::subst_topo_in_expr(&mut r.expr, tm),
            S::Assign(a) => {
                Self::subst_topo_in_expr(&mut a.lhs, tm);
                Self::subst_topo_in_expr(&mut a.rhs, tm);
            }
            S::CompoundAssign(a) => {
                Self::subst_topo_in_expr(&mut a.lhs, tm);
                Self::subst_topo_in_expr(&mut a.rhs, tm);
            }
            S::ForLoop(f) => {
                Self::subst_topo_in_expr(&mut f.iterable, tm);
                for s in &mut f.body {
                    Self::subst_topo_in_stmt(s, tm);
                }
            }
            S::Loop(l) => {
                for s in &mut l.body {
                    Self::subst_topo_in_stmt(s, tm);
                }
            }
            _ => {}
        }
    }

    fn subst_topo_in_expr(
        expr: &mut Expr,
        tm: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) {
        use crate::syntax::Expr as E;
        let recur_block = |stmts: &mut Vec<crate::syntax::Statement>| {
            for s in stmts {
                Self::subst_topo_in_stmt(s, tm);
            }
        };
        match expr {
            E::SpawnOn(e) => {
                e.top = Self::substitute_topology(e.top.clone(), tm);
                recur_block(&mut e.stmts);
                if let Some(r) = e.ret.as_deref_mut() {
                    Self::subst_topo_in_expr(r, tm);
                }
            }
            E::If(e) => {
                Self::subst_topo_in_expr(&mut e.cond, tm);
                recur_block(&mut e.then_block);
                if let Some(eb) = &mut e.else_block {
                    recur_block(eb);
                }
            }
            E::UnsafeBlock(e) => {
                recur_block(&mut e.stmts);
                if let Some(r) = e.ret.as_deref_mut() {
                    Self::subst_topo_in_expr(r, tm);
                }
            }
            E::ComptimeBlock(e) => {
                recur_block(&mut e.stmts);
                if let Some(r) = e.ret.as_deref_mut() {
                    Self::subst_topo_in_expr(r, tm);
                }
            }
            E::BinaryOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::RelationalOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::LogicalOp(e) => {
                Self::subst_topo_in_expr(&mut e.lhs, tm);
                Self::subst_topo_in_expr(&mut e.rhs, tm);
            }
            E::UnaryOp(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::Dereference(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::Borrow(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::AsCast(e) => {
                e.target_ty = Self::substitute_topology_in_type(e.target_ty.clone(), tm);
                Self::subst_topo_in_expr(&mut e.expr, tm);
            }
            E::Transfer(e) => Self::subst_topo_in_expr(&mut e.expr, tm),
            E::TransferPredicate(e) => {
                e.from = Self::substitute_topology(e.from.clone(), tm);
                e.to = Self::substitute_topology(e.to.clone(), tm);
            }
            E::IndexAccess(e) => {
                Self::subst_topo_in_expr(&mut e.base, tm);
                Self::subst_topo_in_expr(&mut e.index, tm);
            }
            E::MemberAccess(e) => Self::subst_topo_in_expr(&mut e.base, tm),
            E::FunctionCall(e) => {
                for a in &mut e.args {
                    Self::subst_topo_in_expr(a, tm);
                }
            }
            E::MethodCall(e) => {
                Self::subst_topo_in_expr(&mut e.base, tm);
                for a in &mut e.args {
                    Self::subst_topo_in_expr(a, tm);
                }
            }
            _ => {}
        }
    }

    pub fn instantiate_function(
        &mut self,
        generic_func: &Function,
        mapping: &std::collections::HashMap<crate::symbol::Symbol, Type>,
        topo_mapping: &std::collections::HashMap<crate::symbol::Symbol, Topology>,
    ) -> Function {
        let mut mangled_name = generic_func.name.to_string();
        let mut sorted_keys: Vec<&crate::symbol::Symbol> = mapping.keys().collect();
        sorted_keys.sort();
        for g_name in sorted_keys {
            if let Some(ty) = mapping.get(g_name) {
                mangled_name.push_str(&format!("${}", ty.mangle()));
            }
        }
        // Distinguish topology instantiations (`f$GPU` vs `f$NPU`).
        let mut topo_keys: Vec<&crate::symbol::Symbol> = topo_mapping.keys().collect();
        topo_keys.sort();
        for g_name in topo_keys {
            if let Some(top) = topo_mapping.get(g_name) {
                mangled_name.push_str(&format!("${:?}", top.kind()));
            }
        }

        let new_params = generic_func
            .params
            .iter()
            .map(|(n, t)| {
                let substituted =
                    Self::substitute_topology_in_type(t.substitute(mapping), topo_mapping);
                (n.clone(), substituted)
            })
            .collect();
        let new_ret = Self::substitute_topology_in_type(
            generic_func.return_type.substitute(mapping),
            topo_mapping,
        );

        let new_body = generic_func
            .body
            .iter()
            .map(|s| {
                let mut s = s.substitute(mapping);
                if !topo_mapping.is_empty() {
                    Self::subst_topo_in_stmt(&mut s, topo_mapping);
                }
                s
            })
            .collect();

        Function {
            name: mangled_name.into(),
            generics: Vec::new(),
            params: new_params,
            topology: Self::substitute_topology(generic_func.topology.clone(), topo_mapping),
            return_type: new_ret,
            requires: generic_func
                .requires
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
            ensures: generic_func
                .ensures
                .iter()
                .map(|e| e.substitute(mapping))
                .collect(),
            // Constraints were discharged at the call site before instantiation.
            where_transfers: Vec::new(),
            body: new_body,
            doc_comment: generic_func.doc_comment.clone(),
        }
    }

    pub fn mangle_path(path: &str) -> String {
        path.replace("/", "_").replace(".", "_")
    }

    pub fn check_function(&mut self, func: &mut Function) {
        if !func.generics.is_empty() {
            return;
        }

        // `main` is the C entry point: its return value is the process exit code,
        // so it must return i32 (#210). A non-i32 `main` is a hard error rather
        // than a silently-discarded value.
        if func.name.as_ref() == "main" && func.return_type != Type::Scalar(ElementType::I32) {
            self.errors.push(format!(
                "`main` must return i32 (the process exit code), found {:?}",
                func.return_type
            ));
        }

        let prev_constraints = self.constraints.clone();
        let prev_ret_ty = self.current_return_type.clone();
        self.current_return_type = Some(func.return_type.clone());
        self.push_scope();

        let prev_top = self.active_topology.clone();
        let prev_mem = self.active_memory.clone();
        self.active_topology = func.topology.clone();
        self.active_memory = self
            .transfer_cost_graph
            .default_memory_for(&self.active_topology);

        // Record the parameter set and clear per-function provenance state for the
        // return-escape analysis (#243). Saved/restored so nested checks (closures) don't
        // clobber the enclosing function's view.
        let prev_params = std::mem::take(&mut self.borrow.current_params);
        let prev_provenance = std::mem::take(&mut self.borrow.ref_provenance);
        for (name, ty) in &func.params {
            self.insert(name.to_string(), ty.clone());
            self.borrow.current_params.insert(name.clone(), ty.clone());
        }
        // Inside a transfer lowering, the raw:: primitives may only touch tiles the
        // transfer was given -- record the parameter names they are allowed to name.
        if self.transfer_lowering_edge.is_some() {
            self.transfer_lowering_params = func.params.iter().map(|(n, _)| n.clone()).collect();
        }

        // Add preconditions (requires) to our constraints
        for req in &func.requires {
            self.constraints.push(req.clone());
        }

        // Pre-scan: gather the value contracts the consumer's asserts impose, so a
        // transfer seam (checked before the consumer's spawn body) can consult them.
        // Only when seam verification is enabled (otherwise we pay nothing for it).
        let prev_contracts = std::mem::take(&mut self.seam_contracts);
        if self.verify_seams {
            Self::collect_assert_contracts(&func.body, &mut self.seam_contracts);
        }

        self.check_block(&mut func.body, &func.return_type.clone());

        self.seam_contracts = prev_contracts;

        // Combine return constraints into a single OR constraint
        if !self.return_constraints.is_empty() {
            let mut combined = self.return_constraints[0].clone();
            for rc in self.return_constraints.iter().skip(1) {
                combined = Expr::LogicalOp(LogicalOpExpr {
                    lhs: Box::new(combined),
                    op: LogicalOp::Or,
                    rhs: Box::new(rc.clone()),
                    span: crate::syntax::Span::default(),
                });
            }
            self.constraints.push(combined);
            self.return_constraints.clear();
        }

        // Verify postconditions (ensures)
        for ens in &func.ensures {
            if !self.prove_expr(ens) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8001,
                    format!(
                        "Function '{}' cannot prove postcondition (ensures) at compile time",
                        func.name
                    ),
                    None,
                );
            }
        }

        // W1009: Unused function parameters
        for (param_name, _) in &func.params {
            let name_str: &str = param_name.as_ref();
            if !name_str.starts_with('_')
                && name_str != "self"
                && !self.used_vars.contains(param_name.as_ref())
            {
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1009,
                    format!("Unused function parameter '{}'", param_name),
                    None,
                );
            }
        }

        // W1001: Unused variable bindings
        for (var_name, var_span) in std::mem::take(&mut self.declared_vars) {
            let name_str: &str = var_name.as_ref();
            if !name_str.starts_with('_') && !self.used_vars.contains(&var_name) {
                let diag = self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1001,
                    format!("Unused variable '{}'", var_name),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&var_span)),
                );
                diag.fix_its.push(crate::diagnostic::FixIt {
                    message: "prefix with underscore to suppress".into(),
                    span: crate::diagnostic::SourceSpan::from_ast_span(&var_span),
                    replacement: format!("_{}", var_name).into(),
                });
            }
        }

        // The whole function has been checked; verify the working set of each memory space
        // it places tiles into fits (or is `overcommit`). Clears the per-function placement map.
        self.check_cumulative_capacity();

        self.pop_scope();
        self.current_return_type = prev_ret_ty;
        self.constraints = prev_constraints;
        self.active_topology = prev_top;
        self.active_memory = prev_mem;
        self.borrow.current_params = prev_params;
        self.borrow.ref_provenance = prev_provenance;
        self.used_vars.clear();
    }

    pub fn parse_ty_str(&self, s: &str) -> Type {
        let mut lexer = crate::lexer::Lexer::new(s);
        let tokens = lexer.tokenize();

        let mut parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(ty) = parser.parse_type() {
            if parser.check(&crate::lexer::TokenType::Eof) {
                return self.resolve_parsed_type(ty);
            }
        }

        let mut expr_parser = crate::parser::Parser::new(&tokens, s);
        if let Ok(expr) = expr_parser.parse_primary_expr() {
            return Type::Const(Box::new(expr));
        }

        Type::Unknown
    }

    fn resolve_parsed_type(&self, ty: Type) -> Type {
        match ty {
            Type::Struct(name, id) => {
                // A nominal type name is a struct *or* an enum (both are represented
                // as `Type::Struct` nominally). Only demote to a generic type
                // parameter when the name matches neither — otherwise an enum type
                // argument like `Option<i32>` keeps a `Generic("Option")` head and
                // fails to unify against `impl<T> Option<T>` during method resolution.
                if self.env.structs.contains_key(&name)
                    || self.env.enums.contains_key(&name)
                    || self.generated_structs.iter().any(|s| s.name == name)
                {
                    Type::Struct(name, id)
                } else {
                    Type::Generic(name, id)
                }
            }
            Type::GenericInstance(base, args) => {
                let resolved_base = Box::new(self.resolve_parsed_type(*base));
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::GenericInstance(resolved_base, resolved_args)
            }
            Type::Pointer(inner, mem, mut_flag) => {
                Type::Pointer(Box::new(self.resolve_parsed_type(*inner)), mem, mut_flag)
            }
            Type::Ref(inner, mem) => Type::Ref(Box::new(self.resolve_parsed_type(*inner)), mem),
            Type::Borrow {
                inner,
                mem_space: mem,
                is_mut: mut_flag,
                region_id: r,
            } => Type::Borrow {
                inner: Box::new(self.resolve_parsed_type(*inner)),
                mem_space: mem.clone(),
                is_mut: mut_flag,
                region_id: r,
            },
            Type::Pinned(inner, top) => {
                Type::Pinned(Box::new(self.resolve_parsed_type(*inner)), top)
            }
            Type::Function(args, ret) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Function(resolved_args, Box::new(self.resolve_parsed_type(*ret)))
            }
            Type::Closure(args, ret) => {
                let resolved_args = args
                    .into_iter()
                    .map(|a| self.resolve_parsed_type(a))
                    .collect();
                Type::Closure(resolved_args, Box::new(self.resolve_parsed_type(*ret)))
            }
            _ => ty,
        }
    }
}
