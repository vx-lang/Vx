//! The type checker's per-analysis state, grouped by the analysis that owns it.
//!
//! `TypeChecker` accumulated a field per piece of state until it carried 42 of them, which is
//! why every new analysis landed there: there was nowhere else for it to go. These four structs
//! are that state sorted into the analyses it belongs to, so a new analysis brings its own struct
//! rather than four more fields.
//!
//! The pattern is [`crate::hir::borrow_cx::BorrowCx`]'s, which did the same for borrow checking.
//! Unlike `BorrowCx` these are plain state with no invariant to protect, so their fields stay
//! reachable; when one grows an invariant, it should grow methods and hide the field the way
//! `BorrowCx` hides `active_borrows`.
//!
//! All four are per-worker, owned by value inside `TypeChecker` -- no shared state and no lock,
//! so they compose with the parallel pipeline's phase isolation.

use crate::symbol::Symbol;
use crate::syntax::{Expr, MemorySpace, StructDecl, Topology, Type};
use crate::{hir::env::Value, syntax::Function};
use std::collections::{HashMap, HashSet};

/// Compile-time evaluation: the constant environment and the constraints gathered from it.
/// One tile placed in a memory space: what it costs, where it lives, and when it happened.
#[derive(Debug, Clone)]
pub struct Placement {
    pub bytes: u64,
    /// The blocks open when it was placed, outermost first. A tile is live for exactly the
    /// program points inside its innermost block.
    pub scope: Vec<u32>,
    /// Position in program order, used to tell a tile placed before a sibling block from one
    /// placed after it has closed.
    pub order: usize,
}

pub struct ConstEvalState {
    /// Scoped constant bindings, innermost last.
    pub env: Vec<HashMap<Symbol, Value>>,
    /// Constraints collected while checking the current function.
    pub constraints: Vec<Expr>,
    /// Constraints a `return` must satisfy.
    pub return_constraints: Vec<Expr>,
}

impl Default for ConstEvalState {
    fn default() -> Self {
        Self {
            // One scope open from the start, matching `TypeChecker::scopes`. Deriving `Default`
            // would start with none, and the first `let` would have nowhere to bind.
            env: vec![HashMap::new()],
            constraints: Vec::new(),
            return_constraints: Vec::new(),
        }
    }
}

/// Monomorphization: the instantiations produced while checking, and the deduction state that
/// produces them.
#[derive(Default)]
pub struct MonoState {
    /// Concrete instantiations produced during checking, with the hash of the arguments.
    pub functions: Vec<(Function, u64)>,
    /// Structs synthesized during checking (closure environments, instantiated generics).
    pub generated_structs: Vec<StructDecl>,
    /// Topology generic parameters (`<D: Topology>`) of the function being instantiated. Set
    /// around the deduction unify at a generic call so `unify_types` can bind a `Pinned<_, D>`
    /// param's topology variable instead of demanding equality.
    pub pending_topo_vars: HashSet<Symbol>,
    /// Topology bindings (`D -> concrete topology`) deduced during that unify, consumed by
    /// `instantiate_function` to specialize `on D` and `Pinned<_, D>`.
    pub pending_topo_bindings: HashMap<Symbol, Topology>,
    /// Signature of each closure literal, keyed by its generated `Closure_N` struct name:
    /// `(param types, return type)`. A closure expression checks to `Struct("Closure_N")`, which
    /// erases the call signature, so unifying it against a `ClosureK<Args.., Ret>` parameter needs
    /// this side table to recover the args/ret and bind the method's generics.
    pub closure_signatures: HashMap<Symbol, (Vec<Type>, Type)>,
    /// Nesting depth of each open closure literal, innermost last.
    pub closure_depths: Vec<usize>,
    /// Variables each open closure literal captured, innermost last.
    pub closure_captures_stack: Vec<HashMap<Symbol, Type>>,
}

/// The memory algebra's seam obligations: the solver that discharges them, the facts they are
/// checked against, and the cost of doing so.
#[derive(Default)]
pub struct SeamState {
    /// Whether to discharge per-seam boundary obligations (the assert pre-scan and the z3
    /// checks). Off by default so ordinary compilation pays nothing and needs no solver;
    /// enabled with `vxc --verify-seams`.
    pub verify: bool,
    /// Persistent z3 process, lazily started on the first seam and reused for all of them so the
    /// marginal per-seam cost is solving time, not process startup.
    pub solver: Option<crate::hir::seam::Solver>,
    /// Per-function pre-scan of `assert(var == const)` facts: the value a consumer requires of
    /// `var`. Populated before statements are checked so a transfer seam (checked before the
    /// consumer's `spawn` body) can consult the downstream contract on the buffer it produces.
    pub contracts: HashMap<String, u64>,
    /// Number of per-seam obligations discharged (eval metric M1).
    pub checks: usize,
    /// Total marginal solving time across all seams, excluding the one-time solver startup.
    pub check_time: std::time::Duration,
    /// One-time cost of spawning the persistent solver and installing its preamble, paid once per
    /// compilation regardless of program size.
    pub init_time: std::time::Duration,
    /// Set immediately before a transfer is checked to mark it a *relaxed* (escape-hatch)
    /// transfer that does not carry a synchronizing release/DMA-completion. Consumed and reset by
    /// `check_transfer_expr`.
    pub pending_relaxed: bool,
    /// The edge and machine of the `impl Transfer` lowering whose body is being checked, when one
    /// is. `Some((from, to, machine))` is what makes the eight `raw::` primitives resolve and
    /// gives their space and capability obligations an edge to check against. `None` everywhere
    /// else -- which is the floor property: outside a lowering, `raw::` names do not exist.
    ///
    /// The machine (the lowering's `for Topology::X`, as its display name) is here because a
    /// capability check that knows only the edge pair cannot tell WHOSE edge: `raw::async_copy`
    /// asked "does any topology's L2 -> SMEM carry copy_engine", so one machine declaring the
    /// engine armed every machine's lowerings for the like-named edge.
    pub lowering_edge: Option<(MemorySpace, MemorySpace, String)>,
    /// The current lowering method's parameter names: the only tiles `raw::` may touch. A local
    /// alias would escape the async-discipline walk (it is name-keyed), so the primitives are
    /// limited to the names the walk can see. Only read while `lowering_edge` is `Some`.
    pub lowering_params: HashSet<Symbol>,
}

/// What an admitted program implies about traffic and capacity: the routes it takes, what it
/// keeps resident, and what it places where.
#[derive(Default)]
pub struct TrafficState {
    /// Every staging route this compilation resolved, in source order: the hops a `transfer`
    /// lowered to and their per-edge costs. An *admitted* program emits no diagnostics, so this
    /// is what `--diagnostics-json` reports for it -- the accept side of the admission verdict.
    pub staging_routes: Vec<crate::report::StagingRoute>,
    /// Per-function, per-space working sets -- the resident sets an admitted program implies.
    pub resident_sets: Vec<crate::report::ResidentSet>,
    /// What each `spawn` region moves, counted from its own code. One entry per `spawn on(...)`
    /// site, in source order.
    pub spawn_regions: Vec<crate::report::SpawnRegionTraffic>,
    /// Per-function working set: `memory space -> {buffer key -> granule-rounded bytes}`. Every
    /// tile placed in a declared space (via `transfer`/`Ref` annotation) is recorded here so the
    /// *cumulative* budget check can sum them and flag a space whose total exceeds `capacity`.
    /// Reset per function.
    /// Each entry is the tile's granule-rounded bytes, the chain of blocks it was placed
    /// inside, and its position in program order.
    ///
    /// The chain is what makes this a peak rather than a sum. A placed tile is released at the
    /// end of the block that placed it, so two tiles in sibling blocks never occupy the space at
    /// the same time, and adding them together describes a program that was never run.
    pub memory_placements: HashMap<MemorySpace, HashMap<String, Placement>>,
    /// Monotonic id for placements with no binding name, so they still count toward the sum.
    pub placement_site: usize,
    /// The *releasing* blocks currently open, outermost first.
    ///
    /// Only scopes the lowering actually frees at the end of are on this chain. That is
    /// narrower than "lexical scope": the release goes to the function's exits whenever the
    /// defining block dominates them, so an unconditionally-entered scope -- a `spawn` region,
    /// whose block ends in an unconditional branch -- holds its tiles until the function
    /// returns. Treating one of those as releasing admits a program that then exceeds the space.
    pub scope_chain: Vec<u32>,
    /// Whether each open scope put an id on `scope_chain`, so `pop_scope` can undo exactly
    /// what its `push` did.
    pub scope_releases: Vec<bool>,
    /// Ids for the blocks in `scope_chain`; a block gets a fresh one each time it is entered.
    pub next_scope_id: u32,
    /// Program order for placements, so "already placed when this one happened" is answerable.
    pub placement_order: usize,
    /// Whether the placement being checked is a hop feeding another transfer.
    ///
    /// A staged route is rewritten into a chain, and each inner hop's tile is read by the hop
    /// above it -- so it is never one of the tiles nobody reads, whatever the binding the whole
    /// chain is eventually bound to does. Attributing the binding's readership to the transit
    /// copy said two of them never coexisted, when the emitted code holds both to the end of the
    /// block.
    pub consumed_by_transfer: bool,
    /// The calls the current function makes, keyed by call-site span (a probe may check the
    /// same expression twice; the first visit's program order wins). Taken, with the
    /// placements, when the function's summary is built.
    pub call_sites: HashMap<String, CallSite>,
    /// One finished capacity summary per checked function, in check order -- what the
    /// cross-call fold reads after every body is done.
    pub capacity_summaries: Vec<crate::hir::check::capacity_fold::FnCapacitySummary>,
}

/// A resolved call, recorded where the checker resolves it: who is called, from inside which
/// open blocks, and where it sits in program order relative to the placements around it.
pub struct CallSite {
    pub callee: String,
    pub scope: Vec<u32>,
    pub order: usize,
    pub span: crate::syntax::Span,
}
