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
    /// How many calls deep the evaluator currently is. A `Cell` because evaluation runs
    /// behind `&self`, and the count has to rise and fall as it descends.
    pub call_depth: std::cell::Cell<u32>,
    /// Set when evaluation stopped because it ran past `MAX_CALL_DEPTH`. Whoever asked for
    /// the value reports it, since the evaluator itself cannot reach the diagnostics.
    pub depth_exceeded: std::cell::Cell<bool>,
    /// Set when a called body held a statement the evaluator cannot run. The call then has
    /// no value, rather than whatever the statements it *could* run happened to leave behind.
    pub unsupported_stmt: std::cell::Cell<bool>,
    /// How many loop iterations the current evaluation has run. A `Cell` because evaluation
    /// runs behind `&self`, and the count has to rise as the loops turn.
    pub loop_steps: std::cell::Cell<u64>,
    /// Set when evaluation stopped because it ran past `MAX_LOOP_STEPS`. Whoever asked for
    /// the value reports it, since the evaluator itself cannot reach the diagnostics.
    pub steps_exceeded: std::cell::Cell<bool>,
    /// How many closure bodies are open around the statement being checked.
    ///
    /// A `comptime` block that is a closure's body is that closure's body, not a request to
    /// compute something now: `| y | comptime { y + 1 }` cannot fold, because `y` is not
    /// known until someone calls it. It is a `constexpr` function, and what has to fold is
    /// the call. An uncalled one is just an unused variable.
    pub closure_body_depth: u32,
    /// How many `comptime` blocks are open around the statement being checked.
    ///
    /// Loops are run only inside one. A run-time loop has nothing to gain from being run at
    /// compile time -- its result is not wanted -- and running it would spend the compiler's
    /// time walking a trip count that belongs to the program. What still happens everywhere
    /// is the settling up: see `settle_loop_consteval`.
    pub comptime_depth: u32,
}

/// Provenance carried by a value while a `comptime` block is interpreted.
///
/// These facts deliberately live beside, rather than inside, [`Value`]. `Value` is the compact
/// representation of a concrete constant used throughout the checker; it cannot represent every
/// reference, aggregate, or callable that can transport an outer mutable reference.
#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct ComptimeValueFacts {
    /// Outer bindings a mutable reference in this value may reach.
    pub reference_origins: HashSet<Symbol>,
    /// Outer storage represented by a generated closure-environment field. Unlike
    /// `reference_origins`, this makes assigning the local binding created from `_env.field` an
    /// escaping write; ordinary local reference aliases do not populate it.
    pub captured_place_origins: HashSet<Symbol>,
    /// Concrete functions this value may invoke when used as a callable.
    pub callable_targets: HashSet<Symbol>,
    /// Captured-environment facts for generated closure call targets. The target's generated
    /// `Closure_N_call` body receives this as its synthetic first `_env` parameter when invoked.
    pub callable_environments: HashMap<Symbol, ComptimeValueFacts>,
    /// Outer bindings a closure held by this value may write when invoked.
    pub captured_writes: HashSet<Symbol>,
    /// The callable's body is unavailable, so a call through it must be conservative.
    pub unknown_callable: bool,
}

impl ComptimeValueFacts {
    /// Join facts from alternative values or paths. Every field is a may-fact, so union is the
    /// sound merge and an unknown callable stays unknown.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.reference_origins
            .extend(other.reference_origins.iter().cloned());
        self.captured_place_origins
            .extend(other.captured_place_origins.iter().cloned());
        self.callable_targets
            .extend(other.callable_targets.iter().cloned());
        for (target, facts) in &other.callable_environments {
            self.callable_environments
                .entry(target.clone())
                .or_default()
                .merge_from(facts);
        }
        self.captured_writes
            .extend(other.captured_writes.iter().cloned());
        self.unknown_callable |= other.unknown_callable;
    }
}

/// The value part of a comptime interpretation result.
///
/// `concrete == None` means the expression was modelled but did not yield a constant. It says
/// nothing about support or effects; those are independent parts of [`ComptimeEvalOutcome`] and
/// [`ComptimeEvalContext`].
/// Component values retained beside the legacy scalar/array [`Value`]. They let the transition
/// interpreter model structs and partially-known arrays without assigning every field or element
/// the union of its siblings' mutable-reference and callable facts.
#[derive(Clone, PartialEq)]
pub(crate) enum ComptimeAggregateValue {
    Array(Vec<ComptimeEvalValue>),
    Struct(HashMap<Symbol, ComptimeEvalValue>),
}

#[derive(Clone, Default, PartialEq)]
pub(crate) struct ComptimeEvalValue {
    pub concrete: Option<Value>,
    pub facts: ComptimeValueFacts,
    pub aggregate: Option<ComptimeAggregateValue>,
}

impl ComptimeEvalValue {
    pub(crate) fn known(value: Value) -> Self {
        Self {
            concrete: Some(value),
            facts: ComptimeValueFacts::default(),
            aggregate: None,
        }
    }

    /// Join values from alternative control-flow paths. A concrete result survives only when both
    /// paths proved the same concrete value; facts always merge as may-facts.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        if self.concrete != other.concrete {
            self.concrete = None;
        }
        let aggregates_match = match (&mut self.aggregate, &other.aggregate) {
            (
                Some(ComptimeAggregateValue::Array(left)),
                Some(ComptimeAggregateValue::Array(right)),
            ) if left.len() == right.len() => {
                for (left, right) in left.iter_mut().zip(right) {
                    left.merge_from(right);
                }
                true
            }
            (
                Some(ComptimeAggregateValue::Struct(left)),
                Some(ComptimeAggregateValue::Struct(right)),
            ) if left.len() == right.len()
                && left.keys().all(|field| right.contains_key(field)) =>
            {
                for (field, left) in left {
                    // The key-set equality above makes this lookup infallible.
                    left.merge_from(right.get(field).expect("matching aggregate field"));
                }
                true
            }
            _ => false,
        };
        if !aggregates_match {
            self.aggregate = None;
        }
        self.facts.merge_from(&other.facts);
    }
}

/// Whether the interpreter has a sound rule for every operation it reached.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ComptimeEvalSupport {
    #[default]
    Supported,
    Unsupported,
}

/// Why an otherwise unsupported shadow interpretation is known to differ from the legacy
/// evaluator. The transition comparator may allow only reasons explicitly backed by fixtures;
/// every other unsupported result remains an unclassified disagreement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ComptimeEvalUnsupportedReason {
    #[default]
    None,
    /// An owner without a recorded transition allowance.
    Unclassified,
}

impl ComptimeEvalUnsupportedReason {
    pub(crate) fn merge_from(&mut self, other: Self) {
        use ComptimeEvalUnsupportedReason::*;
        *self = match (*self, other) {
            (None, reason) => reason,
            (reason, None) => reason,
            _ => Unclassified,
        };
    }
}

impl ComptimeEvalSupport {
    pub(crate) fn merge_from(&mut self, other: Self) {
        if matches!(other, Self::Unsupported) {
            *self = Self::Unsupported;
        }
    }

    pub(crate) fn is_supported(self) -> bool {
        matches!(self, Self::Supported)
    }
}

/// Control flow produced while interpreting a comptime statement or block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum ComptimeEvalFlow {
    #[default]
    Normal,
    Return,
    Break,
    Continue,
    /// Viable abstract paths leave the construct differently. The caller must not continue as
    /// though it had proved any one of the concrete flows.
    Indeterminate,
}

/// One expression, statement, or block result from the unified comptime interpreter.
///
/// This keeps concrete-value knowledge, support, and control flow separate. Possible escaping
/// writes belong to the context because branches must be able to clone and merge that state.
#[derive(Clone, Default, PartialEq)]
pub(crate) struct ComptimeEvalOutcome {
    pub value: ComptimeEvalValue,
    pub support: ComptimeEvalSupport,
    pub unsupported_reason: ComptimeEvalUnsupportedReason,
    pub flow: ComptimeEvalFlow,
    /// A deliberately promoted fail-closed owner. During the transition, most unsupported
    /// results remain observations so they can be compared with the legacy evaluator. This bit
    /// is reserved for the small, fixture-backed set whose effect cannot disappear even when it
    /// only touches comptime-local storage.
    pub requires_refusal: bool,
}

impl ComptimeEvalOutcome {
    pub(crate) fn known(value: Value) -> Self {
        Self {
            value: ComptimeEvalValue::known(value),
            ..Self::default()
        }
    }

    pub(crate) fn unknown() -> Self {
        Self::default()
    }

    pub(crate) fn unsupported() -> Self {
        Self {
            support: ComptimeEvalSupport::Unsupported,
            unsupported_reason: ComptimeEvalUnsupportedReason::Unclassified,
            ..Self::default()
        }
    }

    pub(crate) fn mark_unsupported(&mut self) {
        self.support = ComptimeEvalSupport::Unsupported;
        self.unsupported_reason = ComptimeEvalUnsupportedReason::Unclassified;
    }

    pub(crate) fn refusal() -> Self {
        Self {
            support: ComptimeEvalSupport::Unsupported,
            unsupported_reason: ComptimeEvalUnsupportedReason::Unclassified,
            requires_refusal: true,
            ..Self::default()
        }
    }

    /// Join results from viable abstract paths. Differing flow cannot produce a definite normal
    /// result, so the caller receives `Indeterminate` rather than continuing after the branch.
    pub(crate) fn merge_from(&mut self, other: &Self) {
        self.value.merge_from(&other.value);
        self.support.merge_from(other.support);
        self.unsupported_reason.merge_from(other.unsupported_reason);
        self.requires_refusal |= other.requires_refusal;
        if self.flow != other.flow {
            self.flow = ComptimeEvalFlow::Indeterminate;
            self.value.concrete = None;
        }
    }
}

/// Per-`comptime`-block state for the unified concrete/abstract interpreter.
///
/// It replaces the old pattern of an evaluator plus a separately invoked effect walk. Its lexical
/// scopes carry value facts, while its immutable outer-boundary sets make it possible to identify
/// writes that would disappear when the comptime block is removed.
#[derive(Clone)]
pub(crate) struct ComptimeEvalContext {
    outer_bindings: HashSet<Symbol>,
    outer_reference_bindings: HashSet<Symbol>,
    outer_callable_bindings: HashSet<Symbol>,
    local_scopes: Vec<HashMap<Symbol, ComptimeValueFacts>>,
    analysis_call_depth: u32,
    call_depth_exceeded: bool,
    escaping_writes: HashSet<Symbol>,
    /// Loop iterations run so far in this block, by every loop together, so nested loops share
    /// one `MAX_LOOP_STEPS` budget.
    loop_steps: u64,
}

impl ComptimeEvalContext {
    pub(crate) fn new(
        outer_bindings: HashSet<Symbol>,
        outer_reference_bindings: HashSet<Symbol>,
        outer_callable_bindings: HashSet<Symbol>,
    ) -> Self {
        Self {
            outer_bindings,
            outer_reference_bindings,
            outer_callable_bindings,
            local_scopes: vec![HashMap::new()],
            analysis_call_depth: 0,
            call_depth_exceeded: false,
            escaping_writes: HashSet::new(),
            loop_steps: 0,
        }
    }

    /// Counts one loop iteration, or returns false when the block has used its whole budget.
    pub(crate) fn take_loop_step(&mut self) -> bool {
        if self.loop_steps >= MAX_LOOP_STEPS {
            return false;
        }
        self.loop_steps += 1;
        true
    }

    pub(crate) fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    pub(crate) fn pop_scope(&mut self) {
        assert!(
            self.local_scopes.len() > 1,
            "the comptime root scope must remain open"
        );
        self.local_scopes.pop();
    }

    pub(crate) fn binding_facts(&self, name: &Symbol) -> Option<&ComptimeValueFacts> {
        self.local_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name))
    }

    pub(crate) fn declare(&mut self, name: Symbol, facts: ComptimeValueFacts) {
        self.local_scopes
            .last_mut()
            .expect("comptime context has a root scope")
            .insert(name, facts);
    }

    /// Replace the nearest lexical binding's facts. A missing binding is outer, and therefore is
    /// intentionally not installed as a local shadow.
    pub(crate) fn reassign(&mut self, name: &Symbol, facts: ComptimeValueFacts) {
        if let Some(scope) = self
            .local_scopes
            .iter_mut()
            .rev()
            .find(|scope| scope.contains_key(name))
        {
            scope.insert(name.clone(), facts);
        }
    }

    pub(crate) fn is_outer_binding(&self, name: &Symbol) -> bool {
        self.outer_bindings.contains(name) && self.binding_facts(name).is_none()
    }

    pub(crate) fn outer_reference_binding(&self, name: &Symbol) -> bool {
        self.outer_reference_bindings.contains(name)
    }

    pub(crate) fn outer_callable_binding(&self, name: &Symbol) -> bool {
        self.outer_callable_bindings.contains(name)
    }

    pub(crate) fn note_escaping_write(&mut self, roots: impl IntoIterator<Item = Symbol>) {
        self.escaping_writes.extend(
            roots
                .into_iter()
                .filter(|root| self.outer_bindings.contains(root)),
        );
    }

    pub(crate) fn escaping_write(&self) -> Option<&Symbol> {
        self.escaping_writes
            .iter()
            .min_by(|left, right| left.as_ref().cmp(right.as_ref()))
    }

    /// Enter one concrete call frame. Recursion is allowed: every frame owns a fresh value
    /// environment and lexical fact scope, while the shared depth limit stops nontermination.
    pub(crate) fn push_call(&mut self) -> bool {
        if self.analysis_call_depth >= MAX_CALL_DEPTH {
            self.call_depth_exceeded = true;
            return false;
        }
        self.analysis_call_depth += 1;
        true
    }

    pub(crate) fn call_depth_exceeded(&self) -> bool {
        self.call_depth_exceeded
    }

    pub(crate) fn pop_call(&mut self) {
        self.analysis_call_depth = self.analysis_call_depth.saturating_sub(1);
    }

    /// Merge an alternative branch into this one. Branch-local declarations must have been popped
    /// before the merge. A binding present on only one path is not available afterwards; a binding
    /// present on both paths keeps the union of its may-facts. Escaping writes always union.
    pub(crate) fn merge_branch(&mut self, branch: &Self) {
        debug_assert_eq!(self.local_scopes.len(), branch.local_scopes.len());
        for (scope, other_scope) in self.local_scopes.iter_mut().zip(&branch.local_scopes) {
            scope.retain(|name, _| other_scope.contains_key(name));
            for (name, facts) in scope.iter_mut() {
                facts.merge_from(
                    other_scope
                        .get(name)
                        .expect("retained comptime binding exists in both branches"),
                );
            }
        }
        self.escaping_writes
            .extend(branch.escaping_writes.iter().cloned());
        self.call_depth_exceeded |= branch.call_depth_exceeded;
        self.loop_steps = self.loop_steps.max(branch.loop_steps);
        debug_assert_eq!(self.analysis_call_depth, branch.analysis_call_depth);
    }
}

impl Default for ComptimeEvalContext {
    fn default() -> Self {
        Self::new(HashSet::new(), HashSet::new(), HashSet::new())
    }
}

#[cfg(test)]
mod comptime_eval_tests {
    use super::*;

    fn symbol(name: &str) -> Symbol {
        name.into()
    }

    #[test]
    fn alternative_values_keep_only_an_agreed_constant_and_union_facts() {
        let outside = symbol("outside");
        let mut left = ComptimeEvalValue::known(Value::Int(7));
        left.facts.reference_origins.insert(outside.clone());
        let mut right = ComptimeEvalValue::known(Value::Int(9));
        right.facts.callable_targets.insert(symbol("touch"));

        left.merge_from(&right);

        assert!(left.concrete.is_none());
        assert!(left.facts.reference_origins.contains(&outside));
        assert!(left.facts.callable_targets.contains("touch"));
    }

    #[test]
    fn branch_merge_keeps_may_facts_and_a_deterministic_outer_write() {
        let outside = symbol("outside");
        let mut left = ComptimeEvalContext::new(
            HashSet::from([outside.clone()]),
            HashSet::new(),
            HashSet::new(),
        );
        left.declare(symbol("alias"), ComptimeValueFacts::default());
        let mut right = left.clone();
        let mut right_facts = ComptimeValueFacts::default();
        right_facts.reference_origins.insert(outside.clone());
        right.reassign(&symbol("alias"), right_facts);
        right.note_escaping_write([outside.clone()]);

        left.merge_branch(&right);

        assert!(left
            .binding_facts(&symbol("alias"))
            .is_some_and(|facts| facts.reference_origins.contains(&outside)));
        assert_eq!(left.escaping_write(), Some(&outside));
    }

    #[test]
    fn incompatible_branch_flow_is_never_treated_as_normal() {
        let mut normal = ComptimeEvalOutcome::known(Value::Int(7));
        let returned = ComptimeEvalOutcome {
            value: ComptimeEvalValue::known(Value::Int(7)),
            flow: ComptimeEvalFlow::Return,
            ..ComptimeEvalOutcome::default()
        };

        normal.merge_from(&returned);

        assert_eq!(normal.flow, ComptimeEvalFlow::Indeterminate);
        assert!(normal.value.concrete.is_none());
    }

    #[test]
    fn call_guard_allows_recursion_and_rejects_the_depth_limit() {
        let mut context = ComptimeEvalContext::default();
        assert!(context.push_call());
        assert!(context.push_call());
        context.pop_call();
        context.pop_call();

        for _ in 0..MAX_CALL_DEPTH {
            assert!(context.push_call());
        }
        assert!(!context.push_call());
        assert!(context.call_depth_exceeded());
        for _ in 0..MAX_CALL_DEPTH {
            context.pop_call();
        }
        assert!(context.push_call());
    }
}

/// How many nested calls compile-time evaluation will follow. A recursion that does not
/// terminate used to take the compiler's stack down with it; this turns that into a
/// diagnostic. High enough that ordinary compile-time work never reaches it.
pub const MAX_CALL_DEPTH: u32 = 256;

/// How many loop iterations compile-time evaluation will run, counted across every loop in
/// one evaluation rather than per loop. A loop whose end is never reached would otherwise
/// hang the compiler with no file, no line and no message; this turns that into a
/// diagnostic. High enough that ordinary compile-time work never reaches it.
pub const MAX_LOOP_STEPS: u64 = 1_000_000;

impl Default for ConstEvalState {
    fn default() -> Self {
        Self {
            // One scope open from the start, matching `TypeChecker::scopes`. Deriving `Default`
            // would start with none, and the first `let` would have nowhere to bind.
            env: vec![HashMap::new()],
            constraints: Vec::new(),
            return_constraints: Vec::new(),
            call_depth: std::cell::Cell::new(0),
            depth_exceeded: std::cell::Cell::new(false),
            unsupported_stmt: std::cell::Cell::new(false),
            loop_steps: std::cell::Cell::new(0),
            steps_exceeded: std::cell::Cell::new(false),
            comptime_depth: 0,
            closure_body_depth: 0,
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
    /// How many generic instantiations are open around the call being checked.
    ///
    /// `f<N - 1>()` whose base case is never reached instantiates a new `f` every time
    /// round, each one checked inside the last, and that took the compiler's stack down
    /// with no file, no line and no message.
    pub instantiation_depth: u32,
}

/// How deep a chain of generic instantiations the checker will follow before giving up.
/// High enough that ordinary compile-time recursion never reaches it.
pub const MAX_INSTANTIATION_DEPTH: u32 = 128;

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
    /// Number of per-seam obligations discharged.
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
