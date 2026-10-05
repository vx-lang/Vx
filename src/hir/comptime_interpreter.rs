//! The transition harness for the unified `comptime` interpreter.
//!
//! This is deliberately separate from ordinary `eval_expr`: it is the seam where comptime-only
//! value facts, support, flow, and possible escaping writes meet. Expression-owner work adds
//! concrete semantics here one family at a time; the exhaustive dispatch below makes omission a
//! compile error instead of a silently pure catch-all.

use std::collections::{HashMap, HashSet};

use crate::arch::TransferCostGraph;
use crate::hir::check_state::{
    ComptimeAggregateValue, ComptimeEvalContext, ComptimeEvalFlow, ComptimeEvalOutcome,
    ComptimeEvalSupport, ComptimeEvalUnsupportedReason, ComptimeEvalValue, ComptimeValueFacts,
};
use crate::hir::env::Value;
use crate::symbol::Symbol;
use crate::syntax::*;

/// Recursive comptime calls clone enough lexical state that the compiler worker's ordinary stack
/// can run out before the language-level 256-call limit. Each comptime block runs on one larger
/// stack; every call made while interpreting that block stays on the same worker thread.
// Debug builds retain considerably larger interpreter frames than the optimized CI binary. The
// language permits 256 nested comptime calls, so reserve enough stack for that documented limit
// in both profiles rather than making a valid program overflow only for a debug compiler.
const COMPTIME_RECURSION_STACK_SIZE: usize = 64 * 1024 * 1024;

/// The result of interpreting one `comptime` block.
#[derive(Clone, PartialEq)]
pub(crate) struct ComptimeObservation {
    pub outcome: ComptimeEvalOutcome,
    pub escaping_write: Option<Symbol>,
    pub call_depth_exceeded: bool,
}

#[derive(Clone)]
pub(crate) struct ComptimeFunctionBodies<'bodies> {
    comptime_bodies: &'bodies HashMap<Symbol, Function>,
    syntax_functions: &'bodies HashMap<Symbol, &'bodies Function>,
    mono_functions: &'bodies [(Function, u64)],
    /// Whether a parameter type can carry a mutable reference out of a call. Asked per call,
    /// for the function actually called, rather than for every function in the program each
    /// time a comptime block is checked.
    type_can_carry_mut_reference: &'bodies (dyn Fn(&Type) -> bool + Sync),
}

impl<'bodies> ComptimeFunctionBodies<'bodies> {
    pub(crate) fn new(
        comptime_bodies: &'bodies HashMap<Symbol, Function>,
        syntax_functions: &'bodies HashMap<Symbol, &'bodies Function>,
        mono_functions: &'bodies [(Function, u64)],
        type_can_carry_mut_reference: &'bodies (dyn Fn(&Type) -> bool + Sync),
    ) -> Self {
        Self {
            comptime_bodies,
            syntax_functions,
            mono_functions,
            type_can_carry_mut_reference,
        }
    }

    fn get(&self, name: &Symbol) -> Option<&Function> {
        self.syntax_functions
            .get(name)
            .filter(|function| !function.body.is_empty())
            .copied()
            .or_else(|| self.comptime_bodies.get(name))
            .or_else(|| {
                self.mono_functions
                    .iter()
                    .find(|(function, _)| function.name == *name)
                    .map(|(function, _)| function)
            })
    }

    fn contains(&self, name: &Symbol) -> bool {
        self.get(name).is_some()
    }

    /// Whether the function `name` returns a type that can carry a mutable reference.
    fn returns_mut_reference(&self, name: &Symbol) -> bool {
        self.get(name)
            .is_some_and(|function| (self.type_can_carry_mut_reference)(&function.return_type))
    }

    /// For each parameter of the body `get` returns for `name`, whether it can carry a mutable
    /// reference out of the call.
    fn mutable_reference_params(&self, name: &Symbol) -> Vec<bool> {
        let function = self
            .get(name)
            .expect("mutable-reference parameter facts are asked only for a known function");
        function
            .params
            .iter()
            .map(|(_, ty)| (self.type_can_carry_mut_reference)(ty))
            .collect()
    }
}

/// The owner-complete interpreter for a single `comptime` block.
#[derive(Clone)]
pub(crate) struct ComptimeInterpreter<'graph, 'bodies> {
    env: HashMap<Symbol, ComptimeEvalValue>,
    /// Local names whose values are mutable references, mapped to the place their next write
    /// reaches. Function parameters use a binding in their private call frame; their caller
    /// receives that frame's final value through the call write-back below.
    reference_places: HashMap<Symbol, ComptimeWritePlace>,
    /// Mutable references stored inside a local aggregate, keyed by the aggregate projection
    /// which contains the reference (for example `holder.value`).
    reference_projection_places: HashMap<ComptimeWritePlace, ComptimeWritePlace>,
    /// The variables written in the current call frame, by the root of each place stored to.
    /// A call writes back to its caller only the parameters its body wrote, so a function that
    /// only reads through a mutable reference does not count as writing to it.
    written_roots: HashSet<Symbol>,
    function_bodies: ComptimeFunctionBodies<'bodies>,
    transfer_cost_graph: &'graph TransferCostGraph,
    active_topology: Topology,
    context: ComptimeEvalContext,
}

/// A directly modelled local place that can receive an interpreted write. Anything more complex
/// is deliberately invalidated rather than leaving an older component value behind as certainty.
#[derive(Clone, PartialEq, Eq, Hash)]
enum ComptimeWritePlace {
    Binding(Symbol),
    Projection {
        root: Symbol,
        projections: Vec<ComptimePlaceProjection>,
    },
    Unknown,
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum ComptimePlaceProjection {
    ArrayElement(usize),
    StructField(Symbol),
}

impl<'graph, 'bodies> ComptimeInterpreter<'graph, 'bodies> {
    pub(crate) fn new(
        env: &HashMap<Symbol, Value>,
        function_bodies: ComptimeFunctionBodies<'bodies>,
        transfer_cost_graph: &'graph TransferCostGraph,
        active_topology: Topology,
        context: ComptimeEvalContext,
    ) -> Self {
        Self {
            env: env
                .iter()
                .map(|(name, value)| (name.clone(), ComptimeEvalValue::known(value.clone())))
                .collect(),
            reference_places: HashMap::new(),
            reference_projection_places: HashMap::new(),
            written_roots: HashSet::new(),
            function_bodies,
            transfer_cost_graph,
            active_topology,
            context,
        }
    }

    pub(crate) fn observe_block(
        &mut self,
        stmts: &[Statement],
        tail: Option<&Expr>,
    ) -> ComptimeObservation {
        // The worker is a stack boundary for the whole block, not for an individual call. A
        // loop of independent calls therefore pays for one worker rather than allocating and
        // joining a 64 MiB stack on every iteration.
        std::thread::scope(|scope| {
            let worker = std::thread::Builder::new()
                .name("vx-comptime".into())
                .stack_size(COMPTIME_RECURSION_STACK_SIZE)
                .spawn_scoped(scope, move || {
                    self.observe_block_on_current_stack(stmts, tail)
                });
            match worker {
                Ok(worker) => match worker.join() {
                    Ok(observation) => observation,
                    // The worker only supplies stack space. Do not turn a compiler panic into
                    // an E3033 on user code.
                    Err(payload) => std::panic::resume_unwind(payload),
                },
                // A failed worker allocation cannot safely evaluate the block. Keep the
                // conservative E3033 path rather than crashing the compiler.
                Err(_) => ComptimeObservation {
                    outcome: ComptimeEvalOutcome::refusal(),
                    escaping_write: None,
                    call_depth_exceeded: false,
                },
            }
        })
    }

    fn observe_block_on_current_stack(
        &mut self,
        stmts: &[Statement],
        tail: Option<&Expr>,
    ) -> ComptimeObservation {
        let outcome = self.block_with_tail(stmts, tail);
        ComptimeObservation {
            outcome,
            escaping_write: self.context.escaping_write().cloned(),
            call_depth_exceeded: self.context.call_depth_exceeded(),
        }
    }

    fn unsupported_after(
        &mut self,
        children: impl IntoIterator<Item = ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        Self::unsupported_outcome(children)
    }

    fn unsupported_outcome(
        children: impl IntoIterator<Item = ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        let mut outcome = ComptimeEvalOutcome::unsupported();
        for child in children {
            outcome.value.facts.merge_from(&child.value.facts);
            outcome.support.merge_from(child.support);
            outcome
                .unsupported_reason
                .merge_from(child.unsupported_reason);
            outcome.requires_refusal |= child.requires_refusal;
        }
        outcome.support = ComptimeEvalSupport::Unsupported;
        outcome
    }

    /// Preserve children, then mark a syntax owner whose lack of comptime semantics is already
    /// a live, fixture-backed refusal policy. Other transition-only unsupported results stay
    /// observations until the old/new comparator can classify them.
    fn refusal_after(
        &mut self,
        children: impl IntoIterator<Item = ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        let mut outcome = ComptimeEvalOutcome::refusal();
        for child in children {
            outcome.value.facts.merge_from(&child.value.facts);
            outcome.support.merge_from(child.support);
            outcome
                .unsupported_reason
                .merge_from(child.unsupported_reason);
            outcome.requires_refusal |= child.requires_refusal;
        }
        outcome
    }

    /// Sequence children in evaluation order. Their values need not agree, but their may-facts
    /// and unsupported status must survive even when the enclosing value becomes unknown.
    fn unknown_after(
        &mut self,
        children: impl IntoIterator<Item = ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        let mut outcome = ComptimeEvalOutcome::unknown();
        for child in children {
            outcome.value.facts.merge_from(&child.value.facts);
            outcome.support.merge_from(child.support);
            outcome
                .unsupported_reason
                .merge_from(child.unsupported_reason);
            outcome.requires_refusal |= child.requires_refusal;
            if child.flow != ComptimeEvalFlow::Normal {
                outcome.flow = child.flow;
            }
        }
        outcome
    }

    fn value_facts(&self, expr: &Expr) -> ComptimeValueFacts {
        match expr {
            Expr::Identifier(id) => self.identifier_facts(&id.name),
            Expr::Borrow(borrow) => self.value_facts(&borrow.expr),
            Expr::Dereference(deref) => self.value_facts(&deref.expr),
            Expr::MemberAccess(access) => self.value_facts(&access.base),
            Expr::IndexAccess(access) => {
                let mut facts = self.value_facts(&access.base);
                facts.merge_from(&self.value_facts(&access.index));
                facts
            }
            _ => ComptimeValueFacts::default(),
        }
    }

    fn identifier_facts(&self, name: &Symbol) -> ComptimeValueFacts {
        self.context
            .binding_facts(name)
            .cloned()
            .unwrap_or_else(|| {
                let mut facts = ComptimeValueFacts::default();
                if self.context.outer_reference_binding(name) {
                    facts.reference_origins.insert(name.clone());
                }
                if self.context.outer_callable_binding(name) {
                    // A fat-pointer closure crossed the comptime boundary without the facts of
                    // the environment it captured. Its body is unavailable, but the closure
                    // variable is not itself storage the call may write. `callable_call` still
                    // conservatively records known captured writes and mutable-reference
                    // arguments; do not invent an escaping write named after this callable.
                    facts.unknown_callable = true;
                }
                if self.function_bodies.contains(name) {
                    facts.callable_targets.insert(name.clone());
                }
                facts
            })
    }

    /// Facts for the storage a write or mutable borrow reaches, distinct from the value held in
    /// that storage. Dereferencing an alias reaches its carried reference origins; selecting a
    /// field or index remains in the base storage.
    fn place_facts(&self, expr: &Expr) -> ComptimeValueFacts {
        match expr {
            Expr::Identifier(id) => {
                if let Some(binding) = self.context.binding_facts(&id.name) {
                    // A local aggregate may *contain* a reference, but borrowing the aggregate
                    // (or one of its ordinary scalar fields) does not reach that reference's
                    // pointee. Component selection below carries the field's own facts when it
                    // is the reference-bearing field.
                    if self.env.get(&id.name).is_some_and(|value| {
                        matches!(&value.aggregate, Some(ComptimeAggregateValue::Struct(_)))
                    }) {
                        return ComptimeValueFacts::default();
                    }
                    // A local can name a closure-environment field or a reference parameter.
                    // Both are places whose writes reach the carried outer origin; keeping only
                    // captured-place origins here lost writes such as `param[0] = ...`.
                    let mut facts = ComptimeValueFacts::default();
                    facts
                        .reference_origins
                        .extend(binding.reference_origins.iter().cloned());
                    facts
                        .reference_origins
                        .extend(binding.captured_place_origins.iter().cloned());
                    facts
                } else if self.context.is_outer_binding(&id.name) {
                    let mut facts = ComptimeValueFacts::default();
                    facts.reference_origins.insert(id.name.clone());
                    facts
                } else {
                    ComptimeValueFacts::default()
                }
            }
            Expr::MemberAccess(access) => self.place_facts(&access.base),
            Expr::IndexAccess(access) => self.place_facts(&access.base),
            Expr::Dereference(deref) => self.value_facts(&deref.expr),
            _ => self.value_facts(expr),
        }
    }

    fn block(&mut self, stmts: &[Statement]) -> ComptimeEvalOutcome {
        self.block_with_tail(stmts, None)
    }

    /// Interpret a lexical block, keeping declarations available to its tail expression and
    /// restoring concrete bindings shadowed by the block when it exits. Provenance already had
    /// this discipline in `ComptimeEvalContext`; the value environment must mirror it or a local
    /// `let` in an `if`/`match` arm can leak a stale constant into the enclosing path.
    fn block_with_tail(&mut self, stmts: &[Statement], tail: Option<&Expr>) -> ComptimeEvalOutcome {
        let mut shadowed = HashMap::new();
        self.context.push_scope();
        let mut result = ComptimeEvalOutcome::unknown();
        let mut support = ComptimeEvalSupport::Supported;
        let mut unsupported_reason = ComptimeEvalUnsupportedReason::None;
        let mut requires_refusal = false;
        for stmt in stmts {
            // A lexical binding starts only once execution reaches its declaration. Saving
            // every declaration up front made an unreachable `let total = ...` undo a prior
            // write to the enclosing `total` when the block exited after `break` or `return`.
            if let Statement::LetDecl(decl) = stmt {
                shadowed.entry(decl.name.clone()).or_insert_with(|| {
                    (
                        self.env.get(&decl.name).cloned(),
                        self.reference_places.get(&decl.name).cloned(),
                        self.projection_references_for_root(&decl.name),
                    )
                });
            }
            result = self.statement(stmt);
            support.merge_from(result.support);
            unsupported_reason.merge_from(result.unsupported_reason);
            requires_refusal |= result.requires_refusal;
            if result.flow != ComptimeEvalFlow::Normal {
                break;
            }
        }
        if result.flow == ComptimeEvalFlow::Normal {
            if let Some(tail) = tail {
                result = self.expr(tail);
                support.merge_from(result.support);
                unsupported_reason.merge_from(result.unsupported_reason);
                requires_refusal |= result.requires_refusal;
            }
        }
        result.support.merge_from(support);
        result.unsupported_reason.merge_from(unsupported_reason);
        result.requires_refusal |= requires_refusal;
        self.context.pop_scope();
        for (name, (value, reference_place, projection_references)) in shadowed {
            match value {
                Some(value) => {
                    self.env.insert(name.clone(), value);
                }
                None => {
                    self.env.remove(&name);
                }
            }
            match reference_place {
                Some(place) => {
                    self.reference_places.insert(name.clone(), place);
                }
                None => {
                    self.reference_places.remove(&name);
                }
            }
            self.restore_projection_references_for_root(&name, projection_references);
        }
        result
    }

    fn statement(&mut self, stmt: &Statement) -> ComptimeEvalOutcome {
        match stmt {
            Statement::LetDecl(let_decl) => {
                let outcome = self.expr(&let_decl.expr);
                let reference_place = self.binding_reference_place(&let_decl.expr);
                self.context
                    .declare(let_decl.name.clone(), outcome.value.facts.clone());
                self.env
                    .insert(let_decl.name.clone(), outcome.value.clone());
                if let Some(place) = reference_place {
                    self.reference_places.insert(let_decl.name.clone(), place);
                } else {
                    self.reference_places.remove(&let_decl.name);
                }
                self.record_aggregate_reference_places(&let_decl.name, &let_decl.expr);
                outcome
            }
            Statement::Return(ret) => {
                let mut outcome = ret
                    .expr
                    .as_ref()
                    .map(|expr| self.expr(expr))
                    .unwrap_or_default();
                outcome.flow = ComptimeEvalFlow::Return;
                outcome
            }
            Statement::ExprStmt(expr) => self.expr(&expr.expr),
            Statement::ForLoop(loop_stmt) => self.for_loop(loop_stmt),
            Statement::Assign(assign) => self.assign(&assign.lhs, &assign.rhs),
            Statement::CompoundAssign(assign) => {
                self.compound_assign(&assign.lhs, &assign.op, &assign.rhs)
            }
            Statement::Assert(assert) => self.expr(&assert.expr),
            Statement::Loop(loop_stmt) => self.loop_stmt(loop_stmt),
            Statement::Break(_) => ComptimeEvalOutcome {
                flow: ComptimeEvalFlow::Break,
                ..ComptimeEvalOutcome::default()
            },
            Statement::Continue(_) => ComptimeEvalOutcome {
                flow: ComptimeEvalFlow::Continue,
                ..ComptimeEvalOutcome::default()
            },
            // Freeing memory changes no value the evaluator tracks.
            Statement::Drop(_) => ComptimeEvalOutcome::default(),
            Statement::MacroCall(_) | Statement::Error(_) => {
                // Both forms are supposed to have been eliminated before semantic checking. If
                // either escapes that boundary, it cannot be silently skipped before a later
                // foldable tail.
                ComptimeEvalOutcome::refusal()
            }
        }
    }

    fn assign(&mut self, lhs: &Expr, rhs: &Expr) -> ComptimeEvalOutcome {
        let (left, place) = self.write_place(lhs);
        let right = self.expr(rhs);
        let stored = self.store_place(lhs, place, right.value.clone());
        self.update_reference_binding(lhs, rhs);
        let mut outcome = self.unknown_after([left, right]);
        if !stored {
            outcome.mark_unsupported();
        }
        outcome
    }

    fn compound_assign(&mut self, lhs: &Expr, op: &BinaryOp, rhs: &Expr) -> ComptimeEvalOutcome {
        let (left, place) = self.write_place(lhs);
        let right = self.expr(rhs);
        let mut updated = self.binary(left.clone(), right.clone(), op);
        if !self.store_place(lhs, place, updated.value.clone()) {
            updated.mark_unsupported();
        }
        updated
    }

    /// Evaluate a writable place once, in source evaluation order, and retain the direct local
    /// path needed to update it after the right-hand side has run.
    fn write_place(&mut self, expr: &Expr) -> (ComptimeEvalOutcome, ComptimeWritePlace) {
        let outcome = self.expr(expr);
        (outcome, self.direct_place(expr))
    }

    /// A concrete, local lvalue. A projection stays attached to its root so that a call can
    /// write through `&mut holder.field`, a reborrowed parameter, or any depth of local
    /// aggregate rather than being mistaken for an unrelated scalar binding.
    fn direct_place(&self, expr: &Expr) -> ComptimeWritePlace {
        match expr {
            Expr::Identifier(id) => ComptimeWritePlace::Binding(id.name.clone()),
            Expr::Dereference(deref) => {
                if let Expr::Identifier(id) = &*deref.expr {
                    self.reference_places
                        .get(&id.name)
                        .cloned()
                        // An incoming mutable-reference parameter is represented by its local
                        // binding until it is reborrowed. Keeping that binding lets the context
                        // still report an outer write instead of degrading it to a generic
                        // unsupported operation.
                        .unwrap_or_else(|| ComptimeWritePlace::Binding(id.name.clone()))
                } else {
                    ComptimeWritePlace::Unknown
                }
            }
            Expr::IndexAccess(access) => {
                let Some(index) = self.place_index(&access.index) else {
                    return ComptimeWritePlace::Unknown;
                };
                Self::project_place(
                    self.direct_place(&access.base),
                    ComptimePlaceProjection::ArrayElement(index),
                )
            }
            Expr::MemberAccess(access) => Self::project_place(
                self.direct_place(&access.base),
                ComptimePlaceProjection::StructField(access.member.clone()),
            ),
            _ => ComptimeWritePlace::Unknown,
        }
    }

    fn place_index(&self, expr: &Expr) -> Option<usize> {
        // The destination must be determined without replaying an effectful index expression:
        // its ordinary evaluation already ran while evaluating the lvalue. Loop arithmetic such
        // as `j + 1` is pure, however, and must retain the same concrete index as `j` itself.
        if !Self::is_pure_index_expr(expr) {
            return None;
        }
        let mut probe = self.clone();
        let value = probe.expr(expr).value.concrete?;
        match value {
            Value::Int(index) if index >= 0 => Some(index as usize),
            _ => None,
        }
    }

    fn is_pure_index_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Number(_) | Expr::Identifier(_) => true,
            Expr::BinaryOp(op) => {
                Self::is_pure_index_expr(&op.lhs) && Self::is_pure_index_expr(&op.rhs)
            }
            Expr::UnaryOp(op) => Self::is_pure_index_expr(&op.expr),
            _ => false,
        }
    }

    fn project_place(
        place: ComptimeWritePlace,
        projection: ComptimePlaceProjection,
    ) -> ComptimeWritePlace {
        match place {
            ComptimeWritePlace::Binding(root) => ComptimeWritePlace::Projection {
                root,
                projections: vec![projection],
            },
            ComptimeWritePlace::Projection {
                root,
                mut projections,
            } => {
                projections.push(projection);
                ComptimeWritePlace::Projection { root, projections }
            }
            ComptimeWritePlace::Unknown => ComptimeWritePlace::Unknown,
        }
    }

    /// `Some` means the argument is a mutable-reference argument. Its destination must be
    /// exact: `Some(Unknown)` deliberately makes the enclosing call refuse to fold.
    /// The place a variable reaches after `let r = rhs` or `r = rhs`. A value that may borrow
    /// something but comes from a call or an `if`, `pass(&mut x)`, reaches a place this evaluator
    /// does not know, so writing through `r` or passing it on is refused.
    fn binding_reference_place(&self, rhs: &Expr) -> Option<ComptimeWritePlace> {
        self.reference_source_place(rhs).or_else(|| {
            self.makes_mut_reference(rhs)
                .then_some(ComptimeWritePlace::Unknown)
        })
    }

    /// A struct or array literal holding a mutable reference, as in `Holder { value: &mut x }`.
    /// Passed to a call, the callee can write through it, and there is no place to write back.
    fn literal_holds_mut_reference(&self, expr: &Expr) -> bool {
        let holds = |value: &Expr| {
            self.reference_source_place(value).is_some()
                || self.makes_mut_reference(value)
                || self.literal_holds_mut_reference(value)
        };
        match expr {
            Expr::StructInit(init) => init.fields.iter().any(|(_, value)| holds(value)),
            Expr::Array(array) => array.elements.iter().any(holds),
            _ => false,
        }
    }

    /// Whether a call, `if` or `match` can produce a mutable reference: a call to a function
    /// whose return type can carry one, or a branch whose value can.
    fn makes_mut_reference(&self, expr: &Expr) -> bool {
        let tail_makes = |stmts: &[Statement]| match stmts.last() {
            Some(Statement::ExprStmt(tail)) if !tail.has_semi => {
                self.reference_source_place(&tail.expr).is_some()
                    || self.makes_mut_reference(&tail.expr)
            }
            _ => false,
        };
        match expr {
            Expr::FunctionCall(call) => self.function_bodies.returns_mut_reference(&call.name),
            Expr::If(if_expr) => {
                tail_makes(&if_expr.then_block)
                    || if_expr.else_block.as_deref().is_some_and(tail_makes)
            }
            Expr::Match(match_expr) => match_expr.arms.iter().any(|arm| tail_makes(&arm.body)),
            _ => false,
        }
    }

    fn mutable_argument_place(&self, expr: &Expr) -> Option<ComptimeWritePlace> {
        let direct = self.direct_place(expr);
        if let Some(place) = self.reference_projection_places.get(&direct) {
            return Some(place.clone());
        }
        self.reference_source_place(expr)
    }

    fn update_reference_binding(&mut self, lhs: &Expr, rhs: &Expr) {
        let Expr::Identifier(lhs) = lhs else {
            return;
        };
        let place = self.binding_reference_place(rhs);
        if let Some(place) = place {
            self.reference_places.insert(lhs.name.clone(), place);
        } else {
            self.reference_places.remove(&lhs.name);
        }
    }

    /// The destination carried by an actual mutable-reference expression. An identifier is a
    /// reference only when it was previously introduced as one (or is an outer `&mut` binding);
    /// a normal value copy such as `let n = size` must never manufacture an alias to `size`.
    fn reference_source_place(&self, rhs: &Expr) -> Option<ComptimeWritePlace> {
        match rhs {
            Expr::Borrow(borrow) if borrow.is_mut => Some(self.direct_place(&borrow.expr)),
            Expr::Identifier(rhs) => self.reference_places.get(&rhs.name).cloned().or_else(|| {
                self.context
                    .outer_reference_binding(&rhs.name)
                    .then(|| ComptimeWritePlace::Binding(rhs.name.clone()))
            }),
            Expr::MemberAccess(_) | Expr::IndexAccess(_) => {
                let direct = self.direct_place(rhs);
                if let Some(target) = self.reference_projection_places.get(&direct) {
                    return Some(target.clone());
                }
                (!self.value_facts(rhs).reference_origins.is_empty()).then_some(direct)
            }
            _ => None,
        }
    }

    /// Whether two places share storage: the same root, and one's path a prefix of the other's.
    fn places_overlap(a: &ComptimeWritePlace, b: &ComptimeWritePlace) -> bool {
        fn path(place: &ComptimeWritePlace) -> Option<(&Symbol, &[ComptimePlaceProjection])> {
            match place {
                ComptimeWritePlace::Binding(root) => Some((root, &[])),
                ComptimeWritePlace::Projection { root, projections } => {
                    Some((root, projections.as_slice()))
                }
                ComptimeWritePlace::Unknown => None,
            }
        }
        match (path(a), path(b)) {
            (Some((root_a, path_a)), Some((root_b, path_b))) => {
                root_a == root_b && (path_a.starts_with(path_b) || path_b.starts_with(path_a))
            }
            _ => true,
        }
    }

    fn place_root(place: &ComptimeWritePlace) -> Option<&Symbol> {
        match place {
            ComptimeWritePlace::Binding(root) | ComptimeWritePlace::Projection { root, .. } => {
                Some(root)
            }
            ComptimeWritePlace::Unknown => None,
        }
    }

    fn projection_references_for_root(
        &self,
        root: &Symbol,
    ) -> Vec<(ComptimeWritePlace, ComptimeWritePlace)> {
        self.reference_projection_places
            .iter()
            .filter(|(place, _)| Self::place_root(place) == Some(root))
            .map(|(place, target)| (place.clone(), target.clone()))
            .collect()
    }

    fn restore_projection_references_for_root(
        &mut self,
        root: &Symbol,
        saved: Vec<(ComptimeWritePlace, ComptimeWritePlace)>,
    ) {
        self.reference_projection_places
            .retain(|place, _| Self::place_root(place) != Some(root));
        self.reference_projection_places.extend(saved);
    }

    fn record_aggregate_reference_places(&mut self, root: &Symbol, expr: &Expr) {
        let Expr::StructInit(init) = expr else {
            return;
        };
        for (field, value) in &init.fields {
            let target = self.reference_source_place(value);
            if let Some(target) = target {
                self.reference_projection_places.insert(
                    ComptimeWritePlace::Projection {
                        root: root.clone(),
                        projections: vec![ComptimePlaceProjection::StructField(field.clone())],
                    },
                    target,
                );
            }
        }
    }

    fn store_place(
        &mut self,
        lhs: &Expr,
        place: ComptimeWritePlace,
        value: ComptimeEvalValue,
    ) -> bool {
        match place {
            ComptimeWritePlace::Binding(_) | ComptimeWritePlace::Projection { .. } => {
                if let ComptimeWritePlace::Projection { .. } = &place {
                    self.context
                        .note_escaping_write(self.place_facts(lhs).reference_origins);
                }
                self.store_resolved_place(place, value)
            }
            ComptimeWritePlace::Unknown => {
                self.context
                    .note_escaping_write(self.place_facts(lhs).reference_origins);
                self.invalidate_unknown_local_place(lhs)
            }
        }
    }

    /// An unknown local array index is still a real write, but it cannot affect storage outside
    /// the block. Forget the whole local aggregate so a later read cannot use a stale component;
    /// this is safer and more precise than rejecting a block whose unknown write is provably dead.
    fn invalidate_unknown_local_place(&mut self, expr: &Expr) -> bool {
        let base = match expr {
            Expr::IndexAccess(access) => &access.base,
            Expr::MemberAccess(access) => &access.base,
            _ => return false,
        };
        let place = self.direct_place(base);
        let Some(root) = Self::place_root(&place).cloned() else {
            return false;
        };
        let known = self.env.contains_key(&root) || self.root_write_escapes(&root);
        self.written_roots.insert(root.clone());
        self.invalidate_binding(&root);
        known
    }

    fn store_resolved_place(
        &mut self,
        place: ComptimeWritePlace,
        value: ComptimeEvalValue,
    ) -> bool {
        if let Some(root) = Self::place_root(&place) {
            self.written_roots.insert(root.clone());
        }
        match place {
            ComptimeWritePlace::Binding(name) => {
                self.store_binding(name, value);
                true
            }
            ComptimeWritePlace::Projection { root, projections } => {
                let stored = self.update_local_value(&root, |stored| {
                    Self::update_projection(stored, &projections, &value)
                });
                if !stored {
                    self.invalidate_binding(&root);
                }
                stored || self.root_write_escapes(&root)
            }
            ComptimeWritePlace::Unknown => false,
        }
    }

    fn read_place(&self, place: &ComptimeWritePlace) -> Option<ComptimeEvalValue> {
        match place {
            ComptimeWritePlace::Binding(name) => self.env.get(name).cloned(),
            ComptimeWritePlace::Projection { root, projections } => {
                let value = self.env.get(root)?;
                Self::read_projection(value, projections)
            }
            ComptimeWritePlace::Unknown => None,
        }
    }

    fn read_projection(
        value: &ComptimeEvalValue,
        projections: &[ComptimePlaceProjection],
    ) -> Option<ComptimeEvalValue> {
        let Some((projection, rest)) = projections.split_first() else {
            return Some(value.clone());
        };
        match projection {
            ComptimePlaceProjection::ArrayElement(index) => {
                if let Some(ComptimeAggregateValue::Array(items)) = &value.aggregate {
                    return Self::read_projection(items.get(*index)?, rest);
                }
                if !rest.is_empty() {
                    return None;
                }
                let Value::Array(items) = value.concrete.as_ref()? else {
                    return None;
                };
                Some(ComptimeEvalValue::known(items.get(*index)?.clone()))
            }
            ComptimePlaceProjection::StructField(field) => {
                let ComptimeAggregateValue::Struct(fields) = value.aggregate.as_ref()? else {
                    return None;
                };
                Self::read_projection(fields.get(field)?, rest)
            }
        }
    }

    fn root_write_escapes(&self, name: &Symbol) -> bool {
        self.context.is_outer_binding(name)
            || self
                .context
                .binding_facts(name)
                .is_some_and(|facts| !facts.captured_place_origins.is_empty())
    }

    fn update_projection(
        value: &mut ComptimeEvalValue,
        projections: &[ComptimePlaceProjection],
        replacement: &ComptimeEvalValue,
    ) -> bool {
        let Some((projection, rest)) = projections.split_first() else {
            *value = replacement.clone();
            return true;
        };
        match projection {
            ComptimePlaceProjection::ArrayElement(index) => {
                if let Some(ComptimeAggregateValue::Array(items)) = &mut value.aggregate {
                    let Some(item) = items.get_mut(*index) else {
                        return false;
                    };
                    let updated = Self::update_projection(item, rest, replacement);
                    if updated {
                        Self::refresh_aggregate(value);
                    }
                    return updated;
                }
                if !rest.is_empty() {
                    return false;
                }
                let Some(Value::Array(items)) = &mut value.concrete else {
                    return false;
                };
                let Some(item) = items.get_mut(*index) else {
                    return false;
                };
                let Some(concrete) = replacement.concrete.clone() else {
                    return false;
                };
                *item = concrete;
                value.facts.merge_from(&replacement.facts);
                true
            }
            ComptimePlaceProjection::StructField(field) => {
                let Some(ComptimeAggregateValue::Struct(fields)) = &mut value.aggregate else {
                    return false;
                };
                let Some(item) = fields.get_mut(field) else {
                    return false;
                };
                let updated = Self::update_projection(item, rest, replacement);
                if updated {
                    Self::refresh_aggregate(value);
                }
                updated
            }
        }
    }

    fn store_binding(&mut self, name: Symbol, value: ComptimeEvalValue) {
        if self.context.is_outer_binding(&name) {
            self.context.note_escaping_write([name]);
            return;
        }
        let captured_place = self
            .context
            .binding_facts(&name)
            .map(|binding| binding.captured_place_origins.clone())
            .unwrap_or_default();
        self.context
            .note_escaping_write(captured_place.iter().cloned());
        if captured_place.is_empty() {
            self.context.reassign(&name, value.facts.clone());
            self.env.insert(name, value);
        }
    }

    /// Apply a component write to a local value. A caller cannot rely on a stale component when
    /// the container was unknown or the write could not be represented, so that binding is
    /// invalidated by the caller instead.
    fn update_local_value(
        &mut self,
        name: &Symbol,
        update: impl FnOnce(&mut ComptimeEvalValue) -> bool,
    ) -> bool {
        if self.context.is_outer_binding(name) {
            self.context.note_escaping_write([name.clone()]);
            return false;
        }
        let captured_place = self
            .context
            .binding_facts(name)
            .map(|binding| binding.captured_place_origins.clone())
            .unwrap_or_default();
        self.context
            .note_escaping_write(captured_place.iter().cloned());
        if !captured_place.is_empty() {
            return false;
        }
        let facts = {
            let Some(value) = self.env.get_mut(name) else {
                return false;
            };
            if !update(value) {
                return false;
            }
            value.facts.clone()
        };
        self.context.reassign(name, facts);
        true
    }

    fn invalidate_binding(&mut self, name: &Symbol) {
        if !self.context.is_outer_binding(name) {
            self.env.remove(name);
            self.context.reassign(name, ComptimeValueFacts::default());
        }
    }

    fn refresh_aggregate(value: &mut ComptimeEvalValue) {
        match &value.aggregate {
            Some(ComptimeAggregateValue::Array(items)) => {
                value.concrete = items
                    .iter()
                    .map(|item| item.concrete.clone())
                    .collect::<Option<Vec<_>>>()
                    .map(Value::Array);
                value.facts =
                    items
                        .iter()
                        .fold(ComptimeValueFacts::default(), |mut facts, item| {
                            facts.merge_from(&item.facts);
                            facts
                        });
            }
            Some(ComptimeAggregateValue::Struct(fields)) => {
                value.concrete = None;
                value.facts =
                    fields
                        .values()
                        .fold(ComptimeValueFacts::default(), |mut facts, item| {
                            facts.merge_from(&item.facts);
                            facts
                        });
            }
            None => {}
        }
    }

    fn binary(
        &mut self,
        lhs: ComptimeEvalOutcome,
        rhs: ComptimeEvalOutcome,
        op: &BinaryOp,
    ) -> ComptimeEvalOutcome {
        let concrete = match (&lhs.value.concrete, &rhs.value.concrete) {
            (Some(Value::Int(a)), Some(Value::Int(b))) => match op {
                BinaryOp::Add => a.checked_add(*b).map(Value::Int),
                BinaryOp::Sub => a.checked_sub(*b).map(Value::Int),
                BinaryOp::Mul => a.checked_mul(*b).map(Value::Int),
                BinaryOp::Div => a.checked_div(*b).map(Value::Int),
                BinaryOp::Rem => a.checked_rem(*b).map(Value::Int),
                _ => None,
            },
            (Some(left), Some(right)) => match (left.as_f64(), right.as_f64(), op) {
                (Some(a), Some(b), BinaryOp::Add) => Some(Value::Number(a + b)),
                (Some(a), Some(b), BinaryOp::Sub) => Some(Value::Number(a - b)),
                (Some(a), Some(b), BinaryOp::Mul) => Some(Value::Number(a * b)),
                (Some(a), Some(b), BinaryOp::Div) => Some(Value::Number(a / b)),
                (Some(a), Some(b), BinaryOp::Rem) if b != 0.0 => Some(Value::Number(a % b)),
                _ => None,
            },
            _ => None,
        };
        let mut outcome = self.unknown_after([lhs, rhs]);
        outcome.value.concrete = concrete;
        outcome
    }

    fn relational_value(&self, lhs: &Value, rhs: &Value, op: &RelationalOp) -> Option<Value> {
        let comparison = match (lhs, rhs) {
            // Keep integer comparisons exact: routing large integers through `f64` changes the
            // answer above 2^53.
            (Value::Int(left), Value::Int(right)) => match op {
                RelationalOp::Eq => left == right,
                RelationalOp::NotEq => left != right,
                RelationalOp::Lt => left < right,
                RelationalOp::Gt => left > right,
                RelationalOp::Le => left <= right,
                RelationalOp::Ge => left >= right,
            },
            // Mixed and floating numeric comparisons use the same numeric rule as the legacy
            // evaluator.
            _ if lhs.as_f64().is_some() && rhs.as_f64().is_some() => {
                let (left, right) = (lhs.as_f64()?, rhs.as_f64()?);
                match op {
                    RelationalOp::Eq => left == right,
                    RelationalOp::NotEq => left != right,
                    RelationalOp::Lt => left < right,
                    RelationalOp::Gt => left > right,
                    RelationalOp::Le => left <= right,
                    RelationalOp::Ge => left >= right,
                }
            }
            (Value::Bool(left), Value::Bool(right)) => match op {
                RelationalOp::Eq => left == right,
                RelationalOp::NotEq => left != right,
                _ => return None,
            },
            (Value::Topology(left), Value::Topology(right)) => match op {
                RelationalOp::Eq => self.topologies_equal(left, right)?,
                RelationalOp::NotEq => !self.topologies_equal(left, right)?,
                _ => return None,
            },
            _ => return None,
        };
        Some(Value::Bool(comparison))
    }

    /// Semantic topology equality. `topology_dispatch_id` is a compact placement key and can
    /// collide for distinct slices, so it must never decide a language-level comparison.
    fn topologies_equal(&self, left: &Topology, right: &Topology) -> Option<bool> {
        match (left, right) {
            (Topology::CPU, Topology::CPU)
            | (Topology::CpuAvx512, Topology::CpuAvx512)
            | (Topology::CpuNeon, Topology::CpuNeon)
            | (Topology::AMX, Topology::AMX)
            | (Topology::ANE, Topology::ANE)
            | (Topology::Current, Topology::Current) => Some(true),
            (Topology::GPU(left), Topology::GPU(right))
            | (Topology::NPU(left), Topology::NPU(right))
            | (Topology::AccCore(left), Topology::AccCore(right)) => {
                self.topology_exprs_equal(left, right)
            }
            (
                Topology::Slice(left_topology, left_start, left_end),
                Topology::Slice(right_topology, right_start, right_end),
            ) => Some(
                self.topologies_equal(left_topology, right_topology)?
                    && self.topology_exprs_equal(left_start, right_start)?
                    && self.topology_exprs_equal(left_end, right_end)?,
            ),
            (Topology::Custom(left), Topology::Custom(right)) => Some(left == right),
            _ => Some(false),
        }
    }

    fn topology_exprs_equal(&self, left: &Expr, right: &Expr) -> Option<bool> {
        let mut left_interpreter = self.clone();
        let mut right_interpreter = self.clone();
        let left = left_interpreter.expr(left).value.concrete?;
        let right = right_interpreter.expr(right).value.concrete?;
        match (left, right) {
            (Value::Int(left), Value::Int(right)) => Some(left == right),
            (Value::Number(left), Value::Number(right)) => Some((left - right).abs() < 1e-9),
            (Value::Bool(left), Value::Bool(right)) => Some(left == right),
            (Value::Topology(left), Value::Topology(right)) => self.topologies_equal(&left, &right),
            _ => None,
        }
    }

    fn array(&mut self, elements: Vec<ComptimeEvalOutcome>) -> ComptimeEvalOutcome {
        let concrete = elements
            .iter()
            .map(|outcome| outcome.value.concrete.clone())
            .collect::<Option<Vec<_>>>()
            .map(Value::Array);
        let aggregate = ComptimeAggregateValue::Array(
            elements
                .iter()
                .map(|outcome| outcome.value.clone())
                .collect(),
        );
        let mut outcome = self.unknown_after(elements);
        outcome.value.concrete = concrete;
        outcome.value.aggregate = Some(aggregate);
        outcome
    }

    /// Select one component without inheriting may-facts from its unrelated siblings. The base
    /// still contributes support, refusal, flow, and already-recorded effects through
    /// `unknown_after`; only the selected *value* is precise.
    fn member_access(&mut self, base: ComptimeEvalOutcome, member: &Symbol) -> ComptimeEvalOutcome {
        let selected = match &base.value.aggregate {
            Some(ComptimeAggregateValue::Struct(fields)) => fields.get(member).cloned(),
            _ => None,
        };
        let mut outcome = self.unknown_after([base]);
        if let Some(selected) = selected {
            outcome.value = selected;
        }
        outcome
    }

    fn index_access(
        &mut self,
        base: ComptimeEvalOutcome,
        index: ComptimeEvalOutcome,
    ) -> ComptimeEvalOutcome {
        let selected = match (&base.value.aggregate, &index.value.concrete) {
            (Some(ComptimeAggregateValue::Array(items)), Some(Value::Int(index)))
                if *index >= 0 =>
            {
                items.get(*index as usize).cloned()
            }
            _ => None,
        };
        let concrete = match (&base.value.concrete, &index.value.concrete) {
            (Some(Value::Array(items)), Some(Value::Int(index))) if *index >= 0 => {
                items.get(*index as usize).cloned()
            }
            _ => None,
        };
        let mut outcome = self.unknown_after([base, index]);
        if let Some(selected) = selected {
            outcome.value = selected;
        } else {
            outcome.value.concrete = concrete;
        }
        outcome
    }

    /// Run a direct call whose body is available to comptime evaluation. The call gets a fresh
    /// value environment, but its parameter facts are installed in the context's enclosing call
    /// scope, so `*param = ...` is recognised as a write through the caller's mutable reference.
    ///
    /// Calls without a body or with an arity mismatch are deliberately unsupported during the
    /// transition. Recursive calls use the same isolated-frame model up to the shared depth
    /// limit. Arguments are evaluated before this point, preserving any effect that occurs while
    /// producing one.
    fn known_function_call(
        &mut self,
        target: &Symbol,
        args: Vec<ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        self.known_function_call_with_writebacks(target, args, Vec::new())
    }

    fn known_function_call_with_writebacks(
        &mut self,
        target: &Symbol,
        args: Vec<ComptimeEvalOutcome>,
        writebacks: Vec<Option<ComptimeWritePlace>>,
    ) -> ComptimeEvalOutcome {
        self.known_function_call_on_current_stack(target, args, writebacks)
    }

    fn known_function_call_on_current_stack(
        &mut self,
        target: &Symbol,
        args: Vec<ComptimeEvalOutcome>,
        writebacks: Vec<Option<ComptimeWritePlace>>,
    ) -> ComptimeEvalOutcome {
        let Some(function) = self.function_bodies.get(target).cloned() else {
            return self.unsupported_after(args);
        };
        let mutable_reference_params = self.function_bodies.mutable_reference_params(target);
        if function.params.len() != args.len()
            || (!writebacks.is_empty() && writebacks.len() != args.len())
            || mutable_reference_params.len() != function.params.len()
        {
            return self.unsupported_after(args);
        }
        if !self.context.push_call() {
            return self.unsupported_after(args);
        }

        let saved_env = std::mem::take(&mut self.env);
        let saved_reference_places = std::mem::take(&mut self.reference_places);
        let saved_reference_projection_places =
            std::mem::take(&mut self.reference_projection_places);
        let saved_written_roots = std::mem::take(&mut self.written_roots);
        self.reference_projection_places = saved_reference_projection_places
            .iter()
            .filter(|(place, _)| {
                Self::place_root(place).is_some_and(|root| {
                    function
                        .params
                        .iter()
                        .any(|(parameter, _)| parameter == root)
                })
            })
            .map(|(place, target)| (place.clone(), target.clone()))
            .collect();
        self.context.push_scope();
        for ((((parameter, _), parameter_can_carry_mut_reference), argument), writeback) in function
            .params
            .iter()
            .zip(&mutable_reference_params)
            .zip(&args)
            .zip(writebacks.iter().chain(std::iter::repeat(&None)))
        {
            self.context
                .declare(parameter.clone(), argument.value.facts.clone());
            self.env.insert(parameter.clone(), argument.value.clone());
            if *parameter_can_carry_mut_reference && writeback.is_some() {
                // The parameter's private binding is its lvalue inside this call. When the call
                // returns, its final value is copied to the caller's tracked place below.
                self.reference_places.insert(
                    parameter.clone(),
                    ComptimeWritePlace::Binding(parameter.clone()),
                );
            }
        }
        let body = self.block(&function.body);
        let caller_updates = function
            .params
            .iter()
            .zip(&mutable_reference_params)
            .zip(writebacks.into_iter().chain(std::iter::repeat(None)))
            .filter_map(
                |(((parameter, _), parameter_can_carry_mut_reference), caller)| {
                    (*parameter_can_carry_mut_reference && self.written_roots.contains(parameter))
                        .then_some(caller)
                        .flatten()
                        .map(|caller| (caller, self.env.get(parameter).cloned()))
                },
            )
            .collect::<Vec<_>>();
        self.context.pop_scope();
        self.env = saved_env;
        self.reference_places = saved_reference_places;
        self.reference_projection_places = saved_reference_projection_places;
        // The write-backs below are writes in the caller's frame, so they are recorded there.
        self.written_roots = saved_written_roots;
        let mut writebacks_supported = true;
        for (caller, value) in caller_updates {
            match value {
                Some(value) => {
                    writebacks_supported &= self.store_resolved_place(caller, value);
                }
                None => {
                    writebacks_supported = false;
                }
            }
        }
        self.context.pop_call();

        let mut outcome = match body.flow {
            ComptimeEvalFlow::Return => ComptimeEvalOutcome {
                flow: ComptimeEvalFlow::Normal,
                ..body
            },
            ComptimeEvalFlow::Normal => {
                // A fall-through function has no call value, but every support/refusal fact from
                // its body is still a property of evaluating the call. Dropping that state would
                // let `fn log() { print(..) }` disappear before a later comptime tail folded.
                let mut outcome = body;
                outcome.value.concrete = None;
                outcome.value.aggregate = None;
                outcome
            }
            _ => ComptimeEvalOutcome::unsupported(),
        };
        for argument in args {
            outcome.support.merge_from(argument.support);
            outcome
                .unsupported_reason
                .merge_from(argument.unsupported_reason);
            outcome.requires_refusal |= argument.requires_refusal;
        }
        if !writebacks_supported {
            outcome.mark_unsupported();
        }
        outcome
    }

    fn callable_call(
        &mut self,
        facts: ComptimeValueFacts,
        args: Vec<ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        self.note_opaque_callable(&facts, &args);
        let mut targets = facts.callable_targets.iter();
        let Some(first_target) = targets.next() else {
            // A bodyless direct call (for example FFI), an outer callable, or an otherwise
            // unresolved callable cannot be discarded. Legacy evaluation can otherwise ignore
            // its statement and fold a later tail. `note_opaque_callable` may already have
            // found a concrete outer write; otherwise this remains a generic refusal.
            return self.refusal_after(args);
        };

        let mut merged = self.clone();
        let mut first_args = args.clone();
        if let Some(environment) = facts.callable_environments.get(first_target) {
            first_args.insert(
                0,
                ComptimeEvalOutcome {
                    value: ComptimeEvalValue {
                        concrete: None,
                        facts: environment.clone(),
                        aggregate: None,
                    },
                    ..ComptimeEvalOutcome::default()
                },
            );
        }
        let mut outcome = merged.known_function_call(first_target, first_args);
        for target in targets {
            let mut branch = self.clone();
            let mut branch_args = args.clone();
            if let Some(environment) = facts.callable_environments.get(target) {
                branch_args.insert(
                    0,
                    ComptimeEvalOutcome {
                        value: ComptimeEvalValue {
                            concrete: None,
                            facts: environment.clone(),
                            aggregate: None,
                        },
                        ..ComptimeEvalOutcome::default()
                    },
                );
            }
            let branch_outcome = branch.known_function_call(target, branch_args);
            merged.context.merge_branch(&branch.context);
            merged.env = Self::merge_environments(merged.env, &branch.env);
            merged.reference_places =
                Self::merge_reference_places(merged.reference_places, &branch.reference_places);
            merged.reference_projection_places = Self::merge_projection_reference_places(
                merged.reference_projection_places,
                &branch.reference_projection_places,
            );
            merged.written_roots.extend(branch.written_roots);
            outcome.merge_from(&branch_outcome);
        }
        self.context = merged.context;
        self.env = merged.env;
        self.reference_places = merged.reference_places;
        self.reference_projection_places = merged.reference_projection_places;
        self.written_roots = merged.written_roots;
        outcome
    }

    fn function_call(
        &mut self,
        call: &FunctionCallExpr,
        args: Vec<ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        if self.function_bodies.contains(&call.name) {
            return self.direct_call(&call.name, &call.args, args);
        }
        let facts = self.identifier_facts(&call.name);
        // A function value naming one function, `let cb = touch; cb(&mut x)`, is that call.
        if let (1, Some(target)) = (
            facts.callable_targets.len(),
            facts.callable_targets.iter().next().cloned(),
        ) {
            if !facts.unknown_callable
                && !facts.callable_environments.contains_key(&target)
                && self.function_bodies.contains(&target)
            {
                return self.direct_call(&target, &call.args, args);
            }
        }
        // Through a closure or a choice of functions, writes through a `&mut` argument have no
        // write-back here, so such a call is not folded.
        let writes_through_argument = facts.callable_targets.iter().any(|target| {
            self.function_bodies.contains(target)
                && self
                    .function_bodies
                    .mutable_reference_params(target)
                    .contains(&true)
        });
        if writes_through_argument {
            self.note_opaque_callable(&facts, &args);
            return self.refusal_after(args);
        }
        self.callable_call(facts, args)
    }

    /// A call to the function `name`, whose body is known, with `&mut` arguments written back.
    fn direct_call(
        &mut self,
        name: &Symbol,
        arg_exprs: &[Expr],
        mut args: Vec<ComptimeEvalOutcome>,
    ) -> ComptimeEvalOutcome {
        let function = self
            .function_bodies
            .get(name)
            .cloned()
            .expect("direct_call is made only for a function with a known body");
        let mutable_reference_params = self.function_bodies.mutable_reference_params(name);
        let writebacks = arg_exprs
            .iter()
            .zip(&mutable_reference_params)
            .map(|(arg, parameter_can_carry_mut_reference)| {
                // A reference made by a call, an `if` or a literal, `touch(pass(&mut x))` or
                // `f(Holder { value: &mut x })`, has no place this evaluator can write back
                // to, so the call is refused.
                parameter_can_carry_mut_reference
                    .then(|| {
                        self.mutable_argument_place(arg).or_else(|| {
                            (self.makes_mut_reference(arg) || self.literal_holds_mut_reference(arg))
                                .then_some(ComptimeWritePlace::Unknown)
                        })
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        if writebacks
            .iter()
            .flatten()
            .any(|place| matches!(place, ComptimeWritePlace::Unknown))
        {
            return self.refusal_after(args);
        }
        // Each parameter gets its own copy of what it points to, so two arguments reaching
        // the same variable, as in `f(p, p)` or `f(&mut h, &mut h.n)`, would not see each
        // other's writes. Such a call is not folded.
        let places = writebacks.iter().flatten().collect::<Vec<_>>();
        let overlapping = places
            .iter()
            .enumerate()
            .any(|(i, a)| places[i + 1..].iter().any(|b| Self::places_overlap(a, b)));
        if overlapping {
            return self.unsupported_after(args);
        }
        // The borrow outcome deliberately has no scalar `Value`: `comptime { &mut x }`
        // cannot replace itself with `x`. A known direct callee, however, needs the current
        // pointee value in its private parameter frame so `*param = ...` can compute a
        // write-back. Preserve the borrow's provenance and support state while supplying
        // only that frame-local value.
        for (argument, place) in args.iter_mut().zip(writebacks.iter()) {
            let Some(place) = place.as_ref() else {
                continue;
            };
            if let Some(value) = self.read_place(place) {
                argument.value.concrete = value.concrete.clone();
                argument.value.aggregate = value.aggregate.clone();
                argument.value.facts.merge_from(&value.facts);
            } else if !Self::place_root(place).is_some_and(|root| self.root_write_escapes(root)) {
                return self.refusal_after(args);
            }
        }
        let temporary_projection_places = self.forward_projection_reference_places(
            &function.params,
            &mutable_reference_params,
            arg_exprs,
        );
        let outcome = self.known_function_call_with_writebacks(name, args, writebacks);
        for place in temporary_projection_places {
            self.reference_projection_places.remove(&place);
        }
        outcome
    }

    /// A borrowed aggregate parameter retains the reference destinations of fields inside it.
    /// The temporary keys are installed under the callee's parameter name, then removed as soon
    /// as its frame returns; the caller's aggregate map remains untouched.
    fn forward_projection_reference_places(
        &mut self,
        params: &[(Symbol, Type)],
        mutable_reference_params: &[bool],
        args: &[Expr],
    ) -> Vec<ComptimeWritePlace> {
        let mut inserted = Vec::new();
        for (((parameter, _), parameter_can_carry_mut_reference), arg) in
            params.iter().zip(mutable_reference_params).zip(args)
        {
            if !parameter_can_carry_mut_reference {
                continue;
            }
            let source = match arg {
                Expr::Borrow(borrow) => self.direct_place(&borrow.expr),
                _ => self.direct_place(arg),
            };
            let Some(source_root) = Self::place_root(&source) else {
                continue;
            };
            let mappings = self
                .reference_projection_places
                .iter()
                .filter(|(place, _)| Self::place_root(place) == Some(source_root))
                .map(|(place, target)| (place.clone(), target.clone()))
                .collect::<Vec<_>>();
            for (place, target) in mappings {
                let ComptimeWritePlace::Projection { projections, .. } = place else {
                    continue;
                };
                let forwarded = ComptimeWritePlace::Projection {
                    root: parameter.clone(),
                    projections,
                };
                self.reference_projection_places
                    .insert(forwarded.clone(), target);
                inserted.push(forwarded);
            }
        }
        inserted
    }

    /// An opaque callable cannot be assumed pure. Captured writes and mutable-reference arguments
    /// name outer storage directly, so recording them prevents an indirect/callback call from
    /// being folded away just because its body was unavailable to this transition interpreter.
    fn note_opaque_callable(&mut self, facts: &ComptimeValueFacts, args: &[ComptimeEvalOutcome]) {
        // An empty target set includes a function value that crossed the comptime boundary
        // through an outer alias. Its spelling is unavailable here, but it may still write
        // through every mutable reference it received.
        if facts.unknown_callable || facts.callable_targets.is_empty() {
            self.context
                .note_escaping_write(facts.captured_writes.iter().cloned());
            self.context
                .note_escaping_write(args.iter().flat_map(|arg| {
                    arg.value
                        .facts
                        .reference_origins
                        .iter()
                        .chain(arg.value.facts.captured_place_origins.iter())
                        .cloned()
                }));
        }
    }

    /// Join the variable environment after an abstract branch. Branch-local declarations are
    /// absent from at least one side and therefore disappear; bindings that exist on both paths
    /// retain only a concrete value both paths agree on, while their provenance is a may-fact.
    fn merge_environments(
        mut left: HashMap<Symbol, ComptimeEvalValue>,
        right: &HashMap<Symbol, ComptimeEvalValue>,
    ) -> HashMap<Symbol, ComptimeEvalValue> {
        left.retain(|name, value| {
            let Some(other) = right.get(name) else {
                return false;
            };
            value.merge_from(other);
            true
        });
        left
    }

    /// A reference alias remains usable after an abstract join only when both paths point to
    /// precisely the same local place. Keeping an alias from one arm would make a later `*p =`
    /// look concrete even though the other arm may have made `p` point elsewhere.
    fn merge_reference_places(
        mut left: HashMap<Symbol, ComptimeWritePlace>,
        right: &HashMap<Symbol, ComptimeWritePlace>,
    ) -> HashMap<Symbol, ComptimeWritePlace> {
        left.retain(|name, place| right.get(name).is_some_and(|other| place == other));
        left
    }

    fn merge_projection_reference_places(
        mut left: HashMap<ComptimeWritePlace, ComptimeWritePlace>,
        right: &HashMap<ComptimeWritePlace, ComptimeWritePlace>,
    ) -> HashMap<ComptimeWritePlace, ComptimeWritePlace> {
        left.retain(|place, target| right.get(place).is_some_and(|other| target == other));
        left
    }

    fn if_expr(&mut self, if_expr: &IfExpr) -> ComptimeEvalOutcome {
        let condition = self.expr(&if_expr.cond);
        if condition.flow != ComptimeEvalFlow::Normal {
            return condition;
        }

        match condition.value.concrete {
            Some(Value::Bool(true)) => {
                let mut outcome = self.block(&if_expr.then_block);
                outcome.support.merge_from(condition.support);
                outcome
                    .unsupported_reason
                    .merge_from(condition.unsupported_reason);
                outcome.requires_refusal |= condition.requires_refusal;
                return outcome;
            }
            Some(Value::Bool(false)) => {
                let mut outcome = if_expr
                    .else_block
                    .as_ref()
                    .map(|branch| self.block(branch))
                    .unwrap_or_default();
                outcome.support.merge_from(condition.support);
                outcome
                    .unsupported_reason
                    .merge_from(condition.unsupported_reason);
                outcome.requires_refusal |= condition.requires_refusal;
                return outcome;
            }
            _ => {}
        }

        // An unknown condition has two viable paths. Each starts from the state after evaluating
        // the condition, then the states join: effects and reference origins union, while a
        // concrete value or binding survives only when both paths establish the same one.
        let mut then_interpreter = self.clone();
        let then_outcome = then_interpreter.block(&if_expr.then_block);
        let mut else_interpreter = self.clone();
        let else_outcome = if_expr
            .else_block
            .as_ref()
            .map(|branch| else_interpreter.block(branch))
            .unwrap_or_default();

        then_interpreter
            .context
            .merge_branch(&else_interpreter.context);
        self.context = then_interpreter.context;
        self.env = Self::merge_environments(then_interpreter.env, &else_interpreter.env);
        self.reference_places = Self::merge_reference_places(
            then_interpreter.reference_places,
            &else_interpreter.reference_places,
        );
        self.reference_projection_places = Self::merge_projection_reference_places(
            then_interpreter.reference_projection_places,
            &else_interpreter.reference_projection_places,
        );
        self.written_roots = then_interpreter.written_roots;
        self.written_roots.extend(else_interpreter.written_roots);

        let mut outcome = then_outcome;
        outcome.merge_from(&else_outcome);
        outcome.support.merge_from(condition.support);
        outcome
            .unsupported_reason
            .merge_from(condition.unsupported_reason);
        outcome.requires_refusal |= condition.requires_refusal;
        outcome
    }

    fn pattern_matches(pattern: &Pattern, value: &Value) -> Option<bool> {
        match pattern {
            Pattern::Wildcard | Pattern::Identifier(_) => Some(true),
            Pattern::Literal(Expr::Identifier(id)) if id.name.as_ref() == "true" => {
                Some(matches!(value, Value::Bool(true)))
            }
            Pattern::Literal(Expr::Identifier(id)) if id.name.as_ref() == "false" => {
                Some(matches!(value, Value::Bool(false)))
            }
            Pattern::Literal(Expr::Number(number)) => match (value, Self::number_literal(number)) {
                (Value::Int(value), Some(Value::Int(other))) => Some(value == &other),
                (Value::Number(value), Some(Value::Number(other))) => Some(value == &other),
                (Value::Int(_), Some(Value::Number(_)))
                | (Value::Number(_), Some(Value::Int(_))) => Some(false),
                (_, None) => None,
                _ => Some(false),
            },
            Pattern::Literal(_) | Pattern::EnumVariant(..) => Some(false),
        }
    }

    /// Preserve the syntax-level numeric type while constructing the interpreter value. The
    /// spelling `7f64` has no decimal point, so parsing its text as an integer first silently
    /// changes both the operation selected below and the value written back into HIR.
    fn number_literal(number: &NumberExpr) -> Option<Value> {
        let is_float = number.ty.as_ref().is_some_and(ElementType::is_float)
            || (number.ty.is_none()
                && (number.value.contains('.')
                    || number.value.contains('e')
                    || number.value.contains('E')));
        if is_float {
            number.value.parse::<f64>().ok().map(Value::Number)
        } else {
            number
                .value
                .parse::<i64>()
                .map(Value::Int)
                .or_else(|_| number.value.parse::<f64>().map(Value::Number))
                .ok()
        }
    }

    fn bind_pattern_facts(&mut self, pattern: &Pattern, facts: &ComptimeValueFacts) {
        match pattern {
            Pattern::Identifier(name) => self.context.declare(name.clone(), facts.clone()),
            Pattern::EnumVariant(_, _, Some(payloads)) => {
                for payload in payloads {
                    // Aggregate field precision is an owner of its own. Until it is modelled,
                    // every payload binding receives the scrutinee's may-facts rather than
                    // dropping a possible reference origin on the floor.
                    self.bind_pattern_facts(payload, facts);
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::EnumVariant(_, _, None) => {}
        }
    }

    fn pattern_binding_names(pattern: &Pattern, names: &mut Vec<Symbol>) {
        match pattern {
            Pattern::Identifier(name) => names.push(name.clone()),
            Pattern::EnumVariant(_, _, Some(payloads)) => {
                for payload in payloads {
                    Self::pattern_binding_names(payload, names);
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) | Pattern::EnumVariant(_, _, None) => {}
        }
    }

    fn bind_pattern_value(&mut self, pattern: &Pattern, value: &ComptimeEvalValue) {
        match pattern {
            // A selected identifier arm binds the scrutinee itself. Retaining only its facts
            // made `match 2 { value => value + 1 }` lose the concrete `2` before the arm ran.
            Pattern::Identifier(name) => {
                self.context.declare(name.clone(), value.facts.clone());
                self.env.insert(name.clone(), value.clone());
            }
            // Enum payload components need their own concrete representation. Until that owner
            // exists, keep the existing conservative provenance-only binding for them.
            _ => self.bind_pattern_facts(pattern, &value.facts),
        }
    }

    fn match_arm(&mut self, arm: &MatchArm, scrutinee: &ComptimeEvalValue) -> ComptimeEvalOutcome {
        let mut names = Vec::new();
        Self::pattern_binding_names(&arm.pattern, &mut names);
        let shadowed = names
            .into_iter()
            .map(|name| {
                let value = self.env.get(&name).cloned();
                let reference_place = self.reference_places.get(&name).cloned();
                (name, (value, reference_place))
            })
            .collect::<HashMap<_, _>>();
        self.context.push_scope();
        self.bind_pattern_value(&arm.pattern, scrutinee);
        let outcome = self.block(&arm.body);
        self.context.pop_scope();
        for (name, (value, reference_place)) in shadowed {
            match value {
                Some(value) => {
                    self.env.insert(name.clone(), value);
                }
                None => {
                    self.env.remove(&name);
                }
            }
            match reference_place {
                Some(place) => {
                    self.reference_places.insert(name, place);
                }
                None => {
                    self.reference_places.remove(&name);
                }
            }
        }
        outcome
    }

    fn match_expr(&mut self, match_expr: &MatchExpr) -> ComptimeEvalOutcome {
        let scrutinee = self.expr(&match_expr.expr);
        if scrutinee.flow != ComptimeEvalFlow::Normal {
            return scrutinee;
        }

        let candidates: Vec<&MatchArm> = if let Some(value) = scrutinee.value.concrete.as_ref() {
            let mut matching = Vec::new();
            for arm in &match_expr.arms {
                match Self::pattern_matches(&arm.pattern, value) {
                    Some(true) => {
                        matching.push(arm);
                        break;
                    }
                    Some(false) => {}
                    None => return ComptimeEvalOutcome::unsupported(),
                }
            }
            matching
        } else {
            let mut viable = Vec::new();
            for arm in &match_expr.arms {
                viable.push(arm);
                if matches!(arm.pattern, Pattern::Wildcard | Pattern::Identifier(_)) {
                    break;
                }
            }
            viable
        };
        let Some((first, rest)) = candidates.split_first() else {
            return ComptimeEvalOutcome::unsupported();
        };

        let scrutinee_known = scrutinee.value.concrete.is_some();
        let scrutinee_support = scrutinee.support;
        let scrutinee_requires_refusal = scrutinee.requires_refusal;
        let scrutinee_value = scrutinee.value.clone();
        let mut merged = self.clone();
        let mut outcome = merged.match_arm(first, &scrutinee_value);
        for arm in rest {
            let mut branch = self.clone();
            let branch_outcome = branch.match_arm(arm, &scrutinee_value);
            merged.context.merge_branch(&branch.context);
            merged.env = Self::merge_environments(merged.env, &branch.env);
            merged.reference_places =
                Self::merge_reference_places(merged.reference_places, &branch.reference_places);
            merged.reference_projection_places = Self::merge_projection_reference_places(
                merged.reference_projection_places,
                &branch.reference_projection_places,
            );
            merged.written_roots.extend(branch.written_roots);
            outcome.merge_from(&branch_outcome);
        }
        self.context = merged.context;
        self.env = merged.env;
        self.reference_places = merged.reference_places;
        self.reference_projection_places = merged.reference_projection_places;
        self.written_roots = merged.written_roots;

        // Known scalar patterns have a concrete selected arm. Unknown and enum-pattern cases
        // still need pattern/value modelling before their value can fold, but their effects have
        // been collected above, so fail closed rather than forgetting an arm's write.
        if !scrutinee_known {
            outcome.mark_unsupported();
        }
        outcome.support.merge_from(scrutinee_support);
        outcome.requires_refusal |= scrutinee_requires_refusal;
        outcome
    }

    fn for_loop(&mut self, loop_stmt: &ForLoopStmt) -> ComptimeEvalOutcome {
        // A finite integer range has the same recurrence as the legacy evaluator. Run it
        // concretely so assignments made by one turn are available to the next one; the former
        // zero-or-one abstract join was safe for effects, but could never agree on values.
        let iterable = if let Expr::Range(range) = &*loop_stmt.iterable {
            let start = self.expr(&range.start);
            let end = self.expr(&range.end);
            if start.flow != ComptimeEvalFlow::Normal {
                return start;
            }
            if end.flow != ComptimeEvalFlow::Normal {
                return end;
            }
            if let (Some(Value::Int(start_index)), Some(Value::Int(end_index))) =
                (&start.value.concrete, &end.value.concrete)
            {
                let iterator: Symbol = loop_stmt.iter.clone().into();
                let shadowed = self.env.get(&iterator).cloned();
                let shadowed_reference = self.reference_places.remove(&iterator);
                self.context.push_scope();
                self.context
                    .declare(iterator.clone(), ComptimeValueFacts::default());

                let mut index = *start_index;
                let end_index = *end_index;
                let mut outcome = self.unknown_after([start, end]);
                while index < end_index {
                    if !self.context.take_loop_step() {
                        outcome.mark_unsupported();
                        break;
                    }
                    self.env.insert(
                        iterator.clone(),
                        ComptimeEvalValue::known(Value::Int(index)),
                    );
                    self.reference_places.remove(&iterator);
                    let body = self.block(&loop_stmt.body);
                    outcome.support.merge_from(body.support);
                    outcome
                        .unsupported_reason
                        .merge_from(body.unsupported_reason);
                    outcome.requires_refusal |= body.requires_refusal;
                    match body.flow {
                        ComptimeEvalFlow::Normal | ComptimeEvalFlow::Continue => {}
                        ComptimeEvalFlow::Break => break,
                        ComptimeEvalFlow::Return => {
                            outcome.flow = ComptimeEvalFlow::Return;
                            outcome.value = body.value;
                            break;
                        }
                        ComptimeEvalFlow::Indeterminate => {
                            outcome.flow = ComptimeEvalFlow::Indeterminate;
                            outcome.mark_unsupported();
                            break;
                        }
                    }
                    index += 1;
                }
                self.context.pop_scope();
                match shadowed {
                    Some(value) => {
                        self.env.insert(iterator.clone(), value);
                    }
                    None => {
                        self.env.remove(&iterator);
                    }
                }
                match shadowed_reference {
                    Some(place) => {
                        self.reference_places.insert(iterator, place);
                    }
                    None => {
                        self.reference_places.remove(&iterator);
                    }
                }
                return outcome;
            }
            // Keep the already-evaluated bounds when the recurrence is abstract, so an
            // effectful bound is not visited twice.
            self.unknown_after([start, end])
        } else {
            self.expr(&loop_stmt.iterable)
        };
        if iterable.flow != ComptimeEvalFlow::Normal {
            return iterable;
        }

        // The iterable may be empty, so join the state after zero iterations with the state after
        // one abstract iteration. That retains writes through outer references while preventing
        // the loop variable and body-local declarations from escaping the loop.
        let mut one_iteration = self.clone();
        one_iteration.context.push_scope();
        one_iteration
            .context
            .declare(loop_stmt.iter.clone().into(), ComptimeValueFacts::default());
        let body = one_iteration.block(&loop_stmt.body);
        one_iteration.context.pop_scope();

        self.context.merge_branch(&one_iteration.context);
        self.env = Self::merge_environments(self.env.clone(), &one_iteration.env);
        self.reference_places = Self::merge_reference_places(
            self.reference_places.clone(),
            &one_iteration.reference_places,
        );
        self.reference_projection_places = Self::merge_projection_reference_places(
            self.reference_projection_places.clone(),
            &one_iteration.reference_projection_places,
        );
        self.written_roots.extend(one_iteration.written_roots);
        let mut outcome = ComptimeEvalOutcome::unsupported();
        outcome.support.merge_from(iterable.support);
        outcome
            .unsupported_reason
            .merge_from(iterable.unsupported_reason);
        outcome.requires_refusal |= iterable.requires_refusal || body.requires_refusal;
        outcome
    }

    fn loop_stmt(&mut self, loop_stmt: &LoopStmt) -> ComptimeEvalOutcome {
        // Unlike a `for`, a plain loop has no empty path. Execute until an explicit `break` or
        // `return`; the shared budget gives an unproven recurrence the same fail-closed outcome
        // as the legacy evaluator's loop-step cap.
        let mut outcome = ComptimeEvalOutcome::unknown();
        while self.context.take_loop_step() {
            let body = self.block(&loop_stmt.body);
            outcome.support.merge_from(body.support);
            outcome
                .unsupported_reason
                .merge_from(body.unsupported_reason);
            outcome.requires_refusal |= body.requires_refusal;
            match body.flow {
                ComptimeEvalFlow::Normal | ComptimeEvalFlow::Continue => {}
                ComptimeEvalFlow::Break => return outcome,
                ComptimeEvalFlow::Return => {
                    outcome.flow = ComptimeEvalFlow::Return;
                    outcome.value = body.value;
                    return outcome;
                }
                ComptimeEvalFlow::Indeterminate => {
                    outcome.flow = ComptimeEvalFlow::Indeterminate;
                    outcome.mark_unsupported();
                    return outcome;
                }
            }
        }
        outcome.mark_unsupported();
        outcome.flow = ComptimeEvalFlow::Indeterminate;
        outcome
    }

    fn expr(&mut self, expr: &Expr) -> ComptimeEvalOutcome {
        match expr {
            Expr::Identifier(id) if id.name.as_ref() == "true" => {
                ComptimeEvalOutcome::known(Value::Bool(true))
            }
            Expr::Identifier(id) if id.name.as_ref() == "false" => {
                ComptimeEvalOutcome::known(Value::Bool(false))
            }
            Expr::Identifier(id) => {
                let mut value = self.env.get(&id.name).cloned().unwrap_or_default();
                value.facts.merge_from(&self.value_facts(expr));
                ComptimeEvalOutcome {
                    value,
                    ..Default::default()
                }
            }
            Expr::Number(number) => Self::number_literal(number)
                .map(ComptimeEvalOutcome::known)
                .unwrap_or_else(ComptimeEvalOutcome::unsupported),
            Expr::EnumVariant(variant) => {
                let values = variant
                    .payload
                    .iter()
                    .flatten()
                    .map(|value| self.expr(value))
                    .collect::<Vec<_>>();
                // Enum constants have no concrete representation in the transition value
                // model. Do not let an enum-driven match and its local writes vanish merely
                // because the legacy evaluator happens to continue to a later tail.
                self.refusal_after(values)
            }
            Expr::SizeOf(size) => comptime_sizeof_bytes(&size.target_ty)
                .map(Value::Int)
                .map(ComptimeEvalOutcome::known)
                .unwrap_or_else(ComptimeEvalOutcome::refusal),
            Expr::StringLiteral(_) | Expr::MemorySpace(_) | Expr::MacroCall(_) => {
                // The transition value model cannot faithfully represent these forms.  A raw
                // memory-space or macro node also violates an earlier lowering invariant, but
                // must still be live if it reaches this defensive interpreter boundary.
                ComptimeEvalOutcome::refusal()
            }
            Expr::Transfer(transfer) => {
                let value = self.expr(&transfer.expr);
                self.refusal_after([value])
            }
            Expr::TransferPredicate(predicate) => {
                let from = self.topology(&predicate.from);
                let to = self.topology(&predicate.to);
                let concrete = match (&from.value.concrete, &to.value.concrete) {
                    (Some(Value::Topology(from)), Some(Value::Topology(to))) => {
                        let from_memory = self.transfer_cost_graph.default_memory_for(from);
                        let to_memory = self.transfer_cost_graph.default_memory_for(to);
                        Some(Value::Bool(
                            self.transfer_cost_graph
                                .transfer_path(&from_memory, &to_memory)
                                .is_some(),
                        ))
                    }
                    _ => None,
                };
                let Some(concrete) = concrete else {
                    // The transfer graph can answer reachability only for concrete topology
                    // values. A runtime index must not be collapsed to the device kind and
                    // silently make a predicate statement disappear before a later folded tail.
                    // `refusal_after` still visits both operands, so an escaping write wins over
                    // this generic refusal.
                    return self.refusal_after([from, to]);
                };
                let mut outcome = self.unknown_after([from, to]);
                outcome.value.concrete = Some(concrete);
                outcome
            }
            Expr::FunctionCall(call) => {
                let args = call.args.iter().map(|arg| self.expr(arg)).collect();
                self.function_call(call, args)
            }
            Expr::IndirectCall(call) => {
                let callee = self.expr(&call.callee);
                let facts = callee.value.facts.clone();
                let args = call.args.iter().map(|arg| self.expr(arg)).collect();
                let mut outcome = self.callable_call(facts, args);
                outcome.support.merge_from(callee.support);
                outcome
                    .unsupported_reason
                    .merge_from(callee.unsupported_reason);
                outcome.requires_refusal |= callee.requires_refusal;
                outcome
            }
            Expr::Array(array) => {
                let elements = array
                    .elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect();
                self.array(elements)
            }
            Expr::MemberAccess(access) => {
                let base = self.expr(&access.base);
                self.member_access(base, &access.member)
            }
            Expr::IndexAccess(access) => {
                let base = self.expr(&access.base);
                let index = self.expr(&access.index);
                self.index_access(base, index)
            }
            Expr::MethodCall(call) => {
                let mut children = vec![self.expr(&call.base)];
                children.extend(call.args.iter().map(|arg| self.expr(arg)));
                // Semantic checking normally rewrites a method call to the resolved function,
                // extent access, or intrinsic form. A remaining raw method node has no
                // transition-model semantics, but its eager receiver and arguments still count.
                self.refusal_after(children)
            }
            Expr::BinaryOp(op) => {
                let lhs = self.expr(&op.lhs);
                let rhs = self.expr(&op.rhs);
                self.binary(lhs, rhs, &op.op)
            }
            Expr::RelationalOp(op) => {
                let lhs = self.expr(&op.lhs);
                let rhs = self.expr(&op.rhs);
                let mut outcome = self.unknown_after([lhs.clone(), rhs.clone()]);
                outcome.value.concrete = lhs
                    .value
                    .concrete
                    .as_ref()
                    .zip(rhs.value.concrete.as_ref())
                    .and_then(|(lhs, rhs)| self.relational_value(lhs, rhs, &op.op));
                if matches!(
                    (&lhs.value.concrete, &rhs.value.concrete),
                    (Some(Value::Topology(_)), Some(Value::Topology(_)))
                ) && outcome.value.concrete.is_none()
                {
                    // Neither an unequal topology nor an unresolved slice index may be guessed.
                    // Preserve the operands' effects, then reject the enclosing fold.
                    outcome.mark_unsupported();
                }
                outcome
            }
            Expr::LogicalOp(op) => {
                let lhs = self.expr(&op.lhs);
                if lhs.flow != ComptimeEvalFlow::Normal {
                    return lhs;
                }
                match (&lhs.value.concrete, &op.op) {
                    (Some(Value::Bool(false)), LogicalOp::And) => {
                        let mut outcome = ComptimeEvalOutcome::known(Value::Bool(false));
                        outcome.support.merge_from(lhs.support);
                        outcome
                            .unsupported_reason
                            .merge_from(lhs.unsupported_reason);
                        outcome.requires_refusal |= lhs.requires_refusal;
                        return outcome;
                    }
                    (Some(Value::Bool(true)), LogicalOp::Or) => {
                        let mut outcome = ComptimeEvalOutcome::known(Value::Bool(true));
                        outcome.support.merge_from(lhs.support);
                        outcome
                            .unsupported_reason
                            .merge_from(lhs.unsupported_reason);
                        outcome.requires_refusal |= lhs.requires_refusal;
                        return outcome;
                    }
                    _ => {}
                }
                let rhs = self.expr(&op.rhs);
                let mut outcome = self.unknown_after([lhs.clone(), rhs.clone()]);
                outcome.value.concrete = match (&lhs.value.concrete, &rhs.value.concrete, &op.op) {
                    (Some(Value::Bool(a)), Some(Value::Bool(b)), LogicalOp::And) => {
                        Some(Value::Bool(*a && *b))
                    }
                    (Some(Value::Bool(a)), Some(Value::Bool(b)), LogicalOp::Or) => {
                        Some(Value::Bool(*a || *b))
                    }
                    _ => None,
                };
                outcome
            }
            Expr::UnaryOp(op) => {
                let inner = self.expr(&op.expr);
                let mut outcome = self.unknown_after([inner.clone()]);
                outcome.value.concrete = match (&inner.value.concrete, &op.op) {
                    (Some(Value::Bool(value)), UnaryOp::Not) => Some(Value::Bool(!value)),
                    (Some(Value::Int(value)), UnaryOp::Neg) => value.checked_neg().map(Value::Int),
                    (Some(Value::Number(value)), UnaryOp::Neg) => Some(Value::Number(-value)),
                    _ => None,
                };
                outcome
            }
            Expr::Borrow(borrow) => {
                let inner = self.expr(&borrow.expr);
                let mut outcome = self.unknown_after([inner.clone()]);
                outcome.value.facts = inner.value.facts;
                outcome.value.aggregate = inner.value.aggregate;
                if borrow.is_mut {
                    outcome
                        .value
                        .facts
                        .merge_from(&self.place_facts(&borrow.expr));
                }
                outcome
            }
            Expr::Dereference(deref) => {
                let inner = self.expr(&deref.expr);
                let place = if let Expr::Identifier(id) = &*deref.expr {
                    self.reference_places.get(&id.name)
                } else {
                    None
                };
                if matches!(place, Some(ComptimeWritePlace::Unknown)) {
                    return self.refusal_after([inner]);
                }
                let referenced = place.and_then(|place| self.read_place(place));
                let mut outcome = self.unknown_after([inner.clone()]);
                // Direct calls install the pointee value in their parameter frame. A
                // dereference is the boundary at which that private representation becomes an
                // ordinary value again; a borrow expression itself remains non-foldable.
                if let Some(referenced) = referenced {
                    outcome.value = referenced;
                    outcome.value.facts.merge_from(&inner.value.facts);
                } else {
                    outcome.value.concrete = inner.value.concrete;
                    outcome.value.facts = inner.value.facts;
                    outcome.value.aggregate = inner.value.aggregate;
                }
                outcome
            }
            Expr::UnsafeBlock(block) => self.block_with_tail(&block.stmts, block.ret.as_deref()),
            Expr::ComptimeBlock(block) => self.block_with_tail(&block.stmts, block.ret.as_deref()),
            Expr::StructInit(init) => {
                let fields = init
                    .fields
                    .iter()
                    .map(|(name, value)| (name.clone(), self.expr(value), self.place_facts(value)))
                    .collect::<Vec<_>>();
                let mut outcome = self.unknown_after(
                    fields
                        .iter()
                        .map(|(_, outcome, _)| outcome.clone())
                        .collect::<Vec<_>>(),
                );
                let target: Symbol = format!("{}_call", init.name).into();
                if !init.name.starts_with("Closure_") {
                    // This abstract structure is deliberately not a concrete legacy `Value`:
                    // its components are available for local member/provenance reasoning, but
                    // an ordinary struct literal cannot itself replace a comptime block until
                    // the fold-value representation can spell structs back into source.
                    outcome.value.aggregate = Some(ComptimeAggregateValue::Struct(
                        fields
                            .iter()
                            .map(|(name, outcome, _)| (name.clone(), outcome.value.clone()))
                            .collect(),
                    ));
                }
                if init.name.starts_with("Closure_") && self.function_bodies.contains(&target) {
                    let mut environment = ComptimeValueFacts::default();
                    for (_, field, place) in fields {
                        environment.merge_from(&field.value.facts);
                        // A generated closure environment retains the captured place. This is
                        // deliberately distinct from an ordinary struct literal, whose fields are
                        // values and do not make every outer scalar into a mutable reference.
                        environment
                            .reference_origins
                            .extend(place.reference_origins.iter().cloned());
                        environment
                            .captured_place_origins
                            .extend(place.reference_origins);
                    }
                    outcome.value.facts.callable_targets.insert(target.clone());
                    outcome
                        .value
                        .facts
                        .callable_environments
                        .insert(target, environment.clone());
                    // The closure value itself transports its environment. Preserve that fact
                    // through a borrow of the closure object so the generated `_env` parameter
                    // and its captured-field bindings still name the original outer place.
                    outcome.value.facts.merge_from(&environment);
                }
                outcome
            }
            Expr::Topology(topology) => self.topology(&topology.top),
            Expr::If(if_expr) => self.if_expr(if_expr),
            Expr::Range(range) => {
                let start = self.expr(&range.start);
                let end = self.expr(&range.end);
                // A range used as a value has no spelling in the legacy `Value` model. The
                // `for` owner handles its integer bounds directly above; every other use must
                // evaluate both bounds for effects and then refuse rather than silently vanish.
                self.refusal_after([start, end])
            }
            Expr::Match(match_expr) => self.match_expr(match_expr),
            Expr::Grad(grad) => {
                let args = grad
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Vec<_>>();
                // Differentiation produces a transformed runtime computation, not a value the
                // transition model can represent. Eager operands still retain their effects.
                self.refusal_after(args)
            }
            Expr::Vjp(vjp) => {
                let mut children = vjp
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Vec<_>>();
                children.push(self.expr(&vjp.cotangent));
                self.refusal_after(children)
            }
            Expr::Jvp(jvp) => {
                let mut children = jvp
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Vec<_>>();
                children.push(self.expr(&jvp.tangent));
                self.refusal_after(children)
            }
            Expr::SpawnOn(spawn) => {
                let topology = self.topology(&spawn.top);
                let body = self.block_with_tail(&spawn.stmts, spawn.ret.as_deref());
                self.refusal_after([topology, body])
            }
            Expr::VecMacro(vector) => {
                let elements = vector
                    .elements
                    .iter()
                    .map(|element| self.expr(element))
                    .collect::<Vec<_>>();
                // Checked source normally lowers `vec![..]` into allocation and `push` calls
                // before this interpreter runs. Preserve the same fail-closed rule for a raw
                // macro node too, so a changed lowering order cannot make an allocation-backed
                // vector disappear with a comptime block.
                self.refusal_after(elements)
            }
            // A closure body is deferred until invocation. Checked closures are normally lowered
            // to a generated `Closure_N` struct before this point. If an unlowered literal
            // reaches this late boundary, it has no concrete callable/environment model, so it
            // must refuse the fold rather than letting a later tail erase it. Deliberately do not
            // interpret the body here: creation alone does not run a closure.
            Expr::Closure(_) => ComptimeEvalOutcome::refusal(),
            Expr::AsCast(cast) => {
                let value = self.expr(&cast.expr);
                // The legacy value model has no target type, so using its `Int`/`Number`
                // representation here would silently get narrowing, signedness, pointer, and
                // tensor casts wrong. Evaluate the operand for effects, then refuse the whole
                // block until a typed comptime value can model the conversion. This is live
                // rather than observational because legacy evaluation otherwise drops an
                // unsupported cast statement and folds a later tail.
                self.refusal_after([value])
            }
            Expr::Print(print) => {
                let args = print
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Vec<_>>();
                // Printing is an observable run-time effect. A comptime block disappears, so a
                // local print cannot be left to the legacy evaluator to drop before folding a
                // later tail; arguments are still evaluated first for escaping writes.
                self.refusal_after(args)
            }
            Expr::Println(print) => {
                let args = print
                    .args
                    .iter()
                    .map(|arg| self.expr(arg))
                    .collect::<Vec<_>>();
                self.refusal_after(args)
            }
            Expr::InlineMlir(mlir) => {
                let mut children = mlir
                    .inputs
                    .iter()
                    .map(|(_, value, _)| self.expr(value))
                    .collect::<Vec<_>>();
                for clobber in &mlir.clobbers {
                    children.push(self.expr(clobber));
                    self.context
                        .note_escaping_write(self.place_facts(clobber).reference_origins);
                }
                self.refusal_after(children)
            }
        }
    }

    fn topology(&mut self, topology: &Topology) -> ComptimeEvalOutcome {
        match topology {
            Topology::NPU(index) | Topology::AccCore(index) | Topology::GPU(index) => {
                let index = self.expr(index);
                let index_is_concrete = index.value.concrete.is_some();
                let mut outcome = self.unknown_after([index]);
                if index_is_concrete && outcome.support.is_supported() {
                    outcome.value.concrete = Some(Value::Topology(topology.clone()));
                }
                outcome
            }
            Topology::Slice(base, start, end) => {
                let base = self.topology(base);
                let start = self.expr(start);
                let end = self.expr(end);
                let components_are_concrete = base.value.concrete.is_some()
                    && start.value.concrete.is_some()
                    && end.value.concrete.is_some();
                let mut outcome = self.unknown_after([base, start, end]);
                if components_are_concrete && outcome.support.is_supported() {
                    outcome.value.concrete = Some(Value::Topology(topology.clone()));
                }
                outcome
            }
            Topology::CPU
            | Topology::AMX
            | Topology::ANE
            | Topology::CpuAvx512
            | Topology::CpuNeon
            | Topology::Custom(_) => ComptimeEvalOutcome::known(Value::Topology(topology.clone())),
            Topology::Current => {
                ComptimeEvalOutcome::known(Value::Topology(self.active_topology.clone()))
            }
        }
    }
}

/// The `sizeof<T>()` subset whose layout is available without the code generator's nominal-type
/// registry. It deliberately uses the shared scalar layout table so comptime and lowering agree
/// for every scalar width, including sub-byte scalar spellings that round up to one byte.
fn comptime_sizeof_bytes(ty: &Type) -> Option<i64> {
    match ty {
        Type::Scalar(element) => {
            crate::layout::scalar_size_align(element).map(|(size, _)| size as i64)
        }
        Type::Pointer(..) | Type::Borrow { .. } | Type::Ref(..) => Some(8),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlowered_closure_refuses_without_executing_its_deferred_body() {
        let span = Span::default();
        let outside: Symbol = "outside".into();
        let graph = TransferCostGraph::default();
        let env = HashMap::new();
        let comptime_bodies = HashMap::new();
        let syntax_functions: HashMap<Symbol, &Function> = HashMap::new();
        let mono_functions: Vec<(Function, u64)> = Vec::new();
        let mut interpreter = ComptimeInterpreter::new(
            &env,
            ComptimeFunctionBodies::new(
                &comptime_bodies,
                &syntax_functions,
                &mono_functions,
                &|_| false,
            ),
            &graph,
            Topology::CPU,
            ComptimeEvalContext::new(
                std::collections::HashSet::from([outside.clone()]),
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            ),
        );
        let body = Expr::ComptimeBlock(ComptimeBlockExpr::new(
            vec![Statement::Assign(AssignStmt::new(
                Expr::Identifier(IdentifierExpr::new(outside, span)),
                Expr::Number(NumberExpr::new("9".into(), None, span)),
                span,
            ))],
            None,
            span,
        ));
        let closure = Expr::Closure(ClosureExpr::new(vec![], Box::new(body), span));
        let block = [Statement::LetDecl(LetDeclStmt::new(
            "callback".into(),
            false,
            None,
            closure,
            span,
        ))];
        let tail = Expr::Number(NumberExpr::new("7".into(), None, span));

        let observation = interpreter.observe_block(&block, Some(&tail));

        assert!(observation.outcome.requires_refusal);
        assert_eq!(observation.escaping_write, None);
        assert_eq!(observation.outcome.value.concrete, Some(Value::Int(7)));
    }

    #[test]
    fn raw_statement_boundaries_refuse_before_a_foldable_tail() {
        let span = Span::default();
        let graph = TransferCostGraph::default();
        let env = HashMap::new();
        let comptime_bodies = HashMap::new();
        let syntax_functions: HashMap<Symbol, &Function> = HashMap::new();
        let mono_functions: Vec<(Function, u64)> = Vec::new();
        let tail = Expr::Number(NumberExpr::new("7".into(), None, span));
        let statements = [
            Statement::MacroCall(MacroCallStmt::new(
                "unexpanded".into(),
                TokenTree::Group(vec![]),
                None,
                true,
                span,
            )),
            Statement::Error(span),
        ];

        for statement in statements {
            let mut interpreter = ComptimeInterpreter::new(
                &env,
                ComptimeFunctionBodies::new(
                    &comptime_bodies,
                    &syntax_functions,
                    &mono_functions,
                    &|_| false,
                ),
                &graph,
                Topology::CPU,
                ComptimeEvalContext::default(),
            );
            let observation = interpreter.observe_block(&[statement], Some(&tail));

            assert!(observation.outcome.requires_refusal);
            assert_eq!(observation.outcome.value.concrete, Some(Value::Int(7)));
        }
    }

    #[test]
    fn raw_method_call_refuses_before_a_foldable_tail() {
        let span = Span::default();
        let graph = TransferCostGraph::default();
        let env = HashMap::new();
        let comptime_bodies = HashMap::new();
        let syntax_functions: HashMap<Symbol, &Function> = HashMap::new();
        let mono_functions: Vec<(Function, u64)> = Vec::new();
        let mut interpreter = ComptimeInterpreter::new(
            &env,
            ComptimeFunctionBodies::new(
                &comptime_bodies,
                &syntax_functions,
                &mono_functions,
                &|_| false,
            ),
            &graph,
            Topology::CPU,
            ComptimeEvalContext::default(),
        );
        let method = Expr::MethodCall(MethodCallExpr::new(
            Box::new(Expr::Number(NumberExpr::new("1".into(), None, span))),
            "unlowered".into(),
            None,
            vec![Expr::Number(NumberExpr::new("2".into(), None, span))],
            span,
        ));
        let block = [Statement::ExprStmt(ExprStmtStmt::new(method, true, span))];
        let tail = Expr::Number(NumberExpr::new("7".into(), None, span));

        let observation = interpreter.observe_block(&block, Some(&tail));

        assert!(observation.outcome.requires_refusal);
        assert_eq!(observation.outcome.value.concrete, Some(Value::Int(7)));
    }

    #[test]
    fn opaque_callable_records_captured_outer_writes() {
        let outside: Symbol = "outside".into();
        let graph = TransferCostGraph::default();
        let env = HashMap::new();
        let comptime_bodies = HashMap::new();
        let syntax_functions: HashMap<Symbol, &Function> = HashMap::new();
        let mono_functions: Vec<(Function, u64)> = Vec::new();
        let mut interpreter = ComptimeInterpreter::new(
            &env,
            ComptimeFunctionBodies::new(
                &comptime_bodies,
                &syntax_functions,
                &mono_functions,
                &|_| false,
            ),
            &graph,
            Topology::CPU,
            ComptimeEvalContext::new(
                std::collections::HashSet::from([outside.clone()]),
                std::collections::HashSet::new(),
                std::collections::HashSet::new(),
            ),
        );
        let facts = ComptimeValueFacts {
            unknown_callable: true,
            captured_writes: std::collections::HashSet::from([outside.clone()]),
            ..ComptimeValueFacts::default()
        };

        interpreter.note_opaque_callable(&facts, &[]);

        assert_eq!(interpreter.context.escaping_write(), Some(&outside));
    }
}
