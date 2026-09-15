//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for statements, verifying types, variable definitions, and scoping.
//
//===----------------------------------------------------------------------===//

use std::collections::HashMap;

use super::*;

use crate::hir;
use crate::syntax;

/// What running a statement did to the block it sits in.
///
/// The evaluator used to answer `Option<Value>`, which said "a return produced this" and
/// could not say "a return produced nothing the evaluator could compute", nor anything at
/// all about `break` and `continue`. Running a loop needs all four answers.
pub(crate) enum EvalFlow {
    /// Carry on with the next statement.
    Normal,
    /// A `return` ran. `None` when the evaluator could not compute what it returned -- the
    /// body still ends there, so the statements after it must not run.
    Return(Option<Value>),
    /// Leave the innermost loop.
    Break,
    /// Start the innermost loop's next iteration.
    Continue,
}

impl<'a> TypeChecker<'a> {
    /// Performs semantic analysis on a block of statements.
    ///
    /// A **block** is a sequence of statements enclosed in `{ ... }` that defines a new lexical scope.
    /// This includes function bodies, `if` branches, loop bodies, and raw blocks. Variables declared
    /// inside a block are dropped when the block ends.
    ///
    /// # Liveness Analysis (Block-Local)
    /// Before type-checking and executing the statements, we perform a single O(N) forward
    /// pass over the block to precompute the liveness of all variables within this lexical scope.
    ///
    /// **The Algorithm:**
    /// 1. We instantiate a `last_use` map (`HashMap<crate::symbol::Symbol, usize>`).
    /// 2. We iterate over the block's statements from `0` to `N-1`.
    /// 3. For each statement, we recursively extract all variable identifiers used in that
    ///    statement (`extract_uses_stmt`) and insert them into `last_use` with the current statement index `i`.
    /// 4. By the end of the pass, `last_use[var]` holds the exact index of the *last* statement
    ///    that references `var` within this block.
    /// 5. We hand this map to `self.borrow` (via `enter_block`) and advance its statement cursor with
    ///    `set_stmt`, so the borrow context carries the per-block liveness.
    ///
    /// This allows the Non-Lexical Lifetimes (NLL) borrow checker to query its `is_variable_used_after`
    /// in O(1) time instead of performing an O(N^2) AST tree-walk!
    pub(crate) fn check_block(&mut self, body: &mut [Statement], return_type: &Type) {
        let mut terminated = false;

        // 1. Liveness Analysis Pass
        let last_use = Self::compute_block_liveness(body);

        self.borrow.enter_block(last_use);

        // A block is always checked for real, never speculatively. Force `speculating` off for the
        // loop so a block reached from *inside* a return-type probe (e.g. a generic function body
        // instantiated during the methodcall probe) is still fully checked — reproducing the old
        // hard-coded `silent = false` this loop passed to `check_statement` (#279 R3).
        let saved_speculating = self.speculating;
        self.speculating = false;

        #[allow(clippy::needless_range_loop)]
        for i in 0..body.len() {
            self.borrow.set_stmt(i);
            if terminated {
                let stmt_span = body[i].span();
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1003,
                    "Unreachable code after return, break, or continue",
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&stmt_span)),
                );
                break; // Only warn once per block
            }

            self.check_statement(&mut body[i], return_type, true);
            match &body[i] {
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    terminated = true;
                }
                _ => {}
            }
        }
        self.speculating = saved_speculating;
        self.borrow.exit_block();
    }

    pub(crate) fn compute_block_liveness(
        body: &[Statement],
    ) -> HashMap<crate::symbol::Symbol, usize> {
        let mut last_use = HashMap::new();
        for (i, stmt) in body.iter().enumerate() {
            let mut uses = std::collections::HashSet::new();
            Self::extract_uses_stmt(stmt, &mut uses);
            for var in uses {
                last_use.insert(var.into(), i);
            }
        }
        last_use
    }

    pub(crate) fn check_statement(
        &mut self,
        stmt: &mut Statement,
        return_type: &Type,
        consume: bool,
    ) {
        match stmt {
            Statement::LetDecl(decl) => self.check_let_decl_stmt(decl, consume),
            Statement::ForLoop(floop) => self.check_for_loop_stmt(floop, consume, return_type),
            Statement::Loop(lp) => self.check_loop_stmt(lp, return_type),
            Statement::Break(_) => {}
            Statement::Continue(_) => {}
            Statement::Assign(AssignStmt { lhs, rhs, span: _ }) => {
                self.check_assign_stmt(lhs, None, rhs, consume)
            }
            Statement::CompoundAssign(CompoundAssignStmt {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                let op = op.clone();
                self.check_assign_stmt(lhs, Some(&op), rhs, consume)
            }
            Statement::Return(ret) => self.check_return_stmt(ret, consume, return_type),
            Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: _,
                span: _,
            }) => {
                // Taken before the arguments are checked: checking `&mut a` is what drops
                // `a`'s compile-time value, so afterwards there is nothing left to run the
                // call against.
                let before = self.consteval_snapshot();
                let scopes = self.consteval_scopes();
                let saved_borrows = self.borrow.snapshot();
                self.check_expr_type_flag(expr, consume);
                self.borrow.restore(saved_borrows);
                self.settle_mut_borrow_call(expr, &before, &scopes);
            }
            Statement::Assert(assert) => self.check_assert_stmt(assert, consume, return_type),
            Statement::MacroCall(_) => {
                self.errors.push("Macro failed to expand".to_string());
            }
            Statement::Error(_) => {}
        }
    }

    /// Check a `let` binding: type the initializer, bind the name (annotation wins), record the
    /// binding's reference provenance (#243), and register an equality constraint for an immutable.
    fn check_let_decl_stmt(&mut self, decl: &mut LetDeclStmt, consume: bool) {
        let LetDeclStmt {
            name,
            is_mut: _is_mut,
            ty_ann,
            expr,
            span,
        } = decl;
        // Track declared variables for W1001 (unused variable) detection
        self.declared_vars.push((name.clone(), *span));

        self.current_assignment_target = Some(name.to_string());
        // Forward the annotation as an expected-type hint so a generic call can
        // deduce a return-only topology/type variable from it.
        let prev_expected = self.expected_type.take();
        self.expected_type = ty_ann.clone();
        let ty = self.check_expr_type_flag(expr, consume);
        self.expected_type = prev_expected;
        self.current_assignment_target = None;

        let mut tmp_env = HashMap::new();
        for env in &self.consteval.env {
            for (k, v) in env {
                tmp_env.insert(k.clone(), v.clone());
            }
        }
        if let Some(val) = self.eval_expr(expr, &tmp_env) {
            self.consteval
                .env
                .last_mut()
                .unwrap()
                .insert(name.to_string().into(), val);
        }

        // A `transfer(..)` validates its destination and records the tile it lands, at its own
        // site. The binding adds nothing: checking its type as well reported the same overflow
        // twice, counted one tile as two, and re-litigated a destination the transfer had
        // already accepted.
        let initializer_places_its_own = matches!(expr, Expr::Transfer(_));
        let context = format!("variable '{}'", name);
        let binding_ty = if let Some(ann) = ty_ann {
            // `Unknown` is the poison type of an already-reported failure (an unresolved call, a
            // type parameter nothing binds); a mismatch against it would report that twice.
            if ty != Type::Unknown && !self.is_assignable(ann, &ty) {
                let splat = matches!(
                    (&*ann, &ty),
                    (Type::Tensor(el, dims, _), Type::Scalar(s)) if !dims.is_empty() && el == s
                );
                if splat {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3023,
                        format!(
                            "'{}' is a shaped tensor initialized from a scalar; write \
                             `Tensor<..>::fill(v)` to fill every element with it",
                            name
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                } else {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3001,
                        format!("'{name}' is declared {ann} but its initializer is {ty}"),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
            }
            // Capacity: a placed tensor annotation must fit its memory space.
            if !initializer_places_its_own {
                self.check_type_placement(ann, &context, span);
            }
            self.insert(name.to_string(), ann.clone());
            ann.clone()
        } else {
            // A placement is in the type whether or not the binding repeats it as an annotation,
            // so the inferred type gets the same check. Without this, every placement written
            // only in the initializer -- the ordinary spelling -- escaped capacity admission,
            // element-type admission, and the working set.
            if !initializer_places_its_own {
                self.check_type_placement(&ty, &context, span);
            }
            self.insert(name.to_string(), ty.clone());
            ty
        };

        // Record where a reference binding roots, so a later `return` of it can be
        // checked for escape (#243). A binding whose provenance we can't determine is
        // left unrecorded rather than assumed safe-or-unsafe.
        if Self::is_ref_type(&binding_ty) {
            if let Some(prov) = self.ref_provenance_of(expr) {
                self.borrow.ref_provenance.insert(name.clone(), prov);
            }
        }

        // Shadowing: this binding hides any earlier one with the same name, so earlier
        // facts about the name are now about a dead binding and must not participate in
        // proofs (a stale `i == 1` beside a new `i == 9` proves anything). Neutralized
        // for mutable bindings too -- they record no new fact, but they still shadow.
        self.neutralize_facts_mentioning(name.as_ref());
        if !*_is_mut {
            let id_expr = Expr::Identifier(IdentifierExpr {
                name: name.clone(),
                span: Span::default(),
            });
            // `raw::extent(t)` folds to its literal element count first: the prover cannot
            // lower a function call, so `let n = raw::extent(src)` would otherwise record
            // an unlowerable fact and every bound written against `n` would be unprovable
            // (Vx#353 A2).
            let eq_expr = Expr::RelationalOp(RelationalOpExpr {
                lhs: Box::new(id_expr),
                op: RelationalOp::Eq,
                rhs: Box::new(self.fold_raw_extent(expr)),
                span: Span::default(),
            });
            self.consteval.constraints.push(eq_expr);
        }
    }

    /// Check a `for` loop: type the iterable, bind the induction variable's element type, prove
    /// loop invariants on entry, and check the body.
    fn check_for_loop_stmt(&mut self, floop: &mut ForLoopStmt, consume: bool, return_type: &Type) {
        let ForLoopStmt {
            iter,
            iterable,
            invariants,
            body,
            span: loop_span,
        } = floop;
        let loop_span = *loop_span;
        let iterable_ty = self.check_expr_type_flag(iterable, consume);
        self.push_releasing_scope();

        // If it's Range, it's I64. If it's Iterator, we extract from Option<T>
        // If it's Tensor, we extract the ElementType
        let mut iter_ty = Type::Scalar(ElementType::I64); // fallback

        // Check if it's a generic iterator by synthesizing a `.next()` call
        if matches!(iterable_ty, Type::GenericInstance(..))
            || matches!(iterable_ty, Type::Struct(..))
        {
            use syntax::expr::{Expr, MethodCallExpr};
            use syntax::Span;
            let mut next_call = Expr::MethodCall(MethodCallExpr {
                base: (*iterable).clone(),
                method_name: "next".to_string().into(),
                type_args: None,
                args: vec![],
                span: Span::default(),
            });
            // This will resolve and monomorphize `next`! Its result is `Option<Element>`, so
            // the loop variable takes the payload type. The `Option` base may be spelled
            // `Enum` *or* `Struct` after resolution — accept both, else the element type is
            // lost and the loop variable wrongly falls back to `i64` (E3004 against an i32
            // body, the for-over-iterator typing bug, #242).
            let opt_ty = self.check_expr_type_flag(&mut next_call, consume);
            if let Type::GenericInstance(base, args) = opt_ty {
                if let Type::Enum(name, _) | Type::Struct(name, _) = &*base {
                    if name.as_ref() == "Option" && args.len() == 1 {
                        iter_ty = args[0].clone();
                    }
                }
            }
        } else {
            iter_ty = match iterable_ty {
                Type::GenericInstance(base, args) => {
                    if let Type::Enum(name, _) | Type::Struct(name, _) = &*base {
                        if name.as_ref() == "Option" && args.len() == 1 {
                            args[0].clone()
                        } else {
                            Type::Scalar(ElementType::I64)
                        }
                    } else {
                        Type::Scalar(ElementType::I64)
                    }
                }
                Type::Tensor(el_ty, _, _) => Type::Scalar(el_ty),
                // A scalar iterable is an integer range (`a..b`); the induction variable takes
                // the range's element type, so `for i in 0..10` binds `i: i32` and code like
                // `sum + i` / `return i` type-checks without a coercion (#240). The flat
                // lowerer already types the loop var from the same range bound.
                Type::Scalar(e) => Type::Scalar(e),
                _ => Type::Scalar(ElementType::I64),
            };
        }

        self.insert(iter.clone(), iter_ty); // Still assuming i64 for most things, but it works for our current test cases.

        // Prove invariants hold on entry, then assume them inside the loop
        let prev_constraints_len = self.consteval.constraints.len();

        // A range loop constrains its induction variable: `for i in a..b` gives
        // `a <= i && i < b` for the body. Recorded as prover facts so bounds obligations
        // over `i` (the `raw::` primitives, Vx#353 A2) and `Verified<T>` assertions can
        // close without a hand-written invariant. Skipped when the body reassigns the
        // variable: the checker does not version mutated symbols, so a stale fact would
        // prove false things. The facts sit above `prev_constraints_len`, so the
        // truncation below drops them at loop exit with the invariants.
        // The induction variable shadows any outer binding of the same name; earlier
        // facts about that name are about the dead binding now (see
        // neutralize_facts_mentioning -- a stale outer `i < 2` under an inner
        // `for i in 0..8` forged an out-of-bounds proof in review).
        self.neutralize_facts_mentioning(iter);
        if let Expr::Range(r) = &**iterable {
            if !crate::hir::check::raw::body_reassigns(body, iter) {
                let iter_expr = Expr::Identifier(syntax::expr::IdentifierExpr {
                    name: iter.clone().into(),
                    span: Span::default(),
                });
                let lo = self.fold_raw_extent(&r.start);
                let hi = self.fold_raw_extent(&r.end);
                // Only facts the prover can lower are recorded; an unlowerable bound
                // (a call) would make every later proof in the function warn.
                if crate::hir::check::raw::prover_expressible(&lo) {
                    self.consteval
                        .constraints
                        .push(Expr::RelationalOp(RelationalOpExpr {
                            lhs: Box::new(iter_expr.clone()),
                            op: RelationalOp::Ge,
                            rhs: Box::new(lo),
                            span: Span::default(),
                        }));
                }
                if crate::hir::check::raw::prover_expressible(&hi) {
                    self.consteval
                        .constraints
                        .push(Expr::RelationalOp(RelationalOpExpr {
                            lhs: Box::new(iter_expr),
                            op: RelationalOp::Lt,
                            rhs: Box::new(hi),
                            span: Span::default(),
                        }));
                }
            }
        }
        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven on entry".to_string());
            }
            self.consteval.constraints.push(inv.clone());
        }

        // What is known before the body is walked, and where each name lives. Both are
        // needed after: the walk folds one iteration's worth of assignments into the
        // environment, and settling that up requires the state it started from.
        let before = self.consteval_snapshot();
        let scopes = self.consteval_scopes();

        self.check_block(body, return_type);

        // Check invariants hold after the loop iteration (we don't strictly prove induction here, just checking at end of block)
        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven to hold across iterations".to_string());
            }
        }

        let mut after = before.clone();
        let ran = self.run_loop_for_consteval(&mut after, |checker, env| {
            checker.eval_for_loop(iter, iterable, body, env)
        });
        self.settle_loop_consteval(&before, &scopes, ran, &after);
        self.report_steps_exceeded(&loop_span);

        self.consteval.constraints.truncate(prev_constraints_len);
        self.pop_scope();
    }

    /// Check an infinite `loop`: prove invariants on entry, check the body, then re-prove them
    /// across iterations.
    fn check_loop_stmt(&mut self, lp: &mut LoopStmt, return_type: &Type) {
        let LoopStmt {
            body,
            span: loop_span,
            invariants,
        } = lp;
        let loop_span = *loop_span;
        self.push_releasing_scope();

        let prev_constraints_len = self.consteval.constraints.len();
        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven on entry".to_string());
            }
            self.consteval.constraints.push(inv.clone());
        }

        // See `check_for_loop_stmt`: the single pass the checker makes over the body is not
        // what the program does, so what it folded is settled up once the loop has been run.
        let before = self.consteval_snapshot();
        let scopes = self.consteval_scopes();

        self.check_block(body, return_type);

        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven to hold across iterations".to_string());
            }
        }

        let mut after = before.clone();
        let ran =
            self.run_loop_for_consteval(&mut after, |checker, env| checker.eval_loop(body, env));
        self.settle_loop_consteval(&before, &scopes, ran, &after);
        self.report_steps_exceeded(&loop_span);

        self.consteval.constraints.truncate(prev_constraints_len);
        self.pop_scope();
    }

    /// Check an assignment / compound assignment: type the LHS, then the RHS expecting the LHS
    /// type (#240), verify assignability, and fold a const RHS into the eval environment.
    /// `op` is `Some` for a compound assignment, and carries the operator that sits between
    /// the two sides. It has to be checked here as well as in `check_binaryop_expr`: `a %= b`
    /// never builds a `BinaryOp` expression for that rule to see.
    fn check_assign_stmt(
        &mut self,
        lhs: &mut Expr,
        op: Option<&BinaryOp>,
        rhs: &mut Expr,
        consume: bool,
    ) {
        self.checking_assign_lhs = true;
        let lhs_ty = self.check_expr_type_flag(lhs, false);
        self.checking_assign_lhs = false;

        // Determine target name for NLL
        if let Expr::Identifier(id) = lhs {
            self.current_assignment_target = Some(id.name.to_string());
        } else if let Expr::MemberAccess(ma) = lhs {
            if let Expr::Identifier(id) = &*ma.base {
                self.current_assignment_target = Some(id.name.to_string());
            }
        }

        // Check the RHS expecting the target's type, so an untyped literal is born at that
        // type (`a[i] = 1.0` into a bf16 tensor, `r = 5` into an i64 slot) rather than
        // defaulting and mismatching (#240).
        let rhs_ty = self.check_expr_expecting(rhs, Some(lhs_ty.clone()), consume);
        self.current_assignment_target = None;
        if let Some(op) = op {
            let span = lhs.span();
            if !self.check_restricted_operands(op, &lhs_ty, &rhs_ty, &span) {
                return;
            }
        }
        if !self.is_assignable(&lhs_ty, &rhs_ty) {
            // Named types and a location, like every other type error: this fires on a
            // narrowing store into half storage, where the fix is to write the `as` the
            // language requires rather than to guess what was meant.
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3004,
                format!(
                    "Type mismatch in assignment: cannot assign {} to {}; write an explicit \
                     `as` cast if the conversion is intended",
                    rhs_ty, lhs_ty
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&lhs.span())),
            );
        }

        // Keep the compile-time value of the assigned variable in step with the assignment.
        // An assignment the evaluator cannot compute makes the variable unknown rather than
        // leaving the previous value behind, which later reads would report as a certainty.
        match lhs {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => {
                let tmp_env = self.consteval_snapshot();
                let new_val = self.eval_expr(rhs, &tmp_env);
                if let Some(scope) = self.consteval_scope_of(name.as_ref()) {
                    let env = &mut self.consteval.env[scope];
                    match new_val {
                        Some(val) => env.insert(name.to_string().into(), val),
                        None => env.remove(name.as_ref()),
                    };
                }
            }
            // `a[i] = v`: replace that one element. Anything unknown -- the index, the new
            // value, or the array -- drops the whole array instead of leaving it stale.
            Expr::IndexAccess(IndexAccessExpr {
                base,
                index,
                span: _,
            }) => {
                if let Some(root) = Self::place_root(base) {
                    // Only a write straight into a variable is carried out. A nested place
                    // like `a[i][j]` is not, so the array it belongs to becomes unknown.
                    // No compiling program reaches this today, because a nested array
                    // literal is refused by code generation; it guards the evaluator from
                    // reporting a stale element if that ever changes.
                    let direct = matches!(&**base, Expr::Identifier(_));
                    let tmp_env = self.consteval_snapshot();
                    let index_val = if direct {
                        self.eval_expr(index, &tmp_env)
                    } else {
                        None
                    };
                    let new_val = if direct {
                        self.eval_expr(rhs, &tmp_env)
                    } else {
                        None
                    };
                    if let Some(scope) = self.consteval_scope_of(root.as_ref()) {
                        let env = &mut self.consteval.env[scope];
                        let stored = match (index_val, new_val) {
                            (Some(index_val), Some(val)) => match env.get_mut(root.as_ref()) {
                                Some(Value::Array(items)) => {
                                    match Self::array_index(&index_val, items.len()) {
                                        Some(i) => {
                                            items[i] = val;
                                            true
                                        }
                                        None => false,
                                    }
                                }
                                _ => false,
                            },
                            _ => false,
                        };
                        if !stored {
                            env.remove(root.as_ref());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Which constant scope each known name currently lives in, innermost winning.
    ///
    /// Taken before a loop body is type checked, so that a variable the single pass drops
    /// can still be put back afterwards. `consteval_scope_of` could not find it by then:
    /// the name is gone from every scope, which is exactly the case that matters.
    pub(crate) fn consteval_scopes(&self) -> HashMap<crate::symbol::Symbol, usize> {
        let mut scopes = HashMap::new();
        for (index, scope) in self.consteval.env.iter().enumerate() {
            for name in scope.keys() {
                scopes.insert(name.clone(), index);
            }
        }
        scopes
    }

    /// Settle what the constant environment knows about the variables a loop body writes.
    ///
    /// Type checking walks the body once and folds whatever it can, so afterwards the
    /// environment holds the value after a single iteration. That is not what the program
    /// does. Every name that single pass changed is replaced here with the value the loop
    /// really produces, or dropped when the evaluator could not run the loop -- because the
    /// alternative is reporting the one-iteration value as a certainty, which is how a
    /// correct program comes to be rejected.
    fn settle_loop_consteval(
        &mut self,
        before: &HashMap<crate::symbol::Symbol, Value>,
        scopes: &HashMap<crate::symbol::Symbol, usize>,
        ran: bool,
        after: &HashMap<crate::symbol::Symbol, Value>,
    ) {
        // Whatever the single pass changed, added or dropped is what it touched. Comparing
        // the environment against itself this way needs no list of the statement forms that
        // can write to a variable -- a list that would silently go stale.
        let folded = self.consteval_snapshot();
        let mut touched: Vec<crate::symbol::Symbol> = Vec::new();
        for (name, value) in before {
            if folded.get(name.as_ref()) != Some(value) {
                touched.push(name.clone());
            }
        }
        for name in folded.keys() {
            if !before.contains_key(name.as_ref()) {
                touched.push(name.clone());
            }
        }

        for name in touched {
            let Some(&scope) = scopes.get(name.as_ref()) else {
                // Declared inside the body. Its scope is about to be popped, so there is
                // nothing outside the loop that could read it.
                continue;
            };
            match after.get(name.as_ref()).filter(|_| ran) {
                Some(value) => self.consteval.env[scope].insert(name.clone(), value.clone()),
                None => self.consteval.env[scope].remove(name.as_ref()),
            };
        }
    }

    /// Run a loop for its compile-time value, and say whether the answer can be trusted.
    ///
    /// Only inside a `comptime` block. Elsewhere the value is not wanted, and walking a
    /// run-time loop's trip count would spend the compiler's time to learn nothing -- one
    /// benchmark in the corpus loops a million times, and evaluating it cost a second and
    /// an E8005 about a loop that was never meant to be evaluated.
    fn run_loop_for_consteval(
        &mut self,
        env: &mut HashMap<crate::symbol::Symbol, Value>,
        run: impl FnOnce(&Self, &mut HashMap<crate::symbol::Symbol, Value>) -> EvalFlow,
    ) -> bool {
        if self.consteval.comptime_depth == 0 {
            return false;
        }
        let outer_unsupported = self.consteval.unsupported_stmt.replace(false);
        let flow = run(self, env);
        // A `return` out of the loop leaves the rest of the function unreached, which the
        // checker goes on walking anyway; treating it as run would hand those statements a
        // state the program never arrives in.
        let ran = !self.consteval.unsupported_stmt.get()
            && !self.consteval.steps_exceeded.get()
            && matches!(flow, EvalFlow::Normal);
        self.consteval.unsupported_stmt.set(outer_unsupported);
        ran
    }

    /// Follow what a call written as a statement wrote through its mutable borrows.
    ///
    /// Taking `&mut a` drops `a`'s compile-time value, because whoever holds the borrow can
    /// write through it. The write can be followed now: the call runs with the borrowed
    /// argument bound to the value `a` held, and what the body leaves there is put back.
    /// When the body cannot be run the value stays dropped, which is the old behaviour and
    /// the safe one -- the callee has by now overwritten what the caller was holding.
    pub(crate) fn settle_mut_borrow_call(
        &mut self,
        expr: &Expr,
        before: &HashMap<crate::symbol::Symbol, Value>,
        scopes: &HashMap<crate::symbol::Symbol, usize>,
    ) {
        if self.consteval.comptime_depth == 0 {
            return;
        }
        let Expr::FunctionCall(call) = expr else {
            return;
        };
        let Some(written) = self.eval_call_effects(call, before) else {
            return;
        };
        for (place, value) in written {
            // The name was dropped when the borrow was taken, so its scope has to come
            // from the reading made before that happened.
            if let Some(&scope) = scopes.get(place.as_ref()) {
                self.consteval.env[scope].insert(place, value);
            }
        }
    }

    /// Report an evaluation stopped by the loop budget, and clear the flag so the next one
    /// starts fresh. Called where a value was asked for and a diagnostic can be raised.
    fn report_steps_exceeded(&mut self, span: &Span) {
        if !self.consteval.steps_exceeded.replace(false) {
            return;
        }
        self.consteval.loop_steps.set(0);
        if self.speculating {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8005,
            format!(
                "compile-time evaluation ran more than {} loop iterations and was stopped. A \
                 loop whose end condition is never reached is the usual cause.",
                crate::hir::check_state::MAX_LOOP_STEPS
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// The innermost constant scope holding `name`, if any.
    fn consteval_scope_of(&self, name: &str) -> Option<usize> {
        self.consteval
            .env
            .iter()
            .rposition(|env| env.contains_key(name))
    }

    /// Drop a variable's compile-time value. Used where something happened that the
    /// evaluator cannot follow, so that it stops claiming to know what the variable holds.
    pub(crate) fn consteval_forget(&mut self, name: &str) {
        if let Some(scope) = self.consteval_scope_of(name) {
            self.consteval.env[scope].remove(name);
        }
    }

    /// The variable a place expression writes through: `a` for `a`, `a[i]` and `a[i][j]`.
    fn place_root(expr: &Expr) -> Option<&crate::symbol::Symbol> {
        match expr {
            Expr::Identifier(IdentifierExpr { name, span: _ }) => Some(name),
            Expr::IndexAccess(IndexAccessExpr { base, .. }) => Self::place_root(base),
            _ => None,
        }
    }

    /// Check a `return`: type the returned expression against the declared return type, run the
    /// return-escape analysis (#243), and bind `return` for `ensures` constraints.
    fn check_return_stmt(&mut self, ret: &mut ReturnStmt, consume: bool, return_type: &Type) {
        let ReturnStmt { expr, span } = ret;

        // `return;` -- valid only where there is no value to return. Everything below types a
        // returned expression, so there is nothing left to do once the function type is checked.
        let Some(expr) = expr else {
            if !crate::syntax::is_void_ty(return_type) {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3002,
                    format!("this function returns {return_type}, so `return` needs a value"),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
            return;
        };

        let prev_expected = self.expected_type.take();
        self.expected_type = Some(return_type.clone());
        let ty = self.check_expr_type_flag(expr, consume);
        self.expected_type = prev_expected;

        let mut expected_ty = return_type.clone();
        if let Some(Type::Unknown) = self.current_return_type {
            self.current_return_type = Some(ty.clone());
            expected_ty = ty.clone();
        }

        // `Unknown` is the poison type of an already-reported resolution failure (an undefined
        // variable, a name tombstoned by the import merge): a mismatch against it is pure noise
        // that misdirects from the root cause, so only genuinely-typed values are checked (#294).
        if ty != Type::Unknown && !self.is_assignable(&expected_ty, &ty) {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3002,
                format!(
                    "Type mismatch on return. Expected {}, got {}",
                    expected_ty, ty
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }

        // Return-escape analysis (#243, bc4): a returned reference must root in
        // caller-owned memory. Returning a reference to a function-local — `return &x`
        // for a local `x`, or a binding that reborrows one — leaves a dangling pointer
        // once this frame unwinds.
        if !self.speculating
            && Self::is_ref_type(&ty)
            && self.ref_provenance_of(expr) == Some(crate::hir::env::RefProvenance::Local)
        {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E4005,
                "Cannot return a reference to a local value: it would dangle after the \
                         function returns. A returned reference must borrow from a reference \
                         parameter, not a local."
                    .to_string(),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }

        // Bind 'return' to this expression in the constraints so `ensures` clauses can use it
        let return_ident = Expr::Identifier(IdentifierExpr {
            name: "return".to_string().into(),
            span: *span,
        });
        let return_eq = Expr::RelationalOp(RelationalOpExpr {
            lhs: Box::new(return_ident),
            op: RelationalOp::Eq,
            rhs: Box::new(expr.clone()),
            span: *span,
        });
        self.consteval.return_constraints.push(return_eq);
    }

    /// Check an `assert`: require a boolean condition, evaluate it at comptime when possible, and
    /// either flag a failure or fold it into the SMT constraints (Verified<T> discharge).
    fn check_assert_stmt(&mut self, assert: &mut AssertStmt, consume: bool, return_type: &Type) {
        let AssertStmt { expr, msg, span } = assert;
        let ty = self.check_expr_type_flag(expr, consume);
        if ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Assertion condition must be boolean".to_string());
        }

        let is_verified = matches!(return_type, Type::Verified(_));
        let mut tmp_env = HashMap::new();
        for env in &self.consteval.env {
            for (k, v) in env {
                tmp_env.insert(k.clone(), v.clone());
            }
        }
        let eval_res = self.eval_expr(expr, &tmp_env);
        self.report_depth_exceeded(span);
        self.report_steps_exceeded(span);

        if let Some(Value::Bool(b)) = eval_res {
            if !b {
                let m = msg
                    .clone()
                    .unwrap_or_else(|| "Comptime assertion failed".to_string());
                if is_verified {
                    self.errors
                        .push(format!("Contract violated for Verified return type: {}", m));
                } else {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E8002,
                        format!("Comptime assert failed: {}", m),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
            }
        } else if is_verified {
            // Try to prove mathematically using our SMT constraints
            if !self.prove_expr(expr) {
                self.errors
                    .push("Cannot statically prove assertion for Verified return type".to_string());
            }
        } else {
            // It's a standard dynamic assert, add it to our mathematical constraints
            // so we can prove future Verified<T> return conditions!
            self.consteval.constraints.push(*expr.clone());
        }
    }

    pub(crate) fn prove_expr(&mut self, expr: &Expr) -> bool {
        let mut prover = hir::prover::SmtProver::new();
        for constraint in &self.consteval.constraints {
            if let Err(e) = prover.add_constraint(constraint) {
                // If we can't lower a constraint, we log a warning
                self.errors
                    .push_warning(format!("Could not add constraint to SMT solver: {}", e));
            }
        }

        // To prove `expr` holds under `constraints`, we assert `!expr` and check for unsatisfiability.
        let negated_expr = Expr::UnaryOp(UnaryOpExpr {
            op: UnaryOp::Not,
            expr: Box::new(expr.clone()),
            span: Span::default(),
        });

        if let Err(e) = prover.add_constraint(&negated_expr) {
            self.errors
                .push_warning(format!("Could not lower expression to SMT solver: {}", e));
            return false; // Can't prove
        }

        match prover.prove() {
            Ok(is_sat) => !is_sat, // If unsat, then the expression is proven (valid)
            Err(e) => {
                self.errors.push_warning(format!("SMT solver error: {}", e));
                false
            }
        }
    }

    pub(crate) fn eval_expr(
        &self,
        expr: &Expr,
        env: &HashMap<crate::symbol::Symbol, Value>,
    ) -> Option<Value> {
        match expr {
            Expr::Number(NumberExpr {
                value: n_str,
                ty,
                span: _,
            }) => {
                // An integer literal stays an integer. The suffix decides when there is one;
                // without a suffix the spelling does, since a whole number written without a
                // point is an integer everywhere else in the language.
                let is_float = match ty {
                    Some(t) => t.is_float(),
                    None => n_str.contains('.') || n_str.contains('e') || n_str.contains('E'),
                };
                if !is_float {
                    if let Ok(i) = n_str.parse::<i64>() {
                        return Some(Value::Int(i));
                    }
                }
                n_str.parse::<f64>().ok().map(Value::Number)
            }
            // `Reachable<A, B>`: true iff a transfer path exists in the cost graph. Topology
            // variables have already been substituted during monomorphization.
            Expr::TransferPredicate(e) => {
                let mfrom = self.transfer_cost_graph.default_memory_for(&e.from);
                let mto = self.transfer_cost_graph.default_memory_for(&e.to);
                Some(Value::Bool(
                    self.transfer_cost_graph
                        .transfer_path(&mfrom, &mto)
                        .is_some(),
                ))
            }
            Expr::Identifier(IdentifierExpr { name: n, span: _ }) if n.as_ref() == "true" => {
                Some(Value::Bool(true))
            }
            Expr::Identifier(IdentifierExpr { name: n, span: _ }) if n.as_ref() == "false" => {
                Some(Value::Bool(false))
            }

            Expr::Identifier(IdentifierExpr { name: n, span: _ }) => env.get(n.as_ref()).cloned(),
            Expr::BinaryOp(BinaryOpExpr {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                // Two integers stay integers, so `/` truncates the way the emitted code does
                // and a large value keeps every bit. A result that overflows has no value
                // rather than a wrapped one, which would be baked into the program unnoticed.
                if let (Value::Int(a), Value::Int(b)) = (&l, &r) {
                    let (a, b) = (*a, *b);
                    return match op {
                        BinaryOp::Add => a.checked_add(b).map(Value::Int),
                        BinaryOp::Sub => a.checked_sub(b).map(Value::Int),
                        BinaryOp::Mul => a.checked_mul(b).map(Value::Int),
                        // The checked forms answer None for a zero divisor, and for the one
                        // signed division that overflows.
                        BinaryOp::Div => a.checked_div(b).map(Value::Int),
                        BinaryOp::Rem => a.checked_rem(b).map(Value::Int),
                        // The bit pattern is exact now, but `>>` reads it one way for a
                        // signed type and another for an unsigned one, and this value does
                        // not record which it came from. Left unfolded until it does.
                        BinaryOp::BitAnd
                        | BinaryOp::BitOr
                        | BinaryOp::BitXor
                        | BinaryOp::Shl
                        | BinaryOp::Shr => None,
                        BinaryOp::MatMul => None,
                    };
                }
                // Anything else numeric is float arithmetic, an integer mixed with a float
                // included.
                let (a, b) = (l.as_f64()?, r.as_f64()?);
                match op {
                    BinaryOp::Add => Some(Value::Number(a + b)),
                    BinaryOp::Sub => Some(Value::Number(a - b)),
                    BinaryOp::Mul => Some(Value::Number(a * b)),
                    BinaryOp::Div => Some(Value::Number(a / b)),
                    BinaryOp::Rem => (b != 0.0).then(|| Value::Number(a % b)),
                    // A bit pattern read out of an `f64` would not be the bit pattern the
                    // program is talking about. Left unfolded rather than folded wrongly.
                    BinaryOp::BitAnd
                    | BinaryOp::BitOr
                    | BinaryOp::BitXor
                    | BinaryOp::Shl
                    | BinaryOp::Shr => None,
                    // MatMul not supported for pure numbers at compile time
                    BinaryOp::MatMul => None,
                }
            }
            Expr::RelationalOp(RelationalOpExpr {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                // Two integers compare as integers. Comparing them as floats makes every
                // pair past 2^53 look equal.
                if let (Value::Int(a), Value::Int(b)) = (&l, &r) {
                    let (a, b) = (*a, *b);
                    return Some(Value::Bool(match op {
                        RelationalOp::Eq => a == b,
                        RelationalOp::NotEq => a != b,
                        RelationalOp::Lt => a < b,
                        RelationalOp::Gt => a > b,
                        RelationalOp::Le => a <= b,
                        RelationalOp::Ge => a >= b,
                    }));
                }
                if let (Some(a), Some(b)) = (l.as_f64(), r.as_f64()) {
                    return Some(Value::Bool(match op {
                        RelationalOp::Eq => a == b,
                        RelationalOp::NotEq => a != b,
                        RelationalOp::Lt => a < b,
                        RelationalOp::Gt => a > b,
                        RelationalOp::Le => a <= b,
                        RelationalOp::Ge => a >= b,
                    }));
                }
                match (l, r, op) {
                    (Value::Bool(a), Value::Bool(b), RelationalOp::Eq) => Some(Value::Bool(a == b)),
                    (Value::Bool(a), Value::Bool(b), RelationalOp::NotEq) => {
                        Some(Value::Bool(a != b))
                    }
                    (Value::Topology(a), Value::Topology(b), RelationalOp::Eq) => {
                        Some(Value::Bool(self.topologies_equal(&a, &b)))
                    }
                    (Value::Topology(a), Value::Topology(b), RelationalOp::NotEq) => {
                        Some(Value::Bool(!self.topologies_equal(&a, &b)))
                    }
                    _ => None,
                }
            }
            Expr::LogicalOp(LogicalOpExpr {
                lhs,
                op,
                rhs,
                span: _,
            }) => {
                let l = self.eval_expr(lhs, env)?;
                let r = self.eval_expr(rhs, env)?;
                match (l, r, op) {
                    (Value::Bool(a), Value::Bool(b), LogicalOp::And) => Some(Value::Bool(a && b)),
                    (Value::Bool(a), Value::Bool(b), LogicalOp::Or) => Some(Value::Bool(a || b)),
                    _ => None,
                }
            }
            Expr::UnaryOp(UnaryOpExpr {
                op: UnaryOp::Not,
                expr: inner,
                span: _,
            }) => {
                if let Value::Bool(b) = self.eval_expr(inner, env)? {
                    Some(Value::Bool(!b))
                } else {
                    None
                }
            }
            // `[ a, b, c ]`. Every element has to be known, or the whole array is unknown:
            // a half-built array would let a later index read a value that was never there.
            Expr::Array(ArrayExpr { elements, span: _ }) => {
                let mut items = Vec::with_capacity(elements.len());
                for element in elements {
                    items.push(self.eval_expr(element, env)?);
                }
                Some(Value::Array(items))
            }
            // `a[i]`, where both the array and the index are known at compile time. An index
            // past the end gives no value here; `check_indexaccess_expr` reports it.
            Expr::IndexAccess(IndexAccessExpr {
                base,
                index,
                span: _,
            }) => {
                let Value::Array(items) = self.eval_expr(base, env)? else {
                    return None;
                };
                let index_val = self.eval_expr(index, env)?;
                let i = Self::array_index(&index_val, items.len())?;
                Some(items[i].clone())
            }
            Expr::FunctionCall(FunctionCallExpr {
                name,
                type_args: None,
                args,
                span: _,
            }) => {
                let func = self.callee_body(name.as_ref())?;
                let mut local_env = HashMap::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let arg_val = self.eval_expr(arg_expr, env)?;
                    local_env.insert(func.params[i].0.clone(), arg_val);
                }
                self.enter_call()?;
                let outer_unsupported = self.consteval.unsupported_stmt.replace(false);
                let mut result = None;
                if let EvalFlow::Return(ret_val) = self.eval_block(&func.body, &mut local_env) {
                    result = ret_val;
                }
                // A body holding a statement the evaluator cannot run has not been run.
                // Answering with what the statements it could run left behind would be a
                // guess, and a guess here is reported as a certainty.
                if self.consteval.unsupported_stmt.get() {
                    result = None;
                }
                self.consteval.unsupported_stmt.set(outer_unsupported);
                self.leave_call();
                result
            }
            Expr::Topology(TopologyExpr { top, span: _ }) => {
                if matches!(top, Topology::Current) {
                    Some(Value::Topology(self.active_topology.clone()))
                } else {
                    Some(Value::Topology(top.clone()))
                }
            }
            Expr::If(IfExpr {
                cond,
                then_block,
                else_block,
                span: _,
                is_comptime: _,
            }) => {
                if let Some(Value::Bool(cond_val)) = self.eval_expr(cond, env) {
                    let block = if cond_val {
                        then_block
                    } else if let Some(e) = else_block {
                        e
                    } else {
                        return None;
                    };
                    let mut ret = None;
                    let mut local_env = env.clone();
                    for stmt in block {
                        if let Statement::ExprStmt(ExprStmtStmt {
                            expr: e,
                            has_semi,
                            span: _,
                        }) = stmt
                        {
                            let val = self.eval_expr(e, &local_env);
                            if !*has_semi {
                                ret = val;
                            }
                        } else if let EvalFlow::Return(val) =
                            self.eval_statement(stmt, &mut local_env)
                        {
                            ret = val;
                        }
                    }
                    ret
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// The caller-side variable each argument passed by mutable borrow writes through.
    ///
    /// Two spellings reach the same place. `f(&mut a)` takes the borrow at the call, and
    /// `partition(w, ..)` passes on a borrow the caller already holds. Both are answered
    /// here as the name whose value the callee can change.
    fn mut_borrow_args(func: &Function, args: &[Expr]) -> Vec<(usize, crate::symbol::Symbol)> {
        let mut borrowed = Vec::new();
        for (i, arg) in args.iter().enumerate() {
            let Some((_, param_ty)) = func.params.get(i) else {
                continue;
            };
            if !matches!(param_ty, Type::Borrow { is_mut: true, .. }) {
                continue;
            }
            let place = match arg {
                Expr::Borrow(BorrowExpr {
                    expr,
                    is_mut: true,
                    span: _,
                }) => &**expr,
                other => other,
            };
            if let Expr::Identifier(IdentifierExpr { name, span: _ }) = place {
                borrowed.push((i, name.clone()));
            }
        }
        borrowed
    }

    /// Run a call for what it writes through its mutable borrows, rather than for a value.
    ///
    /// Answers the new value of each borrowed argument, or `None` when the body could not
    /// be run -- in which case the caller must drop those values rather than keep the ones
    /// from before the call, which the callee has by now overwritten.
    pub(crate) fn eval_call_effects(
        &self,
        call: &FunctionCallExpr,
        env: &HashMap<crate::symbol::Symbol, Value>,
    ) -> Option<Vec<(crate::symbol::Symbol, Value)>> {
        let FunctionCallExpr {
            name,
            type_args: None,
            args,
            span: _,
        } = call
        else {
            return None;
        };
        let func = self.callee_body(name.as_ref())?;
        let borrowed = Self::mut_borrow_args(func, args);
        if borrowed.is_empty() {
            return None;
        }

        // A borrowed argument is bound to the value it names, so the body's writes land on
        // it. Every other argument is passed the ordinary way, by value.
        let mut local_env = HashMap::new();
        for (i, arg_expr) in args.iter().enumerate() {
            let param = func.params.get(i)?.0.clone();
            let arg_val = match borrowed.iter().find(|(at, _)| *at == i) {
                Some((_, place)) => env.get(place.as_ref())?.clone(),
                None => self.eval_expr(arg_expr, env)?,
            };
            local_env.insert(param, arg_val);
        }

        self.enter_call()?;
        let outer_unsupported = self.consteval.unsupported_stmt.replace(false);
        self.eval_block(&func.body, &mut local_env);
        let ran = !self.consteval.unsupported_stmt.get();
        self.consteval.unsupported_stmt.set(outer_unsupported);
        self.leave_call();
        if !ran {
            return None;
        }

        let mut written = Vec::new();
        for (i, place) in borrowed {
            let param = func.params.get(i)?.0.as_ref();
            // The body may itself have dropped the value -- an unknown index, say. Then
            // there is nothing to write back and the caller's copy has to go too.
            written.push((place, local_env.get(param)?.clone()));
        }
        Some(written)
    }

    /// The callee's body for compile-time evaluation.
    ///
    /// The module being compiled goes into the resolution env with its non-generic bodies
    /// stripped, so a function defined alongside the caller has nothing to walk there and is
    /// looked up in the bodies kept for compile-time evaluation instead.
    fn callee_body(&self, name: &str) -> Option<&'a Function> {
        if let Some(func) = self.env.syntax_functions.get(name) {
            if !func.body.is_empty() {
                return Some(func);
            }
        }
        self.env.comptime_bodies.get(name)
    }

    /// Step one call deeper, or refuse. `None` stops the evaluation; whoever asked for the
    /// value reports it, since the evaluator cannot reach the diagnostics from `&self`.
    fn enter_call(&self) -> Option<()> {
        let depth = self.consteval.call_depth.get();
        if depth >= crate::hir::check_state::MAX_CALL_DEPTH {
            self.consteval.depth_exceeded.set(true);
            return None;
        }
        self.consteval.call_depth.set(depth + 1);
        Some(())
    }

    fn leave_call(&self) {
        let depth = self.consteval.call_depth.get();
        self.consteval.call_depth.set(depth.saturating_sub(1));
    }

    /// Report an evaluation that ran past the call limit, and clear the flag so the next one
    /// starts fresh. Called where a value was asked for and a diagnostic can be raised.
    fn report_depth_exceeded(&mut self, span: &Span) {
        if !self.consteval.depth_exceeded.replace(false) {
            return;
        }
        self.consteval.call_depth.set(0);
        if self.speculating {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8004,
            format!(
                "compile-time evaluation went more than {} calls deep and was stopped. A \
                 recursive function whose base case is never reached is the usual cause.",
                crate::hir::check_state::MAX_CALL_DEPTH
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// A compile-time array index. `None` for a fraction, a negative number, or an index
    /// at or past the end, so none of those can silently read the wrong element.
    pub(crate) fn array_index(value: &Value, len: usize) -> Option<usize> {
        let i = match value {
            Value::Int(i) => {
                if *i < 0 {
                    return None;
                }
                *i as usize
            }
            // A float index is only an index when it is a whole number. Rounding one onto a
            // neighbouring element would read a value the program never asked for.
            Value::Number(n) => {
                if n.fract() != 0.0 || *n < 0.0 {
                    return None;
                }
                *n as usize
            }
            _ => return None,
        };
        if i < len {
            Some(i)
        } else {
            None
        }
    }

    /// Write one element of a compile-time array. `None` when the index, the new value, or
    /// the array itself is not known; the caller then drops the whole array.
    fn eval_array_store(
        &self,
        name: &crate::symbol::Symbol,
        index: &Expr,
        rhs: &Expr,
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> Option<()> {
        let index_val = self.eval_expr(index, env)?;
        let value = self.eval_expr(rhs, env)?;
        let Some(Value::Array(items)) = env.get_mut(name.as_ref()) else {
            return None;
        };
        let i = Self::array_index(&index_val, items.len())?;
        items[i] = value;
        Some(())
    }

    /// Run every statement of a block, stopping at whatever leaves it early.
    pub(crate) fn eval_block(
        &self,
        stmts: &[Statement],
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> EvalFlow {
        for stmt in stmts {
            match self.eval_statement(stmt, env) {
                EvalFlow::Normal => {}
                leaves => return leaves,
            }
        }
        EvalFlow::Normal
    }

    /// Count one loop iteration, or refuse. `None` stops the evaluation; whoever asked for
    /// the value reports it, since the evaluator cannot reach the diagnostics from `&self`.
    fn step_loop(&self) -> Option<()> {
        let steps = self.consteval.loop_steps.get();
        if steps >= crate::hir::check_state::MAX_LOOP_STEPS {
            self.consteval.steps_exceeded.set(true);
            return None;
        }
        self.consteval.loop_steps.set(steps + 1);
        Some(())
    }

    /// Run `for <iter> in <iterable> { body }`.
    ///
    /// Only a range over two known integers is run. Anything else -- a tensor, an iterator,
    /// a bound the evaluator cannot compute -- is not something it can walk, so it says so
    /// rather than running some other number of iterations.
    fn eval_for_loop(
        &self,
        iter: &str,
        iterable: &Expr,
        body: &[Statement],
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> EvalFlow {
        let Expr::Range(range) = iterable else {
            self.consteval.unsupported_stmt.set(true);
            return EvalFlow::Normal;
        };
        let (Some(Value::Int(start)), Some(Value::Int(end))) = (
            self.eval_expr(&range.start, env),
            self.eval_expr(&range.end, env),
        ) else {
            self.consteval.unsupported_stmt.set(true);
            return EvalFlow::Normal;
        };

        // The induction variable shadows any outer binding of the same name, and the outer
        // one is live again after the loop. Put back rather than dropped: dropping it would
        // make a variable the loop never touched unknown from here on.
        let name: crate::symbol::Symbol = iter.to_string().into();
        let shadowed = env.get(name.as_ref()).cloned();
        let mut left = EvalFlow::Normal;

        let mut i = start;
        while i < end {
            if self.step_loop().is_none() {
                break;
            }
            env.insert(name.clone(), Value::Int(i));
            match self.eval_block(body, env) {
                EvalFlow::Normal | EvalFlow::Continue => {}
                EvalFlow::Break => break,
                ret @ EvalFlow::Return(_) => {
                    left = ret;
                    break;
                }
            }
            // A body holding a statement the evaluator cannot run has not been run, so the
            // iterations after this one would be built on a state that never existed.
            if self.consteval.unsupported_stmt.get() {
                break;
            }
            i += 1;
        }

        match shadowed {
            Some(val) => env.insert(name, val),
            None => env.remove(name.as_ref()),
        };
        left
    }

    /// Run `loop { body }`, which ends at a `break` or a `return` and otherwise runs until
    /// the iteration budget stops it.
    fn eval_loop(
        &self,
        body: &[Statement],
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> EvalFlow {
        loop {
            if self.step_loop().is_none() {
                return EvalFlow::Normal;
            }
            match self.eval_block(body, env) {
                EvalFlow::Normal | EvalFlow::Continue => {}
                EvalFlow::Break => return EvalFlow::Normal,
                ret @ EvalFlow::Return(_) => return ret,
            }
            if self.consteval.unsupported_stmt.get() {
                return EvalFlow::Normal;
            }
        }
    }

    pub(crate) fn eval_statement(
        &self,
        stmt: &Statement,
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> EvalFlow {
        match stmt {
            Statement::LetDecl(LetDeclStmt {
                name,
                is_mut: _,
                ty_ann: _,
                expr,
                span: _,
            }) => {
                // A binding the evaluator cannot compute has to become unknown. Leaving an
                // older binding of the same name in place would answer later reads with it.
                match self.eval_expr(expr, env) {
                    Some(val) => env.insert(name.clone(), val),
                    None => env.remove(name.as_ref()),
                };
                EvalFlow::Normal
            }
            Statement::Assign(AssignStmt {
                lhs: Expr::Identifier(IdentifierExpr { name, span: _ }),
                rhs,
                span: _,
            }) => {
                match self.eval_expr(rhs, env) {
                    Some(val) => env.insert(name.clone(), val),
                    None => env.remove(name.as_ref()),
                };
                EvalFlow::Normal
            }
            // `x op= v` is `x = x op v`. Run as that, so the arithmetic and the overflow
            // rules are the ones `eval_expr` already applies rather than a second copy.
            Statement::CompoundAssign(CompoundAssignStmt { lhs, op, rhs, span }) => {
                let folded = Expr::BinaryOp(BinaryOpExpr {
                    lhs: Box::new(lhs.clone()),
                    op: op.clone(),
                    rhs: Box::new(rhs.clone()),
                    span: *span,
                });
                self.eval_statement(
                    &Statement::Assign(AssignStmt {
                        lhs: lhs.clone(),
                        rhs: folded,
                        span: *span,
                    }),
                    env,
                )
            }
            // `a[i] = v`. If any part of the store is unknown the whole array is dropped:
            // keeping the old contents would report a stale element as a certainty.
            Statement::Assign(AssignStmt {
                lhs:
                    lhs @ Expr::IndexAccess(IndexAccessExpr {
                        base,
                        index,
                        span: _,
                    }),
                rhs,
                span: _,
            }) => {
                if let Some(root) = Self::place_root(lhs) {
                    // Only a write straight into a variable is carried out. A nested place
                    // like `a[i][j]` is not, so the array it belongs to becomes unknown.
                    let direct = matches!(&**base, Expr::Identifier(_));
                    let stored = direct && self.eval_array_store(root, index, rhs, env).is_some();
                    if !stored {
                        env.remove(root.as_ref());
                    }
                }
                EvalFlow::Normal
            }
            // A `return` ends the body whether or not its value could be computed. Carrying
            // on to the next statement would run code the program does not reach.
            Statement::Return(ReturnStmt { expr, span: _ }) => {
                EvalFlow::Return(expr.as_ref().and_then(|e| self.eval_expr(e, env)))
            }
            Statement::ForLoop(ForLoopStmt {
                iter,
                iterable,
                invariants: _,
                body,
                span: _,
            }) => self.eval_for_loop(iter, iterable, body, env),
            Statement::Loop(LoopStmt {
                body,
                invariants: _,
                span: _,
            }) => self.eval_loop(body, env),
            Statement::Break(_) => EvalFlow::Break,
            Statement::Continue(_) => EvalFlow::Continue,
            // A call written as a statement. It is run only when it writes through a
            // mutable borrow -- that is the whole reason to run something for no value.
            // Anything it borrowed must be dropped when it could not be run, because the
            // callee has by now overwritten what the caller was holding.
            Statement::ExprStmt(ExprStmtStmt {
                expr: Expr::FunctionCall(call),
                has_semi: _,
                span: _,
            }) => {
                let borrowed = match self.callee_body(call.name.as_ref()) {
                    Some(func) => Self::mut_borrow_args(func, &call.args),
                    None => Vec::new(),
                };
                if borrowed.is_empty() {
                    self.consteval.unsupported_stmt.set(true);
                    return EvalFlow::Normal;
                }
                match self.eval_call_effects(call, env) {
                    Some(written) => {
                        for (place, value) in written {
                            env.insert(place, value);
                        }
                    }
                    None => {
                        for (_, place) in borrowed {
                            env.remove(place.as_ref());
                        }
                        self.consteval.unsupported_stmt.set(true);
                    }
                }
                EvalFlow::Normal
            }
            // An `if` written as a statement, which is how a loop body decides anything.
            // The chosen block runs against this environment rather than a copy, so what
            // it writes is still there afterwards, and a `break` inside it reaches the
            // loop. `eval_expr` has its own arm for an `if` used as a *value*, which
            // cannot do either: it answers with a value and keeps its writes to itself.
            Statement::ExprStmt(ExprStmtStmt {
                expr:
                    Expr::If(IfExpr {
                        cond,
                        then_block,
                        else_block,
                        is_comptime: _,
                        span: _,
                    }),
                has_semi: _,
                span: _,
            }) => {
                let Some(Value::Bool(taken)) = self.eval_expr(cond, env) else {
                    self.consteval.unsupported_stmt.set(true);
                    return EvalFlow::Normal;
                };
                if taken {
                    self.eval_block(then_block, env)
                } else if let Some(otherwise) = else_block {
                    self.eval_block(otherwise, env)
                } else {
                    EvalFlow::Normal
                }
            }
            // Anything else: not run. Say so, so the call this body belongs to gives no
            // value rather than a half-executed one.
            _ => {
                self.consteval.unsupported_stmt.set(true);
                EvalFlow::Normal
            }
        }
    }

    fn topologies_equal(&self, a: &Topology, b: &Topology) -> bool {
        match (a, b) {
            (Topology::CPU, Topology::CPU) => true,
            (Topology::CpuAvx512, Topology::CpuAvx512) => true,
            (Topology::CpuNeon, Topology::CpuNeon) => true,
            (Topology::AMX, Topology::AMX) => true,
            (Topology::ANE, Topology::ANE) => true,
            (Topology::GPU(expr_a), Topology::GPU(expr_b)) => self.exprs_equal(expr_a, expr_b),
            (Topology::Current, Topology::Current) => true,
            (Topology::NPU(expr_a), Topology::NPU(expr_b)) => self.exprs_equal(expr_a, expr_b),
            (Topology::AccCore(expr_a), Topology::AccCore(expr_b)) => {
                self.exprs_equal(expr_a, expr_b)
            }
            (Topology::Slice(top_a, start_a, end_a), Topology::Slice(top_b, start_b, end_b)) => {
                self.topologies_equal(top_a, top_b)
                    && self.exprs_equal(start_a, start_b)
                    && self.exprs_equal(end_a, end_b)
            }
            _ => false,
        }
    }

    fn exprs_equal(&self, a: &Expr, b: &Expr) -> bool {
        // Try to evaluate both expressions to see if they result in the same value.
        // For static values like indices, this is much better than AST comparison.
        let empty_env = HashMap::new();
        if let (Some(val_a), Some(val_b)) =
            (self.eval_expr(a, &empty_env), self.eval_expr(b, &empty_env))
        {
            match (val_a, val_b) {
                (Value::Number(na), Value::Number(nb)) => return (na - nb).abs() < 1e-9,
                (Value::Bool(ba), Value::Bool(bb)) => return ba == bb,
                _ => {}
            }
        }

        // Fallback to structural AST matching for things we can't fully evaluate
        match (a, b) {
            (
                Expr::Number(NumberExpr { value: va, .. }),
                Expr::Number(NumberExpr { value: vb, .. }),
            ) => va == vb,
            (
                Expr::Identifier(IdentifierExpr { name: na, .. }),
                Expr::Identifier(IdentifierExpr { name: nb, .. }),
            ) => na == nb,
            _ => false,
        }
    }
}
