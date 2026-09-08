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
            Statement::Assign(AssignStmt { lhs, rhs, span: _ })
            | Statement::CompoundAssign(CompoundAssignStmt {
                lhs,
                op: _,
                rhs,
                span: _,
            }) => self.check_assign_stmt(lhs, rhs, consume),
            Statement::Return(ret) => self.check_return_stmt(ret, consume, return_type),
            Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: _,
                span: _,
            }) => {
                let saved_borrows = self.borrow.snapshot();
                self.check_expr_type_flag(expr, consume);
                self.borrow.restore(saved_borrows);
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
            if !self.is_assignable(ann, &ty) {
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
            span: _,
        } = floop;
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

        self.check_block(body, return_type);

        // Check invariants hold after the loop iteration (we don't strictly prove induction here, just checking at end of block)
        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven to hold across iterations".to_string());
            }
        }

        self.consteval.constraints.truncate(prev_constraints_len);
        self.pop_scope();
    }

    /// Check an infinite `loop`: prove invariants on entry, check the body, then re-prove them
    /// across iterations.
    fn check_loop_stmt(&mut self, lp: &mut LoopStmt, return_type: &Type) {
        let LoopStmt {
            body,
            span: _,
            invariants,
        } = lp;
        self.push_releasing_scope();

        let prev_constraints_len = self.consteval.constraints.len();
        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven on entry".to_string());
            }
            self.consteval.constraints.push(inv.clone());
        }

        self.check_block(body, return_type);

        for inv in invariants.iter() {
            if !self.prove_expr(inv) {
                self.errors
                    .push("Loop invariant cannot be proven to hold across iterations".to_string());
            }
        }

        self.consteval.constraints.truncate(prev_constraints_len);
        self.pop_scope();
    }

    /// Check an assignment / compound assignment: type the LHS, then the RHS expecting the LHS
    /// type (#240), verify assignability, and fold a const RHS into the eval environment.
    fn check_assign_stmt(&mut self, lhs: &mut Expr, rhs: &mut Expr, consume: bool) {
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

        if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
            let mut tmp_env = HashMap::new();
            for env in &self.consteval.env {
                for (k, v) in env {
                    tmp_env.insert(k.clone(), v.clone());
                }
            }
            if let Some(val) = self.eval_expr(rhs, &tmp_env) {
                // find the scope that has the variable
                for env in self.consteval.env.iter_mut().rev() {
                    if env.contains_key(name.as_ref()) {
                        env.insert(name.to_string().into(), val);
                        break;
                    }
                }
            }
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
                ty: _,
                span: _,
            }) => {
                if let Ok(n) = n_str.parse::<f64>() {
                    Some(Value::Number(n))
                } else {
                    None
                }
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
                match (l, r, op) {
                    (Value::Number(a), Value::Number(b), BinaryOp::Add) => {
                        Some(Value::Number(a + b))
                    }
                    (Value::Number(a), Value::Number(b), BinaryOp::Sub) => {
                        Some(Value::Number(a - b))
                    }
                    (Value::Number(a), Value::Number(b), BinaryOp::Mul) => {
                        Some(Value::Number(a * b))
                    }
                    (Value::Number(_), Value::Number(_), BinaryOp::MatMul) => {
                        // MatMul not supported for pure numbers at compile time
                        None
                    }
                    (Value::Number(a), Value::Number(b), BinaryOp::Div) => {
                        Some(Value::Number(a / b))
                    }
                    _ => None,
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
                match (l, r, op) {
                    (Value::Number(a), Value::Number(b), RelationalOp::Eq) => {
                        Some(Value::Bool(a == b))
                    }
                    (Value::Number(a), Value::Number(b), RelationalOp::NotEq) => {
                        Some(Value::Bool(a != b))
                    }
                    (Value::Number(a), Value::Number(b), RelationalOp::Lt) => {
                        Some(Value::Bool(a < b))
                    }
                    (Value::Number(a), Value::Number(b), RelationalOp::Gt) => {
                        Some(Value::Bool(a > b))
                    }
                    (Value::Number(a), Value::Number(b), RelationalOp::Le) => {
                        Some(Value::Bool(a <= b))
                    }
                    (Value::Number(a), Value::Number(b), RelationalOp::Ge) => {
                        Some(Value::Bool(a >= b))
                    }
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
            Expr::FunctionCall(FunctionCallExpr {
                name,
                type_args: None,
                args,
                span: _,
            }) => {
                let func = self.env.syntax_functions.get(name.as_ref())?;
                let mut local_env = HashMap::new();
                for (i, arg_expr) in args.iter().enumerate() {
                    let arg_val = self.eval_expr(arg_expr, env)?;
                    local_env.insert(func.params[i].0.clone(), arg_val);
                }
                for stmt in &func.body {
                    if let Some(ret_val) = self.eval_statement(stmt, &mut local_env) {
                        return Some(ret_val);
                    }
                }
                None
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
                        } else {
                            if let Some(val) = self.eval_statement(stmt, &mut local_env) {
                                ret = Some(val);
                            }
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

    pub(crate) fn eval_statement(
        &self,
        stmt: &Statement,
        env: &mut HashMap<crate::symbol::Symbol, Value>,
    ) -> Option<Value> {
        match stmt {
            Statement::LetDecl(LetDeclStmt {
                name,
                is_mut: _,
                ty_ann: _,
                expr,
                span: _,
            }) => {
                if let Some(val) = self.eval_expr(expr, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Assign(AssignStmt {
                lhs: Expr::Identifier(IdentifierExpr { name, span: _ }),
                rhs,
                span: _,
            }) => {
                if let Some(val) = self.eval_expr(rhs, env) {
                    env.insert(name.clone(), val);
                }
                None
            }
            Statement::Return(ReturnStmt { expr, span: _ }) => {
                expr.as_ref().and_then(|e| self.eval_expr(e, env))
            }
            _ => None,
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
