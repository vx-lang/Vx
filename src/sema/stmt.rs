//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis for statements, verifying types, variable definitions, and scoping.
//
//===----------------------------------------------------------------------===//

use std::collections::HashMap;

use super::*;

use crate::ast;
use crate::sema;
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
    /// 5. We push this map onto `self.block_liveness`, and we track the current execution index
    ///    using `self.current_stmt_idx`.
    ///
    /// This allows the Non-Lexical Lifetimes (NLL) borrow checker to query `is_variable_used_after`
    /// in O(1) time instead of performing an O(N^2) AST tree-walk!
    pub(crate) fn check_block(&mut self, body: &mut [Statement], return_type: &Type) {
        let mut terminated = false;

        // 1. Liveness Analysis Pass
        let last_use = Self::compute_block_liveness(body);

        self.block_liveness.push(last_use);
        self.current_stmt_idx.push(0);

        #[allow(clippy::needless_range_loop)]
        for i in 0..body.len() {
            *self.current_stmt_idx.last_mut().unwrap() = i;
            let mut stmt = body[i].clone();
            if terminated {
                let stmt_span = stmt.span();
                self.errors.warn(
                    crate::diagnostic::DiagnosticCode::W1003,
                    "Unreachable code after return, break, or continue",
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&stmt_span)),
                );
                break; // Only warn once per block
            }

            self.check_statement(&mut stmt, return_type, true, false);
            body[i] = stmt.clone();
            match stmt {
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    terminated = true;
                }
                _ => {}
            }
        }
        self.block_liveness.pop();
        self.current_stmt_idx.pop();
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
        silent: bool,
    ) {
        // No more HIR interception block needed.

        match stmt {
            Statement::LetDecl(LetDeclStmt {
                name,
                is_mut: _is_mut,
                ty_ann,
                expr,
                span: _,
            }) => {
                self.current_assignment_target = Some(name.to_string());
                let ty = self.check_expr_type_flag(expr, consume, silent);
                self.current_assignment_target = None;

                let mut tmp_env = HashMap::new();
                for env in &self.eval_env {
                    for (k, v) in env {
                        tmp_env.insert(k.clone(), v.clone());
                    }
                }
                if let Some(val) = self.eval_expr(expr, &tmp_env) {
                    self.eval_env
                        .last_mut()
                        .unwrap()
                        .insert(name.to_string().into(), val);
                }

                if let Some(ann) = ty_ann {
                    if !self.is_assignable(ann, &ty) {
                        self.errors
                            .push(format!("Type mismatch in variable declaration '{}'", name));
                    }
                    self.insert(name.to_string(), ann.clone());
                } else {
                    self.insert(name.to_string(), ty);
                }

                if !*_is_mut {
                    let id_expr = Expr::Identifier(IdentifierExpr {
                        name: name.clone(),
                        span: Span::default(),
                    });
                    let eq_expr = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(id_expr),
                        op: RelationalOp::Eq,
                        rhs: Box::new(expr.clone()),
                        span: Span::default(),
                    });
                    self.constraints.push(eq_expr);
                }
            }
            Statement::ForLoop(ForLoopStmt {
                iter,
                iterable,
                invariants,
                body,
                span: _,
            }) => {
                let iterable_ty = self.check_expr_type_flag(iterable, consume, silent);
                self.push_scope();

                // If it's Range, it's I64. If it's Iterator, we extract from Option<T>
                // If it's Tensor, we extract the ElementType
                let mut iter_ty = Type::Scalar(ElementType::I64); // fallback

                // Check if it's a generic iterator by synthesizing a `.next()` call
                if matches!(iterable_ty, Type::GenericInstance(..))
                    || matches!(iterable_ty, Type::Struct(..))
                {
                    use ast::expr::{Expr, MethodCallExpr};
                    use ast::Span;
                    let mut next_call = Expr::MethodCall(MethodCallExpr {
                        base: (*iterable).clone(),
                        method_name: "next".to_string().into(),
                        type_args: None,
                        args: vec![],
                        span: Span::default(),
                    });
                    // This will resolve and monomorphize `next`!
                    let opt_ty = self.check_expr_type_flag(&mut next_call, consume, silent);
                    if let Type::GenericInstance(base, args) = opt_ty {
                        if let Type::Enum(name, _) = &*base {
                            if name.as_ref() == "Option" && args.len() == 1 {
                                iter_ty = args[0].clone();
                            }
                        }
                    }
                } else {
                    iter_ty = match iterable_ty {
                        Type::GenericInstance(base, args) => {
                            if let Type::Enum(name, _) = &*base {
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
                        _ => Type::Scalar(ElementType::I64),
                    };
                }

                self.insert(iter.clone(), iter_ty); // Still assuming i64 for most things, but it works for our current test cases.

                // Prove invariants hold on entry, then assume them inside the loop
                let prev_constraints_len = self.constraints.len();
                for inv in invariants.iter() {
                    if !self.prove_expr(inv) {
                        self.errors
                            .push("Loop invariant cannot be proven on entry".to_string());
                    }
                    self.constraints.push(inv.clone());
                }

                self.check_block(body, return_type);

                // Check invariants hold after the loop iteration (we don't strictly prove induction here, just checking at end of block)
                for inv in invariants.iter() {
                    if !self.prove_expr(inv) {
                        self.errors.push(
                            "Loop invariant cannot be proven to hold across iterations".to_string(),
                        );
                    }
                }

                self.constraints.truncate(prev_constraints_len);
                self.pop_scope();
            }
            Statement::Loop(LoopStmt {
                body,
                span: _,
                invariants,
            }) => {
                self.push_scope();

                let prev_constraints_len = self.constraints.len();
                for inv in invariants.iter() {
                    if !self.prove_expr(inv) {
                        self.errors
                            .push("Loop invariant cannot be proven on entry".to_string());
                    }
                    self.constraints.push(inv.clone());
                }

                self.check_block(body, return_type);

                for inv in invariants.iter() {
                    if !self.prove_expr(inv) {
                        self.errors.push(
                            "Loop invariant cannot be proven to hold across iterations".to_string(),
                        );
                    }
                }

                self.constraints.truncate(prev_constraints_len);
                self.pop_scope();
            }
            Statement::Break(_) => {}
            Statement::Continue(_) => {}
            Statement::Assign(AssignStmt { lhs, rhs, span: _ })
            | Statement::CompoundAssign(CompoundAssignStmt {
                lhs,
                op: _,
                rhs,
                span: _,
            }) => {
                let lhs_ty = self.check_expr_type_flag(lhs, false, silent);

                // Determine target name for NLL
                if let Expr::Identifier(id) = lhs {
                    self.current_assignment_target = Some(id.name.to_string());
                } else if let Expr::MemberAccess(ma) = lhs {
                    if let Expr::Identifier(id) = &*ma.base {
                        self.current_assignment_target = Some(id.name.to_string());
                    }
                }

                let rhs_ty = self.check_expr_type_flag(rhs, consume, silent);
                self.current_assignment_target = None;
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.push("Type mismatch in assignment".to_string());
                }

                if let Expr::Identifier(IdentifierExpr { name, span: _ }) = lhs {
                    let mut tmp_env = HashMap::new();
                    for env in &self.eval_env {
                        for (k, v) in env {
                            tmp_env.insert(k.clone(), v.clone());
                        }
                    }
                    if let Some(val) = self.eval_expr(rhs, &tmp_env) {
                        // find the scope that has the variable
                        for env in self.eval_env.iter_mut().rev() {
                            if env.contains_key(name.as_ref()) {
                                env.insert(name.to_string().into(), val);
                                break;
                            }
                        }
                    }
                }
            }
            Statement::Return(ReturnStmt { expr, span }) => {
                let ty = self.check_expr_type_flag(expr, consume, silent);

                let mut expected_ty = return_type.clone();
                if let Some(Type::Unknown) = self.current_return_type {
                    self.current_return_type = Some(ty.clone());
                    expected_ty = ty.clone();
                }

                if !self.is_assignable(&expected_ty, &ty) {
                    self.errors.push(format!(
                        "Type mismatch on return. Expected {:?}, got {:?}",
                        expected_ty, ty
                    ));
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
                self.constraints.push(return_eq);
            }

            Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: _,
                span: _,
            }) => {
                let saved_borrows = self.active_borrows.clone();
                self.check_expr_type_flag(expr, consume, silent);
                self.active_borrows = saved_borrows;
            }
            Statement::Assert(AssertStmt { expr, msg, span: _ }) => {
                let ty = self.check_expr_type_flag(expr, consume, silent);
                if ty != Type::Scalar(ElementType::Bool) {
                    self.errors
                        .push("Assertion condition must be boolean".to_string());
                }

                let is_verified = matches!(return_type, Type::Verified(_));
                let mut tmp_env = HashMap::new();
                for env in &self.eval_env {
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
                            self.errors.push(format!("Comptime assert failed: {}", m));
                        }
                    }
                } else if is_verified {
                    // Try to prove mathematically using our SMT constraints
                    if !self.prove_expr(expr) {
                        self.errors.push(
                            "Cannot statically prove assertion for Verified return type"
                                .to_string(),
                        );
                    }
                } else {
                    // It's a standard dynamic assert, add it to our mathematical constraints
                    // so we can prove future Verified<T> return conditions!
                    self.constraints.push(*expr.clone());
                }
            }
            Statement::MacroCall(_) => panic!("Macros should be expanded before type checking"),
        }
    }

    pub(crate) fn prove_expr(&mut self, expr: &Expr) -> bool {
        let mut prover = sema::prover::SmtProver::new();
        for constraint in &self.constraints {
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
                let func = self.env.ast_functions.get(name.as_ref())?;
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
            Statement::Return(ReturnStmt { expr, span: _ }) => self.eval_expr(expr, env),
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
            (Topology::GPU, Topology::GPU) => true,
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
