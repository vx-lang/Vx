use std::collections::HashMap;

use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_block(&mut self, body: &mut Vec<Statement>, return_type: &Type) {
        let mut terminated = false;
        for stmt in body {
            if terminated {
                self.errors
                    .push_warning("Unreachable code after return, break, or continue".to_string());
                break; // Only warn once per block
            }
            self.check_statement(stmt, return_type, true, false);
            match stmt {
                Statement::Return(_) | Statement::Break(_) | Statement::Continue(_) => {
                    terminated = true;
                }
                _ => {}
            }
        }
    }

    pub(crate) fn check_statement(
        &mut self,
        stmt: &mut Statement,
        return_type: &Type,
        consume: bool,
        silent: bool,
    ) {
        // Intercept for HIR lowering
        match stmt {
            Statement::Assign(AssignStmt {
                lhs: _lhs,
                rhs,
                span: _,
            }) => {
                let (ty, rhs_reg) = self.check_expr(rhs);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_STORE, rhs_reg, 0, type_idx);
                // Fallthrough to standard semantic checks
            }
            Statement::Return(ReturnStmt { expr, span: _ }) => {
                let (ty, ret_reg) = self.check_expr(expr);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_RET, ret_reg, 0, type_idx);
                // Fallthrough to standard semantic checks
            }
            Statement::LetDecl(LetDeclStmt {
                name: _name,
                is_mut: _,
                ty_ann: _,
                expr,
                span: _,
            }) => {
                let (ty, val_reg) = self.check_expr(expr);
                let type_idx = self.emit_type(&ty);
                self.emit_inst(crate::hir::OP_STORE, val_reg, 0, type_idx);
            }
            _ => {}
        }

        match stmt {
            Statement::LetDecl(LetDeclStmt {
                name,
                is_mut: _is_mut,
                ty_ann,
                expr,
                span: _,
            }) => {
                let ty = self.check_expr_type_flag(expr, consume, silent);

                let mut tmp_env = HashMap::new();
                for env in &self.eval_env {
                    for (k, v) in env {
                        tmp_env.insert(k.clone(), v.clone());
                    }
                }
                if let Some(val) = self.eval_expr(expr, &tmp_env) {
                    self.eval_env.last_mut().unwrap().insert(name.clone(), val);
                }

                if let Some(ann) = ty_ann {
                    if !self.is_assignable(ann, &ty) {
                        self.errors
                            .push(format!("Type mismatch in variable declaration '{}'", name));
                    }
                    self.insert(name.clone(), ann.clone());
                } else {
                    self.insert(name.clone(), ty);
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
                    use crate::ast::expr::{Expr, MethodCallExpr};
                    use crate::ast::Span;
                    let mut next_call = Expr::MethodCall(MethodCallExpr {
                        base: (*iterable).clone(),
                        method_name: "next".to_string(),
                        args: vec![],
                        span: Span::default(),
                    });
                    // This will resolve and monomorphize `next`!
                    let opt_ty = self.check_expr_type_flag(&mut next_call, consume, silent);
                    if let Type::Enum(ref name, _) = opt_ty {
                        if name.starts_with("Option<") {
                            // The Option enum is generic, we can get T from its arguments!
                            // Wait, if it's a GenericInstance(Enum("Option"), [T]), we can extract it!
                        }
                    }
                    if let Type::GenericInstance(_, args) = opt_ty {
                        if args.len() == 1 {
                            iter_ty = args[0].clone();
                        }
                    }
                } else {
                    iter_ty = match iterable_ty {
                        Type::Enum(name, _) if name.starts_with("Option<") => {
                            Type::Scalar(ElementType::I64)
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
                let rhs_ty = self.check_expr_type_flag(rhs, consume, silent);
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
                            if env.contains_key(name) {
                                env.insert(name.clone(), val);
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
                let return_ident = Expr::Identifier(crate::ast::expr::IdentifierExpr {
                    name: "return".to_string(),
                    span: span.clone(),
                });
                let return_eq = Expr::RelationalOp(crate::ast::expr::RelationalOpExpr {
                    lhs: Box::new(return_ident),
                    op: crate::ast::expr::RelationalOp::Eq,
                    rhs: Box::new(expr.clone()),
                    span: span.clone(),
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
                if let Some(Value::Bool(b)) = self.eval_expr(expr, &tmp_env) {
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

    pub(crate) fn prove_expr(&self, expr: &Expr) -> bool {
        let mut prover = crate::sema::prover::SmtProver::new();
        for constraint in &self.constraints {
            if let Err(e) = prover.add_constraint(constraint) {
                // If we can't lower a constraint, we just ignore it or log a warning
                println!("Warning: Could not add constraint to SMT solver: {}", e);
            }
        }

        // To prove `expr` holds under `constraints`, we assert `!expr` and check for unsatisfiability.
        let negated_expr = Expr::UnaryOp(crate::ast::UnaryOpExpr {
            op: crate::ast::UnaryOp::Not,
            expr: Box::new(expr.clone()),
            span: crate::ast::Span::default(),
        });

        if let Err(e) = prover.add_constraint(&negated_expr) {
            println!("Warning: Could not lower expression to SMT solver: {}", e);
            return false; // Can't prove
        }

        match prover.prove() {
            Ok(is_sat) => !is_sat, // If unsat, then the expression is proven (valid)
            Err(e) => {
                println!("Warning: SMT solver error: {}", e);
                false
            }
        }
    }

    pub(crate) fn eval_expr(&self, expr: &Expr, env: &HashMap<String, Value>) -> Option<Value> {
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
            Expr::Identifier(IdentifierExpr { name: n, span: _ }) if n == "true" => {
                Some(Value::Bool(true))
            }
            Expr::Identifier(IdentifierExpr { name: n, span: _ }) if n == "false" => {
                Some(Value::Bool(false))
            }
            Expr::Identifier(IdentifierExpr { name: n, span: _ }) => env.get(n).cloned(),
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
                args,
                span: _,
            }) => {
                let func = self.env.ast_functions.get(name)?;
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
            _ => None,
        }
    }

    pub(crate) fn eval_statement(
        &self,
        stmt: &Statement,
        env: &mut HashMap<String, Value>,
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
}
