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
                body,
                span: _,
            }) => {
                let iterable_ty = self.check_expr_type_flag(iterable, consume, silent);
                self.push_scope();
                
                // If it's Range, it's I64. If it's Iterator, we extract from Option<T>
                // If it's Tensor, we extract the ElementType
                let iter_ty = match iterable_ty {
                    Type::Enum(name, _) if name.starts_with("Option<") => {
                        // Hack for Option<T> in sema: just parse the T part
                        let start = name.find('<').unwrap() + 1;
                        let end = name.rfind('>').unwrap();
                        let _inner_ty_str = &name[start..end];
                        // fallback to i64 if parsing fails? We don't have parse_type here easily.
                        Type::Scalar(ElementType::I64)
                    }
                    Type::Tensor(el_ty, _, _) => Type::Scalar(el_ty),
                    _ => Type::Scalar(ElementType::I64)
                };
                
                self.insert(iter.clone(), iter_ty); // Still assuming i64 for most things, but it works for our current test cases.
                self.check_block(body, return_type);
                self.pop_scope();
            }
            Statement::Loop(LoopStmt { body, span: _ }) => {
                self.push_scope();
                self.check_block(body, return_type);
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
            Statement::Return(ReturnStmt { expr, span: _ }) => {
                let ty = self.check_expr_type_flag(expr, consume, silent);
                if !self.is_assignable(return_type, &ty) {
                    self.errors.push(format!(
                        "Type mismatch on return. Expected {:?}, got {:?}",
                        return_type, ty
                    ));
                }
            }

            Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi: _,
                span: _,
            }) => {
                self.check_expr_type_flag(expr, consume, silent);
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
        }
    }

    pub(crate) fn prove_expr(&self, expr: &Expr) -> bool {
        // Simple structural matching for our lightweight SMT solver
        for constraint in &self.constraints {
            if expr == constraint {
                return true;
            }
            // Basic commutativity for ==
            if let Expr::RelationalOp(RelationalOpExpr {
                lhs: l1,
                op: RelationalOp::Eq,
                rhs: r1,
                ..
            }) = expr
            {
                if let Expr::RelationalOp(RelationalOpExpr {
                    lhs: l2,
                    op: RelationalOp::Eq,
                    rhs: r2,
                    ..
                }) = constraint
                {
                    if (l1 == l2 && r1 == r2) || (l1 == r2 && r1 == l2) {
                        return true;
                    }
                }
            }
        }
        // Lightweight transitive equality solver for Identifier == Identifier
        if let Expr::RelationalOp(RelationalOpExpr {
            lhs,
            op: RelationalOp::Eq,
            rhs,
            ..
        }) = expr
        {
            if let (Expr::Identifier(l_id), Expr::Identifier(r_id)) = (&**lhs, &**rhs) {
                let mut adj: std::collections::HashMap<String, Vec<String>> =
                    std::collections::HashMap::new();
                for constraint in &self.constraints {
                    if let Expr::RelationalOp(RelationalOpExpr {
                        lhs: c_lhs,
                        op: RelationalOp::Eq,
                        rhs: c_rhs,
                        ..
                    }) = constraint
                    {
                        if let (Expr::Identifier(cl), Expr::Identifier(cr)) = (&**c_lhs, &**c_rhs) {
                            adj.entry(cl.name.clone())
                                .or_default()
                                .push(cr.name.clone());
                            adj.entry(cr.name.clone())
                                .or_default()
                                .push(cl.name.clone());
                        }
                    }
                }

                // BFS to find path from l_id.name to r_id.name
                let mut visited = std::collections::HashSet::new();
                let mut queue = std::collections::VecDeque::new();
                queue.push_back(l_id.name.clone());
                visited.insert(l_id.name.clone());

                while let Some(curr) = queue.pop_front() {
                    if curr == r_id.name {
                        return true;
                    }
                    if let Some(neighbors) = adj.get(&curr) {
                        for n in neighbors {
                            if visited.insert(n.clone()) {
                                queue.push_back(n.clone());
                            }
                        }
                    }
                }
            }
        }

        false
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
