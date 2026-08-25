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
use std::collections::HashMap;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_comptimeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::ComptimeBlock(ComptimeBlockExpr {
                stmts,
                ret,
                span: _,
            }) => {
                self.push_scope();
                let mut ret_ty = self.check_expr_block(stmts, consume);
                if let Some(r) = ret {
                    ret_ty = self.check_expr_type(r);
                }
                self.pop_scope();
                ret_ty
            }

            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_if_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        let if_expr = match expr {
            Expr::If(e) => e,
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        };

        let cond_ty = self.check_expr_type(&mut if_expr.cond);
        if cond_ty != Type::Scalar(ElementType::Bool) {
            self.errors
                .push("Condition in if expression must be of type bool (i1)".to_string());
        }

        if if_expr.is_comptime {
            let mut tmp_env = std::collections::HashMap::new();
            for env in &self.consteval.env {
                for (k, v) in env {
                    tmp_env.insert(k.clone(), v.clone());
                }
            }
            if let Some(Value::Bool(b)) = self.eval_expr(&if_expr.cond, &tmp_env) {
                if b {
                    if_expr.else_block = None;
                } else {
                    if_expr.then_block.clear();
                }
            } else {
                self.errors
                    .push("Cannot statically evaluate comptime if condition".to_string());
            }
        }

        self.push_scope();
        let mut then_ty = Type::Tensor(ElementType::F32, vec![], None);
        if !self.speculating && !if_expr.then_block.is_empty() {
            then_ty = self.check_expr_block(&mut if_expr.then_block, consume);
        }
        self.pop_scope();

        let mut else_ty = Type::Tensor(ElementType::F32, vec![], None);
        if let Some(else_b) = if_expr.else_block.as_mut() {
            if !else_b.is_empty() {
                self.push_scope();
                if !self.speculating {
                    else_ty = self.check_expr_block(else_b, consume);
                }
                self.pop_scope();

                if !if_expr.is_comptime && then_ty != else_ty {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3007,
                        format!(
                            "If expression branches have incompatible types: {:?} and {:?}",
                            then_ty, else_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&if_expr.span)),
                    );
                }
            }
        } else if !if_expr.is_comptime {
            // Without else block, it evaluates to unit (represented as dummy Tensor)
            then_ty = Type::Tensor(ElementType::F32, vec![], None);
        }

        // If it was comptime evaluated to false, the return type should just be the else block type
        if if_expr.is_comptime {
            if if_expr.then_block.is_empty() {
                return else_ty;
            } else {
                return then_ty;
            }
        }

        then_ty
    }

    pub(crate) fn check_unsafeblock_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret: ret_expr,
                span: _,
            }) => {
                let prev_unsafe = self.in_unsafe_block;
                self.in_unsafe_block = true;
                self.push_scope();
                let mut ret_ty = self.check_expr_block(stmts, consume);
                if let Some(r) = ret_expr {
                    ret_ty = self.check_expr_type_flag(r, consume);
                }
                self.pop_scope();
                self.in_unsafe_block = prev_unsafe;
                ret_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_range_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) => {
                // Reconcile the bounds so an untyped literal adopts the other bound's type
                // (`0..n` with `n: i64` → `0` becomes i64), mirroring binary-operand inference (#240).
                let (start_ty, end_ty) = self.check_operand_pair(start, end, true);
                if start_ty != end_ty {
                    self.errors.push(format!(
                        "Range start and end types must match, got {:?} and {:?}",
                        start_ty, end_ty
                    ));
                }
                start_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn bind_pattern_variables(&mut self, pattern: &Pattern, expr_ty: &Type) {
        match pattern {
            Pattern::Identifier(name) => {
                self.insert(name.to_string(), expr_ty.clone());
            }
            Pattern::EnumVariant(enum_name, variant_name, Some(payloads)) => {
                let mut base_name = enum_name.clone();
                if let Some(idx) = enum_name.find('<') {
                    base_name = enum_name[..idx].to_string().into();
                }
                if let Some(enum_decl) = self.env.enums.get(base_name.as_ref()) {
                    if let Some(variant) = enum_decl.variants.iter().find(|v| v.0 == *variant_name)
                    {
                        if let Some(payload_types) = &variant.1 {
                            let mut mapping = std::collections::HashMap::new();
                            if let Type::GenericInstance(_, args) = expr_ty {
                                for (i, param) in enum_decl.generics.iter().enumerate() {
                                    if i < args.len() {
                                        mapping.insert(param.name().into(), args[i].clone());
                                    }
                                }
                            }
                            for (i, p) in payloads.iter().enumerate() {
                                if let Pattern::Identifier(name) = p {
                                    if i < payload_types.len() {
                                        let p_ty = payload_types[i].substitute(&mapping);
                                        self.insert(name.to_string(), p_ty);
                                    } else {
                                        self.insert(name.to_string(), Type::Unknown);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for p in payloads {
                        if let Pattern::Identifier(name) = p {
                            self.insert(name.to_string(), Type::Unknown);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn check_match_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::Match(MatchExpr {
                expr: match_expr,
                arms,
                span: _,
            }) => {
                let expr_ty = self.check_expr_type(match_expr);

                let mut match_ty: Option<Type> = None;
                for arm in arms {
                    self.push_scope();
                    self.bind_pattern_variables(&arm.pattern, &expr_ty);

                    let arm_ty = if !self.speculating {
                        self.check_expr_block(&mut arm.body, consume)
                    } else {
                        Type::Tensor(ElementType::F32, vec![], None)
                    };
                    self.pop_scope();

                    // An arm contributes to the match's value type only if its block
                    // ends in a tail expression (a non-semicolon `ExprStmt`). Arms that
                    // end in `return`/`break`/`continue`, or in a statement like `assert`,
                    // diverge or yield unit and do not constrain the match value — this is
                    // what lets `Option::unwrap` (Some arm `return v`, None arm asserts)
                    // type-check as `-> T`.
                    let yields_value = matches!(
                        arm.body.last(),
                        Some(Statement::ExprStmt(ExprStmtStmt {
                            has_semi: false,
                            ..
                        }))
                    );
                    if yields_value && match_ty.is_none() {
                        match_ty = Some(arm_ty);
                    }
                }

                // When no arm yields a value the match sits in diverging/statement
                // position (every arm returns or aborts). Type it as the expected return
                // type so an implicit `return match { ... }` type-checks; fall back to the
                // historical placeholder when there is no expected return type.
                match_ty.unwrap_or_else(|| {
                    self.current_return_type.clone().unwrap_or(Type::Tensor(
                        ElementType::F32,
                        vec![],
                        None,
                    ))
                })
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_closure_expr(&mut self, expr: &mut Expr, _consume: bool) -> Type {
        match expr {
            Expr::Closure(e) => {
                let struct_name = format!("Closure_{}", self.next_id);
                let func_name = format!("{}_call", struct_name);
                self.next_id += 1;

                let cloned_params = e.params.clone();

                let closure_depth = self.scopes.len();
                self.mono.closure_depths.push(closure_depth);
                self.mono.closure_captures_stack.push(HashMap::new());

                let old_ret = self.current_return_type.take();
                self.current_return_type = Some(Type::Unknown);

                self.push_scope();
                for (name, ty) in &cloned_params {
                    self.insert(name.to_string(), ty.clone());
                }
                let mut b = e.body.clone();
                let expr_ret_ty = self.check_expr_type(&mut b);

                let mut ret_ty = expr_ret_ty;
                if let Some(inferred) = &self.current_return_type {
                    if *inferred != Type::Unknown {
                        ret_ty = inferred.clone();
                    }
                }

                self.pop_scope();
                self.current_return_type = old_ret;

                let captured_vars_map = self.mono.closure_captures_stack.pop().unwrap_or_default();
                self.mono.closure_depths.pop();

                let mut captured_vars: Vec<(crate::symbol::Symbol, Type)> =
                    captured_vars_map.into_iter().collect();
                captured_vars.sort_by(|a, b| a.0.cmp(&b.0)); // Stable layout

                if !self.speculating {
                    // Consume captured variables in the outer scope if they are linear
                    for (name, ty) in &captured_vars {
                        if matches!(ty, Type::Struct(_, _) | Type::Tensor(_, _, _)) {
                            self.consume(name.as_ref());
                        }
                    }
                }

                // Create StructDecl for the environment
                let struct_decl = decl::StructDecl {
                    name: struct_name.clone().into(),
                    generics: vec![],
                    fields: captured_vars.clone(),
                    doc_comment: None,
                };
                self.mono.generated_structs.push(struct_decl);

                // Create the Function for calling the closure
                let mut env_params: Vec<(crate::symbol::Symbol, Type)> = vec![(
                    "_env".to_string().into(),
                    Type::Pointer(
                        Box::new(Type::Struct(struct_name.clone().into(), None)),
                        None,
                        true,
                    ), // &mut env
                )];

                for (name, ty) in &cloned_params {
                    env_params.push((name.clone(), ty.clone()));
                }

                let mut body_stmts = Vec::new();
                for (cap_name, cap_ty) in &captured_vars {
                    let env_access = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(Expr::Identifier(IdentifierExpr::new(
                            "_env".to_string().into(),
                            e.span,
                        ))),
                        member: cap_name.clone(),
                        struct_name: Some(struct_name.clone().into()),
                        span: e.span,
                    });
                    body_stmts.push(Statement::LetDecl(LetDeclStmt {
                        name: cap_name.clone(),
                        is_mut: true,
                        ty_ann: Some(cap_ty.clone()),
                        expr: env_access,
                        span: e.span,
                    }));
                }

                body_stmts.push(Statement::Return(ReturnStmt {
                    expr: *b,
                    span: e.span,
                }));

                let call_func = decl::Function {
                    name: func_name.clone().into(),
                    generics: vec![],
                    params: env_params,
                    topology: self.active_topology.clone(),
                    return_type: ret_ty.clone(),
                    requires: vec![],
                    ensures: vec![],
                    where_transfers: vec![],
                    body: body_stmts,
                    doc_comment: None,
                };

                self.mono.functions.push((call_func, 0));

                let mut fields = Vec::new();
                for (cap_name, _) in &captured_vars {
                    fields.push((
                        cap_name.clone(),
                        Expr::Identifier(IdentifierExpr::new(cap_name.clone(), e.span)),
                    ));
                }

                *expr = Expr::StructInit(StructInitExpr {
                    name: struct_name.clone().into(),
                    fields,
                    type_id: None,
                    span: e.span,
                });

                // Record the closure's call signature so a later `unify_types` can recover it:
                // `Struct("Closure_N")` erases the args/ret, but matching it against a
                // `ClosureK<Args.., Ret>` parameter (e.g. `.map`'s `Closure1<T, NewItem>`) needs
                // them to bind the method's generics. Keyed by the generated struct name.
                let param_tys: Vec<Type> = cloned_params.iter().map(|(_, ty)| ty.clone()).collect();
                self.mono
                    .closure_signatures
                    .insert(struct_name.clone().into(), (param_tys, ret_ty));

                Type::Struct(struct_name.into(), None)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
