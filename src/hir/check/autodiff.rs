//===- check/autodiff.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Autodiff expression checks (`grad` / `vjp` / `jvp`) and the differentiability guard, split out of
// `hir/expr.rs` along the `check_expr_type_flag` dispatch seam (R2, #279). Moves only — zero logic change.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_differentiability(&mut self, func: &Function) {
        match &func.return_type {
            Type::Tensor(_, _, _) | Type::Scalar(_) | Type::Simd(_, _) => {}
            _ => {
                self.errors.push(format!("Function '{}' cannot be differentiated because it returns a non-continuous type: {:?}", func.name, func.return_type));
            }
        }
    }

    pub(crate) fn check_grad_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Grad(GradExpr {
                target_fn,
                args,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.syntax_functions.get(&*target_fn) {
                    f.clone()
                } else {
                    self.errors.push(format!(
                        "Cannot differentiate unknown function '{}'",
                        target_fn
                    ));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for grad target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                func.return_type.clone()
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_vjp_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Vjp(VjpExpr {
                target_fn,
                args,
                cotangent,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.syntax_functions.get(&*target_fn) {
                    f.clone()
                } else {
                    self.errors
                        .push(format!("Cannot vjp unknown function '{}'", target_fn));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for vjp target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                self.check_expr_type(cotangent);
                func.return_type.clone()
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_jvp_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
        match expr {
            Expr::Jvp(JvpExpr {
                target_fn,
                args,
                tangent,
                span: _,
            }) => {
                let func = if let Some(&f) = self.env.syntax_functions.get(&*target_fn) {
                    f.clone()
                } else {
                    self.errors
                        .push(format!("Cannot jvp unknown function '{}'", target_fn));
                    return Type::Tensor(ElementType::F32, vec![], None);
                };
                self.check_differentiability(&func);

                if args.len() != func.params.len() {
                    self.errors.push(format!(
                        "Function {} expects {} arguments, but {} were provided",
                        target_fn,
                        func.params.len(),
                        args.len()
                    ));
                } else {
                    for (i, arg) in args.iter_mut().enumerate() {
                        let arg_type = self.check_expr_type(arg);
                        let param_type = &func.params[i].1;
                        if !self.is_assignable(param_type, &arg_type) {
                            self.errors.push(format!(
                                        "Type mismatch in argument {} for jvp target {}: expected {:?}, got {:?}",
                                        i + 1, target_fn, param_type, arg_type
                                    ));
                        }
                    }
                }
                self.check_expr_type(tangent);
                func.return_type.clone()
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
