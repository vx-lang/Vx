//===- check/autodiff.rs - Vx Compiler --------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Autodiff expression checks (`grad` / `vjp` / `jvp`) and the differentiability guard, split out of
// `hir/expr.rs` along the `check_expr_type_flag` dispatch seam (R2, #279). Moves only — zero logic change.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl<'a> TypeChecker<'a> {
    /// The element type a differentiable value is made of. `None` for a type autodiff has no
    /// shape for at all, such as a struct or a pointer.
    fn differentiable_elem(ty: &Type) -> Option<&ElementType> {
        match ty {
            Type::Tensor(e, _, _) | Type::Scalar(e) | Type::Simd(e, _) => Some(e),
            _ => None,
        }
    }

    /// Whether values of this element type have a derivative. An integer or a bool takes
    /// separated values, so between any two of them there is no limit to take. A generic
    /// element is not decided here -- it is checked once it has been substituted.
    fn is_continuous(e: &ElementType) -> bool {
        e.is_float() || matches!(e, ElementType::Generic(_))
    }

    /// Whether `func` can be differentiated at all, checked at the `grad`/`vjp`/`jvp` call.
    ///
    /// A derivative needs both ends continuous: the result, and the value it is taken with
    /// respect to. The value differentiated is the first parameter -- a later one is an
    /// ordinary argument the function also takes, an index or a count, and it may be discrete.
    pub(crate) fn check_differentiability(&mut self, func: &Function) {
        match Self::differentiable_elem(&func.return_type) {
            None => {
                self.errors.push(format!("Function '{}' cannot be differentiated because it returns a non-continuous type: {:?}", func.name, func.return_type));
            }
            Some(e) if !Self::is_continuous(e) => {
                self.errors.push(format!(
                    "Function '{}' cannot be differentiated because it returns the discrete \
                     type {}",
                    func.name, e
                ));
            }
            Some(_) => {}
        }
        if let Some((param_name, param_ty)) = func.params.first() {
            if let Some(e) = Self::differentiable_elem(param_ty) {
                if !Self::is_continuous(e) {
                    self.errors.push(format!(
                        "Function '{}' cannot be differentiated with respect to its first \
                         parameter: '{}' is the discrete type {}",
                        func.name, param_name, e
                    ));
                }
            }
        }
    }

    pub(crate) fn check_grad_expr(&mut self, expr: &mut Expr) -> Type {
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
                    return Type::Unknown;
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

    pub(crate) fn check_vjp_expr(&mut self, expr: &mut Expr) -> Type {
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
                    return Type::Unknown;
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

    pub(crate) fn check_jvp_expr(&mut self, expr: &mut Expr) -> Type {
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
                    return Type::Unknown;
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
