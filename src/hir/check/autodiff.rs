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
                self.errors.push(format!("Function '{}' cannot be differentiated because it returns a non-continuous type: {}", func.name, func.return_type));
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

    /// Checks the arguments of a `grad`, `vjp` or `jvp` call against `func`'s parameters. Each
    /// argument is checked expecting its parameter's type, so `0.25` passed to an `f64` parameter
    /// is an `f64`, as it is in an ordinary call.
    fn check_autodiff_args(
        &mut self,
        form: &str,
        target_fn: &str,
        func: &Function,
        args: &mut [Expr],
    ) {
        if args.len() != func.params.len() {
            self.errors.push(format!(
                "Function {} expects {} arguments, but {} were provided",
                target_fn,
                func.params.len(),
                args.len()
            ));
            return;
        }
        for (i, (arg, (_, param_type))) in args.iter_mut().zip(&func.params).enumerate() {
            let arg_type = self.check_expr_type_expecting(arg, param_type.clone());
            if !self.is_assignable(param_type, &arg_type) {
                self.errors.push(format!(
                    "Type mismatch in argument {} for {} target {}: expected {:?}, got {:?}",
                    i + 1,
                    form,
                    target_fn,
                    param_type,
                    arg_type
                ));
            }
        }
    }

    /// Checks the seed of a `vjp` or `jvp` call, expecting `expected`.
    fn check_autodiff_seed(
        &mut self,
        form: &str,
        target_fn: &str,
        seed: &mut Expr,
        expected: &Type,
    ) {
        let seed_type = self.check_expr_type_expecting(seed, expected.clone());
        if !self.is_assignable(expected, &seed_type) {
            self.errors.push(format!(
                "Type mismatch in the seed for {} target {}: expected {:?}, got {:?}",
                form, target_fn, expected, seed_type
            ));
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

                self.check_autodiff_args("grad", &*target_fn, &func, args);
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

                self.check_autodiff_args("vjp", &*target_fn, &func, args);
                // The seed multiplies the derivative of the result, so it has the result's type.
                self.check_autodiff_seed("vjp", &*target_fn, cotangent, &func.return_type);
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

                self.check_autodiff_args("jvp", &*target_fn, &func, args);
                // The seed is a direction for the parameter, so it has that parameter's type. The
                // generated code passes it after all the arguments, but Enzyme reads the argument
                // after `x` as `x`'s seed, so only a one-parameter function works.
                match func.params.as_slice() {
                    [(_, param)] => {
                        self.check_autodiff_seed("jvp", &*target_fn, tangent, param);
                    }
                    params => {
                        self.errors.push(format!(
                            "jvp works only on a function with one parameter, but {} has {}",
                            target_fn,
                            params.len()
                        ));
                        self.check_expr_type(tangent);
                    }
                }
                func.return_type.clone()
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
