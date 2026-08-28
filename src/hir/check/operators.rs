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

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_ascast_expr(&mut self, expr: &mut AsCastExpr, consume: bool) -> Type {
        let source_ty = self.check_expr_type_flag(&mut expr.expr, consume);
        let target_ty = expr.target_ty.clone();
        expr.source_ty = Some(source_ty.clone());

        match (&source_ty, &target_ty) {
            (Type::Struct(name, _), Type::Closure(target_args, target_ret))
                if name.starts_with("Closure_") =>
            {
                let call_method_name = format!("{}_call", name);

                // The call method lives in `env.functions` when the generated function is
                // already merged into the module (the GID-annotation recheck), and only in
                // `mono.functions` while the first check is still collecting it. Either way
                // the signature is params + return type; `env.functions` stores them as
                // tuple fields (.2 / .0), not as a `Type::Function`.
                let mut found_func = None;
                if let Some(entry) = self.env.functions.get(&*call_method_name) {
                    found_func = Some((entry.2.clone(), entry.0.clone()));
                } else {
                    for (func, _) in &self.mono.functions {
                        if func.name.as_ref() == call_method_name {
                            let params = func.params.iter().map(|(_, t)| t.clone()).collect();
                            found_func = Some((params, func.return_type.clone()));
                            break;
                        }
                    }
                }

                if let Some((func_args, func_ret)) = found_func {
                    let mut args_match = func_args.len() == target_args.len() + 1;
                    if args_match {
                        for (i, target_arg) in target_args.iter().enumerate() {
                            if target_arg != &func_args[i + 1] {
                                args_match = false;
                                break;
                            }
                        }
                    }

                    if !args_match || target_ret.as_ref() != &func_ret {
                        self.errors.push(format!(
                                "Closure cast signature mismatch. Expected {:?} but found a function with args {:?} and ret {:?}",
                                target_ty, func_args, func_ret
                            ));
                        return Type::Unknown;
                    }

                    return target_ty;
                } else {
                    self.errors.push(format!(
                        "Closure call method '{}' not found for cast.",
                        call_method_name
                    ));
                    return Type::Unknown;
                }
            }
            (Type::Scalar(_), Type::Scalar(_)) => {
                return target_ty;
            }
            (Type::Scalar(_), Type::Pointer(_, _, _)) => {
                // Allow casting integers to pointers (e.g. 0 as *mut T)
                if !self.in_unsafe_block && !self.speculating {
                    self.errors.push(
                        "Casting an integer to a raw pointer requires an unsafe block".to_string(),
                    );
                }
                return target_ty;
            }
            _ => {}
        }

        self.errors.push(format!(
            "Unsupported cast: cannot cast from {:?} to {:?}",
            source_ty, target_ty
        ));
        Type::Unknown
    }

    /// Type-check the two operands of a binary / relational / logical op, letting an untyped
    /// numeric literal on one side adopt the other side's concrete type — the reconciliation an
    /// implicit conversion used to paper over (`i + 1` with `i: i64`, `x < 1.0` with `x: f64`),
    /// now that a literal is untyped until typed by context (#240). If both operands are untyped
    /// literals, or neither is, they are checked in order under the ambient expected type (so
    /// `let x: i64 = 1 + 2` still infers both to `i64`). Returns `(lhs_ty, rhs_ty)`.
    pub(crate) fn check_operand_pair(
        &mut self,
        lhs: &mut Expr,
        rhs: &mut Expr,
        consume: bool,
    ) -> (Type, Type) {
        let lhs_untyped_lit = matches!(&*lhs, Expr::Number(n) if n.ty.is_none());
        let rhs_untyped_lit = matches!(&*rhs, Expr::Number(n) if n.ty.is_none());
        if rhs_untyped_lit && !lhs_untyped_lit {
            let lt = self.check_expr_type_flag(lhs, consume);
            let rt = self.check_expr_expecting(rhs, Some(lt.clone()), consume);
            (lt, rt)
        } else if lhs_untyped_lit && !rhs_untyped_lit {
            let rt = self.check_expr_type_flag(rhs, consume);
            let lt = self.check_expr_expecting(lhs, Some(rt.clone()), consume);
            (lt, rt)
        } else {
            let lt = self.check_expr_type_flag(lhs, consume);
            let rt = self.check_expr_type_flag(rhs, consume);
            (lt, rt)
        }
    }

    pub(crate) fn check_binaryop_expr(&mut self, expr: &mut Expr, consume: bool) -> Type {
        match expr {
            Expr::BinaryOp(BinaryOpExpr { lhs, op, rhs, span }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, consume);

                // Tensor operator overloading (A * B) -> Matmul.
                //
                // A borrowed operand is the non-consuming spelling: `a @ b`
                // moves both tensors, `&a @ &b` reads them. Both are matmuls,
                // so the wrappers are peeled here and the operand's element
                // type, shape and placement are read through them (#335).
                //
                // A tensor could otherwise be an operand of `@` exactly once in
                // a program, and there was no way to say otherwise -- `&a @ &b`
                // did not typecheck, and the stdlib `copy` is a copy-*into*
                // rather than a duplication. That is enough for a test and never
                // enough for a model, whose weights are read once per token.
                if let (Some((el_ty_l, dims_l, top_l)), Some((el_ty_r, dims_r, _top_r))) = (
                    Self::as_tensor_operand(&lhs_ty),
                    Self::as_tensor_operand(&rhs_ty),
                ) {
                    if *op == BinaryOp::MatMul {
                        if el_ty_l != el_ty_r {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E7002,
                                format!("Tensor multiplication requires matching element types, got {:?} and {:?}", el_ty_l, el_ty_r),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                            );
                        }
                        let l_len = dims_l.len();
                        let r_len = dims_r.len();
                        if (l_len != 2 && l_len != 0) || (r_len != 2 && r_len != 0) {
                            self.errors.error_with_code(
                                crate::diagnostic::DiagnosticCode::E7001,
                                format!("Tensor multiplication (matmul) requires 2D tensors, got {}D and {}D", l_len, r_len),
                                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                            );
                            return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                        }
                        // `[m, k] @ [k, n]` is `[m, n]`, and the two `k`s must agree. Computing
                        // the shape here is what lets a declaration be checked at all: a
                        // dims-less result made `is_assignable` skip the comparison, so any
                        // annotation was accepted (Vx#397). A dims-less operand is the dynamic
                        // spelling and keeps the dims-less answer -- nothing to compute or
                        // compare until it carries a shape.
                        if l_len == 2 && r_len == 2 {
                            let empty_env = std::collections::HashMap::new();
                            let dim_of = |v: Option<crate::hir::env::Value>| match v {
                                Some(crate::hir::env::Value::Number(n)) => Some(n),
                                _ => None,
                            };
                            let kl = dim_of(self.eval_expr(&dims_l[1], &empty_env));
                            let kr = dim_of(self.eval_expr(&dims_r[0], &empty_env));
                            if let (Some(kl), Some(kr)) = (kl, kr) {
                                if kl != kr {
                                    self.errors.error_with_code(
                                        crate::diagnostic::DiagnosticCode::E7001,
                                        format!(
                                            "Tensor multiplication requires the inner dimensions \
                                             to agree, got {kl} and {kr}"
                                        ),
                                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                                    );
                                    return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                                }
                            }
                            return Type::Tensor(
                                el_ty_l.clone(),
                                vec![dims_l[0].clone(), dims_r[1].clone()],
                                top_l.clone(),
                            );
                        }
                        return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                    }
                }

                // Slice elementwise (S3): arithmetic where at least one operand is a rank-1
                // float slice. `slice OP slice`, `slice OP scalar`, `scalar OP slice` -> the
                // slice's shape (here `*` is elementwise; matmul is `@`/MatMul, handled above).
                // These lower to vector ops (load/broadcast + arith.{mulf,addf,subf,divf}).
                //
                // Half-precision slices participate as STORAGE (Vx#320): they widen to f32 on
                // load, so any op touching an f16/bf16 slice yields an F32 slice -- storage
                // precision and arithmetic precision are separate decisions, and arithmetic is
                // always f32. Narrowing back happens only at an explicit store into a half
                // tensor, which the assignment rules govern.
                if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                ) {
                    let l_half = Self::is_half_slice(&lhs_ty);
                    let r_half = Self::is_half_slice(&rhs_ty);
                    let l_slice = Self::is_f32_slice(&lhs_ty) || l_half;
                    let r_slice = Self::is_f32_slice(&rhs_ty) || r_half;
                    let l_scalar = matches!(lhs_ty, Type::Scalar(ElementType::F32));
                    let r_scalar = matches!(rhs_ty, Type::Scalar(ElementType::F32));
                    // At least one slice operand; the other must be a slice or an f32 scalar.
                    let slice_op = match (l_slice, r_slice) {
                        (true, true) => true,      // slice OP slice
                        (true, false) => r_scalar, // slice OP scalar
                        (false, true) => l_scalar, // scalar OP slice
                        (false, false) => false,
                    };
                    if slice_op {
                        let shape_side = if l_slice { &lhs_ty } else { &rhs_ty };
                        if l_half || r_half {
                            // The widened result: same shape and placement, f32 element.
                            if let Type::Tensor(_, dims, top) = shape_side {
                                return Type::Tensor(ElementType::F32, dims.clone(), top.clone());
                            }
                        }
                        return shape_side.clone();
                    }
                }

                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3004,
                        format!(
                            "Type mismatch in binary operation: {:?} vs {:?}",
                            lhs_ty, rhs_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
                lhs_ty
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    /// Element type seen through value-carrying wrappers (Pinned/Ref/Borrow/Tensor).
    /// Used so a relational comparison can match element values regardless of wrapper.
    pub(crate) fn scalar_elem(ty: &Type) -> Option<ElementType> {
        match ty {
            Type::Scalar(e) => Some(e.clone()),
            Type::Tensor(e, _, _) => Some(e.clone()),
            Type::Pinned(inner, _) => Self::scalar_elem(inner),
            Type::Ref(inner, _) => Self::scalar_elem(inner),
            Type::Borrow { inner, .. } => Self::scalar_elem(inner),
            _ => None,
        }
    }

    pub(crate) fn check_relationalop_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::RelationalOp(RelationalOpExpr {
                lhs,
                op: _,
                rhs,
                span,
            }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, false);
                // A relational compares element *values*, so wrapper differences
                // (Pinned/Ref/Tensor vs a bare Scalar) are fine as long as the element
                // types agree -- e.g. comparing a device-resident scalar to a constant.
                let compatible = self.is_assignable(&lhs_ty, &rhs_ty)
                    || matches!(
                        (Self::scalar_elem(&lhs_ty), Self::scalar_elem(&rhs_ty)),
                        (Some(a), Some(b)) if a == b
                    );
                if !compatible {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3005,
                        format!(
                            "Type mismatch in relational operation: {:?} vs {:?}",
                            lhs_ty, rhs_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
                Type::Scalar(ElementType::Bool)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_logicalop_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::LogicalOp(LogicalOpExpr {
                lhs,
                op: _,
                rhs,
                span,
            }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, false);
                if !self.is_assignable(&lhs_ty, &rhs_ty) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3006,
                        format!(
                            "Type mismatch in logical operation: {:?} vs {:?}",
                            lhs_ty, rhs_ty
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                    );
                }
                Type::Scalar(ElementType::Bool)
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }

    pub(crate) fn check_unaryop_expr(&mut self, expr: &mut Expr) -> Type {
        match expr {
            Expr::UnaryOp(UnaryOpExpr {
                op,
                expr: inner,
                span: _,
            }) => {
                let inner_ty = self.check_expr_type(inner);
                match op {
                    UnaryOp::Not => Type::Scalar(ElementType::Bool),
                    UnaryOp::Neg => inner_ty,
                }
            }
            _ => panic!("Expected IndexAccess, got {:?}", expr),
        }
    }
}
