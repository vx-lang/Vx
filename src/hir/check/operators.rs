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
    pub(crate) fn check_ascast_expr(
        &mut self,
        expr: &mut AsCastExpr,
        consume: bool,
        silent: bool,
    ) -> Type {
        let source_ty = self.check_expr_type_flag(&mut expr.expr, consume, silent);
        let target_ty = expr.target_ty.clone();
        expr.source_ty = Some(source_ty.clone());

        match (&source_ty, &target_ty) {
            (Type::Struct(name, _), Type::Closure(target_args, target_ret))
                if name.starts_with("Closure_") =>
            {
                let call_method_name = format!("{}_call", name);

                let mut found_func = None;
                if let Some(func_type) = self.env.functions.get(&*call_method_name) {
                    found_func = Some(func_type.0.clone());
                } else {
                    for (func, _) in &self.monomorphized_functions {
                        if func.name.as_ref() == call_method_name {
                            let params = func.params.iter().map(|(_, t)| t.clone()).collect();
                            found_func =
                                Some(Type::Function(params, Box::new(func.return_type.clone())));
                            break;
                        }
                    }
                }

                if let Some(func_type) = found_func {
                    let Type::Function(func_args, func_ret) = func_type else {
                        unreachable!()
                    };
                    let mut args_match = func_args.len() == target_args.len() + 1;
                    if args_match {
                        for (i, target_arg) in target_args.iter().enumerate() {
                            if target_arg != &func_args[i + 1] {
                                args_match = false;
                                break;
                            }
                        }
                    }

                    if !args_match || target_ret.as_ref() != func_ret.as_ref() {
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
                if !self.in_unsafe_block && !silent {
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
        silent: bool,
    ) -> (Type, Type) {
        let lhs_untyped_lit = matches!(&*lhs, Expr::Number(n) if n.ty.is_none());
        let rhs_untyped_lit = matches!(&*rhs, Expr::Number(n) if n.ty.is_none());
        if rhs_untyped_lit && !lhs_untyped_lit {
            let lt = self.check_expr_type_flag(lhs, consume, silent);
            let rt = self.check_expr_expecting(rhs, Some(lt.clone()), consume, silent);
            (lt, rt)
        } else if lhs_untyped_lit && !rhs_untyped_lit {
            let rt = self.check_expr_type_flag(rhs, consume, silent);
            let lt = self.check_expr_expecting(lhs, Some(rt.clone()), consume, silent);
            (lt, rt)
        } else {
            let lt = self.check_expr_type_flag(lhs, consume, silent);
            let rt = self.check_expr_type_flag(rhs, consume, silent);
            (lt, rt)
        }
    }

    pub(crate) fn check_binaryop_expr(
        &mut self,
        expr: &mut Expr,
        consume: bool,
        silent: bool,
    ) -> Type {
        match expr {
            Expr::BinaryOp(BinaryOpExpr { lhs, op, rhs, span }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, consume, silent);

                // Tensor operator overloading (A * B) -> Matmul
                if let (
                    Type::Tensor(el_ty_l, dims_l, top_l),
                    Type::Tensor(el_ty_r, dims_r, _top_r),
                ) = (&lhs_ty, &rhs_ty)
                {
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
                        return Type::Tensor(el_ty_l.clone(), vec![], top_l.clone());
                    }
                }

                // Slice elementwise (S3): arithmetic where at least one operand is a rank-1 f32
                // slice. `slice OP slice`, `slice OP scalar`, `scalar OP slice` -> the slice's
                // shape (here `*` is elementwise; matmul is `@`/MatMul, handled above). These
                // lower to vector ops (load/broadcast + arith.{mulf,addf,subf,divf}).
                if matches!(
                    op,
                    BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
                ) {
                    let l_slice = Self::is_f32_slice(&lhs_ty);
                    let r_slice = Self::is_f32_slice(&rhs_ty);
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
                        return if l_slice { lhs_ty } else { rhs_ty };
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

    pub(crate) fn check_relationalop_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::RelationalOp(RelationalOpExpr {
                lhs,
                op: _,
                rhs,
                span,
            }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, false, silent);
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

    pub(crate) fn check_logicalop_expr(&mut self, expr: &mut Expr, silent: bool) -> Type {
        match expr {
            Expr::LogicalOp(LogicalOpExpr {
                lhs,
                op: _,
                rhs,
                span,
            }) => {
                let (lhs_ty, rhs_ty) = self.check_operand_pair(lhs, rhs, false, silent);
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

    pub(crate) fn check_unaryop_expr(&mut self, expr: &mut Expr, _silent: bool) -> Type {
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
