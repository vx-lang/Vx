//===- lanes.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The lane reductions: `lanes::sum(&t, n)`, `lanes::max(&t, n)`,
// `lanes::argmax(&values, &indices, n)` and their `_unordered` versions. Each one combines the
// first `n` elements of rank-1 tensors, and every lane of the block or the subgroup gets the
// result. docs/lane_reductions.md has the order. `lanes::sort(&mut t, n)` and
// `lanes::sort_descending(&mut t, n)` sort the first `n` elements of a rank-1 tensor of `i64`.
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::{DiagnosticCode, SourceSpan};
use crate::syntax::expr::{Expr, FunctionCallExpr};
use crate::syntax::Type;
use crate::syntax::{ElementType, Span};

/// How a lane reduction combines two values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneOp {
    Sum,
    Max,
    /// The index of the larger value, from a tensor of values and a tensor of indices.
    ArgMax,
    /// A sort in place, from the smallest value up.
    Sort,
    /// A sort in place, from the largest value down.
    SortDescending,
}

impl LaneOp {
    pub fn is_sort(self) -> bool {
        matches!(self, LaneOp::Sort | LaneOp::SortDescending)
    }
}

/// The routines, by the name after `lanes::`.
const ROUTINES: &[(&str, LaneOp, bool)] = &[
    ("sum", LaneOp::Sum, true),
    ("max", LaneOp::Max, true),
    ("sum_unordered", LaneOp::Sum, false),
    ("max_unordered", LaneOp::Max, false),
    ("argmax", LaneOp::ArgMax, true),
    ("argmax_unordered", LaneOp::ArgMax, false),
    ("sort", LaneOp::Sort, true),
    ("sort_descending", LaneOp::SortDescending, true),
];

/// The most elements that a block reduces: 32 groups of 32.
pub const MAX_BLOCK_ELEMENTS: u64 = 1024;

/// The most elements of a sort for each lane of the block or the subgroup. Each lane holds
/// its elements in registers.
pub const MAX_SORT_ELEMENTS_A_LANE: u64 = 8;

/// The operation of a call to a lane reduction, and whether it keeps its order. `None` for any
/// other call.
pub fn lane_routine(name: &str) -> Option<(LaneOp, bool)> {
    let routine = name.strip_prefix("lanes::")?;
    ROUTINES
        .iter()
        .find(|(n, ..)| *n == routine)
        .map(|&(_, op, ordered)| (op, ordered))
}

/// Whether `e` is a call with the reserved prefix `lanes::`, known or not.
pub fn is_lanes_call(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall(fc) if fc.name.as_ref().starts_with("lanes::"))
}

/// The call to a lane sort that is the whole of `e`, also inside `unsafe { .. }`.
pub fn bare_lane_sort(e: &Expr) -> Option<&FunctionCallExpr> {
    bare_lanes_call(e).filter(|fc| lane_routine(&fc.name).is_some_and(|(op, _)| op.is_sort()))
}

/// The call to a lane reduction that is the whole of `e`, also inside `unsafe { .. }`.
pub fn bare_lanes_call(e: &Expr) -> Option<&FunctionCallExpr> {
    match crate::syntax::bare_unsafe_value(e) {
        Expr::FunctionCall(fc) if fc.name.as_ref().starts_with("lanes::") => Some(fc),
        _ => None,
    }
}

impl crate::hir::TypeChecker<'_> {
    /// The type of a call `lanes::<name>(..)`. The size of its tensor goes to `lane_reductions`,
    /// for the checks of `across` loops.
    pub(crate) fn check_lane_routine(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let f32_ty = Type::Scalar(ElementType::F32);
        if lane_routine(name).is_some_and(|(op, _)| op.is_sort()) {
            return self.check_lane_sort(name, args, arg_types, call_span);
        }
        let argmax = lane_routine(name).map(|(op, _)| op) == Some(LaneOp::ArgMax);
        let result = if argmax {
            Type::Scalar(ElementType::I32)
        } else {
            f32_ty.clone()
        };
        if self.speculating {
            return result;
        }
        let span = Some(SourceSpan::from_ast_span(&call_span));
        if lane_routine(name).is_none() {
            let names: Vec<String> = ROUTINES
                .iter()
                .map(|(n, ..)| format!("`lanes::{n}`"))
                .collect();
            self.errors.error_with_code(
                DiagnosticCode::E6032,
                format!(
                    "`{name}` is not a lane routine. The lane routines are {}",
                    names.join(", ")
                ),
                span,
            );
            return f32_ty;
        }
        let wanted = if argmax { 3 } else { 2 };
        if args.len() != wanted {
            let what = if argmax {
                "a borrowed tensor of values, a borrowed tensor of their indices and the number of \
                 elements to combine"
            } else {
                "a borrowed tensor and the number of its elements to combine"
            };
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes {wanted} arguments, {what}, but it got {}",
                    args.len()
                ),
                span,
            );
            return result;
        }
        // The size of a borrowed rank-1 tensor of `elem` whose size is a number.
        let size_of = |t: &Type, elem: ElementType| match t {
            Type::Borrow { inner, .. } => match inner.as_ref() {
                Type::Tensor(e, dims, _) if *e == elem && dims.len() == 1 => {
                    crate::hir::memory::static_extent(&dims[0]).filter(|n| *n >= 1)
                }
                _ => None,
            },
            _ => None,
        };
        match size_of(&arg_types[0], ElementType::F32) {
            Some(n) => {
                self.lane_reductions.insert(call_span, n as u64);
            }
            None => _ = self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow a rank-1 tensor of `f32` with a \
                     size that is a number, such as `&part` for `Tensor<f32, [256], Memory::SMEM>`, \
                     but it is `{}`",
                    arg_types[0]
                ),
                span.clone(),
            ),
        }
        if argmax
            && size_of(&arg_types[1], ElementType::I32) != size_of(&arg_types[0], ElementType::F32)
        {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` must borrow a rank-1 tensor of `i32` indices \
                     of the same size as the values, but it is `{}`",
                    arg_types[1]
                ),
                span.clone(),
            );
        }
        let count = &arg_types[wanted - 1];
        if !matches!(count, Type::Scalar(ElementType::I32)) {
            let which = if argmax { "third" } else { "second" };
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the {which} argument of `{name}` is the number of elements to combine, an \
                     `i32`, but it is `{count}`"
                ),
                span,
            );
        }
        result
    }

    /// The type of a call `lanes::sort(&mut t, n)` or `lanes::sort_descending(&mut t, n)`. The
    /// size of `t` goes to `lane_reductions`, for the checks of `across` loops.
    fn check_lane_sort(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let void = Type::Struct("void".into(), None);
        if self.speculating {
            return void;
        }
        let span = Some(SourceSpan::from_ast_span(&call_span));
        if args.len() != 2 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 2 arguments, a borrowed tensor and the number of its elements \
                     to sort, but it got {}",
                    args.len()
                ),
                span,
            );
            return void;
        }
        let size = match &arg_types[0] {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => match inner.as_ref() {
                Type::Tensor(ElementType::I64, dims, _) if dims.len() == 1 => {
                    crate::hir::memory::static_extent(&dims[0])
                        .filter(|n| *n >= 1 && (*n as u64).is_power_of_two())
                }
                _ => None,
            },
            _ => None,
        };
        match size {
            Some(n) => {
                self.lane_reductions.insert(call_span, n as u64);
            }
            None => {
                _ = self.errors.error_with_code(
                    DiagnosticCode::E3003,
                    format!(
                        "the first argument of `{name}` must borrow a rank-1 tensor of `i64` as \
                     `&mut`, with a size that is a power of two, such as `&mut keys` for \
                     `Tensor<i64, [2048], Memory::SMEM>`, but it is `{}`",
                        arg_types[0]
                    ),
                    span.clone(),
                )
            }
        }
        if !matches!(arg_types[1], Type::Scalar(ElementType::I32)) {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` is the number of elements to sort, an `i32`, \
                     but it is `{}`",
                    arg_types[1]
                ),
                span,
            );
        }
        void
    }
}
