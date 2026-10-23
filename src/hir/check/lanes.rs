//===- lanes.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The lane reductions: `lanes::sum(&t, n)`, `lanes::max(&t, n)` and their `_unordered`
// versions. Each one combines `t[0]` to `t[n - 1]` of a rank-1 `f32` tensor, and every lane of
// the block or the subgroup gets the result. docs/lane_reductions.md has the order.
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
}

/// The routines, by the name after `lanes::`.
const ROUTINES: &[(&str, LaneOp, bool)] = &[
    ("sum", LaneOp::Sum, true),
    ("max", LaneOp::Max, true),
    ("sum_unordered", LaneOp::Sum, false),
    ("max_unordered", LaneOp::Max, false),
];

/// The most elements that a block reduces: 32 groups of 32.
pub const MAX_BLOCK_ELEMENTS: u64 = 1024;

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
        if self.speculating {
            return f32_ty;
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
                    "`{name}` is not a lane reduction. The lane reductions are {}",
                    names.join(", ")
                ),
                span,
            );
            return f32_ty;
        }
        if args.len() != 2 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 2 arguments, a borrowed tensor and the number of its elements \
                     to combine, but it got {}",
                    args.len()
                ),
                span,
            );
            return f32_ty;
        }
        let tensor = match &arg_types[0] {
            Type::Borrow { inner, .. } => match inner.as_ref() {
                Type::Tensor(ElementType::F32, dims, _) if dims.len() == 1 => {
                    crate::hir::memory::static_extent(&dims[0])
                }
                _ => None,
            },
            _ => None,
        };
        match tensor {
            Some(n) if n >= 1 => {
                self.lane_reductions.insert(call_span, n as u64);
            }
            _ => _ = self.errors.error_with_code(
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
        if !matches!(arg_types[1], Type::Scalar(ElementType::I32)) {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` is the number of elements to combine, an \
                     `i32`, but it is `{}`",
                    arg_types[1]
                ),
                span,
            );
        }
        f32_ty
    }
}
