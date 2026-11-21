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
// `lanes::copy_rows_async(&mut tile, &x, &ids, [first, col])` copies the rows of `x` that `ids`
// chooses into a tile in shared memory, in the background.
// `lanes::copy(&mut dst, &src, [i, j])` copies a tile in shared memory to or from a tensor.
// `lanes::fill(&mut tile, v)` sets each element of a tile in shared memory to `v`.
// `lanes::scale_rows(&mut tile, &scale)` multiplies row `i` of a tile by `scale[i]`.
//
// `LANE_ROUTINES` describes each routine once. Both code generators make the op of a call from
// its row, so a new routine needs a row, its op in VxDialect.td and the op's lowering.
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::{DiagnosticCode, SourceSpan};
use crate::syntax::expr::{Expr, FunctionCallExpr};
use crate::syntax::Type;
use crate::syntax::{ElementType, Span};

/// What a lane routine does.
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
    /// A copy of a tile of a tensor into a tile of shared memory, which goes on in the background.
    /// It gives a token.
    CopyAsync,
    /// The end of a copy: `t.wait()` on the token of a `lanes::copy_async`.
    CopyWait,
    /// A copy into a tile of shared memory or out of one, which ends before the call returns.
    Copy,
    /// Sets each element of a tile in shared memory to a value.
    Fill,
    /// `acc = acc + x @ y` or `acc = x @ y` on tiles, in the code of a subgroup.
    Matmul,
    /// Multiplies each row of a tile in shared memory by an element of another tile.
    ScaleRows,
    /// `acc` gets the matmul of two `f4e2m1` tiles, each with a scale for each block of k. The
    /// lanes of a block or of a subgroup compute it together.
    MatmulScaled,
}

impl LaneOp {
    pub fn is_sort(self) -> bool {
        matches!(self, LaneOp::Sort | LaneOp::SortDescending)
    }
}

/// How a program asks for a lane routine.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneSyntax {
    /// A call `lanes::<name>(..)`.
    Call,
    /// A call that the checker writes in place of other code: `lanes::copy_wait(t)` for
    /// `t.wait()`. The message of an unknown routine leaves it out.
    Written,
    /// An assignment with `@` in the code of a subgroup. No name calls it.
    Operator,
}

/// An argument of a lane routine. It says how the code generators lower the argument.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneArg {
    /// A borrowed tensor of rank 1, such as the tensor of a reduction.
    Vector,
    /// A borrowed tensor with as many dimensions as the place of the call has numbers.
    Tile,
    /// A borrowed tensor of any rank, such as the tile of a fill.
    Tensor,
    /// A matrix with one of these element types, written as a place: an operand of `@`.
    Matrix(&'static [ElementType]),
    /// An `i32`, such as the number of elements to combine, or a copy token.
    I32,
    /// The place of a tile, `[i, j, ..]`. The op takes each number as an operand.
    Place,
    /// A number, such as the value of a fill.
    Scalar,
}

impl LaneArg {
    /// Whether the op takes the argument as a memref.
    pub fn is_tensor(self) -> bool {
        matches!(
            self,
            LaneArg::Vector | LaneArg::Tile | LaneArg::Tensor | LaneArg::Matrix(_)
        )
    }
}

/// The value of an attribute of a lane op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneAttr {
    /// A unit attribute, such as `ordered`.
    Unit,
    /// A string, such as the `"sum"` of `kind = "sum"`.
    Str(&'static str),
    /// An `i64`, such as the `32` of `lanes = 32 : i64`.
    I64(u64),
}

/// Where a call to a lane routine runs. It gives the attributes of the op that change from one
/// call to the next.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct LaneSite {
    /// In a region whose `across` loops run on one lane: the loops of the op on the CPU do all of
    /// the work.
    pub serial: bool,
    /// In the code of a subgroup: the lanes of the subgroup share the work, not those of the block.
    pub subgroup: bool,
    /// The tile of the call is `t[w]`, where `w` is the index of a subgroup loop that gives each
    /// subgroup one pass at most.
    pub subgroup_row: bool,
    /// Each tensor of the call is the same row of a tile that the code of the subgroup alone
    /// writes: 0 for `t` or `t[w]`, and `c + 1` for `t[w][c]`.
    pub own_row: Option<u64>,
    /// With `own_row`: the subgroup has the row for the whole kernel, so the op may read it from
    /// the registers of the lanes (docs/lane_reductions.md).
    pub lane_own: bool,
    /// The lanes that share the work, for a routine whose op has `lanes`.
    pub lanes: u64,
}

/// The type of the value of a lane op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LaneValue {
    F32,
    I32,
}

impl LaneValue {
    pub fn element(self) -> ElementType {
        match self {
            LaneValue::F32 => ElementType::F32,
            LaneValue::I32 => ElementType::I32,
        }
    }
}

/// A lane routine, and the op that both code generators make for a call to it.
#[derive(Debug)]
pub struct LaneRoutine {
    /// The name after `lanes::`.
    pub name: &'static str,
    pub syntax: LaneSyntax,
    pub op: LaneOp,
    /// Whether it keeps the order that docs/lane_reductions.md gives.
    pub ordered: bool,
    /// The name of the MLIR op.
    pub mlir: &'static str,
    pub args: &'static [LaneArg],
    /// The type of the value of the op, or `None` for an op with no value.
    pub result: Option<LaneValue>,
    /// The attributes that the op of each call has.
    pub attrs: &'static [(&'static str, LaneAttr)],
    /// Whether the op has `lanes`, the number of lanes that share the work.
    pub lanes: bool,
    /// The argument whose tile can be the row of the subgroup, for `subgroup_row`.
    pub row_of: Option<usize>,
    /// Whether the op can have `own_row`.
    pub own_row: bool,
    /// Whether the call writes the tensor that its first argument borrows.
    pub writes_first: bool,
}

impl LaneRoutine {
    const fn call(
        name: &'static str,
        op: LaneOp,
        mlir: &'static str,
        args: &'static [LaneArg],
        result: Option<LaneValue>,
    ) -> Self {
        LaneRoutine {
            name,
            syntax: LaneSyntax::Call,
            op,
            ordered: true,
            mlir,
            args,
            result,
            attrs: &[],
            lanes: false,
            row_of: None,
            own_row: false,
            writes_first: false,
        }
    }

    const fn with_syntax(self, syntax: LaneSyntax) -> Self {
        LaneRoutine { syntax, ..self }
    }

    const fn unordered(self) -> Self {
        LaneRoutine {
            ordered: false,
            ..self
        }
    }

    const fn with_attrs(self, attrs: &'static [(&'static str, LaneAttr)]) -> Self {
        LaneRoutine { attrs, ..self }
    }

    const fn with_lanes(self) -> Self {
        LaneRoutine {
            lanes: true,
            ..self
        }
    }

    const fn with_row_of(self, arg: usize) -> Self {
        LaneRoutine {
            row_of: Some(arg),
            ..self
        }
    }

    const fn with_own_row(self) -> Self {
        LaneRoutine {
            own_row: true,
            ..self
        }
    }

    const fn writing_first(self) -> Self {
        LaneRoutine {
            writes_first: true,
            ..self
        }
    }

    /// Whether a call is a statement of its own, not the value of a `let`.
    pub fn is_statement(&self) -> bool {
        self.result.is_none()
    }

    /// The position of the routine in `LANE_ROUTINES`.
    pub fn index(&'static self) -> usize {
        LANE_ROUTINES
            .iter()
            .position(|r| std::ptr::eq(r, self))
            .expect("a lane routine is a row of LANE_ROUTINES")
    }

    /// The tile whose row of the subgroup gives the op `subgroup_row`: the argument `row_of`,
    /// without its `&`.
    pub fn row_tile<'e>(&self, args: &[&'e Expr]) -> Option<&'e Expr> {
        let at = self.row_of?;
        match (self.args[at], args.get(at)?) {
            (LaneArg::Matrix(_), e) => Some(e),
            (_, Expr::Borrow(b)) => Some(&b.expr),
            _ => None,
        }
    }

    /// The row of the subgroup that each tensor argument borrows, when they all borrow the same
    /// row. `row_number` gives the row of a tile, or `None` for a tile that is not a row.
    pub fn own_row_of(
        &self,
        args: &[&Expr],
        row_number: impl Fn(&Expr) -> Option<u64>,
    ) -> Option<u64> {
        let rows: Vec<Option<u64>> = args
            .iter()
            .zip(self.args)
            .filter(|(_, kind)| kind.is_tensor())
            .map(|(e, _)| match e {
                Expr::Borrow(b) => row_number(&b.expr),
                _ => None,
            })
            .collect();
        let row = (*rows.first()?)?;
        rows.iter().all(|r| *r == Some(row)).then_some(row)
    }

    /// The kind of each of the `operands` of the op. The place gives the operands that the other
    /// arguments leave. `None` when there are too few.
    pub fn operand_kinds(&self, operands: usize) -> Option<Vec<LaneArg>> {
        let fixed = self.args.iter().filter(|a| **a != LaneArg::Place).count();
        let places = operands.checked_sub(fixed)?;
        let mut kinds = Vec::new();
        for kind in self.args {
            match kind {
                LaneArg::Place => kinds.extend(std::iter::repeat_n(LaneArg::Place, places)),
                kind => kinds.push(*kind),
            }
        }
        (kinds.len() == operands).then_some(kinds)
    }

    /// The attributes of the op of a call at `site`, in the order that both code generators
    /// give them.
    pub fn op_attributes(&self, site: &LaneSite) -> Vec<(&'static str, LaneAttr)> {
        let mut attrs = self.attrs.to_vec();
        if self.lanes {
            attrs.push(("lanes", LaneAttr::I64(site.lanes)));
        }
        if site.serial {
            attrs.push(("serial", LaneAttr::Unit));
        }
        if site.subgroup {
            attrs.push(("subgroup", LaneAttr::Unit));
        }
        if site.subgroup_row {
            assert!(
                self.row_of.is_some(),
                "`{}` has no tile for `subgroup_row`",
                self.name
            );
            attrs.push(("subgroup_row", LaneAttr::Unit));
        }
        if let Some(row) = site.own_row {
            assert!(self.own_row, "`{}` cannot have `own_row`", self.name);
            attrs.push(("own_row", LaneAttr::Unit));
            if site.lane_own {
                attrs.push(("vx.lane_own", LaneAttr::Unit));
            }
            if row > 0 {
                attrs.push(("vx.lane_row", LaneAttr::I64(row)));
            }
        }
        assert!(
            !site.lane_own || site.own_row.is_some(),
            "`vx.lane_own` on `{}` needs `own_row`",
            self.name
        );
        attrs
    }
}

/// The element types of the accumulator of a lane matmul, and of the tiles that it multiplies.
const MATMUL_ACC: &[ElementType] = &[ElementType::F32];
const MATMUL_TILES: &[ElementType] = &[
    ElementType::F32,
    ElementType::BF16,
    ElementType::F16,
    ElementType::F8E4M3,
    ElementType::F8E5M2,
];

const KIND_SUM: (&str, LaneAttr) = ("kind", LaneAttr::Str("sum"));
const KIND_MAX: (&str, LaneAttr) = ("kind", LaneAttr::Str("max"));
const ORDERED: (&str, LaneAttr) = ("ordered", LaneAttr::Unit);

/// The lane routines.
pub const LANE_ROUTINES: &[LaneRoutine] = {
    use LaneArg::{Matrix, Place, Scalar, Tensor, Tile, Vector, I32};
    use LaneOp::*;
    use LaneValue::{F32, I32 as INT};
    let reduce: &[LaneArg] = &[Vector, I32];
    let copy: &[LaneArg] = &[Tile, Tile, Place];
    // The tile, the tensor, the ids of the rows and the place `[first, col]`.
    let copy_rows: &[LaneArg] = &[Tile, Tile, Vector, Place];
    let matmul: &[LaneArg] = &[
        Matrix(MATMUL_ACC),
        Matrix(MATMUL_TILES),
        Matrix(MATMUL_TILES),
    ];
    &[
        LaneRoutine::call("sum", Sum, "vx.lane_reduce", reduce, Some(F32))
            .with_attrs(&[KIND_SUM, ORDERED])
            .with_own_row(),
        LaneRoutine::call("max", Max, "vx.lane_reduce", reduce, Some(F32))
            .with_attrs(&[KIND_MAX, ORDERED])
            .with_own_row(),
        LaneRoutine::call("sum_unordered", Sum, "vx.lane_reduce", reduce, Some(F32))
            .unordered()
            .with_attrs(&[KIND_SUM])
            .with_own_row(),
        LaneRoutine::call("max_unordered", Max, "vx.lane_reduce", reduce, Some(F32))
            .unordered()
            .with_attrs(&[KIND_MAX])
            .with_own_row(),
        LaneRoutine::call(
            "argmax",
            ArgMax,
            "vx.lane_argmax",
            &[Vector, Vector, I32],
            Some(INT),
        )
        .with_attrs(&[ORDERED])
        .with_own_row(),
        LaneRoutine::call(
            "argmax_unordered",
            ArgMax,
            "vx.lane_argmax",
            &[Vector, Vector, I32],
            Some(INT),
        )
        .unordered()
        .with_own_row(),
        LaneRoutine::call("sort", Sort, "vx.lane_sort", reduce, None)
            .with_lanes()
            .writing_first(),
        LaneRoutine::call(
            "sort_descending",
            SortDescending,
            "vx.lane_sort",
            reduce,
            None,
        )
        .with_attrs(&[("descending", LaneAttr::Unit)])
        .with_lanes()
        .writing_first(),
        LaneRoutine::call(
            "copy_async",
            CopyAsync,
            "vx.lane_copy_async",
            copy,
            Some(INT),
        )
        .with_lanes()
        .writing_first(),
        // A copy of the rows that a list of ids chooses. It is a `copy_async` in every rule but its
        // arguments.
        LaneRoutine::call(
            "copy_rows_async",
            CopyAsync,
            "vx.lane_copy_rows_async",
            copy_rows,
            Some(INT),
        )
        .with_lanes()
        .writing_first(),
        LaneRoutine::call("copy_wait", CopyWait, "vx.lane_copy_wait", &[I32], None)
            .with_syntax(LaneSyntax::Written),
        LaneRoutine::call("copy", Copy, "vx.lane_copy", copy, None)
            .with_lanes()
            .with_row_of(1)
            .writing_first(),
        LaneRoutine::call("fill", Fill, "vx.lane_fill", &[Tensor, Scalar], None)
            .with_lanes()
            .with_row_of(0)
            .writing_first(),
        LaneRoutine::call(
            "scale_rows",
            ScaleRows,
            "vx.lane_scale_rows",
            &[Tensor, Tensor],
            None,
        )
        .with_lanes()
        .with_row_of(0)
        .writing_first(),
        LaneRoutine::call(
            "matmul_scaled",
            MatmulScaled,
            "vx.lane_matmul_scaled",
            &[Tensor, Tensor, Tensor, Tensor, Tensor],
            None,
        )
        .with_lanes()
        .with_row_of(0)
        .writing_first(),
        LaneRoutine::call("matmul", Matmul, "vx.lane_matmul", matmul, None)
            .with_syntax(LaneSyntax::Operator)
            .with_lanes()
            .with_row_of(0)
            .writing_first(),
        LaneRoutine::call("matmul_accumulate", Matmul, "vx.lane_matmul", matmul, None)
            .with_syntax(LaneSyntax::Operator)
            .with_attrs(&[("accumulate", LaneAttr::Unit)])
            .with_lanes()
            .with_row_of(0)
            .writing_first(),
    ]
};

/// The name of the copy of rows that a list of ids chooses.
pub const COPY_ROWS_ASYNC: &str = "lanes::copy_rows_async";

/// The most elements that a block reduces: 32 groups of 32.
pub const MAX_BLOCK_ELEMENTS: u64 = 1024;

/// The most elements of a sort for each lane of the block or the subgroup. Each lane holds
/// its elements in registers.
pub const MAX_SORT_ELEMENTS_A_LANE: u64 = 8;

/// The routine of a call `lanes::<name>(..)`. `None` for any other call.
pub fn lane_routine(name: &str) -> Option<&'static LaneRoutine> {
    let routine = name.strip_prefix("lanes::")?;
    LANE_ROUTINES
        .iter()
        .find(|r| r.syntax != LaneSyntax::Operator && r.name == routine)
}

/// The routine of a lane matmul, which adds its products to the old `acc` when `accumulate`.
pub fn matmul_routine(accumulate: bool) -> &'static LaneRoutine {
    let name = if accumulate {
        "matmul_accumulate"
    } else {
        "matmul"
    };
    LANE_ROUTINES
        .iter()
        .find(|r| r.syntax == LaneSyntax::Operator && r.name == name)
        .expect("LANE_ROUTINES has both lane matmuls")
}

/// A call to a lane routine as a message names it. The checker writes `t.wait()` on a copy token
/// as `lanes::copy_wait(t)`, which the program does not show.
pub fn shown_name(name: &str) -> String {
    if name == MATMUL {
        "a matmul of tiles".to_string()
    } else if lane_routine(name).is_some_and(|r| r.op == LaneOp::CopyWait) {
        "the `.wait()` of a copy token".to_string()
    } else {
        format!("`{name}`")
    }
}

/// Whether `e` is a call with the reserved prefix `lanes::`, known or not.
pub fn is_lanes_call(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall(fc) if fc.name.as_ref().starts_with("lanes::"))
}

/// The call to a lane routine that is a statement of its own, the whole of `e`, also inside
/// `unsafe { .. }`.
pub fn bare_lane_statement(e: &Expr) -> Option<&FunctionCallExpr> {
    bare_lanes_call(e).filter(|fc| lane_routine(&fc.name).is_some_and(|r| r.is_statement()))
}

/// The call of `t = lanes::copy_async(..)`, which gives the token variable `t` its next copy, as a
/// pipeline does when it starts the next copy before it waits for this one.
pub fn next_copy(a: &crate::syntax::AssignStmt) -> Option<&FunctionCallExpr> {
    match &a.lhs {
        Expr::Identifier(_) if calls_routine(&a.rhs, LaneOp::CopyAsync) => bare_lanes_call(&a.rhs),
        _ => None,
    }
}

/// Whether `op` is a lane routine that `e` calls.
pub fn calls_routine(e: &Expr, op: LaneOp) -> bool {
    bare_lanes_call(e).is_some_and(|fc| lane_routine(&fc.name).is_some_and(|r| r.op == op))
}

/// The call to a lane reduction that is the whole of `e`, also inside `unsafe { .. }`.
pub fn bare_lanes_call(e: &Expr) -> Option<&FunctionCallExpr> {
    match crate::syntax::bare_unsafe_value(e) {
        Expr::FunctionCall(fc) if fc.name.as_ref().starts_with("lanes::") => Some(fc),
        _ => None,
    }
}

/// The name that the checks of `across` loops give a lane matmul.
pub const MATMUL: &str = "@";

/// `acc = acc + x @ y`, or `acc = x @ y`, where each of `x` and `y` is a tile or `t as f32` of a
/// tile. In the code of a subgroup, every lane of the subgroup computes it together.
pub struct LaneMatmul<'a> {
    pub acc: &'a Expr,
    /// The tiles, without their `as f32`.
    pub lhs: &'a Expr,
    pub rhs: &'a Expr,
    /// Whether the products are added to the old `acc`.
    pub accumulate: bool,
    /// The `@`.
    pub at: Span,
}

/// The matmul of tiles that the assignment `a` is, if it is one.
pub fn lane_matmul(a: &crate::syntax::AssignStmt) -> Option<LaneMatmul<'_>> {
    use crate::syntax::BinaryOp;
    if !is_place(&a.lhs) {
        return None;
    }
    let (product, accumulate) = match &a.rhs {
        Expr::BinaryOp(sum) if sum.op == BinaryOp::Add && same_place(&sum.lhs, &a.lhs) => {
            (&*sum.rhs, true)
        }
        other => (other, false),
    };
    let Expr::BinaryOp(mm) = product else {
        return None;
    };
    if mm.op != BinaryOp::MatMul {
        return None;
    }
    Some(LaneMatmul {
        acc: &a.lhs,
        lhs: matmul_tile(&mm.lhs)?,
        rhs: matmul_tile(&mm.rhs)?,
        accumulate,
        at: mm.span,
    })
}

/// The tile of an operand of a lane matmul: `t`, or `t` of `t as f32`.
fn matmul_tile(e: &Expr) -> Option<&Expr> {
    match e {
        Expr::AsCast(c) if c.target_ty == Type::Scalar(ElementType::F32) && is_place(&c.expr) => {
            Some(&c.expr)
        }
        e if is_place(e) => Some(e),
        _ => None,
    }
}

/// A variable, or an element or a row of one: `t`, `t[w]`, `t[w][1]`.
fn is_place(e: &Expr) -> bool {
    match e {
        Expr::Identifier(_) => true,
        Expr::IndexAccess(ix) => is_place(&ix.base),
        _ => false,
    }
}

/// Whether `a` and `b` are the same place, written the same way.
fn same_place(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Identifier(x), Expr::Identifier(y)) => x.name == y.name,
        (Expr::Number(x), Expr::Number(y)) => x.value == y.value,
        (Expr::IndexAccess(x), Expr::IndexAccess(y)) => {
            same_place(&x.base, &y.base) && same_place(&x.index, &y.index)
        }
        (Expr::BinaryOp(x), Expr::BinaryOp(y)) => {
            x.op == y.op && same_place(&x.lhs, &y.lhs) && same_place(&x.rhs, &y.rhs)
        }
        _ => false,
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
        match lane_routine(name).map(|r| r.op) {
            Some(op) if op.is_sort() => {
                return self.check_lane_sort(name, args, arg_types, call_span)
            }
            Some(LaneOp::CopyAsync) if name == COPY_ROWS_ASYNC => {
                return self.check_copy_rows_async(name, args, arg_types, call_span)
            }
            Some(LaneOp::CopyAsync) => {
                return self.check_copy_async(name, args, arg_types, call_span)
            }
            Some(LaneOp::CopyWait) => {
                return self.check_copy_wait(name, args, arg_types, call_span)
            }
            Some(LaneOp::Copy) => return self.check_copy(name, args, arg_types, call_span),
            Some(LaneOp::Fill) => return self.check_fill(name, args, arg_types, call_span),
            Some(LaneOp::ScaleRows) => {
                return self.check_scale_rows(name, args, arg_types, call_span)
            }
            Some(LaneOp::MatmulScaled) => {
                return self.check_matmul_scaled(name, args, arg_types, call_span)
            }
            _ => {}
        }
        let argmax = lane_routine(name).map(|r| r.op) == Some(LaneOp::ArgMax);
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
            let names: Vec<String> = LANE_ROUTINES
                .iter()
                .filter(|r| r.syntax == LaneSyntax::Call)
                .map(|r| format!("`lanes::{}`", r.name))
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

    /// The type of a call `lanes::copy_async(&mut dst, &src, [i, j, ..])`: a token of the copy of
    /// the tile of `src` that starts at `[i, j, ..]` into `dst`, which is in shared memory. An
    /// element of the tile past the end of `src` copies as zero.
    fn check_copy_async(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let token = crate::hir::check::transfer::token_of(Type::Struct("void".into(), None));
        if self.speculating {
            return token;
        }
        let span = Some(SourceSpan::from_ast_span(&call_span));
        if args.len() != 3 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 3 arguments, a `&mut` tile in shared memory, a borrowed \
                     tensor and the place of the tile in it, such as `[r, c]`, but it got {}",
                    args.len()
                ),
                span,
            );
            return token;
        }
        let shared = |p: &Option<crate::syntax::Placement>| self.in_shared_memory(p);
        let dst = match &arg_types[0] {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => match inner.as_ref() {
                Type::Tensor(e, dims, p)
                    if shared(p)
                        && dims.iter().all(|d| {
                            crate::hir::memory::static_extent(d).is_some_and(|n| n >= 1)
                        })
                        && matches!(crate::hir::memory::element_bytes(e), Some(2 | 4 | 8)) =>
                {
                    Some((e.clone(), dims.len()))
                }
                _ => None,
            },
            _ => None,
        };
        let Some((elem, rank)) = dst else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow a tile in shared memory as `&mut`, \
                     with a size that is a number and elements of 2, 4 or 8 bytes, such as \
                     `&mut tiles[0]` for `Tensor<f32, [2, 64, 64], Memory::SMEM>`, but it is `{}`",
                    arg_types[0]
                ),
                span,
            );
            return token;
        };
        // `x` and `&x` for a parameter `x : &Tensor<..>` both name the tensor.
        let mut src = &arg_types[1];
        while let Type::Borrow { inner, .. } = src {
            src = inner;
        }
        let src_ok = !matches!(arg_types[1], Type::Tensor(..))
            && matches!(src, Type::Tensor(e, dims, _) if *e == elem && dims.len() == rank);
        if !src_ok {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` must borrow a tensor of `{elem}` with {rank} \
                     dimensions, as the tile has, but it is `{}`",
                    arg_types[1]
                ),
                span.clone(),
            );
        }
        let place_ok = matches!(&args[2], Expr::Array(a) if a.elements.len() == rank)
            && matches!(
                &arg_types[2],
                Type::Tensor(ElementType::I32 | ElementType::I64, ..)
            );
        if !place_ok {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the third argument of `{name}` is the place of the tile, written as a list of \
                     {rank} integers such as `[r, c]`"
                ),
                span,
            );
        }
        token
    }

    /// The type of a call `lanes::copy_rows_async(&mut tile, &x, &ids, [first, col])`: a token of
    /// the copy of row `ids[first + r]` of `x`, from column `col`, into row `r` of `tile`, which is
    /// in shared memory. A row whose id is not a row of `x`, or past the end of `ids`, is zeros.
    fn check_copy_rows_async(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let token = crate::hir::check::transfer::token_of(Type::Struct("void".into(), None));
        if self.speculating {
            return token;
        }
        let span = Some(SourceSpan::from_ast_span(&call_span));
        if args.len() != 4 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 4 arguments, a `&mut` tile in shared memory, a borrowed \
                     tensor, a borrowed tensor of `i32` ids and the place `[first, col]`, but it \
                     got {}",
                    args.len()
                ),
                span,
            );
            return token;
        }
        let shared = |p: &Option<crate::syntax::Placement>| self.in_shared_memory(p);
        let elem = match &arg_types[0] {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => match inner.as_ref() {
                Type::Tensor(e, dims, p)
                    if shared(p)
                        && dims.len() == 2
                        && dims.iter().all(|d| {
                            crate::hir::memory::static_extent(d).is_some_and(|n| n >= 1)
                        })
                        && matches!(crate::hir::memory::element_bytes(e), Some(2 | 4 | 8)) =>
                {
                    Some(e.clone())
                }
                _ => None,
            },
            _ => None,
        };
        let Some(elem) = elem else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow a tile with 2 dimensions in \
                     shared memory as `&mut`, with a size that is a number and elements of 2, 4 \
                     or 8 bytes, such as `&mut tiles[0]` for `Tensor<bf16, [2, 16, 128], \
                     Memory::SMEM>`, but it is `{}`",
                    arg_types[0]
                ),
                span,
            );
            return token;
        };
        // `x` and `&x` for a parameter `x : &Tensor<..>` both name the tensor.
        let borrowed = |t: &Type| {
            let mut inner = t;
            while let Type::Borrow { inner: i, .. } = inner {
                inner = i;
            }
            (!matches!(t, Type::Tensor(..))).then_some(inner.clone())
        };
        if !matches!(borrowed(&arg_types[1]), Some(Type::Tensor(e, dims, _)) if e == elem && dims.len() == 2)
        {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` must borrow a tensor of `{elem}` with 2 \
                     dimensions, as the tile has, but it is `{}`",
                    arg_types[1]
                ),
                span.clone(),
            );
        }
        if !matches!(borrowed(&arg_types[2]), Some(Type::Tensor(ElementType::I32, dims, _)) if dims.len() == 1)
        {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the third argument of `{name}` must borrow a tensor of `i32` ids with 1 \
                     dimension, such as `&ids[w]` for `Tensor<i32, [8, 32], Memory::SMEM>`, but \
                     it is `{}`",
                    arg_types[2]
                ),
                span.clone(),
            );
        }
        let place_ok = matches!(&args[3], Expr::Array(a) if a.elements.len() == 2)
            && matches!(
                &arg_types[3],
                Type::Tensor(ElementType::I32 | ElementType::I64, ..)
            );
        if !place_ok {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the fourth argument of `{name}` is `[first, col]`: the index of the id of \
                     the first row in the ids, and the column of `x` where the rows start"
                ),
                span,
            );
        }
        token
    }

    /// Whether a tensor with placement `p` is in a memory that the lanes of a block share.
    fn in_shared_memory(&self, p: &Option<crate::syntax::Placement>) -> bool {
        p.as_ref().is_some_and(|p| {
            self.env.memories.values().any(|d| {
                crate::syntax::MemorySpace::from_name(d.name.as_ref()) == p.space
                    && matches!(
                        d.scope,
                        Some(crate::syntax::Scope::Cta | crate::syntax::Scope::Sm)
                    )
            })
        })
    }

    /// The type of a call `lanes::copy(&mut dst, &src, [i, j, ..])`, which copies a tile in shared
    /// memory to or from the part of a tensor that starts at `[i, j, ..]`. When `dst` is a tile in
    /// shared memory, the copy goes into it, as `lanes::copy_async` does. Otherwise `src` is the
    /// tile, and the copy goes into `dst`. The shape of the tile goes to `lane_copies_out`, for
    /// the proof that the passes of an `across` loop write apart.
    fn check_copy(
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
        if args.len() != 3 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 3 arguments, a `&mut` tensor, a borrowed tensor and the place \
                     of the tile in the tensor that is not in shared memory, such as `[r, c]`, but \
                     it got {}",
                    args.len()
                ),
                span,
            );
            return void;
        }
        // The tensor behind a borrow, and whether the borrow is `&mut`.
        let tensor = |t: &Type| match t {
            Type::Borrow { inner, is_mut, .. } => match inner.as_ref() {
                Type::Tensor(e, dims, p) => Some((e.clone(), dims.clone(), p.clone(), *is_mut)),
                _ => None,
            },
            _ => None,
        };
        let tile_dims = |dims: &[crate::syntax::Dim]| {
            dims.iter()
                .map(|d| {
                    crate::hir::memory::static_extent(d)
                        .filter(|n| *n >= 1)
                        .and_then(|n| u64::try_from(n).ok())
                })
                .collect::<Option<Vec<u64>>>()
        };
        let (dst, src) = (tensor(&arg_types[0]), tensor(&arg_types[1]));
        let Some((elem, dst_dims, dst_place, true)) = dst else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow a tensor as `&mut`, such as \
                     `&mut tiles[0]` or a parameter `c : &mut Tensor<..>`, but it is `{}`",
                    arg_types[0]
                ),
                span,
            );
            return void;
        };
        let into_tile = self.in_shared_memory(&dst_place);
        let tile = match (&src, into_tile) {
            (Some((e, dims, _, _)), true) if *e == elem && dims.len() == dst_dims.len() => {
                tile_dims(&dst_dims)
            }
            (Some((e, dims, p, _)), false)
                if *e == elem && dims.len() == dst_dims.len() && self.in_shared_memory(p) =>
            {
                tile_dims(dims)
            }
            _ => None,
        };
        let Some(tile) = tile else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` copies a tile in shared memory with a size that is a number, into \
                     the first argument or out of the second. Both must borrow tensors of `{elem}` \
                     with {} dimensions, but they are `{}` and `{}`",
                    dst_dims.len(),
                    arg_types[0],
                    arg_types[1]
                ),
                span,
            );
            return void;
        };
        let rank = tile.len();
        let place_ok = matches!(&args[2], Expr::Array(a) if a.elements.len() == rank)
            && matches!(
                &arg_types[2],
                Type::Tensor(ElementType::I32 | ElementType::I64, ..)
            );
        if !place_ok {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the third argument of `{name}` is the place of the tile, written as a list of \
                     {rank} integers such as `[r, c]`"
                ),
                span,
            );
        }
        if !into_tile {
            self.lane_copies_out.insert(call_span, tile);
        }
        void
    }

    /// The type of a call `lanes::fill(&mut tile, v)`, which sets each element of `tile`, a tile
    /// in shared memory with a size that is a number, to `v`.
    fn check_fill(
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
                    "`{name}` takes 2 arguments, a `&mut` tile in shared memory and the value of \
                     each element, but it got {}",
                    args.len()
                ),
                span,
            );
            return void;
        }
        let tile = match &arg_types[0] {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => match inner.as_ref() {
                Type::Tensor(e, dims, p)
                    if self.in_shared_memory(p)
                        && dims.iter().all(|d| {
                            crate::hir::memory::static_extent(d).is_some_and(|n| n >= 1)
                        }) =>
                {
                    Some(e.clone())
                }
                _ => None,
            },
            _ => None,
        };
        let Some(elem) = tile else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow a tile in shared memory as `&mut`, \
                     with a size that is a number, such as `&mut acc[w]` for `Tensor<f32, [4, 32, \
                     32], Memory::SMEM>`, but it is `{}`",
                    arg_types[0]
                ),
                span,
            );
            return void;
        };
        if !matches!(&arg_types[1], Type::Scalar(e) if *e == elem) {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` is the value of each element, so its type is \
                     the element type of the tile, `{elem}`, but it is `{}`",
                    arg_types[1]
                ),
                span,
            );
        }
        void
    }

    /// The type of a call `lanes::scale_rows(&mut tile, &scale)`, which multiplies row `i` of
    /// `tile`, an `f32` matrix in shared memory, by `scale[i]`. `scale` is an `f32` tile in shared
    /// memory with one element for each row. Both sizes are numbers.
    fn check_scale_rows(
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
                    "`{name}` takes 2 arguments, a `&mut` tile in shared memory and a `&` tile of \
                     the scale of each row, but it got {}",
                    args.len()
                ),
                span,
            );
            return void;
        }
        // The sizes of an `f32` tile in shared memory, borrowed as `&mut` or as `&`.
        let f32_tile = |t: &Type, is_mut: bool| -> Option<Vec<i128>> {
            match t {
                Type::Borrow {
                    inner, is_mut: m, ..
                } if *m == is_mut => match inner.as_ref() {
                    Type::Tensor(ElementType::F32, dims, p) if self.in_shared_memory(p) => dims
                        .iter()
                        .map(|d| crate::hir::memory::static_extent(d).filter(|&n| n >= 1))
                        .collect(),
                    _ => None,
                },
                _ => None,
            }
        };
        let Some(&[rows, _]) = f32_tile(&arg_types[0], true).as_deref() else {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the first argument of `{name}` must borrow an `f32` matrix in shared memory as \
                     `&mut`, with sizes that are numbers, such as `&mut acc[w]` for `Tensor<f32, \
                     [4, 16, 64], Memory::SMEM>`, but it is `{}`",
                    arg_types[0]
                ),
                span,
            );
            return void;
        };
        if f32_tile(&arg_types[1], false).as_deref() != Some(&[rows]) {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the second argument of `{name}` must borrow an `f32` tile in shared memory as \
                     `&`, with one element for each of the {rows} rows of the first, such as \
                     `&scale[w]` for `Tensor<f32, [4, {rows}], Memory::SMEM>`, but it is `{}`",
                    arg_types[1]
                ),
                span,
            );
        }
        void
    }

    /// The type of a call `lanes::matmul_scaled(&mut acc, &a, &sa, &b, &sb)`. It adds to
    /// `acc[i][j]` the sum over `k` of `a[i][k] * sa[i][k / B] * b[k][j] * sb[k / B][j]`, where
    /// `B` is the k size of `a` over the k size of `sa`. `acc` is an `f32` matrix, `a` and `b`
    /// are `f4e2m1` matrices, and `sa` and `sb` are matrices of one scale type, `f8e4m3` (NVFP4)
    /// or `f8e8m0` (MXFP4). All five are in shared memory, and their sizes are numbers.
    fn check_matmul_scaled(
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
        if args.len() != 5 {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "`{name}` takes 5 arguments, `&mut acc`, `&a`, `&sa`, `&b` and `&sb`, but it \
                     got {}",
                    args.len()
                ),
                span,
            );
            return void;
        }
        // The two sizes of a matrix of `elem` in shared memory, borrowed as `&mut` or as `&`.
        let matrix = |t: &Type, elem: &ElementType, is_mut: bool| -> Option<[i128; 2]> {
            let Type::Borrow {
                inner, is_mut: m, ..
            } = t
            else {
                return None;
            };
            let Type::Tensor(e, dims, p) = inner.as_ref() else {
                return None;
            };
            if *m != is_mut || e != elem || dims.len() != 2 || !self.in_shared_memory(p) {
                return None;
            }
            let size = |d| crate::hir::memory::static_extent(d).filter(|&n| n >= 1);
            Some([size(&dims[0])?, size(&dims[1])?])
        };
        // NVFP4 has `f8e4m3` scales and MXFP4 `f8e8m0` scales. `sb` has the type of `sa`.
        let scale = match &arg_types[2] {
            Type::Borrow { inner, .. }
                if matches!(inner.as_ref(), Type::Tensor(ElementType::F8E8M0, ..)) =>
            {
                ElementType::F8E8M0
            }
            _ => ElementType::F8E4M3,
        };
        let wanted = [
            ("first", "`acc`", ElementType::F32, true, "an `f32`"),
            ("second", "`a`", ElementType::F4E2M1, false, "an `f4e2m1`"),
            (
                "third",
                "`sa`",
                scale.clone(),
                false,
                "an `f8e4m3` or `f8e8m0`",
            ),
            ("fourth", "`b`", ElementType::F4E2M1, false, "an `f4e2m1`"),
            ("fifth", "`sb`", scale.clone(), false, "a scale"),
        ];
        let mut sizes = Vec::new();
        for (t, (nth, what, elem, is_mut, shown)) in arg_types.iter().zip(wanted) {
            let Some(size) = matrix(t, &elem, is_mut) else {
                let borrow = if is_mut { "`&mut`" } else { "`&`" };
                let shown = if nth == "fifth" {
                    format!("an `{scale}`")
                } else {
                    shown.to_string()
                };
                let like_sa = if nth == "fifth" { ", as `sa` does" } else { "" };
                self.errors.error_with_code(
                    DiagnosticCode::E3003,
                    format!(
                        "the {nth} argument of `{name}`, {what}, must borrow {shown} matrix in \
                         shared memory as {borrow}{like_sa}, with sizes that are numbers, but it is \
                         `{t}`"
                    ),
                    span,
                );
                return void;
            };
            sizes.push(size);
        }
        let [[m, n], [am, k], [sm, sk], [bk, bn], [sbk, sbn]] = sizes[..] else {
            unreachable!("five matrices");
        };
        let fits = am == m && bn == n && bk == k && sm == m && sbn == n && sbk == sk && k % sk == 0;
        if !fits {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!(
                    "the sizes of the tiles of `{name}` do not fit: with `acc` of [M, N] and `a` \
                     of [M, K], `b` must be [K, N], `sa` [M, K / B] and `sb` [K / B, N], where \
                     the block size B divides K. Here `acc` is [{m}, {n}], `a` [{am}, {k}], `sa` \
                     [{sm}, {sk}], `b` [{bk}, {bn}] and `sb` [{sbk}, {sbn}]"
                ),
                span,
            );
        }
        void
    }

    /// The type of `lanes::copy_wait(t)`, which the checker writes for `t.wait()` on the token of
    /// a `lanes::copy_async`.
    fn check_copy_wait(
        &mut self,
        name: &str,
        args: &[Expr],
        arg_types: &[Type],
        call_span: Span,
    ) -> Type {
        let void = Type::Struct("void".into(), None);
        let is_copy_token =
            arg_types.len() == 1 && crate::syntax::types::is_copy_token(&arg_types[0]);
        if !is_copy_token && !self.speculating {
            self.errors.error_with_code(
                DiagnosticCode::E3003,
                format!("`{name}` takes the token of a `lanes::copy_async`"),
                Some(SourceSpan::from_ast_span(&call_span)),
            );
        }
        // The copy ends at the wait, so the borrow of its tile ends there, even when the variable
        // of the token gets another copy later, as in a pipeline.
        if let (Some(Expr::Identifier(token)), false) = (args.first(), self.speculating) {
            self.borrow.forget_borrows_of(token.name.as_ref());
        }
        void
    }
}
