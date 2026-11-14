//===- across.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The `across` loops of a GPU region on the AST code generator: the same blocks, lanes and
// barriers that the flat code generator makes (`src/hir/flatten/across.rs`).
//
// A loop of blocks starts at the block's id and steps by the number of blocks. A lane loop or
// a subgroup loop starts at the lane's id and steps by the lanes. A barrier follows a lane loop
// in the code of a block or a subgroup. Code there that writes memory runs on lane 0, and a
// barrier follows it.
//
//===----------------------------------------------------------------------===//

use std::collections::HashSet;

use melior::ir::attribute::IntegerAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike, OperationMutLike};
use melior::ir::{BlockLike, Identifier, RegionLike, Type, Value, ValueLike};

use super::LowerError;
use crate::arch::LevelKind;
use crate::codegen::generator::MeliorGenerator;
use crate::hir::flatten::across::{across_width, inner_lane_loop, inner_lanes};
use crate::syntax::{self, Expr, ForLoopStmt, Statement};

type Block<'c> = melior::ir::BlockRef<'c, 'c>;

/// The code of a block or of a subgroup.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Level {
    Block,
    Subgroup,
}

/// What the lowering of a region with `across` loops knows at a point of the region.
#[derive(Default)]
pub(crate) struct AcrossMap<'c> {
    /// The lanes of a block.
    pub lanes: u64,
    /// The code of a block or of a subgroup, or `None` in the code of a lane.
    pub level: Option<Level>,
    /// The axis of the next loop of blocks.
    pub blocks: u64,
    /// The lane loop that is the whole body of the lane loop being lowered, by its address.
    pub inner: Option<usize>,
    /// The lanes of a subgroup.
    pub subgroup: Option<u64>,
    /// The names that only lane 0 has.
    pub lane0: HashSet<String>,
    /// The code being lowered runs on lane 0 alone.
    pub on_lane0: bool,
    /// The row of a tile that the code of a subgroup alone writes.
    pub subgroup_row: Option<SubgroupRow>,
    /// In a lane loop from 0 over all the lanes of its level: the index `t` of the loop, and the
    /// level. Only lane `t` writes `tile[t]` in the code of a block, or element `t` of the row of
    /// the subgroup.
    pub lane_element: Option<(String, Level)>,
    /// The code after the nest of blocks, which the last pass to end runs.
    pub tail: Option<Tail<'c>>,
    /// The body of the loop of blocks whose passes end with an arrival, by its address.
    pub arrive_body: Option<usize>,
}

/// The code after the passes of a nest of blocks, or of a group: each pass of the outer loop is
/// a group of the passes of the inner one. Each pass adds 1 to the element of its group in
/// `count` when it ends. The pass that sees the last count runs `stmts` in its block and sets the
/// element back to 0. `flag`, in shared memory, tells the lanes of the pass whether it was the
/// last one of its group.
pub(crate) struct Tail<'c> {
    pub stmts: Vec<Statement>,
    /// The addresses of the statements in the region, which the region does not lower itself.
    pub skip: HashSet<usize>,
    /// The loop whose passes count, by its address.
    pub counted: usize,
    /// The body of the outer loop of a group, which lowers only its inner loop, by its address.
    pub outer_body: Option<usize>,
    /// The variable of the outer loop of a group, and its range.
    pub group: Option<(crate::symbol::Symbol, syntax::RangeExpr)>,
    /// The range of the loop whose passes count.
    pub range: syntax::RangeExpr,
    pub count: Value<'c, 'c>,
    pub flag: Option<Value<'c, 'c>>,
}

/// The row of a tile that the code of a subgroup alone writes.
#[derive(Clone)]
pub(crate) enum SubgroupRow {
    /// `tile[w]`, with `w` the index of the subgroup loop.
    Index(String),
    /// The whole tile, when the subgroup loop has one subgroup.
    Whole,
}

impl AcrossMap<'_> {
    /// When `e` is a row of a tile that the code of this subgroup alone writes, the number that
    /// tells it from the other rows of the subgroup: 0 for `t` or `t[w]`, and `c + 1` for
    /// `t[w][c]` with a number `c`.
    fn subgroup_row_number(&self, e: &Expr) -> Option<u64> {
        let is_own_row = |e: &Expr| match (&self.subgroup_row, e) {
            (Some(SubgroupRow::Whole), Expr::Identifier(_)) => true,
            (Some(SubgroupRow::Index(w)), Expr::IndexAccess(r)) => {
                matches!(&*r.base, Expr::Identifier(_))
                    && matches!(&*r.index, Expr::Identifier(id) if id.name.as_ref() == w)
            }
            _ => false,
        };
        if is_own_row(e) {
            return Some(0);
        }
        match e {
            Expr::IndexAccess(r) if is_own_row(&r.base) => match &*r.index {
                Expr::Number(n) => n.value.parse::<u64>().ok().map(|c| c + 1),
                _ => None,
            },
            _ => None,
        }
    }
}

/// The attributes of a store `ix = ..` that writes element `t` of a tile, which only lane `t`
/// writes: `vx.lane_element`, and `vx.lane_row` for a row after the first. The device copy of a
/// kernel can then keep the element in a register.
pub(crate) fn lane_element_attrs<'c>(
    gen: &MeliorGenerator<'c>,
    ix: &syntax::IndexAccessExpr,
) -> Vec<(Identifier<'c>, melior::ir::Attribute<'c>)> {
    let Some(map) = &gen.across_map else {
        return Vec::new();
    };
    let Some((lane, level)) = &map.lane_element else {
        return Vec::new();
    };
    if !matches!(&*ix.index, Expr::Identifier(id) if id.name.as_ref() == lane) {
        return Vec::new();
    }
    let string = |s: &str| melior::ir::attribute::StringAttribute::new(gen.context, s).into();
    match level {
        Level::Block if matches!(&*ix.base, Expr::Identifier(_)) => {
            vec![(
                Identifier::new(gen.context, "vx.lane_element"),
                string("block"),
            )]
        }
        Level::Subgroup => match map.subgroup_row_number(&ix.base) {
            Some(row) => {
                let mut attrs = vec![(
                    Identifier::new(gen.context, "vx.lane_element"),
                    string("subgroup"),
                )];
                if row > 0 {
                    attrs.push((
                        Identifier::new(gen.context, "vx.lane_row"),
                        IntegerAttribute::new(gen.i64_ty, row as i64).into(),
                    ));
                }
                attrs
            }
            None => Vec::new(),
        },
        _ => Vec::new(),
    }
}

/// `own_row`, and `vx.lane_row` for a row after the first, when each tensor of a lane reduction
/// in the code of a subgroup is the same row of a tile that the subgroup alone writes.
pub(crate) fn own_row_attrs<'c>(
    gen: &MeliorGenerator<'c>,
    tensors: &[Expr],
) -> Vec<(Identifier<'c>, melior::ir::Attribute<'c>)> {
    let Some(map) = &gen.across_map else {
        return Vec::new();
    };
    if gen.across_kinds.last() != Some(&LevelKind::Subgroups) {
        return Vec::new();
    }
    let rows: Vec<Option<u64>> = tensors
        .iter()
        .map(|t| match t {
            Expr::Borrow(b) => map.subgroup_row_number(&b.expr),
            _ => None,
        })
        .collect();
    let Some(Some(row)) = rows.first().copied() else {
        return Vec::new();
    };
    if !rows.iter().all(|r| *r == Some(row)) {
        return Vec::new();
    }
    let mut attrs = vec![(
        Identifier::new(gen.context, "own_row"),
        melior::ir::Attribute::unit(gen.context),
    )];
    if row > 0 {
        attrs.push((
            Identifier::new(gen.context, "vx.lane_row"),
            IntegerAttribute::new(gen.i64_ty, row as i64).into(),
        ));
    }
    attrs
}

/// The lanes of a block of `s`, when its `across` loops map to blocks and lanes.
pub(crate) fn mapped_lanes(s: &syntax::SpawnOnExpr) -> Option<u64> {
    crate::hir::flatten::across::across_block_lanes(&s.stmts)
}

/// The lanes of a subgroup, in the code of a subgroup of a mapped region.
pub(crate) fn subgroup_lanes(gen: &MeliorGenerator<'_>) -> Result<u64, LowerError> {
    gen.across_map
        .as_ref()
        .and_then(|m| m.subgroup)
        .ok_or_else(|| {
            LowerError::from("a subgroup whose number of lanes is not known".to_string())
        })
}

/// The size of each axis of blocks of `stmts`, computed in `block` before the region, and the
/// lanes of a block as axis 3.
pub(crate) fn launch_shape<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[Statement],
    lanes: u64,
    mut block: Block<'c>,
) -> Result<(Vec<(u64, Value<'c, 'c>)>, Block<'c>), LowerError> {
    let mut shape = Vec::new();
    let mut level = Some(stmts);
    let mut axis = 0;
    while let Some(f) = level.and_then(crate::hir::flatten::across::blocks_loop) {
        let Expr::Range(range) = &*f.iterable else {
            return Err(LowerError::from(
                "a loop of blocks over something that is not a range".to_string(),
            ));
        };
        let starts_at_zero = matches!(&*range.start, Expr::Number(n) if n.value.as_ref() == "0");
        let size = match f.across.as_ref().and_then(|a| a.share.as_deref()) {
            Some(n) => {
                let (v, _, b) = gen.generate_expr(n, block)?;
                block = b;
                v
            }
            None if starts_at_zero => {
                let (v, _, b) = gen.generate_expr(&range.end, block)?;
                block = b;
                v
            }
            None => {
                let (end, _, b) = gen.generate_expr(&range.end, block)?;
                let (start, _, b) = gen.generate_expr(&range.start, b)?;
                block = b;
                binary(gen, block, "arith.subi", end, start)?
            }
        };
        if size.r#type() != gen.i32_ty {
            return Err(LowerError::from(
                "a loop of blocks whose size is not `i32`".to_string(),
            ));
        }
        shape.push((axis, size));
        axis += 1;
        level = Some(f.body.as_slice());
    }
    let lanes = constant(gen, block, lanes)?;
    shape.push((3, lanes));
    Ok((shape, block))
}

/// `vx.launch_dim` for each axis, at the start of the region.
pub(crate) fn emit_launch_shape<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    shape: &[(u64, Value<'c, 'c>)],
) -> Result<(), LowerError> {
    for (axis, v) in shape {
        block.append_operation(
            OperationBuilder::new("vx.launch_dim", gen.loc())
                .add_operands(&[*v])
                .add_attributes(&[(
                    Identifier::new(gen.context, "axis"),
                    IntegerAttribute::new(gen.i32_ty, *axis as i64).into(),
                )])
                .build()?,
        );
    }
    Ok(())
}

fn constant<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    v: u64,
) -> Result<Value<'c, 'c>, LowerError> {
    Ok(block
        .append_operation(
            OperationBuilder::new("arith.constant", gen.loc())
                .add_results(&[gen.i32_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "value"),
                    IntegerAttribute::new(gen.i32_ty, v as i64).into(),
                )])
                .build()?,
        )
        .result(0)?
        .into())
}

fn binary<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    op: &str,
    a: Value<'c, 'c>,
    b: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    Ok(block
        .append_operation(
            OperationBuilder::new(op, gen.loc())
                .add_operands(&[a, b])
                .add_results(&[a.r#type()])
                .build()?,
        )
        .result(0)?
        .into())
}

fn compare<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    predicate: melior::dialect::arith::CmpiPredicate,
    a: Value<'c, 'c>,
    b: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LowerError> {
    Ok(block
        .append_operation(
            OperationBuilder::new("arith.cmpi", gen.loc())
                .add_operands(&[a, b])
                .add_results(&[gen.i1_ty])
                .add_attributes(&[(
                    Identifier::new(gen.context, "predicate"),
                    IntegerAttribute::new(gen.i64_ty, predicate as i64).into(),
                )])
                .build()?,
        )
        .result(0)?
        .into())
}

/// A `gpu` id op, `gpu.thread_id x` for example, as an `i32`.
fn gpu_id<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    op: &str,
    axis: u64,
) -> Result<Value<'c, 'c>, LowerError> {
    let name = ["x", "y", "z"]
        .get(axis as usize)
        .ok_or_else(|| LowerError::from(format!("no axis {axis} of blocks")))?;
    let dim = melior::ir::Attribute::parse(gen.context, &format!("#gpu<dim {name}>"))
        .ok_or_else(|| LowerError::from(format!("cannot parse #gpu<dim {name}>")))?;
    let index: Value = block
        .append_operation(
            OperationBuilder::new(op, gen.loc())
                .add_attributes(&[(Identifier::new(gen.context, "dimension"), dim)])
                .add_results(&[gen.index_ty])
                .build()?,
        )
        .result(0)?
        .into();
    Ok(block
        .append_operation(
            OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[index])
                .add_results(&[gen.i32_ty])
                .build()?,
        )
        .result(0)?
        .into())
}

fn lane_id<'c>(gen: &MeliorGenerator<'c>, block: Block<'c>) -> Result<Value<'c, 'c>, LowerError> {
    gpu_id(gen, block, "gpu.thread_id", 0)
}

/// A barrier for the lanes of a block, or of a subgroup.
pub(crate) fn barrier<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    level: Level,
) -> Result<(), LowerError> {
    let mut op = OperationBuilder::new("vx.barrier", gen.loc());
    if level == Level::Subgroup {
        op = op.add_attributes(&[(
            Identifier::new(gen.context, "subgroup"),
            melior::ir::Attribute::unit(gen.context),
        )]);
    }
    block.append_operation(op.build()?);
    Ok(())
}

/// Opens a block that runs when `cond` holds. Returns it and the block after it.
fn open_guard<'c>(
    gen: &MeliorGenerator<'c>,
    block: Block<'c>,
    cond: Value<'c, 'c>,
) -> Result<(Block<'c>, Block<'c>), LowerError> {
    let region = block
        .parent_region()
        .ok_or_else(|| LowerError::from("a block with no region".to_string()))?;
    let then_b = region.append_block(melior::ir::Block::new(&[]));
    let join = region.append_block(melior::ir::Block::new(&[]));
    block.append_operation(
        OperationBuilder::new("cf.cond_br", gen.loc())
            .add_operands(&[cond])
            .add_successors(&[&*then_b, &*join])
            .add_attributes(&[(
                Identifier::new(gen.context, "operandSegmentSizes"),
                melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0, 0]).into(),
            )])
            .build()?,
    );
    Ok((then_b, join))
}

fn close_guard<'c>(
    gen: &MeliorGenerator<'c>,
    end: Option<Block<'c>>,
    join: Block<'c>,
) -> Result<(), LowerError> {
    if let Some(end) = end {
        end.append_operation(
            OperationBuilder::new("cf.br", gen.loc())
                .add_successors(&[&*join])
                .build()?,
        );
    }
    Ok(())
}

/// The kind of `for_loop` when it is an `across` loop of a region whose loops map to blocks and
/// lanes.
pub(crate) fn mapped_kind(gen: &MeliorGenerator<'_>, for_loop: &ForLoopStmt) -> Option<LevelKind> {
    gen.across_map.as_ref()?;
    let a = for_loop.across.as_ref()?;
    match a.kind? {
        LevelKind::Blocks if a.auto_schedule => None,
        kind => Some(kind),
    }
}

/// An `across` loop of a mapped region.
pub(crate) fn lower_across_for<'c>(
    for_loop: &ForLoopStmt,
    kind: LevelKind,
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let map = gen.across_map.as_mut().expect("a mapped region");
    if kind == LevelKind::Blocks {
        // The passes of this loop count.
        if map
            .tail
            .as_ref()
            .is_some_and(|t| t.counted == for_loop as *const ForLoopStmt as usize)
        {
            map.arrive_body = Some(for_loop.body.as_ptr() as usize);
        }
        let axis = map.blocks;
        map.blocks += 1;
        let outer = map.level.replace(Level::Block);
        let first = gpu_id(gen, block, "gpu.block_id", axis)?;
        let step = gpu_id(gen, block, "gpu.grid_dim", axis)?;
        let lowered = lower_strided(for_loop, (first, step), gen, block);
        let map = gen.across_map.as_mut().expect("a mapped region");
        map.level = outer;
        map.blocks -= 1;
        return lowered;
    }
    let addr = for_loop as *const ForLoopStmt as usize;
    // A loop directly in the code of a level is the outside of a nest of lane loops. Another is
    // the whole body of a lane loop.
    let root = map.level;
    if root.is_none() && map.inner != Some(addr) {
        return Err(LowerError::from(
            "a lane loop that is not in the code of a block or a subgroup, and not the \
                    whole body of a lane loop"
                .to_string(),
        ));
    }
    if let Some(sub) = for_loop.across.as_ref().and_then(|a| a.subgroup_lanes) {
        map.subgroup = Some(sub);
    }
    let width = across_width(for_loop)
        .ok_or_else(|| LowerError::from("a lane loop whose lanes are not a number".to_string()))?;
    let inner = match kind {
        LevelKind::Subgroups => map.subgroup.ok_or_else(|| {
            LowerError::from(
                "a subgroup loop on a topology whose subgroups have no size".to_string(),
            )
        })?,
        _ => inner_lanes(for_loop),
    };
    let lanes = map.lanes;
    // A nest with fewer lanes than the block leaves the other lanes out.
    let nest = width * inner;
    let (block, join) = if root == Some(Level::Block) && nest < lanes {
        let lane = lane_id(gen, block)?;
        let n = constant(gen, block, nest)?;
        let cond = compare(
            gen,
            block,
            melior::dialect::arith::CmpiPredicate::Slt,
            lane,
            n,
        )?;
        let (then_b, join) = open_guard(gen, block, cond)?;
        (then_b, Some(join))
    } else {
        (block, None)
    };
    // This lane's pass: its index among the `width` lanes of this loop.
    let mut pass = lane_id(gen, block)?;
    if inner > 1 {
        let by = constant(gen, block, inner)?;
        pass = binary(gen, block, "arith.divsi", pass, by)?;
    }
    if root != Some(Level::Block) {
        let by = constant(gen, block, width)?;
        pass = binary(gen, block, "arith.remsi", pass, by)?;
    }
    let dim = gpu_id(gen, block, "gpu.block_dim", 0)?;
    let w = constant(gen, block, width)?;
    let step = binary(gen, block, "arith.minsi", w, dim)?;

    let map = gen.across_map.as_mut().expect("a mapped region");
    let body_level = (kind == LevelKind::Subgroups).then_some(Level::Subgroup);
    let outer_level = std::mem::replace(&mut map.level, body_level);
    let outer_inner = std::mem::replace(
        &mut map.inner,
        inner_lane_loop(for_loop).map(|g| g as *const ForLoopStmt as usize),
    );
    let index = for_loop.iter.to_string();
    let rebound = crate::hir::flatten::binds_name(&for_loop.body, &index.as_str().into());
    let outer_row = (kind == LevelKind::Subgroups).then(|| {
        let row = if width == 1 {
            Some(SubgroupRow::Whole)
        } else {
            (!rebound).then(|| SubgroupRow::Index(index.clone()))
        };
        std::mem::replace(&mut map.subgroup_row, row)
    });
    let outer_element = map.lane_element.take();
    let from_zero = matches!(&*for_loop.iterable,
        Expr::Range(r) if matches!(&*r.start, Expr::Number(n) if n.value.as_ref() == "0"));
    let all_lanes = match root {
        Some(Level::Block) => true,
        Some(Level::Subgroup) => Some(width) == map.subgroup,
        None => false,
    };
    let shared = for_loop.across.as_ref().is_some_and(|a| a.share.is_some());
    if kind == LevelKind::Lanes && inner == 1 && !rebound && from_zero && all_lanes && !shared {
        map.lane_element = root.map(|level| (index, level));
    }
    let lowered = lower_strided(for_loop, (pass, step), gen, block);
    let map = gen.across_map.as_mut().expect("a mapped region");
    map.level = outer_level;
    map.inner = outer_inner;
    map.lane_element = outer_element;
    if let Some(row) = outer_row {
        map.subgroup_row = row;
    }
    let mut end = lowered?;
    if let Some(join) = join {
        close_guard(gen, end, join)?;
        end = Some(join);
    }
    if let (Some(level), Some(end)) = (root, end) {
        barrier(gen, end, level)?;
    }
    Ok(end)
}

/// The loop `for_loop` over its range, from `start + first` by `step`.
fn lower_strided<'c>(
    for_loop: &ForLoopStmt,
    (first, step): (Value<'c, 'c>, Value<'c, 'c>),
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let Expr::Range(range) = &*for_loop.iterable else {
        return Err(LowerError::from(
            "an `across` loop over something that is not a range".to_string(),
        ));
    };
    let (start, start_ty, block) = gen.generate_expr(&range.start, block)?;
    let (end, end_ty, block) = gen.generate_expr(&range.end, block)?;
    if start_ty != gen.i32_ty || end_ty != gen.i32_ty {
        return Err(LowerError::from(
            "an `across` loop over a range that is not `i32`".to_string(),
        ));
    }
    super::control_flow::lower_marked_range_loop(
        for_loop,
        None,
        (start, end, start_ty),
        Some((first, step)),
        gen,
        block,
    )
}

/// Whether lane 0 alone runs `stmt`, in the code of a level of a mapped region. Then `lane0`
/// gains the names that it sets.
pub(crate) fn runs_on_lane0(gen: &mut MeliorGenerator<'_>, stmt: &Statement) -> Option<Level> {
    let map = gen.across_map.as_mut()?;
    let level = map.level?;
    if map.on_lane0
        || matches!(stmt, Statement::Drop(_))
        || crate::hir::check::across_lanes::holds_across(stmt)
    {
        return None;
    }
    crate::hir::check::across_lanes::runs_on_lane0(std::slice::from_ref(stmt), &mut map.lane0)
        .then_some(level)
}

/// `stmt` on lane 0 of the block or of the subgroup, and then a barrier. The buffers and slots
/// that `stmt` makes move to before the guard, so the code after it can read the names that it
/// declares. A value with no slot gets one.
pub(crate) fn lower_on_lane0<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmt: &Statement,
    level: Level,
    block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    lower_stretch_on_lane0(gen, &[stmt], level, block)
}

/// The statements of a loop body. In the code of a level, a stretch of them with no `across`
/// loop runs on lane 0, with one barrier after it, when it writes memory or reads a name that
/// only lane 0 has. A drop in the stretch runs after the barrier. Then it forgets the names that
/// the statements added to `lane0`, so lane 0 runs no more code than the checker expects.
pub(crate) fn lower_level_statements<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[Statement],
    block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let lane0 = gen.across_map.as_ref().map(|m| m.lane0.clone());
    let lowered = lower_stretches(gen, stmts, block);
    if let (Some(map), Some(lane0)) = (gen.across_map.as_mut(), lane0) {
        map.lane0 = lane0;
    }
    lowered
}

fn lower_stretches<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[Statement],
    mut block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let level = match &gen.across_map {
        Some(m) if !m.on_lane0 => m.level,
        _ => None,
    };
    // The outer loop of a group lowers its inner loop. The code after it runs in the last pass
    // of each group.
    let outer_body = gen
        .across_map
        .as_ref()
        .and_then(|m| m.tail.as_ref())
        .and_then(|t| t.outer_body);
    let stmts = if outer_body == Some(stmts.as_ptr() as usize) {
        &stmts[..1]
    } else {
        stmts
    };
    let arrives = level == Some(Level::Block)
        && gen.across_map.as_ref().and_then(|m| m.arrive_body) == Some(stmts.as_ptr() as usize);
    let Some(level) = level else {
        for stmt in stmts {
            match gen.generate_statement(stmt, block)? {
                Some(b) => block = b,
                None => return Ok(None),
            }
        }
        return Ok(Some(block));
    };
    let holds = crate::hir::check::across_lanes::holds_across;
    let mut i = 0;
    while i < stmts.len() {
        if holds(&stmts[i]) {
            match gen.generate_statement(&stmts[i], block)? {
                Some(b) => block = b,
                None => return Ok(None),
            }
            i += 1;
            continue;
        }
        let start = i;
        while i < stmts.len() && !holds(&stmts[i]) {
            i += 1;
        }
        let part = &stmts[start..i];
        let map = gen.across_map.as_mut().expect("a mapped region");
        if !crate::hir::check::across_lanes::runs_on_lane0(part, &mut map.lane0) {
            // Every lane runs the stretch. The code in it is not the code of a level.
            map.level = None;
            let mut lowered = Ok(());
            for stmt in part {
                match gen.generate_statement(stmt, block) {
                    Ok(Some(b)) => block = b,
                    Ok(None) => {
                        gen.across_map.as_mut().expect("a mapped region").level = Some(level);
                        return Ok(None);
                    }
                    Err(e) => {
                        lowered = Err(e);
                        break;
                    }
                }
            }
            gen.across_map.as_mut().expect("a mapped region").level = Some(level);
            lowered?;
            continue;
        }
        let (work, drops): (Vec<&Statement>, Vec<&Statement>) =
            part.iter().partition(|s| !matches!(s, Statement::Drop(_)));
        let Some(join) = lower_stretch_on_lane0(gen, &work, level, block)? else {
            return Ok(None);
        };
        block = join;
        for stmt in drops {
            match gen.generate_statement(stmt, block)? {
                Some(b) => block = b,
                None => return Ok(None),
            }
        }
    }
    if arrives {
        return arrive(gen, block);
    }
    Ok(Some(block))
}

/// `stmts` on lane 0 of the block or of the subgroup, and then one barrier.
fn lower_stretch_on_lane0<'c>(
    gen: &mut MeliorGenerator<'c>,
    stmts: &[&Statement],
    level: Level,
    block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let mut first = lane_id(gen, block)?;
    if level == Level::Subgroup {
        let sub = gen
            .across_map
            .as_ref()
            .and_then(|m| m.subgroup)
            .ok_or_else(|| {
                LowerError::from("the code of a subgroup whose size is not known".to_string())
            })?;
        let by = constant(gen, block, sub)?;
        first = binary(gen, block, "arith.remsi", first, by)?;
    }
    let zero = constant(gen, block, 0)?;
    let cond = compare(
        gen,
        block,
        melior::dialect::arith::CmpiPredicate::Eq,
        first,
        zero,
    )?;
    let (then_b, join) = open_guard(gen, block, cond)?;
    let map = gen.across_map.as_mut().expect("a mapped region");
    map.on_lane0 = true;
    map.level = None;
    let mut end = Some(then_b);
    let mut lowered = Ok(());
    for stmt in stmts {
        let Some(at) = end else { break };
        match gen.generate_statement(stmt, at) {
            Ok(next) => end = next,
            Err(e) => {
                lowered = Err(e);
                break;
            }
        }
    }
    let map = gen.across_map.as_mut().expect("a mapped region");
    map.on_lane0 = false;
    map.level = Some(level);
    lowered?;
    hoist_allocas(then_b, block)?;
    if let Some(end) = end {
        for stmt in stmts {
            if let Statement::LetDecl(l) = stmt {
                keep_in_a_slot(gen, l.name.as_ref(), block, (then_b, end))?;
            }
        }
    }
    close_guard(gen, end, join)?;
    barrier(gen, join, level)?;
    Ok(Some(join))
}

/// Whether `v` is defined in `block`.
fn defined_in(v: Value<'_, '_>, block: Block<'_>) -> bool {
    melior::ir::operation::OperationResult::try_from(v)
        .ok()
        .and_then(|r| r.owner().block())
        .is_some_and(|b| b.to_raw().ptr == block.to_raw().ptr)
}

/// Whether `v` is an argument of `block`.
fn is_argument_of(v: Value<'_, '_>, block: Block<'_>) -> bool {
    melior::ir::block::BlockArgument::try_from(v)
        .is_ok_and(|a| a.owner().to_raw().ptr == block.to_raw().ptr)
}

/// Moves each `memref.alloca` of a fixed size in `guard` to the end of `before`, ahead of its
/// branch. A store of a buffer into a slot moves too, when both are then outside `guard`: every
/// lane gets the same buffer. The writes of values stay in `guard`.
fn hoist_allocas<'c>(guard: Block<'c>, before: Block<'c>) -> Result<(), LowerError> {
    let branch = before
        .terminator()
        .ok_or_else(|| LowerError::from("the block before a guard has no branch".to_string()))?;
    let mut ops = Vec::new();
    let mut op = guard.first_operation();
    while let Some(o) = op {
        op = o.next_in_block();
        ops.push(o);
    }
    for o in ops {
        let name = o.name();
        let name = name.as_string_ref().as_str().unwrap_or("");
        let movable = match name {
            "memref.alloca" => o.operand_count() == 0,
            "memref.store" => {
                o.operand_count() == 2
                    && o.operand(0).is_ok_and(|v| {
                        melior::ir::r#type::MemRefType::try_from(v.r#type()).is_ok()
                            && !defined_in(v, guard)
                    })
                    && o.operand(1).is_ok_and(|v| !defined_in(v, guard))
            }
            _ => false,
        };
        if movable {
            // SAFETY: the op is in `guard`, and nothing holds a reference to it while it moves.
            let mut o = unsafe { melior::ir::operation::OperationRefMut::from_raw(o.to_raw()) };
            o.move_before(branch);
        }
    }
    Ok(())
}

/// Gives `name`, a value that `end` computes, a slot that `before` makes, so the code after the
/// guard can read it. A name whose value or slot is already outside the guard keeps it.
fn keep_in_a_slot<'c>(
    gen: &mut MeliorGenerator<'c>,
    name: &str,
    before: Block<'c>,
    (guard, end): (Block<'c>, Block<'c>),
) -> Result<(), LowerError> {
    let key: crate::symbol::Symbol = name.into();
    let Some((value, _)) = gen.env.get(&key).copied() else {
        return Ok(());
    };
    if !defined_in(value, guard) && !defined_in(value, end) && !is_argument_of(value, end) {
        return Ok(());
    }
    if gen.allocs.contains(name) || melior::ir::r#type::MemRefType::try_from(value.r#type()).is_ok()
    {
        return Err(LowerError::from(format!(
            "`{name}`, which lane 0 sets, has a buffer or a slot that is not a fixed `memref.alloca`"
        )));
    }
    let branch = before
        .terminator()
        .ok_or_else(|| LowerError::from("the block before a guard has no branch".to_string()))?;
    let elem = value.r#type();
    let slot_ty = Type::parse(gen.context, &format!("memref<{elem}>"))
        .ok_or_else(|| LowerError::ParseType(format!("memref<{elem}>")))?;
    let slot: Value = before
        .insert_operation_before(
            branch,
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_results(&[slot_ty])
                .build()?,
        )
        .result(0)?
        .into();
    end.append_operation(
        OperationBuilder::new("memref.store", gen.loc())
            .add_operands(&[value, slot])
            .build()?,
    );
    gen.env.insert(key, (slot, slot_ty));
    gen.allocs.insert(name.to_string());
    Ok(())
}

/// The count of the passes of the nest of blocks of `s` that ended, made and zeroed in `block`
/// before the region, in the memory of the region. `None` for a region with no code after its
/// nest of blocks.
pub(crate) fn lower_count<'c>(
    gen: &mut MeliorGenerator<'c>,
    s: &syntax::SpawnOnExpr,
    mut block: Block<'c>,
) -> Result<(Option<Tail<'c>>, Block<'c>), LowerError> {
    let Some((space, shared)) = &s.arrivals else {
        return Ok((None, block));
    };
    if shared.is_none() {
        return Err(LowerError::from(
            "code after a nest of blocks on a topology with no shared memory".to_string(),
        ));
    }
    let plan = crate::hir::flatten::across::arrivals(&s.stmts).ok_or_else(|| {
        LowerError::from("a region with code after a nest of blocks has no nest".to_string())
    })?;
    let range = |f: &ForLoopStmt| match &*f.iterable {
        Expr::Range(r) => Ok(r.clone()),
        _ => Err(LowerError::from(
            "a loop of blocks over something that is not a range".to_string(),
        )),
    };
    let counted = range(plan.counted)?;
    // An element for each group counts the passes of it that ended.
    let (groups, group) = match plan.outer {
        Some(o) => {
            let r = range(o)?;
            let (end, _, b) = gen.generate_expr(&r.end, block)?;
            let (start, _, b) = gen.generate_expr(&r.start, b)?;
            block = b;
            let groups = binary(gen, block, "arith.subi", end, start)?;
            (groups, Some((o.iter.as_str().into(), r)))
        }
        None => (constant(gen, block, 1)?, None),
    };
    let size: Value = block
        .append_operation(
            OperationBuilder::new("arith.index_cast", gen.loc())
                .add_operands(&[groups])
                .add_results(&[Type::index(gen.context)])
                .build()?,
        )
        .result(0)?
        .into();
    let ty = Type::parse(gen.context, "memref<?xi32>")
        .ok_or_else(|| LowerError::ParseType("memref<?xi32>".to_string()))?;
    let host: Value = block
        .append_operation(
            OperationBuilder::new("memref.alloc", gen.loc())
                .add_operands(&[size])
                .add_attributes(&[(
                    Identifier::new(gen.context, "operandSegmentSizes"),
                    melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[1, 0]).into(),
                )])
                .add_results(&[ty])
                .build()?,
        )
        .result(0)?
        .into();
    super::expr::emit_zero_fill(gen, &block, host, "i32")?;
    // The count moves to the memory of the region as a tensor that a program moves there. The
    // name has a space, so no name of the program is the same.
    let name: crate::symbol::Symbol = "vx count".into();
    let outer = gen.env.insert(name.clone(), (host, ty));
    let transfer = Expr::Transfer(syntax::TransferExpr {
        expr: Box::new(Expr::Identifier(syntax::IdentifierExpr::new(
            name.clone(),
            s.span,
        ))),
        space: space.clone(),
        device: None,
        cost: None,
        lowering: None,
        is_async: false,
        shared: false,
        span: s.span,
    });
    let lowered = gen.generate_expr(&transfer, block);
    match outer {
        Some(entry) => gen.env.insert(name, entry),
        None => gen.env.remove(&name),
    };
    let (count, _, b) = lowered?;
    block = b;
    let op = melior::ir::operation::OperationResult::try_from(count)
        .map_err(|_| LowerError::from("the count is not the result of an op".to_string()))?
        .owner();
    if op.name().as_string_ref().as_str() != Ok("vx.transfer") {
        return Err(LowerError::from(
            "the count is not the result of a transfer".to_string(),
        ));
    }
    // SAFETY: the op is in `block`, and nothing holds a reference to it while it changes.
    let mut op = unsafe { melior::ir::operation::OperationRefMut::from_raw(op.to_raw()) };
    // `--emit-ptx` names the count by its length.
    op.set_attribute(
        "vx.arrivals",
        melior::ir::attribute::StringAttribute::new(gen.context, plan.length).into(),
    );
    gen.emit_drop(host, None, None, block)?;
    let tail = Tail {
        stmts: plan.after.iter().map(|s| (*s).clone()).collect(),
        skip: plan
            .after
            .iter()
            .map(|s| *s as *const Statement as usize)
            .collect(),
        counted: plan.counted as *const ForLoopStmt as usize,
        outer_body: plan.outer.map(|o| o.body.as_ptr() as usize),
        group,
        range: counted,
        count,
        flag: None,
    };
    Ok((Some(tail), block))
}

/// The flag that tells the lanes of a pass whether it was the last, in the shared memory of the
/// block, at the start of the region.
pub(crate) fn alloc_flag<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
) -> Result<(), LowerError> {
    let Some(tail) = gen.across_map.as_ref().and_then(|m| m.tail.as_ref()) else {
        return Ok(());
    };
    debug_assert!(tail.flag.is_none());
    let ty = Type::parse(gen.context, FLAG_TYPE)
        .ok_or_else(|| LowerError::ParseType(FLAG_TYPE.to_string()))?;
    let flag: Value = block
        .append_operation(
            OperationBuilder::new("memref.alloca", gen.loc())
                .add_attributes(&[
                    (
                        Identifier::new(gen.context, "operandSegmentSizes"),
                        melior::ir::attribute::DenseI32ArrayAttribute::new(gen.context, &[0, 0])
                            .into(),
                    ),
                    (
                        Identifier::new(gen.context, "alignment"),
                        IntegerAttribute::new(gen.i64_ty, 16).into(),
                    ),
                ])
                .add_results(&[ty])
                .build()?,
        )
        .result(0)?
        .into();
    if let Some(tail) = gen.across_map.as_mut().and_then(|m| m.tail.as_mut()) {
        tail.flag = Some(flag);
    }
    Ok(())
}

/// The type of the flag. Address space 3 is the shared memory of every GPU arch that launches
/// `across` loops.
const FLAG_TYPE: &str = "memref<1xi32, 3>";

/// The statements of the region that the last pass runs, and not the region itself.
pub(crate) fn runs_after_the_nest(gen: &MeliorGenerator<'_>, stmt: &Statement) -> bool {
    gen.across_map
        .as_ref()
        .and_then(|m| m.tail.as_ref())
        .is_some_and(|t| t.skip.contains(&(stmt as *const Statement as usize)))
}

/// A call of a function whose body is `body`, an `mlir!` block that the compiler writes, with its
/// inputs by name, type and value.
fn call_block<'c>(
    gen: &mut MeliorGenerator<'c>,
    block: Block<'c>,
    body: &str,
    inputs: &[(&str, &str, Value<'c, 'c>)],
    result: Option<Type<'c>>,
) -> Result<Option<Value<'c, 'c>>, LowerError> {
    let prefix = crate::codegen::flat::mlir_wrapper_prefix(
        "vx_macro_mlir_",
        gen.functions.keys().map(|n| n.as_ref()),
    );
    let name = format!("{prefix}arrive_{}", gen.mlir_block_counter);
    gen.mlir_block_counter += 1;
    let args = inputs
        .iter()
        .map(|(n, t, _)| format!("{n}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");
    let ret = result.map(|t| format!("-> {t}")).unwrap_or_default();
    let source = format!(
        "module {{\n  func.func private @{name}({args}) {ret} {{\n{}\n  }}\n}}",
        body.replace("macro.yield", "return")
    );
    let module = melior::ir::Module::parse(gen.context, &source)
        .ok_or_else(|| LowerError::from(format!("an arrival does not parse:\n{source}")))?;
    let func = module
        .body()
        .first_operation()
        .ok_or_else(|| LowerError::from("an arrival has no function".to_string()))?;
    gen.module.body().append_operation((*func).clone());
    let operands: Vec<Value> = inputs.iter().map(|(_, _, v)| *v).collect();
    let call = block.append_operation(
        OperationBuilder::new("func.call", gen.loc())
            .add_attributes(&[(
                Identifier::new(gen.context, "callee"),
                melior::ir::attribute::FlatSymbolRefAttribute::new(gen.context, &name).into(),
            )])
            .add_operands(&operands)
            .add_results(&result.into_iter().collect::<Vec<_>>())
            .build()?,
    );
    Ok(match result {
        Some(_) => Some(call.result(0)?.into()),
        None => None,
    })
}

/// The end of a pass of the nest of blocks, when code comes after the nest. Lane 0 adds the
/// pass to the count, and the block of the last pass runs the code. The lane loop or the lane-0
/// code that ends the pass ends with a barrier, so the writes of every lane come before the add.
/// Returns the block after the code.
fn arrive<'c>(
    gen: &mut MeliorGenerator<'c>,
    mut block: Block<'c>,
) -> Result<Option<Block<'c>>, LowerError> {
    let map = gen.across_map.as_ref().expect("a mapped region");
    let tail = map
        .tail
        .as_ref()
        .ok_or_else(|| LowerError::from("an arrival with no code after it".to_string()))?;
    let (group_of, range, count) = (tail.group.clone(), tail.range.clone(), tail.count);
    let flag = tail
        .flag
        .ok_or_else(|| LowerError::from("an arrival with no flag".to_string()))?;
    let group = match group_of {
        Some((name, r)) => {
            let id = Expr::Identifier(syntax::IdentifierExpr::new(name, r.span));
            let (outer, _, b) = gen.generate_expr(&id, block)?;
            let (start, _, b) = gen.generate_expr(&r.start, b)?;
            block = b;
            binary(gen, block, "arith.subi", outer, start)?
        }
        None => constant(gen, block, 0)?,
    };
    // The last count of a group: one less than the passes of the counted loop.
    let (end, _, b) = gen.generate_expr(&range.end, block)?;
    let (start, _, b) = gen.generate_expr(&range.start, b)?;
    block = b;
    let passes = binary(gen, block, "arith.subi", end, start)?;
    let one = constant(gen, block, 1)?;
    let last = binary(gen, block, "arith.subi", passes, one)?;
    let lane = lane_id(gen, block)?;
    let zero = constant(gen, block, 0)?;
    let first = compare(
        gen,
        block,
        melior::dialect::arith::CmpiPredicate::Eq,
        lane,
        zero,
    )?;
    let (then_b, join) = open_guard(gen, block, first)?;
    call_block(
        gen,
        then_b,
        crate::hir::flatten::across::arrive_block(),
        &[
            ("%count", "memref<?xi32>", count),
            ("%group", "i32", group),
            ("%last", "i32", last),
            ("%flag", FLAG_TYPE, flag),
        ],
        None,
    )?;
    close_guard(gen, Some(then_b), join)?;
    block = join;
    barrier(gen, block, Level::Block)?;
    let is_last = call_block(
        gen,
        block,
        crate::hir::flatten::across::is_last_block(),
        &[("%flag", FLAG_TYPE, flag)],
        Some(gen.i32_ty),
    )?
    .expect("a result");
    let cond = compare(
        gen,
        block,
        melior::dialect::arith::CmpiPredicate::Eq,
        is_last,
        one,
    )?;
    // Every lane of the last pass runs the code, as the body of its block.
    let (then_b, join) = open_guard(gen, block, cond)?;
    let stmts = tail_stmts(gen);
    let map = gen.across_map.as_mut().expect("a mapped region");
    let blocks = std::mem::replace(&mut map.blocks, 1);
    let outer = map.level.replace(Level::Block);
    let depth = gen.open_block();
    let lowered = lower_level_statements(gen, &stmts, then_b);
    gen.close_blocks_to(depth);
    let map = gen.across_map.as_mut().expect("a mapped region");
    map.level = outer;
    map.blocks = blocks;
    close_guard(gen, lowered?, join)?;
    Ok(Some(join))
}

fn tail_stmts(gen: &MeliorGenerator<'_>) -> Vec<Statement> {
    gen.across_map
        .as_ref()
        .and_then(|m| m.tail.as_ref())
        .map(|t| t.stmts.clone())
        .unwrap_or_default()
}
