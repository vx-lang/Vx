//===- across.rs - Vx Compiler --------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The lowering of `across` loops. A loop of blocks starts at the index of its block and
// steps over the blocks of the grid. A lane loop or a subgroup loop starts at the pass of its
// lane and steps over the lanes of its loop. So every launch shape covers each pass once, and the
// CPU copy of the kernel, with one block of one lane, runs the loops in order.
//
// The end of a lane loop is a barrier. Code between lane loops that writes memory runs on lane 0
// alone, and a barrier follows it (L5).
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::arch::LevelKind;
use crate::bytecode::{
    ACROSS_BLOCK, ACROSS_BLOCKS, ACROSS_LANE, ACROSS_STRIDE, BARRIER_SUBGROUP, LANE_REDUCE_ARGMAX,
    LANE_REDUCE_MAX, LANE_REDUCE_ORDERED, LANE_REDUCE_SUBGROUP,
};
use crate::hir::check::across_lanes;

/// The code of a blocks body or a subgroups body outside its lane loops.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum AcrossLevel {
    Block,
    Subgroup,
}

#[derive(Default)]
pub(super) struct AcrossState {
    /// The lanes of a block of the region being lowered, or 0 outside a region with `across`.
    pub lanes: u64,
    /// The level whose code is being lowered, outside its lane loops. `None` in a lane loop.
    pub level: Option<AcrossLevel>,
    /// The names that only lane 0 has, as the checker found them.
    pub lane0: HashSet<String>,
    /// In code that lane 0 alone runs.
    pub on_lane0: bool,
    /// The loops of blocks around the code.
    pub blocks: u64,
    /// The start and the step of the next loop, for `lower_for_loop`.
    pub next_loop: Option<(Val, Val)>,
    /// The lane loop that is the whole body of the lane loop being lowered.
    pub inner: Option<usize>,
    /// The lanes of a subgroup of the region's topology.
    pub subgroup: Option<u64>,
    /// The levels of the `across` loops around the code, also of those that run as plain loops.
    pub kinds: Vec<LevelKind>,
    /// The code after the nest of blocks, which the last pass to end runs.
    pub tail: Option<Tail>,
    /// The body of the loop of blocks whose passes end with an arrival, by its address.
    pub arrive_body: Option<usize>,
    /// The variable of that loop, which holds the index of its pass.
    pub pass: Option<crate::symbol::Symbol>,
}

/// The code after a nest of blocks. Each pass adds 1 to element 0 of `count` when it ends, and
/// the pass that sees `last`, one less than the number of passes, runs `stmts` in its block.
/// Element `1 + b - start` tells the lanes of pass `b` whether it is the last one.
#[derive(Clone)]
pub(super) struct Tail {
    pub stmts: Vec<Statement>,
    /// The addresses of the statements in the region, which the region does not lower itself.
    pub skip: HashSet<usize>,
    pub count: Val,
    pub start: Val,
    pub last: Val,
}

/// One pass of a nest of blocks has ended, on lane 0: a release fence, an atomic add to the
/// count, and an acquire fence. Each fence is for the device, so the writes of every pass are
/// visible to the last one. The pass's own element of the count says whether it was the last.
const ARRIVE: &str = r#"llvm.fence syncscope("device") release
%vx_zero = arith.constant 0 : index
%vx_one = arith.constant 1 : i32
%vx_old = memref.atomic_rmw addi %vx_one, %count[%vx_zero] : (i32, memref<?xi32>) -> i32
%vx_is_last = arith.cmpi eq, %vx_old, %last : i32
llvm.fence syncscope("device") acquire
%vx_last = arith.extui %vx_is_last : i1 to i32
%vx_slot = arith.index_cast %slot : i32 to index
memref.store %vx_last, %count[%vx_slot] : memref<?xi32>
macro.yield"#;

/// Whether this pass was the last to end, for every lane of its block, after the barrier.
const IS_LAST: &str = r#"%vx_slot = arith.index_cast %slot : i32 to index
%vx_last = memref.load %count[%vx_slot] : memref<?xi32>
macro.yield %vx_last : i32"#;

/// The lanes of a block of a region with `across` loops: the most that a lane loop or a subgroup
/// loop asks for, and 1 with none. `None` for a region with no loop of blocks.
pub(crate) fn across_block_lanes(stmts: &[Statement]) -> Option<u64> {
    blocks_loop(stmts)?;
    let mut lanes = 1;
    across_lanes::visit_stmts(stmts, &mut |s| {
        if let Statement::ForLoop(f) = s {
            let width = across_width(f).unwrap_or(1);
            match f.across.as_ref().and_then(|a| a.kind) {
                Some(LevelKind::Lanes) => lanes = lanes.max(width * inner_lanes(f)),
                Some(LevelKind::Subgroups) => {
                    let sub = f.across.as_ref().and_then(|a| a.subgroup_lanes);
                    lanes = lanes.max(width * sub.unwrap_or(1));
                }
                _ => {}
            }
        }
    });
    Some(lanes)
}

/// The loop of blocks in `stmts`, when the checker found one that the GPU launches. A loop that
/// `-fauto-schedule-levels` made a plain loop is not one.
pub(super) fn blocks_loop(stmts: &[Statement]) -> Option<&crate::syntax::ForLoopStmt> {
    stmts.iter().find_map(|s| match s {
        Statement::ForLoop(f)
            if f.across
                .as_ref()
                .is_some_and(|a| a.kind == Some(LevelKind::Blocks) && !a.auto_schedule) =>
        {
            Some(f)
        }
        _ => None,
    })
}

/// The lanes that a lane loop or a subgroup loop shares its passes among: `n` of
/// `across lanes(n)`, or else its trip. The checker made sure that it is a number.
fn across_width(f: &crate::syntax::ForLoopStmt) -> Option<u64> {
    let number = |e: &Expr| match e {
        Expr::Number(n) => n.value.parse::<u64>().ok(),
        _ => None,
    };
    if let Some(n) = f.across.as_ref()?.share.as_deref() {
        return number(n);
    }
    let Expr::Range(r) = &*f.iterable else {
        return None;
    };
    number(&r.end)?.checked_sub(number(&r.start)?)
}

/// The lane loop that is the whole body of the lane loop `f`, if there is one.
fn inner_lane_loop(f: &crate::syntax::ForLoopStmt) -> Option<&crate::syntax::ForLoopStmt> {
    let mut body = f.body.iter().filter(|s| !matches!(s, Statement::Drop(_)));
    match (body.next(), body.next()) {
        (Some(Statement::ForLoop(g)), None)
            if g.across.as_ref().and_then(|a| a.kind) == Some(LevelKind::Lanes) =>
        {
            Some(g)
        }
        _ => None,
    }
}

/// The lanes of the lane loops inside the lane loop `f`, each the whole body of the one around it.
fn inner_lanes(f: &crate::syntax::ForLoopStmt) -> u64 {
    match inner_lane_loop(f) {
        Some(g) => across_width(g).unwrap_or(1) * inner_lanes(g),
        None => 1,
    }
}

impl Lowerer<'_> {
    fn across_id(&mut self, kind: u64, arg: u64) -> Val {
        self.emit_value(
            Opcode::AcrossId,
            Register(0),
            Register(0),
            ElementType::I32,
            kind | (arg << 2),
        )
    }

    fn const_i32(&mut self, v: u64) -> Val {
        self.emit_value(Opcode::Const, Register(0), Register(0), ElementType::I32, v)
    }

    /// Opens a block that runs when `cond` holds, and returns the block after it.
    fn open_guard(&mut self, cond: Val) -> u32 {
        let then_b = self.new_block();
        let join = self.new_block();
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(then_b, join),
        );
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), then_b as u64);
        join
    }

    fn close_guard(&mut self, join: u32) {
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), join as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), join as u64);
    }

    fn barrier(&mut self, level: AcrossLevel) {
        let imm = match level {
            AcrossLevel::Block => 0,
            AcrossLevel::Subgroup => BARRIER_SUBGROUP,
        };
        self.emit_effect(Opcode::Barrier, Register(0), Register(0), imm);
    }

    /// The statements of a blocks body or a subgroups body (L5). A statement that holds an
    /// `across` loop lowers as it is, and its bodies have stretches of their own. A stretch that
    /// lane 0 alone runs is in a block that only lane 0 enters, and a barrier follows it.
    pub(super) fn lower_stretches(
        &mut self,
        stmts: &[Statement],
        level: AcrossLevel,
    ) -> Lowered<()> {
        let mut i = 0;
        while i < stmts.len() {
            if across_lanes::holds_across(&stmts[i]) {
                let lane0 = self.across.lane0.clone();
                self.lower_stmt(&stmts[i])?;
                self.across.lane0 = lane0;
                i += 1;
                continue;
            }
            let start = i;
            while i < stmts.len() && !across_lanes::holds_across(&stmts[i]) {
                i += 1;
            }
            let part = &stmts[start..i];
            let on_lane0 = across_lanes::runs_on_lane0(part, &mut self.across.lane0);
            let outer = self.across.level.take();
            let join = if on_lane0 {
                // The first lane of the block, or of the subgroup.
                let lane = self.across_id(ACROSS_LANE, 0);
                let first = match level {
                    AcrossLevel::Block => lane,
                    AcrossLevel::Subgroup => {
                        let sub = self.subgroup_lanes()?;
                        let width = self.const_i32(sub);
                        self.emit_value(Opcode::Rem, lane.reg, width.reg, ElementType::I32, 0)
                    }
                };
                let zero = self.const_i32(0);
                let cond = self.emit_value(
                    Opcode::Cmp,
                    first.reg,
                    zero.reg,
                    ElementType::Bool,
                    rel_code(&RelationalOp::Eq),
                );
                self.across.on_lane0 = true;
                Some(self.open_guard(cond))
            } else {
                None
            };
            let lowered = part.iter().try_for_each(|s| self.lower_stmt(s));
            self.across.on_lane0 = false;
            self.across.level = outer;
            lowered?;
            if let Some(join) = join {
                self.close_guard(join);
                self.barrier(level);
            }
        }
        if level == AcrossLevel::Block && self.across.arrive_body == Some(stmts.as_ptr() as usize) {
            let name = self.across.pass.clone().ok_or(crate::emitter_gap!())?;
            let id = crate::syntax::IdentifierExpr::new(name, crate::syntax::Span::default());
            let pass = self.lower_expr(&Expr::Identifier(id))?;
            self.arrive(pass)?;
        }
        Ok(())
    }

    /// The end of a pass `pass` of the nest of blocks, when code comes after the nest. Lane 0 adds
    /// the pass to the count, and the block of the last pass runs the code. The lane loop or the
    /// lane-0 code that ends the pass ends with a barrier, so the writes of every lane come before
    /// the add. Code that every lane runs writes no memory.
    fn arrive(&mut self, pass: Val) -> Lowered<()> {
        let tail = self.across.tail.clone().ok_or(crate::emitter_gap!())?;
        let offset = self.emit_value(Opcode::Sub, pass.reg, tail.start.reg, ElementType::I32, 0);
        let one = self.const_i32(1);
        let slot = self.emit_value(Opcode::Add, offset.reg, one.reg, ElementType::I32, 0);
        let lane = self.across_id(ACROSS_LANE, 0);
        let zero = self.const_i32(0);
        let first = self.emit_value(
            Opcode::Cmp,
            lane.reg,
            zero.reg,
            ElementType::Bool,
            rel_code(&RelationalOp::Eq),
        );
        let lane0 = self.open_guard(first);
        self.inline(
            ARRIVE,
            &[
                ("%count", "memref<?xi32>", &tail.count),
                ("%last", "i32", &tail.last),
                ("%slot", "i32", &slot),
            ],
            true,
        );
        self.close_guard(lane0);
        self.barrier(AcrossLevel::Block);
        let is_last = self.inline(
            IS_LAST,
            &[
                ("%count", "memref<?xi32>", &tail.count),
                ("%slot", "i32", &slot),
            ],
            false,
        );
        let cond = self.emit_value(
            Opcode::Cmp,
            is_last.reg,
            one.reg,
            ElementType::Bool,
            rel_code(&RelationalOp::Eq),
        );
        // Every lane of the last pass runs the code, as the body of its block.
        let last = self.open_guard(cond);
        let blocks = std::mem::replace(&mut self.across.blocks, 1);
        let outer = self.across.level.replace(AcrossLevel::Block);
        let lowered = self.lower_block(&tail.stmts);
        self.across.level = outer;
        self.across.blocks = blocks;
        lowered?;
        self.close_guard(last);
        Ok(())
    }

    /// An `mlir!` block that the compiler writes, with its inputs by name, type and value.
    fn inline(&mut self, body: &str, inputs: &[(&str, &str, &Val)], void: bool) -> Val {
        for (_, _, v) in inputs {
            self.emit_effect(Opcode::Arg, v.reg, Register(0), 0);
        }
        let imm = self.inline_blocks.len() as u64;
        self.inline_blocks.push(InlineBlock {
            inputs: inputs
                .iter()
                .map(|(n, t, _)| (n.to_string(), t.to_string()))
                .collect(),
            body: body.to_string(),
            void,
            line: 0,
            column: 0,
        });
        self.emit_typed(
            Opcode::InlineMlir,
            Register(0),
            Register(0),
            LoweredTy::Scalar(ElementType::I32),
            imm,
        )
    }

    /// `lanes::sum(&t, n)` and the other lane reductions. The checker allows a call only in the
    /// code of a block or a subgroup, so the innermost level says which.
    pub(super) fn lower_lane_reduce(
        &mut self,
        fc: &crate::syntax::expr::FunctionCallExpr,
        op: crate::hir::check::lanes::LaneOp,
        ordered: bool,
    ) -> Lowered<Val> {
        let argmax = op == crate::hir::check::lanes::LaneOp::ArgMax;
        let (n, tensors) = fc.args.split_last().ok_or(crate::emitter_gap!())?;
        if tensors.len() != if argmax { 2 } else { 1 } {
            return Err(crate::emitter_gap!());
        }
        let mut regs = Vec::new();
        for t in tensors {
            let t = self.lower_expr(t)?;
            if !matches!(&t.ty, LoweredTy::Tensor { shape, .. } if shape.len() == 1) {
                return Err(Decline::TypeNotModelled {
                    what: "a lane reduction of something that is not a rank-1 tensor",
                });
            }
            regs.push(t.reg);
        }
        let n = self.lower_expr(n)?;
        let mut imm = 0;
        match op {
            crate::hir::check::lanes::LaneOp::Max => imm |= LANE_REDUCE_MAX,
            crate::hir::check::lanes::LaneOp::ArgMax => imm |= LANE_REDUCE_ARGMAX,
            crate::hir::check::lanes::LaneOp::Sum => {}
        }
        if ordered {
            imm |= LANE_REDUCE_ORDERED;
        }
        if self.across.kinds.last() == Some(&LevelKind::Subgroups) {
            imm |= LANE_REDUCE_SUBGROUP;
        }
        if argmax {
            self.emit_effect(Opcode::Arg, n.reg, Register(0), 0);
            return Ok(self.emit_value(
                Opcode::LaneReduce,
                regs[0],
                regs[1],
                ElementType::I32,
                imm,
            ));
        }
        Ok(self.emit_value(Opcode::LaneReduce, regs[0], n.reg, ElementType::F32, imm))
    }

    /// The lanes of a subgroup of the region's topology.
    fn subgroup_lanes(&self) -> Lowered<u64> {
        self.across.subgroup.ok_or(Decline::Unsupported {
            what: "a subgroup loop on a topology whose subgroups have no size",
        })
    }

    /// An `across` loop of the kind `kind`.
    pub(super) fn lower_across_for(
        &mut self,
        f: &crate::syntax::ForLoopStmt,
        kind: LevelKind,
    ) -> Lowered<()> {
        if kind == LevelKind::Blocks {
            // The outer loop of blocks is the nest whose passes count.
            if self.across.blocks == 0 && self.across.tail.is_some() {
                self.across.arrive_body = Some(f.body.as_ptr() as usize);
                self.across.pass = Some(f.iter.as_str().into());
            }
            let axis = self.across.blocks;
            let first = self.across_id(ACROSS_BLOCK, axis);
            let step = self.across_id(ACROSS_BLOCKS, axis);
            self.across.next_loop = Some((first, step));
            self.across.blocks += 1;
            let outer = self.across.level.replace(AcrossLevel::Block);
            let lowered = self.lower_plain_for(f);
            self.across.level = outer;
            self.across.blocks -= 1;
            return lowered;
        }
        let addr = f as *const crate::syntax::ForLoopStmt as usize;
        // A loop directly in code of a level is the outside of a nest of lane loops. Another is
        // the whole body of a lane loop.
        let root = self.across.level;
        // The checker refuses a lane loop in a lane loop that is not its whole body.
        if root.is_none() && self.across.inner != Some(addr) {
            return Err(crate::emitter_gap!());
        }
        if let Some(sub) = f.across.as_ref().and_then(|a| a.subgroup_lanes) {
            self.across.subgroup = Some(sub);
        }
        let width = across_width(f).ok_or(crate::emitter_gap!())?;
        let inner = match kind {
            LevelKind::Subgroups => self.subgroup_lanes()?,
            _ => inner_lanes(f),
        };
        // A nest with fewer lanes than the block leaves the other lanes out.
        let nest = width * inner;
        let mask = root == Some(AcrossLevel::Block) && nest < self.across.lanes;
        let join = if mask {
            let lane = self.across_id(ACROSS_LANE, 0);
            let lanes = self.const_i32(nest);
            let cond = self.emit_value(
                Opcode::Cmp,
                lane.reg,
                lanes.reg,
                ElementType::Bool,
                rel_code(&RelationalOp::Lt),
            );
            Some(self.open_guard(cond))
        } else {
            None
        };
        // This lane's pass: its index among the `width` lanes of this loop.
        let mut pass = self.across_id(ACROSS_LANE, 0);
        if inner > 1 {
            let by = self.const_i32(inner);
            pass = self.emit_value(Opcode::Div, pass.reg, by.reg, ElementType::I32, 0);
        }
        if root != Some(AcrossLevel::Block) {
            let by = self.const_i32(width);
            pass = self.emit_value(Opcode::Rem, pass.reg, by.reg, ElementType::I32, 0);
        }
        let step = self.across_id(ACROSS_STRIDE, width);
        self.across.next_loop = Some((pass, step));
        let body_level = match kind {
            LevelKind::Subgroups => Some(AcrossLevel::Subgroup),
            _ => None,
        };
        let outer_level = std::mem::replace(&mut self.across.level, body_level);
        let outer_inner = std::mem::replace(
            &mut self.across.inner,
            inner_lane_loop(f).map(|g| g as *const crate::syntax::ForLoopStmt as usize),
        );
        let lowered = self.lower_plain_for(f);
        self.across.level = outer_level;
        self.across.inner = outer_inner;
        lowered?;
        if let Some(join) = join {
            self.close_guard(join);
        }
        if let Some(level) = root {
            self.barrier(level);
        }
        Ok(())
    }
}
