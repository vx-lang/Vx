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
    LaneCallImm, ACROSS_BLOCK, ACROSS_BLOCKS, ACROSS_LANE, ACROSS_STRIDE, BARRIER_SUBGROUP,
    LANE_OWN_LANES_SHIFT, LANE_OWN_SUBGROUP, LANE_OWN_WHOLE, LANE_ROW_SHIFT,
    TENSOR_STORE_BLOCK_LANE_ELEMENT, TENSOR_STORE_SUBGROUP_LANE_ELEMENT,
};
use crate::hir::check::across_lanes::{self, lane_own_access, LaneOwn};
use crate::hir::check::lanes::{LaneArg, LaneRoutine, LaneSite};

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
    /// The names that only lane 0 has. Each block forgets the names that it adds, so lane 0 runs
    /// no more code than the checker expects.
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
    /// The row of a tile that the code of a subgroup alone writes. `None` when the body of the
    /// subgroup loop binds the loop's index again.
    pub subgroup_row: Option<SubgroupRow>,
    /// The subgroup loop gives each subgroup one pass at most, so its index is the number of the
    /// subgroup, the same in each such loop.
    pub subgroup_once: bool,
    /// In a lane loop from 0 over all the lanes of its level: the index `t` of the loop, and the
    /// level. Only lane `t` writes `tile[t]` in the code of a block, or element `t` of the row of
    /// the subgroup.
    pub lane_element: Option<(Symbol, AcrossLevel)>,
    /// With `lane_element`: the lanes of the lane loop, and whether the subgroup loop around it
    /// gives each subgroup one pass at most.
    pub lane_loop: Option<(u64, bool)>,
}

/// The row of a tile that the code of a subgroup alone writes.
#[derive(Clone)]
pub(super) enum SubgroupRow {
    /// `tile[w]`, with `w` the index of the subgroup loop.
    Index(Symbol),
    /// The whole tile, when the subgroup loop has one subgroup.
    Whole,
}

/// The code after the passes of a nest of blocks, or of a group: each pass of the outer loop is
/// a group of the passes of the inner one. Each pass adds 1 to the element of its group in
/// `count` when it ends, and the pass that sees `last`, one less than the passes of a group, runs
/// `stmts` in its block and sets the element back to 0. `flag`, in shared memory, tells the lanes
/// of the pass whether it was the last one of its group.
#[derive(Clone)]
pub(super) struct Tail {
    pub stmts: Vec<Statement>,
    /// The addresses of the statements in the region, which the region does not lower itself.
    pub skip: HashSet<usize>,
    /// The loop whose passes count, by its address.
    pub counted: usize,
    /// The body of the outer loop of a group, which lowers only its inner loop, by its address.
    pub outer_body: Option<usize>,
    /// The variable of the outer loop of a group, and its range.
    pub group: Option<(crate::symbol::Symbol, crate::syntax::RangeExpr)>,
    /// The range of the loop whose passes count. The kernel computes the last count from it, so
    /// it reads no value of the host but the parameters.
    pub range: crate::syntax::RangeExpr,
    pub count: Val,
    /// The shared memory of a block, by its dispatch id, and the flag in it.
    pub shared: u64,
    pub flag: Option<Val>,
}

/// One pass of a nest of blocks has ended, on lane 0: an atomic add to the count of its group,
/// with release and acquire order for the device, so the writes of every pass are visible to the
/// last one. `memref.atomic_rmw` has no scope, so it would order for the whole system, which
/// costs NVIDIA two more fences. The last pass of a group sets the count back to 0, so the
/// next launch can use it again, and the flag says whether this pass was the last. The flag is
/// in address space 3, the shared memory of every GPU arch that launches `across` loops.
const ARRIVE: &str = r#"%vx_base = memref.extract_aligned_pointer_as_index %count : memref<?xi32> -> index
%vx_buf, %vx_off, %vx_size, %vx_stride = memref.extract_strided_metadata %count : memref<?xi32> -> memref<i32>, index, index, index
%vx_group = arith.index_cast %group : i32 to index
%vx_index = arith.addi %vx_off, %vx_group : index
%vx_base_int = arith.index_cast %vx_base : index to i64
%vx_ptr = llvm.inttoptr %vx_base_int : i64 to !llvm.ptr
%vx_index_int = arith.index_cast %vx_index : index to i64
%vx_at_count = llvm.getelementptr %vx_ptr[%vx_index_int] : (!llvm.ptr, i64) -> !llvm.ptr, i32
%vx_one = arith.constant 1 : i32
%vx_old = llvm.atomicrmw add %vx_at_count, %vx_one syncscope("device") acq_rel : !llvm.ptr, i32
%vx_is_last = arith.cmpi eq, %vx_old, %last : i32
scf.if %vx_is_last {
  %vx_none = arith.constant 0 : i32
  memref.store %vx_none, %count[%vx_group] : memref<?xi32>
}
%vx_last = arith.extui %vx_is_last : i1 to i32
%vx_at = arith.constant 0 : index
memref.store %vx_last, %flag[%vx_at] : memref<1xi32, 3>
macro.yield"#;

/// Whether this pass was the last of its group, for every lane of its block, after the barrier.
const IS_LAST: &str = r#"%vx_at = arith.constant 0 : index
%vx_last = memref.load %flag[%vx_at] : memref<1xi32, 3>
macro.yield %vx_last : i32"#;

/// Where the code after a nest of blocks is, in a region with such code.
pub(crate) struct Arrivals<'a> {
    /// The loop whose passes count.
    pub counted: &'a crate::syntax::ForLoopStmt,
    /// The statements that the last pass runs.
    pub after: Vec<&'a Statement>,
    /// The outer loop of a group, when the code is the code of a group.
    pub outer: Option<&'a crate::syntax::ForLoopStmt>,
    /// The length of the count, which `--emit-ptx` uses in its name: "one" for a whole nest,
    /// "axis0" when the outer loop of a group goes over all of the first axis, and "" otherwise.
    pub length: &'static str,
}

/// The loops and the statements of the code after the nest of blocks in `stmts`, or of the code
/// of a group: each pass of the outer loop is a group of the passes of the inner one.
pub(crate) fn arrivals(stmts: &[Statement]) -> Option<Arrivals<'_>> {
    let at = stmts
        .iter()
        .position(|s| matches!(s, Statement::ForLoop(f) if f.across.is_some()))?;
    let Statement::ForLoop(nest) = &stmts[at] else {
        return None;
    };
    let grouped = nest.body.len() > 1 && crate::hir::check::across::same_level(&nest.body[0], nest);
    if grouped {
        let Statement::ForLoop(inner) = &nest.body[0] else {
            return None;
        };
        let whole_axis = nest.across.as_ref().is_some_and(|a| a.share.is_none())
            && matches!(&*nest.iterable, Expr::Range(r)
                if matches!(&*r.start, Expr::Number(n) if n.value.as_ref() == "0"));
        return Some(Arrivals {
            counted: inner,
            after: nest.body[1..].iter().collect(),
            outer: Some(nest),
            length: if whole_axis { "axis0" } else { "" },
        });
    }
    let rest = &stmts[at + 1..];
    // The code after the nest drops the names it declares. A drop of a name from before the nest
    // stays in the body that every block runs.
    let declared: Vec<_> = rest
        .iter()
        .filter_map(|s| match s {
            Statement::LetDecl(l) => Some(&l.name),
            _ => None,
        })
        .collect();
    let after = rest
        .iter()
        .filter(|s| !matches!(s, Statement::Drop(d) if !declared.contains(&&d.name)))
        .filter(|s| !crate::hir::check::across::same_level(s, nest))
        .collect();
    Some(Arrivals {
        counted: nest,
        after,
        outer: None,
        length: "one",
    })
}

/// The ops that end one pass of a nest of blocks, on lane 0, as an `mlir!` block.
pub(crate) fn arrive_block() -> &'static str {
    ARRIVE
}

/// The ops that read whether this pass was the last, as an `mlir!` block.
pub(crate) fn is_last_block() -> &'static str {
    IS_LAST
}

/// The lanes of a block of a region with `across` loops: the most that a lane loop, a subgroup
/// loop or an `mlir!` block with `threads` asks for, and 1 with none. `None` for a region with no loop of blocks.
pub(crate) fn across_block_lanes(stmts: &[Statement]) -> Option<u64> {
    blocks_loop(stmts)?;
    let mut lanes = 1;
    across_lanes::visit_stmts(stmts, &mut |s| {
        if let Some(n) = across_lanes::threadgroup_block(s).and_then(|m| m.threads) {
            lanes = lanes.max(u64::from(n));
        }
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
pub(crate) fn blocks_loop(stmts: &[Statement]) -> Option<&crate::syntax::ForLoopStmt> {
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
pub(crate) fn across_width(f: &crate::syntax::ForLoopStmt) -> Option<u64> {
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

/// Whether a lane loop or a subgroup loop from 0 gives each of its lanes one pass at most: it has
/// at most `width` passes, and its `width` groups of `inner` lanes fit in the `lanes` of the block.
pub(crate) fn one_pass_each(
    f: &crate::syntax::ForLoopStmt,
    width: u64,
    inner: u64,
    lanes: u64,
) -> bool {
    let trip = match &*f.iterable {
        Expr::Range(r) => match &*r.end {
            Expr::Number(n) => n.value.parse::<u64>().ok(),
            _ => None,
        },
        _ => None,
    };
    trip.is_some_and(|t| t <= width) && width.checked_mul(inner).is_some_and(|n| n <= lanes)
}

/// The lane loop that is the whole body of the lane loop `f`, if there is one.
pub(crate) fn inner_lane_loop(
    f: &crate::syntax::ForLoopStmt,
) -> Option<&crate::syntax::ForLoopStmt> {
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
pub(crate) fn inner_lanes(f: &crate::syntax::ForLoopStmt) -> u64 {
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

    /// The `i32` that a launch takes for the passes of a loop of blocks, from a `size` of the
    /// loop's type. A launch has at most `i32::MAX` blocks on an axis, so a 64-bit size that is
    /// more stops the program, with the line of the loop in the message.
    pub(super) fn launch_count(&mut self, size: Val, line: usize) -> Lowered<Val> {
        let elem = match &size.ty {
            LoweredTy::Scalar(ElementType::I32) => return Ok(size),
            LoweredTy::Scalar(e @ (ElementType::I64 | ElementType::U64 | ElementType::USize)) => {
                e.clone()
            }
            _ => {
                return Err(Decline::Unsupported {
                    what: "an `across` loop of blocks whose size is not `i32` or a 64-bit integer",
                })
            }
        };
        let compare = |me: &mut Self, bound: &str, op: RelationalOp| -> Lowered<Val> {
            let bound = me.lower_number_literal(bound, &elem)?;
            Ok(me.emit_value(
                Opcode::Cmp,
                size.reg,
                bound.reg,
                ElementType::Bool,
                rel_code(&op),
            ))
        };
        let mut fits = compare(self, &i32::MAX.to_string(), RelationalOp::Le)?;
        if elem == ElementType::I64 {
            let above = compare(self, &i32::MIN.to_string(), RelationalOp::Ge)?;
            fits = self.emit_value(Opcode::BitAnd, fits.reg, above.reg, ElementType::Bool, 0);
        }
        let message = self.strings.len() as u64;
        self.strings.push(format!(
            "the loop of blocks at line {line} has more passes than a launch can have, which is {}",
            i32::MAX
        ));
        self.emit_effect(Opcode::Abort, fits.reg, Register(0), message);
        Ok(self.emit_value(Opcode::Cast, size.reg, Register(0), ElementType::I32, 0))
    }

    /// `a - b` for two bounds or indices of a loop of blocks, as an `i32`. The difference is at
    /// most the passes of the loop, which `launch_count` checks. A 64-bit one is taken in `i64`:
    /// a `usize` subtraction would check in each pass that it does not go below 0.
    fn difference_as_i32(&mut self, a: Val, b: Val) -> Lowered<Val> {
        let LoweredTy::Scalar(elem) = a.ty.clone() else {
            return Err(crate::emitter_gap!());
        };
        if elem == ElementType::I32 {
            return Ok(self.emit_value(Opcode::Sub, a.reg, b.reg, elem, 0));
        }
        let a = self.emit_value(Opcode::Cast, a.reg, Register(0), ElementType::I64, 0);
        let b = self.emit_value(Opcode::Cast, b.reg, Register(0), ElementType::I64, 0);
        let d = self.emit_value(Opcode::Sub, a.reg, b.reg, ElementType::I64, 0);
        Ok(self.emit_value(Opcode::Cast, d.reg, Register(0), ElementType::I32, 0))
    }

    /// The `i32` index of a block or a lane, as a value of the integer type `elem` of its loop. A
    /// `usize` goes through `u32`: the index is never below 0, and a cast from `i32` to `usize`
    /// would check that in each pass.
    pub(super) fn widen_index(&mut self, index: Val, elem: &ElementType) -> Val {
        if *elem == ElementType::I32 {
            return index;
        }
        let index = if elem.is_signed_int() {
            index
        } else {
            self.emit_value(Opcode::Cast, index.reg, Register(0), ElementType::U32, 0)
        };
        self.emit_value(Opcode::Cast, index.reg, Register(0), elem.clone(), 0)
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
        // The outer loop of a group lowers its inner loop. The code after it runs in the last
        // pass of each group.
        let outer = self.across.tail.as_ref().and_then(|t| t.outer_body);
        let stmts = if outer == Some(stmts.as_ptr() as usize) {
            &stmts[..1]
        } else {
            stmts
        };
        let mut i = 0;
        while i < stmts.len() {
            if across_lanes::holds_across(&stmts[i]) {
                self.lower_stmt(&stmts[i])?;
                // The code after an `mlir!` block with `threads` may read what any thread of it
                // wrote.
                if across_lanes::threadgroup_block(&stmts[i]).is_some() {
                    self.barrier(level);
                }
                i += 1;
                continue;
            }
            let start = i;
            i = across_lanes::stretch_end(stmts, start, &self.across.lane0);
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
            self.arrive()?;
        }
        Ok(())
    }

    /// The end of a pass of the nest of blocks, when code comes after the nest. Lane 0 adds
    /// the pass to the count, and the block of the last pass runs the code. The lane loop or the
    /// lane-0 code that ends the pass ends with a barrier, so the writes of every lane come before
    /// the add. Code that every lane runs writes no memory.
    fn arrive(&mut self) -> Lowered<()> {
        let tail = self.across.tail.clone().ok_or(crate::emitter_gap!())?;
        let group = match &tail.group {
            Some((name, range)) => {
                let id = crate::syntax::IdentifierExpr::new(
                    name.clone(),
                    crate::syntax::Span::default(),
                );
                let outer = self.lower_expr(&Expr::Identifier(id))?;
                let start = self.lower_expr(&range.start)?;
                self.difference_as_i32(outer, start)?
            }
            None => self.const_i32(0),
        };
        // The last count of a group: one less than the passes of the counted loop.
        let end = self.lower_expr(&tail.range.end)?;
        let start = self.lower_expr(&tail.range.start)?;
        let passes = self.difference_as_i32(end, start)?;
        let one = self.const_i32(1);
        let last = self.emit_value(Opcode::Sub, passes.reg, one.reg, ElementType::I32, 0);
        let flag = tail.flag.clone().ok_or(crate::emitter_gap!())?;
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
                ("%group", "i32", &group),
                ("%last", "i32", &last),
                ("%flag", "memref<1xi32, 3>", &flag),
            ],
            true,
        );
        self.close_guard(lane0);
        self.barrier(AcrossLevel::Block);
        let is_last = self.inline(IS_LAST, &[("%flag", "memref<1xi32, 3>", &flag)], false);
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
            threads: false,
        });
        self.emit_typed(
            Opcode::InlineMlir,
            Register(0),
            Register(0),
            LoweredTy::Scalar(ElementType::I32),
            imm,
        )
    }

    /// A call to a lane routine, or a lane matmul: the operands go in `Arg`s, and then a
    /// `LaneCall`. The checker allows a call only in the code of a block or a subgroup, so the
    /// innermost level says which. The value is `None` for a routine with no value.
    pub(super) fn lower_lane_call(
        &mut self,
        routine: &'static LaneRoutine,
        args: &[&Expr],
    ) -> Lowered<Option<Val>> {
        if args.len() != routine.args.len() {
            return Err(crate::emitter_gap!());
        }
        let mut operands = Vec::new();
        let mut place = None;
        let mut tile_ranks = Vec::new();
        for (&arg, &kind) in args.iter().zip(routine.args) {
            if kind == LaneArg::Place {
                let Expr::Array(at) = arg else {
                    return Err(crate::emitter_gap!());
                };
                for e in &at.elements {
                    operands.push(self.lower_expr(e)?.reg);
                }
                place = Some(at.elements.len());
                continue;
            }
            let v = self.lower_expr(arg)?;
            if kind.is_tensor() {
                let LoweredTy::Tensor { elem, shape } = &v.ty else {
                    return Err(Decline::TypeNotModelled {
                        what: "a lane routine of something that is not a tensor",
                    });
                };
                match kind {
                    LaneArg::Vector if shape.len() != 1 => {
                        return Err(Decline::TypeNotModelled {
                            what: "a lane routine of something that is not a rank-1 tensor",
                        })
                    }
                    LaneArg::Matrix(elems) if shape.len() != 2 || !elems.contains(elem) => {
                        return Err(Decline::TypeNotModelled {
                            what: "a lane matmul of something that is not a matrix of floats",
                        })
                    }
                    LaneArg::Tile => tile_ranks.push(shape.len()),
                    _ => {}
                }
            }
            operands.push(v.reg);
        }
        if tile_ranks.iter().any(|r| Some(*r) != place) {
            return Err(Decline::TypeNotModelled {
                what: "a lane copy whose tensors do not have the rank of its place",
            });
        }
        let subgroup = self.across.kinds.last() == Some(&LevelKind::Subgroups);
        let lanes = match (routine.lanes, subgroup) {
            (false, _) => 0,
            (true, true) => self.subgroup_lanes()?,
            (true, false) => self.across.lanes,
        };
        let own_row = if routine.own_row && subgroup {
            routine.own_row_of(args, |e| self.subgroup_row_number(e))
        } else {
            None
        };
        let site = LaneSite {
            serial: false,
            subgroup,
            subgroup_row: routine
                .row_tile(args)
                .is_some_and(|t| self.row_of_subgroup(t)),
            own_row,
            lane_own: own_row.is_some()
                && (self.across.subgroup_once
                    || matches!(self.across.subgroup_row, Some(SubgroupRow::Whole))),
            lanes,
        };
        for r in &operands {
            self.emit_effect(Opcode::Arg, *r, Register(0), 0);
        }
        let imm = LaneCallImm {
            routine: routine.index(),
            operands: operands.len(),
            site,
        }
        .encode();
        Ok(match routine.result {
            Some(value) => Some(self.emit_value(
                Opcode::LaneCall,
                Register(0),
                Register(0),
                value.element(),
                imm,
            )),
            None => {
                self.emit_effect(Opcode::LaneCall, Register(0), Register(0), imm);
                None
            }
        })
    }

    /// `acc = acc + (a as f32) @ (b as f32)` in the code of a subgroup or a block: an effect on
    /// `acc`, which every lane of the subgroup or the block computes. A barrier after it lets each
    /// lane read the elements that the other lanes wrote.
    pub(super) fn lower_lane_matmul(
        &mut self,
        m: &crate::hir::check::lanes::LaneMatmul,
    ) -> Lowered<()> {
        assert!(
            matches!(
                self.across.kinds.last(),
                Some(LevelKind::Subgroups | LevelKind::Blocks)
            ),
            "a lane matmul is in the code of a block or a subgroup"
        );
        let level = if self.across.kinds.last() == Some(&LevelKind::Blocks) {
            AcrossLevel::Block
        } else {
            AcrossLevel::Subgroup
        };
        let routine = crate::hir::check::lanes::matmul_routine(m.accumulate);
        self.lower_lane_call(routine, &[m.acc, m.lhs, m.rhs])?;
        self.barrier(level);
        Ok(())
    }

    /// `TENSOR_STORE_BLOCK_LANE_ELEMENT` or `TENSOR_STORE_SUBGROUP_LANE_ELEMENT` when the store
    /// `ix = ..` writes element `t` of `lane_element`, and otherwise 0.
    pub(super) fn lane_element_store(&self, ix: &crate::syntax::expr::IndexAccessExpr) -> u64 {
        let Some((lane, level)) = &self.across.lane_element else {
            return 0;
        };
        if !matches!(&*ix.index, Expr::Identifier(id) if &id.name == lane) {
            return 0;
        }
        match level {
            AcrossLevel::Block if matches!(&*ix.base, Expr::Identifier(_)) => {
                TENSOR_STORE_BLOCK_LANE_ELEMENT
            }
            AcrossLevel::Subgroup => match self.subgroup_row_number(&ix.base) {
                Some(row) => TENSOR_STORE_SUBGROUP_LANE_ELEMENT | (row << LANE_ROW_SHIFT),
                None => 0,
            },
            _ => 0,
        }
    }

    /// `LANE_OWN_SUBGROUP` or `LANE_OWN_WHOLE`, with the lanes of the lane loop, when `ix` is an
    /// element that this lane alone accesses, and otherwise 0. The tensor under it is a variable,
    /// not a reference that could point into another tensor.
    pub(super) fn lane_own(&self, ix: &crate::syntax::expr::IndexAccessExpr) -> u64 {
        let (Some((lane, level)), Some((lanes, once))) =
            (&self.across.lane_element, self.across.lane_loop)
        else {
            return 0;
        };
        let tensor = |e: &Expr| {
            matches!(e, Expr::Identifier(_))
                && matches!(
                    self.infer_ast_type(e),
                    Some(crate::syntax::Type::Tensor(..))
                )
        };
        let own = |e: &Expr| match (level, &self.across.subgroup_row) {
            (AcrossLevel::Block, _) | (AcrossLevel::Subgroup, Some(SubgroupRow::Whole)) => {
                tensor(e).then_some(LaneOwn::Whole)
            }
            (AcrossLevel::Subgroup, Some(SubgroupRow::Index(w))) if once => match e {
                Expr::IndexAccess(r)
                    if tensor(&r.base)
                        && matches!(&*r.index, Expr::Identifier(id) if &id.name == w) =>
                {
                    Some(LaneOwn::Subgroup)
                }
                _ => None,
            },
            _ => None,
        };
        match lane_own_access(ix, lane.as_ref(), own) {
            Some(LaneOwn::Subgroup) => LANE_OWN_SUBGROUP | (lanes << LANE_OWN_LANES_SHIFT),
            Some(LaneOwn::Whole) => LANE_OWN_WHOLE | (lanes << LANE_OWN_LANES_SHIFT),
            None => 0,
        }
    }

    /// Whether `e` is the row of the subgroup, `t[w]`, in a subgroup loop whose index `w` is the
    /// number of the subgroup.
    fn row_of_subgroup(&self, e: &Expr) -> bool {
        self.across.subgroup_once && self.subgroup_row_number(e) == Some(0)
    }

    /// When `e` is a row of a tile that the code of this subgroup alone writes, the number that
    /// tells it from the other rows of the subgroup: 0 for `t` or `t[w]`, and `c + 1` for
    /// `t[w][c]` with a number `c`. Each row keeps its own register on the device.
    fn subgroup_row_number(&self, e: &Expr) -> Option<u64> {
        let is_own_row = |e: &Expr| match (&self.across.subgroup_row, e) {
            (Some(SubgroupRow::Whole), Expr::Identifier(_)) => true,
            (Some(SubgroupRow::Index(w)), Expr::IndexAccess(r)) => {
                matches!(&*r.base, Expr::Identifier(_))
                    && matches!(&*r.index, Expr::Identifier(id) if &id.name == w)
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
            // The passes of this loop count.
            if self
                .across
                .tail
                .as_ref()
                .is_some_and(|t| t.counted == f as *const _ as usize)
            {
                self.across.arrive_body = Some(f.body.as_ptr() as usize);
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
        let index: Symbol = f.iter.as_str().into();
        let rebound = binds_name(&f.body, &index);
        let subgroup_row = match kind {
            LevelKind::Subgroups => {
                let row = if width == 1 {
                    Some(SubgroupRow::Whole)
                } else {
                    (!rebound).then(|| SubgroupRow::Index(index.clone()))
                };
                Some(std::mem::replace(&mut self.across.subgroup_row, row))
            }
            _ => None,
        };
        let lane_element = self.across.lane_element.take();
        let lane_loop = self.across.lane_loop.take();
        let from_zero = matches!(&*f.iterable,
            Expr::Range(r) if matches!(&*r.start, Expr::Number(n) if &*n.value == "0"));
        let once = kind == LevelKind::Subgroups
            && from_zero
            && one_pass_each(f, width, inner, self.across.lanes);
        let outer_once = std::mem::replace(&mut self.across.subgroup_once, once);
        let all_lanes = match root {
            Some(AcrossLevel::Block) => true,
            Some(AcrossLevel::Subgroup) => Some(width) == self.across.subgroup,
            None => false,
        };
        let shared = f.across.as_ref().is_some_and(|a| a.share.is_some());
        if kind == LevelKind::Lanes && inner == 1 && !rebound && from_zero && all_lanes && !shared {
            self.across.lane_element = root.map(|level| (index, level));
            self.across.lane_loop = Some((width, outer_once));
        }
        let lowered = self.lower_plain_for(f);
        self.across.lane_element = lane_element;
        self.across.lane_loop = lane_loop;
        self.across.subgroup_once = outer_once;
        if let Some(outer) = subgroup_row {
            self.across.subgroup_row = outer;
        }
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
