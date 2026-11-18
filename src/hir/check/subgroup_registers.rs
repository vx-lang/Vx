//===- subgroup_registers.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The tiles in shared memory that a GPU keeps in registers. On NVIDIA tensor cores (`sm_80` and
// later) and on the Apple GPU, the lowering keeps the sums of a lane matmul in the registers of
// each subgroup when the tile has no other use: `expandLaneMatmulWithoutTile` and
// `keepSimdgroupSumsInRegisters` in src/dialect/VxLowering.cpp. Such a tile takes no shared
// memory, so E6009 and E6010 leave it out. The rule here follows the lowering. When the two
// disagree, the lowering finds that the kernel needs more shared memory than its space has, and
// reports it.
//
//===----------------------------------------------------------------------===//

use std::collections::{HashMap, HashSet};

use crate::arch::LevelKind;
use crate::syntax::expr::Expr;
use crate::syntax::{ElementType, Statement};

/// The most f32 registers of a lane that the sums of one row of a subgroup take: a tile of 64 x
/// 64 (`laneMatmulFragmentRegisters` in src/dialect/VxLowering.cpp).
const REGISTERS_A_LANE: u64 = 128;

/// The GPU of a kernel, as far as matmuls of tiles in registers go.
#[derive(Clone, Copy, PartialEq)]
enum Gpu {
    /// `mma.sync.m16n8k16`: tiles of f16 or bf16, in sizes of 16 x 8 x 16.
    TensorCores,
    /// The simdgroup matrices of the Apple GPU: tiles of f16 or f32, each one alone, in sizes of 8.
    Simdgroups,
}

/// What the walk of one kernel found.
#[derive(Default)]
struct Uses {
    /// Tiles with a use that the lowering does not take.
    other: HashSet<String>,
    /// Tiles that a lane matmul adds to.
    added_to: HashSet<String>,
}

impl crate::hir::TypeChecker<'_> {
    /// Leaves the tiles that stay in registers out of the working set of their space, and reports
    /// the E6009 of each other tile that waited for this.
    pub(crate) fn leave_out_tiles_in_registers(&mut self, body: &[Statement]) {
        let mut tiles: HashMap<String, (ElementType, Vec<u64>)> = HashMap::new();
        for placements in self.traffic.memory_placements.values() {
            for p in placements.values() {
                if let (Some(name), Some(tile)) = (&p.holder, &p.shared_tile) {
                    tiles.insert(name.clone(), tile.clone());
                }
            }
        }
        let mut registers = HashSet::new();
        if !tiles.is_empty() {
            let mut lets: HashMap<String, usize> = HashMap::new();
            count_lets(body, &mut lets);
            // A name that two `let`s bind could name either tile.
            tiles.retain(|name, _| lets.get(name) == Some(&1));
            crate::syntax::visit_exprs(body, &mut |e| {
                if let Expr::SpawnOn(sp) = e {
                    registers.extend(self.tiles_in_registers(sp, &tiles));
                }
            });
        }
        for placements in self.traffic.memory_placements.values_mut() {
            placements.retain(|_, p| {
                !p.holder
                    .as_ref()
                    .is_some_and(|name| registers.contains(name))
            });
            for p in placements.values_mut() {
                if let Some(held) = p.held_e6009.take() {
                    self.errors
                        .error_with_code(
                            crate::diagnostic::DiagnosticCode::E6009,
                            held.message,
                            held.span,
                        )
                        .facts = Some(held.facts);
                }
            }
        }
    }

    /// The tiles of `tiles` that the kernel of `sp` keeps in registers.
    fn tiles_in_registers(
        &self,
        sp: &crate::syntax::expr::SpawnOnExpr,
        tiles: &HashMap<String, (ElementType, Vec<u64>)>,
    ) -> Vec<String> {
        let desc = self.transfer_cost_graph.descriptor(&sp.top.kind());
        let gpu = match desc.and_then(|d| d.arch.as_deref()) {
            Some("applegpu") => Gpu::Simdgroups,
            // A kernel for NVIDIA without a `chip:` compiles for sm_80.
            Some("nvptx64")
                if desc
                    .and_then(|d| d.chip.as_deref())
                    .is_none_or(has_mma_sync) =>
            {
                Gpu::TensorCores
            }
            _ => return Vec::new(),
        };
        let Some(block_lanes) = crate::hir::flatten::across::across_block_lanes(&sp.stmts) else {
            return Vec::new();
        };
        let walk = Walk {
            tiles,
            gpu,
            block_lanes,
        };
        let mut uses = Uses::default();
        walk.stmts(&sp.stmts, None, &mut uses);
        uses.added_to
            .into_iter()
            .filter(|name| !uses.other.contains(name))
            .filter(|name| {
                let (_, sizes) = &tiles[name];
                sizes[1] * sizes[2] / 32 <= REGISTERS_A_LANE
            })
            .collect()
    }
}

/// Whether `chip`, such as `sm_90a`, has `mma.sync` for f16 and bf16: `sm_80` and later.
fn has_mma_sync(chip: &str) -> bool {
    let digits: String = chip
        .strip_prefix("sm_")
        .unwrap_or("")
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().is_ok_and(|v| v >= 80)
}

/// Counts the `let`s of each name in `stmts`, inside blocks and loops too.
fn count_lets(stmts: &[Statement], lets: &mut HashMap<String, usize>) {
    for s in stmts {
        match s {
            Statement::LetDecl(l) => *lets.entry(l.name.to_string()).or_default() += 1,
            Statement::ForLoop(f) => count_lets(&f.body, lets),
            Statement::Loop(l) => count_lets(&l.body, lets),
            _ => {}
        }
    }
    crate::syntax::visit_exprs(stmts, &mut |e| {
        let inner: &[Statement] = match e {
            Expr::UnsafeBlock(b) => &b.stmts,
            Expr::SpawnOn(b) => &b.stmts,
            Expr::If(i) => {
                count_lets(&i.then_block, lets);
                i.else_block.as_deref().unwrap_or(&[])
            }
            _ => &[],
        };
        count_lets(inner, lets);
    });
}

struct Walk<'a> {
    tiles: &'a HashMap<String, (ElementType, Vec<u64>)>,
    gpu: Gpu,
    block_lanes: u64,
}

impl Walk<'_> {
    /// `row` is the index of the subgroup loop around `stmts`, when that loop gives each subgroup
    /// one pass at most, so that `t[row]` is the row of the subgroup.
    fn stmts(&self, stmts: &[Statement], row: Option<&str>, uses: &mut Uses) {
        for s in stmts {
            match s {
                Statement::ForLoop(f) => {
                    self.other(&f.iterable, uses);
                    let kind = f.across.as_ref().and_then(|a| a.kind);
                    let inner_row = match kind {
                        Some(LevelKind::Subgroups) => self.subgroup_row(f),
                        // A lane routine is not in a lane loop.
                        Some(LevelKind::Lanes) => None,
                        _ => row,
                    };
                    self.stmts(&f.body, inner_row, uses);
                }
                Statement::Loop(l) => self.stmts(&l.body, row, uses),
                Statement::ExprStmt(e) => {
                    match crate::hir::check::lanes::bare_lane_statement(&e.expr) {
                        Some(fc) => self.lane_call(fc, row, uses),
                        None => self.other(&e.expr, uses),
                    }
                }
                Statement::Assign(a) => match crate::hir::check::lanes::lane_matmul(a) {
                    Some(m) => self.matmul(&m, row, uses),
                    None => {
                        self.other(&a.lhs, uses);
                        self.other(&a.rhs, uses);
                    }
                },
                Statement::Drop(_) => {}
                other => crate::syntax::visit_exprs(std::slice::from_ref(other), &mut |e| {
                    self.mark(e, uses)
                }),
            }
        }
    }

    /// The index of the subgroup loop `f` when each subgroup gets one pass at most, as
    /// `one_pass_each` decides for the lowering.
    fn subgroup_row<'f>(&self, f: &'f crate::syntax::ForLoopStmt) -> Option<&'f str> {
        let from_zero = matches!(&*f.iterable,
            Expr::Range(r) if matches!(&*r.start, Expr::Number(n) if &*n.value == "0"));
        let width = crate::hir::flatten::across::across_width(f)?;
        let lanes = f.across.as_ref()?.subgroup_lanes?;
        let once = from_zero
            && width > 1
            && crate::hir::flatten::across::one_pass_each(f, width, lanes, self.block_lanes);
        let rebound = crate::hir::flatten::binds_name(&f.body, &f.iter.as_str().into());
        (once && !rebound).then_some(f.iter.as_str())
    }

    /// Each tile that `e` names has a use that the lowering does not take.
    fn other(&self, e: &Expr, uses: &mut Uses) {
        crate::syntax::visit_expr(e, &mut |x| self.mark(x, uses));
    }

    fn mark(&self, e: &Expr, uses: &mut Uses) {
        if let Expr::Identifier(id) = e {
            if self.tiles.contains_key(id.name.as_ref()) {
                uses.other.insert(id.name.to_string());
            }
        }
    }

    /// The tile of which `e` is the row of the subgroup, `t[row]`.
    fn row_of<'e>(&self, e: &'e Expr, row: Option<&str>) -> Option<&'e str> {
        let (Expr::IndexAccess(ix), Some(row)) = (e, row) else {
            return None;
        };
        match (&*ix.base, &*ix.index) {
            (Expr::Identifier(t), Expr::Identifier(i))
                if i.name.as_ref() == row && self.tiles.contains_key(t.name.as_ref()) =>
            {
                Some(t.name.as_ref())
            }
            _ => None,
        }
    }

    /// `lanes::fill(&mut t[w], v)`, `lanes::scale_rows(&mut t[w], &s)` and
    /// `lanes::copy(dst, &t[w], at)`, with `dst` a tensor that is not in shared memory.
    fn lane_call(
        &self,
        fc: &crate::syntax::expr::FunctionCallExpr,
        row: Option<&str>,
        uses: &mut Uses,
    ) {
        use crate::hir::check::lanes::{lane_routine, LaneOp};
        let borrowed = |e: &Expr, is_mut: bool| match e {
            Expr::Borrow(b) if b.is_mut == is_mut => self.row_of(&b.expr, row).is_some(),
            _ => false,
        };
        let op = lane_routine(&fc.name).map(|routine| routine.op);
        let at = match op {
            Some(LaneOp::Fill | LaneOp::ScaleRows) => fc
                .args
                .first()
                .is_some_and(|a| borrowed(a, true))
                .then_some(0),
            Some(LaneOp::Copy) => (fc.args.get(1).is_some_and(|a| borrowed(a, false))
                && !fc.args.first().is_some_and(|d| self.in_shared_memory(d)))
            .then_some(1),
            _ => None,
        };
        for (i, a) in fc.args.iter().enumerate() {
            if Some(i) != at {
                self.other(a, uses);
            }
        }
    }

    /// Whether `e`, or the tensor that it indexes, is a tile in shared memory.
    fn in_shared_memory(&self, e: &Expr) -> bool {
        let mut e = match e {
            Expr::Borrow(b) => &*b.expr,
            e => e,
        };
        while let Expr::IndexAccess(ix) = e {
            e = &ix.base;
        }
        matches!(e, Expr::Identifier(id) if self.tiles.contains_key(id.name.as_ref()))
    }

    /// `t[w] = t[w] + x @ y`, with tiles `x` and `y` that the matmul of the GPU takes.
    fn matmul(&self, m: &crate::hir::check::lanes::LaneMatmul, row: Option<&str>, uses: &mut Uses) {
        self.other(m.lhs, uses);
        self.other(m.rhs, uses);
        let Some(acc) = self.row_of(m.acc, row).filter(|_| m.accumulate) else {
            self.other(m.acc, uses);
            return;
        };
        let (elem, sizes) = &self.tiles[acc];
        let [_, rows, cols] = sizes[..] else {
            uses.other.insert(acc.to_string());
            return;
        };
        if *elem != ElementType::F32 {
            uses.other.insert(acc.to_string());
            return;
        }
        let fits = match (self.tile(m.lhs), self.tile(m.rhs)) {
            (Some((a, [m_, k])), Some((b, [k2, n]))) if m_ == rows && n == cols && k == k2 => {
                match self.gpu {
                    // `mma.sync` takes two tiles of one type.
                    Gpu::TensorCores => {
                        a == b
                            && matches!(a, ElementType::F16 | ElementType::BF16)
                            && rows % 16 == 0
                            && cols % 8 == 0
                            && k % 16 == 0
                            && k > 0
                    }
                    // Each tile loads into its own simdgroup matrix, of f16 or f32.
                    Gpu::Simdgroups => {
                        [a, b]
                            .iter()
                            .all(|t| matches!(t, ElementType::F16 | ElementType::F32))
                            && rows % 8 == 0
                            && cols % 8 == 0
                            && k % 8 == 0
                            && k > 0
                    }
                }
            }
            _ => false,
        };
        if fits {
            uses.added_to.insert(acc.to_string());
        } else {
            uses.other.insert(acc.to_string());
        }
    }

    /// The element type and the 2 sizes of the tile `e`: a tile `x`, or the row `x[i]` of one.
    fn tile(&self, e: &Expr) -> Option<(ElementType, [u64; 2])> {
        let (name, rows) = match e {
            Expr::Identifier(id) => (id.name.as_ref(), false),
            Expr::IndexAccess(ix) => match &*ix.base {
                Expr::Identifier(id) => (id.name.as_ref(), true),
                _ => return None,
            },
            _ => return None,
        };
        let (elem, sizes) = self.tiles.get(name)?;
        match (rows, sizes.as_slice()) {
            (false, [m, k]) | (true, [_, m, k]) => Some((elem.clone(), [*m, *k])),
            _ => None,
        }
    }
}
