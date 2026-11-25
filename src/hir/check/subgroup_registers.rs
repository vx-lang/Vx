//===- subgroup_registers.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The tiles of sums that a GPU keeps in registers. On NVIDIA tensor cores (`sm_80` and later) and
// on the Apple GPU, the lowering keeps the sums of a lane matmul in the registers of each subgroup
// when the tile has no other use: `expandLaneMatmulWithoutTile` and
// `keepSimdgroupSumsInRegisters` in src/dialect/VxLowering.cpp. The rule here follows the
// lowering.
//
// A tile in shared memory that stays in registers takes no shared memory, so E6009 and E6010
// leave it out. When the rule and the lowering disagree, the lowering finds that the kernel needs
// more shared memory than its space has, and reports it.
//
// A tile in `Memory::REG` that the code of a block makes must stay in registers. Each use that
// the rule does not take is an error (E6033), and so is a part of a lane that does not fit in the
// registers (E6009). The lowering reports the tiles that it does not keep in registers, too.
//
// A tile of sums in shared memory that does not stay in registers makes the kernel slower, so
// the rule warns about it (W1037), and names the use that keeps it in memory.
//
//===----------------------------------------------------------------------===//

use std::collections::{HashMap, HashSet};

use crate::arch::LevelKind;
use crate::diagnostic::{DiagnosticCode, SourceSpan};
use crate::syntax::expr::Expr;
use crate::syntax::{ElementType, Span, Statement, Type};

/// The most f32 registers of a lane that the sums of one row of a subgroup take: a tile of 64 x
/// 64 (`laneMatmulFragmentRegisters` in src/dialect/VxLowering.cpp).
const REGISTERS_A_LANE: u64 = 128;

/// The lanes of a subgroup that holds a row of a tile of sums.
const SUBGROUP_LANES: u64 = 32;

/// The uses of the row of a tile of sums that keep the tile in registers.
const REGISTER_USES: &str = "a `lanes::fill`, a `lanes::scale_rows`, a lane matmul that adds to \
                             it, or a `lanes::copy` out of it into a tensor that is not in shared \
                             memory";

/// The GPU of a kernel, as far as matmuls of tiles in registers go.
#[derive(Clone, Copy, PartialEq)]
enum Gpu {
    /// `mma.sync`: tiles of f16 or bf16 in sizes of 16 x 8 x 16, and with `fp8`, tiles of fp8 in
    /// sizes of 16 x 8 x 32, or an f16 tile by an fp8 tile.
    TensorCores { fp8: bool },
    /// The simdgroup matrices of the Apple GPU: tiles of f16 or f32, each one alone, in sizes of 8.
    Simdgroups,
}

/// What the walk of one kernel found.
#[derive(Default)]
struct Uses {
    /// Tiles with a use that the lowering does not take, and the first such use.
    other: HashMap<String, Span>,
    /// Tiles that a lane matmul adds to, with the type that the matmul multiplies.
    added_to: HashMap<String, ElementType>,
    /// Tiles that a lane matmul writes, and the first such matmul.
    written: HashMap<String, Span>,
    /// Tiles with a lane matmul that the lowering does not take, and the first such matmul.
    not_taken: HashMap<String, (Span, NotTaken)>,
}

/// Why the lowering does not take a lane matmul.
#[derive(Clone, Copy)]
enum NotTaken {
    /// It is not `t[w] = t[w] + a @ b`, with a row of `t` for each subgroup.
    Form,
    /// The GPU has no matmul for its tiles that keeps the sums in registers.
    Tiles,
}

impl Uses {
    fn other(&mut self, name: &str, at: Span) {
        self.other.entry(name.to_string()).or_insert(at);
    }

    fn not_taken(&mut self, name: &str, at: Span, why: NotTaken) {
        self.other(name, at);
        self.not_taken.entry(name.to_string()).or_insert((at, why));
    }
}

/// Why a tile in registers cannot stay there.
enum Refused {
    /// The GPU of the kernel has no matmul that keeps sums in registers.
    NoGpu,
    /// A use that needs the tile in memory.
    Use(Span),
    /// No lane matmul adds to the tile.
    NoMatmul,
}

impl crate::hir::TypeChecker<'_> {
    /// Leaves the tiles that stay in registers out of the working set of their space, and reports
    /// the E6009 of each other tile that waited for this. Reports each tile in registers that
    /// cannot stay there.
    pub(crate) fn leave_out_tiles_in_registers(&mut self, body: &[Statement]) {
        let h = crate::hir::memory::MemoryHierarchy::build(self.env.memories.values().copied());
        let mut tiles: HashMap<String, (ElementType, Vec<u64>)> = HashMap::new();
        for placements in self.traffic.memory_placements.values() {
            for p in placements.values() {
                if let (Some(name), Some(tile)) = (&p.holder, &p.shared_tile) {
                    tiles.insert(name.clone(), tile.clone());
                }
            }
        }
        // Each tile in registers, with its position, and its space and the capacity of that.
        let mut in_registers: HashMap<String, (Span, String, Option<u64>)> = HashMap::new();
        for (name, at, ty) in std::mem::take(&mut self.block_register_tiles) {
            let Type::Tensor(elem, dims, Some(place)) = ty else {
                continue;
            };
            let sizes: Option<Vec<u64>> = dims
                .iter()
                .map(|d| d.literal().and_then(|v| v.parse::<u64>().ok()))
                .collect();
            if let Some(sizes) = sizes {
                tiles.insert(name.clone(), (elem, sizes));
            }
            let cap = h
                .descriptor(&place.space)
                .and_then(|d| d.capacity)
                .map(|c| c.0);
            in_registers.insert(name, (at, place.space.name(), cap));
        }
        let across_indices = std::mem::take(&mut self.across_register_indices);
        let mut lets: HashMap<String, usize> = HashMap::new();
        count_lets(body, &mut lets);
        // A name that two `let`s bind could name either tile.
        tiles.retain(|name, _| lets.get(name) == Some(&1));
        // The tiles that only take uses that the lowering takes.
        let mut added = HashSet::new();
        let mut refused: HashMap<String, Refused> = HashMap::new();
        let mut written: HashMap<String, Span> = HashMap::new();
        let mut not_taken: HashMap<String, (Span, NotTaken)> = HashMap::new();
        if !tiles.is_empty() {
            crate::syntax::visit_exprs(body, &mut |e| {
                if let Expr::SpawnOn(sp) = e {
                    match self.tiles_in_registers(sp, &tiles) {
                        Some(uses) => {
                            added.extend(
                                uses.added_to
                                    .into_keys()
                                    .filter(|name| !uses.other.contains_key(name)),
                            );
                            for (name, at) in uses.other {
                                refused.insert(name, Refused::Use(at));
                            }
                            written.extend(uses.written);
                            not_taken.extend(uses.not_taken);
                        }
                        None => {
                            let mut made = HashMap::new();
                            count_lets(&sp.stmts, &mut made);
                            for name in made.into_keys() {
                                refused.insert(name, Refused::NoGpu);
                            }
                        }
                    }
                }
            });
        }
        // The tiles in shared memory that stay in registers: those that fit.
        let registers: HashSet<String> = added
            .iter()
            .filter(|name| {
                let (_, sizes) = &tiles[*name];
                sizes[1] * sizes[2] / SUBGROUP_LANES <= REGISTERS_A_LANE
            })
            .cloned()
            .collect();
        for (name, (at, space, cap)) in &in_registers {
            let shape = tiles
                .get(name)
                .filter(|(elem, sizes)| *elem == ElementType::F32 && sizes.len() == 3);
            let Some((_, sizes)) = shape else {
                self.e6033(
                    format!(
                        "`{name}` is a tensor in registers that the code of a block makes, so it \
                         must be a tile of `f32` sums with a row for each subgroup. Its sizes \
                         must be numbers, as in `Tensor<f32, [4, 16, 64], Memory::REG>`"
                    ),
                    at,
                );
                continue;
            };
            if !added.contains(name) {
                let (message, at) = match refused.get(name).unwrap_or(&Refused::NoMatmul) {
                    Refused::NoGpu => (
                        format!(
                            "the GPU of this kernel cannot keep `{name}` in registers. That needs \
                             the `mma.sync` of an NVIDIA GPU, `sm_80` or later, or the \
                             simdgroup matrices of the Apple GPU"
                        ),
                        at,
                    ),
                    Refused::Use(use_at) => (
                        format!(
                            "this use of `{name}` needs it in memory, but it is a tile in \
                             registers. Each use must take the row of its subgroup, \
                             `{name}[w]`: {REGISTER_USES}"
                        ),
                        use_at,
                    ),
                    Refused::NoMatmul => (
                        format!(
                            "no lane matmul adds to `{name}`, so it cannot stay in registers. A \
                             tile in registers holds the sums of a lane matmul"
                        ),
                        at,
                    ),
                };
                self.e6033(message, at);
                continue;
            }
            let sums = sizes[1] * sizes[2] / SUBGROUP_LANES;
            let bytes = sums * 4;
            match cap {
                Some(cap) if bytes > *cap => {
                    self.errors
                        .error_with_code(
                            DiagnosticCode::E6009,
                            format!(
                                "each lane holds {bytes} bytes of `{name}`, but memory space \
                                 '{space}' has capacity {cap} bytes"
                            ),
                            Some(SourceSpan::from_ast_span(at)),
                        )
                        .facts = Some(crate::diagnostic::DiagnosticFacts::Capacity {
                        space: space.clone(),
                        required_bytes: bytes,
                        available_bytes: *cap,
                        tiles: None,
                    });
                }
                _ if sums > REGISTERS_A_LANE => self.e6033(
                    format!(
                        "each lane holds {sums} sums of `{name}`, and the compiler keeps at most \
                         {REGISTERS_A_LANE} sums of a lane in registers: a 64 x 64 tile for each \
                         subgroup"
                    ),
                    at,
                ),
                _ => {}
            }
        }
        let mut in_shared_memory: Vec<(&String, &Span)> = written
            .iter()
            .filter(|(name, _)| !registers.contains(*name) && !in_registers.contains_key(*name))
            .collect();
        in_shared_memory.sort_by_key(|(_, at)| (at.line, at.column));
        for (name, matmul) in in_shared_memory {
            let (elem, sizes) = &tiles[name];
            let why = match (not_taken.get(name), refused.get(name)) {
                (Some(&(_, why)), _) => Err(why),
                (None, Some(Refused::Use(at))) => Ok(*at),
                _ => Err(NotTaken::Tiles),
            };
            self.warn_sums_in_shared_memory(name, elem, sizes, matmul, why);
        }
        for (root, var, at) in across_indices {
            if !in_registers.contains_key(&root) {
                self.e6033(
                    format!(
                        "the register tensor '{root}' needs an index that is known when \
                         compiling: '{var}' is the variable of an `across` loop, which the \
                         compiler does not unroll. A tensor in a `scope: thread` space stays in \
                         registers, which have no address"
                    ),
                    &at,
                );
            }
        }
        for placements in self.traffic.memory_placements.values_mut() {
            placements.retain(|_, p| {
                !p.holder
                    .as_ref()
                    .is_some_and(|name| registers.contains(name) || in_registers.contains_key(name))
            });
            for p in placements.values_mut() {
                if let Some(held) = p.held_e6009.take() {
                    self.errors
                        .error_with_code(DiagnosticCode::E6009, held.message, held.span)
                        .facts = Some(held.facts);
                }
            }
        }
    }

    /// W1037 for the tile `name` of the sums of the lane matmul at `matmul`, which stays in shared
    /// memory. `why` is the use that needs the tile in memory, or why the lowering does not take
    /// the matmul.
    fn warn_sums_in_shared_memory(
        &mut self,
        name: &str,
        elem: &ElementType,
        sizes: &[u64],
        matmul: &Span,
        why: Result<Span, NotTaken>,
    ) {
        let bytes =
            sizes.iter().product::<u64>() * crate::hir::memory::element_bytes(elem).unwrap_or(4);
        let stay = format!(
            "the tile `{name}` of the sums of this matmul stays in shared memory, {bytes} bytes"
        );
        let sums = match sizes {
            [_, rows, cols] => rows * cols / SUBGROUP_LANES,
            _ => 0,
        };
        let (message, at) = match why {
            Ok(use_at) => (
                format!(
                    "this use of `{name}` keeps the tile of the sums of the matmul on line {} in \
                     shared memory, {bytes} bytes. For the tile to stay in registers, each use \
                     must take the row of its subgroup, `{name}[w]`: {REGISTER_USES}",
                    matmul.line
                ),
                use_at,
            ),
            Err(NotTaken::Form) => (
                format!(
                    "{stay}. The compiler keeps the tile in registers only for `{name}[w] = \
                     {name}[w] + a @ b`, with a row of `{name}` for each subgroup. `w` must be \
                     the variable of a loop across subgroups that gives each subgroup one pass"
                ),
                *matmul,
            ),
            Err(NotTaken::Tiles) if sums > REGISTERS_A_LANE => (
                format!(
                    "{stay}, because each lane would hold {sums} sums. The compiler keeps at most \
                     {REGISTERS_A_LANE} sums of a lane in registers: a 64 x 64 tile for each \
                     subgroup"
                ),
                *matmul,
            ),
            Err(NotTaken::Tiles) => (
                format!(
                    "{stay}. This GPU has no matmul for these tiles that keeps the sums in \
                     registers"
                ),
                *matmul,
            ),
        };
        let d = self.errors.warn(
            DiagnosticCode::W1037,
            message,
            Some(SourceSpan::from_ast_span(&at)),
        );
        if why.is_ok() {
            d.notes.push(crate::diagnostic::Note {
                message: "the matmul is here".into(),
                span: Some(SourceSpan::from_ast_span(matmul)),
            });
        }
        d.notes.push(crate::diagnostic::Note {
            message: format!(
                "with `{name}` in `Memory::REG`, the compiler keeps the tile in registers or \
                 reports an error"
            )
            .into(),
            span: None,
        });
    }

    fn e6033(&mut self, message: String, at: &Span) {
        self.errors.error_with_code(
            DiagnosticCode::E6033,
            message,
            Some(SourceSpan::from_ast_span(at)),
        );
    }

    /// The uses of the tiles of `tiles` in the kernel of `sp`. None when the GPU of the kernel
    /// keeps no tiles in registers.
    fn tiles_in_registers(
        &self,
        sp: &crate::syntax::expr::SpawnOnExpr,
        tiles: &HashMap<String, (ElementType, Vec<u64>)>,
    ) -> Option<Uses> {
        let desc = self.transfer_cost_graph.descriptor(&sp.top.kind());
        let gpu = match desc.and_then(|d| d.arch.as_deref()) {
            Some("applegpu") => Gpu::Simdgroups,
            // A kernel for NVIDIA without a `chip:` compiles for sm_80.
            Some("nvptx64") => {
                let chip = desc.and_then(|d| d.chip.as_deref()).map_or(80, chip_number);
                if chip < 80 {
                    return None;
                }
                Gpu::TensorCores { fp8: chip >= 89 }
            }
            _ => return None,
        };
        let block_lanes = crate::hir::flatten::across::across_block_lanes(&sp.stmts)?;
        let walk = Walk {
            tiles,
            gpu,
            block_lanes,
        };
        let mut uses = Uses::default();
        walk.stmts(&sp.stmts, None, None, &mut uses);
        Some(uses)
    }
}

/// The number of `chip`, such as 90 for `sm_90a`, and 0 when it has none.
fn chip_number(chip: &str) -> u32 {
    let digits: String = chip
        .strip_prefix("sm_")
        .unwrap_or("")
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().unwrap_or(0)
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
    /// one pass at most, so that `t[row]` is the row of the subgroup. `kind` is the level of the
    /// `across` loop around `stmts`, or `None` outside one.
    fn stmts(
        &self,
        stmts: &[Statement],
        row: Option<&str>,
        kind: Option<LevelKind>,
        uses: &mut Uses,
    ) {
        for s in stmts {
            match s {
                Statement::ForLoop(f) => {
                    self.other(&f.iterable, uses);
                    let loop_kind = f.across.as_ref().and_then(|a| a.kind);
                    let inner_row = match loop_kind {
                        Some(LevelKind::Subgroups) => self.subgroup_row(f),
                        // A lane routine is not in a lane loop.
                        Some(LevelKind::Lanes) => None,
                        _ => row,
                    };
                    self.stmts(&f.body, inner_row, loop_kind.or(kind), uses);
                }
                Statement::Loop(l) => self.stmts(&l.body, row, kind, uses),
                Statement::ExprStmt(e) => {
                    match crate::hir::check::lanes::bare_lane_statement(&e.expr) {
                        Some(fc) => self.lane_call(fc, row, uses),
                        None => self.other(&e.expr, uses),
                    }
                }
                Statement::Assign(a) => match crate::hir::check::lanes::lane_matmul(a) {
                    Some(m) => self.matmul(&m, row, kind, uses),
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
                uses.other(id.name.as_ref(), id.span);
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

    /// Whether `e`, or the tensor that it indexes, is a tile in shared memory or in registers.
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
    fn matmul(
        &self,
        m: &crate::hir::check::lanes::LaneMatmul,
        row: Option<&str>,
        kind: Option<LevelKind>,
        uses: &mut Uses,
    ) {
        self.other(m.lhs, uses);
        self.other(m.rhs, uses);
        // A block matmul keeps its accumulator in the registers of the warpgroup
        // when the `wgmma.mma_async` of `VX_WGMMA=1` takes it, which this
        // analysis does not model. Leave its tiles in the working set.
        if kind != Some(LevelKind::Subgroups) {
            self.other(m.acc, uses);
            return;
        }
        let mut root = m.acc;
        while let Expr::IndexAccess(ix) = root {
            root = &ix.base;
        }
        if let Expr::Identifier(id) = root {
            if self.tiles.contains_key(id.name.as_ref()) {
                uses.written.entry(id.name.to_string()).or_insert(m.at);
            }
        }
        let Some(acc) = self.row_of(m.acc, row).filter(|_| m.accumulate) else {
            if let Expr::Identifier(id) = root {
                if self.tiles.contains_key(id.name.as_ref()) {
                    uses.not_taken(id.name.as_ref(), m.at, NotTaken::Form);
                }
            }
            self.other(m.acc, uses);
            return;
        };
        let (elem, sizes) = &self.tiles[acc];
        let [_, rows, cols] = sizes[..] else {
            uses.not_taken(acc, m.at, NotTaken::Form);
            return;
        };
        let multiplied = match (
            self.tile(m.lhs, m.lhs_source),
            self.tile(m.rhs, m.rhs_source),
        ) {
            (Some((a, [m_, k])), Some((b, [k2, n])))
                if *elem == ElementType::F32 && m_ == rows && n == cols && k == k2 =>
            {
                self.multiplies(&a, &b, rows, cols, k)
            }
            _ => None,
        };
        match multiplied {
            // All the matmuls that add to a tile multiply one type.
            Some(t) if *uses.added_to.entry(acc.to_string()).or_insert(t.clone()) == t => {}
            _ => uses.not_taken(acc, m.at, NotTaken::Tiles),
        }
    }

    /// The type that the matmul of the GPU multiplies for tiles of `a` and `b` of `rows` x `k` and
    /// `k` x `cols`, as `mmaElementFor` in src/dialect/VxLowering.cpp decides. None when it takes
    /// no such tiles.
    fn multiplies(
        &self,
        a: &ElementType,
        b: &ElementType,
        rows: u64,
        cols: u64,
        k: u64,
    ) -> Option<ElementType> {
        use ElementType::{BF16, F16, F32, F8E4M3, F8E5M2};
        let is_fp8 = |t: &ElementType| matches!(t, F8E4M3 | F8E5M2);
        match self.gpu {
            Gpu::TensorCores { fp8 } => {
                let elem = match (a, b) {
                    (a, b) if a == b && matches!(a, F16 | BF16) => a.clone(),
                    (a, b) if a == b && is_fp8(a) && fp8 => a.clone(),
                    (F16, b) if is_fp8(b) && fp8 => F16,
                    (a, F16) if is_fp8(a) && fp8 => F16,
                    _ => return None,
                };
                let depth = if is_fp8(&elem) { 32 } else { 16 };
                (rows.is_multiple_of(16)
                    && cols.is_multiple_of(8)
                    && k.is_multiple_of(depth)
                    && k > 0)
                    .then_some(elem)
            }
            // Each tile loads into its own simdgroup matrix, of f16 or f32.
            Gpu::Simdgroups => ([a, b].iter().all(|t| matches!(t, F16 | F32))
                && rows.is_multiple_of(8)
                && cols.is_multiple_of(8)
                && k.is_multiple_of(8)
                && k > 0)
                .then_some(F32),
        }
    }

    /// The element type and the 2 sizes of the tile `e`: from `source`, its type before its `as
    /// f32`, or for a tile `x` or the row `x[i]` of one, from the type of `x`.
    fn tile(&self, e: &Expr, source: Option<&Type>) -> Option<(ElementType, [u64; 2])> {
        if let Some(Type::Tensor(elem, dims, _)) = source {
            let sizes: Option<Vec<u64>> = dims
                .iter()
                .map(|d| d.literal().and_then(|v| v.parse::<u64>().ok()))
                .collect();
            return match sizes.as_deref() {
                Some(&[m, k]) => Some((elem.clone(), [m, k])),
                _ => None,
            };
        }
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
