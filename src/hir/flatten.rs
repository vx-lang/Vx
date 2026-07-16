//===- flatten.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// HIR lowering (C1, #197): the instruction-selection pass that flattens a
// type-checked function *body* from the tree AST into the worker's
// `local_hir_stream` (`Vec<HirInstruction>`), the flat bytecode Phase-7 codegen
// consumes. Distinct from `hir/lower_ast.rs`, which builds an arena *tree* HIR.
//
// SSA-by-position: the instruction at index `i` defines `Register(i)`; operands
// name earlier instructions by index; `type_idx` indexes `local_type_stream`.
// See docs/discussions/implementation_plans/hir_flattening.md.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hir::bytecode::{HirInstruction, Opcode, Register, TypeIdx};
use crate::session::LocalWorkerState;
use crate::symbol::Symbol;
use crate::syntax::{
    BinaryOp, ElementType, Expr, Function, NumberExpr, RelationalOp, Statement, Type, UnaryOp,
};
use std::collections::HashMap;

/// The stable GID of a primitive scalar type: module 0 (builtin) + a content hash of the element
/// name. Single source of truth so a scalar has the *same* GID whether it appears in a signature
/// (`pipeline.rs`) or a lowered body — the flat type stream must agree on identity.
pub fn scalar_gid(elem: &ElementType) -> TypeId {
    let sym = crate::hash::DefPath::Named(&format!("$prim::{elem:?}")).compute_symbol_hash();
    TypeId::new(0, sym, 0, 0)
}

/// `type_idx` sentinel for *effect* instructions (`Store`/`Br`/`CondBr`/`BlockStart`) that produce
/// no result value and therefore have no result type.
const NO_TYPE: u32 = u32::MAX;

/// A lowered value: the SSA register holding it and its (scalar) element type.
#[derive(Clone)]
struct Val {
    reg: Register,
    ty: ElementType,
}

/// How an in-scope name is materialized.
#[derive(Clone)]
enum Binding {
    /// Straight-line SSA: the name aliases an existing value register (no control flow).
    Reg(Val),
    /// Memory model: the name is a stack slot (`Alloca`); reads emit `SlotLoad`, writes `Store`, so
    /// the value survives across basic blocks. `ty` is the slot's element type.
    Slot { reg: Register, ty: ElementType },
}

/// Per-function lowering accumulator. Instructions and their result-type GIDs are built into local
/// buffers and only committed to the worker on full success, so a partial (aborted) lowering leaves
/// no trace.
struct Lowerer {
    code: Vec<HirInstruction>,
    types: Vec<TypeId>,
    scope: HashMap<Symbol, Binding>,
    /// When set, named locals live in memory (`Alloca`/`Store`/`SlotLoad`) so values survive across
    /// basic blocks — chosen for functions with control flow, matching the AST codegen's
    /// `alloca`-backed locals. Straight-line functions stay pure-SSA (bindings are `Reg`).
    memory: bool,
    /// Next basic-block id to hand out (0 is the entry block).
    next_block: u32,
    /// Enclosing loops: `(continue_target, break_target)` block ids. `continue` branches to the
    /// first (the header for `loop`, the increment latch for `for`), `break` to the second (exit).
    loop_stack: Vec<(u32, u32)>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            types: Vec::new(),
            scope: HashMap::new(),
            memory: false,
            next_block: 1,
            loop_stack: Vec::new(),
        }
    }

    /// Emit a *value* instruction defining a fresh SSA register (= its stream position) with result
    /// type `ty`, and return the value it produces.
    fn emit_value(
        &mut self,
        opcode: Opcode,
        o1: Register,
        o2: Register,
        ty: ElementType,
        imm: u64,
    ) -> Val {
        let type_idx = TypeIdx(self.types.len() as u32);
        self.types.push(scalar_gid(&ty));
        let reg = Register(self.code.len() as u32);
        self.code
            .push(HirInstruction::new(opcode, o1, o2, type_idx, imm));
        Val { reg, ty }
    }

    /// Emit an *effect* instruction (no result value): `type_idx` is the [`NO_TYPE`] sentinel.
    fn emit_effect(&mut self, opcode: Opcode, o1: Register, o2: Register, imm: u64) {
        self.code
            .push(HirInstruction::new(opcode, o1, o2, TypeIdx(NO_TYPE), imm));
    }

    fn new_block(&mut self) -> u32 {
        let b = self.next_block;
        self.next_block += 1;
        b
    }

    /// Whether the last emitted instruction is a block terminator — so we don't append a second one
    /// (e.g. a `Br` to the merge after a branch already `return`ed).
    fn block_terminated(&self) -> bool {
        matches!(
            self.code.last().map(|i| i.opcode),
            Some(Opcode::Ret | Opcode::Br | Opcode::CondBr)
        )
    }

    /// Bind a fresh local name to a value: an SSA alias in straight-line mode, or an `Alloca` slot
    /// (+ initializing `Store`) in memory mode.
    fn bind_local(&mut self, name: Symbol, v: Val) {
        if self.memory {
            let slot = self.emit_value(Opcode::Alloca, Register(0), Register(0), v.ty.clone(), 0);
            self.emit_effect(Opcode::Store, slot.reg, v.reg, 0);
            self.scope.insert(
                name,
                Binding::Slot {
                    reg: slot.reg,
                    ty: v.ty,
                },
            );
        } else {
            self.scope.insert(name, Binding::Reg(v));
        }
    }

    /// Assign to an already-bound name: a `Store` to its slot (memory mode) or an SSA rebind
    /// (straight-line). `None` if the name is unbound.
    fn assign_local(&mut self, name: &Symbol, v: Val) -> Option<()> {
        match self.scope.get(name)?.clone() {
            Binding::Slot { reg, .. } => {
                self.emit_effect(Opcode::Store, reg, v.reg, 0);
                Some(())
            }
            Binding::Reg(_) => {
                self.scope.insert(name.clone(), Binding::Reg(v));
                Some(())
            }
        }
    }

    /// Lower an expression to the register holding its result (emitting instructions as needed).
    /// Returns `None` for anything outside the current subset — the caller then aborts.
    fn lower_expr(&mut self, e: &Expr) -> Option<Val> {
        match e {
            Expr::Number(n) => {
                let elem = number_elem(n)?;
                let imm = encode_imm(n.value.as_ref(), &elem)?;
                Some(self.emit_value(Opcode::Const, Register(0), Register(0), elem, imm))
            }
            // A name read: an SSA alias (no instruction) or a `SlotLoad` from its memory slot.
            Expr::Identifier(id) => match self.scope.get(&id.name)?.clone() {
                Binding::Reg(v) => Some(v),
                Binding::Slot { reg, ty } => {
                    Some(self.emit_value(Opcode::SlotLoad, reg, Register(0), ty, 0))
                }
            },
            Expr::BinaryOp(b) => {
                let l = self.lower_expr(&b.lhs)?;
                let r = self.lower_expr(&b.rhs)?;
                let op = binop_opcode(&b.op)?;
                // Operands are type-checked to a common type; the result carries the lhs type.
                Some(self.emit_value(op, l.reg, r.reg, l.ty, 0))
            }
            // A comparison yields a `bool`; the relation is carried in `imm`.
            Expr::RelationalOp(r) => {
                let l = self.lower_expr(&r.lhs)?;
                let rhs = self.lower_expr(&r.rhs)?;
                Some(self.emit_value(
                    Opcode::Cmp,
                    l.reg,
                    rhs.reg,
                    ElementType::Bool,
                    rel_code(&r.op),
                ))
            }
            Expr::UnaryOp(u) => {
                let v = self.lower_expr(&u.expr)?;
                let op = match u.op {
                    UnaryOp::Neg => Opcode::Neg,
                    UnaryOp::Not => Opcode::Not,
                };
                Some(self.emit_value(op, v.reg, Register(0), v.ty, 0))
            }
            // A scalar `as` cast: the result carries the (scalar) target type.
            Expr::AsCast(c) => {
                let v = self.lower_expr(&c.expr)?;
                let target = scalar_of(&c.target_ty)?;
                Some(self.emit_value(Opcode::Cast, v.reg, Register(0), target, 0))
            }
            _ => None,
        }
    }

    /// Lower an `if`/`else` statement to basic blocks + branches (memory mode only, so mutated or
    /// cross-block locals are already in slots). Early `return` in a branch is honored: the trailing
    /// `Br` to the merge is skipped when the branch already terminated.
    fn lower_if(&mut self, e: &crate::syntax::IfExpr) -> Option<()> {
        let cond = self.lower_expr(&e.cond)?;
        let then_b = self.new_block();
        let (else_b, merge_b) = match &e.else_block {
            Some(_) => (self.new_block(), self.new_block()),
            None => {
                let m = self.new_block();
                (m, m) // no else: the "else" edge goes straight to the merge block
            }
        };
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(then_b, else_b),
        );

        // then block
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), then_b as u64);
        for s in &e.then_block {
            self.lower_stmt(s)?;
        }
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
        }

        // else block (only when distinct from the merge)
        if let Some(else_stmts) = &e.else_block {
            self.emit_effect(Opcode::BlockStart, Register(0), Register(0), else_b as u64);
            for s in else_stmts {
                self.lower_stmt(s)?;
            }
            if !self.block_terminated() {
                self.emit_effect(Opcode::Br, Register(0), Register(0), merge_b as u64);
            }
        }

        // merge block — subsequent statements continue here
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), merge_b as u64);
        Some(())
    }

    /// Lower an infinite `loop { body }`: a header block the body branches back to, plus an exit
    /// block that `break` targets. `continue` re-enters the header.
    fn lower_loop(&mut self, body: &[Statement]) -> Option<()> {
        let header = self.new_block();
        let exit = self.new_block();
        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), header as u64);
        self.loop_stack.push((header, exit)); // continue -> header, break -> exit
        for s in body {
            self.lower_stmt(s)?;
        }
        self.loop_stack.pop();
        // Back-edge, unless the body already terminated every path (e.g. ended in `break`/`return`).
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        }
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), exit as u64);
        Some(())
    }

    /// Lower `for i in a..b { body }` over a scalar exclusive range. The induction variable and the
    /// (once-evaluated) bound live in slots so they cross blocks; `continue` targets the increment
    /// latch (so it doesn't skip the step), `break` the exit.
    fn lower_for(&mut self, f: &crate::syntax::ForLoopStmt) -> Option<()> {
        let Expr::Range(range) = &*f.iterable else {
            return None; // only integer ranges for now
        };
        let start = self.lower_expr(&range.start)?;
        let end = self.lower_expr(&range.end)?;
        let ty = start.ty.clone();
        // Induction variable `i` and the loop bound both need to survive across blocks -> slots.
        let i_slot = self.emit_value(Opcode::Alloca, Register(0), Register(0), ty.clone(), 0);
        self.emit_effect(Opcode::Store, i_slot.reg, start.reg, 0);
        let end_slot = self.emit_value(Opcode::Alloca, Register(0), Register(0), ty.clone(), 0);
        self.emit_effect(Opcode::Store, end_slot.reg, end.reg, 0);
        self.scope.insert(
            f.iter.as_str().into(),
            Binding::Slot {
                reg: i_slot.reg,
                ty: ty.clone(),
            },
        );

        let header = self.new_block();
        let body_b = self.new_block();
        let latch = self.new_block();
        let exit = self.new_block();

        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);
        // header: cond = i < end
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), header as u64);
        let i_val = self.emit_value(Opcode::SlotLoad, i_slot.reg, Register(0), ty.clone(), 0);
        let end_val = self.emit_value(Opcode::SlotLoad, end_slot.reg, Register(0), ty.clone(), 0);
        let cond = self.emit_value(
            Opcode::Cmp,
            i_val.reg,
            end_val.reg,
            ElementType::Bool,
            rel_code(&RelationalOp::Lt),
        );
        self.emit_effect(
            Opcode::CondBr,
            cond.reg,
            Register(0),
            pack_targets(body_b, exit),
        );

        // body
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), body_b as u64);
        self.loop_stack.push((latch, exit)); // continue -> latch, break -> exit
        for s in &f.body {
            self.lower_stmt(s)?;
        }
        self.loop_stack.pop();
        if !self.block_terminated() {
            self.emit_effect(Opcode::Br, Register(0), Register(0), latch as u64);
        }

        // latch: i = i + 1; back to header
        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), latch as u64);
        let i2 = self.emit_value(Opcode::SlotLoad, i_slot.reg, Register(0), ty.clone(), 0);
        let one = self.emit_value(Opcode::Const, Register(0), Register(0), ty.clone(), 1);
        let inc = self.emit_value(Opcode::Add, i2.reg, one.reg, ty, 0);
        self.emit_effect(Opcode::Store, i_slot.reg, inc.reg, 0);
        self.emit_effect(Opcode::Br, Register(0), Register(0), header as u64);

        self.emit_effect(Opcode::BlockStart, Register(0), Register(0), exit as u64);
        Some(())
    }

    /// Lower `spawn on (<topology>) { body }` into a `Spawn`/`SpawnEnd`-delimited region carrying the
    /// topology dispatch id. First cut: statement-form spawn (no yielded value) with a straight-line
    /// body, in a straight-line function — nested control flow and value-producing spawn are
    /// deferred so the region stays a linear instruction range.
    fn lower_spawn(&mut self, s: &crate::syntax::SpawnOnExpr) -> Option<()> {
        if s.ret.is_some() || self.memory || body_has_control_flow(&s.stmts) {
            return None;
        }
        let top_id = crate::arch::topology_dispatch_id(&s.top);
        self.emit_effect(Opcode::Spawn, Register(0), Register(0), top_id as u64);
        for stmt in &s.stmts {
            self.lower_stmt(stmt)?;
        }
        self.emit_effect(Opcode::SpawnEnd, Register(0), Register(0), 0);
        Some(())
    }

    /// Lower a statement. `None` aborts the whole function's lowering.
    fn lower_stmt(&mut self, s: &Statement) -> Option<()> {
        match s {
            Statement::LetDecl(l) => {
                let v = self.lower_expr(&l.expr)?;
                self.bind_local(l.name.clone(), v);
                Some(())
            }
            Statement::Return(r) => {
                let v = self.lower_expr(&r.expr)?;
                self.emit_value(Opcode::Ret, v.reg, Register(0), v.ty, 0);
                Some(())
            }
            // `name = expr` (simple identifier target only).
            Statement::Assign(a) => {
                let name = simple_ident(&a.lhs)?;
                let v = self.lower_expr(&a.rhs)?;
                self.assign_local(&name, v)
            }
            Statement::ExprStmt(e) => match &e.expr {
                Expr::If(iff) => self.lower_if(iff),
                Expr::SpawnOn(sp) => self.lower_spawn(sp),
                other => {
                    self.lower_expr(other)?;
                    Some(())
                }
            },
            Statement::Loop(l) => self.lower_loop(&l.body),
            Statement::ForLoop(f) => self.lower_for(f),
            // `break`/`continue` branch to the enclosing loop's exit/continue target.
            Statement::Break(_) => {
                let (_, brk) = *self.loop_stack.last()?;
                self.emit_effect(Opcode::Br, Register(0), Register(0), brk as u64);
                Some(())
            }
            Statement::Continue(_) => {
                let (cont, _) = *self.loop_stack.last()?;
                self.emit_effect(Opcode::Br, Register(0), Register(0), cont as u64);
                Some(())
            }
            _ => None,
        }
    }

    /// Append the built stream onto the worker: body types extend `local_type_stream` (after any
    /// signature types already there), and each value instruction's local `type_idx` is rebased to
    /// the absolute index in that stream (effect instructions keep the [`NO_TYPE`] sentinel).
    fn commit(self, worker: &mut LocalWorkerState) {
        let base = worker.local_type_stream.len() as u32;
        worker.local_type_stream.extend(self.types);
        for mut ins in self.code {
            if ins.type_idx.0 != NO_TYPE {
                ins.type_idx = TypeIdx(ins.type_idx.0 + base);
            }
            worker.local_hir_stream.push(ins);
        }
    }
}

/// Lower a whole function body to flat HIR bytecode on `worker`. **Atomic**: returns `false` and
/// leaves the worker untouched if any construct is outside the current subset, so
/// `local_hir_stream` is only ever a complete, correct lowering or empty (keep-green). Returns
/// `true` when the full body lowered.
pub fn lower_function_to_hir(func: &Function, worker: &mut LocalWorkerState) -> bool {
    match try_lower(func) {
        Some(lw) => {
            lw.commit(worker);
            true
        }
        None => false,
    }
}

fn try_lower(func: &Function) -> Option<Lowerer> {
    let mut lw = Lowerer::new();
    // Control flow forces the memory model so locals survive across basic blocks (like the AST
    // codegen). Straight-line functions stay pure-SSA.
    lw.memory = body_has_control_flow(&func.body);
    if lw.memory {
        lw.emit_effect(Opcode::BlockStart, Register(0), Register(0), 0); // entry block
    }
    // Parameters: materialize the incoming value (`Load` imm = index), then bind (a slot in memory
    // mode, an SSA register otherwise).
    for (i, (name, ty)) in func.params.iter().enumerate() {
        let elem = scalar_of(ty)?;
        let incoming = lw.emit_value(Opcode::Load, Register(0), Register(0), elem, i as u64);
        lw.bind_local(name.clone(), incoming);
    }
    for stmt in &func.body {
        lw.lower_stmt(stmt)?;
    }
    Some(lw)
}

/// Whether the (top-level) body contains control flow (`if`/`loop`/`for`) — the trigger for the
/// memory model, so mutated or loop-carried locals survive across basic blocks. Nested control flow
/// rides on its enclosing top-level construct, and the `lower_*` helpers recurse in memory mode.
fn body_has_control_flow(stmts: &[Statement]) -> bool {
    stmts.iter().any(|s| match s {
        Statement::Loop(_) | Statement::ForLoop(_) => true,
        Statement::ExprStmt(e) => matches!(e.expr, Expr::If(_)),
        _ => false,
    })
}

fn simple_ident(e: &Expr) -> Option<Symbol> {
    match e {
        Expr::Identifier(id) => Some(id.name.clone()),
        _ => None,
    }
}

/// Pack an `if`'s two branch targets into `CondBr`'s `imm`: `then | (else << 32)`.
fn pack_targets(then_b: u32, else_b: u32) -> u64 {
    (then_b as u64) | ((else_b as u64) << 32)
}

/// The scalar element type of a parameter type, or `None` for non-scalars / generic scalars (which
/// C1.1 does not lower).
fn scalar_of(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(e.clone()),
        _ => None,
    }
}

fn binop_opcode(op: &BinaryOp) -> Option<Opcode> {
    Some(match op {
        BinaryOp::Add => Opcode::Add,
        BinaryOp::Sub => Opcode::Sub,
        BinaryOp::Mul => Opcode::Mul,
        BinaryOp::Div => Opcode::Div,
        BinaryOp::MatMul => Opcode::Matmul,
    })
}

/// The `Cmp` relation code stored in `imm` (kept in sync with codegen's decoding).
fn rel_code(op: &RelationalOp) -> u64 {
    match op {
        RelationalOp::Eq => 0,
        RelationalOp::NotEq => 1,
        RelationalOp::Lt => 2,
        RelationalOp::Gt => 3,
        RelationalOp::Le => 4,
        RelationalOp::Ge => 5,
    }
}

/// The element type of a numeric literal: its checked annotation when concrete, else inferred from
/// the spelling.
fn number_elem(n: &NumberExpr) -> Option<ElementType> {
    match &n.ty {
        Some(ElementType::Generic(_)) | None => infer_elem(n.value.as_ref()),
        Some(e) => Some(e.clone()),
    }
}

fn infer_elem(s: &str) -> Option<ElementType> {
    if s.contains('.') || s.contains('e') || s.contains('E') {
        Some(ElementType::F64)
    } else {
        Some(ElementType::I64)
    }
}

fn is_float(e: &ElementType) -> bool {
    matches!(
        e,
        ElementType::F16 | ElementType::F32 | ElementType::F64 | ElementType::BF16
    )
}

/// Encode a literal's raw 64-bit `imm`: float bit-pattern for float types, else the integer value.
fn encode_imm(s: &str, elem: &ElementType) -> Option<u64> {
    if is_float(elem) {
        Some(s.parse::<f64>().ok()?.to_bits())
    } else if *elem == ElementType::Bool {
        match s {
            "true" => Some(1),
            "false" => Some(0),
            _ => s.parse::<u64>().ok(),
        }
    } else if let Ok(i) = s.parse::<i64>() {
        Some(i as u64)
    } else {
        s.parse::<u64>().ok()
    }
}

/// Debug-only structural check on a worker's flat HIR: every value `type_idx` is in-bounds (effect
/// instructions carry the [`NO_TYPE`] sentinel), every operand an opcode reads names a
/// strictly-earlier instruction (temporaries are block-local, so the linear check still captures SSA
/// dominance), and every branch targets a declared block. One worker holds exactly one function's
/// stream.
#[cfg(debug_assertions)]
pub fn verify_hir_stream(worker: &LocalWorkerState) {
    let n_types = worker.local_type_stream.len() as u32;
    // Declared basic blocks (a `BlockStart` per block id) — branch targets must land in this set.
    let blocks: std::collections::HashSet<u64> = worker
        .local_hir_stream
        .iter()
        .filter(|ins| ins.opcode == Opcode::BlockStart)
        .map(|ins| ins.imm)
        .collect();

    for (i, ins) in worker.local_hir_stream.iter().enumerate() {
        let i = i as u32;
        if ins.type_idx.0 != NO_TYPE {
            assert!(
                ins.type_idx.0 < n_types,
                "HIR type_idx {} out of bounds ({}) at instruction {i}",
                ins.type_idx.0,
                n_types
            );
        }
        match ins.opcode {
            // Binary: both operands read.
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Matmul
            | Opcode::Cmp
            | Opcode::Store => {
                assert!(
                    ins.operand1.0 < i && ins.operand2.0 < i,
                    "HIR operand not dominated at instruction {i}"
                );
            }
            // Unary: operand1 read.
            Opcode::Ret | Opcode::Cast | Opcode::Neg | Opcode::Not | Opcode::SlotLoad => assert!(
                ins.operand1.0 < i,
                "HIR operand not dominated at instruction {i}"
            ),
            Opcode::CondBr => {
                assert!(
                    ins.operand1.0 < i,
                    "HIR cond not dominated at instruction {i}"
                );
                let then_b = ins.imm & 0xFFFF_FFFF;
                let else_b = ins.imm >> 32;
                assert!(
                    blocks.contains(&then_b) && blocks.contains(&else_b),
                    "HIR CondBr targets undeclared block(s) at instruction {i}"
                );
            }
            Opcode::Br => assert!(
                blocks.contains(&ins.imm),
                "HIR Br targets an undeclared block at instruction {i}"
            ),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::GlobalSession;
    use std::sync::Arc;

    fn parse_fn(src: &str) -> Function {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let prog = parser.parse().expect("parse failed");
        prog.functions.into_iter().next().expect("expected one fn")
    }

    fn worker() -> LocalWorkerState {
        LocalWorkerState::new(Arc::new(GlobalSession::new(1)))
    }

    fn opcodes(w: &LocalWorkerState) -> Vec<Opcode> {
        w.local_hir_stream.iter().map(|i| i.opcode).collect()
    }

    fn count(w: &LocalWorkerState, op: Opcode) -> usize {
        w.local_hir_stream.iter().filter(|i| i.opcode == op).count()
    }

    #[test]
    fn if_else_lowers_to_basic_blocks_and_memory_locals() {
        let f = parse_fn(
            "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        // Control flow forced the memory model: slots for `a` and `x`.
        assert!(count(&w, Opcode::Alloca) >= 2, "alloca slots for a and x");
        assert!(count(&w, Opcode::SlotLoad) >= 1);
        assert_eq!(count(&w, Opcode::CondBr), 1);
        assert_eq!(
            count(&w, Opcode::Br),
            2,
            "then and else each branch to merge"
        );
        assert_eq!(count(&w, Opcode::BlockStart), 4, "entry, then, else, merge");
        verify_hir_stream(&w);
    }

    #[test]
    fn if_without_else_targets_merge_directly() {
        let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } return x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::CondBr), 1);
        assert_eq!(
            count(&w, Opcode::Br),
            1,
            "then branches to merge; else edge is merge"
        );
        assert_eq!(count(&w, Opcode::BlockStart), 3, "entry, then, merge");
        verify_hir_stream(&w);
    }

    #[test]
    fn early_return_in_branch_emits_no_trailing_branch() {
        let f = parse_fn("fn c(a: i32) -> i32 { if a < 0 { return 0; } return a; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        // The then-block terminates with Ret, so no `Br` to the merge is appended.
        assert_eq!(
            count(&w, Opcode::Br),
            0,
            "returning branch emits no trailing Br"
        );
        assert_eq!(count(&w, Opcode::Ret), 2);
        assert_eq!(count(&w, Opcode::CondBr), 1);
        verify_hir_stream(&w);
    }

    #[test]
    fn straight_line_reassignment_stays_pure_ssa() {
        // No control flow -> SSA mode: reassignment is a rebind, no memory ops or blocks.
        let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; x = a + a; return x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::Alloca), 0);
        assert_eq!(count(&w, Opcode::Store), 0);
        assert_eq!(count(&w, Opcode::BlockStart), 0);
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Add, Opcode::Ret],
            "a materialized once, x rebinds to a+a"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn for_range_loop_lowers_header_body_latch_exit() {
        let f = parse_fn(
            "fn sum(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::CondBr), 1, "loop condition test");
        assert_eq!(
            count(&w, Opcode::BlockStart),
            5,
            "entry, header, body, latch, exit"
        );
        // entry->header, body->latch, latch->header
        assert_eq!(count(&w, Opcode::Br), 3);
        // The induction var increments: an Add feeding a Store in the latch.
        assert!(
            count(&w, Opcode::Add) >= 2,
            "body add + induction increment"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn infinite_loop_with_break_verifies() {
        let f = parse_fn(
            "fn f(a: i32) -> i32 { let mut x = a; loop { x = x - 1; if x < 0 { break; } } return x; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert!(count(&w, Opcode::CondBr) >= 1, "the if condition");
        // loop header/exit + entry + the if's then/merge blocks.
        assert!(count(&w, Opcode::BlockStart) >= 4);
        verify_hir_stream(&w); // every branch (incl. the break) targets a declared block
    }

    #[test]
    fn for_loop_with_continue_verifies() {
        let f = parse_fn(
            "fn f(n: i32) -> i32 { let mut s = 0; for i in 0..n { if i < 2 { continue; } s = s + i; } return s; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        verify_hir_stream(&w);
    }

    #[test]
    fn for_over_non_range_aborts() {
        // A non-range iterable (here a call) is outside the C1.2c subset -> atomic abort.
        let f = parse_fn("fn f(a: i32) -> i32 { for i in gen() { } return a; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn break_outside_loop_aborts() {
        let f = parse_fn("fn f() -> i32 { break; return 0; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn spawn_wraps_body_in_region_markers() {
        let f = parse_fn(
            "fn k(a: i32) -> i32 { spawn on (Topology::GPU) { let x = a + 1; } return a; }",
        );
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(count(&w, Opcode::Spawn), 1);
        assert_eq!(count(&w, Opcode::SpawnEnd), 1);
        // The Spawn carries the topology dispatch id, and the body (`a + 1`) lowered between the
        // region markers.
        let spawn = w
            .local_hir_stream
            .iter()
            .find(|i| i.opcode == Opcode::Spawn)
            .unwrap();
        assert_eq!(
            spawn.imm,
            crate::arch::topology_dispatch_id(&crate::syntax::Topology::GPU) as u64
        );
        assert!(
            count(&w, Opcode::Add) >= 1,
            "body arithmetic lowered inside the region"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn value_producing_spawn_aborts() {
        // A spawn that yields a value (no trailing `;`) is deferred -> atomic abort.
        let f = parse_fn("fn k(a: i32) -> i32 { spawn on (Topology::GPU) { a + 1 } }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }

    #[test]
    fn lowers_params_arithmetic_and_return() {
        let f = parse_fn("fn add(a: i32, b: i32) -> i32 { return a + b; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));

        // Load a, Load b, Add(a,b), Ret(add).
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Load, Opcode::Add, Opcode::Ret]
        );
        let add = w.local_hir_stream[2];
        assert_eq!(
            (add.operand1.0, add.operand2.0),
            (0, 1),
            "Add reads the two params"
        );
        assert_eq!(
            w.local_hir_stream[3].operand1.0, 2,
            "Ret reads the Add result"
        );
        // One type per instruction, all in bounds.
        assert_eq!(w.local_type_stream.len(), 4);
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_let_binding_and_reuse() {
        // y = x * x; return y + x  ->  Load x, Mul(0,0), (let y=1), Add(1,0), Ret(2)
        let f = parse_fn("fn sq(x: i32) -> i32 { let y = x * x; return y + x; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Mul, Opcode::Add, Opcode::Ret]
        );
        let mul = w.local_hir_stream[1];
        assert_eq!(
            (mul.operand1.0, mul.operand2.0),
            (0, 0),
            "x * x reads param twice"
        );
        let add = w.local_hir_stream[2];
        assert_eq!(
            (add.operand1.0, add.operand2.0),
            (1, 0),
            "y + x reads Mul result and x"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_integer_literal_immediate() {
        let f = parse_fn("fn seven() -> i64 { return 7; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Const, Opcode::Ret]);
        assert_eq!(
            w.local_hir_stream[0].imm, 7,
            "Const carries the literal value"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_float_literal_as_bit_pattern() {
        let f = parse_fn("fn half() -> f64 { return 0.5; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(w.local_hir_stream[0].imm, 0.5f64.to_bits());
    }

    #[test]
    fn lowers_comparison_to_bool() {
        let f = parse_fn("fn lt(a: i32, b: i32) -> bool { return a < b; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(
            opcodes(&w),
            vec![Opcode::Load, Opcode::Load, Opcode::Cmp, Opcode::Ret]
        );
        let cmp = w.local_hir_stream[2];
        assert_eq!((cmp.operand1.0, cmp.operand2.0), (0, 1));
        assert_eq!(cmp.imm, 2, "Lt relation code");
        // The Cmp result type is bool.
        assert_eq!(
            w.local_type_stream[cmp.type_idx.0 as usize],
            scalar_gid(&ElementType::Bool)
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_scalar_cast_with_target_type() {
        let f = parse_fn("fn widen(a: i32) -> i64 { return a as i64; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Cast, Opcode::Ret]);
        let cast = w.local_hir_stream[1];
        assert_eq!(cast.operand1.0, 0, "cast reads the source");
        assert_eq!(
            w.local_type_stream[cast.type_idx.0 as usize],
            scalar_gid(&ElementType::I64),
            "cast result carries the target type"
        );
        verify_hir_stream(&w);
    }

    #[test]
    fn lowers_unary_negation() {
        let f = parse_fn("fn neg(a: i32) -> i32 { return -a; }");
        let mut w = worker();
        assert!(lower_function_to_hir(&f, &mut w));
        assert_eq!(opcodes(&w), vec![Opcode::Load, Opcode::Neg, Opcode::Ret]);
        assert_eq!(w.local_hir_stream[1].operand1.0, 0);
        verify_hir_stream(&w);
    }

    #[test]
    fn aborts_atomically_on_unsupported_construct() {
        // A call is outside the C1.1 subset -> abort, worker untouched.
        let f = parse_fn("fn f(a: i32) -> i32 { return g(a); }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty(), "no partial stream on abort");
        assert!(w.local_type_stream.is_empty(), "no partial types on abort");
    }

    #[test]
    fn aborts_on_non_scalar_parameter() {
        let f = parse_fn("fn f(s: Widget) -> i32 { return 0; }");
        let mut w = worker();
        assert!(!lower_function_to_hir(&f, &mut w));
        assert!(w.local_hir_stream.is_empty());
    }
}
