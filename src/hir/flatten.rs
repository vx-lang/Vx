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

/// A lowered value: the SSA register holding it and its (scalar) element type.
#[derive(Clone)]
struct Val {
    reg: Register,
    ty: ElementType,
}

/// Per-function lowering accumulator. Instructions and their result-type GIDs are built into local
/// buffers and only committed to the worker on full success, so a partial (aborted) lowering leaves
/// no trace.
struct Lowerer {
    code: Vec<HirInstruction>,
    types: Vec<TypeId>,
    scope: HashMap<Symbol, Val>,
}

impl Lowerer {
    fn new() -> Self {
        Self {
            code: Vec::new(),
            types: Vec::new(),
            scope: HashMap::new(),
        }
    }

    /// Emit one instruction defining a fresh SSA register (= its stream position) with result type
    /// `ty`, and return the value it produces. Each emitted instruction contributes exactly one
    /// entry to the type buffer (`type_idx` local index == instruction index).
    fn emit(
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

    /// Lower an expression to the register holding its result (emitting instructions as needed).
    /// Returns `None` for anything outside the C1.1 scalar subset — the caller then aborts.
    fn lower_expr(&mut self, e: &Expr) -> Option<Val> {
        match e {
            Expr::Number(n) => {
                let elem = number_elem(n)?;
                let imm = encode_imm(n.value.as_ref(), &elem)?;
                Some(self.emit(Opcode::Const, Register(0), Register(0), elem, imm))
            }
            // A name read yields the register the local/param is bound to (pure SSA, no instruction).
            Expr::Identifier(id) => self.scope.get(&id.name).cloned(),
            Expr::BinaryOp(b) => {
                let l = self.lower_expr(&b.lhs)?;
                let r = self.lower_expr(&b.rhs)?;
                let op = binop_opcode(&b.op)?;
                // Operands are type-checked to a common type; the result carries the lhs type.
                Some(self.emit(op, l.reg, r.reg, l.ty, 0))
            }
            // A comparison yields a `bool`; the relation is carried in `imm`.
            Expr::RelationalOp(r) => {
                let l = self.lower_expr(&r.lhs)?;
                let rhs = self.lower_expr(&r.rhs)?;
                Some(self.emit(
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
                Some(self.emit(op, v.reg, Register(0), v.ty, 0))
            }
            // A scalar `as` cast: the result carries the (scalar) target type.
            Expr::AsCast(c) => {
                let v = self.lower_expr(&c.expr)?;
                let target = scalar_of(&c.target_ty)?;
                Some(self.emit(Opcode::Cast, v.reg, Register(0), target, 0))
            }
            _ => None,
        }
    }

    /// Lower a statement. `None` aborts the whole function's lowering.
    fn lower_stmt(&mut self, s: &Statement) -> Option<()> {
        match s {
            // `let x = e` binds `x` to `e`'s result register (immutable SSA alias; C1.1 has no store).
            Statement::LetDecl(l) => {
                let v = self.lower_expr(&l.expr)?;
                self.scope.insert(l.name.clone(), v);
                Some(())
            }
            Statement::Return(r) => {
                let v = self.lower_expr(&r.expr)?;
                self.emit(Opcode::Ret, v.reg, Register(0), v.ty, 0);
                Some(())
            }
            Statement::ExprStmt(e) => {
                self.lower_expr(&e.expr)?;
                Some(())
            }
            _ => None,
        }
    }

    /// Append the built stream onto the worker: body types extend `local_type_stream` (after any
    /// signature types already there), and each instruction's local `type_idx` is rebased to the
    /// absolute index in that stream.
    fn commit(self, worker: &mut LocalWorkerState) {
        let base = worker.local_type_stream.len() as u32;
        worker.local_type_stream.extend(self.types);
        for mut ins in self.code {
            ins.type_idx = TypeIdx(ins.type_idx.0 + base);
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
    // Parameters become `Load` instructions at the top (imm = param index), giving each a register.
    for (i, (name, ty)) in func.params.iter().enumerate() {
        let elem = scalar_of(ty)?;
        let val = lw.emit(Opcode::Load, Register(0), Register(0), elem, i as u64);
        lw.scope.insert(name.clone(), val);
    }
    for stmt in &func.body {
        lw.lower_stmt(stmt)?;
    }
    Some(lw)
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

/// Debug-only structural check on a worker's flat HIR: every `type_idx` is in-bounds, and every
/// operand an opcode actually reads names a strictly-earlier instruction (SSA dominance for the
/// straight-line subset). One worker holds exactly one function's stream.
#[cfg(debug_assertions)]
pub fn verify_hir_stream(worker: &LocalWorkerState) {
    let n_types = worker.local_type_stream.len() as u32;
    for (i, ins) in worker.local_hir_stream.iter().enumerate() {
        let i = i as u32;
        assert!(
            ins.type_idx.0 < n_types,
            "HIR type_idx {} out of bounds ({}) at instruction {i}",
            ins.type_idx.0,
            n_types
        );
        match ins.opcode {
            // Binary: both operands read.
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::Matmul
            | Opcode::Cmp => {
                assert!(
                    ins.operand1.0 < i && ins.operand2.0 < i,
                    "HIR operand not dominated at instruction {i}"
                );
            }
            // Unary: operand1 read.
            Opcode::Ret | Opcode::Cast | Opcode::Neg | Opcode::Not => assert!(
                ins.operand1.0 < i,
                "HIR operand not dominated at instruction {i}"
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
