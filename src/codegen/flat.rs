//===- flat.rs - Vx Compiler ---------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Flat codegen (C2, #200): lower a function's flat HIR stream (`local_hir_stream`
// + `local_type_stream`, produced by `hir/flatten.rs`) to MLIR, driven by the
// instruction array instead of an AST walk (doc §Phase 7 "O(1) array codegen").
//
// First increment: the straight-line *scalar arithmetic* subset (params, const,
// add/sub/mul/div, return). Emits a `func.func` as text (SSA name = producing
// instruction's register), which the caller parses + verifies with melior. The
// AST path stays the oracle: `emit_function_mlir` returns `None` for any stream
// using opcodes outside this subset, so nothing half-lowered is ever emitted.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hir::bytecode::{HirInstruction, Opcode};
use crate::hir::flatten::scalar_gid;
use crate::syntax::{ElementType, Function, Type};

/// The MLIR type string for a scalar element type. Integers are signless (signedness lives in the
/// op, e.g. `divsi`/`divui`); `bool` is `i1`.
fn mlir_scalar(elem: &ElementType) -> Option<&'static str> {
    use ElementType::*;
    Some(match elem {
        F16 => "f16",
        F32 => "f32",
        F64 => "f64",
        BF16 => "bf16",
        I4 | U4 => "i4",
        I8 | U8 => "i8",
        I16 | U16 => "i16",
        I32 | U32 => "i32",
        I64 | U64 => "i64",
        I128 | U128 => "i128",
        Bool => "i1",
        Generic(_) => return None,
    })
}

fn is_float(e: &ElementType) -> bool {
    matches!(
        e,
        ElementType::F16 | ElementType::F32 | ElementType::F64 | ElementType::BF16
    )
}

fn is_signed(e: &ElementType) -> bool {
    use ElementType::*;
    matches!(e, I4 | I8 | I16 | I32 | I64 | I128)
}

/// Recover the element type a type-stream GID stands for. The stream stores content-hash GIDs
/// (`scalar_gid`), so we invert by testing the finite set of scalar variants — the flat-driven
/// counterpart of reading a scalar type off the AST.
fn elem_of_gid(gid: TypeId) -> Option<ElementType> {
    use ElementType::*;
    [
        F16, F32, F64, BF16, I4, U4, I8, U8, I16, U16, I32, U32, I64, U64, I128, U128, Bool,
    ]
    .into_iter()
    .find(|e| scalar_gid(e) == gid)
}

fn scalar_of(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(ElementType::Generic(_)) => None,
        Type::Scalar(e) => Some(e.clone()),
        _ => None,
    }
}

/// The arith op mnemonic for a binary opcode at a given element type.
fn arith_op(op: Opcode, e: &ElementType) -> Option<&'static str> {
    let f = is_float(e);
    Some(match op {
        Opcode::Add => {
            if f {
                "arith.addf"
            } else {
                "arith.addi"
            }
        }
        Opcode::Sub => {
            if f {
                "arith.subf"
            } else {
                "arith.subi"
            }
        }
        Opcode::Mul => {
            if f {
                "arith.mulf"
            } else {
                "arith.muli"
            }
        }
        Opcode::Div => {
            if f {
                "arith.divf"
            } else if is_signed(e) {
                "arith.divsi"
            } else {
                "arith.divui"
            }
        }
        _ => return None,
    })
}

/// Emit a `func.func` for `func` from its flat HIR body, or `None` if the stream uses any construct
/// outside the C2.0 scalar-arithmetic subset (the AST path stays the oracle there). The returned
/// text is a bare `func.func` op; wrap it in a `module { … }` before parsing.
pub fn emit_function_mlir(
    func: &Function,
    hir: &[HirInstruction],
    types: &[TypeId],
) -> Option<String> {
    // Signature (taken from the resolved AST signature; the *body* is flat-driven).
    let mut params = Vec::new();
    for (i, (_, ty)) in func.params.iter().enumerate() {
        params.push(format!("%arg{}: {}", i, mlir_scalar(&scalar_of(ty)?)?));
    }
    let ret_elem = match &func.return_type {
        Type::Scalar(e) if !matches!(e, ElementType::Generic(_)) => Some(e.clone()),
        Type::Scalar(_) => return None,
        _ => None, // treat non-scalar returns as void for this subset
    };

    let ty_at = |ti: u32| -> Option<ElementType> { elem_of_gid(*types.get(ti as usize)?) };

    let mut names: Vec<String> = vec![String::new(); hir.len()];
    let mut body = String::new();
    let mut ret_line: Option<String> = None;

    for (idx, ins) in hir.iter().enumerate() {
        match ins.opcode {
            // Parameter materialization: the register *is* the block argument, no op emitted.
            Opcode::Load => names[idx] = format!("%arg{}", ins.imm),
            Opcode::Const => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let lit = if is_float(&e) {
                    format!("{:?}", f64::from_bits(ins.imm))
                } else {
                    (ins.imm as i64).to_string()
                };
                let n = format!("%v{idx}");
                body += &format!("  {n} = arith.constant {lit} : {mt}\n");
                names[idx] = n;
            }
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let op = arith_op(ins.opcode, &e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                let b = names.get(ins.operand2.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = {op} {a}, {b} : {mt}\n");
                names[idx] = n.clone();
            }
            Opcode::Ret => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                ret_line = Some(format!("  func.return {a} : {mt}\n"));
            }
            // Anything else (control flow, spawn, memory, matmul, …) is out of the C2.0 subset.
            _ => return None,
        }
    }

    // A scalar-returning function must actually return a value.
    if ret_elem.is_some() && ret_line.is_none() {
        return None;
    }

    let ret_sig = match &ret_elem {
        Some(e) => format!(" -> {}", mlir_scalar(e)?),
        None => String::new(),
    };
    let mut out = format!(
        "func.func @{}({}){} {{\n",
        func.name,
        params.join(", "),
        ret_sig
    );
    out += &body;
    out += &ret_line.unwrap_or_else(|| "  func.return\n".to_string());
    out += "}\n";
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::flatten::lower_function_to_hir;
    use crate::session::{GlobalSession, LocalWorkerState};
    use std::sync::Arc;

    fn parse_fn(src: &str) -> Function {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let prog = parser.parse().expect("parse failed");
        prog.functions.into_iter().next().expect("expected one fn")
    }

    /// Lower a function to flat HIR, emit MLIR, and check it parses + verifies in a real MLIR
    /// context — proving the flat stream produces valid MLIR end to end.
    fn emit_and_verify(src: &str) -> String {
        let f = parse_fn(src);
        let mut w = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        assert!(lower_function_to_hir(&f, &mut w), "function lowers");
        let mlir = emit_function_mlir(&f, &w.local_hir_stream, &w.local_type_stream)
            .expect("emits flat MLIR");

        use melior::ir::operation::OperationLike;
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        let context = melior::Context::new();
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        let module = melior::ir::Module::parse(&context, &format!("module {{\n{mlir}}}\n"))
            .unwrap_or_else(|| panic!("emitted MLIR failed to parse:\n{mlir}"));
        assert!(
            module.as_operation().verify(),
            "emitted MLIR failed to verify:\n{mlir}"
        );
        mlir
    }

    #[test]
    fn emits_verifiable_integer_add() {
        let mlir = emit_and_verify("fn add(a: i32, b: i32) -> i32 { return a + b; }");
        assert!(mlir.contains("arith.addi %arg0, %arg1 : i32"), "{mlir}");
        assert!(mlir.contains("func.return"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_float_arithmetic() {
        let mlir = emit_and_verify("fn f(a: f64, b: f64) -> f64 { return a * b + b; }");
        assert!(mlir.contains("arith.mulf"), "{mlir}");
        assert!(mlir.contains("arith.addf"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_constant_and_signed_div() {
        let mlir = emit_and_verify("fn g(a: i32) -> i32 { return a / 2; }");
        assert!(mlir.contains("arith.constant 2 : i32"), "{mlir}");
        assert!(mlir.contains("arith.divsi"), "{mlir}");
    }

    #[test]
    fn declines_control_flow_subset() {
        // A function with control flow lowers to a memory/branch stream that C2.0 doesn't emit yet
        // -> `None`, so the AST path stays the oracle for it.
        let f = parse_fn("fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } return x; }");
        let mut w = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        assert!(lower_function_to_hir(&f, &mut w));
        assert!(emit_function_mlir(&f, &w.local_hir_stream, &w.local_type_stream).is_none());
    }
}
