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
// Current subset: scalar arithmetic (params, const, add/sub/mul/div, compare,
// return), intra-function control flow (basic-block markers, (conditional)
// branches, and `alloca`/store/load'd scalar locals — the memory model the flat
// HIR uses so values cross blocks), and fixed-arity scalar calls (`func.call`,
// callee resolved via the registry's `fn_sigs`). `emit_function_mlir` emits one
// `func.func` as text (SSA name = producing instruction's register; blocks become
// `^bbN:` labels); `emit_module_mlir` emits a whole program (needed for calls).
// The caller parses + verifies with melior. The AST path stays the oracle: the
// emitter returns `None` for any stream (or module) using opcodes outside this
// subset, so nothing half-lowered is ever emitted.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hir::bytecode::{HirInstruction, Opcode};
use crate::hir::flatten::scalar_gid;
use crate::registry::ImmutableGlobalRegistry;
use crate::syntax::{ElementType, Function, Type};
use std::collections::HashMap;

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

/// The `arith.cmp{i,f}` op + textual predicate for a `Cmp` on operands of element type `e`, given
/// the relation code stored in the instruction's `imm` (0=Eq,1=Ne,2=Lt,3=Gt,4=Le,5=Ge — kept in
/// sync with `flatten::rel_code`). Integers use signed vs. unsigned predicates by the element's
/// signedness; floats use the ordered predicates.
fn cmp_op(rel: u64, e: &ElementType) -> Option<(&'static str, &'static str)> {
    if is_float(e) {
        let pred = match rel {
            0 => "oeq",
            1 => "one",
            2 => "olt",
            3 => "ogt",
            4 => "ole",
            5 => "oge",
            _ => return None,
        };
        Some(("arith.cmpf", pred))
    } else {
        let s = is_signed(e);
        let pred = match rel {
            0 => "eq",
            1 => "ne",
            2 => {
                if s {
                    "slt"
                } else {
                    "ult"
                }
            }
            3 => {
                if s {
                    "sgt"
                } else {
                    "ugt"
                }
            }
            4 => {
                if s {
                    "sle"
                } else {
                    "ule"
                }
            }
            5 => {
                if s {
                    "sge"
                } else {
                    "uge"
                }
            }
            _ => return None,
        };
        Some(("arith.cmpi", pred))
    }
}

/// A resolved callee for the flat emitter: the MLIR symbol name to `func.call`, and its scalar
/// return element type (`None` for a non-scalar/void return, which this subset declines). Keyed by
/// the callee's GID — the identity a `Call` instruction carries in its `type_idx`.
pub struct Callee {
    pub name: String,
    pub ret: Option<ElementType>,
}

/// GID → callee: the reverse of the registry's name-keyed `fn_sigs`. A `Call`'s `type_idx` resolves
/// to a callee GID; this map recovers the symbol name (for `func.call @name`) and the return type
/// (for the call's result type) without a name→AST walk.
pub type CalleeMap = HashMap<TypeId, Callee>;

/// Build the GID→callee map from the frozen registry's function signatures.
pub fn build_callee_map(registry: &ImmutableGlobalRegistry) -> CalleeMap {
    registry
        .fn_sigs
        .iter()
        .map(|(name, sig)| {
            (
                sig.gid,
                Callee {
                    name: name.to_string(),
                    ret: scalar_of(&sig.ret_ty),
                },
            )
        })
        .collect()
}

/// Emit a whole module — every function as a concatenated bare `func.func` — or `None` if *any*
/// function is outside the current subset (module-level keep-green atomicity: a partially lowered
/// module is never emitted, so the AST path stays the oracle for the whole program). Calls resolve
/// through the frozen registry's `fn_sigs`. Wrap the result in `module { … }` before parsing.
pub fn emit_module_mlir(
    funcs: &[(&Function, &[HirInstruction], &[TypeId])],
    registry: &ImmutableGlobalRegistry,
) -> Option<String> {
    let callees = build_callee_map(registry);
    let mut out = String::new();
    for (func, hir, types) in funcs {
        out += &emit_function_mlir(func, hir, types, &callees)?;
    }
    Some(out)
}

/// Emit a `func.func` for `func` from its flat HIR body, or `None` if the stream uses any construct
/// outside the current subset (scalar arithmetic + intra-function control flow + fixed-arity scalar
/// calls; the AST path stays the oracle there). `callees` resolves a `Call`'s callee GID to a symbol
/// name + return type. The returned text is a bare `func.func` op; wrap it in a `module { … }` before
/// parsing.
pub fn emit_function_mlir(
    func: &Function,
    hir: &[HirInstruction],
    types: &[TypeId],
    callees: &CalleeMap,
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
    // The scalar element type each register carries, indexed by register (= instruction position in
    // the stream) — the type-valued parallel to `names` above. As the stream is walked, each value-
    // producing instruction records its result type here, so any later instruction can recover the
    // type of a register it *reads*. This is the flat-driven stand-in for reading an operand's type
    // off the AST: there is no AST node to consult, so we reconstruct types as we go.
    //
    // Why not just read each instruction's own `type_idx`? Most values are self-describing that way,
    // but two consumers need an *operand's* type, which their own `type_idx` doesn't give:
    //   - `Cmp`: its own result type is `bool`, but choosing `cmpi`/`cmpf` + the signed/unsigned
    //     predicate needs the *operands'* type -> `etypes[operand1]`.
    //   - `Store`: an effect instruction (its `type_idx` is the no-type sentinel), but printing
    //     `memref<T>` needs the slot's element type -> the `Alloca` records it, `Store` reads it back.
    // Calls use it too: a `func.call`'s argument types come from `etypes[arg_reg]`.
    //
    // Scalar-only today (`ElementType`); brick 3 (#200) widens it to tensor/aggregate types.
    let mut etypes: Vec<Option<ElementType>> = vec![None; hir.len()];
    let elem_at = |etypes: &[Option<ElementType>], r: u32| -> Option<ElementType> {
        etypes.get(r as usize)?.clone()
    };
    let mut body = String::new();
    // Whether the block currently being emitted has a terminator yet (a block must end in one).
    let mut terminated = false;
    // Argument value registers accumulated by the `Arg`s that immediately precede a `Call`; the
    // `Call` consumes its `imm` trailing entries (a nested inner call sits between its own `Arg`s and
    // the outer ones, so each call's args are exactly the tail — see `flatten::lower_call`).
    let mut pending_args: Vec<u32> = Vec::new();

    for (idx, ins) in hir.iter().enumerate() {
        match ins.opcode {
            // Parameter materialization: the register *is* the block argument, no op emitted.
            Opcode::Load => {
                names[idx] = format!("%arg{}", ins.imm);
                etypes[idx] = ty_at(ins.type_idx.0);
            }
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
                etypes[idx] = Some(e);
            }
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let op = arith_op(ins.opcode, &e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                let b = names.get(ins.operand2.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = {op} {a}, {b} : {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Scalar comparison → `i1`; the relation is in `imm`, the operand type comes from the
            // first operand's tracked type (this instruction's own type is `bool`, the result).
            Opcode::Cmp => {
                let e = elem_at(&etypes, ins.operand1.0)?;
                let mt = mlir_scalar(&e)?;
                let (op, pred) = cmp_op(ins.imm, &e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                let b = names.get(ins.operand2.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = {op} {pred}, {a}, {b} : {mt}\n");
                names[idx] = n;
                etypes[idx] = Some(ElementType::Bool);
            }
            // A named local's stack slot: a rank-0 memref, matching the AST codegen's scalar locals.
            Opcode::Alloca => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = memref.alloca() : memref<{mt}>\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Store a value into a slot (no result); the memref type is the slot's element type.
            Opcode::Store => {
                let e = elem_at(&etypes, ins.operand1.0)?;
                let mt = mlir_scalar(&e)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let val = names.get(ins.operand2.0 as usize)?;
                body += &format!("  memref.store {val}, {slot}[] : memref<{mt}>\n");
            }
            // Load a value back from a slot; the result type is the slot's element (this
            // instruction's own `type_idx`).
            Opcode::SlotLoad => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let slot = names.get(ins.operand1.0 as usize)?;
                let n = format!("%v{idx}");
                body += &format!("  {n} = memref.load {slot}[] : memref<{mt}>\n");
                names[idx] = n;
                etypes[idx] = Some(e);
            }
            // Block markers → MLIR blocks. Block 0 is the func's entry block (implicit; it carries the
            // params), so it gets no label; every other id opens `^bbN:`.
            Opcode::BlockStart => {
                if ins.imm != 0 {
                    body += &format!("^bb{}:\n", ins.imm);
                }
                terminated = false;
            }
            Opcode::Br => {
                body += &format!("  cf.br ^bb{}\n", ins.imm);
                terminated = true;
            }
            // `imm` packs the two targets as `then | (else << 32)` (see `flatten::pack_targets`).
            Opcode::CondBr => {
                let cond = names.get(ins.operand1.0 as usize)?;
                let then_b = ins.imm & 0xffff_ffff;
                let else_b = ins.imm >> 32;
                body += &format!("  cf.cond_br {cond}, ^bb{then_b}, ^bb{else_b}\n");
                terminated = true;
            }
            Opcode::Ret => {
                let e = ty_at(ins.type_idx.0)?;
                let mt = mlir_scalar(&e)?;
                let a = names.get(ins.operand1.0 as usize)?;
                body += &format!("  func.return {a} : {mt}\n");
                terminated = true;
            }
            // One argument of the following `Call`: record its value register (no op emitted).
            Opcode::Arg => pending_args.push(ins.operand1.0),
            // A fixed-arity call. `type_idx` is the callee's GID (resolved to name + return type via
            // `callees`); `imm` is the arg count, taken from the tail of `pending_args`. Emit
            // `%r = func.call @name(%a, %b) : (Ta, Tb) -> Tret`.
            Opcode::Call => {
                let gid = *types.get(ins.type_idx.0 as usize)?;
                let callee = callees.get(&gid)?;
                let ret = callee.ret.clone()?; // scalar-returning calls only in this subset
                let rt = mlir_scalar(&ret)?;
                let n = ins.imm as usize;
                if pending_args.len() < n {
                    return None;
                }
                let args = pending_args.split_off(pending_args.len() - n);
                let mut arg_names = Vec::with_capacity(n);
                let mut arg_types = Vec::with_capacity(n);
                for a in &args {
                    arg_names.push(names.get(*a as usize)?.clone());
                    let e = elem_at(&etypes, *a)?;
                    arg_types.push(mlir_scalar(&e)?);
                }
                let nm = format!("%v{idx}");
                body += &format!(
                    "  {nm} = func.call @{}({}) : ({}) -> {rt}\n",
                    callee.name,
                    arg_names.join(", "),
                    arg_types.join(", "),
                );
                names[idx] = nm;
                etypes[idx] = Some(ret);
            }
            // Anything else (spawn, the non-scalar surface, matmul, …) is outside this subset.
            _ => return None,
        }
    }

    // Every block must end in a terminator. A void function falls through to a bare `return`; a
    // scalar-returning function whose final block isn't terminated is either ill-typed or has an
    // unreachable trailing block (no value to return) — decline it, leaving the AST path the oracle.
    if !terminated {
        match ret_elem {
            None => body += "  func.return\n",
            Some(_) => return None,
        }
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
        let mlir = emit_function_mlir(
            &f,
            &w.local_hir_stream,
            &w.local_type_stream,
            &CalleeMap::new(),
        )
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

    /// Lower a whole program to flat HIR and emit the module, then check it parses + verifies in a
    /// real MLIR context — proving the module emitter (calls included) produces valid MLIR end to
    /// end. Mirrors the pipeline's registry build so callees resolve through `fn_sigs`.
    fn emit_module_and_verify(src: &str) -> String {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut prog = parser.parse().expect("parse failed");
        prog.module_path = "crate::t".into();
        let mut mods = vec![prog];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map);
        let registry = crate::pipeline::build_frozen_registry(&mods).expect("registry builds");
        let session = Arc::new(GlobalSession::with_registry(1, registry));

        let mut lowered = Vec::new();
        for f in &mods[0].functions {
            let mut w = LocalWorkerState::new(session.clone());
            assert!(lower_function_to_hir(f, &mut w), "function lowers");
            lowered.push(w);
        }
        let funcs: Vec<(&Function, &[HirInstruction], &[TypeId])> = mods[0]
            .functions
            .iter()
            .zip(&lowered)
            .map(|(f, w)| {
                (
                    f,
                    w.local_hir_stream.as_slice(),
                    w.local_type_stream.as_slice(),
                )
            })
            .collect();
        let mlir = emit_module_mlir(&funcs, &session.registry).expect("emits flat module");

        use melior::ir::operation::OperationLike;
        let dialects = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&dialects);
        let context = melior::Context::new();
        context.append_dialect_registry(&dialects);
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
    fn emits_verifiable_scalar_call() {
        // The module emitter emits both `func.func`s; the call site resolves `add` through the
        // registry's `fn_sigs` and prints a matching call signature.
        let mlir = emit_module_and_verify(
            "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
             fn main() -> i32 { return add(3, 4); }",
        );
        assert!(mlir.contains("func.call @add("), "{mlir}");
        assert!(mlir.contains("(i32, i32) -> i32"), "{mlir}");
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
    fn emits_verifiable_if_else() {
        // Control flow: the memory model (alloca/store/load'd locals) + a `cf` diamond. The compare
        // drives the conditional branch; both branches reconverge at the merge block.
        let mlir = emit_and_verify(
            "fn c(a: i32) -> i32 { let mut x = a; if a < 0 { x = 0; } else { x = 1; } return x; }",
        );
        assert!(mlir.contains("memref.alloca() : memref<i32>"), "{mlir}");
        assert!(mlir.contains("arith.cmpi slt, "), "{mlir}");
        assert!(mlir.contains("cf.cond_br "), "{mlir}");
        assert!(mlir.contains("cf.br ^bb"), "{mlir}");
    }

    #[test]
    fn emits_verifiable_for_loop() {
        // A `for` range loop: header (compare + cond_br), body, increment latch, exit — all wired
        // through slots for the induction variable and accumulator.
        let mlir = emit_and_verify(
            "fn sum(n: i32) -> i32 { let mut s = 0; for i in 0..n { s = s + i; } return s; }",
        );
        assert!(mlir.contains("arith.cmpi slt, "), "{mlir}");
        assert!(mlir.contains("memref.load "), "{mlir}");
        assert!(mlir.contains("memref.store "), "{mlir}");
        assert!(mlir.contains("cf.cond_br "), "{mlir}");
    }

    #[test]
    fn declines_out_of_subset_scalar_op() {
        // A scalar `as` cast (`Cast` opcode) lowers to the flat HIR but is outside the emitter's
        // current subset -> `None`, so the AST path stays the oracle for it.
        let f = parse_fn("fn c(a: i32) -> i64 { return a as i64; }");
        let mut w = LocalWorkerState::new(Arc::new(GlobalSession::new(1)));
        assert!(lower_function_to_hir(&f, &mut w));
        assert!(emit_function_mlir(
            &f,
            &w.local_hir_stream,
            &w.local_type_stream,
            &CalleeMap::new()
        )
        .is_none());
    }
}
