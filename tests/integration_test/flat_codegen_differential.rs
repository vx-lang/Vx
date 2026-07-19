//===- flat_codegen_differential.rs - Vx Compiler --------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// Differential testing of the *flat* codegen path against the AST path, the
// oracle (#200). For a function the flat HIR lowers today, the flat path
// (local_hir_stream -> `flat::emit_module_mlir`) must produce MLIR that, run
// through the *same* production lowering (`lower_to_llvm`) and JIT, yields the
// *same* result as the AST path. Parity here is what lets `vxc` eventually flip
// to the flat path. Coverage grows as the flat HIR + emitter widen (control
// flow, calls, then the non-scalar surface #199).
//===----------------------------------------------------------------------===//
use vxc::codegen::{lower_to_llvm, register_vx_dialect, MeliorGenerator};
use vxc::hir::flatten::lower_function_to_hir;
use vxc::hir::{GlobalAstEnv, TypeChecker};
use vxc::jit::execute_mlir;
use vxc::lexer::Lexer;
use vxc::parser::Parser;
use vxc::session::{GlobalSession, LocalWorkerState};
use vxc::syntax::Program;

fn make_context() -> melior::Context {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    register_vx_dialect(&context);
    context
}

fn parse(src: &str) -> Program {
    let mut lexer = Lexer::new(src);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(&tokens, src);
    parser.parse().expect("parse failed")
}

/// JIT the given LLVM-dialect MLIR and return the process exit code (`main`'s
/// return value): `execute_mlir` yields `Ok` for a zero exit and an `Err`
/// carrying the code otherwise.
fn exit_code(llvm_mlir: &str) -> i32 {
    match execute_mlir(llvm_mlir, vec![], 0, false) {
        Ok(_) => 0,
        Err(e) => e
            .rsplit(':')
            .next()
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or_else(|| panic!("unexpected execute_mlir error: {e}")),
    }
}

/// Normalize JIT stdout for comparison: the `printMemref*` helper prints a
/// non-deterministic heap pointer (`base@ = 0x...`); replace every `0x<hex>` with
/// a placeholder so the shape/strides/data (which *are* deterministic) compare.
fn normalize(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(pos) = rest.find("0x") {
        out.push_str(&rest[..pos]);
        out.push_str("0x<ptr>");
        let after = &rest[pos + 2..];
        let hexlen = after.chars().take_while(|c| c.is_ascii_hexdigit()).count();
        rest = &after[hexlen..];
    }
    out.push_str(rest);
    out
}

/// The (normalized) stdout of running the given LLVM-dialect MLIR through the JIT.
/// Panics if the program exits non-zero (a printing program returns 0).
fn run_output(llvm_mlir: &str) -> String {
    match execute_mlir(llvm_mlir, vec![], 0, false) {
        Ok(out) => normalize(&out),
        Err(e) => panic!("printing program exited non-zero: {e}"),
    }
}

/// The lowered LLVM-dialect MLIR for `main` compiled through the AST codegen (the oracle).
fn ast_llvm(src: &str) -> String {
    let mut program = parse(src);
    let program_arr = [program.clone()];
    let env = GlobalAstEnv::build(&program_arr);
    let mut worker = LocalWorkerState::new(std::sync::Arc::new(GlobalSession::new(1)));
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    assert_eq!(checker.errors.error_count(), 0, "AST type-checks");

    let context = make_context();
    let mut codegen = MeliorGenerator::new(&context, "diff_ast".to_string());
    let module_syntaxes = std::collections::HashMap::new();
    codegen.generate(&program, &module_syntaxes).unwrap();
    let mut module = codegen.into_module();
    lower_to_llvm(&context, &mut module).expect("AST lower_to_llvm");
    module.as_operation().to_string()
}

/// Exit code of `main` compiled through the AST codegen (the oracle).
fn ast_exit_code(src: &str) -> i32 {
    exit_code(&ast_llvm(src))
}

/// The lowered LLVM-dialect MLIR for the whole program compiled through the *flat*
/// path, or `None` if the flat HIR / emitter declines any function (outside the
/// current subset). Lowers *all* functions (so calls resolve to their callee
/// `func.func`) through the frozen registry, then emits one module via
/// `flat::emit_module_mlir`.
fn flat_llvm(src: &str) -> Option<String> {
    let mut program = parse(src);
    program.module_path = "crate::diff".into();
    let mut mods = vec![program];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map);
    let registry = vxc::pipeline::build_frozen_registry(&mods).ok()?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

    // Type-check so the type checker annotates each `StructInit` with its struct GID (a scratch
    // worker; the annotation lands on the AST, which the per-function lowering below then reads).
    let env_mods = mods.clone();
    let env = GlobalAstEnv::build(&env_mods);
    {
        let mut scratch = LocalWorkerState::new(session.clone());
        let mut checker = TypeChecker::new(&env, &mut scratch);
        for f in &mut mods[0].functions {
            checker.check_function(f);
        }
    }

    // Lower every function into its own worker; decline the whole program if any
    // function is outside the flat subset (module-level keep-green atomicity).
    let mut lowered: Vec<LocalWorkerState> = Vec::new();
    for f in &mods[0].functions {
        let mut worker = LocalWorkerState::new(session.clone());
        if !lower_function_to_hir(f, &mut worker) {
            return None;
        }
        lowered.push(worker);
    }
    let funcs: Vec<(&_, &[_], &[_])> = mods[0]
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
    let tensor_types: Vec<_> = lowered
        .iter()
        .flat_map(|w| w.local_tensor_types.iter().cloned())
        .collect();
    let body = vxc::codegen::flat::emit_module_mlir(&funcs, &session.registry, &tensor_types)?;

    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &format!("module {{\n{body}}}\n"))
        .expect("flat MLIR parses");
    lower_to_llvm(&context, &mut module).expect("flat lower_to_llvm");
    Some(module.as_operation().to_string())
}

/// Exit code of the program compiled through the *flat* path (`None` if declined).
fn flat_exit_code(src: &str) -> Option<i32> {
    Some(exit_code(&flat_llvm(src)?))
}

/// The parity assertion for a *printing* program: the flat path lowers it, and its
/// (normalized) JIT stdout equals the AST path's.
fn assert_output_parity(src: &str) {
    let flat = flat_llvm(src).expect("flat path lowers this printing program");
    let ast = ast_llvm(src);
    let flat_out = run_output(&flat);
    let ast_out = run_output(&ast);
    assert_eq!(
        flat_out, ast_out,
        "flat print output diverged from the AST path for `{src}`\nflat:\n{flat_out}\nast:\n{ast_out}"
    );
}

/// The core parity assertion: the flat path lowers `main`, and its JIT exit code
/// equals both the AST path's and the expected value.
fn assert_parity(src: &str, expected: i32) {
    let flat = flat_exit_code(src).expect("flat path lowers this main");
    let ast = ast_exit_code(src);
    assert_eq!(
        ast, expected,
        "AST oracle differs from expected for `{src}`"
    );
    assert_eq!(
        flat, ast,
        "flat codegen diverged from the AST path for `{src}` (flat={flat}, ast={ast})"
    );
}

#[test]
fn flat_matches_ast_integer_arithmetic() {
    assert_parity("fn main() -> i32 { return 3 + 4 * 5; }", 23);
}

#[test]
fn flat_matches_ast_subtraction_and_div() {
    assert_parity("fn main() -> i32 { return 100 - 84 / 2; }", 58);
}

#[test]
fn flat_matches_ast_with_a_scalar_param_chain() {
    // A helper-free chain so the whole `main` stays in the flat subset.
    assert_parity("fn main() -> i32 { return (2 + 3) * (10 - 3); }", 35);
}

#[test]
fn flat_matches_ast_if_without_else() {
    // `if` with a fall-through merge: the branch runs, then both edges reconverge
    // and the function returns the mutated local.
    assert_parity(
        "fn main() -> i32 { let mut x = 5; if x < 10 { x = x + 100; } return x; }",
        105,
    );
}

#[test]
fn flat_matches_ast_if_else() {
    // A full if/else diamond; the taken branch decides the returned value.
    assert_parity(
        "fn main() -> i32 { let a = 3; let mut y = 0; if a < 0 { y = 1; } else { y = 2; } return y; }",
        2,
    );
}

#[test]
fn flat_matches_ast_for_range_accumulator() {
    // A `for` range loop accumulating 0+1+2+3+4 through slot-backed locals.
    assert_parity(
        "fn main() -> i32 { let mut s = 0; for i in 0..5 { s = s + i; } return s; }",
        10,
    );
}

#[test]
fn flat_matches_ast_loop_with_break() {
    // An infinite `loop` exited by `break` from a nested `if` — exercises the
    // loop back-edge, the break target, and a compare inside the body.
    assert_parity(
        "fn main() -> i32 { let mut i = 0; loop { if i >= 3 { break; } i = i + 1; } return i; }",
        3,
    );
}

#[test]
fn flat_matches_ast_scalar_helper_call() {
    // A `main` that calls a scalar helper — exercises the module-level emitter
    // (both `func.func`s in one module) + callee resolution via `fn_sigs`.
    assert_parity(
        "fn add(a: i32, b: i32) -> i32 { return a + b; }\n\
         fn main() -> i32 { return add(20, 22); }",
        42,
    );
}

#[test]
fn flat_matches_ast_call_inside_expression() {
    // The call is one operand of a larger arithmetic expression in `main`.
    assert_parity(
        "fn mul(a: i32, b: i32) -> i32 { return a * b; }\n\
         fn main() -> i32 { return mul(6, 7) - 2; }",
        40,
    );
}

#[test]
fn flat_matches_ast_nested_calls() {
    // A call whose argument is itself a call — the flat stream nests `Arg`/`Call`
    // pairs, so each `Call` must consume exactly its own trailing args.
    assert_parity(
        "fn inc(x: i32) -> i32 { return x + 1; }\n\
         fn dbl(x: i32) -> i32 { return x + x; }\n\
         fn main() -> i32 { return dbl(inc(9)); }",
        20,
    );
}

#[test]
fn flat_matches_ast_call_from_control_flow() {
    // Bricks 1+2 together: a helper called from inside an `if` in `main`, its
    // argument read from a slot-backed local.
    assert_parity(
        "fn sq(x: i32) -> i32 { return x * x; }\n\
         fn main() -> i32 { let n = 5; let mut r = 0; if n > 0 { r = sq(n); } return r; }",
        25,
    );
}

#[test]
fn flat_matches_ast_struct_field_sum() {
    // A struct built in place, its fields read back and summed — exercises the
    // aggregate `llvm.alloca` + GEP + `llvm.load`/`store` field path.
    assert_parity(
        "struct Point { x: i32, y: i32 }\n\
         fn main() -> i32 { let p = Point { x: 3, y: 4 }; return p.x + p.y; }",
        7,
    );
}

#[test]
fn flat_matches_ast_struct_field_in_control_flow() {
    // Bricks 1+3 together: struct fields feed an `if` condition and the returned
    // value (a struct slot and a scalar slot coexist in the same function).
    assert_parity(
        "struct Box { lo: i32, hi: i32 }\n\
         fn main() -> i32 { let b = Box { lo: 2, hi: 9 }; let mut r = 0; if b.lo < b.hi { r = b.hi - b.lo; } return r; }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_element_read() {
    // The minimal self-contained tensor program: allocate a tensor, fill it with
    // scalar-element stores, and read one element back (no reduction/cast, so the
    // result is an i32 exit code). Exercises the tensor-type side table +
    // memref.alloc/store/load through the real JIT.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
         q[3] = 8; return q[2]; }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_row_element_read() {
    // A rank-2 tensor: fill it, then read an element through a row sub-view
    // (`q[1][2]`) — exercises `memref.reinterpret_cast` (row) + `memref.load`
    // through the strided row, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<i32>([2, 3]); q[0][0] = 1; q[0][1] = 2; \
         q[0][2] = 3; q[1][0] = 4; q[1][1] = 5; q[1][2] = 6; return q[1][2]; }",
        6,
    );
}

#[test]
fn flat_matches_ast_tensor_param_passed_to_helper() {
    // A tensor built in `main`, passed by value to a helper with a tensor param
    // (a `memref` in the signature) that indexes it. Exercises tensor params +
    // tensor call arguments, flat-vs-AST.
    assert_parity(
        "fn get(q: Tensor<i32, [4]>, i: i32) -> i32 { return q[i]; }\n\
         fn main() -> i32 { let mut q = Tensor<i32>([4]); q[0] = 5; q[1] = 6; q[2] = 7; \
         q[3] = 8; return get(q, 2); }",
        7,
    );
}

#[test]
fn flat_matches_ast_tensor_row_sum_reduction() {
    // A float `sum` over a static-sized row (`q[0]`, a reinterpret_cast sub-view)
    // fed into a compare so the exit code is an i32: sum([1,2,3,4]) = 10 > 9 sets
    // r = 1. Exercises reinterpret_cast + vector.load + vector.reduction through
    // the real JIT, flat-vs-AST. (The AST oracle can only reduce a static-sized
    // slice, so the reduction is over a row, not the whole dynamic tensor.)
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let mut r = 0; if sum(q[0]) > 9.0 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_elementwise_row_store() {
    // The FlashAttention write-path shape: an elementwise scalar-broadcast multiply
    // over a row (`q[0] * 2.0`) stored back into a row (`o[0] = …`), then an element
    // read + compare so the exit is an i32. o[0] = [2,4,6,8]; o[0][1] = 4 > 3.5 -> 1.
    // Exercises vector.load/broadcast + arith.mulf + vector.store, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let mut o = Tensor<f32>([2, 4]); o[0] = q[0] * 2.0; \
         let mut r = 0; if o[0][1] > 3.5 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_flashattention_write_path() {
    // Capstone: the FlashAttention inner write path composed end to end through the
    // flat path -- `o[0] = v[0] * (dot(q[0], k[0]) * scale)`. dot([1,2,3,4],[1,1,1,1])
    // = 10; * 0.5 = 5; v[0] * 5 = [10,10,10,10]; o[0][0] = 10 > 9 -> r = 1. Exercises
    // reduction (dot) + scalar multiply + elementwise broadcast + row store + read,
    // all together, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { \
           let mut q = Tensor<f32>([1, 4]); q[0][0] = 1.0; q[0][1] = 2.0; q[0][2] = 3.0; q[0][3] = 4.0; \
           let mut k = Tensor<f32>([1, 4]); k[0][0] = 1.0; k[0][1] = 1.0; k[0][2] = 1.0; k[0][3] = 1.0; \
           let mut v = Tensor<f32>([1, 4]); v[0][0] = 2.0; v[0][1] = 2.0; v[0][2] = 2.0; v[0][3] = 2.0; \
           let mut o = Tensor<f32>([1, 4]); \
           let scale = 0.5; \
           o[0] = v[0] * (dot(q[0], k[0]) * scale); \
           let mut r = 0; if o[0][0] > 9.0 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_transfer() {
    // `transfer(q, Memory::NPU_HBM)` moves (copies) the tensor to a memory space;
    // the vx→standard lowering makes it an alloc + memref.copy. Read an element of
    // the copy back for an i32 exit: o[0][2] = q[0][2] = 3.0 > 2.5 -> r = 1.
    // Exercises vx.transfer, flat-vs-AST.
    assert_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 4]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[0][2] = 3.0; q[0][3] = 4.0; let o = transfer(q, Memory::NPU_HBM); let mut r = 0; \
         if o[0][2] > 2.5 { r = 1; } return r; }",
        1,
    );
}

#[test]
fn flat_matches_ast_tensor_print_output() {
    // Print a filled tensor and compare the JIT *stdout* (not the exit code) of the
    // flat path against the AST path -- the printMemrefF32 dump (shape/strides/data)
    // must match after normalizing the non-deterministic base pointer.
    assert_output_parity(
        "fn main() -> i32 { let mut q = Tensor<f32>([2, 2]); q[0][0] = 1.0; q[0][1] = 2.0; \
         q[1][0] = 3.0; q[1][1] = 4.0; print(q); return 0; }",
    );
}

#[test]
fn flat_declines_scalar_cast_leaving_ast_the_oracle() {
    // A scalar `as` cast is still outside the flat emitter's subset (the `Cast`
    // opcode lowers to the flat HIR, but the emitter declines it; see #214) -> the
    // flat path yields `None`, so the AST path stays the sole oracle (no false
    // parity claim). Becomes a parity case once #214 lands.
    let src = "fn main() -> i32 { let a = 7; return a as i64 as i32; }";
    assert!(flat_exit_code(src).is_none());
}
