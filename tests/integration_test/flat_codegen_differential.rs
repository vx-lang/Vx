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
    let mut program = parser.parse().expect("parse failed");
    // Expand macros (`print!`/`println!` -> `Expr::Print`/`Expr::Println`, plus any user macros)
    // before name resolution, exactly as the real pipeline does — otherwise a `MacroCall` reaches
    // resolution and panics.
    let mut global_macros = std::collections::HashMap::new();
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::syntax::MacroExpander::new(&global_macros);
    expander
        .expand_module(&mut program)
        .expect("macro expansion failed");
    program
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
    // Append the monomorphs the checker collected (method-call rewrites like `x.sq()` -> `f32$sq`,
    // generic instances) so their bodies emit and the rewritten calls resolve — as the driver does.
    for (f, _) in std::mem::take(&mut checker.monomorphized_functions) {
        program.functions.push(f);
    }

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

    // Type-check so the checker annotates each `StructInit` with its struct GID and collects
    // monomorphizations (method-call rewrites like `x.sq()` -> `f32$sq`, generic instances). A scratch
    // worker; the annotation lands on the AST, which the per-function lowering below then reads.
    let env_mods = mods.clone();
    let env = GlobalAstEnv::build(&env_mods);
    let monos = {
        let mut scratch = LocalWorkerState::new(session.clone());
        let mut checker = TypeChecker::new(&env, &mut scratch);
        for f in &mut mods[0].functions {
            checker.check_function(f);
        }
        checker.monomorphized_functions
    };
    if !monos.is_empty() {
        // Append the monomorph bodies, then re-resolve + rebuild the registry so they land in
        // `fn_sigs` (a rewritten `f32$sq(x)` resolves its callee) — mirroring the driver's flat path.
        for (f, _) in monos {
            mods[0].functions.push(f);
        }
        let symbol_map = vxc::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map);
    }
    let registry = vxc::pipeline::build_frozen_registry(&mods).ok()?;
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

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
    let string_tables: Vec<&[String]> = lowered
        .iter()
        .map(|w| w.local_string_table.as_slice())
        .collect();
    let body = vxc::codegen::flat::emit_module_mlir(
        &funcs,
        &session.registry,
        &tensor_types,
        &string_tables,
    )?;

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
fn flat_matches_ast_if_expression() {
    // A value-position `if` (`let m: i32 = if a > b { a } else { b }`, #201): lowered to blocks + a
    // result slot each branch stores into, then loaded. JIT-matches the AST oracle. This is the
    // `let m_new = if tm > m { tm } else { m }` shape from the attention corpus.
    assert_parity(
        "fn main() -> i32 { let a = 3; let b = 7; let m: i32 = if a > b { a } else { b }; return m; }",
        7,
    );
    // With side-effecting leading statements in a branch before the trailing value.
    assert_parity(
        "fn main() -> i32 { let x = 5; \
         let r: i32 = if x > 0 { let t = x * 2; t + 1 } else { 0 }; return r; }",
        11,
    );
}

#[test]
fn flat_matches_ast_assert_is_a_noop() {
    // `assert(cond, msg)` emits no runtime check in *either* path (the AST codegen treats it as a
    // compile-time seam fact), so a program with asserts JIT-matches — both ignore them.
    assert_parity(
        "fn main() -> i32 { let x = 5; assert(x == 5, \"ok\"); assert(x > 0, \"pos\"); return x; }",
        5,
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
fn flat_matches_ast_extern_call() {
    // An `extern` (libm `sqrtf`, linked by the JIT via `-lm`) now resolves + emits through the flat
    // path: registered in `fn_sigs`, called, and declared `func.func private` from the call's own
    // signature. `sqrtf(16.0) = 4.0`; compare the printed output flat-vs-AST.
    assert_output_parity(
        "extern { safe fn sqrtf(x: f32) -> f32; }\n\
         fn main() -> i32 { print(sqrtf(16.0)); return 0; }",
    );
}

#[test]
fn flat_matches_ast_scalar_method_call() {
    // Method dispatch through the flat path (#217): the type checker rewrites `(3.0).sq()` ->
    // `f32$sq(3.0)` and monomorphizes the body; the harness appends + registers the monomorph, so the
    // flat path lowers the caller + the `f32$sq` body together. `3*3 = 9`; output matches the AST path.
    assert_output_parity(
        "trait Sq { fn sq(self: Self) -> f32; }\n\
         impl Sq for f32 { fn sq(self: f32) -> f32 { return self * self; } }\n\
         fn main() -> i32 { print((3.0f32).sq()); return 0; }",
    );
}

#[test]
fn flat_matches_ast_unsafe_extern_call() {
    // The stdlib math-wrapper shape: a (non-safe) extern called inside an `unsafe { .. }` value block
    // (`impl Math for f32 { fn sqrt(self) { return unsafe { sqrtf(self) }; } }`). `unsafe` is
    // transparent to lowering, so both lower through the flat path; stdout matches the AST oracle.
    assert_output_parity(
        "extern { fn sqrtf(x: f32) -> f32; }\n\
         fn main() -> i32 { print(unsafe { sqrtf(16.0) }); return 0; }",
    );
}

#[test]
fn flat_matches_ast_print_string_literal() {
    // A bare string `print!` (the `print!("Success: ")` shape from the ffi corpus, #225): the flat
    // path records the bytes in a string side table, emits an `llvm.mlir.global` for them, and calls
    // `@print_str`. Stdout must match the AST oracle, which does the same.
    assert_output_parity("fn main() -> i32 { print!(\"Stdio Success: \"); return 0; }");
}

#[test]
fn flat_matches_ast_print_string_and_scalar() {
    // The common `print!("label", value)` shape: a string arg emits `PrintStr`, the scalar arg emits
    // `Print`, in order and with no separators — byte-identical to the AST path.
    assert_output_parity("fn main() -> i32 { let v = 42; print!(\"x=\", v); return 0; }");
}

#[test]
fn flat_matches_ast_println_string() {
    // `println!` prints its args then a trailing newline. The flat path reuses `PrintStr` for the
    // newline (a `"\n"` string is byte-identical to the AST path's `println()` runtime call), so the
    // full line — label, value, newline — matches the oracle.
    assert_output_parity("fn main() -> i32 { let v = 7; println!(\"count: \", v); return 0; }");
}

#[test]
fn flat_matches_ast_with_memory_method() {
    // `Tensor<..>(..).with_memory(Memory::X)` annotates a tensor's home memory for the seam/type
    // analysis but emits no op — the flat path lowers it as a transparent pass-through of the receiver
    // tensor, matching the AST codegen (#226). The device-placement transfer *methods*
    // (`to_device`/`to_host`) are already rewritten to `Expr::Transfer` by the type checker.
    assert_output_parity(
        "fn main() -> i32 { let mut c = Tensor<f32>([2]).with_memory(Memory::NPU_HBM); \
         c[0] = 3.0; c[1] = 4.0; print(c[0]); return 0; }",
    );
}

#[test]
fn flat_matches_ast_tensor_store_element_coercion() {
    // A default-`f32` float literal stored into a non-`f32` tensor is coerced to the element type at
    // the store (`bf16` -> `arith.truncf`), matching the AST's `coerce_type` before its `memref.store`
    // — without it the store is ill-typed. Read the stored values back via `as f32` and sum so the
    // (correctly truncated) value is observable: 4 × 1.5 = 6.0.
    assert_output_parity(
        "fn main() -> i32 { let mut a = Tensor<bf16>([4]); for i in 0..4 { a[i] = 1.5; } \
         let mut s = 0.0f32; for i in 0..4 { s = s + (a[i] as f32); } print(s); return 0; }",
    );
}

#[test]
fn flat_matches_ast_enum_construct_and_match() {
    // A payload-free (C-like) enum: `Dir::East` constructs the discriminant ordinal (a bare `i32`),
    // an enum-typed param passes it, and `match d { Dir::.. => .. }` is an eq-compare chain over the
    // ordinal (#227). `East`=2, `West`=3 -> `code` returns 3, 4 -> 3 + 4 = 7.
    assert_parity(
        "enum Dir { North, South, East, West }\n\
         fn code(d: Dir) -> i32 { let mut r = 0; \
           match d { Dir::North => { r = 1; } Dir::South => { r = 2; } \
                     Dir::East => { r = 3; } Dir::West => { r = 4; } } return r; }\n\
         fn main() -> i32 { return code(Dir::East) + code(Dir::West); }",
        7,
    );
}

#[test]
fn flat_matches_ast_enum_match_wildcard() {
    // The wildcard arm is the unconditional default: `Color::Green` matches neither listed arm, so
    // it falls through to `_ => 99`.
    assert_parity(
        "enum Color { Red, Green, Blue }\n\
         fn pick(c: Color) -> i32 { let mut r = 0; \
           match c { Color::Red => { r = 10; } Color::Blue => { r = 30; } _ => { r = 99; } } return r; }\n\
         fn main() -> i32 { return pick(Color::Green); }",
        99,
    );
}

#[test]
fn flat_matches_ast_sizeof_value() {
    // `sizeof<T>()` folds to a compile-time `i64` constant of `T`'s byte size, matching the AST
    // codegen (#228). `sizeof<f64>()` = 8, `sizeof<i8>()` = 1 -> 9.
    assert_parity(
        "fn main() -> i32 { let s: i64 = sizeof<f64>(); let t: i64 = sizeof<i8>(); \
         return (s + t) as i32; }",
        9,
    );
}

#[test]
fn flat_matches_ast_comptime_block() {
    // A `comptime { .. }` block lowers transparently — its `sizeof` folds to a constant and its
    // `assert` is a runtime no-op, so at runtime it has no observable effect and `main` returns 0
    // (#228). Matches the AST codegen, which lowers the block the same way.
    assert_parity(
        "fn main() -> i32 { comptime { let s: i64 = sizeof<f64>(); assert(s == 8); } return 0; }",
        0,
    );
}

#[test]
fn flat_matches_ast_return_widening_coercion() {
    // `return 7` — a default-`i32` literal — from an `-> i64` function coerces to `i64` at the
    // return (`arith.extsi`), matching the AST's `coerce_type`. Without it the `func.return` type
    // contradicts the signature. `wide()` = 7.
    assert_parity(
        "fn wide() -> i64 { return 7; } fn main() -> i32 { return wide() as i32; }",
        7,
    );
}

#[test]
fn flat_matches_ast_borrow_tensor_print() {
    // `print(&t)` where `t` is a tensor (`npu_lowering_execution.vx` shape, #230): a tensor is a
    // memref — already a reference — so the borrow `&t` is transparent (yields the tensor itself),
    // matching the AST codegen. The `printMemref` dump (shape/strides/data) matches the oracle.
    assert_output_parity(
        "fn main() -> i32 { let mut a = Tensor<f32>([2]); a[0] = 1.0; a[1] = 2.0; \
         print(&a); return 0; }",
    );
}

#[test]
fn flat_matches_ast_nested_value_if() {
    // A value-`if` whose then-branch is itself a value-`if` (`expr_assignment.vx` shape, #229): the
    // inner if lowers through `lower_expr`'s `Expr::If` arm (its result type inferred from the branch
    // value) into the outer slot. `x=25` -> `25>10` -> `25>20` -> 100.
    assert_parity(
        "fn nested(x: i32) -> i32 { let val: i32 = if x > 10 { if x > 20 { 100 } else { 50 } } \
         else { 0 }; return val; }\n\
         fn main() -> i32 { return nested(25); }",
        100,
    );
}

#[test]
fn flat_matches_ast_value_if_implicit_return() {
    // A value-`if` as a function's implicit return (no annotation): the parser rewrites the trailing
    // `if` to `return if ..`, which lowers through the `Expr::If` arm. `implicit(3)` -> `3>10` false
    // -> `3 + 9` = 12.
    assert_parity(
        "fn implicit(x: i32) -> i32 { if x > 10 { x + 5 } else { x + 9 } }\n\
         fn main() -> i32 { return implicit(3); }",
        12,
    );
}

#[test]
fn flat_matches_ast_value_if_call_argument() {
    // A value-`if` nested as a call argument, in a function with no other control flow (so it lowers
    // in pure-SSA mode — the `if`'s blocks are self-contained). `5 > 2` -> 42.
    assert_parity(
        "fn id(x: i32) -> i32 { x }\n\
         fn main() -> i32 { return id(if 5 > 2 { 42 } else { 7 }); }",
        42,
    );
}

#[test]
fn flat_matches_ast_value_if_else_if_compound_assign() {
    // An `else if` chain as a compound-assign RHS: the `else` branch's trailing value is itself an
    // `if`, lowered through the `Expr::If` arm into the outer slot. `x=15` -> not `>20`, is `>10` ->
    // 50; `r = 1 + 50` = 51.
    assert_parity(
        "fn main() -> i32 { let x = 15; let mut r = 1; \
         r += if x > 20 { 100 } else if x > 10 { 50 } else { 0 }; return r; }",
        51,
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
fn flat_matches_ast_struct_return() {
    // A function returns a struct by value (#215): the callee builds it in a slot, loads the
    // `!llvm.struct` and returns it; the caller spills the returned value to a slot and reads fields.
    assert_parity(
        "struct P { x: i32, y: i32 }\n\
         fn mk() -> P { return P { x: 3, y: 4 }; }\n\
         fn main() -> i32 { let p = mk(); return p.x + p.y; }",
        7,
    );
}

#[test]
fn flat_matches_ast_struct_return_used_directly() {
    // Return a struct and read a field off the call result without an intermediate binding.
    assert_parity(
        "struct P { x: i32, y: i32 }\n\
         fn mk() -> P { return P { x: 10, y: 5 }; }\n\
         fn main() -> i32 { let p = mk(); return p.x - p.y; }",
        5,
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
fn flat_matches_ast_integer_negation() {
    // `-x` on an integer (#214): the emitter now lowers `Neg` (`0 - x`), so it JIT-matches the AST.
    assert_parity(
        "fn main() -> i32 { let a = 5; let b = -a; return b + 12; }",
        7,
    );
}

#[test]
fn flat_matches_ast_float_negation() {
    // `-x` on a float now lowers (`arith.negf`, #214); the negated value is printed for stdout parity.
    assert_output_parity("fn main() -> i32 { let x = 3.0f32; print(-x); return 0; }");
}

#[test]
fn flat_matches_ast_scalar_casts() {
    // Scalar `as` casts now lower + emit through the flat path (#214): a chained widen/narrow, and an
    // int->float->int round trip, both JIT-match the AST oracle.
    assert_parity("fn main() -> i32 { let a = 7; return a as i64 as i32; }", 7);
    assert_parity(
        "fn main() -> i32 { let a = 100; let f = a as f32; return f as i32; }",
        100,
    );
}

/// Compile a "library" module to a serialized `.vxlib` interface (frozen registry + flat-HIR bodies).
fn build_lib_interface(path: &str, src: &str) -> Vec<u8> {
    let mut prog = parse(src);
    prog.module_path = path.into();
    let mut mods = vec![prog];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map);
    vxc::pipeline::emit_module_interface(&mods).expect("emit module interface")
}

/// The stdlib-decoupling endgame in miniature (#220 stage 4): a program links an imported function's
/// **body from a precompiled `.vxlib` artifact**, with the library's source *never parsed* in the
/// consumer compile. Proves the whole mechanism end to end through the flat path — producer (harvest +
/// serialize) → deserialize → registry merge → flat codegen links `body_of` → JIT.
#[test]
fn program_links_a_function_body_from_a_vxlib_artifact() {
    use vxc::syntax::{ElementType, Function, Topology, Type};

    // 1. Producer: a library module -> a `.vxlib` interface (bytes). This is the only place its
    //    source is ever seen; the consumer below works purely from these bytes.
    let lib_bytes = build_lib_interface(
        "crate::mathlib",
        "fn double(x: i32) -> i32 { return x * 2; }",
    );

    // 2. Consumer: a program that CALLS `double`, compiled with no access to the library's AST.
    let mut app = parse("fn main() -> i32 { return double(21); }");
    app.module_path = "crate::app".into();
    let mut mods = vec![app];
    let symbol_map = vxc::resolver::build_symbol_map(&mods);
    mods[0].resolve_names(&symbol_map);

    // Fold the precompiled library interface into the app's frozen registry (no parse of the lib).
    let mut registry = vxc::pipeline::build_frozen_registry(&mods).expect("app registry");
    let lib = vxc::metadata::deserialize_registry_interface(&lib_bytes).expect("deserialize lib");
    registry.merge_from(lib);
    let session = std::sync::Arc::new(GlobalSession::with_registry(1, registry));

    // Lower the app's own functions -- `double(21)` resolves via the merged `fn_sigs`.
    let mut lowered = Vec::new();
    for f in &mods[0].functions {
        let mut worker = LocalWorkerState::new(session.clone());
        assert!(
            lower_function_to_hir(f, &mut worker),
            "app fn lowers to flat HIR"
        );
        lowered.push(worker);
    }

    // Pull `double`'s body from the artifact and give it a signature-only `Function` to emit against.
    let double_gid = session
        .registry
        .fn_sigs
        .get(&vxc::symbol::Symbol::from("double"))
        .expect("double resolves from the merged interface")
        .gid;
    let body = session
        .registry
        .body_of(double_gid)
        .expect("double's body came from the .vxlib artifact")
        .clone();
    let synth = Function {
        name: body.name.clone(),
        generics: vec![],
        params: body
            .params
            .iter()
            .enumerate()
            .map(|(i, t)| {
                (
                    vxc::symbol::Symbol::from(format!("a{i}").as_str()),
                    t.clone(),
                )
            })
            .collect(),
        topology: Topology::CPU,
        return_type: body.ret_ty.clone(),
        requires: vec![],
        ensures: vec![],
        where_transfers: vec![],
        body: vec![],
        doc_comment: None,
    };
    assert_eq!(synth.return_type, Type::Scalar(ElementType::I32));

    // Emit one module: the app's `main` + the imported `double` (body from the artifact).
    let mut funcs: Vec<(&Function, &[_], &[_])> = mods[0]
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
    funcs.push((&synth, body.hir.as_slice(), body.types.as_slice()));

    let mlir = vxc::codegen::flat::emit_module_mlir(&funcs, &session.registry, &[], &[])
        .expect("flat codegen emits the linked module");
    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &format!("module {{\n{mlir}}}\n"))
        .expect("linked flat MLIR parses");
    lower_to_llvm(&context, &mut module).expect("flat lower_to_llvm");
    assert_eq!(
        exit_code(&module.as_operation().to_string()),
        42,
        "double(21) linked from the .vxlib artifact returns 42"
    );
}

#[test]
fn flat_matches_ast_string_value_pointer_arg() {
    // A string literal in *value* position (#231): bound to a local, then passed as an `!llvm.ptr`
    // argument to a pointer-typed parameter — the shape of `vx_stdout_write(msg, 14)`. The callee
    // ignores the pointer and returns the length, so the exit code is observable without a deref or a
    // runtime symbol (self-contained). Exercises `StringConst` + pointer param + pointer arg.
    assert_parity(
        "fn take(p: *const u8, n: i32) -> i32 { return n; }\n\
         fn main() -> i32 { let msg = \"Hello!\\n\"; let r = take(msg, 14); return r; }",
        14,
    );
}

#[test]
fn flat_matches_ast_pointer_return_and_memory_slot() {
    // The FFI pointer ABI (#235): a pointer-*returning* function (like an FFI allocator), a pointer
    // local that must survive across basic blocks (the `if` forces the memory model, so the pointer
    // lives in an `llvm.alloca` slot), and a pointer argument read back from that slot. All Vx-defined,
    // so no external symbol is linked — the exit code alone proves the pointer threaded through.
    assert_parity(
        "fn make(x: *const u8) -> *const u8 { return x; }\n\
         fn take(p: *const u8) -> i32 { return 7; }\n\
         fn main() -> i32 {\n\
         let s = \"hi\";\n\
         let p = make(s);\n\
         let mut r = 0;\n\
         if r == 0 { r = take(p); }\n\
         return r;\n\
         }",
        7,
    );
}

#[test]
fn flat_matches_ast_corpus_ffi_stdio_strings() {
    // The #231 corpus driver: string *values* bound to locals and passed to the `vx_stdout_write`/
    // `vx_stderr_write` FFI externs (`!llvm.ptr` arguments), plus print-position strings (#225). Its
    // JIT stdout must match the AST oracle byte for byte.
    assert_output_parity(&corpus("ffi_stdio.vx"));
}

#[test]
fn flat_matches_ast_corpus_ffi_option() {
    // The #235 corpus driver: FFI externs returning/taking `*mut i8` (`!llvm.ptr`), a pointer local
    // stored across an `if` (memory-mode pointer slot), and a `Bool`-returning extern as the `if`
    // condition. Its JIT stdout must match the AST oracle.
    assert_output_parity(&corpus("ffi_option.vx"));
}

#[test]
fn flat_coerces_int_literal_argument_to_wider_param() {
    // #236: a default-`i32` literal passed to an `i64` parameter. The type checker records the
    // coercion on the literal (born as `i64`), so the flat `func.call` operand is `i64` and matches
    // the callee — before this, the flat path emitted an `i32` arg and failed MLIR verification.
    assert_parity(
        "fn wants_i64(n: i64) -> i64 { return n; }\n\
         fn main() -> i32 { return wants_i64(42) as i32; }",
        42,
    );
}

#[test]
fn flat_coerces_nonliteral_int_argument_to_wider_param() {
    // #236: a non-literal argument (an `i32` local) passed to an `i64` parameter. The checker can't
    // re-type the identifier, so it wraps it in an `as` cast — exercising the `AsCast` path (which
    // both backends lower identically). Parity with the AST oracle confirms no divergence.
    assert_parity(
        "fn wants_i64(n: i64) -> i64 { return n; }\n\
         fn main() -> i32 { let x = 7; let r = wants_i64(x); return r as i32; }",
        7,
    );
}

#[test]
fn flat_coerces_float_argument_to_wider_param() {
    // #236: an `f32` local widened to an `f64` parameter (`arith.extf`), the float analogue of the
    // integer cases. The coerced value flows through and truncates back to the `i32` exit code.
    assert_parity(
        "fn wants_f64(x: f64) -> f64 { return x; }\n\
         fn main() -> i32 { let a: f32 = 2.0; return wants_f64(a) as i32; }",
        2,
    );
}

/// Read a corpus program from `tests/backend/pass/`. The `RUN`/`CHECK`/`EXPECT`
/// and license lines are `//` comments the parser ignores.
fn corpus(name: &str) -> String {
    std::fs::read_to_string(format!("tests/backend/pass/{name}"))
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
}

#[test]
fn flat_matches_ast_corpus_slice_reductions() {
    // A real corpus program end to end through the flat path: tensor alloc +
    // scalar-element stores + typed row bindings + dot/sum/max/min reductions + a
    // `for`-loop scalar oracle + scalar-element stores of the results + `print(o)`.
    // The printed output must match the AST oracle.
    assert_output_parity(&corpus("slice_reductions.vx"));
}

#[test]
fn flat_matches_ast_corpus_linear_attention() {
    // An attention-corpus program (no softmax/exp): tensor allocs, `for` loops,
    // and `print`. Its printed output must match the AST oracle through the flat
    // path.
    assert_output_parity(&corpus("linear_attention.vx"));
}
