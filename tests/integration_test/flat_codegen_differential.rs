//===- flat_codegen_differential.rs - Vx Compiler --------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
// C2 (#200): differential testing of the *flat* codegen path against the AST
// path (the oracle). For a function the flat HIR lowers today (straight-line
// scalar arithmetic), the flat path (local_hir_stream -> `flat::emit_function_
// mlir`) must produce MLIR that, run through the *same* production lowering
// (`lower_to_llvm`) and JIT, yields the *same* result as the AST path. Parity
// here is what lets `vxc` eventually flip to the flat path (C3). Coverage grows
// as C1/C2 widen (control flow, calls, then the non-scalar surface #199).
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

/// Exit code of `main` compiled through the AST codegen (the oracle).
fn ast_exit_code(src: &str) -> i32 {
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
    exit_code(&module.as_operation().to_string())
}

/// Exit code of `main` compiled through the *flat* path, or `None` if the flat
/// HIR / emitter declines the function (outside the current subset).
fn flat_exit_code(src: &str) -> Option<i32> {
    let program = parse(src);
    let main = program
        .functions
        .iter()
        .find(|f| f.name.as_ref() == "main")?;

    let mut worker = LocalWorkerState::new(std::sync::Arc::new(GlobalSession::new(1)));
    if !lower_function_to_hir(main, &mut worker) {
        return None;
    }
    let body = vxc::codegen::flat::emit_function_mlir(
        main,
        &worker.local_hir_stream,
        &worker.local_type_stream,
    )?;

    let context = make_context();
    let mut module = melior::ir::Module::parse(&context, &format!("module {{\n{body}}}\n"))
        .expect("flat MLIR parses");
    lower_to_llvm(&context, &mut module).expect("flat lower_to_llvm");
    Some(exit_code(&module.as_operation().to_string()))
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
fn flat_declines_scalar_cast_leaving_ast_the_oracle() {
    // A scalar `as` cast is still outside the flat emitter's subset (the `Cast`
    // opcode lowers to the flat HIR, but the emitter declines it) -> the flat path
    // yields `None`, so the AST path stays the sole oracle (no false parity claim).
    let src = "fn main() -> i32 { let a = 7; return a as i64 as i32; }";
    assert!(flat_exit_code(src).is_none());
}
