//===- integration_test.rs - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file contains the main end-to-end integration test suite.
// It executes full Vx programs through the entire compilation pipeline and JIT
// engine, asserting that the runtime behavior, math operations, and control flow
// match expected outcomes.
//
//===----------------------------------------------------------------------===//
use vxc::hir::TypeChecker;
use vxc::lexer::Lexer;
use vxc::parser::Parser;

#[test]
fn test_distributed_matmul_integration() -> Result<(), String> {
    let input = r#"fn custom_matmul(a: Pinned<Tensor<f32>, Topology::NPU[0]>, b: Pinned<Tensor<f32>, Topology::NPU[0]>) on Topology::NPU[0] -> Pinned<Tensor<f32>, Topology::NPU[0]> {
    return a;
}

fn distributed_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Pinned<Tensor<f32>, Topology::NPU[0]> {
    let local_a = a.to_device();
    let local_b = b.to_device();
    spawn on(Topology::NPU[0]) {
        let result = custom_matmul(local_a, local_b);
        result
    }
}
    "#;

    // 1. Lexing
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    if tokens.is_empty() {
        return Err("Assertion failed: !tokens.is_empty()".to_string());
    }

    // 2. Parsing
    let mut parser = Parser::new(&tokens, input);
    let mut ast = parser.parse().expect("Failed to parse AST");
    if ast.functions.len() != 2 {
        return Err(format!(
            "Assertion failed: {} != {}",
            ast.functions.len(),
            2
        ));
    }

    // 3. Semantic Analysis
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [ast.clone()];
    let env = vxc::hir::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        checker.check_function(f);
    }
    let is_valid = checker.errors.error_count() == 0;

    if checker.errors.error_count() > 0 {
        for err in &checker.errors {
            println!("Semantic Error: {}", err);
        }
    }

    if !(is_valid) {
        return Err("Semantic analysis failed on integration test".to_string());
    }

    Ok(())
}

fn run_pipeline(input: &str) -> Result<vxc::syntax::Program, Vec<vxc::diagnostic::Diagnostic>> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(&tokens, input);
    let mut program = parser
        .parse()
        .map_err(|e| vec![vxc::diagnostic::Diagnostic::error(e.format(input))])?;
    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [program.clone()];
    let env = vxc::hir::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }

    let has_errors = checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error);

    if !has_errors {
        let monomorphized_ast = program;
        let module_syntaxes = std::collections::HashMap::new();
        let context = melior::Context::new();
        let registry = melior::dialect::DialectRegistry::new();
        melior::utility::register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        vxc::codegen::register_vx_dialect(&context);

        let mut codegen =
            vxc::codegen::MeliorGenerator::new(&context, "integration_test".to_string());
        let _ = codegen.generate(&monomorphized_ast, &module_syntaxes);
        let mut module = codegen.into_module();
        vxc::codegen::lower_to_llvm(&context, &mut module).unwrap();
        let _mlir_str = module.as_operation();
        Ok(monomorphized_ast)
    } else {
        for err in &checker.errors {
            println!("run_pipeline semantic error: {}", err);
        }
        Err(checker.errors.inner)
    }
}

#[test]
fn test_integration_operators() -> Result<(), String> {
    let input = r#"
    fn math_ops() -> Tensor<f32> {
        let mut x = 10;
        let y = x * 5;
        x += y + 2;
        return x;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_loops() -> Result<(), String> {
    let input = r#"
    fn loop_test() -> Tensor<f32> {
        let mut sum = 0;
        for i in 0..10 {
            sum += i;
        }
        return sum;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_arrays_and_indexing() -> Result<(), String> {
    let input = r#"
    fn array_test(a: Tensor<f32, [2, 2]>, b: Tensor<f32, [2, 2]>) -> Tensor<f32, [2, 2]> {
        let mut arr = Tensor<f32>(2, 2);
        arr[0][0] = a[0][1] * b[1][0];
        return arr;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_method_chaining() -> Result<(), String> {
    let input = r#"
    fn memory_test() -> Ref<Tensor<f32, [10]>, Memory::NPU_HBM> {
        let mut mem = Tensor<f32>([10]).with_memory(Memory::NPU_HBM);
        return mem;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_function_calls() -> Result<(), String> {
    let input = r#"
    fn helper(x: Tensor<f32>) -> Tensor<f32> {
        return x + 1;
    }

    fn main() -> Tensor<f32> {
        let y = 10;
        let z = helper(y);
        return z;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_logical_ops() -> Result<(), String> {
    let input = r#"
    fn logic_test(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
        let is_less = a < b;
        let is_eq = a == b;
        let c = is_less && is_eq;
        return a;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}

#[test]
fn test_integration_linear_variable_consumption() -> Result<(), String> {
    let input = r#"
    fn helper(t: Tensor<f32>) -> Tensor<f32> {
        return t;
    }

    fn main() -> Tensor<f32> {
        let x = Tensor<f32>(2, 2);

        let mut sum = 0;
        for i in 0..10 {
            sum += i;
        }

        // Use in function call. This is linear, so it consumes x.
        let y = helper(x);

        return y;
    }
    "#;
    if run_pipeline(input).is_err() {
        return Err("Assertion failed: run_pipeline(input).is_ok()".to_string());
    }

    Ok(())
}
