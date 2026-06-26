//===- cargo-vx-bench.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::env;
use std::fs;
use vxc::lexer::Lexer;
use vxc::parser;

fn run_benchmark(path: &std::path::Path) -> Result<(String, f32)> {
    let file_name = path.file_name().unwrap().to_str().unwrap().to_string();
    let source = fs::read_to_string(path).context("Failed to read benchmark file")?;

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();
    let mut parser = parser::Parser::new(&tokens, &source);

    let mut ast = parser
        .parse()
        .map_err(|e| anyhow::anyhow!("Parse Error: {}", e.format(&source)))?;

    let mut has_main = false;
    for func in &mut ast.functions {
        if func.name.as_ref() == "main" {
            func.name = std::sync::Arc::from("__user_main");
            has_main = true;
            break;
        }
    }

    if has_main {
        let harness_code = "
extern {
    fn vx_get_time() -> f32;
    fn vx_print_float(val: f32) -> i32;
}
fn main() -> i32 {
    unsafe {
        let __start = vx_get_time();
        let _ = __user_main();
        let __end = vx_get_time();
        let _ = vx_print_float(__end - __start);
    }
    return 0;
}
";
        let mut lexer2 = Lexer::new(harness_code);
        let tokens2 = lexer2.tokenize();
        let mut parser2 = parser::Parser::new(&tokens2, harness_code);
        let harness_ast = parser2.parse().unwrap();

        ast.functions.extend(harness_ast.functions);
        ast.externs.extend(harness_ast.externs);
    }

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [ast.clone()];
    let env = vxc::hir::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = vxc::hir::TypeChecker::new(&env, &mut worker);
    for f in &mut ast.functions {
        checker.check_function(f);
    }

    if !checker.errors.is_empty() {
        let mut err_msg = String::new();
        for err in checker.errors {
            err_msg.push_str(&format!(" - {}\n", err));
        }
        anyhow::bail!("Semantic Errors:\n{}", err_msg);
    }

    let monomorphized_ast = ast;
    let module_syntaxes = std::collections::HashMap::new();
    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, file_name.clone());
    let _ = codegen.generate(&monomorphized_ast, &module_syntaxes);
    let mut module = codegen.into_module();
    vxc::codegen::lower_to_llvm(&context, &mut module)
        .map_err(|e| anyhow::anyhow!("Lowering Error: {:?}", e))?;
    let mlir_str = format!("{}", module.as_operation());

    let output = vxc::jit::execute_mlir(&mlir_str, vec![], 3, false)
        .map_err(|e| anyhow::anyhow!("Execution Error: {}", e))?;

    let re_time = regex::Regex::new(r"\[([0-9]+\.[0-9]+)\]").unwrap();
    if let Some(last_match) = re_time.captures_iter(&output).last() {
        let time_f = last_match[1].parse::<f32>().unwrap();
        Ok((file_name, time_f))
    } else {
        anyhow::bail!("No timing output found. Raw output:\n{}", output);
    }
}

fn main() -> Result<()> {
    println!("=====================================");
    println!("       Vx Benchmark Runner           ");
    println!("=====================================\n");

    let benchmarks_dir = env::current_dir()?.join("benchmarks");

    if !benchmarks_dir.exists() {
        anyhow::bail!("Error: 'benchmarks' directory not found.");
    }

    let entries = fs::read_dir(&benchmarks_dir)?;
    let mut paths = Vec::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|ext| ext == "vx") {
            paths.push(path);
        }
    }

    let results: Vec<Result<(String, f32)>> = paths
        .par_iter()
        .map(|path| {
            let file_name = path.file_name().unwrap().to_str().unwrap();
            println!("▶ Benchmarking {:<30}", file_name);

            let res = run_benchmark(path);
            match &res {
                Ok((_, time_f)) => println!("{} -> {:.4}s", file_name, time_f),
                Err(e) => println!("{} -> FAILED: {}", file_name, e),
            }
            res
        })
        .collect();

    println!("\n=====================================");
    println!("             Summary                 ");
    println!("=====================================");
    let mut success_results: Vec<(String, f32)> =
        results.into_iter().filter_map(|r| r.ok()).collect();
    success_results.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, time) in &success_results {
        println!("{:<30} {:.4}s", name, time);
    }

    Ok(())
}
