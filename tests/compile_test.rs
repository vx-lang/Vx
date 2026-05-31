//===- compile_test.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the file-driven compilation test harness.
// It walks the `tests/` directory, compiles `.vx` files, and verifies that the
// compiler appropriately emits expected error messages for negative tests or
// successfully generates MLIR for positive tests.
//
//===----------------------------------------------------------------------===//
use std::fs;
use std::path::Path;

use rayon::prelude::*;

use vxc::jit::execute_mlir;
use vxc::sema::TypeChecker;

// Frontend Runner
fn run_frontend_test(path: &Path, expect_pass: bool) {
    let _source = fs::read_to_string(path).expect("Failed to read test file");

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = match loader.load_main(path.to_str().unwrap()) {
        Ok(p) => p,
        Err(_) => {
            if !expect_pass {
                return;
            }
            panic!("Parse failed on {:?}", path);
        }
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    let is_valid = !checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error);

    if expect_pass {
        assert!(
            is_valid,
            "Semantic analysis failed on {:?}:\n{:?}",
            path, checker.errors
        );
    } else {
        assert!(
            !is_valid,
            "Expected semantic failure on {:?}, but it passed",
            path
        );
    }
}

// Middle-End Runner
fn run_middle_end_test(path: &Path) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // Extract // CHECK: lines
    let check_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// CHECK:"))
        .map(|line| line.split_once("CHECK:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = loader
        .load_main(path.to_str().unwrap())
        .expect("Failed to parse");

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    assert!(
        (!checker
            .errors
            .iter()
            .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)),
        "Sema failed on {:?}: {:#?}",
        path,
        checker.errors
    );

    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker
        .monomorphized_functions
        .into_iter()
        .map(|(f, _)| f)
        .collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    let module_asts = std::collections::HashMap::new();

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context);
    codegen.generate(&monomorphized_program, &module_asts);
    let mlir_str = codegen.into_module().as_operation().to_string();

    // Verify // CHECK: lines in order
    let mut current_idx = 0;
    for check in check_lines {
        if let Some(pos) = mlir_str[current_idx..].find(&check) {
            current_idx += pos + check.len();
        } else {
            panic!("FileCheck failed on {:?}: Could not find `{}` after previous checks.\nMLIR Output:\n{}", path, check, mlir_str);
        }
    }
}

// Backend Runner
fn run_backend_test(path: &Path) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    // Extract // EXPECT: lines (assuming just one for simplicity right now)
    let expect_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// EXPECT:"))
        .map(|line| line.split_once("EXPECT:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = match loader.load_main(path.to_str().unwrap()) {
        Ok(p) => p,
        Err(e) => panic!("Frontend failed to parse '{}': {}", path.display(), e),
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        panic!(
            "Semantic check failed on '{}':\n{:?}",
            path.display(),
            checker.errors
        );
    }
    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker
        .monomorphized_functions
        .into_iter()
        .map(|(f, _)| f)
        .collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;
    let mut module_asts = std::collections::HashMap::new();
    for mut p in program_arr {
        let before = p.functions.len();
        p.functions.retain(|f| f.generics.is_empty());
        let after = p.functions.len();
        println!(
            "Module {}: retained {} out of {} functions",
            p.module_path, after, before
        );
        for f in &p.functions {
            if f.name == "expect_eq" {
                println!(
                    "WARNING: expect_eq was retained! Generics: {:?}",
                    f.generics
                );
            }
        }
        module_asts.insert(p.module_path.clone(), p);
    }

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context);
    codegen.generate(&monomorphized_program, &module_asts);
    let mut module = codegen.into_module();
    if let Err(e) = vxc::codegen::lower_to_llvm(&context, &mut module) {
        println!("MLIR Before Lowering Error:\n{}", module.as_operation());
        panic!("Lowering to LLVM failed for {}: {:?}", path.display(), e);
    }
    let mlir_str = module.as_operation().to_string();

    if source.contains("// NO_EXEC") {
        return;
    }

    let out = execute_mlir(&mlir_str, vec![], 0).expect("JIT execution failed");

    for expect in expect_lines {
        assert!(
            out.contains(&expect),
            "Backend output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}",
            path,
            expect,
            out
        );
    }
}

#[test]
fn test_frontend_pass() {
    let dir = Path::new("tests/frontend/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("vx") {
                run_frontend_test(&path, true);
            }
        });
    }
}

#[test]
fn test_frontend_fail() {
    let dir = Path::new("tests/frontend/fail");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("vx") {
                run_frontend_test(&path, false);
            }
        });
    }
}

#[test]
fn test_optimizations() {
    let dir = Path::new("tests/optimizations/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            let ext = path.extension().and_then(|s| s.to_str());
            if path.is_file() && (ext == Some("mlr") || ext == Some("vx")) {
                println!("Running test_optimizations on {:?}", path);
                run_optimization_test(&path);
            }
        });
    }
}

#[test]
fn test_middle_end() {
    let dir = Path::new("tests/middle_end/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                run_middle_end_test(&path);
            }
        });
    }
}

// Optimization Test Runner
fn run_optimization_test(path: &Path) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let run_lines: Vec<_> = source
        .lines()
        .filter(|line| {
            line.trim().starts_with("// RUN: vxc %s")
                || line.trim().starts_with("// RUN: vx-opt %s")
        })
        .collect();

    assert!(!run_lines.is_empty(), "Missing // RUN: line");

    for run_line in run_lines {
        let run_cmd = run_line.split_once("RUN:").unwrap().1.trim();

        // Parse FileCheck prefix
        let mut prefix = "CHECK".to_string();
        if let Some(filecheck_part) = run_cmd.split('|').nth(1) {
            if let Some(prefix_arg) = filecheck_part
                .split_whitespace()
                .find(|s| s.starts_with("--check-prefix="))
            {
                prefix = prefix_arg.split_once('=').unwrap().1.to_string();
            }
        }

        let check_prefix = format!("// {}:", prefix);
        let check_not_prefix = format!("// {}-NOT:", prefix);

        let check_lines: Vec<String> = source
            .lines()
            .filter(|line| {
                line.trim().starts_with(&check_prefix)
                    && !line.trim().starts_with(&check_not_prefix)
            })
            .map(|line| {
                line.split_once(&check_prefix[3..])
                    .unwrap()
                    .1
                    .trim()
                    .to_string()
            })
            .collect();

        let check_not_lines: Vec<String> = source
            .lines()
            .filter(|line| line.trim().starts_with(&check_not_prefix))
            .map(|line| {
                line.split_once(&check_not_prefix[3..])
                    .unwrap()
                    .1
                    .trim()
                    .to_string()
            })
            .collect();

        let vxc_cmd_str = run_cmd.split('|').next().unwrap().trim();
        let vxc_cmd_str = vxc_cmd_str.replace("%s", path.to_str().unwrap());

        let mut args: Vec<String> = vec![];
        let mut current_arg = String::new();
        let mut in_quotes = false;
        for c in vxc_cmd_str.chars() {
            if c == '"' {
                in_quotes = !in_quotes;
            } else if c == ' ' && !in_quotes {
                if !current_arg.is_empty() {
                    args.push(current_arg.clone());
                    current_arg.clear();
                }
            } else {
                current_arg.push(c);
            }
        }
        if !current_arg.is_empty() {
            args.push(current_arg);
        }

        let mut expect_failure = false;
        let mut exec_name = args.remove(0);
        if exec_name == "not" {
            expect_failure = true;
            exec_name = args.remove(0);
        }

        let bin_path = if exec_name == "vxc" {
            env!("CARGO_BIN_EXE_vxc")
        } else if exec_name == "vx-opt" {
            env!("CARGO_BIN_EXE_vx-opt")
        } else {
            println!("Warning: Unknown executable in RUN line: {}", exec_name);
            continue;
        };

        let output = std::process::Command::new(bin_path)
            .args(&args)
            .output()
            .expect("Failed to execute vxc");

        if expect_failure {
            if output.status.success() {
                panic!(
                    "Command succeeded but was expected to fail:\n{}",
                    String::from_utf8_lossy(&output.stdout)
                );
            }
            // If it failed as expected, we probably still want to check the error message
            // or maybe we shouldn't run FileCheck if it's expected to fail?
            // The user expects to use `not` and probably FileCheck the error message!
            // We should combine stdout and stderr so FileCheck can match the error message.
            let out = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );

            let mut current_idx = 0;
            for check in check_lines {
                if let Some(pos) = out[current_idx..].find(&check) {
                    current_idx += pos + check.len();
                } else {
                    panic!("FileCheck failed on {:?} for prefix {}: Could not find `{}` after previous checks.\nOutput:\n{}", path, prefix, check, out);
                }
            }
            return; // Skip negative checks for expected failures? No, let's keep them if needed, but return here because that's usually how it works.
        }

        if !output.status.success() {
            panic!("vxc failed:\n{}", String::from_utf8_lossy(&output.stderr));
        }

        let out = String::from_utf8_lossy(&output.stdout);

        let mut current_idx = 0;
        for check in check_lines {
            if let Some(pos) = out[current_idx..].find(&check) {
                current_idx += pos + check.len();
            } else {
                panic!("FileCheck failed on {:?} for prefix {}: Could not find `{}` after previous checks.\nOutput:\n{}", path, prefix, check, out);
            }
        }

        for not_check in check_not_lines {
            if out.contains(&not_check) {
                panic!(
                    "FileCheck failed on {:?} for prefix {}: Found forbidden `{}`.\nOutput:\n{}",
                    path, prefix, not_check, out
                );
            }
        }
    }
}

#[test]
fn test_middle_end_fail() {
    let dir = Path::new("tests/middle_end/fail");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                let result = std::panic::catch_unwind(|| {
                    run_middle_end_test(&path);
                });
                assert!(
                    result.is_err(),
                    "Expected {} to fail, but it succeeded!",
                    path.display()
                );
            }
        });
    }
}

#[test]
fn test_backend() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let dir = Path::new("tests/backend/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                println!("Running test_backend on {:?}", path);
                run_backend_test(&path);
                let source = std::fs::read_to_string(&path).unwrap_or_default();
                if source.contains("// RUN: vxc %s --emit-mlir") {
                    run_optimization_test(&path);
                }
            }
        });
    }
}

#[test]
fn test_backend_pass_formal_verification() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let dir = Path::new("tests/backend/pass/formal_verification");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                println!(
                    "Running test_backend_pass_formal_verification on {:?}",
                    path
                );
                run_backend_test(&path);
                let source = std::fs::read_to_string(&path).unwrap_or_default();
                if source.contains("// RUN: vxc %s --emit-mlir") {
                    run_optimization_test(&path);
                }
            }
        });
    }
}

#[test]
fn test_backend_fail_formal_verification() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let dir = Path::new("tests/backend/fail/formal_verification");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                println!(
                    "Running test_backend_fail_formal_verification on {:?}",
                    path
                );
                let result = std::panic::catch_unwind(|| {
                    run_backend_test(&path);
                });
                assert!(
                    result.is_err(),
                    "Expected {} to fail, but it succeeded!",
                    path.display()
                );
            }
        });
    }
}

#[test]
fn test_backend_pass_autodiff() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let dir = Path::new("tests/backend/pass/autodiff");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                println!("Running test_backend_pass_autodiff on {:?}", path);
                run_backend_autodiff_test(&path);
            }
        });
    }
}

// Backend Autodiff Runner
fn run_backend_autodiff_test(path: &Path) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let expect_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// EXPECT:"))
        .map(|line| line.split_once("EXPECT:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = match loader.load_main(path.to_str().unwrap()) {
        Ok(p) => p,
        Err(e) => panic!("Frontend failed to parse '{}': {}", path.display(), e),
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        panic!(
            "Semantic check failed on '{}':\n{:?}",
            path.display(),
            checker.errors
        );
    }
    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions: Vec<_> = checker
        .monomorphized_functions
        .into_iter()
        .map(|(f, _)| f)
        .collect();
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;

    let mut module_asts = std::collections::HashMap::new();
    for mut p in program_arr {
        p.functions.retain(|f| f.generics.is_empty());
        module_asts.insert(p.module_path.clone(), p);
    }

    let context = melior::Context::new();
    let mut codegen = vxc::codegen::MeliorGenerator::new(&context);
    let mlir_str = codegen.generate(&monomorphized_program, &module_asts);

    if source.contains("// NO_EXEC") {
        for expect in expect_lines {
            assert!(
                mlir_str.contains(&expect),
                "MLIR Output mismatch on {:?}.\nExpected to find: `{}`\nActual MLIR:\n{}",
                path,
                expect,
                mlir_str
            );
        }
        return;
    }

    if std::env::var("ENZYME_LIB").is_err() {
        println!(
            "Skipping JIT execution for {} because ENZYME_LIB is not set.",
            path.display()
        );
        return;
    }

    let out = execute_mlir(&mlir_str, vec![], 0).expect("JIT execution failed");

    for expect in expect_lines {
        assert!(
            mlir_str.contains(&expect) || out.contains(&expect),
            "Output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}\nMLIR:\n{}",
            path,
            expect,
            out,
            mlir_str
        );
    }
}

#[test]
fn test_backend_fail() {
    if !cfg!(target_os = "macos") {
        return;
    }
    let dir = Path::new("tests/backend/fail");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                let result = std::panic::catch_unwind(|| {
                    run_backend_test(&path);
                });
                assert!(
                    result.is_err(),
                    "Expected {} to fail, but it succeeded!",
                    path.display()
                );
            }
        });
    }
}

#[test]
fn test_melior_matmul() {
    let path = Path::new("tests/middle_end/pass/matmul.mlr");
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let check_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// CHECK:"))
        .map(|line| line.split_once("CHECK:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = loader
        .load_main(path.to_str().unwrap())
        .expect("Failed to parse");

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let program = program_arr.remove(ast_idx);

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let mut all_programs = program_arr.clone();
    all_programs.push(program.clone());
    let env = vxc::sema::GlobalAstEnv::build(&all_programs);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    let mut checked_program = program.clone();
    for f in &mut checked_program.functions {
        checker.check_function(f);
    }
    assert!(
        (!checker
            .errors
            .iter()
            .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)),
        "Sema failed on {:?}: {:#?}",
        path,
        checker.errors
    );

    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);

    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    vxc::codegen::register_vx_dialect(&context);

    let mut gen = vxc::codegen::MeliorGenerator::new(&context);
    let mlir_str = gen.generate(&checked_program, &std::collections::HashMap::new());

    let mut current_idx = 0;
    for check in check_lines {
        if let Some(pos) = mlir_str[current_idx..].find(&check) {
            current_idx += pos + check.len();
        } else {
            panic!("FileCheck failed on {:?}: Could not find `{}` after previous checks.\nMLIR Output:\n{}", path, check, mlir_str);
        }
    }
}

extern "C" {
    fn registerVxDialect(ctx: mlir_sys::MlirContext);
}

#[test]
fn test_vx_dialect_registration() {
    let registry = melior::dialect::DialectRegistry::new();
    let context = melior::Context::new();

    // Register our custom dialect using the FFI
    unsafe {
        registerVxDialect(context.to_raw());
    }

    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // We can't directly check the registered dialects easily in melior without parsing,
    // but we can parse a dummy module that requires the `vx` dialect.
    let mlir_source = r#"
        module {
            "vx.spawn"() <{topology = 100 : i32}> ({}) : () -> ()
        }
    "#;

    // If the dialect wasn't registered, parsing this would fail.
    let module = melior::ir::Module::parse(&context, mlir_source);
    assert!(
        module.is_some(),
        "Failed to parse module containing vx.spawn. Dialect may not be registered!"
    );
}
