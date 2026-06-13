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
fn run_frontend_test(path: &Path, expect_pass: bool) -> Result<(), String> {
    let _source = fs::read_to_string(path).expect("Failed to read test file");

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = match loader.load_main(path.to_str().unwrap()) {
        Ok(p) => p,
        Err(e) => {
            if !expect_pass {
                return Ok(());
            }
            return Err(format!("Parse failed on {:?}: {}", path, e));
        }
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::ast::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            if !expect_pass {
                return Ok(());
            }
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        if !expect_pass {
            return Ok(());
        }
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

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
        if !is_valid {
            return Err(format!(
                "Semantic analysis failed on {:?}:\n{:#?}",
                path, checker.errors
            ));
        }
    } else {
        if is_valid {
            return Err(format!(
                "Expected semantic failure on {:?}, but it passed",
                path
            ));
        }
    }
    Ok(())
}

// Middle-End Runner
fn run_middle_end_test(path: &Path) -> Result<(), String> {
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

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::ast::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

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
        return Err(format!("Sema failed on {:?}: {:#?}", path, checker.errors));
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

    let context = melior::Context::new();
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let module_asts = std::collections::HashMap::new();

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    codegen
        .generate(&monomorphized_program, &module_asts)
        .unwrap();
    let mlir_str = codegen.into_module().as_operation().to_string();

    // Verify // CHECK: lines in order
    let mut current_idx = 0;
    for check in check_lines {
        if let Some(pos) = mlir_str[current_idx..].find(&check) {
            current_idx += pos + check.len();
        } else {
            return Err(format!("FileCheck failed on {:?}: Could not find `{}` after previous checks.\nMLIR Output:\n{}", path, check, mlir_str));
        }
    }
    Ok(())
}

// Backend Runner
fn run_backend_test(path: &Path) -> Result<(), String> {
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
        Err(e) => {
            return Err(format!(
                "Frontend failed to parse '{}': {}",
                path.display(),
                e
            ))
        }
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::ast::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

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
        return Err(format!(
            "Semantic check failed on '{}':\n{:#?}",
            path.display(),
            checker.errors
        ));
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
    monomorphized_program
        .structs
        .extend(checker.generated_structs);
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
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    codegen
        .generate(&monomorphized_program, &module_asts)
        .unwrap();
    let mut module = codegen.into_module();
    if let Err(e) = vxc::codegen::lower_to_llvm(&context, &mut module) {
        println!("MLIR Before Lowering Error:\n{}", module.as_operation());
        return Err(format!(
            "Lowering to LLVM failed for {}: {:?}",
            path.display(),
            e
        ));
    }
    let mlir_str = module.as_operation().to_string();

    if source.contains("// NO_EXEC") {
        return Ok(());
    }

    if source.contains("// REQUIRES: macos") && !cfg!(target_os = "macos") {
        return Ok(());
    }

    let out = execute_mlir(&mlir_str, vec![], 0, false).expect("JIT execution failed");

    for expect in expect_lines {
        if !out.contains(&expect) {
            return Err(format!(
                "Backend output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}",
                path, expect, out
            ));
        }
    }
    Ok(())
}

#[test]
fn test_frontend_pass() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    if let Err(e) = run_frontend_test(&path, true) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

#[test]
fn test_frontend_fail() -> Result<(), String> {
    let has_z3 = std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_ok();
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail"),
        |path| {
            let path_str = path.to_string_lossy();
            if !has_z3
                && (path_str.contains("formal_verification")
                    || path_str.contains("unimplemented_smt"))
            {
                return Err("z3 is not installed".to_string());
            }
            run_shell_tests(path)
        },
    )
}

#[test]
fn test_optimizations() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/optimizations/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                let ext = path.extension().and_then(|s| s.to_str());
                if path.is_file() && (ext == Some("mlr") || ext == Some("vx")) {
                    println!("Running test_optimizations on {:?}", path);
                    if let Err(e) = run_optimization_test(&path) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

#[test]
fn test_middle_end() -> Result<(), String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    if let Err(e) = run_middle_end_test(&path) {
                        return Some(e);
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

// Optimization Test Runner
fn run_optimization_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let run_lines: Vec<_> = source
        .lines()
        .filter(|line| {
            line.trim().starts_with("// RUN: vxc %s")
                || line.trim().starts_with("// RUN: not vxc %s")
                || line.trim().starts_with("// RUN: vx-opt %s")
        })
        .collect();

    if run_lines.is_empty() {
        return Err("Missing // RUN: line".to_string());
    }

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

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&path, &mut hasher);
        let hash = std::hash::Hasher::finish(&hasher);

        // Ensure temporary files are completely contained within the project repository
        let tmp_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_tmp");
        std::fs::create_dir_all(&tmp_dir).unwrap_or_default();
        let t_val = tmp_dir.join(format!(
            "vxc_test_{}_{:x}",
            path.file_stem().unwrap().to_string_lossy(),
            hash
        ));

        let vxc_cmd_str = run_cmd.split('|').next().unwrap().trim();
        let vxc_cmd_str = vxc_cmd_str
            .replace("%s", path.to_str().unwrap())
            .replace("%t", t_val.to_str().unwrap());

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
            .env("RUST_BACKTRACE", "1")
            .output()
            .expect("Failed to execute vxc");

        if expect_failure {
            if output.status.success() {
                return Err(format!(
                    "Command succeeded but was expected to fail:\n{}",
                    String::from_utf8_lossy(&output.stdout)
                ));
            }
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
                    return Err(format!("FileCheck failed on {:?} for prefix {}: Could not find `{}` after previous checks.\nOutput:\n{}", path, prefix, check, out));
                }
            }
            continue;
        }

        if !output.status.success() {
            return Err(format!(
                "vxc failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

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
                return Err(format!("FileCheck failed on {:?} for prefix {}: Could not find `{}` after previous checks.\nOutput:\n{}", path, prefix, check, out));
            }
        }

        for not_check in check_not_lines {
            if out.contains(&not_check) {
                return Err(format!(
                    "FileCheck failed on {:?} for prefix {}: Found forbidden `{}`.\nOutput:\n{}",
                    path, prefix, not_check, out
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn test_middle_end_fail() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/fail"),
        |path| {
            if run_middle_end_test(path).is_ok() {
                return Err(format!(
                    "Expected {} to fail, but it succeeded!",
                    path.display()
                ));
            }
            Ok(())
        },
    )
}

#[test]
fn test_backend() -> Result<(), String> {
    std::env::set_var("RUST_BACKTRACE", "1");
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/pass"),
        |path| {
            println!("Running test_backend on {:?}", path);
            run_backend_test(path)?;
            let source = std::fs::read_to_string(path).unwrap_or_default();
            if source.contains("// RUN: vxc %s --emit-mlir") {
                run_optimization_test(path)?;
            }
            Ok(())
        },
    )
}

#[test]
fn test_frontend_pass_formal_verification() -> Result<(), String> {
    if std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err("z3 is not installed".to_string());
    }
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/pass/formal_verification"),
        |path| {
            println!(
                "Running test_frontend_pass_formal_verification on {:?}",
                path
            );
            run_frontend_test(path, true)
        },
    )
}

#[test]
fn test_frontend_fail_formal_verification() -> Result<(), String> {
    if std::process::Command::new("z3")
        .arg("--version")
        .output()
        .is_err()
    {
        return Err("z3 is not installed".to_string());
    }
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail/formal_verification"),
        |path| {
            println!(
                "Running test_frontend_fail_formal_verification on {:?}",
                path
            );
            run_shell_tests(path)
        },
    )
}

#[test]
fn test_frontend_fail_unimplemented_smt() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/frontend/fail/unimplemented_smt"),
        |path| {
            println!("Running test_frontend_fail_unimplemented_smt on {:?}", path);
            run_shell_tests(path)
        },
    )
}

#[test]
fn test_backend_pass_autodiff() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/pass/autodiff"),
        |path| {
            println!("Running test_backend_pass_autodiff on {:?}", path);
            run_backend_autodiff_test(path)
        },
    )
}

// Backend Autodiff Runner
fn run_backend_autodiff_test(path: &Path) -> Result<(), String> {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let expect_lines: Vec<String> = source
        .lines()
        .filter(|line| line.trim().starts_with("// EXPECT:"))
        .map(|line| line.split_once("EXPECT:").unwrap().1.trim().to_string())
        .collect();

    let mut loader = vxc::module_loader::ModuleLoader::new();
    let mut program_arr = match loader.load_main(path.to_str().unwrap()) {
        Ok(p) => p,
        Err(e) => {
            return Err(format!(
                "Frontend failed to parse '{}': {}",
                path.display(),
                e
            ))
        }
    };

    let ast_idx = program_arr
        .iter()
        .position(|p| p.module_path == path.to_str().unwrap())
        .unwrap();
    let mut program = program_arr.remove(ast_idx);

    let mut global_macros = std::collections::HashMap::new();
    for p in &program_arr {
        for mac in &p.macros {
            global_macros.insert(mac.name.clone(), mac.rules.clone());
        }
    }
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::ast::MacroExpander::new(&global_macros);
    for p in &mut program_arr {
        if let Err(e) = expander.expand_module(p) {
            return Err(format!(
                "Macro expansion failed on {}: {}",
                p.module_path, e
            ));
        }
    }
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

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
        return Err(format!(
            "Semantic check failed on '{}':\n{:#?}",
            path.display(),
            checker.errors
        ));
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
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let mut codegen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    codegen
        .generate(&monomorphized_program, &module_asts)
        .unwrap();

    let mut module = codegen.into_module();
    if let Err(e) = vxc::codegen::lower_to_llvm(&context, &mut module) {
        println!("MLIR Before Lowering Error:\n{}", module.as_operation());
        return Err(format!(
            "Lowering to LLVM failed for {}: {:?}",
            path.display(),
            e
        ));
    }
    let mlir_str = module.as_operation().to_string();

    if source.contains("// NO_EXEC") {
        for expect in expect_lines {
            if !mlir_str.contains(&expect) {
                return Err(format!(
                    "MLIR Output mismatch on {:?}.\nExpected to find: `{}`\nActual MLIR:\n{}",
                    path, expect, mlir_str
                ));
            }
        }
        return Ok(());
    }

    if std::env::var("ENZYME_LIB").is_err() {
        println!(
            "Skipping JIT execution for {} because ENZYME_LIB is not set.",
            path.display()
        );
        return Ok(());
    }

    let out = execute_mlir(&mlir_str, vec![], 0, false).expect("JIT execution failed");

    for expect in expect_lines {
        if !mlir_str.contains(&expect) && !out.contains(&expect) {
            return Err(format!(
                "Output mismatch on {:?}.\nExpected to find: `{}`\nActual Output:\n{}\nMLIR:\n{}",
                path, expect, out, mlir_str
            ));
        }
    }
    Ok(())
}

#[test]
fn test_backend_fail() -> Result<(), String> {
    run_directory_tests(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/backend/fail"),
        run_shell_tests,
    )
}

// --- Test Runners ---
fn run_directory_tests<F>(dir: std::path::PathBuf, test_fn: F) -> Result<(), String>
where
    F: Fn(&std::path::Path) -> Result<(), String> + Sync + Send,
{
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        let errors: Vec<String> = entries
            .into_par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_fn(&path)));
                    match result {
                        Ok(Err(msg)) => return Some(format!("Test {:?} failed: {}", path, msg)),
                        Err(e) => {
                            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                                s.to_string()
                            } else if let Some(s) = e.downcast_ref::<String>() {
                                s.to_string()
                            } else {
                                "Unknown panic".to_string()
                            };
                            return Some(format!("Test {:?} panicked: {}", path, msg));
                        }
                        Ok(Ok(())) => {}
                    }
                }
                None
            })
            .collect();
        if !errors.is_empty() {
            return Err(format!(
                "The following tests failed:\n\n{}",
                errors.join("\n\n")
            ));
        }
    }
    Ok(())
}

fn run_shell_tests(path: &Path) -> Result<(), String> {
    let source = std::fs::read_to_string(path).unwrap_or_default();
    let run_lines: Vec<_> = source
        .lines()
        .filter(|l| l.trim_start().starts_with("// RUN:"))
        .collect();
    if !run_lines.is_empty() {
        let vxc_dir = Path::new(env!("CARGO_BIN_EXE_vxc")).parent().unwrap();
        let current_path = std::env::var("PATH").unwrap_or_default();
        let new_path = format!("{}:{}", vxc_dir.display(), current_path);
        for run_line in run_lines {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&path, &mut hasher);
            let hash = std::hash::Hasher::finish(&hasher);

            // Ensure temporary files are completely contained within the project repository
            let tmp_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target/test_tmp");
            std::fs::create_dir_all(&tmp_dir).unwrap_or_default();
            let t_val = tmp_dir.join(format!(
                "vxc_test_{}_{:x}",
                path.file_stem().unwrap().to_string_lossy(),
                hash
            ));

            let cmd = run_line
                .split_once("RUN:")
                .unwrap()
                .1
                .trim()
                .replace("%s", path.to_str().unwrap())
                .replace("%t", t_val.to_str().unwrap());
            let output = std::process::Command::new("sh")
                .arg("-c")
                .arg(&cmd)
                .env("PATH", &new_path)
                .output()
                .expect("Failed to execute shell command");
            if !output.status.success() {
                return Err(format!(
                    "Command '{}' failed for test {:?}\nStdout:\n{}\nStderr:\n{}",
                    cmd,
                    path,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ));
            }
        }
    } else {
        return Err(format!("Test with no RUN line: {:?}", path));
    }
    Ok(())
}

#[test]
fn test_melior_matmul() -> Result<(), String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/middle_end/pass/matmul.mlr");
    let source = fs::read_to_string(&path).expect("Failed to read test file");

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

    let mut global_macros = std::collections::HashMap::new();
    for mac in &program.macros {
        global_macros.insert(mac.name.clone(), mac.rules.clone());
    }
    let mut expander = vxc::ast::MacroExpander::new(&global_macros);
    if let Err(e) = expander.expand_module(&mut program) {
        return Err(format!("Macro expansion failed on {:?}: {}", path, e));
    }

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
    if checker
        .errors
        .iter()
        .any(|d| d.level == vxc::diagnostic::DiagnosticLevel::Error)
    {
        return Err(format!("Sema failed on {:?}: {:#?}", path, checker.errors));
    }

    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);

    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    vxc::codegen::register_vx_dialect(&context);

    let mut gen = vxc::codegen::MeliorGenerator::new(&context, "test".to_string());
    let mlir_str = gen
        .generate(&checked_program, &std::collections::HashMap::new())
        .unwrap();

    let mut current_idx = 0;
    for check in check_lines {
        if let Some(pos) = mlir_str[current_idx..].find(&check) {
            current_idx += pos + check.len();
        } else {
            return Err(format!("FileCheck failed on {:?}: Could not find `{}` after previous checks.\nMLIR Output:\n{}", path, check, mlir_str));
        }
    }
    Ok(())
}

extern "C" {
    fn registerVxDialect(ctx: mlir_sys::MlirContext);
}

#[test]
fn test_vx_dialect_registration() -> Result<(), String> {
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
    if module.is_none() {
        return Err(
            "Failed to parse module containing vx.spawn. Dialect may not be registered!"
                .to_string(),
        );
    }
    Ok(())
}
