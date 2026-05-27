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
use vxc::codegen::MlirGenerator;
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
    let is_valid = checker.errors.is_empty();

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
        checker.errors.is_empty(),
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
    vxc::melior_codegen::register_vx_dialect(&context);

    let module_asts = std::collections::HashMap::new();

    let mlir_str = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut codegen = vxc::melior_codegen::MeliorGenerator::new(&context);
        codegen.generate(&monomorphized_program, &module_asts);
        codegen.into_module().as_operation().to_string()
    }))
    .unwrap_or_else(|_| {
        let mut codegen = vxc::codegen::MlirGenerator::new();
        codegen.generate(&monomorphized_program, &module_asts)
    });

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
    if !checker.errors.is_empty() {
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

    let mut codegen = MlirGenerator::new();
    let mlir_str = codegen.generate(&monomorphized_program, &module_asts);

    if source.contains("// NO_EXEC") {
        return;
    }

    let out = execute_mlir(&mlir_str).expect("JIT execution failed");

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
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mlr") {
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
        checker.errors.is_empty(),
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
    vxc::melior_codegen::register_vx_dialect(&context);

    let module_asts = std::collections::HashMap::new();

    let mlir_str = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut codegen = vxc::melior_codegen::MeliorGenerator::new(&context);
        codegen.generate(&monomorphized_program, &module_asts);
        codegen.into_module().as_operation().to_string()
    }))
    .unwrap_or_else(|_| {
        let mut codegen = vxc::codegen::MlirGenerator::new();
        codegen.generate(&monomorphized_program, &module_asts)
    });

    let temp_mlir = format!("{}_opt_temp.mlir", path.file_name().unwrap().to_string_lossy());
    let mut file = std::fs::File::create(&temp_mlir).unwrap();
    std::io::Write::write_all(&mut file, mlir_str.as_bytes()).unwrap();

    let mlir_opt_out = std::process::Command::new("/opt/homebrew/opt/llvm/bin/mlir-opt")
        .args([vxc::jit::OPTIMIZATION_PIPELINE, &temp_mlir])
        .output()
        .expect("Failed to execute mlir-opt");

    let _ = std::fs::remove_file(&temp_mlir);

    if !mlir_opt_out.status.success() {
        panic!("mlir-opt failed:\n{}", String::from_utf8_lossy(&mlir_opt_out.stderr));
    }

    let out = String::from_utf8_lossy(&mlir_opt_out.stdout);

    let mut current_idx = 0;
    for check in check_lines {
        if let Some(pos) = out[current_idx..].find(&check) {
            current_idx += pos + check.len();
        } else {
            panic!("FileCheck failed on {:?}: Could not find `{}` after previous checks.\nMLIR Output:\n{}", path, check, out);
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
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                println!("Running test_backend on {:?}", path);
                run_backend_test(&path);
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
    if !checker.errors.is_empty() {
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
    let mut codegen = vxc::melior_codegen::MeliorGenerator::new(&context);
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

    let out = execute_mlir(&mlir_str).expect("JIT execution failed");

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
        checker.errors.is_empty(),
        "Sema failed on {:?}: {:#?}",
        path,
        checker.errors
    );

    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);

    let context = melior::Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    let mut gen = vxc::melior_codegen::MeliorGenerator::new(&context);
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
