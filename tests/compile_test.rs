use std::fs;
use std::path::Path;

use rayon::prelude::*;
use vxc::codegen::MlirGenerator;
use vxc::jit::execute_mlir;
use vxc::lexer::Lexer;
use vxc::parser::Parser;
use vxc::sema::TypeChecker;

// Frontend Runner
fn run_frontend_test(path: &Path, expect_pass: bool) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();

    let mut parser = Parser::new(tokens, &source);
    let mut program = match parser.parse() {
        Ok(p) => p,
        Err(_) => {
            if !expect_pass {
                return;
            }
            panic!("Parse failed on {:?}", path);
        }
    };

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [program.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
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

    let mut lexer = Lexer::new(&source);
    let mut parser = Parser::new(lexer.tokenize(), &source);
    let mut program = parser.parse().expect("Failed to parse");

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [program.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
    let mut worker = vxc::session::LocalWorkerState::new(global_session.clone());
    let mut checker = TypeChecker::new(&env, &mut worker);
    for f in &mut program.functions {
        checker.check_function(f);
    }
    assert!(
        checker.errors.is_empty(),
        "Sema failed: {:#?}",
        checker.errors
    );

    let mut monomorphized_program = program;
    let mut orig_functions = monomorphized_program.functions;
    orig_functions.retain(|f| f.generics.is_empty());

    let mut new_functions = checker.monomorphized_functions;
    new_functions.extend(orig_functions);
    monomorphized_program.functions = new_functions;

    let module_asts = std::collections::HashMap::new();
    let mut codegen = MlirGenerator::new();
    let mlir_str = codegen.generate(&monomorphized_program, &module_asts);

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

    let mut lexer = Lexer::new(&source);
    let mut parser = Parser::new(lexer.tokenize(), &source);
    let mut program = match parser.parse() {
        Ok(p) => p,
        Err(e) => panic!("Frontend failed to parse '{}': {}", path.display(), e),
    };

    let global_session = std::sync::Arc::new(vxc::session::GlobalSession::new(1));
    let program_arr = [program.clone()];
    let env = vxc::sema::GlobalAstEnv::build(&program_arr);
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
    let monomorphized_program = program;
    let module_asts = std::collections::HashMap::new();

    let mut codegen = MlirGenerator::new();
    let mlir_str = codegen.generate(&monomorphized_program, &module_asts);

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
    let dir = Path::new("tests/backend/pass");
    if dir.exists() {
        let entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap()).collect();
        entries.into_par_iter().for_each(|entry| {
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("vx") {
                run_backend_test(&path);
            }
        });
    }
}

#[test]
fn test_backend_fail() {
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
