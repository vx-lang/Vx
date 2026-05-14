use std::fs;
use std::path::Path;

use akarc::codegen::MlirGenerator;
use akarc::jit::execute_mlir;
use akarc::lexer::Lexer;
use akarc::parser::Parser;
use akarc::sema::TypeChecker;

// Frontend Runner
fn run_frontend_test(path: &Path, expect_pass: bool) {
    let source = fs::read_to_string(path).expect("Failed to read test file");

    let mut lexer = Lexer::new(&source);
    let tokens = lexer.tokenize();

    let mut parser = Parser::new(tokens);
    let ast_res = parser.parse();

    if !expect_pass && ast_res.is_err() {
        return; // Expected fail and failed parsing
    }

    let ast = ast_res.unwrap_or_else(|_| panic!("Parse failed on {:?}", path));

    let mut checker = TypeChecker::new();
    let is_valid = checker.check_program(&ast);

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
    let mut parser = Parser::new(lexer.tokenize());
    let ast = parser.parse().unwrap();
    let mut checker = TypeChecker::new();
    assert!(
        checker.check_program(&ast),
        "Semantic analysis failed on {:?}",
        path
    );

    let mut codegen = MlirGenerator::new();
    let mlir_str = codegen.generate(&ast);

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
    let mut parser = Parser::new(lexer.tokenize());
    let ast = parser.parse().unwrap();
    let mut checker = TypeChecker::new();
    assert!(
        checker.check_program(&ast),
        "Semantic analysis failed on {:?}: {:?}",
        path,
        checker.errors
    );

    let mut codegen = MlirGenerator::new();
    let mlir_str = codegen.generate(&ast);

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
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("ak") {
                run_frontend_test(&path, true);
            }
        }
    }
}

#[test]
fn test_frontend_fail() {
    let dir = Path::new("tests/frontend/fail");
    if dir.exists() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("ak") {
                run_frontend_test(&path, false);
            }
        }
    }
}

#[test]
fn test_middle_end() {
    let dir = Path::new("tests/middle_end");
    if dir.exists() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("ak") {
                run_middle_end_test(&path);
            }
        }
    }
}

#[test]
fn test_backend() {
    let dir = Path::new("tests/backend");
    if dir.exists() {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) == Some("ak") {
                run_backend_test(&path);
            }
        }
    }
}
