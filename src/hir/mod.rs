//===- mod.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis module for the Vx compiler. Defines the type checking environment and validation rules.
//
//===----------------------------------------------------------------------===//

use crate::syntax::*;

pub mod borrow_cx;
pub mod check;
pub mod check_state;
mod comptime_interpreter;
pub mod decl_check;
pub mod env;
pub mod expr;
pub mod flatten;
pub mod memory;
pub mod places;
pub mod provenance;
pub mod prover;
pub mod seam;
pub mod solver;
pub mod stmt;

pub use env::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    /// The type checker attaches the initialized struct's resolved GID to the `StructInit`
    /// expression (#199), so the flat-HIR lowerer can reach its registry layout without
    /// re-resolving the name. The GID is now resolved through the frozen registry's
    /// `ModuleInterface` (`resolve_unique_nominal`), not a borrowed-AST side map (#219), so the test
    /// freezes a real registry and takes the same registry lookup as its oracle.
    #[test]
    fn structinit_is_annotated_with_struct_gid() {
        let input = r#"
struct Point { x: i32, y: i32 }
fn make() -> Point {
    return Point { x: 1, y: 2 };
}
"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        program.module_path = "crate::t".into();

        // Resolve names, then freeze the registry so `StructInit` GID resolution has an oracle.
        let mut mods = vec![program];
        let symbol_map = crate::resolver::build_symbol_map(&mods);
        mods[0].resolve_names(&symbol_map, &[]);
        let registry = crate::pipeline::build_frozen_registry(&mods).expect("registry builds");
        let expected = registry
            .resolve_unique_nominal(&crate::symbol::Symbol::from("Point"))
            .expect("Point resolves via the interface");

        let mut program = mods.pop().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::with_registry(1, registry),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        for f in &mut program.functions {
            checker.check_function(f);
        }
        assert_eq!(checker.errors.error_count(), 0, "type-checks cleanly");

        let make = program
            .functions
            .iter()
            .find(|f| f.name.as_ref() == "make")
            .unwrap();
        let mut annotated = None;
        for stmt in &make.body {
            if let crate::syntax::Statement::Return(r) = stmt {
                if let Some(crate::syntax::Expr::StructInit(si)) = &r.expr {
                    annotated = si.type_id;
                }
            }
        }
        assert_eq!(
            annotated,
            Some(expected),
            "the `Point {{ .. }}` construction carries Point's GID"
        );
    }

    /// R3 (#279): speculative checking is carried by the `speculating` field, not a threaded
    /// `silent` parameter. This guards the three mechanisms the field replaced: (1) `speculating`
    /// suppresses diagnostics (a probe is silent), (2) a callee does not clobber the caller's flag,
    /// and (3) the "fresh check" entry `check_expr_type` forces the flag off for its subtree (so an
    /// independent subtree reached under a probe is still fully checked) then restores it. An
    /// undefined-variable reference is the probe: `check_identifier_expr` pushes its diagnostic
    /// exactly when `!self.speculating`.
    #[test]
    fn r3_speculating_gates_diagnostics_and_the_fresh_entry_forces_it_off() {
        let input = "fn f() -> i32 { return 0; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);

        let undefined = || {
            crate::syntax::Expr::Identifier(crate::syntax::IdentifierExpr::new(
                crate::symbol::Symbol::from("undefined_xyz"),
                crate::syntax::Span::default(),
            ))
        };

        // A checker starts non-speculative.
        assert!(!checker.speculating, "checker starts non-speculative");

        // (1a) A real check of an undefined variable emits a diagnostic.
        let mut e = undefined();
        let before = checker.errors.error_count();
        checker.check_expr_type_flag(&mut e, true);
        assert_eq!(
            checker.errors.error_count(),
            before + 1,
            "a real (non-speculative) check of an undefined variable emits one error"
        );

        // (1b) + (2) The same reference under `speculating = true` emits nothing, and the callee
        // leaves the flag as it found it.
        let mut e = undefined();
        let mid = checker.errors.error_count();
        checker.speculating = true;
        checker.check_expr_type_flag(&mut e, true);
        assert_eq!(
            checker.errors.error_count(),
            mid,
            "a speculative check suppresses the diagnostic"
        );
        assert!(
            checker.speculating,
            "the callee does not clobber the caller's speculating flag"
        );

        // (3) The "fresh check" entry forces `speculating` off for its subtree even under an
        // ambient probe, then restores the ambient value.
        let mut e = undefined();
        let before = checker.errors.error_count();
        checker.check_expr_type(&mut e);
        assert_eq!(
            checker.errors.error_count(),
            before + 1,
            "check_expr_type forces a fresh (emitting) check even while speculating"
        );
        assert!(
            checker.speculating,
            "check_expr_type restores the ambient speculating = true"
        );
    }

    #[test]
    fn test_sema_distributed_matmul() {
        let input = r#"
fn custom_matmul(a: Tensor<f32, [?, ?]>, b: Tensor<f32, [?, ?]>) -> Tensor<f32, [?, ?]> {
    return a;
}

fn distributed_matmul(a: Tensor<f32, [?, ?]>, b: Tensor<f32, [?, ?]>) -> Tensor<f32, [?, ?]> {
    let local_a = transfer(a, Memory::NPU_HBM);
    let local_b = transfer(b, Memory::NPU_HBM);
    spawn on(Topology::NPU[0]) {
        let result = local_a; 
        // In a real kernel, we would have explicit NPU intrinsics here.
        // For this test, we verify the spawn and transfer syntax parses.
    }
    return a;
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.error_count() == 0
        };

        for err in &checker.errors {
            println!("Error: {}", err);
        }
        assert!(success);
        assert!(checker.errors.error_count() == 0);
    }

    #[test]
    fn test_sema_type_mismatch() {
        let input = r#"
fn bad_matmul() -> Tensor<f32, [?, ?]> {
    return undefined_variable;
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.error_count() == 0
        };
        assert!(!success);
        assert!(checker.errors.error_count() > 0);
    }

    #[test]
    fn test_sema_struct_and_pointers() {
        let input = r#"
        struct Config {
            value: Tensor<f32, [?, ?]>
        }

        fn test_pointers(c: &mut Config) -> bool {
            unsafe {
                let ptr: *mut Config = c;
                let val = *ptr;
            }
            return c.value < 20.0f32;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        assert!(
            {
                for f in &mut program.functions {
                    checker.check_function(f);
                }
                checker.errors.error_count() == 0
            },
            "Semantic checking failed: {:?}",
            checker.errors
        );
    }

    #[test]
    fn test_sema_extern_unsafe() {
        let input = r#"
        extern "C" {
            fn malloc(size: Tensor<f32, [?, ?]>) -> *mut Tensor<f32, [?, ?]>;
        }

        fn safe_wrapper() -> *mut Tensor<f32, [?, ?]> {
            return malloc(1024); // ERROR: unsafe function call
        }

        fn safe_wrapper_fixed() -> *mut Tensor<f32, [?, ?]> {
            unsafe {
                return malloc(1024);
            }
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);

        let success = {
            for f in &mut program.functions {
                checker.check_function(f);
            }
            checker.errors.error_count() == 0
        };
        assert!(!success);
        assert!(checker.errors.iter().any(|e| {
            e.message
                .contains("Call to unsafe function 'malloc' is unsafe")
        }));
    }

    #[test]
    fn test_sema_as_ptr_and_len() {
        // `as_ptr()` on a tensor is a pointer to its first element, in the tensor's memory.
        // `len()` answers the outermost extent as an i32, like `extent`, and the checker
        // rewrites it to the `$extent` read both backends lower.
        let input = r#"
        fn test_methods(t: Tensor<f32, [?, ?]>) -> i32 {
            let ptr: *const f32 = t.as_ptr();
            let mut_ptr: *mut f32 = t.as_mut_ptr();
            let length: i32 = t.len();
            return length;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);

        assert!(
            {
                for f in &mut program.functions {
                    checker.check_function(f);
                }
                checker.errors.error_count() == 0
            },
            "Semantic checking failed for methods: {:?}",
            checker.errors
        );
        let crate::syntax::Statement::LetDecl(decl) = &program.functions[0].body[2] else {
            panic!("expected the `let length` statement");
        };
        let crate::syntax::Expr::IndexAccess(ix) = &decl.expr else {
            panic!(
                "len() was not rewritten to an indexed read: {:?}",
                decl.expr
            );
        };
        let crate::syntax::Expr::MemberAccess(ma) = ix.base.as_ref() else {
            panic!("the rewritten read is not a member access: {:?}", ix.base);
        };
        assert_eq!(ma.member.as_ref(), "$extent");
        let crate::syntax::Expr::Number(axis) = ix.index.as_ref() else {
            panic!("the axis is not a literal: {:?}", ix.index);
        };
        assert_eq!(axis.value.as_ref(), "0", "len() reads the outermost axis");
    }

    #[test]
    fn test_sema_liveness_analysis() {
        let input = r#"
        fn test_liveness() -> i32 {
            let a = 1;
            let b = a + 2;
            print(a);
            print(b);
            let c = 3;
            return c;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();

        // Grab the body of the first function
        let body = &program.functions[0].body;

        // The block is:
        // 0: let a = 1;
        // 1: let b = a + 2;
        // 2: print(a);
        // 3: print(b);
        // 4: let c = 3;
        // 5: return c;

        let liveness = TypeChecker::compute_block_liveness(body);

        assert_eq!(
            liveness.get("a"),
            Some(&2),
            "a is last used in print(a) at index 2"
        );
        assert_eq!(
            liveness.get("b"),
            Some(&3),
            "b is last used in print(b) at index 3"
        );
        assert_eq!(
            liveness.get("c"),
            Some(&5),
            "c is last used in return c at index 5"
        );
    }

    #[test]
    fn liveness_includes_a_use_in_a_comptime_block_result() {
        let input = r#"
        fn f() -> i32 {
            let value = 1;
            let result = comptime { value };
            return result;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().expect("comptime block parses");

        let liveness = TypeChecker::compute_block_liveness(&program.functions[0].body);

        assert_eq!(
            liveness.get("value"),
            Some(&1),
            "a value used by a comptime block result remains live at that statement"
        );
    }

    #[test]
    fn liveness_includes_a_use_in_an_unsafe_block_result() {
        let input = r#"
        fn f() -> i32 {
            let value = 1;
            let result = unsafe { value };
            return result;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().expect("unsafe block parses");

        let liveness = TypeChecker::compute_block_liveness(&program.functions[0].body);

        assert_eq!(
            liveness.get("value"),
            Some(&1),
            "a value used by an unsafe block result remains live at that statement"
        );
    }

    #[test]
    fn liveness_includes_an_autodiff_argument() {
        let input = r#"
        fn cube(v: f32) -> f32 { return v * v * v; }
        fn f(x: f32) -> f32 { return grad(cube, x); }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().expect("autodiff program parses");

        let liveness = TypeChecker::compute_block_liveness(&program.functions[1].body);

        assert_eq!(
            liveness.get("x"),
            Some(&0),
            "an autodiff argument is read by the expression that contains it"
        );
    }

    #[test]
    fn test_sema_linear_move_consumed() {
        // A Tensor<f32, [?, ?]> is linear: using it once consumes it, second use is an error.
        let input = r#"
        fn test() -> i32 {
            let a : Tensor<f32, []> = 1.0;
            let b = a;
            let c = a;
            return 0;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        for f in &mut program.functions {
            checker.check_function(f);
        }
        assert!(
            checker.errors.error_count() > 0,
            "Expected error for double-use of linear variable"
        );
        assert!(
            checker
                .errors
                .iter()
                .any(|e| e.message.contains("moved or consumed linear variable")),
            "Expected 'moved or consumed linear variable' error, got: {:?}",
            checker.errors
        );
    }

    #[test]
    fn test_sema_scalar_not_consumed() {
        // Scalars (i32) are NOT linear — they can be reused freely.
        let input = r#"
        fn test() -> i32 {
            let x = 42;
            let y = x + 1;
            let z = x + 2;
            return z;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        for f in &mut program.functions {
            checker.check_function(f);
        }
        assert!(
            checker.errors.error_count() == 0,
            "Scalars should not be consumed on use: {:?}",
            checker.errors
        );
    }

    #[test]
    fn test_sema_borrow_blocks_access() {
        // A *live* mutable borrow blocks direct access to the original variable. `y` is used after the
        // access (`use_ref(y)`), so its loan of `x` is alive at `let z = x` and the access is rejected.
        // (Without that later use `y` would be a dead borrow under NLL — #276 — and `let z = x` would
        // correctly compile; `use_ref(y)` is what makes this a genuine conflict, as in the companion
        // fixture `borrow_use_after_mut.vx`.)
        let input = r#"
        fn use_ref(r : &mut Tensor<f32, [?, ?]>) -> i32 {
            return 0;
        }
        fn test() -> i32 {
            let mut x : Tensor<f32, []> = 1.0;
            let y = &mut x;
            let z = x;
            use_ref(y);
            return 0;
        }
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        for f in &mut program.functions {
            checker.check_function(f);
        }
        assert!(
            !checker.errors.is_empty(),
            "Expected error for accessing mutably borrowed variable"
        );
        assert!(
            checker
                .errors
                .iter()
                .any(|e| e.message.contains("mutably borrowed")),
            "Expected 'mutably borrowed' error, got: {:?}",
            checker.errors
        );
    }

    // A relaxed cross-device transfer whose consumer asserts a value on the buffer.
    const SEAM_ASSERT_PROGRAM: &str = r#"
fn k(x: Pinned<Tensor<i32, [?, ?]>, Topology::NPU[0]>)
     on Topology::NPU[0] -> Pinned<Tensor<i32, [?, ?]>, Topology::NPU[0]> { return x; }
fn f(a: Tensor<i32, [?, ?]>) -> Pinned<Tensor<i32, [?, ?]>, Topology::NPU[0]> {
    let local_a = a.to_device_relaxed();
    spawn on(Topology::NPU[0]) {
        assert(local_a == 42);
        let r = k(local_a);
        r
    }
}
"#;

    fn equality_condition(name: &str, value: &str) -> Expr {
        let span = Span::default();
        Expr::RelationalOp(RelationalOpExpr::new(
            Box::new(Expr::Identifier(IdentifierExpr::new(name.into(), span))),
            RelationalOp::Eq,
            Box::new(Expr::Number(NumberExpr::new(value.to_string(), None, span))),
            span,
        ))
    }

    fn assert_eq_const(name: &str, value: &str) -> Statement {
        Statement::Assert(AssertStmt::new(
            Box::new(equality_condition(name, value)),
            None,
            Span::default(),
        ))
    }

    fn nested_consumer_assert(name: &str) -> Expr {
        let span = Span::default();
        Expr::UnsafeBlock(UnsafeBlockExpr::new(
            vec![assert_eq_const(name, "42")],
            Some(Box::new(Expr::Number(NumberExpr::new(
                "1".to_string(),
                None,
                span,
            )))),
            span,
        ))
    }

    #[test]
    fn seam_assert_prescan_scans_eager_expression_children() {
        let span = Span::default();
        let cases = vec![
            (
                "function_argument",
                Expr::FunctionCall(FunctionCallExpr::new(
                    "callee".into(),
                    None,
                    vec![nested_consumer_assert("function_argument")],
                    span,
                )),
            ),
            (
                "indirect_argument",
                Expr::IndirectCall(IndirectCallExpr::new(
                    Box::new(Expr::Identifier(IdentifierExpr::new("callee".into(), span))),
                    vec![nested_consumer_assert("indirect_argument")],
                    span,
                )),
            ),
            (
                "indirect_callee",
                Expr::IndirectCall(IndirectCallExpr::new(
                    Box::new(nested_consumer_assert("indirect_callee")),
                    vec![],
                    span,
                )),
            ),
            (
                "array_element",
                Expr::Array(ArrayExpr::new(
                    vec![nested_consumer_assert("array_element")],
                    span,
                )),
            ),
            (
                "member_base",
                Expr::MemberAccess(MemberAccessExpr::new(
                    Box::new(nested_consumer_assert("member_base")),
                    "field".into(),
                    span,
                )),
            ),
            (
                "index_expression",
                Expr::IndexAccess(IndexAccessExpr::new(
                    Box::new(Expr::Identifier(IdentifierExpr::new("array".into(), span))),
                    Box::new(nested_consumer_assert("index_expression")),
                    span,
                )),
            ),
            (
                "index_base",
                Expr::IndexAccess(IndexAccessExpr::new(
                    Box::new(nested_consumer_assert("index_base")),
                    Box::new(Expr::Number(NumberExpr::new("0".to_string(), None, span))),
                    span,
                )),
            ),
            (
                "method_argument",
                Expr::MethodCall(MethodCallExpr::new(
                    Box::new(Expr::Identifier(IdentifierExpr::new(
                        "receiver".into(),
                        span,
                    ))),
                    "method".into(),
                    None,
                    vec![nested_consumer_assert("method_argument")],
                    span,
                )),
            ),
            (
                "method_base",
                Expr::MethodCall(MethodCallExpr::new(
                    Box::new(nested_consumer_assert("method_base")),
                    "method".into(),
                    None,
                    vec![],
                    span,
                )),
            ),
            (
                "binary_rhs",
                Expr::BinaryOp(BinaryOpExpr::new(
                    Box::new(Expr::Number(NumberExpr::new("1".to_string(), None, span))),
                    BinaryOp::Add,
                    Box::new(nested_consumer_assert("binary_rhs")),
                    span,
                )),
            ),
            (
                "relational_rhs",
                Expr::RelationalOp(RelationalOpExpr::new(
                    Box::new(Expr::Number(NumberExpr::new("1".to_string(), None, span))),
                    RelationalOp::Eq,
                    Box::new(nested_consumer_assert("relational_rhs")),
                    span,
                )),
            ),
            (
                "unary_operand",
                Expr::UnaryOp(UnaryOpExpr::new(
                    UnaryOp::Not,
                    Box::new(nested_consumer_assert("unary_operand")),
                    span,
                )),
            ),
            (
                "borrow_operand",
                Expr::Borrow(BorrowExpr::new(
                    Box::new(nested_consumer_assert("borrow_operand")),
                    false,
                    span,
                )),
            ),
            (
                "dereference_operand",
                Expr::Dereference(DereferenceExpr::new(
                    Box::new(nested_consumer_assert("dereference_operand")),
                    span,
                )),
            ),
            (
                "transfer_operand",
                Expr::Transfer(TransferExpr::new(
                    Box::new(nested_consumer_assert("transfer_operand")),
                    MemorySpace::CPUDRAM,
                    span,
                )),
            ),
            (
                "enum_payload",
                Expr::EnumVariant(EnumVariantExpr::new(
                    "Result".into(),
                    "Ok".into(),
                    Some(vec![nested_consumer_assert("enum_payload")]),
                    span,
                )),
            ),
            (
                "struct_field",
                Expr::StructInit(StructInitExpr::new(
                    "Record".into(),
                    vec![("field".into(), nested_consumer_assert("struct_field"))],
                    span,
                )),
            ),
            (
                "range_end",
                Expr::Range(RangeExpr::new(
                    Box::new(Expr::Number(NumberExpr::new("0".to_string(), None, span))),
                    Box::new(nested_consumer_assert("range_end")),
                    span,
                )),
            ),
            (
                "grad_argument",
                Expr::Grad(GradExpr::new(
                    "target".into(),
                    vec![nested_consumer_assert("grad_argument")],
                    span,
                )),
            ),
            (
                "vjp_cotangent",
                Expr::Vjp(VjpExpr::new(
                    "target".into(),
                    vec![Expr::Number(NumberExpr::new("1".to_string(), None, span))],
                    Box::new(nested_consumer_assert("vjp_cotangent")),
                    span,
                )),
            ),
            (
                "vjp_argument",
                Expr::Vjp(VjpExpr::new(
                    "target".into(),
                    vec![nested_consumer_assert("vjp_argument")],
                    Box::new(Expr::Number(NumberExpr::new("1".to_string(), None, span))),
                    span,
                )),
            ),
            (
                "jvp_tangent",
                Expr::Jvp(JvpExpr::new(
                    "target".into(),
                    vec![Expr::Number(NumberExpr::new("1".to_string(), None, span))],
                    Box::new(nested_consumer_assert("jvp_tangent")),
                    span,
                )),
            ),
            (
                "jvp_argument",
                Expr::Jvp(JvpExpr::new(
                    "target".into(),
                    vec![nested_consumer_assert("jvp_argument")],
                    Box::new(Expr::Number(NumberExpr::new("1".to_string(), None, span))),
                    span,
                )),
            ),
            (
                "vec_element",
                Expr::VecMacro(VecMacroExpr::new(
                    vec![nested_consumer_assert("vec_element")],
                    span,
                )),
            ),
            (
                "cast_operand",
                Expr::AsCast(AsCastExpr {
                    expr: Box::new(nested_consumer_assert("cast_operand")),
                    target_ty: Type::Scalar(ElementType::I32),
                    source_ty: None,
                    span,
                }),
            ),
            (
                "print_argument",
                Expr::Print(PrintExpr::new(
                    vec![nested_consumer_assert("print_argument")],
                    span,
                )),
            ),
            (
                "println_argument",
                Expr::Println(PrintlnExpr::new(
                    vec![nested_consumer_assert("println_argument")],
                    span,
                )),
            ),
            (
                "inline_mlir_input",
                Expr::InlineMlir(InlineMlirExpr {
                    inputs: vec![(
                        "arg".into(),
                        nested_consumer_assert("inline_mlir_input"),
                        "i32".to_string(),
                    )],
                    clobbers: vec![],
                    returns: None,
                    dialects: vec![],
                    block_str: String::new(),
                    span,
                }),
            ),
            (
                "inline_mlir_clobber",
                Expr::InlineMlir(InlineMlirExpr {
                    inputs: vec![],
                    clobbers: vec![nested_consumer_assert("inline_mlir_clobber")],
                    returns: None,
                    dialects: vec![],
                    block_str: String::new(),
                    span,
                }),
            ),
            (
                "topology_index",
                Expr::Topology(TopologyExpr::new(
                    Topology::NPU(Box::new(nested_consumer_assert("topology_index"))),
                    span,
                )),
            ),
            (
                "transfer_predicate_index",
                Expr::TransferPredicate(TransferPredicateExpr {
                    from: Topology::GPU(Box::new(nested_consumer_assert(
                        "transfer_predicate_index",
                    ))),
                    to: Topology::CPU,
                    span,
                }),
            ),
            ("unsafe_block", nested_consumer_assert("unsafe_block")),
            (
                "comptime_block",
                Expr::ComptimeBlock(ComptimeBlockExpr::new(
                    vec![assert_eq_const("comptime_block", "42")],
                    None,
                    span,
                )),
            ),
            (
                "spawn_body",
                Expr::SpawnOn(SpawnOnExpr::new(
                    Topology::CPU,
                    vec![assert_eq_const("spawn_body", "42")],
                    None,
                    span,
                )),
            ),
            (
                "spawn_topology_index",
                Expr::SpawnOn(SpawnOnExpr::new(
                    Topology::NPU(Box::new(nested_consumer_assert("spawn_topology_index"))),
                    vec![],
                    None,
                    span,
                )),
            ),
        ];

        for (name, expr) in cases {
            let mut contracts = std::collections::HashMap::new();
            TypeChecker::scan_expr_for_asserts(&expr, &mut contracts);
            assert_eq!(
                contracts.get(name),
                Some(&42u64),
                "the scanner must visit the {name} child: {contracts:?}"
            );
        }

        let closure = Expr::Closure(ClosureExpr::new(
            vec![],
            Box::new(nested_consumer_assert("closure_body")),
            span,
        ));
        let mut contracts = std::collections::HashMap::new();
        TypeChecker::scan_expr_for_asserts(&closure, &mut contracts);
        assert!(
            !contracts.contains_key("closure_body"),
            "a closure body is deferred and must not impose a contract: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_scans_topology_children() {
        let span = Span::default();
        let expr = Expr::TransferPredicate(TransferPredicateExpr {
            from: Topology::Slice(
                Box::new(Topology::NPU(Box::new(nested_consumer_assert(
                    "slice_base_index",
                )))),
                Box::new(nested_consumer_assert("slice_start")),
                Box::new(nested_consumer_assert("slice_end")),
            ),
            to: Topology::AccCore(Box::new(nested_consumer_assert("predicate_to_index"))),
            span,
        });
        let mut contracts = std::collections::HashMap::new();
        TypeChecker::scan_expr_for_asserts(&expr, &mut contracts);
        for name in [
            "slice_base_index",
            "slice_start",
            "slice_end",
            "predicate_to_index",
        ] {
            assert_eq!(
                contracts.get(name),
                Some(&42),
                "the scanner must visit the {name} topology child: {contracts:?}"
            );
        }
    }

    #[test]
    fn seam_assert_prescan_scans_eager_statement_children() {
        let span = Span::default();
        let cases = vec![
            (
                "let_initializer",
                Statement::LetDecl(LetDeclStmt::new(
                    "binding".to_string(),
                    false,
                    None,
                    nested_consumer_assert("let_initializer"),
                    span,
                )),
            ),
            (
                "expression_statement",
                Statement::ExprStmt(ExprStmtStmt::new(
                    nested_consumer_assert("expression_statement"),
                    true,
                    span,
                )),
            ),
            (
                "return_expression",
                Statement::Return(ReturnStmt::new(
                    Some(nested_consumer_assert("return_expression")),
                    span,
                )),
            ),
            (
                "assignment_rhs",
                Statement::Assign(AssignStmt::new(
                    Expr::Identifier(IdentifierExpr::new("target".into(), span)),
                    nested_consumer_assert("assignment_rhs"),
                    span,
                )),
            ),
            (
                "assignment_lhs",
                Statement::Assign(AssignStmt::new(
                    nested_consumer_assert("assignment_lhs"),
                    Expr::Number(NumberExpr::new("0".to_string(), None, span)),
                    span,
                )),
            ),
            (
                "compound_assignment_rhs",
                Statement::CompoundAssign(CompoundAssignStmt::new(
                    Expr::Identifier(IdentifierExpr::new("target".into(), span)),
                    BinaryOp::Add,
                    nested_consumer_assert("compound_assignment_rhs"),
                    span,
                )),
            ),
            (
                "compound_assignment_lhs",
                Statement::CompoundAssign(CompoundAssignStmt::new(
                    nested_consumer_assert("compound_assignment_lhs"),
                    BinaryOp::Add,
                    Expr::Number(NumberExpr::new("0".to_string(), None, span)),
                    span,
                )),
            ),
            (
                "for_iterable",
                Statement::ForLoop(ForLoopStmt::new(
                    "index".to_string(),
                    Box::new(nested_consumer_assert("for_iterable")),
                    vec![],
                    vec![],
                    span,
                )),
            ),
        ];

        for (name, statement) in cases {
            let mut contracts = std::collections::HashMap::new();
            TypeChecker::collect_assert_contracts(&[statement], &mut contracts);
            assert_eq!(
                contracts.get(name),
                Some(&42u64),
                "the scanner must visit the {name} child: {contracts:?}"
            );
        }
    }

    #[test]
    fn seam_assert_prescan_keeps_only_unconditional_branch_and_loop_contracts() {
        let span = Span::default();
        let condition = Box::new(Expr::Identifier(IdentifierExpr::new(
            "condition".into(),
            span,
        )));

        let cases = vec![
            (
                "if_disagrees",
                Expr::If(IfExpr::new(
                    false,
                    condition.clone(),
                    vec![assert_eq_const("if_disagrees", "42")],
                    Some(vec![assert_eq_const("if_disagrees", "7")]),
                    span,
                )),
                None,
            ),
            (
                "if_single_branch",
                Expr::If(IfExpr::new(
                    false,
                    condition.clone(),
                    vec![assert_eq_const("if_single_branch", "42")],
                    None,
                    span,
                )),
                None,
            ),
            (
                "if_shared",
                Expr::If(IfExpr::new(
                    false,
                    condition,
                    vec![assert_eq_const("if_shared", "42")],
                    Some(vec![assert_eq_const("if_shared", "42")]),
                    span,
                )),
                Some(42),
            ),
        ];

        for (name, expr, expected) in cases {
            let mut contracts = std::collections::HashMap::new();
            TypeChecker::scan_expr_for_asserts(&expr, &mut contracts);
            assert_eq!(
                contracts.get(name).copied(),
                expected,
                "the scanner must retain only unconditional if facts: {contracts:?}"
            );
        }

        let loop_stmt = Statement::Loop(LoopStmt::new(
            vec![equality_condition("loop_invariant", "42")],
            vec![assert_eq_const("loop_body", "7")],
            span,
        ));
        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&[loop_stmt], &mut contracts);
        assert_eq!(contracts.get("loop_invariant"), Some(&42));
        assert!(
            !contracts.contains_key("loop_body"),
            "a loop body may not run and cannot impose a contract: {contracts:?}"
        );

        let for_loop = Statement::ForLoop(ForLoopStmt::new(
            "index".to_string(),
            Box::new(Expr::Number(NumberExpr::new("0".to_string(), None, span))),
            vec![equality_condition("for_invariant", "42")],
            vec![assert_eq_const("for_body", "7")],
            span,
        ));
        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&[for_loop], &mut contracts);
        assert_eq!(contracts.get("for_invariant"), Some(&42));
        assert!(
            !contracts.contains_key("for_body"),
            "a for-loop body may not run and cannot impose a contract: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_handles_asserted_disjunctions_and_short_circuiting() {
        let span = Span::default();
        let disjunction = Statement::Assert(AssertStmt::new(
            Box::new(Expr::LogicalOp(LogicalOpExpr::new(
                Box::new(equality_condition("left_disjunct", "42")),
                LogicalOp::Or,
                Box::new(equality_condition("right_disjunct", "7")),
                span,
            ))),
            None,
            span,
        ));
        let asserted_short_circuit = Statement::Assert(AssertStmt::new(
            Box::new(Expr::LogicalOp(LogicalOpExpr::new(
                Box::new(Expr::Identifier(IdentifierExpr::new(
                    "predicate".into(),
                    span,
                ))),
                LogicalOp::Or,
                Box::new(nested_consumer_assert("asserted_rhs")),
                span,
            ))),
            None,
            span,
        ));
        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(
            &[disjunction, asserted_short_circuit],
            &mut contracts,
        );
        assert!(
            !contracts.contains_key("left_disjunct")
                && !contracts.contains_key("right_disjunct")
                && !contracts.contains_key("asserted_rhs"),
            "an asserted disjunction establishes neither operand: {contracts:?}"
        );

        for operator in [LogicalOp::And, LogicalOp::Or] {
            let expr = Expr::LogicalOp(LogicalOpExpr::new(
                Box::new(nested_consumer_assert("logical_lhs")),
                operator,
                Box::new(nested_consumer_assert("logical_rhs")),
                span,
            ));
            let mut contracts = std::collections::HashMap::new();
            TypeChecker::scan_expr_for_asserts(&expr, &mut contracts);
            assert_eq!(
                contracts.get("logical_lhs"),
                Some(&42),
                "the always-evaluated logical left operand must be scanned: {contracts:?}"
            );
            assert!(
                !contracts.contains_key("logical_rhs"),
                "the short-circuited logical right operand must be skipped: {contracts:?}"
            );
        }
    }

    #[test]
    fn test_seam_assert_prescan_extracts_contract() {
        // The pre-scan recovers the consumer's `assert(local_a == 42)` from inside the
        // spawn body -- the conclusion of the boundary contract -- as a pure AST walk,
        // independent of where the transfer is checked (the ordering fix).
        let mut lexer = Lexer::new(SEAM_ASSERT_PROGRAM);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, SEAM_ASSERT_PROGRAM);
        let program = parser.parse().unwrap();
        let f = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&f.body, &mut contracts);
        assert_eq!(
            contracts.get("local_a"),
            Some(&42u64),
            "pre-scan should extract local_a == 42 from the spawn body, got {:?}",
            contracts
        );
    }

    #[test]
    fn seam_assert_prescan_collects_an_assert_nested_in_a_binary_expression() {
        let source = r#"
fn f(value: i32) -> i32 {
    let result = 1 + if 1 == 1 {
        assert(value == 42);
        2
    } else {
        assert(value == 42);
        3
    };
    return result;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("nested if expression parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("value"),
            Some(&42u64),
            "the pre-scan must collect asserts inside binary-expression operands: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_collects_each_conjunct_of_an_assert() {
        let source = r#"
fn f(left: i32, right: i32) -> i32 {
    assert(left == 42 && right == 7);
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("assertion fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("left"),
            Some(&42u64),
            "the first assertion conjunct must be collected: {contracts:?}"
        );
        assert_eq!(
            contracts.get("right"),
            Some(&7u64),
            "the second assertion conjunct must be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_collects_an_assert_nested_in_an_assert_condition() {
        let source = r#"
fn f(value: i32) -> i32 {
    assert(if 1 == 1 {
        assert(value == 42);
        true
    } else {
        assert(value == 42);
        true
    });
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("nested assertion fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("value"),
            Some(&42u64),
            "an assertion nested in an assertion condition must be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_collects_for_loop_invariant_contracts() {
        let source = r#"
fn f(value: i32) -> i32 {
    for i in 0..4 invariant value == 42 {
        let copy = i;
    }
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("for-loop invariant fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("value"),
            Some(&42u64),
            "a for-loop invariant contract must be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_collects_loop_invariant_contracts() {
        let source = r#"
fn f(value: i32) -> i32 {
    loop invariant value == 42 {
        break;
    }
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("loop invariant fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("value"),
            Some(&42u64),
            "a loop invariant contract must be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_discards_contracts_that_differ_between_match_arms() {
        let source = r#"
fn f(selector: i32, value: i32) -> i32 {
    match selector {
        0 => { assert(value == 42); }
        _ => { assert(value == 7); }
    }
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("match fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert!(
            !contracts.contains_key("value"),
            "a contract that differs by match arm must not be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_keeps_contracts_shared_by_every_match_arm() {
        let source = r#"
fn f(selector: i32, value: i32) -> i32 {
    match selector {
        0 => { assert(value == 42); }
        _ => { assert(value == 42); }
    }
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("match fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert_eq!(
            contracts.get("value"),
            Some(&42u64),
            "a contract shared by every match arm must be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_skips_short_circuited_and_rhs() {
        let source = r#"
fn f(predicate: bool, value: i32) -> i32 {
    let result = predicate && if 1 == 1 {
        assert(value == 42);
        true
    } else {
        false
    };
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("logical-and fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert!(
            !contracts.contains_key("value"),
            "a contract in a short-circuited && right operand must not be collected: {contracts:?}"
        );
    }

    #[test]
    fn seam_assert_prescan_skips_short_circuited_or_rhs() {
        let source = r#"
fn f(predicate: bool, value: i32) -> i32 {
    let result = predicate || if 1 == 1 {
        assert(value == 42);
        true
    } else {
        false
    };
    return 0;
}
        "#;
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, source);
        let program = parser.parse().expect("logical-or fixture parses");
        let function = program
            .functions
            .iter()
            .find(|function| function.name.as_ref() == "f")
            .expect("function f is present");

        let mut contracts = std::collections::HashMap::new();
        TypeChecker::collect_assert_contracts(&function.body, &mut contracts);

        assert!(
            !contracts.contains_key("value"),
            "a contract in a short-circuited || right operand must not be collected: {contracts:?}"
        );
    }

    #[test]
    fn test_seam_check_off_by_default() {
        // With seam verification disabled (the default), a relaxed transfer raises no
        // E6004 -- the obligation (and its z3 dependency) is opt-in via --verify-seams.
        let mut lexer = Lexer::new(SEAM_ASSERT_PROGRAM);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, SEAM_ASSERT_PROGRAM);
        let mut program = parser.parse().unwrap();

        let program_arr = [program.clone()];
        let env = GlobalAstEnv::build(&program_arr);
        let mut worker = crate::session::LocalWorkerState::new(std::sync::Arc::new(
            crate::session::GlobalSession::new(1),
        ));
        let mut checker = TypeChecker::new(&env, &mut worker);
        assert!(!checker.seam.verify, "seam verification must default off");
        for func in &mut program.functions {
            checker.check_function(func);
        }
        assert!(
            checker
                .errors
                .iter()
                .all(|d| d.code != Some(crate::diagnostic::DiagnosticCode::E6004)),
            "no seam (E6004) diagnostic should be emitted when --verify-seams is off: {:?}",
            checker.errors
        );
    }
}
