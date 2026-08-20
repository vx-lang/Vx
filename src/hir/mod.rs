//===- mod.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Semantic analysis module for the Vx compiler. Defines the type checking environment and validation rules.
//
//===----------------------------------------------------------------------===//

use crate::syntax::*;

pub mod arena;
pub mod borrow_cx;
pub mod bytecode;
pub mod check;
pub mod env;
pub mod expr;
pub mod flatten;
pub mod lower_ast;
pub mod memory;
pub mod places;
pub mod provenance;
pub mod prover;
pub mod seam;
pub mod solver;
pub mod stmt;

pub use bytecode::*;
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
        mods[0].resolve_names(&symbol_map);
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
                if let crate::syntax::Expr::StructInit(si) = &r.expr {
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
fn custom_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    return a;
}

fn distributed_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
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
fn bad_matmul() -> Tensor {
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
            value: Tensor<f32>
        }

        fn test_pointers(c: &mut Config) -> Tensor<Bool> {
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
            fn malloc(size: Tensor<f32>) -> *mut Tensor<f32>;
        }

        fn safe_wrapper() -> *mut Tensor<f32> {
            return malloc(1024); // ERROR: unsafe function call
        }

        fn safe_wrapper_fixed() -> *mut Tensor<f32> {
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
        assert!(checker.errors.iter().any(|e| e
            .message
            .contains("Call to unsafe function 'malloc' is unsafe")));
    }

    #[test]
    fn test_sema_as_ptr_and_len() {
        let input = r#"
        fn test_methods(t: Tensor<f32>) -> Tensor<i64> {
            let ptr: *const Tensor<f32> = t.as_ptr();
            let mut_ptr: *mut Tensor<f32> = t.as_mut_ptr();
            let length: Tensor<i64> = t.len();
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
    fn test_sema_linear_move_consumed() {
        // A Tensor is linear: using it once consumes it, second use is an error.
        let input = r#"
        fn test() -> i32 {
            let a : Tensor<f32> = 1.0;
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
        fn use_ref(r : &mut Tensor<f32>) -> i32 {
            return 0;
        }
        fn test() -> i32 {
            let mut x : Tensor<f32> = 1.0;
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
fn k(x: Pinned<Tensor<i32>, Topology::NPU[0]>)
     on Topology::NPU[0] -> Pinned<Tensor<i32>, Topology::NPU[0]> { return x; }
fn f(a: Tensor<i32>) -> Pinned<Tensor<i32>, Topology::NPU[0]> {
    let local_a = a.to_device_relaxed();
    spawn on(Topology::NPU[0]) {
        assert(local_a == 42);
        let r = k(local_a);
        r
    }
}
"#;

    #[test]
    fn test_seam_assert_prescan_extracts_contract() {
        // The pre-scan recovers the consumer's `assert(local_a == 42)` from inside the
        // spawn body -- the conclusion of the boundary contract -- as a pure AST walk,
        // independent of where the transfer is checked (the ordering fix).
        let mut lexer = Lexer::new(SEAM_ASSERT_PROGRAM);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, SEAM_ASSERT_PROGRAM);
        let program = parser.parse().unwrap();
        let f = program.functions.last().unwrap(); // `f`

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
        assert!(!checker.verify_seams, "seam verification must default off");
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
