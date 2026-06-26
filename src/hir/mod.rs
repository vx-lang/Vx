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

pub mod bytecode;
pub mod env;
pub mod expr;
pub mod prover;
pub mod stmt;

pub use bytecode::*;
pub use env::*;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

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
        // A mutable borrow should block direct access to the original variable.
        let input = r#"
        fn test() -> i32 {
            let mut x : Tensor<f32> = 1.0;
            let y = &mut x;
            let z = x;
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
}
