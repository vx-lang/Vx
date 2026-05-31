use crate::ast::*;

pub mod env;
pub mod expr;
pub mod stmt;

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
        let mut parser = Parser::new(tokens, input);
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
            checker.errors.is_empty()
        };

        for err in &checker.errors {
            println!("Error: {}", err);
        }
        assert!(success);
        assert!(checker.errors.is_empty());
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
        let mut parser = Parser::new(tokens, input);
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
            checker.errors.is_empty()
        };
        assert!(!success);
        assert!(!checker.errors.is_empty());
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
        let mut parser = Parser::new(lexer.tokenize(), input);
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
                checker.errors.is_empty()
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
        let mut parser = Parser::new(lexer.tokenize(), input);
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
            checker.errors.is_empty()
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
        let mut parser = Parser::new(lexer.tokenize(), input);
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
                checker.errors.is_empty()
            },
            "Semantic checking failed for methods: {:?}",
            checker.errors
        );
    }
}
