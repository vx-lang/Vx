use vxc::lexer::Lexer;
use vxc::parser::Parser;
use vxc::sema::TypeChecker;

#[test]
fn test_distributed_matmul_integration() {
    let input = r#"
fn custom_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    return a;
}

fn distributed_matmul(a: Tensor<f32>, b: Tensor<f32>) -> Tensor<f32> {
    let local_a = a.to_device();
    let local_b = b.to_device();
    spawn on(Topology::NPU[0]) {
        let result = custom_matmul(local_a, local_b);
        return result;
    }
}
    "#;

    // 1. Lexing
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    assert!(!tokens.is_empty());

    // 2. Parsing
    let mut parser = Parser::new(tokens, input);
    let mut ast = parser.parse().expect("Failed to parse AST");
    assert_eq!(ast.functions.len(), 2);

    // 3. Semantic Analysis
    let mut checker = TypeChecker::new();
    let is_valid = checker.check_program(&mut ast).is_ok();

    if !checker.errors.is_empty() {
        for err in &checker.errors {
            println!("Semantic Error: {}", err);
        }
    }

    assert!(is_valid, "Semantic analysis failed on integration test");
}

fn run_pipeline(input: &str) -> Result<vxc::ast::Program, Vec<String>> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens, input);
    let mut program = parser.parse().map_err(|e| vec![e])?;
    let mut checker = TypeChecker::new();
    match checker.check_program(&mut program) {
        Ok((monomorphized_ast, module_asts)) => {
            let mut codegen = vxc::codegen::MlirGenerator::new();
            let _mlir_str = codegen.generate(&monomorphized_ast, &module_asts);
            Ok(monomorphized_ast)
        }
        Err(errs) => {
            for err in &errs {
                println!("run_pipeline semantic error: {}", err);
            }
            Err(errs)
        }
    }
}

#[test]
fn test_integration_operators() {
    let input = r#"
    fn math_ops() -> Tensor {
        let mut x = 10;
        let y = x * 5;
        x += y + 2;
        return x;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}

#[test]
fn test_integration_loops() {
    let input = r#"
    fn loop_test() -> Tensor {
        let mut sum = 0;
        for i in 0..10 {
            sum += i;
        }
        return sum;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}

#[test]
fn test_integration_arrays_and_indexing() {
    let input = r#"
    fn array_test(a: Tensor, b: Tensor) -> Tensor {
        let mut arr = Tensor([a.shape[0], b.shape[1]]);
        arr[0][0] = a[0][1] * b[1][0];
        return arr;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}

#[test]
fn test_integration_method_chaining() {
    let input = r#"
    fn memory_test() -> Ref<Tensor, Memory::NPU_HBM> {
        let mut mem = Tensor([10]).with_memory(Memory::NPU_HBM);
        return mem;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}

#[test]
fn test_integration_function_calls() {
    let input = r#"
    fn helper(x: Tensor) -> Tensor {
        return x + 1;
    }
    
    fn main() -> Tensor {
        let y = 10;
        let z = helper(y);
        return z;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}

#[test]
fn test_integration_logical_ops() {
    let input = r#"
    fn logic_test(a: Tensor, b: Tensor) -> Tensor {
        let is_less = a < b;
        let is_eq = a == b;
        let c = is_less && is_eq;
        return a;
    }
    "#;
    assert!(run_pipeline(input).is_ok());
}
