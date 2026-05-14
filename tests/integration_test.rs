use akarc::lexer::Lexer;
use akarc::parser::Parser;
use akarc::sema::TypeChecker;

#[test]
fn test_distributed_matmul_integration() {
    let input = r#"
fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>, b: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_a = transfer(a, Memory::NPU_HBM);
        let local_b = transfer(b, Memory::NPU_HBM);
        let result = custom_matmul(local_a, local_b);
        return transfer(result, Memory::Host_DRAM);
    }
}
    "#;

    // 1. Lexing
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    assert!(!tokens.is_empty());

    // 2. Parsing
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().expect("Failed to parse AST");
    assert_eq!(ast.functions.len(), 1);

    // 3. Semantic Analysis
    let mut checker = TypeChecker::new();
    let is_valid = checker.check_program(&ast);

    if !checker.errors.is_empty() {
        for err in &checker.errors {
            println!("Semantic Error: {}", err);
        }
    }

    assert!(is_valid, "Semantic analysis failed on integration test");
}

fn run_pipeline(input: &str) -> Result<akarc::ast::Program, Vec<String>> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize();
    let mut parser = Parser::new(tokens);
    let ast = parser.parse().map_err(|e| vec![e])?;
    let mut checker = TypeChecker::new();
    if checker.check_program(&ast) {
        let mut codegen = akarc::codegen::MlirGenerator::new();
        let _mlir_str = codegen.generate(&ast);
        Ok(ast)
    } else {
        for err in &checker.errors {
            println!("run_pipeline semantic error: {}", err);
        }
        Err(checker.errors)
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
