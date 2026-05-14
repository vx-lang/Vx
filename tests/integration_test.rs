use akarc::lexer::Lexer;
use akarc::parser::Parser;
use akarc::sema::TypeChecker;

#[test]
fn test_distributed_matmul_integration() {
    let input = r#"
fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_data = transfer(a, Memory::NPU_HBM);
        let result = custom_matmul(local_data);
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
