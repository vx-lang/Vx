use std::panic;
use vxc::lexer::Lexer;
use vxc::parser::Parser;

fn fuzz_parser(input: &str) {
    let result = panic::catch_unwind(|| {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let _ = parser.parse();
    });

    if result.is_err() {
        println!("Parser panicked on input: {:?}", input);
        std::process::exit(1);
    }
}

#[test]
fn test_custom_fuzzer() {
    let inputs = vec![
        "fn main() -> i32 { return 1 + 2; }",
        "fn main() { let a = 1; }",
        "let x: f32 = 1.0;",
        "()()()(((())))",
        "fn main() -> Tensor<f32, [10, 10]> { return a * b; }",
        "struct Test { a: i32, b: f32 }",
        "fn bad_syntax( { } () -> -> ",
        "let a: bool = true && false || !true;",
        "+ - * / % ^ & | ! ~ < > = <= >= == !=",
        "let a = Tensor_f32(1, 2, 3);",
        "return return return return",
    ];

    // Fuzz with permutations and garbage
    for _ in 0..100 {
        for base in &inputs {
            let garbage = format!("{} garbage {} more {}", base, base, base);
            fuzz_parser(&garbage);
        }
    }
}
