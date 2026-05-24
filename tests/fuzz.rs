use proptest::prelude::*;
use std::panic;
use vxc::lexer::Lexer;
use vxc::parser::Parser;

fn fuzz_parser(input: &str) {
    let _ = panic::catch_unwind(|| {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let _ = parser.parse();
    });
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]
    #[test]
    fn test_parser_does_not_panic(s in "\\PC*") {
        fuzz_parser(&s);
    }

    #[test]
    fn test_parser_ascii(s in "[ -~]*") {
        fuzz_parser(&s);
    }

    #[test]
    fn test_parser_autodiff(s in "(grad|vjp|jvp) *\\( *[a-zA-Z_][a-zA-Z0-9_]* *(, *[a-zA-Z_][a-zA-Z0-9_]*)* *\\) *") {
        fuzz_parser(&s);
    }
}
