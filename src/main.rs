pub mod ast;
pub mod lexer;
pub mod parser;
use lexer::Lexer;
use std::env;
use std::fs;

fn main() {
    println!("Akar Compiler (akarc) - Bootstrap Phase (Rust)");
    println!("==============================================");

    let args: Vec<String> = env::args().collect();
    let mut parse_only = false;
    let mut filename = "";

    for arg in args.iter().skip(1) {
        if arg == "-p" {
            parse_only = true;
        } else {
            filename = arg;
        }
    }

    if !filename.is_empty() {
        let source = match fs::read_to_string(filename) {
            Ok(src) => src,
            Err(e) => {
                eprintln!("Failed to open file: {} - {}", filename, e);
                return;
            }
        };

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        
        if parse_only {
            let mut parser = parser::Parser::new(tokens);
            match parser.parse() {
                Ok(ast) => println!("{:#?}", ast),
                Err(e) => eprintln!("Parse Error: {}", e),
            }
        } else {
            println!("Successfully lexed {} tokens.", tokens.len());
            for t in tokens {
                println!("Token: [{:?}] at line {}:{}", t.kind, t.line, t.column);
            }
        }
    } else {
        println!("Usage: akarc [-p] <source_file.ak>");
    }
}
