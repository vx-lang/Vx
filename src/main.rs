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
    if args.len() > 1 {
        let filename = &args[1];
        let source = match fs::read_to_string(filename) {
            Ok(src) => src,
            Err(e) => {
                eprintln!("Failed to open file: {} - {}", filename, e);
                return;
            }
        };

        let mut lexer = Lexer::new(&source);
        let tokens = lexer.tokenize();
        
        println!("Successfully lexed {} tokens.", tokens.len());
        for t in tokens {
            println!("Token: [{:?}] at line {}:{}", t.kind, t.line, t.column);
        }
    } else {
        println!("Usage: akarc <source_file.ak>");
    }
}
