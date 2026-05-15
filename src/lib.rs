pub mod ast;
pub mod ast_printer;
pub mod borrow;
pub mod codegen;
pub mod error;
pub mod formatter;
pub mod gid;
pub mod hash;
pub mod jit;
pub mod lexer;
pub mod parser;
pub mod pipeline;
pub mod registry;
pub mod sema;

/// Convenience API for parsing a string representation of a module into an VxModule (AST).
/// Useful for unit testing and interactive REPLs.
pub fn parse_module(source: &str) -> Result<ast::VxModule, String> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize()?;
    let mut parser = parser::Parser::new(tokens);
    parser.parse()
}
