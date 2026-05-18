pub mod ast;
pub mod ast_printer;
pub mod borrow;
pub mod codegen;
pub mod error;
pub mod formatter;
pub mod gid;
pub mod hash;
pub mod hir;
pub mod jit;
pub mod lexer;
pub mod metadata;
pub mod module_loader;
pub mod parallel_architecture_verifier;
pub mod parser;
pub mod pipeline;
pub mod plugin;
pub mod registry;
pub mod resolver;
pub mod sema;
pub mod session;

/// Convenience API for parsing a string representation of a module into an VxModule (AST).
/// Useful for unit testing and interactive REPLs.
pub fn parse_module(source: &str) -> Result<ast::VxModule, String> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = parser::Parser::new(tokens, source);
    parser.parse()
}
