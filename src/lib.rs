//===- lib.rs - Vx Compiler ------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file is the primary library entry point for the Vx compiler crate.
// It exports all public modules, encompassing the lexer, parser, semantic analyzer,
// borrow checker, and the MLIR code generation backend, allowing the compiler to
// be embedded or tested modularly.
//
//===----------------------------------------------------------------------===//
pub mod arch;
pub mod ast;
pub mod ast_printer;
pub mod borrow;
pub mod codegen;
pub mod diagnostic;
pub mod driver;
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
pub mod scratch;
pub mod sema;
pub mod session;
pub mod symbol;

/// Convenience API for parsing a string representation of a module into an VxModule (AST).
/// Useful for unit testing and interactive REPLs.
pub fn parse_module(source: &str) -> Result<ast::VxModule, crate::error::Error> {
    let mut lexer = lexer::Lexer::new(source);
    let tokens = lexer.tokenize();
    let mut parser = parser::Parser::new(&tokens, source);
    parser
        .parse()
        .map_err(|e| crate::error::Error(e.format(source)))
}
