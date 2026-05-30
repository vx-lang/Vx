pub mod decl;
pub mod expr;
pub mod stmt;
pub mod types;

//===- parser.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Recursive Descent Parser for the Vx language.
// It consumes tokens produced by the lexer and constructs the hierarchical Abstract
// Syntax Tree (AST), enforcing the grammatical rules and structural syntax of
// the Vx language.
//
//===----------------------------------------------------------------------===//
use crate::ast::*;
use crate::lexer::{Token, TokenType};

pub struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    generic_params: Vec<String>, // Tracks generic parameters in scope
    source: &'a str,
}

impl From<&str> for crate::ast::Function {
    fn from(source: &str) -> Self {
        // Strip out 'pub' keyword if the user provided it as an example,
        // since Vx currently expects functions to start with 'fn'.
        let cleaned_source = source.trim().trim_start_matches("pub ");

        let mut lexer = crate::lexer::Lexer::new(cleaned_source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, cleaned_source);
        parser
            .parse_function()
            .expect("Failed to parse function source")
    }
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Token>, source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            generic_params: Vec::new(),
            source,
        }
    }

    pub(crate) fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    pub(crate) fn advance(&mut self) -> &Token {
        let token = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        token
    }

    pub(crate) fn check(&self, kind: &TokenType) -> bool {
        &self.peek().kind == kind
    }

    #[allow(dead_code)]
    pub(crate) fn match_token(&mut self, kind: &TokenType) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub(crate) fn consume(&mut self, kind: &TokenType, msg: &str) -> Result<&Token, String> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            let token = self.peek();
            Err(crate::error::format_compiler_error(
                self.source,
                token.line,
                token.column,
                token.length.max(1),
                msg,
            ))
        }
    }
}
