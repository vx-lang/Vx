use crate::ast;
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

impl From<&str> for Function {
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
        self.peek_n(0)
    }

    pub(crate) fn peek_n(&self, offset: usize) -> &Token {
        if self.pos + offset < self.tokens.len() {
            &self.tokens[self.pos + offset]
        } else {
            &self.tokens[self.tokens.len() - 1] // return EOF
        }
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

    pub(crate) fn parse_token_tree(&mut self) -> Result<TokenTree, String> {
        let peek = self.peek();
        match &peek.kind {
            TokenType::LeftParen | TokenType::LeftBrace | TokenType::LeftBracket => {
                let delim_kind = peek.kind.clone();
                let delim = match delim_kind {
                    TokenType::LeftParen => Delimiter::Parenthesis,
                    TokenType::LeftBrace => Delimiter::Brace,
                    TokenType::LeftBracket => Delimiter::Bracket,
                    _ => unreachable!(),
                };
                let closing_delim = match delim_kind {
                    TokenType::LeftParen => TokenType::RightParen,
                    TokenType::LeftBrace => TokenType::RightBrace,
                    TokenType::LeftBracket => TokenType::RightBracket,
                    _ => unreachable!(),
                };
                self.advance(); // consume opening delimiter
                let mut inner = Vec::new();
                while !self.check(&closing_delim) && !self.check(&TokenType::Eof) {
                    inner.push(self.parse_token_tree()?);
                }
                self.consume(&closing_delim, "Expected closing delimiter")?;
                Ok(TokenTree::Delimited(delim, inner))
            }
            TokenType::Eof => {
                let token = self.peek();
                Err(crate::error::format_compiler_error(
                    self.source,
                    token.line,
                    token.column,
                    token.length.max(1),
                    "Unexpected EOF while parsing token tree",
                ))
            }
            _ => {
                let token = self.advance().clone();
                Ok(TokenTree::Token(token))
            }
        }
    }
}
