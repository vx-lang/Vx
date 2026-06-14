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

#[derive(Debug, Clone, PartialEq)]
pub enum ParserError {
    UnexpectedToken {
        expected: String,
        found: crate::lexer::OwnedToken,
    },
    Custom {
        message: String,
        token: crate::lexer::OwnedToken,
    },
    EndOfFile {
        expected: String,
    },
}

impl ParserError {
    pub fn format(&self, source: &str) -> String {
        match self {
            ParserError::UnexpectedToken { expected, found } => {
                crate::error::format_compiler_error(
                    source,
                    found.line,
                    found.column,
                    found.length.max(1),
                    &format!("Unexpected token {:?}. Expected {}", found.kind, expected),
                )
            }
            ParserError::Custom { message, token } => crate::error::format_compiler_error(
                source,
                token.line,
                token.column,
                token.length.max(1),
                message,
            ),
            ParserError::EndOfFile { expected } => {
                format!("Unexpected end of file. Expected {}", expected)
            }
        }
    }
}

pub(crate) type ParseResult<T> = Result<T, ParserError>;

pub struct Parser<'a> {
    tokens: &'a [Token<'a>],
    pos: usize,
    generic_params: Vec<String>, // Tracks generic parameters in scope
    source: &'a str,
}

impl<'a> Parser<'a> {
    pub fn parse_fn_from_str(source: &'a str) -> Function {
        let cleaned_source = source.trim().trim_start_matches("pub ");
        let mut lexer = crate::lexer::Lexer::new(cleaned_source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, cleaned_source);
        parser
            .parse_function()
            .map_err(|e| e.format(cleaned_source))
            .expect("Failed to parse function source")
    }
    pub fn new(tokens: &'a [Token<'a>], source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            generic_params: Vec::new(),
            source,
        }
    }

    pub(crate) fn peek(&self) -> &'a Token<'a> {
        self.peek_n(0)
    }

    pub(crate) fn peek_n(&self, offset: usize) -> &'a Token<'a> {
        self.tokens
            .get(self.pos + offset)
            .unwrap_or_else(|| self.tokens.last().expect("Token stream cannot be empty"))
    }

    pub(crate) fn advance(&mut self) -> &'a Token<'a> {
        let token = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        token
    }

    pub(crate) fn check(&self, kind: &TokenType<'a>) -> bool {
        &self.peek().kind == kind
    }

    #[allow(dead_code)]
    pub(crate) fn match_token(&mut self, kind: &TokenType<'a>) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    pub(crate) fn consume(
        &mut self,
        kind: &TokenType<'a>,
        msg: &str,
    ) -> ParseResult<&'a Token<'a>> {
        if self.check(kind) {
            Ok(self.advance())
        } else if self.peek().kind == TokenType::Eof {
            Err(ParserError::EndOfFile {
                expected: msg.to_string(),
            })
        } else {
            Err(ParserError::UnexpectedToken {
                expected: msg.to_string(),
                found: self.peek().clone().into_owned(),
            })
        }
    }

    pub(crate) fn error(&self, msg: &str) -> ParserError {
        let token = self.peek();
        ParserError::Custom {
            message: msg.to_string(),
            token: token.clone().into_owned(),
        }
    }

    pub(crate) fn parse_token_tree(&mut self) -> ParseResult<TokenTree> {
        let peek = self.peek();
        match &peek.kind {
            TokenType::LeftParen | TokenType::LeftBrace | TokenType::LeftBracket => {
                let delim = match &peek.kind {
                    TokenType::LeftParen => Delimiter::Parenthesis,
                    TokenType::LeftBrace => Delimiter::Brace,
                    TokenType::LeftBracket => Delimiter::Bracket,
                    _ => unreachable!(),
                };
                let closing_delim = match &peek.kind {
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
                Err(ParserError::Custom {
                    message: "Unexpected EOF while parsing token tree".to_string(),
                    token: token.clone().into_owned(),
                })
            }
            _ => {
                let token = self.advance().clone().into_owned();
                Ok(TokenTree::Token(token))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    #[test]
    fn test_parser_error_unexpected_token() {
        let input = "fn foo +";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);

        let err = parser
            .consume(&TokenType::LeftParen, "Expected '('")
            .unwrap_err();
        match &err {
            ParserError::UnexpectedToken { expected, found } => {
                assert!(expected.contains("("));
                assert_eq!(found.kind, crate::lexer::OwnedTokenType::Fn);
            }
            _ => panic!("Expected UnexpectedToken error, got {:?}", err),
        }

        let formatted = err.format(input);
        assert!(formatted.contains("("));
        assert!(formatted.contains("fn foo +"));
    }

    #[test]
    fn test_parser_error_eof() {
        let input = "";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);

        let err = parser
            .consume(&TokenType::LeftParen, "Expected '('")
            .unwrap_err();
        match &err {
            ParserError::EndOfFile { expected } => {
                assert!(expected.contains("("));
            }
            _ => panic!("Expected EndOfFile error, got {:?}", err),
        }

        let formatted = err.format(input);
        assert!(formatted.contains("Unexpected end of file"));
        assert!(formatted.contains("("));
    }
}
