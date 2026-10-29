//! Read source ranges with the existing grammar, without parsing or transforming a function body.

use super::{ParseResult, Parser};
use crate::lexer::TokenType;
use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ClauseKind {
    On,
    Requires,
    Ensures,
    Where,
}

#[derive(Debug)]
pub(crate) struct HeaderRanges {
    pub name: usize,
    pub generics: Range<usize>,
    pub open: usize,
    pub close: usize,
    pub parameters: Vec<(Range<usize>, Range<usize>)>,
    pub return_type: Range<usize>,
    pub clauses: Vec<(ClauseKind, Range<usize>)>,
    pub terminator: usize,
}

impl Parser<'_> {
    pub(crate) fn formatting_header(&mut self) -> ParseResult<'_, HeaderRanges> {
        self.match_token(&TokenType::Unsafe);
        self.consume(&TokenType::Fn, "Expected 'fn'")?;
        let name = self.pos;
        self.expect_identifier("Expected function name")?;
        let generic_start = self.pos;
        self.parse_generic_params()?;
        let generics = generic_start..self.pos;
        let open = self.pos;
        self.consume(&TokenType::LeftParen, "Expected '('")?;
        let mut parameters = Vec::new();
        while !self.check(&TokenType::RightParen) {
            let start = self.pos;
            self.match_token(&TokenType::Mut);
            self.expect_identifier("Expected parameter name")?;
            self.consume(&TokenType::Colon, "Expected ':'")?;
            let type_start = self.pos;
            self.parse_type()?;
            parameters.push((start..self.pos, type_start..self.pos));
            if !self.match_token(&TokenType::Comma) {
                break;
            }
            if self.check(&TokenType::RightParen) {
                return Err(self.error("Trailing parameter commas are not supported"));
            }
        }
        let close = self.pos;
        self.consume(&TokenType::RightParen, "Expected ')'")?;
        let mut clauses = Vec::new();
        if self.check(&TokenType::On) {
            let start = self.pos;
            self.advance();
            self.parse_topology()?;
            clauses.push((ClauseKind::On, start..self.pos));
        }
        self.consume(&TokenType::Arrow, "Expected '->'")?;
        let type_start = self.pos;
        self.parse_return_type()?;
        let return_type = type_start..self.pos;
        for (keyword, kind) in [
            (TokenType::Requires, ClauseKind::Requires),
            (TokenType::Ensures, ClauseKind::Ensures),
        ] {
            while self.check(&keyword) {
                let start = self.pos;
                self.advance();
                self.parse_expr()?;
                clauses.push((kind.clone(), start..self.pos));
            }
        }
        if self.check(&TokenType::Where) {
            let start = self.pos;
            self.advance();
            loop {
                let constraint = self.expect_identifier("Expected Reachable constraint")?;
                if constraint != "Reachable" {
                    return Err(self.error("Unsupported where constraint"));
                }
                self.consume(&TokenType::LeftAngle, "Expected '<'")?;
                self.expect_identifier("Expected topology name")?;
                self.consume(&TokenType::Comma, "Expected ','")?;
                self.expect_identifier("Expected topology name")?;
                self.consume(&TokenType::RightAngle, "Expected '>'")?;
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            clauses.push((ClauseKind::Where, start..self.pos));
        }
        if !matches!(
            self.peek().kind,
            TokenType::LeftBrace | TokenType::Semicolon
        ) {
            return Err(self.error("Unsupported function header suffix"));
        }
        Ok(HeaderRanges {
            name,
            generics,
            open,
            close,
            parameters,
            return_type,
            clauses,
            terminator: self.pos,
        })
    }
}
