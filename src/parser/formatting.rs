//! Read source ranges and check grammar coverage without compiler analysis or source transformations.

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
    pub(crate) fn formatting_inherit_generics(&mut self, names: Vec<String>) {
        self.generic_params.extend(names);
    }

    pub(crate) fn formatting_scope_generics(&mut self) -> ParseResult<'_, Vec<String>> {
        if self.match_token(&TokenType::Impl) {
        } else if self.match_token(&TokenType::Trait) {
            self.expect_identifier("Expected trait name")?;
        } else {
            return Ok(Vec::new());
        }
        self.parse_generic_params()?;
        Ok(self.generic_params.clone())
    }

    pub(crate) fn formatting_body(&mut self) -> ParseResult<'_, ()> {
        if self.match_token(&TokenType::Semicolon) {
            return Ok(());
        }
        self.consume(&TokenType::LeftBrace, "Expected function body")?;
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut Vec::new())?;
        }
        self.consume(&TokenType::RightBrace, "Expected closing body delimiter")?;
        Ok(())
    }

    pub(crate) fn formatting_statements(&mut self) -> ParseResult<'_, ()> {
        while !self.check(&TokenType::Eof) {
            self.parse_statement_into(&mut Vec::new())?;
        }
        Ok(())
    }

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
        let return_type = if self.match_token(&TokenType::Arrow) {
            let start = self.pos;
            self.parse_return_type()?;
            start..self.pos
        } else if matches!(
            self.peek().kind,
            TokenType::LeftBrace | TokenType::Semicolon
        ) {
            // Keep support for incomplete editor signatures that the formatter already accepted.
            self.pos..self.pos
        } else {
            return Err(self.error("Unsupported function header suffix"));
        };
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
