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
use crate::lexer::{Token, TokenType};
use crate::syntax::*;

#[derive(Debug, Clone, PartialEq)]
pub enum ParserError<'a> {
    UnexpectedToken {
        expected: String,
        found: crate::lexer::Token<'a>,
    },
    Custom {
        message: String,
        token: crate::lexer::Token<'a>,
    },
    EndOfFile {
        expected: String,
    },
}

impl<'a> ParserError<'a> {
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

pub(crate) type ParseResult<'a, T> = Result<T, ParserError<'a>>;

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
        let index = self.pos + offset;
        if index < self.tokens.len() {
            &self.tokens[index]
        } else {
            self.tokens.last().expect("Token stream cannot be empty")
        }
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
    ) -> ParseResult<'a, &'a Token<'a>> {
        if self.check(kind) {
            Ok(self.advance())
        } else if self.peek().kind == TokenType::Eof {
            Err(ParserError::EndOfFile {
                expected: msg.to_string(),
            })
        } else {
            Err(ParserError::UnexpectedToken {
                expected: msg.to_string(),
                found: self.peek().clone(),
            })
        }
    }

    pub(crate) fn error(&self, msg: &str) -> ParserError<'a> {
        let token = self.peek();
        ParserError::Custom {
            message: msg.to_string(),
            token: token.clone(),
        }
    }

    pub(crate) fn error_at(&self, token: &Token<'a>, msg: &str) -> ParserError<'a> {
        ParserError::Custom {
            message: msg.to_string(),
            token: token.clone(),
        }
    }

    pub(crate) fn parse_token_tree(&mut self) -> ParseResult<'a, TokenTree> {
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
                    token: token.clone(),
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
                assert_eq!(found.kind, crate::lexer::TokenTypeBase::Fn);
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

    fn parse_type(input: &str) -> crate::syntax::Type {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser.parse_type().expect("Failed to parse type")
    }

    fn parse_topology(input: &str) -> crate::syntax::Topology {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser.parse_topology().expect("Failed to parse topology")
    }

    fn parse_memory_space(input: &str) -> crate::syntax::MemorySpace {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser
            .parse_memory_space()
            .expect("Failed to parse memory space")
    }

    #[test]
    fn test_parse_topology_static_variants() {
        assert_eq!(parse_topology("Topology::CPU"), Topology::CPU);
        assert_eq!(parse_topology("Topology::AMX"), Topology::AMX);
        assert_eq!(parse_topology("Topology::ANE"), Topology::ANE);
        assert_eq!(parse_topology("Topology::GPU"), Topology::GPU);
        assert_eq!(parse_topology("Topology::Current"), Topology::Current);
        assert_eq!(parse_topology("Topology::CpuAvx512"), Topology::CpuAvx512);
        assert_eq!(parse_topology("Topology::CPU_AVX512"), Topology::CpuAvx512);
        assert_eq!(parse_topology("Topology::CpuNeon"), Topology::CpuNeon);
        assert_eq!(parse_topology("Topology::CPU_Neon"), Topology::CpuNeon);
    }

    #[test]
    fn test_parse_topology_custom() {
        // Any identifier that is not a built-in is a user-defined topology.
        assert_eq!(
            parse_topology("Topology::MyTPU"),
            Topology::Custom(crate::symbol::Symbol::from("MyTPU"))
        );
    }

    #[test]
    fn test_parse_topology_decl_registers_descriptor() {
        // A `Topology <Name> { memory: ... }` declaration registers a descriptor. Uses a
        // unique name so parse-time global registration can't perturb other tests.
        let input = "Topology MyDeclTPU { memory: Memory::Local_SRAM, visible: [Memory::CPU_DRAM] }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser.parse_topology_decl().unwrap();

        let d = crate::arch::topology_descriptor(&crate::syntax::TopologyKind::Custom(
            crate::symbol::Symbol::from("MyDeclTPU"),
        ))
        .expect("descriptor should be registered by the declaration");
        assert_eq!(d.default_space, crate::syntax::MemorySpace::LocalSRAM);
        // Declared `visible` plus the always-visible default space.
        assert!(d.visibility.contains(&crate::syntax::MemorySpace::CPUDRAM));
        assert!(d.visibility.contains(&crate::syntax::MemorySpace::LocalSRAM));
    }

    #[test]
    fn test_parse_topology_npu_with_index() {
        let top = parse_topology("Topology::NPU[0]");
        if let Topology::NPU(expr) = top {
            if let Expr::Number(n) = &*expr {
                assert_eq!(n.value.as_ref(), "0");
            } else {
                panic!("Expected Number expression in NPU index");
            }
        } else {
            panic!("Expected NPU topology, got {:?}", top);
        }
    }

    #[test]
    fn test_parse_topology_acccore_with_index() {
        let top = parse_topology("Topology::AccCore[2]");
        if let Topology::AccCore(expr) = top {
            if let Expr::Number(n) = &*expr {
                assert_eq!(n.value.as_ref(), "2");
            } else {
                panic!("Expected Number expression in AccCore index");
            }
        } else {
            panic!("Expected AccCore topology, got {:?}", top);
        }
    }

    #[test]
    fn test_parse_topology_non_builtin_is_custom() {
        // The topology set is open: a non-built-in identifier parses as a user-defined
        // topology rather than erroring (its memory model comes from the registry).
        assert_eq!(
            parse_topology("Topology::Quantum"),
            Topology::Custom(crate::symbol::Symbol::from("Quantum"))
        );
    }

    #[test]
    fn test_parse_topology_npu_missing_index_error() {
        // NPU without [] should fail
        let input = "Topology::NPU";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let err = parser.parse_topology().unwrap_err();
        let msg = err.format(input);
        assert!(msg.contains("Expected index for NPU"), "Got: {}", msg);
    }

    #[test]
    fn test_parse_memory_space_all_variants() {
        assert_eq!(parse_memory_space("Memory::CPU_DRAM"), MemorySpace::CPUDRAM);
        assert_eq!(parse_memory_space("Memory::NPU_HBM"), MemorySpace::NPUHBM);
        assert_eq!(
            parse_memory_space("Memory::Local_SRAM"),
            MemorySpace::LocalSRAM
        );
        assert_eq!(parse_memory_space("Memory::NIC_RAM"), MemorySpace::NicRam);
        assert_eq!(
            parse_memory_space("Memory::Remote_HBM"),
            MemorySpace::RemoteHbm
        );
    }

    #[test]
    fn test_parse_memory_space_unknown_error() {
        let input = "Memory::GDDR";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let err = parser.parse_memory_space().unwrap_err();
        let msg = err.format(input);
        assert!(msg.contains("Unknown memory space GDDR"), "Got: {}", msg);
    }

    #[test]
    fn test_parse_tensor_type_plain() {
        let ty = parse_type("Tensor");
        assert_eq!(
            ty,
            Type::Tensor(ElementType::F32, vec![], None),
            "Plain Tensor should default to f32"
        );
    }

    #[test]
    fn test_parse_tensor_type_with_element_type() {
        let ty = parse_type("Tensor<i64>");
        assert_eq!(ty, Type::Tensor(ElementType::I64, vec![], None));
    }

    #[test]
    fn test_parse_pinned_type() {
        let ty = parse_type("Pinned<i32, Topology::GPU>");
        if let Type::Pinned(inner, top) = ty {
            assert_eq!(*inner, Type::Scalar(ElementType::I32));
            assert_eq!(top, Topology::GPU);
        } else {
            panic!("Expected Pinned type, got {:?}", ty);
        }
    }

    #[test]
    fn test_parse_ref_type() {
        let ty = parse_type("Ref<f32, Memory::NPU_HBM>");
        if let Type::Ref(inner, mem) = ty {
            assert_eq!(*inner, Type::Scalar(ElementType::F32));
            assert_eq!(mem, MemorySpace::NPUHBM);
        } else {
            panic!("Expected Ref type, got {:?}", ty);
        }
    }

    #[test]
    fn test_parse_topology_npu_slice() {
        let top = parse_topology("Topology::NPU[0..4]");
        if let Topology::Slice(inner_top, start, end) = top {
            // Inner topology should be NPU(0)
            assert!(matches!(*inner_top, Topology::NPU(_)));
            // Start should be 0, end should be 4
            if let Expr::Number(n) = *start {
                assert_eq!(n.value, "0".into());
            } else {
                panic!("Expected Number for slice start, got {:?}", start);
            }
            if let Expr::Number(n) = *end {
                assert_eq!(n.value, "4".into());
            } else {
                panic!("Expected Number for slice end, got {:?}", end);
            }
        } else {
            panic!("Expected Topology::Slice, got {:?}", top);
        }
    }
}
