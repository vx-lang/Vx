pub mod decl;
pub mod expr;
pub mod macro_expand;
pub mod stmt;
pub mod types;
pub use macro_expand::MacroExpander;

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

    #[test]
    fn a_type_argument_that_is_not_a_plain_name_parses_as_a_type() {
        use crate::parser::types::parse_type_text;
        use crate::syntax::{ElementType, Type};

        // A turbofish argument reaches the checker as text. A pointer spelling has to come back
        // out as a pointer, not as a generic parameter whose name is `*mut i8` (Vx#415).
        assert_eq!(
            parse_type_text("Option<*mut i8>"),
            Some(Type::GenericInstance(
                Box::new(Type::Struct("Option".into(), None)),
                vec![Type::Pointer(
                    Box::new(Type::Scalar(ElementType::I8)),
                    None,
                    true
                )],
            ))
        );
        assert_eq!(
            parse_type_text("Option<i32>"),
            Some(Type::GenericInstance(
                Box::new(Type::Struct("Option".into(), None)),
                vec![Type::Scalar(ElementType::I32)],
            ))
        );
        // A bare name is the nominal itself, and trailing text is not a type at all.
        assert_eq!(
            parse_type_text("Option"),
            Some(Type::Struct("Option".into(), None))
        );
        assert_eq!(parse_type_text("i32 and then some"), None);
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
        assert_eq!(parse_topology("Topology::GPU"), Topology::gpu(0));
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
    fn test_parse_topology_decl_returns_descriptor() {
        // A `Topology <Name> { memory: ... }` declaration parses into a `TopologyDecl` carried on
        // the AST (no global registration) -- the parser returns it.
        let input =
            "Topology MyDeclTPU { memory: Memory::Local_SRAM, visible: [Memory::CPU_DRAM, Memory::Local_SRAM] }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let decl = parser.parse_topology_decl().unwrap();

        assert_eq!(decl.name, crate::symbol::Symbol::from("MyDeclTPU"));
        assert_eq!(
            decl.descriptor.default_space,
            crate::syntax::MemorySpace::LocalSRAM
        );
        // Declared `visible` plus the always-visible default space.
        assert!(decl
            .descriptor
            .visibility
            .contains(&crate::syntax::MemorySpace::CPUDRAM));
        assert!(decl
            .descriptor
            .visibility
            .contains(&crate::syntax::MemorySpace::LocalSRAM));
    }

    #[test]
    fn test_parse_topology_decl_with_transfer() {
        // A `transfer <from> -> <to> : <cost>` clause becomes a declared morphism (a cost
        // graph edge) on the returned descriptor.
        let input = "Topology EdgeTPU { memory: Memory::Local_SRAM \
                     transfer Memory::GPU_HBM -> Memory::Local_SRAM : 7 }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let decl = parser.parse_topology_decl().unwrap();

        assert_eq!(
            decl.descriptor.transfers,
            vec![crate::arch::TransferEdge {
                from: crate::syntax::MemorySpace::GpuHbm,
                to: crate::syntax::MemorySpace::LocalSRAM,
                cost: crate::arch::EdgeCost::Fixed(7),
                sync: true,
                copy_engine: false,
            }]
        );
    }

    #[test]
    fn test_parse_transfer_edge_markers_compose() {
        // The trailing markers compose in any order: `relaxed copy_engine` declares a
        // relaxed edge with a hardware copy engine (Vx#353 A2). Each is independent --
        // the consistency grade answers visibility, the engine answers capability.
        let input = "Topology EdgeTPU { memory: Memory::Local_SRAM \
                     transfer Memory::GPU_HBM -> Memory::Local_SRAM relaxed copy_engine }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let decl = parser.parse_topology_decl().unwrap();

        assert_eq!(
            decl.descriptor.transfers,
            vec![crate::arch::TransferEdge {
                from: crate::syntax::MemorySpace::GpuHbm,
                to: crate::syntax::MemorySpace::LocalSRAM,
                cost: crate::arch::EdgeCost::Derived,
                sync: false,
                copy_engine: true,
            }]
        );
    }

    /// The three edge-cost forms (vx-review#10). An edge carries exactly one cost source, so the
    /// grammar has to be able to express "no cost declared" — otherwise a hop whose cost is
    /// derivable from its endpoints' bandwidths has no legal spelling, and every `fleet/` file was
    /// forced to declare a second one (E6013).
    #[test]
    fn transfer_edges_carry_one_of_three_cost_forms() {
        use crate::arch::EdgeCost;
        use crate::syntax::{Bandwidth, RatePer};
        let edges = |src: &str| {
            let mut lexer = Lexer::new(src);
            let tokens = lexer.tokenize();
            let mut p = Parser::new(&tokens, src);
            p.parse_topology_decl().unwrap().descriptor.transfers
        };

        // No cost: reachability only, cost comes from the endpoints' bandwidths.
        let e = edges("Topology T { memory: Memory::GPU_HBM, transfer Memory::GPU_HBM -> Memory::Local_SRAM }");
        assert_eq!(e[0].cost, EdgeCost::Derived);

        // A link bandwidth: for a hop whose endpoints do not nest, where containment can derive
        // nothing -- a host<->device link is a property of PCIe, not of either memory.
        let e = edges(
            "Topology T { memory: Memory::GPU_HBM, transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 64 GB/s }",
        );
        assert_eq!(
            e[0].cost,
            EdgeCost::Rate(Bandwidth {
                bytes: 64_000_000_000,
                per: RatePer::Second
            })
        );

        // A bare integer stays the legacy unitless relative cost, and the `relaxed` marker after
        // it still parses -- the disambiguation must not swallow the consistency grade.
        let e = edges("Topology T { memory: Memory::GPU_HBM, transfer Memory::GPU_HBM -> Memory::Local_SRAM : 300 relaxed }");
        assert_eq!(e[0].cost, EdgeCost::Fixed(300));
        assert!(!e[0].sync);
    }

    #[test]
    fn test_parse_topology_decl_custom_memory_space() {
        // A declaration can name a novel (user-defined) memory space, not just a built-in.
        let input = "Topology AcmeMemTPU { memory: Memory::AcmeSRAM }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let decl = parser.parse_topology_decl().unwrap();

        assert_eq!(
            decl.descriptor.default_space,
            crate::syntax::MemorySpace::Custom(crate::symbol::Symbol::from("AcmeSRAM"))
        );
    }

    fn parse_memory_decl(input: &str) -> crate::syntax::MemoryDecl {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser
            .parse_memory_decl()
            .expect("Failed to parse memory declaration")
    }

    #[test]
    fn test_parse_memory_decl_full() {
        use crate::syntax::{Bandwidth, ByteSize, Management, MemorySpace, RatePer};
        let d = parse_memory_decl(
            "Memory SMEM { within: Memory::GPU_HBM, capacity: 228 KiB, \
             bandwidth: 128 B/cyc, managed: explicit, granule: 16 KiB }",
        );
        assert_eq!(d.name, crate::symbol::Symbol::from("SMEM"));
        assert_eq!(d.parent, Some(MemorySpace::GpuHbm));
        assert_eq!(d.capacity, Some(ByteSize(228 * 1024)));
        assert_eq!(
            d.bandwidth,
            Some(Bandwidth {
                bytes: 128,
                per: RatePer::Cycle
            })
        );
        assert_eq!(d.managed, Management::Explicit);
        assert_eq!(d.granule, Some(ByteSize(16 * 1024)));
    }

    #[test]
    fn test_parse_memory_decl_minimal_defaults() {
        use crate::syntax::Management;
        // Only a name: every field optional, `managed` defaults to `cached`.
        let d = parse_memory_decl("Memory RMEM {}");
        assert_eq!(d.name, crate::symbol::Symbol::from("RMEM"));
        assert_eq!(d.parent, None);
        assert_eq!(d.capacity, None);
        assert_eq!(d.bandwidth, None);
        assert_eq!(d.granule, None);
        assert_eq!(d.managed, Management::Cached);
    }

    /// A novel parent space, and the unit convention that the fleet's provenance depends on:
    /// **SI spellings are decimal, IEC binary** (`src/units.rs`).
    ///
    /// This is the regression test for a 9% error. `TB` was read as 2^40 while every `spec:`
    /// citation in `fleet/` quotes a vendor decimal figure, so the compiler treated each link as
    /// ~10% faster than its own citation claimed and understated every predicted transfer time.
    /// The two assertions below are deliberately spelled as literal products rather than as
    /// `ByteUnit::factor()` calls, so this test cannot agree with a wrong implementation by
    /// sharing its arithmetic.
    #[test]
    fn test_parse_memory_decl_custom_parent_and_units() {
        use crate::syntax::{Bandwidth, ByteSize, MemorySpace, RatePer};
        let d = parse_memory_decl(
            "Memory L2 { within: Memory::AcmePool, capacity: 192 GiB, bandwidth: 8 TB/s }",
        );
        assert_eq!(
            d.parent,
            Some(MemorySpace::Custom(crate::symbol::Symbol::from("AcmePool")))
        );
        // IEC capacity: binary.
        assert_eq!(d.capacity, Some(ByteSize(192 * 1024 * 1024 * 1024)));
        // SI bandwidth: decimal. 8 TB/s is 8e12 B/s, which is what a spec sheet means.
        assert_eq!(
            d.bandwidth,
            Some(Bandwidth {
                bytes: 8_000_000_000_000,
                per: RatePer::Second
            })
        );
    }

    /// The same digits under the two spellings must not produce the same bytes -- the property
    /// whose absence was the bug. Also pins that a fractional mantissa is exact: `3.35 TB/s` is
    /// 3.35e12 on the nose, not a float-rounded neighbour.
    #[test]
    fn si_and_iec_spellings_are_distinct_and_fractions_are_exact() {
        use crate::syntax::ByteSize;
        let si = parse_memory_decl("Memory A { capacity: 1 GB }");
        let iec = parse_memory_decl("Memory A { capacity: 1 GiB }");
        assert_eq!(si.capacity, Some(ByteSize(1_000_000_000)));
        assert_eq!(iec.capacity, Some(ByteSize(1_073_741_824)));
        assert_ne!(si.capacity, iec.capacity);

        let d = parse_memory_decl("Memory B { bandwidth: 3.35 TB/s }");
        assert_eq!(d.bandwidth.unwrap().bytes, 3_350_000_000_000);
    }

    #[test]
    fn test_memory_decl_collected_and_indexed_in_global_env() {
        // A `Memory` decl at top level lands on `Program.memories` (not a global registry),
        // is preserved by clone_signature, and is indexed by GlobalAstEnv for sema.
        let input = "Memory TMEM { capacity: 256 KiB }\nfn main() -> i32 { return 0; }";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().expect("program should parse");

        assert_eq!(program.memories.len(), 1);
        assert_eq!(
            program.memories[0].name,
            crate::symbol::Symbol::from("TMEM")
        );
        // Descriptors survive the signature clone used across the pipeline.
        assert_eq!(program.clone_signature().memories, program.memories);

        let programs = vec![program];
        let env = crate::hir::GlobalAstEnv::build(&programs);
        let m = env
            .memories
            .get(&crate::symbol::Symbol::from("TMEM"))
            .expect("TMEM should be indexed in GlobalAstEnv");
        assert_eq!(m.capacity, Some(crate::syntax::ByteSize(256 * 1024)));
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
    fn test_parse_memory_space_non_builtin_is_custom() {
        // The memory-space set is open: a non-built-in identifier parses as a user-defined
        // memory space rather than erroring.
        let input = "Memory::GDDR";
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        assert_eq!(
            parser.parse_memory_space().unwrap(),
            crate::syntax::MemorySpace::Custom(crate::symbol::Symbol::from("GDDR"))
        );
    }

    /// Every spelling that omits the shape is refused, and each of the three that state one
    /// parses to a different type. The dims-less `Tensor` used to carry all of them (Vx#399).
    #[test]
    fn a_tensor_without_a_shape_does_not_parse() {
        for input in ["Tensor", "Tensor<f32>", "Tensor<i64>"] {
            let mut lexer = Lexer::new(input);
            let tokens = lexer.tokenize();
            let mut parser = Parser::new(&tokens, input);
            let err = parser
                .parse_type()
                .expect_err("a Tensor without a shape must not parse");
            assert!(
                format!("{err:?}").contains("Tensor needs its shape"),
                "{input} was refused for the wrong reason: {err:?}"
            );
        }
    }

    #[test]
    fn each_stated_shape_parses_to_its_own_type() {
        assert_eq!(
            parse_type("Tensor<f32, []>"),
            Type::Tensor(ElementType::F32, vec![], None),
            "an empty dimension list is rank 0"
        );
        assert_eq!(
            parse_type("DynTensor<i64>"),
            Type::DynTensor(ElementType::I64, None),
            "a run-time shape is a DynTensor"
        );
        let shaped = parse_type("Tensor<f32, [2, 3]>");
        let Type::Tensor(el, dims, None) = shaped else {
            panic!("expected a shaped tensor, got {shaped:?}");
        };
        assert_eq!(el, ElementType::F32);
        assert_eq!(dims.len(), 2);
    }

    #[test]
    fn test_parse_pinned_type() {
        let ty = parse_type("Pinned<i32, Topology::GPU>");
        if let Type::Pinned(inner, top) = ty {
            assert_eq!(*inner, Type::Scalar(ElementType::I32));
            assert_eq!(top, Topology::gpu(0));
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
