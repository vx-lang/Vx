//===- decl.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx declarations, including functions, structs, impl blocks, and foreign modules.
//
//===----------------------------------------------------------------------===//

use super::*;

impl<'a> Parser<'a> {
    pub(crate) fn expect_identifier(&mut self, msg: &str) -> ParseResult<'a, String> {
        match self.advance().kind {
            TokenType::Identifier(s) => Ok(s.to_string()),
            TokenType::Transfer => Ok("transfer".to_string()),
            _ => Err(self.error(msg)),
        }
    }

    fn parse_comma_separated_params(
        &mut self,
    ) -> ParseResult<'a, Vec<(crate::symbol::Symbol, Type)>> {
        let mut params: Vec<(crate::symbol::Symbol, crate::syntax::types::Type)> = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                let name = self.expect_identifier("Expected parameter name")?;
                self.consume(&TokenType::Colon, "Expected ':'")?;
                let ty = self.parse_type()?;
                params.push((name.into(), ty));

                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        Ok(params)
    }

    pub(crate) fn parse_generic_params(&mut self) -> ParseResult<'a, Vec<GenericParam>> {
        let mut generics = Vec::new();
        if self.match_token(&TokenType::LeftAngle) {
            while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
                if self.check(&TokenType::Identifier("const")) {
                    self.advance(); // consume const
                    let name = self.expect_identifier("Expected const parameter name")?;
                    self.consume(&TokenType::Colon, "Expected ':' after const parameter name")?;
                    let ty = self.parse_type()?;
                    self.generic_params.push(name.clone());
                    generics.push(GenericParam::Const {
                        name: name.into(),
                        ty,
                    });
                } else {
                    let name = self.expect_identifier("Expected generic parameter name")?;
                    self.generic_params.push(name.clone());
                    let mut bound = None;
                    if self.match_token(&TokenType::Colon) {
                        bound = match self.advance().kind.clone() {
                            TokenType::Identifier(s) => Some(s.to_string()),
                            // `<D: Topology>` -- `Topology` is a keyword, not an identifier.
                            TokenType::Topology => Some("Topology".to_string()),
                            _ => return Err(self.error("Expected trait bound identifier")),
                        };
                    }
                    generics.push(GenericParam::Type {
                        name: name.into(),
                        bound: bound.map(|s| s.into()),
                    });
                }
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            self.consume(
                &TokenType::RightAngle,
                "Expected '>' after generic parameters",
            )?;
        }
        Ok(generics)
    }

    pub fn parse_function(&mut self) -> ParseResult<'a, Function> {
        self.consume(&TokenType::Fn, "Expected 'fn'")?;

        let name = self.expect_identifier("Expected function name")?;

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftParen, "Expected '(' after function name")?;
        let params = self.parse_comma_separated_params()?;
        self.consume(&TokenType::RightParen, "Expected ')'")?;

        let mut topology = Topology::CPU;
        if self.match_token(&TokenType::On) {
            topology = self.parse_topology()?;
        }

        if !self.match_token(&TokenType::Arrow) {
            // In Vx, '->' is currently required for functions in parse_function
            return Err(self.error("Expected '->'"));
        }
        let return_type = self.parse_type()?;

        let mut requires = Vec::new();
        while self.match_token(&TokenType::Requires) {
            requires.push(self.parse_expr()?);
        }

        let mut ensures = Vec::new();
        while self.match_token(&TokenType::Ensures) {
            ensures.push(self.parse_expr()?);
        }

        // `where Reachable<A, B> [, Reachable<C, D>]*` -- topology transfer constraints.
        //
        // The name says what the constraint actually tests: a transfer PATH exists from A to
        // B in the cost graph. It does not name any particular lowering, and its arguments are
        // topologies, not memory spaces. It was called `Transfer<A, B>` until Vx#353, which is
        // the name the edge lowering now uses (`impl Transfer<Memory::A, Memory::B> for
        // Topology::X`) with a different kind of argument -- so the two had to part.
        let mut where_transfers: Vec<(crate::symbol::Symbol, crate::symbol::Symbol)> = Vec::new();
        if self.match_token(&TokenType::Where) {
            loop {
                let cname = self
                    .expect_identifier("Expected a `Reachable<A, B>` constraint after 'where'")?;
                if cname != "Reachable" {
                    // `Transfer` gets its own message: it was the old spelling of exactly this
                    // constraint, so a program carrying it is out of date rather than wrong.
                    if cname == "Transfer" {
                        return Err(self.error(
                            "`where Transfer<A, B>` is now `where Reachable<A, B>`; the name \
                             `Transfer` belongs to the edge lowering (`impl Transfer<Memory::A, \
                             Memory::B> for Topology::X`), whose arguments are memory spaces",
                        ));
                    }
                    return Err(self.error(&format!(
                        "Unsupported where-constraint '{}' (only `Reachable<A, B>` is supported)",
                        cname
                    )));
                }
                self.consume(&TokenType::LeftAngle, "Expected '<' after Reachable")?;
                let a = self.expect_identifier("Expected a topology name in Reachable<A, B>")?;
                self.consume(&TokenType::Comma, "Expected ',' in Reachable<A, B>")?;
                let b = self.expect_identifier("Expected a topology name in Reachable<A, B>")?;
                self.consume(&TokenType::RightAngle, "Expected '>' after Reachable<A, B>")?;
                where_transfers.push((a.into(), b.into()));
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            body.push(self.parse_statement()?);
        }

        // Transform implicit return
        if let Some(Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi,
            span,
        })) = body.last().cloned()
        {
            if !has_semi {
                let lsyntax_idx = body.len() - 1;
                body[lsyntax_idx] = Statement::Return(ReturnStmt { expr, span });
            }
        }

        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(Function {
            name: name.into(),
            generics,
            params,
            topology,
            return_type,
            requires,
            ensures,
            where_transfers,
            body,
            doc_comment: None,
        })
    }

    /// Parse a user-defined topology declaration and register its descriptor:
    ///
    /// ```text
    /// Topology <Name> {
    ///     memory: Memory::<Space>            // required: default placement
    ///     visible: [Memory::<Space>, ...]    // optional: extra unified-memory reach
    /// }
    /// ```
    ///
    /// The whole effect is registering the descriptor in the global topology registry
    /// (`crate::arch`), so nothing is stored in the AST. `Topology::<Name>` uses elsewhere
    /// then resolve to `Topology::Custom(<Name>)` and pick up this description.
    pub(crate) fn parse_topology_decl(&mut self) -> ParseResult<'a, crate::arch::TopologyDecl> {
        self.consume(&TokenType::Topology, "Expected 'Topology'")?;
        let name = match &self.advance().kind {
            TokenType::Identifier(s) => crate::symbol::Symbol::from(*s),
            _ => return Err(self.error("Expected a name after 'Topology'")),
        };
        self.consume(
            &TokenType::LeftBrace,
            "Expected '{' in topology declaration",
        )?;

        let mut default_space: Option<MemorySpace> = None;

        let mut arch: Option<crate::symbol::Symbol> = None;
        let mut visibility: Vec<MemorySpace> = Vec::new();
        let mut visible_given = false;
        let mut transfers: Vec<crate::arch::TransferEdge> = Vec::new();

        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            // `transfer Memory::<From> -> Memory::<To> : <cost> [relaxed|sync]` -- a declared
            // morphism (an edge added to the cost graph), with an optional consistency grade
            // (default synchronizing). Uses the `transfer` keyword, so no colon.
            if self.check(&TokenType::Transfer) {
                self.advance();
                let from = self.parse_memory_space()?;
                self.consume(&TokenType::Arrow, "Expected '->' in transfer clause")?;
                let to = self.parse_memory_space()?;
                // The cost is optional, and when present may be either a link bandwidth
                // (`: 64 GB/s`) or a unitless relative cost (`: 300`). Omitting it declares
                // reachability and leaves the cost to the endpoints' bandwidths -- which is the
                // normal case for a hop between nested spaces, and the only way to avoid declaring
                // a second cost for an edge that already has a derivable one (E6013).
                let cost = if self.match_token(&TokenType::Colon) {
                    self.parse_edge_cost()?
                } else {
                    crate::arch::EdgeCost::Derived
                };
                // Optional trailing markers, in any order: a `relaxed` / `sync` consistency
                // grade (default synchronizing), and `copy_engine` -- the declaration that a
                // hardware engine (a DMA, Ampere's cp.async) can drive this hop. `copy_engine`
                // is what makes `raw::async_copy` legal in a lowering for this edge (Vx#353 A2).
                let mut sync = true;
                let mut copy_engine = false;
                while let TokenType::Identifier(s) = &self.peek().kind {
                    match *s {
                        "relaxed" => {
                            self.advance();
                            sync = false;
                        }
                        "sync" => {
                            self.advance();
                            sync = true;
                        }
                        "copy_engine" => {
                            self.advance();
                            copy_engine = true;
                        }
                        _ => break,
                    }
                }
                transfers.push(crate::arch::TransferEdge {
                    from,
                    to,
                    cost,
                    sync,
                    copy_engine,
                });
                self.match_token(&TokenType::Comma);
                continue;
            }

            let field = match &self.advance().kind {
                TokenType::Identifier(s) => s.to_string(),
                other => {
                    return Err(
                        self.error(&format!("Expected a topology field name, got {:?}", other))
                    )
                }
            };
            self.consume(&TokenType::Colon, "Expected ':' after topology field")?;
            match field.as_str() {
                // The instruction set this topology executes. A machine description that
                // does not say this cannot answer what code to emit for it, and nothing
                // else in the file implies it -- a filename and a comment are not readable
                // by the compiler.
                "arch" => {
                    arch = Some(match &self.advance().kind {
                        TokenType::Identifier(s) => crate::symbol::Symbol::from(&**s),
                        other => {
                            return Err(self.error(&format!(
                                "`arch:` expects an identifier (x86_64, aarch64, nvptx64, amdgcn), got {:?}",
                                other
                            )))
                        }
                    });
                }
                "memory" => default_space = Some(self.parse_memory_space()?),
                "visible" => {
                    visible_given = true;
                    self.consume(&TokenType::LeftBracket, "Expected '[' after 'visible'")?;
                    while !self.check(&TokenType::RightBracket) && !self.check(&TokenType::Eof) {
                        visibility.push(self.parse_memory_space()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                    self.consume(&TokenType::RightBracket, "Expected ']' after visible list")?;
                }
                other => return Err(self.error(&format!("Unknown topology field '{}'", other))),
            }
            self.match_token(&TokenType::Comma);
        }
        self.consume(
            &TokenType::RightBrace,
            "Expected '}' to close topology declaration",
        )?;

        let default_space = match default_space {
            Some(s) => s,
            None => return Err(self.error("topology declaration needs a `memory:` field")),
        };
        // With no explicit `visible`, a topology sees exactly its own default space. An
        // explicit `visible` list is respected verbatim, so an inconsistent one (default
        // not listed) is caught by the coherence check (E6005) rather than silently fixed.
        if !visible_given {
            visibility.push(default_space.clone());
        }

        // Carry the descriptor on the AST (no global registration): the caller pushes it onto
        // `Program.topologies`, and sema seeds the per-compilation cost graph from there.
        Ok(crate::arch::TopologyDecl {
            name,
            descriptor: crate::arch::TopologyDescriptor {
                arch,
                default_space,
                visibility,
                transfers,
            },
        })
    }

    /// `Memory <Name> { within:, capacity:, bandwidth:, managed:, granule: }` -- a first-class
    /// memory-space declaration. Every field is optional except the name. Returned as a
    /// `MemoryDecl` on the AST (no global registration, unlike `parse_topology_decl`).
    pub(crate) fn parse_memory_decl(&mut self) -> ParseResult<'a, crate::syntax::MemoryDecl> {
        self.consume(&TokenType::Memory, "Expected 'Memory'")?;
        let name = match &self.advance().kind {
            TokenType::Identifier(s) => crate::symbol::Symbol::from(*s),
            _ => return Err(self.error("Expected a name after 'Memory'")),
        };
        self.consume(&TokenType::LeftBrace, "Expected '{' in memory declaration")?;

        let mut parent: Option<MemorySpace> = None;
        let mut capacity: Option<crate::syntax::ByteSize> = None;
        let mut bandwidth: Option<crate::syntax::Bandwidth> = None;
        let mut clock_hz: Option<u64> = None;
        let mut replicas: Option<u64> = None;
        let mut managed = crate::syntax::Management::default();
        let mut granule: Option<crate::syntax::ByteSize> = None;
        let mut scope: Option<crate::syntax::Scope> = None;
        let mut overcommit = false;
        let mut crossing = crate::syntax::Crossing::default();

        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let field = match &self.advance().kind {
                TokenType::Identifier(s) => s.to_string(),
                other => {
                    return Err(
                        self.error(&format!("Expected a memory field name, got {:?}", other))
                    )
                }
            };
            // `overcommit` is a bare flag (no `: value`).
            if field == "overcommit" {
                overcommit = true;
                self.match_token(&TokenType::Comma);
                continue;
            }
            self.consume(&TokenType::Colon, "Expected ':' after memory field")?;
            match field.as_str() {
                "within" => parent = Some(self.parse_memory_space()?),
                "scope" => {
                    let s = match &self.advance().kind {
                        TokenType::Identifier(s) => s.to_string(),
                        other => {
                            return Err(self.error(&format!(
                                "`scope:` expects device/sm/cta/thread, got {:?}",
                                other
                            )))
                        }
                    };
                    scope = Some(match s.as_str() {
                        "device" => crate::syntax::Scope::Device,
                        "sm" => crate::syntax::Scope::Sm,
                        "cta" => crate::syntax::Scope::Cta,
                        "thread" => crate::syntax::Scope::Thread,
                        other => {
                            return Err(self.error(&format!(
                                "`scope:` expects device/sm/cta/thread, got '{}'",
                                other
                            )))
                        }
                    });
                }
                "capacity" => capacity = Some(self.parse_byte_size()?),
                "granule" => granule = Some(self.parse_byte_size()?),
                "bandwidth" => bandwidth = Some(self.parse_bandwidth()?),
                "clock" => clock_hz = Some(self.parse_clock()?),
                "replicas" => replicas = Some(self.parse_count()?),
                "crossing" => {
                    let kind = match &self.advance().kind {
                        TokenType::Identifier(s) => s.to_string(),
                        other => {
                            return Err(self.error(&format!(
                                "Expected 'streamed' or 'sequenced' for `crossing:`, got {:?}",
                                other
                            )))
                        }
                    };
                    crossing = match kind.as_str() {
                        "streamed" => crate::syntax::Crossing::Streamed,
                        "sequenced" => crate::syntax::Crossing::Sequenced,
                        other => {
                            return Err(self.error(&format!(
                                "`crossing:` expects 'streamed' or 'sequenced', got '{}'",
                                other
                            )))
                        }
                    };
                }
                "managed" => {
                    let kind = match &self.advance().kind {
                        TokenType::Identifier(s) => s.to_string(),
                        other => {
                            return Err(self.error(&format!(
                                "Expected 'explicit' or 'cached' for `managed:`, got {:?}",
                                other
                            )))
                        }
                    };
                    managed = match kind.as_str() {
                        "explicit" => crate::syntax::Management::Explicit,
                        "cached" => crate::syntax::Management::Cached,
                        other => {
                            return Err(self.error(&format!(
                                "`managed:` expects 'explicit' or 'cached', got '{}'",
                                other
                            )))
                        }
                    };
                }
                other => return Err(self.error(&format!("Unknown memory field '{}'", other))),
            }
            self.match_token(&TokenType::Comma);
        }
        self.consume(
            &TokenType::RightBrace,
            "Expected '}' to close memory declaration",
        )?;

        Ok(crate::syntax::MemoryDecl {
            name,
            parent,
            capacity,
            bandwidth,
            clock_hz,
            replicas,
            managed,
            granule,
            scope,
            overcommit,
            crossing,
            doc_comment: None,
        })
    }

    /// A size literal like `256 KiB`, normalized to bytes.
    ///
    /// SI spellings are decimal, IEC binary. Capacities are usually the IEC case -- a "228 KB"
    /// shared-memory figure is 228 KiB in fact -- so a file should say `KiB` and mean it rather
    /// than rely on the reader knowing which convention the parser picked.
    fn parse_byte_size(&mut self) -> ParseResult<'a, crate::syntax::ByteSize> {
        let mantissa = self.parse_number_text("a byte size")?;
        let unit = self.parse_byte_unit()?;
        Ok(crate::syntax::ByteSize(self.scaled_bytes(&mantissa, unit)?))
    }

    /// A bandwidth literal like `8 TB/s` or `128 B/cyc`.
    ///
    /// SI spellings are decimal and IEC binary (`crate::units`), so a figure copied from a vendor
    /// spec sheet means what the sheet meant. Bandwidth is the case that motivated the rule:
    /// NVIDIA's "3.35 TB/s" is 3.35e12 B/s, and reading `TB` as 2^40 understated every predicted
    /// transfer time by 9%.
    fn parse_bandwidth(&mut self) -> ParseResult<'a, crate::syntax::Bandwidth> {
        let mantissa = self.parse_number_text("a bandwidth")?;
        let unit = self.parse_byte_unit()?;
        let per = self.parse_rate_denominator()?;
        Ok(crate::syntax::Bandwidth {
            bytes: self.scaled_bytes(&mantissa, unit)?,
            per,
        })
    }

    /// A clock literal like `1.98 GHz`, stored in hertz.
    ///
    /// Exists so a `B/cyc` bandwidth can be converted against a `B/s` one exactly. Every fleet
    /// machine mixes the two -- L2 quoted in TB/s, SMEM in B/cyc -- and before this the mixed path
    /// was simply not derivable, which is how the `L2->SMEM` seam went unpriced.
    fn parse_clock(&mut self) -> ParseResult<'a, u64> {
        let mantissa = self.parse_number_text("a clock frequency")?;
        let unit_str = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            other => {
                return Err(self.error(&format!(
                    "Expected a frequency unit ({}) after `clock:`, got {:?}",
                    crate::units::FreqUnit::ALL,
                    other
                )))
            }
        };
        let unit = crate::units::FreqUnit::parse(&unit_str).ok_or_else(|| {
            self.error(&format!(
                "Unknown frequency unit '{}'; expected one of {}",
                unit_str,
                crate::units::FreqUnit::ALL
            ))
        })?;
        crate::units::to_hertz(&mantissa, unit).map_err(|e| {
            self.error(&format!(
                "`clock: {} {}` is not a whole number of hertz ({:?})",
                mantissa, unit_str, e
            ))
        })
    }

    /// A plain positive integer count, for `replicas:`.
    fn parse_count(&mut self) -> ParseResult<'a, u64> {
        let text = self.parse_number_text("a count")?;
        let n: u64 = text.parse().map_err(|_| {
            self.error(&format!(
                "`replicas:` expects a whole number, got '{}'",
                text
            ))
        })?;
        if n == 0 {
            return Err(self.error("`replicas:` must be positive"));
        }
        Ok(n)
    }

    /// The `/s` or `/cyc` of a rate.
    fn parse_rate_denominator(&mut self) -> ParseResult<'a, crate::syntax::RatePer> {
        self.consume(
            &TokenType::Slash,
            "Expected '/' in bandwidth (e.g. `8 TB/s`)",
        )?;
        let per_str = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            other => {
                return Err(self.error(&format!(
                    "Expected 's' or 'cyc' after '/' in bandwidth, got {:?}",
                    other
                )))
            }
        };
        match per_str.as_str() {
            "s" => Ok(crate::syntax::RatePer::Second),
            "cyc" => Ok(crate::syntax::RatePer::Cycle),
            other => Err(self.error(&format!(
                "bandwidth denominator must be 's' or 'cyc', got '{}'",
                other
            ))),
        }
    }

    /// The value after `transfer A -> B :` — either a link bandwidth (`64 GB/s`) or a unitless
    /// relative cost (`300`).
    ///
    /// Disambiguated by whether a byte unit follows the number. The only other things that may
    /// follow are the `relaxed`/`sync` markers and a `,`, and no unit spelling collides with those.
    fn parse_edge_cost(&mut self) -> ParseResult<'a, crate::arch::EdgeCost> {
        let mantissa = self.parse_number_text("a transfer cost or link bandwidth")?;
        let is_rate = matches!(
            &self.peek().kind,
            TokenType::Identifier(s) if crate::units::ByteUnit::parse(s).is_some()
        );
        if is_rate {
            let unit = self.parse_byte_unit()?;
            let bytes = self.scaled_bytes(&mantissa, unit)?;
            let per = self.parse_rate_denominator()?;
            Ok(crate::arch::EdgeCost::Rate(crate::syntax::Bandwidth {
                bytes,
                per,
            }))
        } else {
            let cost = mantissa.parse::<u32>().map_err(|_| {
                self.error(&format!(
                    "transfer cost must be a non-negative integer or a link bandwidth \
                     (e.g. `64 GB/s`), got '{mantissa}'"
                ))
            })?;
            Ok(crate::arch::EdgeCost::Fixed(cost))
        }
    }

    /// The source text of a number literal, unparsed.
    ///
    /// Text rather than `f64` so [`crate::units::to_bytes`] can do exact integer arithmetic: `3.35`
    /// is not representable in binary floating point, and a parse-multiply-round pipeline is
    /// correct only for the particular values it happens to be given.
    fn parse_number_text(&mut self, what: &str) -> ParseResult<'a, String> {
        match &self.advance().kind {
            TokenType::Number(s) => Ok(s.to_string()),
            other => Err(self.error(&format!("Expected {}, got {:?}", what, other))),
        }
    }

    /// Exact mantissa-times-unit, with the failure reported where the reader can see the figure.
    fn scaled_bytes(
        &mut self,
        mantissa: &str,
        unit: crate::units::ByteUnit,
    ) -> ParseResult<'a, u64> {
        crate::units::to_bytes(mantissa, unit).map_err(|e| {
            let why = match e {
                crate::units::UnitError::Malformed => "not a decimal number".to_string(),
                crate::units::UnitError::Overflow => {
                    "too large for a 64-bit byte count".to_string()
                }
                crate::units::UnitError::NotWholeBytes => format!(
                    "not a whole number of bytes ({mantissa} x {} does not divide evenly); \
                     write it in a smaller unit",
                    unit.factor()
                ),
            };
            self.error(&format!("invalid size '{mantissa}': {why}"))
        })
    }

    /// A byte-unit suffix. SI spellings (`GB`) are decimal, IEC (`GiB`) binary -- see
    /// [`crate::units`] for why both exist and why the distinction is enforced.
    fn parse_byte_unit(&mut self) -> ParseResult<'a, crate::units::ByteUnit> {
        let unit = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            other => {
                return Err(self.error(&format!(
                    "Expected a size unit ({}), got {:?}",
                    crate::units::ByteUnit::ALL,
                    other
                )))
            }
        };
        crate::units::ByteUnit::parse(&unit).ok_or_else(|| {
            self.error(&format!(
                "Unknown size unit '{}' (expected {})",
                unit,
                crate::units::ByteUnit::ALL
            ))
        })
    }

    pub(crate) fn parse_struct_decl(&mut self) -> ParseResult<'a, StructDecl> {
        self.consume(&TokenType::Struct, "Expected 'struct'")?;

        let name = self.expect_identifier("Expected struct name")?;

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut fields: Vec<(crate::symbol::Symbol, crate::syntax::types::Type)> = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let f_name = self.expect_identifier("Expected field name")?;
            self.consume(&TokenType::Colon, "Expected ':'")?;
            let f_type = self.parse_type()?;
            fields.push((f_name.into(), f_type));

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(StructDecl {
            name: name.into(),
            generics,
            fields,
            doc_comment: None,
        })
    }

    pub(crate) fn parse_enum_decl(&mut self) -> ParseResult<'a, EnumDecl> {
        self.consume(&TokenType::Enum, "Expected 'enum'")?;

        let name = self.expect_identifier("Expected enum name")?;

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut variants: Vec<(
            crate::symbol::Symbol,
            Option<Vec<crate::syntax::types::Type>>,
        )> = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let v_name = self.expect_identifier("Expected enum variant name")?;

            let mut payload = None;
            if self.match_token(&TokenType::LeftParen) {
                let mut types = Vec::new();
                if !self.check(&TokenType::RightParen) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')' after enum payload")?;
                payload = Some(types);
            }

            variants.push((v_name.into(), payload));

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(EnumDecl {
            name: name.into(),
            generics,
            variants,
            doc_comment: None,
        })
    }

    pub(crate) fn parse_extern_block(&mut self) -> ParseResult<'a, Vec<ExternDecl>> {
        self.consume(&TokenType::Extern, "Expected 'extern'")?;

        // Optional "C" ABI string literal (we ignore it for now but parse it if it exists)
        if let TokenType::StringLiteral(s) = &self.peek().kind {
            if s == "C" {
                self.advance();
            }
        }

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut externs = Vec::new();

        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let is_safe = self.match_token(&TokenType::Safe);
            self.consume(&TokenType::Fn, "Expected 'fn'")?;
            let name = self.expect_identifier("Expected function name")?;

            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let params = self.parse_comma_separated_params()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;

            self.consume(&TokenType::Arrow, "Expected '->'")?;
            let return_type = self.parse_type()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;

            externs.push(ExternDecl {
                name: name.into(),
                is_safe,
                params,
                return_type,
            });
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        Ok(externs)
    }

    pub(crate) fn parse_trait_decl(&mut self) -> ParseResult<'a, TraitDecl> {
        self.consume(&TokenType::Trait, "Expected 'trait'")?;
        let name = self.expect_identifier("Expected trait name")?;
        let generics = self.parse_generic_params()?;
        self.consume(&TokenType::LeftBrace, "Expected '{'")?;

        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.consume(&TokenType::Fn, "Expected 'fn' in trait")?;
            let method_name = self.expect_identifier("Expected method name")?;
            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let params = self.parse_comma_separated_params()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            self.consume(&TokenType::Arrow, "Expected '->'")?;
            let return_type = self.parse_type()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            methods.push(MethodSignature {
                name: method_name.into(),
                params,
                return_type,
            });
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(TraitDecl {
            name: name.into(),
            generics,
            methods,
        })
    }

    pub(crate) fn parse_impl_block(&mut self) -> ParseResult<'a, ImplBlock> {
        self.consume(&TokenType::Impl, "Expected 'impl'")?;

        let generics = self.parse_generic_params()?;

        // `impl<T> Transfer ...` reaches here (the dispatch only looks one token past `impl`, so
        // a generic list hides the trait name). Without this check, `parse_type` consumes
        // `Transfer` as an ordinary trait name and the error lands somewhere in the header,
        // while the actual offender is never named.
        //
        // A lowering is instantiated per EDGE, not per type, so a type parameter has nothing to
        // bind to -- the same reason a generic `fn` inside a lowering body is refused (E6015).
        let generic_transfer = self.check(&TokenType::Transfer)
            || matches!(&self.peek().kind, TokenType::Identifier(s) if &**s == "Transfer");
        if !generics.is_empty() && generic_transfer {
            return Err(self.error(
                "'impl Transfer' does not take generic parameters; a transfer lowering is \
                 declared per edge, for one machine \
                 (`impl Transfer<Memory::A, Memory::B> for Topology::X { ... }`)",
            ));
        }

        // Either `impl Trait for Type` or `impl Type`
        let mut trait_name = None;
        let target_type;

        // Since we don't have lookahead to distinguish `impl Trait for Type` from `impl Type`,
        // if we see `Identifier` followed by `for`, it's a trait. Otherwise it's a type.
        // Note: parse_type handles `Struct(name)`, which is an identifier!
        // We can just peek ahead.
        let parsed_type = self.parse_type()?;
        if self.check(&TokenType::For) {
            self.advance(); // consume 'for'
            if let Type::Struct(name, _) = parsed_type {
                trait_name = Some(name);
            } else if let Type::GenericInstance(inner, _) = parsed_type {
                if let Type::Struct(name, _) = *inner {
                    trait_name = Some(name);
                } else {
                    return Err(self.error("Expected trait name before 'for'"));
                }
            } else {
                return Err(self.error("Expected trait name before 'for'"));
            }
            target_type = self.parse_type()?;
        } else {
            target_type = parsed_type;
        }

        self.consume(&TokenType::LeftBrace, "Expected '{' after impl target")?;
        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let mut doc_comment: Option<String> = None;
            while let TokenType::DocComment(c) = &self.peek().kind {
                let text = c.to_string();
                if let Some(existing) = &mut doc_comment {
                    existing.push('\n');
                    existing.push_str(&text);
                } else {
                    doc_comment = Some(text);
                }
                self.advance();
            }

            let mut method = self.parse_function()?;
            method.doc_comment = doc_comment;
            methods.push(method);
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(ImplBlock {
            generics,
            trait_name,
            target_type,
            methods,
        })
    }

    pub(crate) fn parse_import_decl(&mut self) -> ParseResult<'a, ImportDecl> {
        self.consume(&TokenType::Import, "Expected 'import'")?;
        let mut path = Vec::new();
        loop {
            let ident = self.expect_identifier("Expected identifier in import path")?;
            path.push(ident.into());
            if self.match_token(&TokenType::DoubleColon) {
                continue;
            } else {
                break;
            }
        }
        self.consume(&TokenType::Semicolon, "Expected ';' after import path")?;
        Ok(ImportDecl { path })
    }

    pub(crate) fn parse_macro_def(&mut self) -> ParseResult<'a, MacroDefDecl> {
        let span_start_line = self.peek().line;
        let span_start_column = self.peek().column;
        self.consume(&TokenType::MacroRules, "Expected 'macro_rules!'")?;

        let name = self.expect_identifier("Expected macro name")?;

        self.consume(&TokenType::LeftBrace, "Expected '{' for macro rules body")?;

        let mut rules = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            // Matcher: ( ... )
            self.consume(&TokenType::LeftParen, "Expected '(' for macro matcher")?;
            let mut matcher = Vec::new();
            while !self.check(&TokenType::RightParen) && !self.check(&TokenType::Eof) {
                matcher.push(self.parse_token_tree()?);
            }
            self.consume(&TokenType::RightParen, "Expected ')' for macro matcher")?;

            self.consume(&TokenType::FatArrow, "Expected '=>' after macro matcher")?;

            // Transcriber: { ... }
            self.consume(&TokenType::LeftBrace, "Expected '{' for macro transcriber")?;
            let mut transcriber = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                transcriber.push(self.parse_token_tree()?);
            }
            self.consume(&TokenType::RightBrace, "Expected '}' for macro transcriber")?;

            // Optional semicolon after a rule
            if self.check(&TokenType::Semicolon) {
                self.advance();
            }

            rules.push(MacroRule {
                matcher,
                transcriber,
            });
        }
        self.consume(
            &TokenType::RightBrace,
            "Expected '}' after macro rules body",
        )?;

        Ok(MacroDefDecl {
            name: name.into(),
            rules,
            span: Span {
                line: span_start_line,
                column: span_start_column,
                length: 0,
            },
        })
    }

    pub fn parse(&mut self) -> ParseResult<'a, Program> {
        let mut imports = Vec::new();
        let mut externs = Vec::new();
        let mut topologies = Vec::new();
        let mut memories = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut functions = Vec::new();
        let mut macros = Vec::new();
        let mut transfer_impls = Vec::new();
        while !self.check(&TokenType::Eof) {
            let mut doc_comment: Option<String> = None;
            while let TokenType::DocComment(c) = &self.peek().kind {
                let text = c.to_string();
                if let Some(existing) = &mut doc_comment {
                    existing.push('\n');
                    existing.push_str(&text);
                } else {
                    doc_comment = Some(text);
                }
                self.advance();
            }

            if self.check(&TokenType::Import) {
                imports.push(self.parse_import_decl()?);
            } else if self.check(&TokenType::MacroRules) {
                macros.push(self.parse_macro_def()?);
            } else if self.check(&TokenType::Extern) {
                externs.extend(self.parse_extern_block()?);
            } else if self.check(&TokenType::Trait) {
                traits.push(self.parse_trait_decl()?);
            } else if self.check(&TokenType::Impl) {
                // `impl Transfer<Memory::A, Memory::B> for Topology::X { ... }` is a transfer
                // lowering, not an ordinary trait impl. The trait name `Transfer` is reserved for
                // it, so one token of lookahead separates the two: any `impl Transfer...` is a
                // lowering, and `parse_transfer_impl` reports what is wrong with the rest of the
                // header. Reserving the name is what makes this a one-token decision -- before
                // Vx#353 the deciding difference between a lowering and a trait impl was the
                // capital letter in `Transfer` versus the `transfer` keyword.
                let is_transfer_impl = matches!(
                    &self.peek_n(1).kind,
                    TokenType::Identifier(s) if &**s == "Transfer"
                ) || matches!(self.peek_n(1).kind, TokenType::Transfer);
                if is_transfer_impl {
                    let mut t = self.parse_transfer_impl()?;
                    t.doc_comment = doc_comment;
                    transfer_impls.push(t);
                } else {
                    impls.push(self.parse_impl_block()?);
                }
            } else if self.check(&TokenType::Struct) {
                let mut s = self.parse_struct_decl()?;
                s.doc_comment = doc_comment;
                structs.push(s);
            } else if self.check(&TokenType::Enum) {
                let mut e = self.parse_enum_decl()?;
                e.doc_comment = doc_comment;
                enums.push(e);
            } else if self.check(&TokenType::Fn) {
                let mut f = self.parse_function()?;
                f.doc_comment = doc_comment;
                functions.push(f);
            } else if self.check(&TokenType::Topology) {
                // `Topology <Name> { memory: Memory::X, visible: [Memory::Y, ...] }` declares a
                // user-defined topology. Its descriptor is carried on the AST (`Program.topologies`),
                // like a memory space -- no global registry -- and seeded into the per-compilation
                // cost graph by sema.
                topologies.push(self.parse_topology_decl()?);
            } else if self.check(&TokenType::Memory) {
                // `Memory <Name> { within:, capacity:, bandwidth:, managed:, granule: }` declares
                // a first-class memory space. The full descriptor is stored on the AST (not a
                // global registry); sema indexes it via `GlobalAstEnv`.
                let mut m = self.parse_memory_decl()?;
                m.doc_comment = doc_comment;
                memories.push(m);
            } else {
                return Err(self.error(&format!(
                    "Unexpected token at top level: {:?}",
                    self.peek().kind
                )));
            }
        }
        Ok(Program {
            module_path: self.source.to_string().into(), // Default fallback, should be overridden by pipeline
            imports,
            macros,
            externs,
            structs,
            enums,
            traits,
            impls,
            functions,
            topologies,
            memories,
            transfer_impls,
        })
    }

    /// `impl Transfer<Memory::A, Memory::B> for Topology::X { fn ... }` — a transfer lowering
    /// (docs/custom_transfer_contract.md). The body is ordinary `fn` items, parsed by
    /// `parse_function` like an impl block's methods.
    ///
    /// The bodies are macro-expanded, structurally checked (E6015), and type-checked like impl
    /// methods -- a body that errors at top level errors identically inside a lowering. The
    /// `raw::` primitive obligations and the emitted code are the rest of the contract.
    ///
    /// The `for Topology::X` clause is required, because a lowering is code for one machine's
    /// edge. Two parts can declare the same edge and move it with different instructions
    /// (`cp.async` on Ampere, TMA on Hopper), and before Vx#353 only one of them could say so.
    pub(crate) fn parse_transfer_impl(&mut self) -> ParseResult<'a, TransferImplDecl> {
        self.consume(&TokenType::Impl, "Expected 'impl'")?;
        // The old spelling, `impl transfer Memory::A -> Memory::B`, used the lowercase keyword.
        // It has no place to name a topology, which is the whole reason for the change, so it
        // gets a message pointing at the new form rather than a generic parse error.
        if self.check(&TokenType::Transfer) {
            return Err(self.error(
                "`impl transfer Memory::A -> Memory::B` is now \
                 `impl Transfer<Memory::A, Memory::B> for Topology::X`; a lowering is code for \
                 one machine's edge, so it has to name the machine",
            ));
        }
        let trait_name = self.expect_identifier("Expected 'Transfer' after 'impl'")?;
        if trait_name != "Transfer" {
            return Err(self.error(&format!(
                "Expected 'Transfer' after 'impl', got '{}'",
                trait_name
            )));
        }
        // `impl Transfer for SomeType` was the implicit-movement opt-in until Vx#353. It is
        // spelled `Relocatable` now, and it is a different question: whether a value may move
        // implicitly, not how bytes cross an edge.
        if self.check(&TokenType::For) {
            return Err(self.error(
                "`Transfer` names an edge lowering and takes two memory spaces \
                 (`impl Transfer<Memory::A, Memory::B> for Topology::X`). For a type that may \
                 move implicitly across a topology boundary, the trait is `Relocatable`",
            ));
        }
        self.consume(
            &TokenType::LeftAngle,
            "Expected '<' after 'Transfer' (`impl Transfer<Memory::A, Memory::B> for \
             Topology::X`)",
        )?;
        let from = self.parse_memory_space()?;
        self.consume(&TokenType::Comma, "Expected ',' in 'impl Transfer<A, B>'")?;
        let to = self.parse_memory_space()?;
        self.consume(
            &TokenType::RightAngle,
            "Expected '>' after 'impl Transfer<A, B>'",
        )?;
        self.consume(
            &TokenType::For,
            "Expected 'for Topology::X' after 'impl Transfer<A, B>': a lowering is code for one \
             machine's edge, so it has to name the machine",
        )?;
        let topology = self.parse_topology()?;
        self.consume(
            &TokenType::LeftBrace,
            "Expected '{' after 'impl Transfer' header",
        )?;
        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let mut doc_comment: Option<String> = None;
            while let TokenType::DocComment(c) = &self.peek().kind {
                let text = c.to_string();
                if let Some(existing) = &mut doc_comment {
                    existing.push('\n');
                    existing.push_str(&text);
                } else {
                    doc_comment = Some(text);
                }
                self.advance();
            }
            if !self.check(&TokenType::Fn) {
                return Err(self.error(&format!(
                    "Expected 'fn' inside 'impl Transfer' (a lowering is functions only), got {:?}",
                    self.peek().kind
                )));
            }
            let mut method = self.parse_function()?;
            method.doc_comment = doc_comment;
            methods.push(method);
        }
        self.consume(
            &TokenType::RightBrace,
            "Expected '}' to close 'impl Transfer'",
        )?;
        Ok(TransferImplDecl {
            from,
            to,
            topology,
            methods,
            doc_comment: None,
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use rstest::rstest;

    #[rstest]
    #[case("fn main() -> Tensor {}")]
    fn test_parse_empty_function(#[case] input: &str) {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.functions[0].name.as_ref(), "main");
    }

    #[test]
    fn transfer_impl_parses_and_is_carried() {
        // The wrapper from docs/custom_transfer_contract.md:
        // `impl Transfer<Memory::A, Memory::B> for Topology::X { fn ... }`. The body is ordinary
        // Vx -- that is the design's whole point -- so an ordinary function must parse inside it
        // unchanged.
        let input = r#"
/// Fills a distinct destination, so this is the copy shape.
impl Transfer<Memory::L2, Memory::SMEM> for Topology::Dev {
    /// One cooperative copy loop.
    fn move_tile(n: i32) -> i32 {
        let mut i = 0;
        loop {
            if i >= n { break; }
            i = i + 1;
        }
        return i;
    }
}
"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.transfer_impls.len(), 1);
        let t = &program.transfer_impls[0];
        assert_eq!(t.from, crate::syntax::MemorySpace::from_name("L2"));
        assert_eq!(t.to, crate::syntax::MemorySpace::from_name("SMEM"));
        // The machine the lowering is for, which is the half the header used to have no
        // room for. Without it the edge pair was the whole key, across the compilation.
        assert_eq!(t.topology.display_name(), "Dev");
        assert_eq!(t.methods.len(), 1);
        assert_eq!(t.methods[0].name.as_ref(), "move_tile");
        // Doc comments attach at both levels, mirroring impl blocks.
        assert!(t.doc_comment.as_deref().unwrap().contains("copy shape"));
        assert!(t.methods[0]
            .doc_comment
            .as_deref()
            .unwrap()
            .contains("cooperative copy"));
        // And it did NOT leak into the trait-impl list.
        assert!(program.impls.is_empty());
    }

    #[test]
    fn transfer_impl_does_not_shadow_trait_impls() {
        // The dispatch is one token of lookahead on a keyword; a plain impl block right next to a
        // transfer impl must still land in `impls`. This is the regression the lookahead could
        // cause and must not.
        let input = r#"
struct Pair { a: i32 }
impl Pair {
    fn get(x: i32) -> i32 { return x; }
}
impl Transfer<Memory::GPU_HBM, Memory::L2> for Topology::Dev {
    fn stage() -> i32 { return 0; }
}
"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.impls.len(), 1);
        assert_eq!(program.transfer_impls.len(), 1);
        assert_eq!(program.impls[0].methods[0].name.as_ref(), "get");
    }

    #[rstest]
    // One space, not an edge: the header names the movement, so it needs both ends.
    #[case("impl Transfer<Memory::L2> for Topology::Dev { fn f() -> i32 { return 0; } }")]
    // No `for` clause. A lowering is code for one machine's edge, so the machine is required
    // -- this is the case the whole change exists to make impossible to leave out.
    #[case("impl Transfer<Memory::A, Memory::B> { fn f() -> i32 { return 0; } }")]
    // A topology name, not a bare identifier, after `for`.
    #[case("impl Transfer<Memory::A, Memory::B> for Dev { fn f() -> i32 { return 0; } }")]
    // The header takes memory spaces, not bare identifiers.
    #[case("impl Transfer<L2, SMEM> for Topology::Dev { fn f() -> i32 { return 0; } }")]
    // A lowering is functions only; a stray declaration inside is an error, not skipped.
    #[case("impl Transfer<Memory::A, Memory::B> for Topology::Dev { let x = 1; }")]
    // Unclosed body reaches Eof rather than looping forever.
    #[case("impl Transfer<Memory::A, Memory::B> for Topology::Dev { fn f() -> i32 { return 0; }")]
    // The old spelling is refused rather than silently reinterpreted.
    #[case("impl transfer Memory::A -> Memory::B { fn f() -> i32 { return 0; } }")]
    fn transfer_impl_rejects_malformed_headers(#[case] input: &str) {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        assert!(parser.parse().is_err(), "should reject: {input}");
    }

    #[test]
    fn transfer_impl_bodies_are_macro_expanded() {
        // Macro expansion walks functions and impl methods; a lowering body must expand too, or
        // a MacroCall node survives for every downstream pass to trip on (Vx#352 review finding).
        let input = r#"
macro_rules! one { () => { 1 } }
impl Transfer<Memory::L2, Memory::SMEM> for Topology::Dev {
    fn move_tile() -> i32 { return one!(); }
}
"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let mut program = parser.parse().unwrap();
        // Mirror the driver: collect rules into the name-keyed map the expander takes.
        let mut rules = std::collections::HashMap::new();
        for mac in &program.macros {
            rules.insert(mac.name.clone(), mac.rules.clone());
        }
        let expander = crate::syntax::MacroExpander::new(&rules);
        expander.expand_module(&mut program).unwrap();
        let body = format!("{:?}", program.transfer_impls[0].methods[0].body);
        assert!(
            !body.contains("MacroCall"),
            "macro call survived expansion inside a transfer lowering: {body}"
        );
    }

    #[test]
    fn transfer_impl_signature_clone_keeps_bodies() {
        // The reverse of what this test pinned until #353 A4, and the reversal is the
        // point. `clone_signature` strips function bodies so the parallel pipeline can
        // share a light Program, and a lowering body used to go with them -- but a
        // lowering's body is now a fact the checker reads at every transfer site: it is
        // what the derived traffic count is counted FROM.
        //
        // Stripped, the count came back as a confident zero with nothing to indicate a
        // body had gone missing. A derived figure that silently reports nothing when its
        // input vanished is worse than no figure, so the weight argument loses: these are
        // copy loops, a few statements each.
        let input = r#"
impl Transfer<Memory::L2, Memory::SMEM> for Topology::Dev {
    fn move_tile(n: i32) -> i32 { return n; }
}
fn ordinary(n: i32) -> i32 { return n; }
"#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        let sig = program.clone_signature();
        assert_eq!(sig.transfer_impls.len(), 1);
        assert!(
            !sig.transfer_impls[0].methods[0].body.is_empty(),
            "a lowering body must survive the signature clone -- the traffic counter reads it"
        );
        // The exemption is scoped to lowerings: an ordinary function beside it is still
        // stripped, so the light-Program property the clone exists for is intact.
        let ordinary = sig
            .functions
            .iter()
            .find(|f| f.name.as_ref() == "ordinary")
            .expect("the ordinary function survives as a signature");
        assert!(
            ordinary.body.is_empty(),
            "ordinary function bodies are still stripped"
        );
    }

    #[test]
    fn test_parse_distributed_matmul() {
        let input = r#"
fn distributed_matmul(a: Ref<Tensor, Memory::CPU_DRAM>, b: Ref<Tensor, Memory::CPU_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_a = transfer(a, Memory::NPU_HBM);
        let local_b = transfer(b, Memory::NPU_HBM);
        let result = custom_matmul(local_a, local_b);
        return transfer(result, Memory::CPU_DRAM);
    }
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);

        let func = &program.functions[0];
        assert_eq!(func.name.as_ref(), "distributed_matmul");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].0.as_ref(), "a");

        // Assert return type is Verified<Tensor>
        assert_eq!(
            func.return_type,
            Type::Verified(Box::new(Type::Tensor(ElementType::F32, vec![], None)))
        );

        // Assert body has one statement (spawn on)
        assert_eq!(func.body.len(), 1);
        if let Statement::Return(ReturnStmt {
            expr:
                Expr::SpawnOn(SpawnOnExpr {
                    top,
                    stmts,
                    ret: _,
                    span: _,
                }),
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(
                *top,
                // A topology index is a compile-time dimension, stamped to its default type at
                // parse time (#240) rather than left untyped like a value-position literal.
                Topology::NPU(Box::new(Expr::Number(NumberExpr {
                    value: "0".to_string().into(),
                    ty: Some(ElementType::I32),
                    span: Span::default()
                })))
            );
            assert_eq!(stmts.len(), 4);
        } else {
            panic!("Expected SpawnOn statement, got {:#?}", &func.body[0]);
        }
    }

    #[test]
    fn test_parse_let_mut_with_type() {
        let input = "fn main() -> Tensor { let mut x: Tensor = Tensor([1, 2]); }";
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        let func = &program.functions[0];
        if let Statement::LetDecl(LetDeclStmt {
            name,
            is_mut,
            ty_ann: ty,
            expr,
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(name.as_ref(), "x");
            assert!(is_mut);
            assert_eq!(ty, &Some(Type::Tensor(ElementType::F32, vec![], None)));
            if let Expr::FunctionCall(FunctionCallExpr {
                name: func_name,
                args,
                span: _,
                type_args: _,
            }) = expr
            {
                assert_eq!(func_name.as_ref(), "Tensor");
                assert_eq!(args.len(), 1);
                if let Expr::Array(ArrayExpr { elements, span: _ }) = &args[0] {
                    assert_eq!(elements.len(), 2);
                } else {
                    panic!("Expected array");
                }
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected LetDecl");
        }
    }

    #[test]
    fn test_parse_for_loop() {
        let input = "fn main() -> Tensor { for i in 0..10 { x = 5; } }";
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        if let Statement::ForLoop(ForLoopStmt {
            iter,
            iterable,
            body,
            invariants: _,
            span: _,
        }) = &program.functions[0].body[0]
        {
            assert_eq!(iter, "i");
            if let Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) = &**iterable
            {
                assert_eq!(
                    **start,
                    Expr::Number(NumberExpr {
                        value: "0".to_string().into(),
                        ty: None,
                        span: Span::default()
                    })
                );
                assert_eq!(
                    **end,
                    Expr::Number(NumberExpr {
                        value: "10".to_string().into(),
                        ty: None,
                        span: Span::default()
                    })
                );
            } else {
                panic!("Expected Range expression for iterable");
            }
            assert_eq!(body.len(), 1);
            if let Statement::Assign(AssignStmt { lhs, rhs, span: _ }) = &body[0] {
                if let Expr::Identifier(id) = lhs {
                    assert_eq!(id.name.as_ref(), "x");
                } else {
                    panic!("Expected Identifier");
                }
                if let Expr::Number(num) = rhs {
                    assert_eq!(num.value.as_ref(), "5");
                    assert_eq!(num.ty, None);
                } else {
                    panic!("Expected Number");
                }
            } else {
                panic!("Expected Assign");
            }
        } else {
            panic!("Expected ForLoop");
        }
    }

    #[test]
    fn test_parse_compound_assign() {
        let input = "fn main() -> Tensor { x[0] += y * z; }";
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        if let Statement::CompoundAssign(CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span: _,
        }) = &program.functions[0].body[0]
        {
            assert_eq!(*op, BinaryOp::Add);
            if let Expr::IndexAccess(IndexAccessExpr {
                base: arr,
                index: idx,
                span: _,
            }) = lhs
            {
                if let Expr::Identifier(id) = &**arr {
                    assert_eq!(id.name.as_ref(), "x");
                } else {
                    panic!("Expected Identifier");
                }
                if let Expr::Number(num) = &**idx {
                    assert_eq!(num.value.as_ref(), "0");
                    // Unsuffixed literals parse untyped and are inferred at type-check (#240).
                    assert_eq!(num.ty, None);
                } else {
                    panic!("Expected Number");
                }
            } else {
                panic!("Expected IndexAccess");
            }

            if let Expr::BinaryOp(BinaryOpExpr {
                lhs: left,
                op: binop,
                rhs: right,
                span: _,
            }) = rhs
            {
                assert_eq!(*binop, BinaryOp::Mul);
                if let Expr::Identifier(id) = &**left {
                    assert_eq!(id.name.as_ref(), "y");
                } else {
                    panic!("Expected Identifier");
                }
                if let Expr::Identifier(id) = &**right {
                    assert_eq!(id.name.as_ref(), "z");
                } else {
                    panic!("Expected Identifier");
                }
            } else {
                panic!("Expected BinaryOp");
            }
        } else {
            panic!("Expected CompoundAssign");
        }
    }

    #[test]
    fn test_parse_member_and_method() {
        let input = "fn main() -> Tensor { x.shape.with_memory(Memory::NPU_HBM); }";
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        if let Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        }) = &program.functions[0].body[0]
        {
            if let Expr::MethodCall(MethodCallExpr {
                base: obj,
                method_name: method,
                args,
                span: _,
                type_args: _,
            }) = expr
            {
                assert_eq!(method.as_ref(), "with_memory");
                assert_eq!(args.len(), 1);
                if let Expr::MemberAccess(MemberAccessExpr {
                    base: inner_obj,
                    member,
                    struct_name: _,
                    span: _,
                }) = &**obj
                {
                    assert_eq!(member.as_ref(), "shape");
                    if let Expr::Identifier(id) = &**inner_obj {
                        assert_eq!(id.name.as_ref(), "x");
                    } else {
                        panic!("Expected Identifier");
                    }
                } else {
                    panic!("Expected MemberAccess");
                }
            } else {
                panic!("Expected MethodCall");
            }
        } else {
            panic!("Expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_full_custom_matmul() {
        let input = r#"
        fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>, b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
            spawn on(Topology::NPU[0]) {
                let mut result: Tensor = Tensor([a.shape[0], b.shape[1]]).with_memory(Memory::NPU_HBM);
                for i in 0..a.shape[0] {
                    for j in 0..b.shape[1] {
                        result[i][j] = 0;
                        for k in 0..a.shape[1] {
                            result[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                return Verified(result);
            }
        }
        "#;
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);
        let func = &program.functions[0];
        assert_eq!(func.name.as_ref(), "custom_matmul");
        if let Statement::Return(ReturnStmt {
            expr:
                Expr::SpawnOn(SpawnOnExpr {
                    top: _,
                    stmts,
                    ret: _,
                    span: _,
                }),
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(stmts.len(), 3); // Let, For, Return
        } else {
            panic!("Expected SpawnOn, got {:#?}", &func.body[0]);
        }
    }

    #[test]
    fn test_parse_struct_and_pointers() {
        let input = r#"
        struct Config {
            value: Tensor<i32>,
            threshold: Tensor<f32>
        }

        fn update_config(c: &mut Config) -> Tensor<Bool> {
            unsafe {
                let ptr: *mut Config = &mut c;
                *ptr = Config { value: 10, threshold: 0.5 };
            }
            return c.value < 20;
        }
        "#;
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();

        assert_eq!(program.structs.len(), 1);
        assert_eq!(program.structs[0].name.as_ref(), "Config");
        assert_eq!(program.structs[0].fields.len(), 2);
        assert_eq!(program.structs[0].fields[0].0.as_ref(), "value");

        assert_eq!(program.functions.len(), 1);
        let func = &program.functions[0];
        assert_eq!(func.name.as_ref(), "update_config");

        // Param should be &mut Config
        let param_ty = &func.params[0].1;
        if let Type::Borrow {
            inner,
            mem_space: None,
            is_mut: true,
            ..
        } = param_ty
        {
            if let Type::Struct(s, _) = &**inner {
                assert_eq!(s.as_ref(), "Config");
            } else {
                panic!("Expected Struct");
            }
        } else {
            panic!("Expected Borrow");
        }

        // Body should have unsafe block
        if let Statement::ExprStmt(ExprStmtStmt {
            expr:
                Expr::UnsafeBlock(UnsafeBlockExpr {
                    stmts,
                    ret: None,
                    span: _,
                }),
            has_semi: _,
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(stmts.len(), 2);
        } else {
            panic!("Expected UnsafeBlock");
        }
    }

    #[test]
    fn test_parse_extern() {
        let input = r#"
        extern "C" {
            fn malloc(size: Tensor<i32>) -> *mut Tensor<f32>;
        }
        "#;
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();

        assert_eq!(program.externs.len(), 1);
        assert_eq!(program.externs[0].name.as_ref(), "malloc");
        assert_eq!(program.externs[0].params.len(), 1);
        if let Type::Pointer(inner, None, true) = &program.externs[0].return_type {
            assert_eq!(**inner, Type::Tensor(ElementType::F32, vec![], None));
        } else {
            panic!("Expected pointer return type");
        }
    }

    #[test]
    fn test_parse_implicit_return() {
        let input = r#"
fn stdout_write(buffer: *const u8, len: i64) -> i64 {
    vx_stdout_write(buffer, len)
}

fn stderr_write(buffer: *const u8, len: i64) -> i64 {
    vx_stderr_write(buffer, len);
}
        "#;
        let tokens = Lexer::new(input).tokenize();
        let mut parser = Parser::new(&tokens, input);
        let program = parser.parse().unwrap();

        assert_eq!(program.functions.len(), 2);

        // stdout_write should have a Return statement (implicit return converted)
        let func1 = &program.functions[0];
        if let Statement::Return(ReturnStmt { expr, span: _ }) = &func1.body[0] {
            if let Expr::FunctionCall(FunctionCallExpr {
                name,
                args,
                span: _,
                type_args: _,
            }) = expr
            {
                assert_eq!(name.as_ref(), "vx_stdout_write");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected FunctionCall in implicit return");
            }
        } else {
            panic!("Expected Return statement from implicit return");
        }

        // stderr_write should have an ExprStmt with has_semicolon = true
        let func2 = &program.functions[1];
        if let Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi: has_semicolon,
            span: _,
        }) = &func2.body[0]
        {
            assert!(*has_semicolon);
            if let Expr::FunctionCall(FunctionCallExpr {
                name,
                args,
                span: _,
                type_args: _,
            }) = expr
            {
                assert_eq!(name.as_ref(), "vx_stderr_write");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected FunctionCall in ExprStmt");
            }
        } else {
            panic!("Expected ExprStmt with semicolon");
        }
    }
}
