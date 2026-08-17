//===- types.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx type signatures, including generic arguments, memory spaces, and topologies.
//
//===----------------------------------------------------------------------===//

use super::*;

impl<'a> Parser<'a> {
    pub(crate) fn parse_topology(&mut self) -> ParseResult<'a, Topology> {
        self.consume(&TokenType::Topology, "Expected 'Topology'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Topology'")?;
        let ident = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            _ => return Err(self.error("Expected hardware identifier after Topology::")),
        };
        match ident.as_ref() {
            "CPU" => Ok(Topology::CPU),
            "Current" => Ok(Topology::Current),
            "NPU" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let mut expr = self.parse_expr()?;
                    super::expr::stamp_dim_literals(&mut expr);
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    // If the expression is a range like 0..4, convert to Slice
                    if let Expr::Range(RangeExpr { start, end, .. }) = expr {
                        Ok(Topology::Slice(
                            Box::new(Topology::NPU(start.clone())),
                            start,
                            end,
                        ))
                    } else {
                        Ok(Topology::NPU(Box::new(expr)))
                    }
                } else {
                    Err(self.error("Expected index for NPU"))
                }
            }
            "AccCore" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let mut expr = self.parse_expr()?;
                    super::expr::stamp_dim_literals(&mut expr);
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    Ok(Topology::AccCore(Box::new(expr)))
                } else {
                    Err(self.error("Expected index for AccCore"))
                }
            }
            "AMX" => Ok(Topology::AMX),
            "ANE" => Ok(Topology::ANE),
            "GPU" => {
                // The index is optional, unlike NPU's: bare `Topology::GPU` predates
                // multi-device support and means device 0, which is what every program
                // written against it meant.
                if self.match_token(&TokenType::LeftBracket) {
                    let mut expr = self.parse_expr()?;
                    super::expr::stamp_dim_literals(&mut expr);
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    if let Expr::Range(RangeExpr { start, end, .. }) = expr {
                        Ok(Topology::Slice(
                            Box::new(Topology::GPU(start.clone())),
                            start,
                            end,
                        ))
                    } else {
                        Ok(Topology::GPU(Box::new(expr)))
                    }
                } else {
                    Ok(Topology::gpu(0))
                }
            }
            "CpuAvx512" | "CPU_AVX512" => Ok(Topology::CpuAvx512),
            "CpuNeon" | "CPU_Neon" => Ok(Topology::CpuNeon),
            // Any other identifier is a user-defined topology, kept as a bare name here; its
            // descriptor is resolved later from the AST-carried declarations
            // (`Program.topologies`, indexed by `GlobalAstEnv`). Because the topology set is open,
            // an unknown built-in typo is indistinguishable from a custom name here.
            other => Ok(Topology::Custom(crate::symbol::Symbol::from(other))),
        }
    }

    /// Parse a topology in an operand position (a `Reachable<A, B>` argument): either a
    /// `Topology::X` reference or a bare identifier naming a topology variable.
    pub(crate) fn parse_topology_operand(&mut self) -> ParseResult<'a, Topology> {
        if self.check(&TokenType::Topology) {
            self.parse_topology()
        } else {
            let n = self.expect_identifier("Expected a topology name in Reachable<A, B>")?;
            Ok(Topology::Custom(n.into()))
        }
    }

    pub(crate) fn parse_memory_space(&mut self) -> ParseResult<'a, MemorySpace> {
        self.consume(&TokenType::Memory, "Expected 'Memory'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Memory'")?;
        let ident = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            _ => return Err(self.error("Expected memory identifier after Memory::")),
        };
        // Built-in names map to their variants; any other identifier is a user-defined space
        // (the set is open, like topologies). Single source of truth: `MemorySpace::from_name`.
        Ok(MemorySpace::from_name(&ident))
    }

    pub(crate) fn parse_type(&mut self) -> ParseResult<'a, Type> {
        if self.match_token(&TokenType::Ampersand) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner = self.parse_type()?;
            Ok(Type::Borrow {
                inner: Box::new(inner),
                mem_space: None,
                is_mut,
                // Not yet bound to a scope depth — the borrow checker assigns the real region during
                // checking. Reserved sentinel, never a real depth (#267).
                region_id: crate::borrow::REGION_UNSET as usize,
            })
        } else if self.match_token(&TokenType::Star) {
            let is_mut = if self.check(&TokenType::Mut) {
                self.advance();
                true
            } else if self.check(&TokenType::Identifier("const")) {
                self.advance();
                false
            } else {
                return Err(self.error("Expected 'mut' or 'const' after '*'"));
            };
            let inner = self.parse_type()?;
            Ok(Type::Pointer(Box::new(inner), None, is_mut))
        } else if self.match_token(&TokenType::Ref) {
            self.consume(&TokenType::LeftAngle, "Expected '<'")?;
            let inner = self.parse_type()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let mem = self.parse_memory_space()?;
            self.consume(&TokenType::RightAngle, "Expected '>'")?;
            Ok(Type::Ref(Box::new(inner), mem))
        } else if self.match_token(&TokenType::Verified) {
            self.consume(&TokenType::LeftAngle, "Expected '<'")?;
            let inner = self.parse_type()?;
            self.consume(&TokenType::RightAngle, "Expected '>'")?;
            Ok(Type::Verified(Box::new(inner)))
        } else if self.match_token(&TokenType::Pinned) {
            self.consume(&TokenType::LeftAngle, "Expected '<'")?;
            let inner = self.parse_type()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let top = self.parse_topology()?;
            self.consume(&TokenType::RightAngle, "Expected '>'")?;
            Ok(Type::Pinned(Box::new(inner), top))
        } else if self.check(&TokenType::Topology) && self.peek_n(1).kind != TokenType::DoubleColon
        {
            // A bare `Topology` in type position (e.g. `Vec<Topology>`, `let t: Topology`)
            // is the runtime topology-value type: an i32 discriminant, i.e. the stable
            // `arch::topology_dispatch_id`. Placement annotations like
            // `Pinned<T, Topology::GPU>` are parsed via `parse_topology` (above), so a
            // bare `Topology` keyword here is always the value type.
            self.advance();
            Ok(Type::Scalar(ElementType::I32))
        } else if self.match_token(&TokenType::LeftAngle) {
            let n = match &self.advance().kind {
                TokenType::Number(s) => s
                    .parse::<usize>()
                    .map_err(|_| self.error("Expected integer for SIMD size"))?,
                _ => return Err(self.error("Expected number after '<' in SIMD type")),
            };
            match &self.advance().kind {
                TokenType::Identifier(s) if *s == "x" => {}
                _ => return Err(self.error("Expected 'x' after size in SIMD type")),
            }
            let el_ty_ident = match &self.advance().kind {
                TokenType::Identifier(s) => *s,
                _ => return Err(self.error("Expected element type after 'x' in SIMD type")),
            };
            let el_ty = std::str::FromStr::from_str(el_ty_ident)
                .map_err(|_| self.error(&format!("Unknown SIMD element type {}", el_ty_ident)))?;
            self.consume(
                &TokenType::RightAngle,
                "Expected '>' after SIMD element type",
            )?;
            Ok(Type::Simd(el_ty, n))
        } else if self.match_token(&TokenType::Fn) {
            self.consume(&TokenType::LeftParen, "Expected '(' after 'fn'")?;
            let mut params = Vec::new();
            while !self.check(&TokenType::RightParen) && !self.check(&TokenType::Eof) {
                params.push(self.parse_type()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            self.consume(
                &TokenType::RightParen,
                "Expected ')' after function parameters",
            )?;
            self.consume(&TokenType::Arrow, "Expected '->' after function parameters")?;
            let ret = self.parse_type()?;
            Ok(Type::Function(params, Box::new(ret)))
        } else if self.match_token(&TokenType::OrOr) {
            self.consume(&TokenType::Arrow, "Expected '->' after closure parameters")?;
            let ret = self.parse_type()?;
            Ok(Type::Closure(Vec::new(), Box::new(ret)))
        } else if self.match_token(&TokenType::Pipe) {
            let mut params = Vec::new();
            if !self.check(&TokenType::Pipe) {
                while !self.check(&TokenType::Pipe) && !self.check(&TokenType::Eof) {
                    params.push(self.parse_type()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(
                &TokenType::Pipe,
                "Expected '|' after closure type parameters",
            )?;
            self.consume(&TokenType::Arrow, "Expected '->' after closure parameters")?;
            let ret = self.parse_type()?;
            Ok(Type::Closure(params, Box::new(ret)))
        } else {
            if let TokenType::Identifier(s) = &self.peek().kind {
                if self.generic_params.iter().any(|p| p == *s) {
                    let s = s.to_string();
                    self.advance();
                    return Ok(Type::Generic(s.into(), None));
                }
            }

            self.parse_named_type()
        }
    }

    pub(crate) fn parse_generic_type_args(&mut self) -> ParseResult<'a, Vec<Type>> {
        let mut type_args = Vec::new();
        while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
            let is_expr = match &self.peek().kind {
                TokenType::Number(_) | TokenType::StringLiteral(_) => true,
                TokenType::Identifier(s) if *s == "true" || *s == "false" => true,
                _ => false,
            };

            if is_expr {
                let expr = self.parse_primary_expr()?;
                type_args.push(Type::Const(Box::new(expr)));
            } else {
                let saved_pos = self.pos;
                if let Ok(ty) = self.parse_type() {
                    type_args.push(ty);
                } else {
                    self.pos = saved_pos;
                    if let Ok(expr) = self.parse_primary_expr() {
                        type_args.push(Type::Const(Box::new(expr)));
                    } else {
                        return Err(
                            self.error("Expected type or constant expression in generic arguments")
                        );
                    }
                }
            }

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(
            &TokenType::RightAngle,
            "Expected '>' after generic type arguments",
        )?;
        Ok(type_args)
    }

    pub(crate) fn parse_named_type(&mut self) -> ParseResult<'a, Type> {
        let ident = match &self.advance().kind {
            TokenType::Identifier(s) => s.to_string(),
            _ => return Err(self.error("Expected type identifier")),
        };

        // Qualified nominal path: `Mod::Sub::Name`. The leading segments name the *defining* module
        // and the leaf is the type; name resolution (`resolve_names`, #194) splits it back and
        // attaches the defining module's GID. Builtins (`Tensor`, scalars) are never qualified, so
        // only take this branch when a `::` actually follows the first segment. The full `::`-joined
        // path is kept as the nominal's name Symbol so the AST shape is unchanged.
        if self.check(&TokenType::DoubleColon) {
            let mut segments = vec![ident];
            while self.match_token(&TokenType::DoubleColon) {
                segments
                    .push(self.expect_identifier("Expected identifier after '::' in type path")?);
            }
            let base_type = Type::Struct(
                crate::symbol::Symbol::from(segments.join("::").as_ref()),
                None,
            );
            if self.match_token(&TokenType::LeftAngle) {
                let type_args = self.parse_generic_type_args()?;
                return Ok(Type::GenericInstance(Box::new(base_type), type_args));
            }
            return Ok(base_type);
        }

        match ident.as_ref() {
            "Tensor" => {
                let mut el_ty = ElementType::F32;
                if let TokenType::LeftAngle = &self.peek().kind {
                    self.advance(); // consume '<'
                    let ty_ident = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected element type after '<'")),
                    };
                    el_ty = if let Ok(parsed_ty) = std::str::FromStr::from_str(ty_ident) {
                        parsed_ty
                    } else if self.generic_params.iter().any(|p| p.as_str() == ty_ident) {
                        ElementType::Generic(ty_ident.into())
                    } else {
                        return Err(self.error(&format!("Unknown element type {}", ty_ident)));
                    };
                    let mut dims = Vec::new();
                    if self.match_token(&TokenType::Comma) {
                        if self.match_token(&TokenType::LeftBracket) {
                            while !self.check(&TokenType::RightBracket)
                                && !self.check(&TokenType::Eof)
                            {
                                let mut dim = self.parse_expr()?;
                                super::expr::stamp_dim_literals(&mut dim);
                                dims.push(dim);
                                if !self.match_token(&TokenType::Comma) {
                                    break;
                                }
                            }
                            self.consume(
                                &TokenType::RightBracket,
                                "Expected ']' after Tensor dimensions",
                            )?;
                        } else {
                            return Err(self.error("Expected '[' for Tensor dimensions"));
                        }
                    }

                    let mut top = None;
                    if self.match_token(&TokenType::Comma) && self.check(&TokenType::Topology) {
                        top = Some(self.parse_topology()?);
                    }

                    self.consume(
                        &TokenType::RightAngle,
                        "Expected '>' after Tensor parameters",
                    )?;
                    return Ok(Type::Tensor(el_ty, dims, top));
                }
                Ok(Type::Tensor(el_ty, Vec::new(), None))
            }
            "Matrix" => Ok(Type::Matrix),
            _ => {
                if let Ok(el_ty) = std::str::FromStr::from_str(&ident) {
                    return Ok(Type::Scalar(el_ty));
                }

                // Check for GenericInstance like Config<f32>
                let base_type = Type::Struct(crate::symbol::Symbol::from(ident.as_ref()), None);
                if self.match_token(&TokenType::LeftAngle) {
                    let type_args = self.parse_generic_type_args()?;
                    Ok(Type::GenericInstance(Box::new(base_type), type_args))
                } else {
                    Ok(base_type)
                }
            }
        }
    }
}
