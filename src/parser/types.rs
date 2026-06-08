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
    pub(crate) fn parse_topology(&mut self) -> Result<Topology, String> {
        self.consume(&TokenType::Topology, "Expected 'Topology'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Topology'")?;
        let ident = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected hardware identifier after Topology::")),
        };
        match ident.as_str() {
            "CPU" => Ok(Topology::CPU),
            "Current" => Ok(Topology::Current),
            "NPU" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let expr = self.parse_expr()?;
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    Ok(Topology::NPU(Box::new(expr)))
                } else {
                    Err(self.error("Expected index for NPU"))
                }
            }
            "AccCore" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let expr = self.parse_expr()?;
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    Ok(Topology::AccCore(Box::new(expr)))
                } else {
                    Err(self.error("Expected index for AccCore"))
                }
            }
            "AMX" => Ok(Topology::AMX),
            "ANE" => Ok(Topology::ANE),
            "GPU" => Ok(Topology::GPU),
            "CPU_AVX512" => Ok(Topology::CPU_AVX512),
            "CPU_Neon" => Ok(Topology::CPU_Neon),
            _ => Err(format!("Unknown topology {}", ident)),
        }
    }

    pub(crate) fn parse_memory_space(&mut self) -> Result<MemorySpace, String> {
        self.consume(&TokenType::Memory, "Expected 'Memory'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Memory'")?;
        let ident = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected memory identifier after Memory::")),
        };
        match ident.as_str() {
            "CPU_DRAM" => Ok(MemorySpace::CPUDRAM),
            "NPU_HBM" => Ok(MemorySpace::NPUHBM),
            "Local_SRAM" => Ok(MemorySpace::LocalSRAM),
            "NIC_RAM" => Ok(MemorySpace::NIC_RAM),
            "Remote_HBM" => Ok(MemorySpace::Remote_HBM),
            _ => Err(format!("Unknown memory space {}", ident)),
        }
    }

    pub(crate) fn parse_type(&mut self) -> Result<Type, String> {
        if self.match_token(&TokenType::Ampersand) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner = self.parse_type()?;
            Ok(Type::Borrow(Box::new(inner), None, is_mut, 4095))
        } else if self.match_token(&TokenType::Star) {
            let is_mut = if self.check(&TokenType::Mut) {
                self.advance();
                true
            } else if self.check(&TokenType::Identifier("const".to_string())) {
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
        } else if self.match_token(&TokenType::LeftAngle) {
            let n_token = self.advance().clone();
            let n = match n_token.kind {
                TokenType::Number(s) => s
                    .parse::<usize>()
                    .map_err(|_| "Expected integer for SIMD size".to_string())?,
                _ => return Err(self.error("Expected number after '<' in SIMD type")),
            };
            let x_token = self.advance().clone();
            match x_token.kind {
                TokenType::Identifier(ref s) if s == "x" => {}
                _ => return Err(self.error("Expected 'x' after size in SIMD type")),
            }
            let el_ty_ident = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err(self.error("Expected element type after 'x' in SIMD type")),
            };
            let el_ty = std::str::FromStr::from_str(el_ty_ident.as_str())
                .map_err(|_| format!("Unknown SIMD element type {}", el_ty_ident))?;
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
            let token = self.peek().clone();
            if let TokenType::Identifier(ref s) = token.kind {
                if self.generic_params.contains(s) {
                    self.advance();
                    return Ok(Type::Generic(s.clone(), None));
                }
            }

            self.parse_named_type()
        }
    }

    pub(crate) fn parse_generic_type_args(&mut self) -> Result<Vec<Type>, String> {
        let mut type_args = Vec::new();
        while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
            let saved_pos = self.pos;
            if let Ok(ty) = self.parse_type() {
                type_args.push(ty);
            } else {
                self.pos = saved_pos;
                if let Ok(expr) = self.parse_primary_expr() {
                    type_args.push(Type::Const(Box::new(expr)));
                } else {
                    return Err(
                        "Expected type or constant expression in generic arguments".to_string()
                    );
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

    pub(crate) fn parse_named_type(&mut self) -> Result<Type, String> {
        let ident = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected type identifier")),
        };
        match ident.as_str() {
            "Tensor" => {
                let mut el_ty = ElementType::F32;
                if let TokenType::LeftAngle = &self.peek().kind {
                    self.advance(); // consume '<'
                    let ty_ident = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected element type after '<'")),
                    };
                    el_ty = if let Ok(parsed_ty) = std::str::FromStr::from_str(ty_ident.as_str()) {
                        parsed_ty
                    } else if self.generic_params.contains(&ty_ident) {
                        ElementType::Generic(ty_ident)
                    } else {
                        return Err(format!("Unknown element type {}", ty_ident));
                    };
                    let mut dims = Vec::new();
                    if self.match_token(&TokenType::Comma) {
                        if self.match_token(&TokenType::LeftBracket) {
                            while !self.check(&TokenType::RightBracket)
                                && !self.check(&TokenType::Eof)
                            {
                                dims.push(self.parse_expr()?);
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
                    if self.match_token(&TokenType::Comma) {
                        if self.check(&TokenType::Topology) {
                            top = Some(self.parse_topology()?);
                        }
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
                if let Ok(el_ty) = std::str::FromStr::from_str(ident.as_str()) {
                    return Ok(Type::Scalar(el_ty));
                }

                // Check for GenericInstance like Config<f32>
                let base_type = Type::Struct(ident, None);
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
