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
            _ => return Err("Expected hardware identifier after Topology::".to_string()),
        };
        match ident.as_str() {
            "Host" => Ok(Topology::Host),
            "Current" => Ok(Topology::Current),
            "NPU" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let expr = self.parse_expr()?;
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    Ok(Topology::NPU(Box::new(expr)))
                } else {
                    Err("Expected index for NPU".to_string())
                }
            }
            "AccCore" => {
                if self.match_token(&TokenType::LeftBracket) {
                    let expr = self.parse_expr()?;
                    self.consume(&TokenType::RightBracket, "Expected ']'")?;
                    Ok(Topology::AccCore(Box::new(expr)))
                } else {
                    Err("Expected index for AccCore".to_string())
                }
            }
            "AMX" => Ok(Topology::AMX),
            "ANE" => Ok(Topology::ANE),
            "GPU" => Ok(Topology::GPU),
            "Host_AVX512" => Ok(Topology::Host_AVX512),
            "Host_Neon" => Ok(Topology::Host_Neon),
            _ => Err(format!("Unknown topology {}", ident)),
        }
    }

    pub(crate) fn parse_memory_space(&mut self) -> Result<MemorySpace, String> {
        self.consume(&TokenType::Memory, "Expected 'Memory'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Memory'")?;
        let ident = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected memory identifier after Memory::".to_string()),
        };
        match ident.as_str() {
            "Host_DRAM" => Ok(MemorySpace::HostDRAM),
            "NPU_HBM" => Ok(MemorySpace::NPUHBM),
            "Local_SRAM" => Ok(MemorySpace::LocalSRAM),
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
                return Err("Expected 'mut' or 'const' after '*'".to_string());
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
                _ => return Err("Expected number after '<' in SIMD type".to_string()),
            };
            let x_token = self.advance().clone();
            match x_token.kind {
                TokenType::Identifier(ref s) if s == "x" => {}
                _ => return Err("Expected 'x' after size in SIMD type".to_string()),
            }
            let el_ty_ident = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected element type after 'x' in SIMD type".to_string()),
            };
            let el_ty = match el_ty_ident.as_str() {
                "f32" => ElementType::F32,
                "f64" => ElementType::F64,
                "f16" => ElementType::F16,
                "bf16" => ElementType::BF16,
                "i8" => ElementType::I8,
                "i16" => ElementType::I16,
                "i32" => ElementType::I32,
                "i64" => ElementType::I64,
                "u8" => ElementType::U8,
                "u16" => ElementType::U16,
                "u32" => ElementType::U32,
                "u64" => ElementType::U64,
                _ => return Err(format!("Unknown SIMD element type {}", el_ty_ident)),
            };
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

            let ident = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected type identifier".to_string()),
            };
            match ident.as_str() {
                "Tensor" => {
                    let mut el_ty = ElementType::F32;
                    if let TokenType::LeftAngle = &self.peek().kind {
                        self.advance(); // consume '<'
                        let ty_ident = match self.advance().kind.clone() {
                            TokenType::Identifier(s) => s,
                            _ => return Err("Expected element type after '<'".to_string()),
                        };
                        el_ty = match ty_ident.as_str() {
                            "f32" => ElementType::F32,
                            "f64" => ElementType::F64,
                            "bf16" => ElementType::BF16,
                            "i32" => ElementType::I32,
                            "i64" => ElementType::I64,
                            "Bool" => ElementType::Bool,
                            _ => {
                                if self.generic_params.contains(&ty_ident) {
                                    ElementType::Generic(ty_ident)
                                } else {
                                    return Err(format!("Unknown element type {}", ty_ident));
                                }
                            }
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
                                return Err("Expected '[' for Tensor dimensions".to_string());
                            }
                        }

                        let mut top = None;
                        if self.match_token(&TokenType::Comma)
                            && self.match_token(&TokenType::Identifier("Topology".to_string()))
                        {
                            self.consume(&TokenType::DoubleColon, "Expected '::' after Topology")?;
                            // Need to parse Topology... For now let's just parse the basic ones
                            if let TokenType::Identifier(t_name) = &self.peek().kind {
                                let t = t_name.clone();
                                self.advance();
                                if t == "ANE" {
                                    top = Some(Topology::ANE);
                                } else if t == "Host" {
                                    top = Some(Topology::Host);
                                } else if t == "AMX" {
                                    top = Some(Topology::AMX);
                                } else if t == "GPU" {
                                    top = Some(Topology::GPU);
                                }
                            }
                        }

                        match self.advance().kind {
                            TokenType::RightAngle => {}
                            _ => return Err("Expected '>' after Tensor parameters".to_string()),
                        }
                        return Ok(Type::Tensor(el_ty, dims, top));
                    }
                    Ok(Type::Tensor(el_ty, Vec::new(), None))
                }
                "Matrix" => Ok(Type::Matrix),
                "f32" => Ok(Type::Scalar(ElementType::F32)),
                "f64" => Ok(Type::Scalar(ElementType::F64)),
                "f16" => Ok(Type::Scalar(ElementType::F16)),
                "bf16" => Ok(Type::Scalar(ElementType::BF16)),
                "i8" => Ok(Type::Scalar(ElementType::I8)),
                "i16" => Ok(Type::Scalar(ElementType::I16)),
                "i32" => Ok(Type::Scalar(ElementType::I32)),
                "i64" => Ok(Type::Scalar(ElementType::I64)),
                "i128" => Ok(Type::Scalar(ElementType::I128)),
                "u8" => Ok(Type::Scalar(ElementType::U8)),
                "u16" => Ok(Type::Scalar(ElementType::U16)),
                "u32" => Ok(Type::Scalar(ElementType::U32)),
                "u64" => Ok(Type::Scalar(ElementType::U64)),
                "u128" => Ok(Type::Scalar(ElementType::U128)),
                "bool" | "Bool" => Ok(Type::Scalar(ElementType::Bool)),
                _ => {
                    // Check for GenericInstance like Config<f32>
                    if self.check(&TokenType::LeftAngle) {
                        self.advance(); // consume '<'
                        let mut type_args = Vec::new();
                        while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
                            // Try to parse an expression if it's a number literal
                            if let TokenType::Number(_) = self.peek().kind {
                                if let Ok(expr) = self.parse_expr() {
                                    type_args.push(Type::Const(Box::new(expr)));
                                } else {
                                    type_args.push(self.parse_type()?);
                                }
                            } else {
                                type_args.push(self.parse_type()?);
                            }
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                        self.consume(
                            &TokenType::RightAngle,
                            "Expected '>' after generic type arguments",
                        )?;
                        Ok(Type::GenericInstance(
                            Box::new(Type::Struct(ident, None)),
                            type_args,
                        ))
                    } else {
                        Ok(Type::Struct(ident, None))
                    }
                }
            }
        }
    }
}
