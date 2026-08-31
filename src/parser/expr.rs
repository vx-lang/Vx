//===- expr.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx expressions. Handles precedence climbing for binary operators and method chains.
//
//===----------------------------------------------------------------------===//

use super::*;

/// Split a numeric literal token into its numeric part and *explicit* element type.
///
/// Only a suffix (`80i32`, `1.5f64`) types the literal. An unsuffixed literal is *untyped*
/// (`None`) — Rust's model (#240): it adopts the expected type of its context during
/// type-checking, falling back to [`default_number_elem`] when no context supplies one. This
/// keeps implicit numeric conversion out of the language: the literal is born at the right type
/// rather than silently coerced.
pub(crate) fn infer_number_literal(s: &str) -> Result<(&str, Option<ElementType>), String> {
    let idx = s.find(|c: char| c.is_alphabetic() || c == '_');
    let (num_part, suffix_part) = match idx {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    };

    let el_ty = if !suffix_part.is_empty() {
        Some(suffix_part.parse::<ElementType>()?)
    } else {
        None
    };
    Ok((num_part, el_ty))
}

/// The spelling-based default element type for an *untyped* numeric literal — used when no
/// context supplies an expected type (a bare `42` or `3.14`). Reproduces the historical parser
/// defaults so an un-annotated literal keeps the type it always had: a decimal is `f32` (`f64`
/// only when it overflows `f32`); an integer is the narrowest of `i32`/`i64`/`i128` it fits.
///
/// This is the single source of the fallback default, shared by the type checker's untyped-literal
/// arm and the flat lowerer's `infer_elem`, so both backends agree on an un-annotated literal's
/// type (a differential-parity requirement). `num_part` is already suffix-stripped.
pub(crate) fn default_number_elem(num_part: &str) -> ElementType {
    if num_part.contains('.') {
        if let Ok(f32_val) = num_part.parse::<f32>() {
            if f32_val.is_infinite() {
                if let Ok(f64_val) = num_part.parse::<f64>() {
                    if f64_val.is_finite() {
                        return ElementType::F64;
                    }
                }
            }
        }
        ElementType::F32
    } else if num_part.parse::<i32>().is_ok() {
        ElementType::I32
    } else if num_part.parse::<i64>().is_ok() {
        ElementType::I64
    } else {
        ElementType::I128
    }
}

/// Give untyped numeric literals in a *type-position* dimension (a tensor shape or topology index)
/// their concrete default type. Unlike a value-position literal — which stays untyped and adopts
/// its type from context during type-checking (#240) — a dimension is a compile-time integer whose
/// type never varies. It also participates in structural type equality (a parsed `Tensor<f32, 10>`
/// must equal a synthesized one, whose dimensions are built typed), so it must be typed at parse
/// time rather than left for inference. Walks the compound dimension forms (ranges, arithmetic).
pub(crate) fn stamp_dim_literals(expr: &mut Expr) {
    match expr {
        Expr::Number(n) if n.ty.is_none() => {
            n.ty = Some(default_number_elem(&n.value));
        }
        Expr::Range(RangeExpr { start, end, .. }) => {
            stamp_dim_literals(start);
            stamp_dim_literals(end);
        }
        Expr::BinaryOp(b) => {
            stamp_dim_literals(&mut b.lhs);
            stamp_dim_literals(&mut b.rhs);
        }
        _ => {}
    }
}
impl<'a> Parser<'a> {
    pub(crate) fn parse_expr(&mut self) -> ParseResult<'a, Expr> {
        self.parse_binary_expr(0)
    }

    pub(crate) fn parse_binary_expr(&mut self, precedence: u8) -> ParseResult<'a, Expr> {
        let mut left = self.parse_primary_expr()?;

        while let Some(op_prec) = self.get_operator_precedence(&self.peek().kind) {
            if op_prec < precedence {
                break;
            }
            let token = self.advance().clone();
            match token.kind {
                TokenType::OrOr => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::LogicalOp(LogicalOpExpr {
                        lhs: Box::new(left),
                        op: LogicalOp::Or,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::AndAnd => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::LogicalOp(LogicalOpExpr {
                        lhs: Box::new(left),
                        op: LogicalOp::And,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::EqEq => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::Eq,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::NotEq => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::NotEq,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::LessEq => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::Le,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::GreaterEq => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::Ge,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::LeftAngle => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::Lt,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::RightAngle => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::RelationalOp(RelationalOpExpr {
                        lhs: Box::new(left),
                        op: RelationalOp::Gt,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::Plus => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::BinaryOp(BinaryOpExpr {
                        lhs: Box::new(left),
                        op: BinaryOp::Add,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::Minus => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::BinaryOp(BinaryOpExpr {
                        lhs: Box::new(left),
                        op: BinaryOp::Sub,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::Star => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::BinaryOp(BinaryOpExpr {
                        lhs: Box::new(left),
                        op: BinaryOp::Mul,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::At => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::BinaryOp(BinaryOpExpr {
                        lhs: Box::new(left),
                        op: BinaryOp::MatMul,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::Slash => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::BinaryOp(BinaryOpExpr {
                        lhs: Box::new(left),
                        op: BinaryOp::Div,
                        rhs: Box::new(right),
                        span: Span::default(),
                    });
                }
                TokenType::DoubleDot => {
                    let right = self.parse_binary_expr(op_prec + 1)?;
                    left = Expr::Range(RangeExpr {
                        start: Box::new(left),
                        end: Box::new(right),
                        span: Span::default(),
                    });
                }
                _ => return Err(self.error("Unknown binary operator")),
            };
        }

        Ok(left)
    }

    pub(crate) fn get_operator_precedence(&self, kind: &TokenType) -> Option<u8> {
        match kind {
            TokenType::OrOr => Some(10),
            TokenType::AndAnd => Some(20),
            TokenType::EqEq | TokenType::NotEq => Some(30),
            TokenType::LessEq
            | TokenType::GreaterEq
            | TokenType::LeftAngle
            | TokenType::RightAngle => Some(40),
            TokenType::DoubleDot => Some(45),
            TokenType::Plus | TokenType::Minus => Some(50),
            TokenType::Star | TokenType::Slash | TokenType::At => Some(60),
            _ => None,
        }
    }

    pub(crate) fn parse_pattern(&mut self) -> ParseResult<'a, Pattern> {
        if self.match_token(&TokenType::Identifier("_")) {
            return Ok(Pattern::Wildcard);
        }
        let token = self.peek().clone();
        match token.kind {
            TokenType::Number(_) | TokenType::StringLiteral(_) => {
                let expr = self.parse_primary_expr()?; // parses number or string
                Ok(Pattern::Literal(expr))
            }
            TokenType::Identifier(s) => {
                let mut enum_name = s.to_string();
                self.advance();
                if self.check(&TokenType::LeftAngle) {
                    self.advance(); // consume '<'
                    let ty_ident = match &self.advance().kind {
                        TokenType::Identifier(s) => s.to_string(),
                        _ => return Err(self.error("Expected type identifier in generic pattern")),
                    };
                    self.consume(&TokenType::RightAngle, "Expected '>' in generic pattern")?;
                    enum_name = format!("{}<{}>", enum_name, ty_ident);
                }
                if self.match_token(&TokenType::DoubleColon) {
                    let variant_name = match self.advance().kind.clone() {
                        TokenType::Identifier(v) => v.to_string(),
                        _ => return Err(self.error("Expected variant name")),
                    };
                    let mut payload = None;
                    if self.match_token(&TokenType::LeftParen) {
                        let mut p = Vec::new();
                        if !self.check(&TokenType::RightParen) {
                            loop {
                                p.push(self.parse_pattern()?);
                                if !self.match_token(&TokenType::Comma) {
                                    break;
                                }
                            }
                        }
                        self.consume(&TokenType::RightParen, "Expected ')'")?;
                        payload = Some(p);
                    }
                    Ok(Pattern::EnumVariant(
                        enum_name.into(),
                        variant_name.into(),
                        payload,
                    ))
                } else {
                    Ok(Pattern::Identifier(enum_name.into()))
                }
            }
            _ => Err(self.error(&format!("Unexpected token in pattern: {:?}", token.kind))),
        }
    }

    pub(crate) fn parse_identifier_expr(
        &mut self,
        mut call_name: String,
        span: Span,
    ) -> ParseResult<'a, Expr> {
        if call_name == "sizeof" {
            self.consume(&TokenType::LeftAngle, "Expected '<' after sizeof")?;
            let target_ty = self.parse_type()?;
            self.consume(&TokenType::RightAngle, "Expected '>' after sizeof type")?;
            self.consume(&TokenType::LeftParen, "Expected '(' after sizeof<...>")?;
            self.consume(&TokenType::RightParen, "Expected ')' after sizeof<...>(")?;
            return Ok(Expr::SizeOf(SizeOfExpr {
                target_ty,
                span: Span::default(),
            }));
        }

        // `Reachable<A, B>` as a value: a comptime transferability predicate. Same constraint
        // as the `where` clause of the same name, usable in `if comptime`. Spelled
        // `Transfer<A, B>` until Vx#353; see the `where` parser for why it moved.
        if call_name == "Reachable" && self.check(&TokenType::LeftAngle) {
            self.advance(); // consume '<'
            let from = self.parse_topology_operand()?;
            self.consume(&TokenType::Comma, "Expected ',' in Reachable<A, B>")?;
            let to = self.parse_topology_operand()?;
            self.consume(&TokenType::RightAngle, "Expected '>' after Reachable<A, B>")?;
            return Ok(Expr::TransferPredicate(TransferPredicateExpr {
                from,
                to,
                span,
            }));
        }

        let mut parsed_type_args = None;
        if self.check(&TokenType::LeftAngle) {
            let saved_pos = self.pos;
            self.advance(); // consume '<'
            if let Ok(type_args) = self.parse_generic_type_args() {
                if !type_args.is_empty() {
                    parsed_type_args = Some(type_args);
                }
            } else {
                self.pos = saved_pos;
            }
        }
        if self.check(&TokenType::DoubleColon) {
            let t1 = self.tokens.get(self.pos + 1).map(|t| &t.kind);
            let t2 = self.tokens.get(self.pos + 2).map(|t| &t.kind);
            let has_paren = matches!(
                (t1, t2),
                (Some(TokenType::Identifier(_)), Some(TokenType::LeftParen))
                    | (Some(TokenType::Identifier(_)), Some(TokenType::LeftAngle))
            );

            if has_paren {
                Self::apply_type_args(&mut call_name, parsed_type_args.take());

                self.advance(); // consume '::'
                if let TokenType::Identifier(method_name) = self.peek().kind.clone() {
                    self.advance(); // consume method name
                    call_name = format!("{}::{}", call_name, method_name);

                    if self.check(&TokenType::LeftAngle) {
                        let saved_pos = self.pos;
                        self.advance(); // consume '<'
                        if let Ok(type_args) = self.parse_generic_type_args() {
                            if !type_args.is_empty() {
                                parsed_type_args = Some(type_args);
                            }
                        } else {
                            self.pos = saved_pos;
                        }
                    }
                }
            }
        }
        if self.match_token(&TokenType::LeftParen) {
            let mut args = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Ok(Expr::FunctionCall(FunctionCallExpr {
                name: call_name.into(),
                type_args: parsed_type_args,
                args,
                span: Span::default(),
            }))
        } else if self.check(&TokenType::LeftBrace) {
            let is_struct_init = matches!(
                (
                    self.tokens.get(self.pos).map(|t| &t.kind),
                    self.tokens.get(self.pos + 1).map(|t| &t.kind),
                    self.tokens.get(self.pos + 2).map(|t| &t.kind),
                ),
                (Some(TokenType::LeftBrace), Some(TokenType::RightBrace), _)
                    | (
                        Some(TokenType::LeftBrace),
                        Some(TokenType::Identifier(_)),
                        Some(TokenType::Colon),
                    )
            );

            if is_struct_init {
                self.advance(); // consume '{'
                let mut fields = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    let token_kind = self.advance().kind.clone();
                    let f_name = match token_kind {
                        TokenType::Identifier(f) => f.to_string(),
                        _ => {
                            return Err(self.error(&format!(
                                "Expected field name in struct init, found {:?}",
                                token_kind
                            )))
                        }
                    };
                    self.consume(&TokenType::Colon, "Expected ':'")?;
                    let f_expr = self.parse_expr()?;
                    fields.push((f_name.into(), f_expr));
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Self::apply_type_args(&mut call_name, parsed_type_args);
                Ok(Expr::StructInit(StructInitExpr {
                    name: call_name.into(),
                    fields,
                    type_id: None,
                    span: Span::default(),
                }))
            } else if self.match_token(&TokenType::DoubleColon) {
                self.parse_enum_variant_expr(&mut call_name, parsed_type_args)
            } else {
                Self::apply_type_args(&mut call_name, parsed_type_args);
                Ok(Expr::Identifier(IdentifierExpr {
                    name: call_name.into(),
                    span,
                }))
            }
        } else if self.match_token(&TokenType::DoubleColon) {
            self.parse_enum_variant_expr(&mut call_name, parsed_type_args)
        } else {
            Self::apply_type_args(&mut call_name, parsed_type_args);
            Ok(Expr::Identifier(IdentifierExpr {
                name: call_name.into(),
                span,
            }))
        }
    }

    /// Applies parsed generic type arguments to a call name, e.g. `Foo` + `[i32, f64]` → `Foo<i32, f64>`.
    fn apply_type_args(call_name: &mut String, type_args: Option<Vec<Type>>) {
        if let Some(tys) = type_args {
            let ty_args_str = tys
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            *call_name = format!("{}<{}>", call_name, ty_args_str);
        }
    }

    /// Parses an enum variant expression after `::` has been consumed.
    /// Expects `VariantName` optionally followed by `(args...)`.
    fn parse_enum_variant_expr(
        &mut self,
        call_name: &mut String,
        parsed_type_args: Option<Vec<Type>>,
    ) -> ParseResult<'a, Expr> {
        let variant = match self.advance().kind.clone() {
            TokenType::Identifier(v) => v,
            _ => return Err(self.error("Expected enum variant after ::")),
        };
        let mut payload = None;
        if self.match_token(&TokenType::LeftParen) {
            let mut args = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    args.push(self.parse_expr()?);
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::RightParen, "Expected ')' after enum payload")?;
            payload = Some(args);
        }
        Self::apply_type_args(call_name, parsed_type_args);
        Ok(Expr::EnumVariant(EnumVariantExpr {
            enum_name: call_name.clone().into(),
            variant_name: variant.to_string().into(),
            payload,
            span: Span::default(),
        }))
    }

    pub(crate) fn parse_primary_expr(&mut self) -> ParseResult<'a, Expr> {
        if self.match_token(&TokenType::Bang) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::UnaryOp(UnaryOpExpr {
                op: UnaryOp::Not,
                expr: Box::new(inner),
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::Minus) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::UnaryOp(UnaryOpExpr {
                op: UnaryOp::Neg,
                expr: Box::new(inner),
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::Ampersand) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::Borrow(BorrowExpr {
                expr: Box::new(inner),
                is_mut,
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::Star) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::Dereference(DereferenceExpr {
                expr: Box::new(inner),
                ty: None,
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::Unsafe) {
            self.consume(&TokenType::LeftBrace, "Expected '{' after unsafe")?;
            let mut stmts = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                stmts.push(self.parse_statement()?);
            }
            let mut ret = None;
            if let Some(Statement::ExprStmt(ExprStmtStmt {
                expr,
                has_semi,
                span: _,
            })) = stmts.last().cloned()
            {
                if !has_semi {
                    stmts.pop();
                    ret = Some(Box::new(expr));
                }
            }
            self.consume(&TokenType::RightBrace, "Expected '}'")?;
            return Ok(Expr::UnsafeBlock(UnsafeBlockExpr {
                stmts,
                ret,
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::If) {
            let is_comptime = self.match_token(&TokenType::Comptime);
            let cond = self.parse_expr()?;
            self.consume(&TokenType::LeftBrace, "Expected '{'")?;
            let mut then_block = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                then_block.push(self.parse_statement()?);
            }
            self.consume(&TokenType::RightBrace, "Expected '}'")?;

            let mut else_block = None;
            if self.match_token(&TokenType::Else) {
                if self.check(&TokenType::If) {
                    // `else if`
                    let inner_if = self.parse_primary_expr()?;
                    else_block = Some(vec![Statement::ExprStmt(ExprStmtStmt {
                        expr: inner_if,
                        has_semi: false,
                        span: Span::default(),
                    })]);
                } else {
                    self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                    let mut block = Vec::new();
                    while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                        block.push(self.parse_statement()?);
                    }
                    self.consume(&TokenType::RightBrace, "Expected '}'")?;
                    else_block = Some(block);
                }
            }
            return Ok(Expr::If(IfExpr {
                is_comptime,
                cond: Box::new(cond),
                then_block,
                else_block,
                span: Span::default(),
            }));
        } else if self.match_token(&TokenType::Match) {
            let expr = self.parse_expr()?;
            self.consume(&TokenType::LeftBrace, "Expected '{' after match expr")?;
            let mut arms = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                let pattern = self.parse_pattern()?;
                self.consume(&TokenType::FatArrow, "Expected '=>' after pattern")?;
                let mut body = Vec::new();
                if self.match_token(&TokenType::LeftBrace) {
                    while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                        body.push(self.parse_statement()?);
                    }
                    self.consume(&TokenType::RightBrace, "Expected '}'")?;
                    self.match_token(&TokenType::Comma); // optional comma
                } else {
                    body.push(self.parse_statement()?);
                    self.match_token(&TokenType::Comma); // optional comma
                }
                arms.push(MatchArm { pattern, body });
            }
            self.consume(&TokenType::RightBrace, "Expected '}'")?;
            return Ok(Expr::Match(MatchExpr {
                expr: Box::new(expr),
                arms,
                span: Span::default(),
            }));
        }

        let peeked_token = self.peek().clone();
        let mut expr = match peeked_token.kind {
            TokenType::Transfer => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after transfer")?;
                let inner = self.parse_expr()?;
                self.consume(&TokenType::Comma, "Expected ','")?;
                let mem = self.parse_memory_space()?;
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                Expr::Transfer(TransferExpr {
                    expr: Box::new(inner),
                    space: mem,
                    cost: None,
                    lowering: None,
                    span: Span::default(),
                })
            }
            TokenType::LeftBracket => {
                self.advance();
                let mut elements = Vec::new();
                if !self.check(&TokenType::RightBracket) {
                    loop {
                        elements.push(self.parse_expr()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightBracket, "Expected ']'")?;
                Expr::Array(ArrayExpr {
                    elements,
                    span: Span::default(),
                })
            }
            TokenType::Memory => Expr::MemorySpace(MemorySpaceExpr {
                space: self.parse_memory_space()?,
                span: Span::default(),
            }),
            TokenType::Topology => Expr::Topology(TopologyExpr {
                top: self.parse_topology()?,
                span: Span::default(),
            }),
            TokenType::Verified => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after Verified")?;
                let inner = self.parse_expr()?;
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                Expr::FunctionCall(FunctionCallExpr {
                    name: "Verified".to_string().into(),
                    type_args: None,
                    args: vec![inner],
                    span: Span::default(),
                })
            }
            TokenType::Grad => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'grad'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => {
                        return Err(
                            self.error("Expected function identifier as first argument to grad")
                        )
                    }
                };
                let mut args = Vec::new();
                if self.match_token(&TokenType::Comma) && !self.check(&TokenType::RightParen) {
                    loop {
                        args.push(self.parse_expr()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                Expr::Grad(GradExpr {
                    target_fn: target_fn.into(),
                    args,
                    span: Span::default(),
                })
            }
            TokenType::Vjp => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'vjp'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => {
                        return Err(
                            self.error("Expected function identifier as first argument to vjp")
                        )
                    }
                };
                self.consume(&TokenType::Comma, "Expected comma after function name")?;
                let mut all_args = Vec::new();
                if !self.check(&TokenType::RightParen) {
                    loop {
                        all_args.push(self.parse_expr()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                if all_args.is_empty() {
                    return Err(self.error("Expected cotangent argument for vjp"));
                }
                let cotangent = all_args.pop().unwrap();
                Expr::Vjp(VjpExpr {
                    target_fn: target_fn.into(),
                    args: all_args,
                    cotangent: Box::new(cotangent),
                    span: Span::default(),
                })
            }
            TokenType::Jvp => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'jvp'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => {
                        return Err(
                            self.error("Expected function identifier as first argument to jvp")
                        )
                    }
                };
                self.consume(&TokenType::Comma, "Expected comma after function name")?;
                let mut all_args = Vec::new();
                if !self.check(&TokenType::RightParen) {
                    loop {
                        all_args.push(self.parse_expr()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                if all_args.is_empty() {
                    return Err(self.error("Expected tangent argument for jvp"));
                }
                let tangent = all_args.pop().unwrap();
                Expr::Jvp(JvpExpr {
                    target_fn: target_fn.into(),
                    args: all_args,
                    tangent: Box::new(tangent),
                    span: Span::default(),
                })
            }
            _ => {
                let token = self.advance().clone();
                match token.kind {
                    TokenType::LeftParen => {
                        let expr = self.parse_expr()?;
                        self.consume(&TokenType::RightParen, "Expected ')' after expression")?;
                        expr
                    }
                    TokenType::Identifier(s) => {
                        let ident_line = token.line;
                        let ident_end_col = token.column + token.length;

                        let mut is_macro = false;
                        if self.check(&TokenType::Bang) {
                            let bang_token = self.peek();
                            if bang_token.line == ident_line && bang_token.column == ident_end_col {
                                is_macro = true;
                            } else {
                                return Err(self.error_at(&token, &format!("Macro invocations must not have spaces between the macro name and '!'. Did you mean `{}!`?", s)));
                            }
                        }

                        if is_macro {
                            self.advance(); // consume Bang
                            let token_tree = self.parse_token_tree()?;
                            let mut block_tree = None;
                            if self.check(&TokenType::LeftBrace) {
                                block_tree = Some(self.parse_token_tree()?);
                            }
                            Expr::MacroCall(MacroCallExpr {
                                name: s.to_string().into(),
                                token_tree,
                                block_tree,
                                span: Span::default(),
                            })
                        } else {
                            let span = crate::syntax::Span {
                                line: token.line,
                                column: token.column,
                                length: token.length,
                            };
                            self.parse_identifier_expr(s.to_string(), span)?
                        }
                    }
                    TokenType::Return => self.parse_identifier_expr(
                        "return".to_string(),
                        crate::syntax::Span {
                            line: token.line,
                            column: token.column,
                            length: token.length,
                        },
                    )?,
                    TokenType::Number(s) => {
                        let (num_str, el_ty) =
                            infer_number_literal(s).map_err(|e| self.error_at(&token, &e))?;

                        Expr::Number(NumberExpr {
                            value: num_str.into(),
                            ty: el_ty,
                            span: Span::default(),
                        })
                    }
                    TokenType::StringLiteral(s) => Expr::StringLiteral(StringLiteralExpr {
                        value: s.into(),
                        span: Span::default(),
                    }),

                    TokenType::Comptime => {
                        self.consume(&TokenType::LeftBrace, "Expected '{' for comptime block")?;
                        let mut stmts = Vec::new();
                        let mut ret_expr = None;
                        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                            let stmt = self.parse_statement()?;
                            if self.check(&TokenType::RightBrace) {
                                if let Statement::ExprStmt(ExprStmtStmt {
                                    expr: e,
                                    has_semi,
                                    span: _,
                                }) = stmt
                                {
                                    if !has_semi {
                                        ret_expr = Some(Box::new(e));
                                    }
                                    break;
                                }
                            }
                            stmts.push(stmt);
                        }
                        self.consume(&TokenType::RightBrace, "Expected '}'")?;
                        Expr::ComptimeBlock(ComptimeBlockExpr {
                            stmts,
                            ret: ret_expr,
                            span: Span::default(),
                        })
                    }
                    TokenType::Pipe | TokenType::OrOr => {
                        let mut params = Vec::new();
                        if token.kind == TokenType::Pipe {
                            if !self.check(&TokenType::Pipe) {
                                loop {
                                    let name = match self.advance().kind.clone() {
                                        TokenType::Identifier(s) => s.to_string(),
                                        _ => {
                                            return Err(
                                                self.error("Expected identifier in closure params")
                                            )
                                        }
                                    };
                                    let ty = if self.match_token(&TokenType::Colon) {
                                        self.parse_type()?
                                    } else {
                                        Type::Unknown
                                    };
                                    params.push((name.into(), ty));
                                    if !self.match_token(&TokenType::Comma) {
                                        break;
                                    }
                                }
                            }
                            self.consume(&TokenType::Pipe, "Expected '|' after closure params")?;
                        }
                        let body = Box::new(self.parse_expr()?);
                        Expr::Closure(ClosureExpr {
                            params,
                            body,
                            ret_ty: None,
                            captures: vec![],
                            span: Span::default(),
                        })
                    }
                    TokenType::Spawn => {
                        self.consume(&TokenType::On, "Expected 'on' after 'spawn'")?;
                        self.consume(&TokenType::LeftParen, "Expected '('")?;
                        let top = self.parse_topology()?;
                        self.consume(&TokenType::RightParen, "Expected ')'")?;
                        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                        let mut stmts = Vec::new();
                        let mut ret_expr = None;
                        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                            let stmt = self.parse_statement()?;
                            if self.check(&TokenType::RightBrace) {
                                if let Statement::ExprStmt(ExprStmtStmt {
                                    expr: e,
                                    has_semi,
                                    span: _,
                                }) = &stmt
                                {
                                    if !*has_semi {
                                        ret_expr = Some(Box::new(e.clone()));
                                        break;
                                    }
                                }
                            }
                            stmts.push(stmt);
                        }
                        self.consume(&TokenType::RightBrace, "Expected '}'")?;
                        Expr::SpawnOn(SpawnOnExpr {
                            top,
                            stmts,
                            ret: ret_expr,
                            span: Span::default(),
                        })
                    }
                    _ => {
                        return Err(self.error_at(
                            &token,
                            &format!("Expected expression, found {:?}", token.kind),
                        ))
                    }
                }
            }
        };

        // Postfix operators: .member, .method(), [index]
        loop {
            if self.match_token(&TokenType::Dot) {
                let ident = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => return Err(self.error("Expected identifier after '.'")),
                };
                if self.match_token(&TokenType::LeftParen) {
                    let mut args = Vec::new();
                    if !self.check(&TokenType::RightParen) {
                        loop {
                            args.push(self.parse_expr()?);
                            if !self.match_token(&TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    self.consume(&TokenType::RightParen, "Expected ')'")?;
                    expr = Expr::MethodCall(MethodCallExpr {
                        base: Box::new(expr),
                        method_name: ident.to_string().into(),
                        type_args: None,
                        args,
                        span: Span::default(),
                    });
                } else {
                    expr = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(expr),
                        member: ident.into(),
                        struct_name: None,
                        span: Span::default(),
                    });
                }
            } else if self.match_token(&TokenType::LeftParen) {
                let mut args = Vec::new();
                if !self.check(&TokenType::RightParen) {
                    loop {
                        args.push(self.parse_expr()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                expr = Expr::IndirectCall(IndirectCallExpr {
                    callee: Box::new(expr),
                    args,
                    target_func_ty: None,
                    span: Span::default(),
                });
            } else if self.match_token(&TokenType::LeftBracket) {
                let index = self.parse_expr()?;
                self.consume(&TokenType::RightBracket, "Expected ']'")?;
                expr = Expr::IndexAccess(IndexAccessExpr {
                    base: Box::new(expr),
                    index: Box::new(index),
                    span: Span::default(),
                });
            } else if self.match_token(&TokenType::As) {
                let target_ty = self.parse_type()?;
                expr = Expr::AsCast(AsCastExpr {
                    expr: Box::new(expr),
                    target_ty,
                    source_ty: None,
                    span: Span::default(),
                });
            } else {
                break;
            }
        }

        Ok(expr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse_expr(input: &str) -> Expr {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(&tokens, input);
        parser.parse_expr().expect("Failed to parse expression")
    }

    #[test]
    fn test_parse_enum_variant_no_payload() {
        let input = "Option::None";
        let expr = parse_expr(input);
        if let Expr::EnumVariant(EnumVariantExpr {
            enum_name,
            variant_name,
            payload,
            span: _,
        }) = expr
        {
            assert_eq!(enum_name.as_ref(), "Option");
            assert_eq!(variant_name.as_ref(), "None");
            assert!(payload.is_none());
        } else {
            panic!("Expected EnumVariant, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_enum_variant_with_payload() {
        let input = "Option::Some(x)";
        let expr = parse_expr(input);
        // Enum variants with payloads are parsed as FunctionCall in the parser phase.
        // They are later rewritten to EnumVariant by Sema.
        if let Expr::FunctionCall(FunctionCallExpr {
            name,
            args,
            span: _,
            type_args: _,
        }) = expr
        {
            assert_eq!(name.as_ref(), "Option::Some");
            assert_eq!(args.len(), 1);
            if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &args[0] {
                assert_eq!(name.as_ref(), "x");
            } else {
                panic!("Expected Identifier payload");
            }
        } else {
            panic!("Expected FunctionCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_generic_enum_variant() {
        let input = "Option<i32>::Some(x)";
        let expr = parse_expr(input);
        // Similarly, generic enum variants with payloads are FunctionCalls initially.
        if let Expr::FunctionCall(FunctionCallExpr {
            name,
            args,
            span: _,
            type_args: _,
        }) = expr
        {
            assert_eq!(name.as_ref(), "Option<i32>::Some");
            assert_eq!(args.len(), 1);
        } else {
            panic!("Expected FunctionCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_generic_vector() {
        let input = "Vec<i32>::new()";
        let expr = parse_expr(input);
        if let Expr::FunctionCall(FunctionCallExpr {
            name,
            args,
            span: _,
            type_args: _,
        }) = expr
        {
            assert_eq!(name.as_ref(), "Vec<i32>::new");
            assert_eq!(args.len(), 0);
        } else {
            panic!("Expected FunctionCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_generic_vector_push() {
        let input = "vec.push(10)";
        let expr = parse_expr(input);
        if let Expr::MethodCall(MethodCallExpr {
            base,
            method_name,
            args,
            span: _,
            type_args: _,
        }) = expr
        {
            if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &*base {
                assert_eq!(name.as_ref(), "vec");
            } else {
                panic!("Expected Identifier base");
            }
            assert_eq!(method_name.as_ref(), "push");
            assert_eq!(args.len(), 1);
            if let Expr::Number(NumberExpr { value, .. }) = &args[0] {
                assert_eq!(value.as_ref(), "10");
            } else {
                panic!("Expected Number payload");
            }
        } else {
            panic!("Expected MethodCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_iterators() {
        let input = "vec.iter().map(f)";
        let expr = parse_expr(input);
        if let Expr::MethodCall(MethodCallExpr {
            base,
            method_name,
            args,
            span: _,
            type_args: _,
        }) = expr
        {
            assert_eq!(method_name.as_ref(), "map");
            assert_eq!(args.len(), 1);
            if let Expr::MethodCall(MethodCallExpr {
                base: inner_base,
                method_name: inner_method_name,
                args: inner_args,
                span: _,
                type_args: _,
            }) = &*base
            {
                assert_eq!(inner_method_name.as_ref(), "iter");
                assert_eq!(inner_args.len(), 0);
                if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &**inner_base {
                    assert_eq!(name.as_ref(), "vec");
                } else {
                    panic!("Expected Identifier inner_base");
                }
            } else {
                panic!("Expected inner MethodCall");
            }
        } else {
            panic!("Expected MethodCall, got {:?}", expr);
        }
    }

    // ---- infer_number_literal tests ----

    #[test]
    fn test_unsuffixed_literal_is_untyped() {
        // An unsuffixed literal carries no type — it is inferred from context (#240).
        let (num, ty) = infer_number_literal("42").unwrap();
        assert_eq!(num, "42");
        assert_eq!(ty, None);
    }

    #[test]
    fn test_default_integer_i32() {
        assert_eq!(default_number_elem("42"), ElementType::I32);
    }

    #[test]
    fn test_default_integer_large_promotes_to_i64() {
        // 3_000_000_000 overflows i32 but fits i64
        assert_eq!(default_number_elem("3000000000"), ElementType::I64);
    }

    #[test]
    fn test_default_integer_huge_promotes_to_i128() {
        // Overflows i64
        assert_eq!(
            default_number_elem("99999999999999999999"),
            ElementType::I128
        );
    }

    #[test]
    fn test_default_float_defaults_to_f32() {
        assert_eq!(default_number_elem("3.14"), ElementType::F32);
    }

    #[test]
    fn test_infer_scientific_notation_is_suffix_split() {
        // infer_number_literal splits at first alphabetic char.
        // "1e10" → num_part="1", suffix="e10" → Err (e10 is not a valid ElementType)
        // This is a known limitation: scientific notation is not supported directly.
        assert!(infer_number_literal("1e10").is_err());
        // "3.4e39" → num_part="3.4", suffix="e39" → Err
        assert!(infer_number_literal("3.4e39").is_err());
    }

    #[test]
    fn test_infer_suffixed_literal() {
        let (num, ty) = infer_number_literal("42i64").unwrap();
        assert_eq!(num, "42");
        assert_eq!(ty, Some(ElementType::I64));
    }

    #[test]
    fn test_infer_suffixed_float_literal() {
        let (num, ty) = infer_number_literal("3.14f64").unwrap();
        assert_eq!(num, "3.14");
        assert_eq!(ty, Some(ElementType::F64));
    }

    // ---- Binary operator precedence tests ----

    #[test]
    fn test_parse_binary_add_mul_precedence() {
        // "a + b * c" should parse as Add(a, Mul(b, c)) due to * having higher precedence
        let expr = parse_expr("a + b * c");
        if let Expr::BinaryOp(BinaryOpExpr { lhs, op, rhs, .. }) = expr {
            assert_eq!(op, BinaryOp::Add);
            assert!(matches!(&*lhs, Expr::Identifier(_)));
            if let Expr::BinaryOp(BinaryOpExpr { op: inner_op, .. }) = &*rhs {
                assert_eq!(*inner_op, BinaryOp::Mul);
            } else {
                panic!("Expected Mul on RHS, got {:?}", rhs);
            }
        } else {
            panic!("Expected BinaryOp, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_logical_and_or_precedence() {
        // "a && b || c" → Or(And(a, b), c) because && binds tighter than ||
        let expr = parse_expr("a && b || c");
        if let Expr::LogicalOp(LogicalOpExpr { op, lhs, .. }) = expr {
            assert_eq!(op, LogicalOp::Or);
            if let Expr::LogicalOp(LogicalOpExpr { op: inner_op, .. }) = &*lhs {
                assert_eq!(*inner_op, LogicalOp::And);
            } else {
                panic!("Expected And on LHS, got {:?}", lhs);
            }
        } else {
            panic!("Expected LogicalOp, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_relational_less_equal() {
        let expr = parse_expr("a <= b");
        if let Expr::RelationalOp(RelationalOpExpr { op, .. }) = expr {
            assert_eq!(op, RelationalOp::Le);
        } else {
            panic!("Expected RelationalOp, got {:?}", expr);
        }
    }

    // ---- Unary operator tests ----

    #[test]
    fn test_parse_unary_negation() {
        let expr = parse_expr("-x");
        if let Expr::UnaryOp(UnaryOpExpr {
            op, expr: inner, ..
        }) = expr
        {
            assert_eq!(op, UnaryOp::Neg);
            assert!(matches!(&*inner, Expr::Identifier(_)));
        } else {
            panic!("Expected UnaryOp(Neg), got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_unary_not() {
        let expr = parse_expr("!flag");
        if let Expr::UnaryOp(UnaryOpExpr {
            op, expr: inner, ..
        }) = expr
        {
            assert_eq!(op, UnaryOp::Not);
            if let Expr::Identifier(IdentifierExpr { name, .. }) = &*inner {
                assert_eq!(name.as_ref(), "flag");
            } else {
                panic!("Expected Identifier, got {:?}", inner);
            }
        } else {
            panic!("Expected UnaryOp(Not), got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_borrow_immutable() {
        let expr = parse_expr("&x");
        if let Expr::Borrow(BorrowExpr { is_mut, .. }) = expr {
            assert!(!is_mut);
        } else {
            panic!("Expected Borrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_borrow_mutable() {
        let expr = parse_expr("&mut x");
        if let Expr::Borrow(BorrowExpr { is_mut, .. }) = expr {
            assert!(is_mut);
        } else {
            panic!("Expected Borrow, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_dereference() {
        let expr = parse_expr("*ptr");
        if let Expr::Dereference(DereferenceExpr { expr: inner, .. }) = expr {
            if let Expr::Identifier(IdentifierExpr { name, .. }) = &*inner {
                assert_eq!(name.as_ref(), "ptr");
            } else {
                panic!("Expected Identifier, got {:?}", inner);
            }
        } else {
            panic!("Expected Dereference, got {:?}", expr);
        }
    }

    // ---- Expression kind tests ----

    #[test]
    fn test_parse_function_call() {
        let expr = parse_expr("foo(a, b, c)");
        if let Expr::FunctionCall(FunctionCallExpr { name, args, .. }) = expr {
            assert_eq!(name.as_ref(), "foo");
            assert_eq!(args.len(), 3);
        } else {
            panic!("Expected FunctionCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_function_call_no_args() {
        let expr = parse_expr("bar()");
        if let Expr::FunctionCall(FunctionCallExpr { name, args, .. }) = expr {
            assert_eq!(name.as_ref(), "bar");
            assert_eq!(args.len(), 0);
        } else {
            panic!("Expected FunctionCall, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_struct_init() {
        let input = "Point { x: 1, y: 2 }";
        let expr = parse_expr(input);
        if let Expr::StructInit(StructInitExpr { name, fields, .. }) = expr {
            assert_eq!(name.as_ref(), "Point");
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].0.as_ref(), "x");
            assert_eq!(fields[1].0.as_ref(), "y");
        } else {
            panic!("Expected StructInit, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_array_literal() {
        let expr = parse_expr("[1, 2, 3]");
        if let Expr::Array(ArrayExpr { elements, .. }) = expr {
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Array, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_range() {
        let expr = parse_expr("0..10");
        if let Expr::Range(RangeExpr { start, end, .. }) = expr {
            if let Expr::Number(NumberExpr { value, .. }) = &*start {
                assert_eq!(value.as_ref(), "0");
            } else {
                panic!("Expected Number start");
            }
            if let Expr::Number(NumberExpr { value, .. }) = &*end {
                assert_eq!(value.as_ref(), "10");
            } else {
                panic!("Expected Number end");
            }
        } else {
            panic!("Expected Range, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_member_access_chain() {
        let expr = parse_expr("a.b.c");
        // Should be MemberAccess(MemberAccess(a, b), c)
        if let Expr::MemberAccess(MemberAccessExpr { base, member, .. }) = expr {
            assert_eq!(member.as_ref(), "c");
            if let Expr::MemberAccess(MemberAccessExpr {
                base: inner_base,
                member: inner_member,
                ..
            }) = &*base
            {
                assert_eq!(inner_member.as_ref(), "b");
                if let Expr::Identifier(IdentifierExpr { name, .. }) = &**inner_base {
                    assert_eq!(name.as_ref(), "a");
                } else {
                    panic!("Expected Identifier at root");
                }
            } else {
                panic!("Expected inner MemberAccess, got {:?}", base);
            }
        } else {
            panic!("Expected MemberAccess, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_matmul_operator() {
        let expr = parse_expr("a @ b");
        if let Expr::BinaryOp(BinaryOpExpr { op, .. }) = expr {
            assert_eq!(op, BinaryOp::MatMul);
        } else {
            panic!("Expected BinaryOp(MatMul), got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_index_access() {
        let expr = parse_expr("arr[0]");
        if let Expr::IndexAccess(IndexAccessExpr { base, index, .. }) = expr {
            if let Expr::Identifier(IdentifierExpr { name, .. }) = &*base {
                assert_eq!(name.as_ref(), "arr");
            } else {
                panic!("Expected Identifier base");
            }
            if let Expr::Number(NumberExpr { value, .. }) = &*index {
                assert_eq!(value.as_ref(), "0");
            } else {
                panic!("Expected Number index");
            }
        } else {
            panic!("Expected IndexAccess, got {:?}", expr);
        }
    }

    // ---- If expression tests ----

    #[test]
    fn test_parse_if_basic() {
        let expr = parse_expr("if x { 1; }");
        if let Expr::If(IfExpr {
            is_comptime,
            cond,
            then_block,
            else_block,
            ..
        }) = expr
        {
            assert!(!is_comptime);
            assert!(matches!(&*cond, Expr::Identifier(_)));
            assert_eq!(then_block.len(), 1);
            assert!(else_block.is_none());
        } else {
            panic!("Expected If, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_if_comptime() {
        let expr = parse_expr("if comptime x { 1; }");
        if let Expr::If(IfExpr { is_comptime, .. }) = expr {
            assert!(is_comptime);
        } else {
            panic!("Expected If(comptime), got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_if_else() {
        let expr = parse_expr("if x { 1; } else { 2; }");
        if let Expr::If(IfExpr {
            then_block,
            else_block,
            ..
        }) = expr
        {
            assert_eq!(then_block.len(), 1);
            assert!(else_block.is_some());
            assert_eq!(else_block.unwrap().len(), 1);
        } else {
            panic!("Expected If with else, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_if_else_if_chain() {
        let expr = parse_expr("if a { 1; } else if b { 2; } else { 3; }");
        if let Expr::If(IfExpr { else_block, .. }) = expr {
            // else if → else_block = Some([ExprStmt(If(...))])
            let else_b = else_block.expect("Expected else block");
            assert_eq!(else_b.len(), 1);
            if let Statement::ExprStmt(ExprStmtStmt { expr: inner, .. }) = &else_b[0] {
                assert!(matches!(inner, Expr::If(_)));
            } else {
                panic!("Expected inner If in else-if chain");
            }
        } else {
            panic!("Expected If, got {:?}", expr);
        }
    }

    // ---- Match expression tests ----

    #[test]
    fn test_parse_match_with_wildcard() {
        let expr = parse_expr("match x { _ => { 0; } }");
        if let Expr::Match(MatchExpr {
            expr: scrutinee,
            arms,
            ..
        }) = expr
        {
            assert!(matches!(&*scrutinee, Expr::Identifier(_)));
            assert_eq!(arms.len(), 1);
            assert_eq!(arms[0].pattern, Pattern::Wildcard);
        } else {
            panic!("Expected Match, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_match_with_literal_and_identifier() {
        let expr = parse_expr("match x { 1 => { a; } y => { b; } }");
        if let Expr::Match(MatchExpr { arms, .. }) = expr {
            assert_eq!(arms.len(), 2);
            assert!(matches!(
                &arms[0].pattern,
                Pattern::Literal(Expr::Number(_))
            ));
            assert!(matches!(&arms[1].pattern, Pattern::Identifier(_)));
        } else {
            panic!("Expected Match, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_match_enum_variant_pattern() {
        let expr = parse_expr("match x { Option::None => { 0; } Option::Some(v) => { v; } }");
        if let Expr::Match(MatchExpr { arms, .. }) = expr {
            assert_eq!(arms.len(), 2);
            if let Pattern::EnumVariant(enum_name, variant, payload) = &arms[0].pattern {
                assert_eq!(enum_name.as_ref(), "Option");
                assert_eq!(variant.as_ref(), "None");
                assert!(payload.is_none());
            } else {
                panic!("Expected EnumVariant pattern for first arm");
            }
            if let Pattern::EnumVariant(_, variant, payload) = &arms[1].pattern {
                assert_eq!(variant.as_ref(), "Some");
                assert!(payload.is_some());
                assert_eq!(payload.as_ref().unwrap().len(), 1);
            } else {
                panic!("Expected EnumVariant pattern for second arm");
            }
        } else {
            panic!("Expected Match, got {:?}", expr);
        }
    }

    // ---- Unsafe block expression tests ----

    #[test]
    fn test_parse_unsafe_block_with_stmts() {
        let expr = parse_expr("unsafe { call(); }");
        if let Expr::UnsafeBlock(UnsafeBlockExpr { stmts, ret, .. }) = expr {
            assert_eq!(stmts.len(), 1);
            assert!(ret.is_none());
        } else {
            panic!("Expected UnsafeBlock, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_unsafe_block_tail_expr() {
        let expr = parse_expr("unsafe { x }");
        if let Expr::UnsafeBlock(UnsafeBlockExpr { stmts, ret, .. }) = expr {
            assert!(stmts.is_empty());
            assert!(ret.is_some());
            assert!(matches!(&*ret.unwrap(), Expr::Identifier(_)));
        } else {
            panic!("Expected UnsafeBlock with tail, got {:?}", expr);
        }
    }

    // ---- Comptime block expression tests ----

    #[test]
    fn test_parse_comptime_block_tail_expr() {
        let expr = parse_expr("comptime { 42 }");
        if let Expr::ComptimeBlock(ComptimeBlockExpr { stmts, ret, .. }) = expr {
            assert!(stmts.is_empty());
            assert!(ret.is_some());
            if let Expr::Number(NumberExpr { value, .. }) = &*ret.unwrap() {
                assert_eq!(value.as_ref(), "42");
            } else {
                panic!("Expected Number in comptime ret");
            }
        } else {
            panic!("Expected ComptimeBlock, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_comptime_block_with_stmts() {
        let expr = parse_expr("comptime { let x : i32 = 1; x }");
        if let Expr::ComptimeBlock(ComptimeBlockExpr { stmts, ret, .. }) = expr {
            assert_eq!(stmts.len(), 1);
            assert!(ret.is_some());
        } else {
            panic!("Expected ComptimeBlock, got {:?}", expr);
        }
    }

    // ---- Grad expression tests ----

    #[test]
    fn test_parse_grad() {
        let expr = parse_expr("grad(f, x, y)");
        if let Expr::Grad(GradExpr {
            target_fn, args, ..
        }) = expr
        {
            assert_eq!(target_fn.as_ref(), "f");
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected Grad, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_grad_no_args() {
        let expr = parse_expr("grad(f)");
        if let Expr::Grad(GradExpr {
            target_fn, args, ..
        }) = expr
        {
            assert_eq!(target_fn.as_ref(), "f");
            assert!(args.is_empty());
        } else {
            panic!("Expected Grad with no args, got {:?}", expr);
        }
    }

    // ---- Vjp expression tests ----

    #[test]
    fn test_parse_vjp() {
        let expr = parse_expr("vjp(f, x, y, ct)");
        if let Expr::Vjp(VjpExpr {
            target_fn,
            args,
            cotangent,
            ..
        }) = expr
        {
            assert_eq!(target_fn.as_ref(), "f");
            // Last arg split off as cotangent
            assert_eq!(args.len(), 2);
            assert!(matches!(&*cotangent, Expr::Identifier(_)));
        } else {
            panic!("Expected Vjp, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_vjp_single_arg_is_cotangent() {
        // vjp(f, ct) — ct is the cotangent, args should be empty
        let expr = parse_expr("vjp(f, ct)");
        if let Expr::Vjp(VjpExpr {
            target_fn,
            args,
            cotangent,
            ..
        }) = expr
        {
            assert_eq!(target_fn.as_ref(), "f");
            assert!(args.is_empty());
            if let Expr::Identifier(IdentifierExpr { name, .. }) = &*cotangent {
                assert_eq!(name.as_ref(), "ct");
            } else {
                panic!("Expected Identifier cotangent");
            }
        } else {
            panic!("Expected Vjp, got {:?}", expr);
        }
    }

    // ---- Jvp expression tests ----

    #[test]
    fn test_parse_jvp() {
        let expr = parse_expr("jvp(f, x, tn)");
        if let Expr::Jvp(JvpExpr {
            target_fn,
            args,
            tangent,
            ..
        }) = expr
        {
            assert_eq!(target_fn.as_ref(), "f");
            assert_eq!(args.len(), 1);
            if let Expr::Identifier(IdentifierExpr { name, .. }) = &*tangent {
                assert_eq!(name.as_ref(), "tn");
            } else {
                panic!("Expected Identifier tangent");
            }
        } else {
            panic!("Expected Jvp, got {:?}", expr);
        }
    }

    // ---- Closure expression tests ----

    #[test]
    fn test_parse_closure_typed_params() {
        let expr = parse_expr("|x: i32, y: f32| x");
        if let Expr::Closure(ClosureExpr { params, body, .. }) = expr {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0].0.as_ref(), "x");
            assert_eq!(params[1].0.as_ref(), "y");
            assert!(matches!(&*body, Expr::Identifier(_)));
        } else {
            panic!("Expected Closure, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_closure_zero_arg() {
        let expr = parse_expr("|| 42");
        if let Expr::Closure(ClosureExpr { params, body, .. }) = expr {
            assert!(params.is_empty());
            assert!(matches!(&*body, Expr::Number(_)));
        } else {
            panic!("Expected Closure(zero-arg), got {:?}", expr);
        }
    }

    // ---- AsCast expression tests ----

    #[test]
    fn test_parse_as_cast() {
        let expr = parse_expr("x as f64");
        if let Expr::AsCast(AsCastExpr {
            expr: inner,
            target_ty,
            ..
        }) = expr
        {
            assert!(matches!(&*inner, Expr::Identifier(_)));
            assert_eq!(target_ty.to_string(), "f64");
        } else {
            panic!("Expected AsCast, got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_as_cast_chained() {
        // x as i32 as f64 → AsCast(AsCast(x, i32), f64)
        let expr = parse_expr("x as i32 as f64");
        if let Expr::AsCast(AsCastExpr {
            expr: inner,
            target_ty,
            ..
        }) = expr
        {
            assert_eq!(target_ty.to_string(), "f64");
            assert!(matches!(&*inner, Expr::AsCast(_)));
        } else {
            panic!("Expected chained AsCast, got {:?}", expr);
        }
    }

    // ---- IndirectCall expression tests ----

    #[test]
    fn test_parse_indirect_call() {
        // A parenthesized expression followed by (args) produces an IndirectCall
        let expr = parse_expr("(get_fn())(a, b)");
        if let Expr::IndirectCall(IndirectCallExpr { callee, args, .. }) = expr {
            assert_eq!(args.len(), 2);
            assert!(matches!(&*callee, Expr::FunctionCall(_)));
        } else {
            panic!("Expected IndirectCall, got {:?}", expr);
        }
    }

    // ---- SizeOf expression tests ----

    #[test]
    fn test_parse_sizeof() {
        let expr = parse_expr("sizeof<i32>()");
        if let Expr::SizeOf(SizeOfExpr { target_ty, .. }) = expr {
            assert_eq!(target_ty.to_string(), "i32");
        } else {
            panic!("Expected SizeOf, got {:?}", expr);
        }
    }

    // ---- StringLiteral expression tests ----

    #[test]
    fn test_parse_string_literal() {
        let expr = parse_expr("\"hello world\"");
        if let Expr::StringLiteral(StringLiteralExpr { value, .. }) = expr {
            assert_eq!(value.as_ref(), "hello world");
        } else {
            panic!("Expected StringLiteral, got {:?}", expr);
        }
    }

    // ---- Dereference chain test ----

    #[test]
    fn test_parse_dereference_chain() {
        // **ptr → Dereference(Dereference(ptr))
        let expr = parse_expr("**ptr");
        if let Expr::Dereference(DereferenceExpr { expr: inner, .. }) = expr {
            if let Expr::Dereference(DereferenceExpr {
                expr: innermost, ..
            }) = &*inner
            {
                assert!(matches!(&**innermost, Expr::Identifier(_)));
            } else {
                panic!("Expected inner Dereference, got {:?}", inner);
            }
        } else {
            panic!("Expected Dereference, got {:?}", expr);
        }
    }

    // ---- Complex expression composition tests ----

    #[test]
    fn test_parse_borrow_then_deref() {
        // *&x → Dereference(Borrow(x))
        let expr = parse_expr("*&x");
        if let Expr::Dereference(DereferenceExpr { expr: inner, .. }) = expr {
            assert!(matches!(&*inner, Expr::Borrow(_)));
        } else {
            panic!("Expected Dereference(Borrow), got {:?}", expr);
        }
    }

    #[test]
    fn test_parse_negation_of_function_call() {
        // -foo(x) → UnaryOp(Neg, FunctionCall)
        let expr = parse_expr("-foo(x)");
        if let Expr::UnaryOp(UnaryOpExpr {
            op, expr: inner, ..
        }) = expr
        {
            assert_eq!(op, UnaryOp::Neg);
            assert!(matches!(&*inner, Expr::FunctionCall(_)));
        } else {
            panic!("Expected UnaryOp(Neg, FunctionCall), got {:?}", expr);
        }
    }
}
