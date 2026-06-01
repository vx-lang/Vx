use super::*;

impl<'a> Parser<'a> {
    pub(crate) fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_binary_expr(0)
    }

    pub(crate) fn parse_binary_expr(&mut self, precedence: u8) -> Result<Expr, String> {
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
                _ => return Err("Unknown binary operator".to_string()),
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

    pub(crate) fn parse_pattern(&mut self) -> Result<Pattern, String> {
        if self.match_token(&TokenType::Identifier("_".to_string())) {
            return Ok(Pattern::Wildcard);
        }
        let token = self.peek().clone();
        match token.kind {
            TokenType::Number(_) | TokenType::StringLiteral(_) => {
                let expr = self.parse_primary_expr()?; // parses number or string
                Ok(Pattern::Literal(expr))
            }
            TokenType::Identifier(s) => {
                self.advance();
                if self.match_token(&TokenType::DoubleColon) {
                    let variant_name = match &self.advance().kind {
                        TokenType::Identifier(v) => v.clone(),
                        _ => return Err("Expected variant name".to_string()),
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
                    Ok(Pattern::EnumVariant(s, variant_name, payload))
                } else {
                    Ok(Pattern::Identifier(s))
                }
            }
            _ => Err(format!("Unexpected token in pattern: {:?}", token.kind)),
        }
    }

    pub(crate) fn parse_primary_expr(&mut self) -> Result<Expr, String> {
        if self.match_token(&TokenType::Bang) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::UnaryOp(UnaryOpExpr {
                op: UnaryOp::Not,
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

        let mut expr = if self.match_token(&TokenType::Transfer) {
            self.consume(&TokenType::LeftParen, "Expected '(' after transfer")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let mem = self.parse_memory_space()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::Transfer(TransferExpr {
                expr: Box::new(inner),
                space: mem,
                span: Span::default(),
            })
        } else if self.match_token(&TokenType::LeftBracket) {
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
        } else if self.check(&TokenType::Memory) {
            Expr::MemorySpace(MemorySpaceExpr {
                space: self.parse_memory_space()?,
                span: Span::default(),
            })
        } else if self.check(&TokenType::Topology) {
            Expr::Topology(TopologyExpr {
                top: self.parse_topology()?,
                span: Span::default(),
            })
        } else if self.check(&TokenType::Verified) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after Verified")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::FunctionCall(FunctionCallExpr {
                name: "Verified".to_string(),
                args: vec![inner],
                span: Span::default(),
            })
        } else if self.check(&TokenType::Grad) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after 'grad'")?;
            let target_fn = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => {
                    return Err("Expected function identifier as first argument to grad".to_string())
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
                target_fn,
                args,
                span: Span::default(),
            })
        } else if self.check(&TokenType::Vjp) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after 'vjp'")?;
            let target_fn = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => {
                    return Err("Expected function identifier as first argument to vjp".to_string())
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
                return Err("Expected cotangent argument for vjp".to_string());
            }
            let cotangent = all_args.pop().unwrap();
            Expr::Vjp(VjpExpr {
                target_fn,
                args: all_args,
                cotangent: Box::new(cotangent),
                span: Span::default(),
            })
        } else if self.check(&TokenType::Jvp) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after 'jvp'")?;
            let target_fn = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => {
                    return Err("Expected function identifier as first argument to jvp".to_string())
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
                return Err("Expected tangent argument for jvp".to_string());
            }
            let tangent = all_args.pop().unwrap();
            Expr::Jvp(JvpExpr {
                target_fn,
                args: all_args,
                tangent: Box::new(tangent),
                span: Span::default(),
            })
        } else {
            let token = self.advance().clone();
            match token.kind {
                TokenType::LeftParen => {
                    let expr = self.parse_expr()?;
                    self.consume(&TokenType::RightParen, "Expected ')' after expression")?;
                    expr
                }
                TokenType::Identifier(s) => {
                    let mut call_name = s;
                    if call_name == "Tensor" {
                        if let TokenType::LeftAngle = &self.peek().kind {
                            self.advance(); // consume '<'
                            let ty_ident = match self.advance().kind.clone() {
                                TokenType::Identifier(s) => s,
                                _ => return Err("Expected element type after '<'".to_string()),
                            };
                            match self.advance().kind {
                                TokenType::RightAngle => {}
                                _ => return Err("Expected '>' after element type".to_string()),
                            }
                            call_name = format!("Tensor_{}", ty_ident);
                        }
                    }
                    if self.check(&TokenType::DoubleColon) {
                        let has_paren = matches!(
                            (
                                self.tokens.get(self.pos + 1).map(|t| &t.kind),
                                self.tokens.get(self.pos + 2).map(|t| &t.kind),
                            ),
                            (Some(TokenType::Identifier(_)), Some(TokenType::LeftParen))
                        );

                        if has_paren {
                            self.advance(); // consume '::'
                            if let TokenType::Identifier(method_name) = self.peek().kind.clone() {
                                self.advance(); // consume method name
                                call_name = format!("{}::{}", call_name, method_name);
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
                        Expr::FunctionCall(FunctionCallExpr {
                            name: call_name,
                            args,
                            span: Span::default(),
                        })
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
                            while !self.check(&TokenType::RightBrace)
                                && !self.check(&TokenType::Eof)
                            {
                                let token_kind = self.advance().kind.clone();
                                let f_name = match token_kind {
                                    TokenType::Identifier(f) => f,
                                    _ => {
                                        return Err(format!(
                                            "Expected field name in struct init, found {:?}",
                                            token_kind
                                        ))
                                    }
                                };
                                self.consume(&TokenType::Colon, "Expected ':'")?;
                                let f_expr = self.parse_expr()?;
                                fields.push((f_name, f_expr));
                                if !self.match_token(&TokenType::Comma) {
                                    break;
                                }
                            }
                            self.consume(&TokenType::RightBrace, "Expected '}'")?;
                            Expr::StructInit(StructInitExpr {
                                name: call_name,
                                fields,
                                span: Span::default(),
                            })
                        } else if self.match_token(&TokenType::DoubleColon) {
                            let variant = match self.advance().kind.clone() {
                                TokenType::Identifier(v) => v,
                                _ => return Err("Expected enum variant after ::".to_string()),
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
                                self.consume(
                                    &TokenType::RightParen,
                                    "Expected ')' after enum payload",
                                )?;
                                payload = Some(args);
                            }
                            Expr::EnumVariant(EnumVariantExpr {
                                enum_name: call_name,
                                variant_name: variant,
                                payload,
                                span: Span::default(),
                            })
                        } else {
                            Expr::Identifier(IdentifierExpr {
                                name: call_name,
                                span: Span::default(),
                            })
                        }
                    } else if self.match_token(&TokenType::DoubleColon) {
                        let variant = match self.advance().kind.clone() {
                            TokenType::Identifier(v) => v,
                            _ => return Err("Expected enum variant after ::".to_string()),
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
                            self.consume(
                                &TokenType::RightParen,
                                "Expected ')' after enum payload",
                            )?;
                            payload = Some(args);
                        }
                        Expr::EnumVariant(EnumVariantExpr {
                            enum_name: call_name,
                            variant_name: variant,
                            payload,
                            span: Span::default(),
                        })
                    } else {
                        Expr::Identifier(IdentifierExpr {
                            name: call_name,
                            span: Span::default(),
                        })
                    }
                }
                TokenType::Number(s) => {
                    let mut num_str = s.clone();
                    let mut suffix_str = String::new();

                    if let Some(idx) = s.find(|c: char| c.is_alphabetic() || c == '_') {
                        num_str = s[..idx].to_string();
                        suffix_str = s[idx..].to_string();
                    }

                    let el_ty = if !suffix_str.is_empty() {
                        match suffix_str.as_str() {
                            "f16" => Some(crate::ast::ElementType::F16),
                            "f32" => Some(crate::ast::ElementType::F32),
                            "f64" => Some(crate::ast::ElementType::F64),
                            "bf16" => Some(crate::ast::ElementType::BF16),
                            "i8" => Some(crate::ast::ElementType::I8),
                            "i16" => Some(crate::ast::ElementType::I16),
                            "i32" => Some(crate::ast::ElementType::I32),
                            "i64" => Some(crate::ast::ElementType::I64),
                            "i128" => Some(crate::ast::ElementType::I128),
                            "u8" => Some(crate::ast::ElementType::U8),
                            "u16" => Some(crate::ast::ElementType::U16),
                            "u32" => Some(crate::ast::ElementType::U32),
                            "u64" => Some(crate::ast::ElementType::U64),
                            "u128" => Some(crate::ast::ElementType::U128),
                            _ => return Err(format!("Unknown number suffix '{}'", suffix_str)),
                        }
                    } else {
                        // Rust-like defaults: i32 for integers, f32 for floats in ML context
                        if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                            Some(crate::ast::ElementType::F32)
                        } else {
                            Some(crate::ast::ElementType::I32)
                        }
                    };

                    Expr::Number(NumberExpr {
                        value: num_str,
                        ty: el_ty,
                        span: Span::default(),
                    })
                }
                TokenType::StringLiteral(s) => Expr::StringLiteral(StringLiteralExpr {
                    value: s,
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
                    Expr::SpawnOn(crate::ast::SpawnOnExpr {
                        top,
                        stmts,
                        ret: ret_expr,
                        span: Span::default(),
                    })
                }
                _ => return Err(format!("Expected expression, found {:?}", token.kind)),
            }
        };

        // Postfix operators: .member, .method(), [index]
        loop {
            if self.match_token(&TokenType::Dot) {
                let ident = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err("Expected identifier after '.'".to_string()),
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
                        method_name: ident,
                        args,
                        span: Span::default(),
                    });
                } else {
                    expr = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(expr),
                        member: ident,
                        struct_name: None,
                        span: Span::default(),
                    });
                }
            } else if self.match_token(&TokenType::LeftBracket) {
                let index = self.parse_expr()?;
                self.consume(&TokenType::RightBracket, "Expected ']'")?;
                expr = Expr::IndexAccess(IndexAccessExpr {
                    base: Box::new(expr),
                    index: Box::new(index),
                    span: Span::default(),
                });
            } else {
                break;
            }
        }

        Ok(expr)
    }
}
