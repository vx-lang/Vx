use super::*;

pub(crate) fn infer_number_literal(
    s: &str,
) -> Result<(String, Option<crate::ast::ElementType>), String> {
    let mut num_str = s.to_string();
    let mut suffix_str = String::new();

    if let Some(idx) = s.find(|c: char| c.is_alphabetic() || c == '_') {
        num_str = s[..idx].to_string();
        suffix_str = s[idx..].to_string();
    }

    let el_ty = if !suffix_str.is_empty() {
        match suffix_str.parse::<crate::ast::ElementType>() {
            Ok(el) => Some(el),
            Err(e) => return Err(e),
        }
    } else {
        if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
            if let Ok(f32_val) = num_str.parse::<f32>() {
                if f32_val.is_infinite() {
                    if let Ok(f64_val) = num_str.parse::<f64>() {
                        if f64_val.is_finite() {
                            return Ok((num_str, Some(crate::ast::ElementType::F64)));
                        }
                    }
                }
            }
            Some(crate::ast::ElementType::F32)
        } else {
            if num_str.parse::<i32>().is_err() {
                if num_str.parse::<i64>().is_ok() {
                    Some(crate::ast::ElementType::I64)
                } else {
                    Some(crate::ast::ElementType::I128)
                }
            } else {
                Some(crate::ast::ElementType::I32)
            }
        }
    };
    Ok((num_str, el_ty))
}
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
                let mut enum_name = s.clone();
                self.advance();
                if self.check(&TokenType::LeftAngle) {
                    self.advance(); // consume '<'
                    let ty_ident = match &self.advance().kind {
                        TokenType::Identifier(s) => s.clone(),
                        _ => return Err("Expected type identifier in generic pattern".to_string()),
                    };
                    self.consume(&TokenType::RightAngle, "Expected '>' in generic pattern")?;
                    enum_name = format!("{}<{}>", enum_name, ty_ident);
                }
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
                    Ok(Pattern::EnumVariant(enum_name, variant_name, payload))
                } else {
                    Ok(Pattern::Identifier(enum_name))
                }
            }
            _ => Err(format!("Unexpected token in pattern: {:?}", token.kind)),
        }
    }

    pub(crate) fn parse_identifier_expr(&mut self, mut call_name: String) -> Result<Expr, String> {
        if matches!(self.peek().kind, TokenType::LeftAngle) {
            let mut is_generic = true;
            let mut type_args = Vec::new();
            let mut j = 1;
            while !matches!(self.peek_n(j).kind, TokenType::RightAngle)
                && !matches!(self.peek_n(j).kind, TokenType::Eof)
            {
                if let TokenType::Identifier(ref s) = self.peek_n(j).kind {
                    type_args.push(s.clone());
                    j += 1;
                    if matches!(self.peek_n(j).kind, TokenType::Comma) {
                        j += 1;
                    }
                } else {
                    is_generic = false;
                    break;
                }
            }
            if is_generic
                && matches!(self.peek_n(j).kind, TokenType::RightAngle)
                && !type_args.is_empty()
            {
                self.advance(); // consume '<'
                for _ in 0..j {
                    self.advance();
                }
                let ty_args_str = type_args.join(", ");
                if call_name == "Tensor" {
                    call_name = format!("Tensor_{}", type_args[0]);
                } else {
                    call_name = format!("{}<{}>", call_name, ty_args_str);
                }
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
                self.advance(); // consume '::'
                if let TokenType::Identifier(method_name) = self.peek().kind.clone() {
                    self.advance(); // consume method name
                    call_name = format!("{}::{}", call_name, method_name);

                    if matches!(self.peek().kind, TokenType::LeftAngle) {
                        let mut is_generic = true;
                        let mut type_args = Vec::new();
                        let mut j = 1;
                        while !matches!(self.peek_n(j).kind, TokenType::RightAngle)
                            && !matches!(self.peek_n(j).kind, TokenType::Eof)
                        {
                            if let TokenType::Identifier(ref s) = self.peek_n(j).kind {
                                type_args.push(s.clone());
                                j += 1;
                                if matches!(self.peek_n(j).kind, TokenType::Comma) {
                                    j += 1;
                                }
                            } else {
                                is_generic = false;
                                break;
                            }
                        }
                        if is_generic
                            && matches!(self.peek_n(j).kind, TokenType::RightAngle)
                            && !type_args.is_empty()
                        {
                            self.advance(); // consume '<'
                            for _ in 0..j {
                                self.advance();
                            }
                            let ty_args_str = type_args.join(", ");
                            call_name = format!("{}<{}>", call_name, ty_args_str);
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
                name: call_name,
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
                Ok(Expr::StructInit(StructInitExpr {
                    name: call_name,
                    fields,
                    span: Span::default(),
                }))
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
                    self.consume(&TokenType::RightParen, "Expected ')' after enum payload")?;
                    payload = Some(args);
                }
                Ok(Expr::EnumVariant(EnumVariantExpr {
                    enum_name: call_name,
                    variant_name: variant,
                    payload,
                    span: Span::default(),
                }))
            } else {
                Ok(Expr::Identifier(IdentifierExpr {
                    name: call_name,
                    span: Span::default(),
                }))
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
                self.consume(&TokenType::RightParen, "Expected ')' after enum payload")?;
                payload = Some(args);
            }
            Ok(Expr::EnumVariant(EnumVariantExpr {
                enum_name: call_name,
                variant_name: variant,
                payload,
                span: Span::default(),
            }))
        } else {
            Ok(Expr::Identifier(IdentifierExpr {
                name: call_name,
                span: Span::default(),
            }))
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
                    name: "Verified".to_string(),
                    args: vec![inner],
                    span: Span::default(),
                })
            }
            TokenType::Grad => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'grad'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => {
                        return Err(
                            "Expected function identifier as first argument to grad".to_string()
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
                    target_fn,
                    args,
                    span: Span::default(),
                })
            }
            TokenType::Vjp => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'vjp'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => {
                        return Err(
                            "Expected function identifier as first argument to vjp".to_string()
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
                    return Err("Expected cotangent argument for vjp".to_string());
                }
                let cotangent = all_args.pop().unwrap();
                Expr::Vjp(VjpExpr {
                    target_fn,
                    args: all_args,
                    cotangent: Box::new(cotangent),
                    span: Span::default(),
                })
            }
            TokenType::Jvp => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'jvp'")?;
                let target_fn = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => {
                        return Err(
                            "Expected function identifier as first argument to jvp".to_string()
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
                    return Err("Expected tangent argument for jvp".to_string());
                }
                let tangent = all_args.pop().unwrap();
                Expr::Jvp(JvpExpr {
                    target_fn,
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
                        if self.match_token(&TokenType::Bang) {
                            let token_tree = self.parse_token_tree()?;
                            Expr::MacroCall(MacroCallExpr {
                                name: s,
                                token_tree,
                                span: Span::default(),
                            })
                        } else {
                            self.parse_identifier_expr(s)?
                        }
                    }
                    TokenType::Return => self.parse_identifier_expr("return".to_string())?,
                    TokenType::Number(s) => {
                        let (num_str, el_ty) = infer_number_literal(&s)?;

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
                    TokenType::Pipe | TokenType::OrOr => {
                        let mut params = Vec::new();
                        if token.kind == TokenType::Pipe {
                            if !self.check(&TokenType::Pipe) {
                                loop {
                                    let name = match self.advance().kind.clone() {
                                        TokenType::Identifier(s) => s,
                                        _ => {
                                            return Err(
                                                "Expected identifier in closure params".to_string()
                                            )
                                        }
                                    };
                                    let ty = if self.match_token(&TokenType::Colon) {
                                        self.parse_type()?
                                    } else {
                                        Type::Unknown
                                    };
                                    params.push((name, ty));
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
                        Expr::SpawnOn(crate::ast::SpawnOnExpr {
                            top,
                            stmts,
                            ret: ret_expr,
                            span: Span::default(),
                        })
                    }
                    _ => return Err(format!("Expected expression, found {:?}", token.kind)),
                }
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
        let mut parser = Parser::new(tokens, input);
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
            assert_eq!(enum_name, "Option");
            assert_eq!(variant_name, "None");
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
        }) = expr
        {
            assert_eq!(name, "Option::Some");
            assert_eq!(args.len(), 1);
            if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &args[0] {
                assert_eq!(name, "x");
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
        }) = expr
        {
            assert_eq!(name, "Option<i32>::Some");
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
        }) = expr
        {
            assert_eq!(name, "Vec<i32>::new");
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
        }) = expr
        {
            if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &*base {
                assert_eq!(name, "vec");
            } else {
                panic!("Expected Identifier base");
            }
            assert_eq!(method_name, "push");
            assert_eq!(args.len(), 1);
            if let Expr::Number(NumberExpr { value, .. }) = &args[0] {
                assert_eq!(value, "10");
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
        }) = expr
        {
            assert_eq!(method_name, "map");
            assert_eq!(args.len(), 1);
            if let Expr::MethodCall(MethodCallExpr {
                base: inner_base,
                method_name: inner_method_name,
                args: inner_args,
                span: _,
            }) = &*base
            {
                assert_eq!(inner_method_name, "iter");
                assert_eq!(inner_args.len(), 0);
                if let Expr::Identifier(IdentifierExpr { name, span: _ }) = &**inner_base {
                    assert_eq!(name, "vec");
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
}
