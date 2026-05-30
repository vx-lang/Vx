use super::*;

impl<'a> Parser<'a> {
    pub(crate) fn parse_statement(&mut self) -> Result<Statement, String> {
        let token = self.peek().clone();
        match token.kind {
            TokenType::Let => {
                self.advance();
                let mut is_mut = false;
                if self.match_token(&TokenType::Mut) {
                    is_mut = true;
                }
                let name = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err("Expected identifier after let".to_string()),
                };
                let mut type_annotation = None;
                if self.match_token(&TokenType::Colon) {
                    type_annotation = Some(self.parse_type()?);
                }
                self.consume(&TokenType::Equals, "Expected '='")?;
                let expr = self.parse_expr()?;
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::LetDecl(LetDeclStmt {
                    name,
                    is_mut,
                    ty_ann: type_annotation,
                    expr,
                    span: Span::default(),
                }))
            }
            TokenType::Comptime => {
                self.advance();
                self.consume(&TokenType::LeftBrace, "Expected '{' after 'comptime'")?;
                let mut stmts = Vec::new();
                while self.peek().kind != TokenType::RightBrace
                    && self.peek().kind != TokenType::Eof
                {
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
                self.consume(&TokenType::RightBrace, "Expected '}' after comptime block")?;
                Ok(Statement::ExprStmt(ExprStmtStmt {
                    expr: Expr::ComptimeBlock(ComptimeBlockExpr {
                        stmts,
                        ret,
                        span: Span::default(),
                    }),
                    has_semi: true,
                    span: Span::default(),
                }))
            }
            TokenType::Assert => {
                self.advance();
                self.consume(&TokenType::LeftParen, "Expected '(' after 'assert'")?;
                let expr = self.parse_expr()?;
                let mut msg = None;
                if self.match_token(&TokenType::Comma) {
                    if let TokenType::StringLiteral(s) = self.peek().kind.clone() {
                        msg = Some(s);
                        self.advance();
                    } else {
                        return Err(
                            "Expected string literal message after comma in assert".to_string()
                        );
                    }
                }
                self.consume(
                    &TokenType::RightParen,
                    "Expected ')' after assert condition",
                )?;
                self.consume(&TokenType::Semicolon, "Expected ';' after assert statement")?;
                Ok(Statement::Assert(AssertStmt {
                    expr: Box::new(expr),
                    msg,
                    span: Span::default(),
                }))
            }
            TokenType::Return => {
                self.advance();
                let expr = self.parse_expr()?;
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Return(ReturnStmt {
                    expr,
                    span: Span::default(),
                }))
            }

            TokenType::For => {
                self.advance();
                let iter = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err("Expected identifier after 'for'".to_string()),
                };
                self.consume(&TokenType::In, "Expected 'in' after for iterator")?;
                let start = self.parse_expr()?;
                self.consume(&TokenType::DoubleDot, "Expected '..' in range")?;
                let end = self.parse_expr()?;
                self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                let mut stmts = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    stmts.push(self.parse_statement()?);
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Ok(Statement::ForLoop(ForLoopStmt {
                    iter,
                    start: Box::new(start),
                    end: Box::new(end),
                    body: stmts,
                    span: Span::default(),
                }))
            }
            _ => {
                let expr = self.parse_expr()?;
                if self.match_token(&TokenType::Equals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::Assign(AssignStmt {
                        lhs: expr,
                        rhs,
                        span: Span::default(),
                    }))
                } else if self.match_token(&TokenType::PlusEquals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::CompoundAssign(CompoundAssignStmt {
                        lhs: expr,
                        op: BinaryOp::Add,
                        rhs,
                        span: Span::default(),
                    }))
                } else {
                    let mut has_semicolon = true;
                    match &expr {
                        Expr::UnsafeBlock(UnsafeBlockExpr { .. })
                        | Expr::ComptimeBlock(ComptimeBlockExpr { .. })
                        | Expr::SpawnOn(crate::ast::SpawnOnExpr { .. })
                        | Expr::If(IfExpr { .. }) => {
                            has_semicolon = self.match_token(&TokenType::Semicolon);
                        }
                        _ => {
                            if self.check(&TokenType::RightBrace) {
                                has_semicolon = false; // Optional at end of block
                            } else {
                                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                            }
                        }
                    }
                    Ok(Statement::ExprStmt(ExprStmtStmt {
                        expr,
                        has_semi: has_semicolon,
                        span: Span::default(),
                    }))
                }
            }
        }
    }
}
