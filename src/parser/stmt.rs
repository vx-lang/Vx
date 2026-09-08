//===- stmt.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx statements, including let bindings, loops, and return statements.
//
//===----------------------------------------------------------------------===//

use super::*;

impl<'a> Parser<'a> {
    fn parse_expr_or_assign_stmt(&mut self, expr: Expr) -> ParseResult<'a, Statement> {
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
                | Expr::SpawnOn(SpawnOnExpr { .. })
                | Expr::If(IfExpr { .. })
                | Expr::Match(MatchExpr { .. }) => {
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

    pub(crate) fn parse_statement(&mut self) -> ParseResult<'a, Statement> {
        let token = self.peek();
        let token_line = token.line;
        let token_col = token.column;
        let token_len = token.length;

        match &token.kind {
            TokenType::Let => {
                self.advance();
                let mut is_mut = false;
                if self.match_token(&TokenType::Mut) {
                    is_mut = true;
                }
                let name = match &self.advance().kind {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => return Err(self.error("Expected identifier after let")),
                };
                let mut type_annotation = None;
                if self.match_token(&TokenType::Colon) {
                    type_annotation = Some(self.parse_type()?);
                }
                self.consume(&TokenType::Equals, "Expected '='")?;
                let expr = self.parse_expr()?;
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::LetDecl(LetDeclStmt {
                    // The `let` keyword's own token: a diagnostic about the binding (a type
                    // mismatch, an unused variable) points at the declaration, not at 0:0.
                    name: name.into(),
                    is_mut,
                    ty_ann: type_annotation,
                    expr,
                    span: Span {
                        line: token_line,
                        column: token_col,
                        length: token_len,
                    },
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
                if let Some(Statement::ExprStmt(stmt)) = stmts.last() {
                    if !stmt.has_semi {
                        let last = stmts.pop().unwrap();
                        if let Statement::ExprStmt(expr_stmt) = last {
                            ret = Some(Box::new(expr_stmt.expr));
                        }
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
                    if let TokenType::StringLiteral(s) = &self.peek().kind {
                        msg = Some(s.to_string());
                        self.advance();
                    } else {
                        return Err(
                            self.error("Expected string literal message after comma in assert")
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
                // `return;` -- an early exit from a `void` function. Without this the parser
                // demanded an expression and reported `Expected expression, found Semicolon`,
                // so a void function had no way to return before its last statement.
                let expr = if self.check(&TokenType::Semicolon) {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Return(ReturnStmt {
                    expr,
                    span: Span {
                        line: token_line,
                        column: token_col,
                        length: token_len,
                    },
                }))
            }
            TokenType::Loop => {
                self.advance();
                let mut invariants = Vec::new();
                while self.match_token(&TokenType::Invariant) {
                    self.consume(&TokenType::LeftParen, "Expected '(' after 'invariant'")?;
                    invariants.push(self.parse_expr()?);
                    self.consume(
                        &TokenType::RightParen,
                        "Expected ')' after invariant expression",
                    )?;
                }
                self.consume(&TokenType::LeftBrace, "Expected '{' after loop")?;
                let mut body = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    body.push(self.parse_statement()?);
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Ok(Statement::Loop(LoopStmt {
                    invariants,
                    body,
                    span: Span::default(),
                }))
            }
            TokenType::Break => {
                self.advance();
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Break(BreakStmt {
                    span: Span::default(),
                }))
            }
            TokenType::Continue => {
                self.advance();
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Continue(ContinueStmt {
                    span: Span::default(),
                }))
            }
            TokenType::For => {
                self.advance();
                let iter = match &self.advance().kind {
                    TokenType::Identifier(s) => s.to_string(),
                    _ => return Err(self.error("Expected identifier after 'for'")),
                };
                self.consume(&TokenType::In, "Expected 'in' after for iterator")?;
                let iterable = self.parse_expr()?;
                let mut invariants = Vec::new();
                while self.match_token(&TokenType::Invariant) {
                    self.consume(&TokenType::LeftParen, "Expected '(' after 'invariant'")?;
                    invariants.push(self.parse_expr()?);
                    self.consume(
                        &TokenType::RightParen,
                        "Expected ')' after invariant expression",
                    )?;
                }
                self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                let mut stmts = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    stmts.push(self.parse_statement()?);
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Ok(Statement::ForLoop(ForLoopStmt {
                    iter,
                    iterable: Box::new(iterable),
                    invariants,
                    body: stmts,
                    span: Span::default(),
                }))
            }
            TokenType::Identifier(s) => {
                let ident_line = token_line;
                let ident_end_col = token_col + token_len;
                let next_token = self.peek_n(1);

                let mut is_macro = false;
                if next_token.kind == TokenType::Bang {
                    if next_token.line == ident_line && next_token.column == ident_end_col {
                        is_macro = true;
                    } else {
                        return Err(self.error(&format!("Macro invocations must not have spaces between the macro name and '!'. Did you mean `{}!`?", s)));
                    }
                }

                if is_macro {
                    let name = s.to_string();
                    self.advance(); // consume identifier
                    self.advance(); // consume '!'
                    let token_tree = self.parse_token_tree()?;
                    let mut block_tree = None;
                    if self.check(&TokenType::LeftBrace) {
                        block_tree = Some(self.parse_token_tree()?);
                    }
                    let has_semi = self.match_token(&TokenType::Semicolon); // optional semicolon for statement macros
                    return Ok(Statement::MacroCall(MacroCallStmt {
                        name: name.into(),
                        token_tree,
                        block_tree,
                        has_semi,
                        span: Span::default(),
                    }));
                }
                // fallback to expression parsing
                let expr = self.parse_expr()?;
                self.parse_expr_or_assign_stmt(expr)
            }
            _ => {
                let expr = self.parse_expr()?;
                self.parse_expr_or_assign_stmt(expr)
            }
        }
    }
}
