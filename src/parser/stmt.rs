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
    /// The compound assignment at the cursor, consumed, or `None`. Each one means the
    /// binary operator of the same name applied to the two sides, so the statement they
    /// build is checked and lowered by the same rules as `a = a OP b` -- there is no
    /// second implementation of what `%` or `>>` does.
    fn compound_assign_op(&mut self) -> Option<BinaryOp> {
        // Two tokens, so it has to be tried before the single-token spellings: `>>=`
        // begins with a `>` that nothing else here would claim, but `>=` would.
        if let Some(op) = self.shift_assign_at() {
            self.advance();
            self.advance();
            return Some(op);
        }
        let op = match self.peek().kind {
            TokenType::PlusEquals => BinaryOp::Add,
            TokenType::MinusEquals => BinaryOp::Sub,
            TokenType::StarEquals => BinaryOp::Mul,
            TokenType::SlashEquals => BinaryOp::Div,
            TokenType::PercentEquals => BinaryOp::Rem,
            TokenType::AmpersandEquals => BinaryOp::BitAnd,
            TokenType::PipeEquals => BinaryOp::BitOr,
            TokenType::CaretEquals => BinaryOp::BitXor,
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    fn parse_expr_or_assign_stmt(&mut self, expr: Expr) -> ParseResult<'a, Statement> {
        if self.match_token(&TokenType::Equals) {
            let rhs = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::Assign(AssignStmt {
                lhs: expr,
                rhs,
                span: Span::default(),
            }))
        } else if let Some(op) = self.compound_assign_op() {
            let rhs = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::CompoundAssign(CompoundAssignStmt {
                lhs: expr,
                op,
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
                if !is_mut && self.check(&TokenType::LeftParen) {
                    return self.parse_tuple_let(Span {
                        line: token_line,
                        column: token_col,
                        length: token_len,
                    });
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
                    self.parse_statement_into(&mut stmts)?;
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
                    invariants.push(self.parse_expr()?);
                }
                self.consume(&TokenType::LeftBrace, "Expected '{' after loop")?;
                let mut body = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    self.parse_statement_into(&mut body)?;
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
                    invariants.push(self.parse_expr()?);
                }
                self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                let mut stmts = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    self.parse_statement_into(&mut stmts)?;
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Ok(Statement::ForLoop(ForLoopStmt {
                    iter,
                    iterable: Box::new(iterable),
                    invariants,
                    body: stmts,
                    span: Span::default(),
                    next_fn: None,
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

    /// `let (a, mut b, _) = e;`: a `let` of `e` under a name no source can spell, then one `let`
    /// per name, of the element in its place. Nested tuples recurse the same way.
    ///
    /// The first `let` is returned and the rest wait in `pending_stmts` for
    /// `parse_statement_into`, since a statement parses to one statement.
    fn parse_tuple_let(&mut self, span: Span) -> ParseResult<'a, Statement> {
        let pattern = self.parse_tuple_pattern()?;
        let mut type_annotation = None;
        if self.match_token(&TokenType::Colon) {
            type_annotation = Some(self.parse_type()?);
        }
        self.consume(&TokenType::Equals, "Expected '='")?;
        let expr = self.parse_expr()?;
        self.consume(&TokenType::Semicolon, "Expected ';'")?;
        let mut out = Vec::new();
        self.lower_tuple_pattern(pattern, type_annotation, expr, span, &mut out);
        let first = out.remove(0);
        self.pending_stmts.extend(out);
        Ok(first)
    }

    fn parse_tuple_pattern(&mut self) -> ParseResult<'a, TuplePattern> {
        self.consume(&TokenType::LeftParen, "Expected '('")?;
        let mut elems = Vec::new();
        while !self.check(&TokenType::RightParen) {
            if self.check(&TokenType::LeftParen) {
                elems.push(self.parse_tuple_pattern()?);
            } else {
                let is_mut = self.match_token(&TokenType::Mut);
                let name = self.expect_identifier("Expected a name in a tuple pattern")?;
                elems.push(if name == "_" && !is_mut {
                    TuplePattern::Ignore
                } else {
                    TuplePattern::Bind(name, is_mut)
                });
            }
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightParen, "Expected ')' after a tuple pattern")?;
        self.tuple_struct(elems.len())?;
        Ok(TuplePattern::Tuple(elems))
    }

    fn lower_tuple_pattern(
        &mut self,
        pattern: TuplePattern,
        ty_ann: Option<crate::syntax::Type>,
        expr: Expr,
        span: Span,
        out: &mut Vec<Statement>,
    ) {
        match pattern {
            TuplePattern::Ignore => {}
            TuplePattern::Bind(name, is_mut) => out.push(Statement::LetDecl(LetDeclStmt {
                name: name.into(),
                is_mut,
                ty_ann,
                expr,
                span,
            })),
            TuplePattern::Tuple(elems) => {
                let temp = format!("$tuple{}", self.tuple_lets);
                self.tuple_lets += 1;
                out.push(Statement::LetDecl(LetDeclStmt {
                    name: temp.clone().into(),
                    is_mut: false,
                    ty_ann,
                    expr,
                    span,
                }));
                for (i, elem) in elems.into_iter().enumerate() {
                    let element = Expr::MemberAccess(MemberAccessExpr {
                        base: Box::new(Expr::Identifier(IdentifierExpr {
                            name: temp.clone().into(),
                            span,
                        })),
                        member: format!("_{i}").into(),
                        struct_name: None,
                        span,
                    });
                    self.lower_tuple_pattern(elem, None, element, span, out);
                }
            }
        }
    }
}

/// The left of a tuple `let`.
enum TuplePattern {
    /// `_`: the element is not bound.
    Ignore,
    /// A name, and whether it was written `mut`.
    Bind(String, bool),
    Tuple(Vec<TuplePattern>),
}
