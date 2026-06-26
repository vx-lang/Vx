//===- macro_expand.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// AST transformations for expanding macros into inline code during compilation.
//
//===----------------------------------------------------------------------===//

use super::*;
use crate::syntax::{Delimiter, Span};
use crate::lexer::OwnedTokenType;
use std::collections::HashMap;

pub struct MacroExpander<'a> {
    pub macros: &'a HashMap<crate::symbol::Symbol, Vec<MacroRule>>,
}

fn take_expr(expr: &mut expr::Expr) -> expr::Expr {
    std::mem::replace(
        expr,
        expr::Expr::Number(expr::NumberExpr::new(
            "0".into(),
            None,
            crate::syntax::Span::default(),
        )),
    )
}
impl<'a> MacroExpander<'a> {
    pub fn new(macros: &'a HashMap<crate::symbol::Symbol, Vec<MacroRule>>) -> Self {
        Self { macros }
    }

    fn parse_expanded_expr(
        &self,
        tokens: &[crate::lexer::OwnedToken],
    ) -> Result<expr::Expr, String> {
        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");
        parser.parse_expr().map_err(|e| format!("{:?}", e))
    }

    fn parse_expanded_exprs(
        &self,
        tokens: &[crate::lexer::OwnedToken],
    ) -> Result<Vec<expr::Expr>, String> {
        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");
        let mut exprs = Vec::new();
        while !parser.check(&crate::lexer::TokenType::Eof) {
            exprs.push(parser.parse_expr().map_err(|e| format!("{:?}", e))?);
            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                break;
            }
        }
        Ok(exprs)
    }

    pub fn expand_module(&mut self, module: &mut VxModule) -> Result<(), String> {
        // Expand top level decls
        for func in &mut module.functions {
            self.expand_function(func)?;
        }
        for impl_block in &mut module.impls {
            for func in &mut impl_block.methods {
                self.expand_function(func)?;
            }
        }
        Ok(())
    }

    fn expand_function(&mut self, func: &mut Function) -> Result<(), String> {
        let body = std::mem::take(&mut func.body);
        let mut new_body = Vec::with_capacity(body.len());
        for stmt in body {
            new_body.extend(self.expand_stmt(stmt)?);
        }
        func.body = new_body;
        Ok(())
    }

    fn expand_stmt(&mut self, mut stmt: stmt::Statement) -> Result<Vec<stmt::Statement>, String> {
        if let stmt::Statement::MacroCall(call) = stmt {
            let expanded_expr =
                self.expand_macro_call(&call.name, &call.token_tree, &call.block_tree)?;
            let recursively_expanded = self.expand_expr(expanded_expr)?;
            return Ok(vec![stmt::Statement::ExprStmt(stmt::ExprStmtStmt {
                expr: recursively_expanded,
                has_semi: call.has_semi,
                span: call.span,
            })]);
        }
        // Recurse into children
        self.expand_stmt_children(&mut stmt)?;
        Ok(vec![stmt])
    }

    fn expand_stmt_children(&mut self, stmt: &mut stmt::Statement) -> Result<(), String> {
        match stmt {
            stmt::Statement::ExprStmt(e) => {
                e.expr = self.expand_expr(take_expr(&mut e.expr))?;
            }
            stmt::Statement::LetDecl(l) => {
                l.expr = self.expand_expr(take_expr(&mut l.expr))?;
            }
            stmt::Statement::Assign(a) => {
                a.lhs = self.expand_expr(take_expr(&mut a.lhs))?;
                a.rhs = self.expand_expr(take_expr(&mut a.rhs))?;
            }
            stmt::Statement::CompoundAssign(a) => {
                a.lhs = self.expand_expr(take_expr(&mut a.lhs))?;
                a.rhs = self.expand_expr(take_expr(&mut a.rhs))?;
            }
            stmt::Statement::Return(r) => {
                r.expr = self.expand_expr(take_expr(&mut r.expr))?;
            }
            stmt::Statement::Assert(a) => {
                *a.expr = self.expand_expr(take_expr(&mut a.expr))?;
            }
            stmt::Statement::Loop(l) => {
                let body = std::mem::take(&mut l.body);
                let mut new_body = Vec::with_capacity(body.len());
                for s in body {
                    new_body.extend(self.expand_stmt(s)?);
                }
                l.body = new_body;
            }
            stmt::Statement::ForLoop(f) => {
                *f.iterable = self.expand_expr(take_expr(&mut f.iterable))?;
                let body = std::mem::take(&mut f.body);
                let mut new_body = Vec::with_capacity(body.len());
                for s in body {
                    new_body.extend(self.expand_stmt(s)?);
                }
                f.body = new_body;
            }
            _ => {}
        }
        Ok(())
    }

    fn expand_expr(&mut self, mut expr: expr::Expr) -> Result<expr::Expr, String> {
        if let expr::Expr::MacroCall(call) = expr {
            let expanded =
                self.expand_macro_call(&call.name, &call.token_tree, &call.block_tree)?;
            return self.expand_expr(expanded);
        }
        // Traverse and expand
        match &mut expr {
            expr::Expr::BinaryOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::RelationalOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::LogicalOp(b) => {
                *b.lhs = self.expand_expr(take_expr(&mut b.lhs))?;
                *b.rhs = self.expand_expr(take_expr(&mut b.rhs))?;
            }
            expr::Expr::Range(b) => {
                *b.start = self.expand_expr(take_expr(&mut b.start))?;
                *b.end = self.expand_expr(take_expr(&mut b.end))?;
            }
            expr::Expr::UnaryOp(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::Borrow(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::Dereference(u) => {
                *u.expr = self.expand_expr(take_expr(&mut u.expr))?;
            }
            expr::Expr::MemberAccess(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
            }
            expr::Expr::IndexAccess(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
                *m.index = self.expand_expr(take_expr(&mut m.index))?;
            }
            expr::Expr::Match(m) => {
                *m.expr = self.expand_expr(take_expr(&mut m.expr))?;
                for arm in &mut m.arms {
                    let body = std::mem::take(&mut arm.body);
                    let mut new_body = Vec::with_capacity(body.len());
                    for s in body {
                        new_body.extend(self.expand_stmt(s)?);
                    }
                    arm.body = new_body;
                }
            }
            expr::Expr::FunctionCall(f) => {
                for arg in &mut f.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::MethodCall(m) => {
                *m.base = self.expand_expr(take_expr(&mut m.base))?;
                for arg in &mut m.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::StructInit(s) => {
                for (_, e) in &mut s.fields {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::Array(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::VecMacro(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(take_expr(e))?;
                }
            }
            expr::Expr::If(i) => {
                *i.cond = self.expand_expr(take_expr(&mut i.cond))?;
                let body = std::mem::take(&mut i.then_block);
                let mut new_body = Vec::with_capacity(body.len());
                for s in body {
                    new_body.extend(self.expand_stmt(s)?);
                }
                i.then_block = new_body;
                if let Some(else_b) = &mut i.else_block {
                    let body = std::mem::take(else_b);
                    let mut new_body = Vec::with_capacity(body.len());
                    for s in body {
                        new_body.extend(self.expand_stmt(s)?);
                    }
                    *else_b = new_body;
                }
            }
            expr::Expr::UnsafeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                let body = std::mem::take(&mut u.stmts);
                let mut new_body = Vec::with_capacity(body.len());
                for s in body {
                    new_body.extend(self.expand_stmt(s)?);
                }
                u.stmts = new_body;
            }
            expr::Expr::ComptimeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                let body = std::mem::take(&mut u.stmts);
                let mut new_body = Vec::with_capacity(body.len());
                for s in body {
                    new_body.extend(self.expand_stmt(s)?);
                }
                u.stmts = new_body;
            }
            expr::Expr::Closure(c) => {
                *c.body = self.expand_expr(take_expr(&mut c.body))?;
            }
            expr::Expr::SpawnOn(s) => {
                if let Some(ret) = &mut s.ret {
                    **ret = self.expand_expr(take_expr(ret))?;
                }
                let body = std::mem::take(&mut s.stmts);
                let mut new_body = Vec::with_capacity(body.len());
                for st in body {
                    new_body.extend(self.expand_stmt(st)?);
                }
                s.stmts = new_body;
            }
            expr::Expr::Grad(g) => {
                for arg in &mut g.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
            }
            expr::Expr::Vjp(v) => {
                for arg in &mut v.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
                *v.cotangent = self.expand_expr(take_expr(&mut v.cotangent))?;
            }
            expr::Expr::Jvp(j_expr) => {
                for arg in &mut j_expr.args {
                    *arg = self.expand_expr(take_expr(arg))?;
                }
                *j_expr.tangent = self.expand_expr(take_expr(&mut j_expr.tangent))?;
            }
            _ => {}
        }
        Ok(expr)
    }

    fn expand_macro_call(
        &mut self,
        name: &str,
        tt: &TokenTree,
        block_tree: &Option<TokenTree>,
    ) -> Result<expr::Expr, String> {
        println!("Expanding macro call: {}!", name);

        if name == "mlir" {
            return self.expand_mlir_macro(tt, block_tree);
        }

        if name == "vec" {
            return self.expand_vec_macro(tt);
        }
        if name == "print" {
            return self.expand_print_macro(tt);
        }
        if name == "println" {
            return self.expand_println_macro(tt);
        }

        let rules = self
            .macros
            .get(name)
            .ok_or_else(|| format!("Macro {} not found", name))?;

        let input_tokens = match tt {
            TokenTree::Delimited(_, inner) => {
                let mut tokens = Vec::new();
                for i in inner {
                    tokens.extend(self.flatten_tt(i));
                }
                tokens
            }
            _ => self.flatten_tt(tt),
        };

        for rule in rules {
            if let Ok(captures) = self.match_rule(&rule.matcher, &input_tokens) {
                let mut transcribed = self.transcribe(&rule.transcriber, &captures)?;
                // IMPORTANT: The parser expects an EOF token at the end!
                transcribed.push(crate::lexer::OwnedToken {
                    kind: OwnedTokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                return self.parse_expanded_expr(&transcribed);
            }
        }

        Err(format!("No matching rule found for macro {}", name))
    }

    fn flatten_tt(&self, tt: &TokenTree) -> Vec<crate::lexer::OwnedToken> {
        let mut tokens = Vec::new();
        match tt {
            TokenTree::Token(t) => tokens.push(t.clone()),
            TokenTree::Group(inner) => {
                for i in inner {
                    tokens.extend(self.flatten_tt(i));
                }
            }
            TokenTree::Delimited(delim, inner) => {
                let (open, close) = match delim {
                    Delimiter::Parenthesis => {
                        (OwnedTokenType::LeftParen, OwnedTokenType::RightParen)
                    }
                    Delimiter::Brace => (OwnedTokenType::LeftBrace, OwnedTokenType::RightBrace),
                    Delimiter::Bracket => {
                        (OwnedTokenType::LeftBracket, OwnedTokenType::RightBracket)
                    }
                };
                tokens.push(crate::lexer::OwnedToken {
                    kind: open,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                for i in inner {
                    tokens.extend(self.flatten_tt(i));
                }
                tokens.push(crate::lexer::OwnedToken {
                    kind: close,
                    line: 0,
                    column: 0,
                    length: 0,
                });
            }
        }
        tokens
    }

    fn match_rule(
        &self,
        matcher: &[TokenTree],
        input: &[crate::lexer::OwnedToken],
    ) -> Result<HashMap<crate::symbol::Symbol, Vec<crate::lexer::OwnedToken>>, String> {
        let mut captures = HashMap::new();
        let mut matcher_tokens = Vec::new();
        for tt in matcher {
            matcher_tokens.extend(self.flatten_tt(tt));
        }

        let mut i = 0; // input index
        let mut j = 0; // matcher index

        while j < matcher_tokens.len() {
            let m_tok = &matcher_tokens[j];

            if m_tok.kind == OwnedTokenType::Dollar && j + 2 < matcher_tokens.len() {
                let name_tok = &matcher_tokens[j + 1];
                let colon_tok = &matcher_tokens[j + 2];

                if let OwnedTokenType::Identifier(name) = &name_tok.kind {
                    if colon_tok.kind == OwnedTokenType::Colon && j + 3 < matcher_tokens.len() {
                        let kind_tok = &matcher_tokens[j + 3];
                        if let OwnedTokenType::Identifier(kind) = &kind_tok.kind {
                            // Match a meta-variable
                            if kind.as_ref() == "expr" {
                                // Simplified: just grab tokens until the next matcher token is found or EOF
                                let mut captured = Vec::new();
                                if j + 4 < matcher_tokens.len() {
                                    let next_m_tok = &matcher_tokens[j + 4];
                                    while i < input.len() && input[i].kind != next_m_tok.kind {
                                        captured.push(input[i].clone());
                                        i += 1;
                                    }
                                } else {
                                    while i < input.len() {
                                        captured.push(input[i].clone());
                                        i += 1;
                                    }
                                }
                                captures.insert(name.clone(), captured);
                                j += 4;
                                continue;
                            }
                        }
                    }
                }
            }

            // Literal match
            if i >= input.len() {
                return Err("Input ended unexpectedly".to_string());
            }
            if matcher_tokens[j].kind != input[i].kind {
                return Err(format!(
                    "Token mismatch: expected {:?}, got {:?}",
                    matcher_tokens[j].kind, input[i].kind
                ));
            }
            i += 1;
            j += 1;
        }

        if i < input.len() {
            return Err("Trailing input tokens".to_string());
        }

        Ok(captures)
    }

    fn transcribe(
        &self,
        transcriber: &[TokenTree],
        captures: &HashMap<crate::symbol::Symbol, Vec<crate::lexer::OwnedToken>>,
    ) -> Result<Vec<crate::lexer::OwnedToken>, String> {
        let mut tokens = Vec::new();
        let mut transcriber_tokens = Vec::new();
        for tt in transcriber {
            transcriber_tokens.extend(self.flatten_tt(tt));
        }

        let mut j = 0;
        while j < transcriber_tokens.len() {
            let m_tok = &transcriber_tokens[j];
            if m_tok.kind == OwnedTokenType::Dollar && j + 1 < transcriber_tokens.len() {
                let name_tok = &transcriber_tokens[j + 1];
                if let OwnedTokenType::Identifier(name) = &name_tok.kind {
                    if let Some(captured) = captures.get(name) {
                        tokens.extend(captured.clone());
                        j += 2;
                        continue;
                    }
                }
            }
            tokens.push(m_tok.clone());
            j += 1;
        }

        // Append EOF to ensure parser completes
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });

        Ok(tokens)
    }

    fn expand_vec_macro(&mut self, tt: &TokenTree) -> Result<expr::Expr, String> {
        let elements = match tt {
            TokenTree::Delimited(_, inner) => inner,
            _ => return Err("Expected delimited token tree for vec!".to_string()),
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(self.flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = self.parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::VecMacro(expr::VecMacroExpr {
            elements: exprs,
            span: Span::default(),
        }))
    }

    fn expand_print_macro(&mut self, tt: &TokenTree) -> Result<expr::Expr, String> {
        let elements = match tt {
            TokenTree::Delimited(_, inner) => inner,
            _ => return Err("Expected delimited token tree for print!".to_string()),
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(self.flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = self.parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::Print(expr::PrintExpr {
            args: exprs,
            span: Span::default(),
        }))
    }

    fn expand_println_macro(&mut self, tt: &TokenTree) -> Result<expr::Expr, String> {
        let elements = match tt {
            TokenTree::Delimited(_, inner) => inner,
            _ => return Err("Expected delimited token tree for println!".to_string()),
        };
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(self.flatten_tt(t));
        }
        tokens.push(crate::lexer::OwnedToken {
            kind: OwnedTokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });
        let exprs = self.parse_expanded_exprs(&tokens)?;
        Ok(expr::Expr::Println(expr::PrintlnExpr {
            args: exprs,
            span: Span::default(),
        }))
    }

    fn expand_mlir_macro(
        &mut self,
        tt: &TokenTree,
        block_tree: &Option<TokenTree>,
    ) -> Result<expr::Expr, String> {
        let mut inputs = Vec::new();
        let mut clobbers = Vec::new();
        let mut returns = None;
        let mut dialects = Vec::new();

        let tokens = match tt {
            TokenTree::Delimited(_, inner) => {
                let mut t = Vec::new();
                for i in inner {
                    t.extend(self.flatten_tt(i));
                }
                t.push(crate::lexer::OwnedToken {
                    kind: crate::lexer::OwnedTokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                t
            }
            _ => return Err("Expected delimited token tree for mlir!".to_string()),
        };

        let lexed_tokens: Vec<_> = tokens.iter().map(|t| t.as_token()).collect();
        let mut parser = crate::parser::Parser::new(&lexed_tokens, "");

        while !parser.check(&crate::lexer::TokenType::Eof) {
            let field_name = match &parser.advance().kind {
                crate::lexer::TokenType::Identifier(ident) => ident.to_string(),
                _ => {
                    return Err(
                        "Expected 'inputs', 'clobbers', 'returns', or 'dialects'".to_string()
                    )
                }
            };
            parser
                .consume(&crate::lexer::TokenType::Colon, "Expected ':'")
                .map_err(|e| e.format(""))?;

            match field_name.as_ref() {
                "inputs" => {
                    parser
                        .consume(&crate::lexer::TokenType::LeftParen, "Expected '('")
                        .map_err(|e| e.format(""))?;
                    if !parser.check(&crate::lexer::TokenType::RightParen) {
                        loop {
                            let is_percent = match parser.peek().kind {
                                crate::lexer::TokenType::Unknown('%') => {
                                    parser.advance();
                                    true
                                }
                                _ => false,
                            };
                            let arg_name = match &parser.advance().kind {
                                crate::lexer::TokenType::Identifier(ident) => {
                                    if is_percent {
                                        format!("%{}", ident)
                                    } else {
                                        ident.to_string()
                                    }
                                }
                                _ => return Err("Expected identifier in inputs".to_string()),
                            };
                            parser
                                .consume(&crate::lexer::TokenType::Equals, "Expected '='")
                                .map_err(|e| e.format(""))?;
                            let expr = parser.parse_expr().map_err(|e| e.format(""))?;
                            parser
                                .consume(&crate::lexer::TokenType::Colon, "Expected ':'")
                                .map_err(|e| e.format(""))?;

                            let mut ty_str = String::new();
                            let mut angle_depth = 0;
                            while !parser.check(&crate::lexer::TokenType::Eof) {
                                if angle_depth == 0
                                    && (parser.check(&crate::lexer::TokenType::Comma)
                                        || parser.check(&crate::lexer::TokenType::RightParen))
                                {
                                    break;
                                }
                                let tok = parser.advance();
                                if tok.kind == crate::lexer::TokenType::LeftAngle {
                                    angle_depth += 1;
                                } else if tok.kind == crate::lexer::TokenType::RightAngle {
                                    angle_depth -= 1;
                                }
                                ty_str.push_str(&tok.kind.to_string());
                            }
                            inputs.push((arg_name.into(), expr, ty_str));

                            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    parser
                        .consume(&crate::lexer::TokenType::RightParen, "Expected ')'")
                        .map_err(|e| e.format(""))?;
                }
                "clobbers" => {
                    parser
                        .consume(&crate::lexer::TokenType::LeftBracket, "Expected '['")
                        .map_err(|e| e.format(""))?;
                    if !parser.check(&crate::lexer::TokenType::RightBracket) {
                        loop {
                            clobbers.push(parser.parse_expr().map_err(|e| e.format(""))?);
                            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    parser
                        .consume(&crate::lexer::TokenType::RightBracket, "Expected ']'")
                        .map_err(|e| e.format(""))?;
                }
                "returns" => {
                    if parser.match_token(&crate::lexer::TokenType::Identifier("void")) {
                        returns = None;
                    } else {
                        returns = Some(parser.parse_type().map_err(|e| e.format(""))?);
                    }
                }
                "dialects" => {
                    parser
                        .consume(&crate::lexer::TokenType::LeftBracket, "Expected '['")
                        .map_err(|e| e.format(""))?;
                    if !parser.check(&crate::lexer::TokenType::RightBracket) {
                        loop {
                            match &parser.advance().kind {
                                crate::lexer::TokenType::StringLiteral(lit) => {
                                    dialects.push(lit.to_string());
                                }
                                _ => return Err("Expected string literal in dialects".to_string()),
                            }
                            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                                break;
                            }
                        }
                    }
                    parser
                        .consume(&crate::lexer::TokenType::RightBracket, "Expected ']'")
                        .map_err(|e| e.format(""))?;
                }
                _ => return Err(format!("Unknown field '{}' in mlir! macro", field_name)),
            }

            parser.match_token(&crate::lexer::TokenType::Comma);
        }

        let block_str = if let Some(TokenTree::Delimited(_, inner)) = block_tree {
            let mut t = Vec::new();
            for i in inner {
                t.extend(self.flatten_tt(i));
            }

            let mut s = String::new();
            let mut current_line = 0;
            let mut current_col = 0;

            for tok in t {
                if current_line == 0 {
                    current_line = tok.line;
                    current_col = tok.column;
                }

                if tok.line > current_line {
                    for _ in 0..(tok.line - current_line) {
                        s.push('\n');
                    }
                    current_col = 1;
                    current_line = tok.line;
                }

                if tok.column > current_col {
                    for _ in 0..(tok.column - current_col) {
                        s.push(' ');
                    }
                }

                s.push_str(&tok.kind.to_string());
                current_col = tok.column + tok.length;
            }
            s
        } else {
            return Err("mlir! macro requires a trailing block".to_string());
        };

        Ok(expr::Expr::InlineMlir(expr::InlineMlirExpr {
            inputs,
            clobbers,
            returns,
            dialects,
            block_str,
            span: Span::default(),
        }))
    }
}
