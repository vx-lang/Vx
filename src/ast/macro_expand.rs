use super::*;
use std::collections::HashMap;

pub struct MacroExpander<'a> {
    pub macros: &'a HashMap<String, Vec<MacroRule>>,
}

impl<'a> MacroExpander<'a> {
    pub fn new(macros: &'a HashMap<String, Vec<MacroRule>>) -> Self {
        Self { macros }
    }

    pub fn expand_module(&mut self, module: &mut VxModule) -> Result<(), String> {
        // Expand top level decls
        for func in &mut module.functions {
            self.expand_function(func)?;
        }
        Ok(())
    }

    fn expand_function(&mut self, func: &mut Function) -> Result<(), String> {
        let mut i = 0;
        while i < func.body.len() {
            let stmt = func.body.remove(i);
            let mut expanded_stmts = self.expand_stmt(stmt)?;
            for s in expanded_stmts.drain(..).rev() {
                func.body.insert(i, s);
            }
            i += 1;
        }
        Ok(())
    }

    fn expand_stmt(&mut self, mut stmt: stmt::Statement) -> Result<Vec<stmt::Statement>, String> {
        if let stmt::Statement::MacroCall(call) = stmt {
            let expanded_expr = self.expand_macro_call(&call.name, &call.token_tree)?;
            let recursively_expanded = self.expand_expr(expanded_expr)?;
            // for now, a macro call statement returns the expanded expr as a statement
            return Ok(vec![stmt::Statement::ExprStmt(stmt::ExprStmtStmt {
                expr: recursively_expanded,
                has_semi: true,
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
                e.expr = self.expand_expr(e.expr.clone())?;
            }
            stmt::Statement::LetDecl(l) => {
                l.expr = self.expand_expr(l.expr.clone())?;
            }
            stmt::Statement::Assign(a) => {
                a.lhs = self.expand_expr(a.lhs.clone())?;
                a.rhs = self.expand_expr(a.rhs.clone())?;
            }
            stmt::Statement::CompoundAssign(a) => {
                a.lhs = self.expand_expr(a.lhs.clone())?;
                a.rhs = self.expand_expr(a.rhs.clone())?;
            }
            stmt::Statement::Return(r) => {
                r.expr = self.expand_expr(r.expr.clone())?;
            }
            stmt::Statement::Assert(a) => {
                a.expr = Box::new(self.expand_expr(*a.expr.clone())?);
            }
            stmt::Statement::Loop(l) => {
                let mut i = 0;
                while i < l.body.len() {
                    let s = l.body.remove(i);
                    let mut expanded = self.expand_stmt(s)?;
                    for e in expanded.drain(..).rev() {
                        l.body.insert(i, e);
                    }
                    i += 1;
                }
            }
            stmt::Statement::ForLoop(f) => {
                f.iterable = Box::new(self.expand_expr(*f.iterable.clone())?);
                let mut i = 0;
                while i < f.body.len() {
                    let s = f.body.remove(i);
                    let mut expanded = self.expand_stmt(s)?;
                    for e in expanded.drain(..).rev() {
                        f.body.insert(i, e);
                    }
                    i += 1;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn expand_expr(&mut self, mut expr: expr::Expr) -> Result<expr::Expr, String> {
        if let expr::Expr::MacroCall(call) = expr {
            let expanded = self.expand_macro_call(&call.name, &call.token_tree)?;
            return self.expand_expr(expanded);
        }
        // Traverse and expand
        match &mut expr {
            expr::Expr::BinaryOp(b) => {
                b.lhs = Box::new(self.expand_expr(*b.lhs.clone())?);
                b.rhs = Box::new(self.expand_expr(*b.rhs.clone())?);
            }
            expr::Expr::RelationalOp(b) => {
                b.lhs = Box::new(self.expand_expr(*b.lhs.clone())?);
                b.rhs = Box::new(self.expand_expr(*b.rhs.clone())?);
            }
            expr::Expr::LogicalOp(b) => {
                b.lhs = Box::new(self.expand_expr(*b.lhs.clone())?);
                b.rhs = Box::new(self.expand_expr(*b.rhs.clone())?);
            }
            expr::Expr::Range(b) => {
                b.start = Box::new(self.expand_expr(*b.start.clone())?);
                b.end = Box::new(self.expand_expr(*b.end.clone())?);
            }
            expr::Expr::UnaryOp(u) => {
                u.expr = Box::new(self.expand_expr(*u.expr.clone())?);
            }
            expr::Expr::Borrow(u) => {
                u.expr = Box::new(self.expand_expr(*u.expr.clone())?);
            }
            expr::Expr::Dereference(u) => {
                u.expr = Box::new(self.expand_expr(*u.expr.clone())?);
            }
            expr::Expr::MemberAccess(m) => {
                m.base = Box::new(self.expand_expr(*m.base.clone())?);
            }
            expr::Expr::IndexAccess(m) => {
                m.base = Box::new(self.expand_expr(*m.base.clone())?);
                m.index = Box::new(self.expand_expr(*m.index.clone())?);
            }
            expr::Expr::Match(m) => {
                m.expr = Box::new(self.expand_expr(*m.expr.clone())?);
                for arm in &mut m.arms {
                    let mut i = 0;
                    while i < arm.body.len() {
                        let s = arm.body.remove(i);
                        let mut expanded = self.expand_stmt(s)?;
                        for e in expanded.drain(..).rev() {
                            arm.body.insert(i, e);
                        }
                        i += 1;
                    }
                }
            }
            expr::Expr::FunctionCall(f) => {
                for arg in &mut f.args {
                    *arg = self.expand_expr(arg.clone())?;
                }
            }
            expr::Expr::MethodCall(m) => {
                m.base = Box::new(self.expand_expr(*m.base.clone())?);
                for arg in &mut m.args {
                    *arg = self.expand_expr(arg.clone())?;
                }
            }
            expr::Expr::StructInit(s) => {
                for (_, e) in &mut s.fields {
                    *e = self.expand_expr(e.clone())?;
                }
            }
            expr::Expr::Array(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(e.clone())?;
                }
            }
            expr::Expr::VecMacro(a) => {
                for e in &mut a.elements {
                    *e = self.expand_expr(e.clone())?;
                }
            }
            expr::Expr::If(i) => {
                i.cond = Box::new(self.expand_expr(*i.cond.clone())?);
                let mut j = 0;
                while j < i.then_block.len() {
                    let s = i.then_block.remove(j);
                    let mut expanded = self.expand_stmt(s)?;
                    for e in expanded.drain(..).rev() {
                        i.then_block.insert(j, e);
                    }
                    j += 1;
                }
                if let Some(else_b) = &mut i.else_block {
                    let mut j = 0;
                    while j < else_b.len() {
                        let s = else_b.remove(j);
                        let mut expanded = self.expand_stmt(s)?;
                        for e in expanded.drain(..).rev() {
                            else_b.insert(j, e);
                        }
                        j += 1;
                    }
                }
            }
            expr::Expr::UnsafeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    *ret = Box::new(self.expand_expr(*ret.clone())?);
                }
                let mut j = 0;
                while j < u.stmts.len() {
                    let s = u.stmts.remove(j);
                    let mut expanded = self.expand_stmt(s)?;
                    for e in expanded.drain(..).rev() {
                        u.stmts.insert(j, e);
                    }
                    j += 1;
                }
            }
            expr::Expr::ComptimeBlock(u) => {
                if let Some(ret) = &mut u.ret {
                    *ret = Box::new(self.expand_expr(*ret.clone())?);
                }
                let mut j = 0;
                while j < u.stmts.len() {
                    let s = u.stmts.remove(j);
                    let mut expanded = self.expand_stmt(s)?;
                    for e in expanded.drain(..).rev() {
                        u.stmts.insert(j, e);
                    }
                    j += 1;
                }
            }
            expr::Expr::Closure(c) => {
                c.body = Box::new(self.expand_expr(*c.body.clone())?);
            }
            expr::Expr::SpawnOn(s) => {
                if let Some(ret) = &mut s.ret {
                    *ret = Box::new(self.expand_expr(*ret.clone())?);
                }
                let mut j = 0;
                while j < s.stmts.len() {
                    let st = s.stmts.remove(j);
                    let mut expanded = self.expand_stmt(st)?;
                    for e in expanded.drain(..).rev() {
                        s.stmts.insert(j, e);
                    }
                    j += 1;
                }
            }
            expr::Expr::Grad(g) => {
                for arg in &mut g.args {
                    *arg = self.expand_expr(arg.clone())?;
                }
            }
            expr::Expr::Vjp(v) => {
                for arg in &mut v.args {
                    *arg = self.expand_expr(arg.clone())?;
                }
                v.cotangent = Box::new(self.expand_expr(*v.cotangent.clone())?);
            }
            expr::Expr::Jvp(j_expr) => {
                for arg in &mut j_expr.args {
                    *arg = self.expand_expr(arg.clone())?;
                }
                j_expr.tangent = Box::new(self.expand_expr(*j_expr.tangent.clone())?);
            }
            _ => {}
        }
        Ok(expr)
    }

    fn expand_macro_call(&mut self, name: &str, tt: &TokenTree) -> Result<expr::Expr, String> {
        println!("Expanding macro call: {}!", name);

        if name == "vec" {
            return self.expand_vec_macro(tt);
        }

        let rules = self
            .macros
            .get(name)
            .ok_or_else(|| format!("Macro {} not found", name))?;

        let input_tokens = self.flatten_tt(tt);

        for rule in rules {
            if let Ok(captures) = self.match_rule(&rule.matcher, &input_tokens) {
                let mut transcribed = self.transcribe(&rule.transcriber, &captures)?;
                // IMPORTANT: The parser expects an EOF token at the end!
                transcribed.push(crate::lexer::Token {
                    kind: crate::lexer::TokenType::Eof,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                let mut parser = crate::parser::Parser::new(transcribed, "");
                return parser.parse_expr();
            }
        }

        Err(format!("No matching rule found for macro {}", name))
    }

    fn flatten_tt(&self, tt: &TokenTree) -> Vec<crate::lexer::Token> {
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
                    crate::ast::Delimiter::Parenthesis => (
                        crate::lexer::TokenType::LeftParen,
                        crate::lexer::TokenType::RightParen,
                    ),
                    crate::ast::Delimiter::Brace => (
                        crate::lexer::TokenType::LeftBrace,
                        crate::lexer::TokenType::RightBrace,
                    ),
                    crate::ast::Delimiter::Bracket => (
                        crate::lexer::TokenType::LeftBracket,
                        crate::lexer::TokenType::RightBracket,
                    ),
                };
                tokens.push(crate::lexer::Token {
                    kind: open,
                    line: 0,
                    column: 0,
                    length: 0,
                });
                for i in inner {
                    tokens.extend(self.flatten_tt(i));
                }
                tokens.push(crate::lexer::Token {
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
        input: &[crate::lexer::Token],
    ) -> Result<HashMap<String, Vec<crate::lexer::Token>>, String> {
        let mut captures = HashMap::new();
        let mut matcher_tokens = Vec::new();
        for tt in matcher {
            matcher_tokens.extend(self.flatten_tt(tt));
        }

        let mut i = 0; // input index
        let mut j = 0; // matcher index

        while j < matcher_tokens.len() {
            let m_tok = &matcher_tokens[j];

            if m_tok.kind == crate::lexer::TokenType::Dollar {
                if j + 2 < matcher_tokens.len() {
                    let name_tok = &matcher_tokens[j + 1];
                    let colon_tok = &matcher_tokens[j + 2];

                    if let crate::lexer::TokenType::Identifier(name) = &name_tok.kind {
                        if colon_tok.kind == crate::lexer::TokenType::Colon {
                            if j + 3 < matcher_tokens.len() {
                                let kind_tok = &matcher_tokens[j + 3];
                                if let crate::lexer::TokenType::Identifier(kind) = &kind_tok.kind {
                                    // Match a meta-variable
                                    if kind == "expr" {
                                        // Simplified: just grab tokens until the next matcher token is found or EOF
                                        let mut captured = Vec::new();
                                        if j + 4 < matcher_tokens.len() {
                                            let next_m_tok = &matcher_tokens[j + 4];
                                            while i < input.len()
                                                && input[i].kind != next_m_tok.kind
                                            {
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
        captures: &HashMap<String, Vec<crate::lexer::Token>>,
    ) -> Result<Vec<crate::lexer::Token>, String> {
        let mut tokens = Vec::new();
        let mut transcriber_tokens = Vec::new();
        for tt in transcriber {
            transcriber_tokens.extend(self.flatten_tt(tt));
        }

        let mut j = 0;
        while j < transcriber_tokens.len() {
            let m_tok = &transcriber_tokens[j];
            if m_tok.kind == crate::lexer::TokenType::Dollar {
                if j + 1 < transcriber_tokens.len() {
                    let name_tok = &transcriber_tokens[j + 1];
                    if let crate::lexer::TokenType::Identifier(name) = &name_tok.kind {
                        if let Some(captured) = captures.get(name) {
                            tokens.extend(captured.clone());
                            j += 2;
                            continue;
                        }
                    }
                }
            }
            tokens.push(m_tok.clone());
            j += 1;
        }

        // Append EOF to ensure parser completes
        tokens.push(crate::lexer::Token {
            kind: crate::lexer::TokenType::Eof,
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

        // Extract tokens
        let mut tokens = Vec::new();
        for t in elements {
            tokens.extend(self.flatten_tt(t));
        }

        // Append EOF token
        tokens.push(crate::lexer::Token {
            kind: crate::lexer::TokenType::Eof,
            line: 0,
            column: 0,
            length: 0,
        });

        let mut parser = crate::parser::Parser::new(tokens, "");
        let mut exprs = Vec::new();
        while !parser.check(&crate::lexer::TokenType::Eof) {
            exprs.push(parser.parse_expr()?);
            if !parser.match_token(&crate::lexer::TokenType::Comma) {
                break;
            }
        }

        Ok(expr::Expr::VecMacro(expr::VecMacroExpr {
            elements: exprs,
            span: crate::ast::Span::default(),
        }))
    }
}
