use crate::ast::*;
use crate::lexer::{Token, TokenType};

pub struct Parser<'a> {
    tokens: Vec<Token>,
    pos: usize,
    generic_params: Vec<String>, // Tracks generic parameters in scope
    source: &'a str,
}

impl From<&str> for crate::ast::Function {
    fn from(source: &str) -> Self {
        // Strip out 'pub' keyword if the user provided it as an example,
        // since Vx currently expects functions to start with 'fn'.
        let cleaned_source = source.trim().trim_start_matches("pub ");

        let mut lexer = crate::lexer::Lexer::new(cleaned_source);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, cleaned_source);
        parser
            .parse_function()
            .expect("Failed to parse function source")
    }
}

impl<'a> Parser<'a> {
    pub fn new(tokens: Vec<Token>, source: &'a str) -> Self {
        Self {
            tokens,
            pos: 0,
            generic_params: Vec::new(),
            source,
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let token = &self.tokens[self.pos];
        if self.pos < self.tokens.len() - 1 {
            self.pos += 1;
        }
        token
    }

    fn check(&self, kind: &TokenType) -> bool {
        &self.peek().kind == kind
    }

    #[allow(dead_code)]
    fn match_token(&mut self, kind: &TokenType) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    #[allow(dead_code)]
    fn consume(&mut self, kind: &TokenType, msg: &str) -> Result<&Token, String> {
        if self.check(kind) {
            Ok(self.advance())
        } else {
            let token = self.peek();
            Err(crate::error::format_compiler_error(
                self.source,
                token.line,
                token.column,
                token.length.max(1),
                msg,
            ))
        }
    }

    fn parse_topology(&mut self) -> Result<Topology, String> {
        self.consume(&TokenType::Topology, "Expected 'Topology'")?;
        self.consume(&TokenType::DoubleColon, "Expected '::' after 'Topology'")?;
        let ident = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected hardware identifier after Topology::".to_string()),
        };
        match ident.as_str() {
            "Host" => Ok(Topology::Host),
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
            _ => Err(format!("Unknown topology {}", ident)),
        }
    }

    fn parse_memory_space(&mut self) -> Result<MemorySpace, String> {
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

    fn parse_type(&mut self) -> Result<Type, String> {
        if self.match_token(&TokenType::Ampersand) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner = self.parse_type()?;
            Ok(Type::Borrow(Box::new(inner), None, is_mut))
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
                            _ => return Err(format!("Unknown element type {}", ty_ident)),
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
                "Bool" => Ok(Type::Scalar(ElementType::Bool)),
                _ => {
                    // Check for GenericInstance like Config<f32>
                    if self.check(&TokenType::LeftAngle) {
                        self.advance(); // consume '<'
                        let mut type_args = Vec::new();
                        while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
                            type_args.push(self.parse_type()?);
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

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_binary_expr(0)
    }

    fn parse_binary_expr(&mut self, precedence: u8) -> Result<Expr, String> {
        let mut left = self.parse_primary_expr()?;

        while let Some(op_prec) = self.get_operator_precedence(&self.peek().kind) {
            if op_prec < precedence {
                break;
            }
            let token = self.advance().clone();
            let op = match token.kind {
                TokenType::OrOr => BinaryOp::Or,
                TokenType::AndAnd => BinaryOp::And,
                TokenType::EqEq => BinaryOp::Eq,
                TokenType::NotEq => BinaryOp::NotEq,
                TokenType::LessEq => BinaryOp::Le,
                TokenType::GreaterEq => BinaryOp::Ge,
                TokenType::LeftAngle => BinaryOp::Lt,
                TokenType::RightAngle => BinaryOp::Gt,
                TokenType::Plus => BinaryOp::Add,
                TokenType::Minus => BinaryOp::Sub,
                TokenType::Star => BinaryOp::Mul,
                TokenType::Slash => BinaryOp::Div,
                _ => return Err("Unknown binary operator".to_string()),
            };
            let right = self.parse_binary_expr(op_prec + 1)?;
            left = Expr::BinaryOp(Box::new(left), op, Box::new(right), Span::default());
        }

        Ok(left)
    }

    fn get_operator_precedence(&self, kind: &TokenType) -> Option<u8> {
        match kind {
            TokenType::OrOr => Some(10),
            TokenType::AndAnd => Some(20),
            TokenType::EqEq | TokenType::NotEq => Some(30),
            TokenType::LessEq
            | TokenType::GreaterEq
            | TokenType::LeftAngle
            | TokenType::RightAngle => Some(40),
            TokenType::Plus | TokenType::Minus => Some(50),
            TokenType::Star | TokenType::Slash => Some(60),
            _ => None,
        }
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, String> {
        if self.match_token(&TokenType::Bang) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::UnaryOp(
                UnaryOp::Not,
                Box::new(inner),
                Span::default(),
            ));
        } else if self.match_token(&TokenType::Ampersand) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::Borrow(Box::new(inner), is_mut, Span::default()));
        } else if self.match_token(&TokenType::Star) {
            let inner = self.parse_primary_expr()?;
            return Ok(Expr::Dereference(Box::new(inner), Span::default()));
        } else if self.match_token(&TokenType::Unsafe) {
            self.consume(&TokenType::LeftBrace, "Expected '{' after unsafe")?;
            let mut stmts = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                stmts.push(self.parse_statement()?);
            }
            self.consume(&TokenType::RightBrace, "Expected '}'")?;
            return Ok(Expr::UnsafeBlock(stmts, None, Span::default()));
        }

        let mut expr = if self.match_token(&TokenType::Transfer) {
            self.consume(&TokenType::LeftParen, "Expected '(' after transfer")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let mem = self.parse_memory_space()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::Transfer(Box::new(inner), mem, Span::default())
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
            Expr::Array(elements, Span::default())
        } else if self.check(&TokenType::Memory) {
            Expr::MemorySpace(self.parse_memory_space()?, Span::default())
        } else if self.check(&TokenType::Topology) {
            Expr::Topology(self.parse_topology()?, Span::default())
        } else if self.check(&TokenType::Verified) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after Verified")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::FunctionCall("Verified".to_string(), vec![inner], Span::default())
        } else {
            let token = self.advance().clone();
            match token.kind {
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
                        Expr::FunctionCall(call_name, args, Span::default())
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
                            Expr::StructInit(call_name, fields, Span::default())
                        } else if self.match_token(&TokenType::DoubleColon) {
                            let variant = match self.advance().kind.clone() {
                                TokenType::Identifier(v) => v,
                                _ => return Err("Expected enum variant after ::".to_string()),
                            };
                            Expr::EnumVariant(call_name, variant, Span::default())
                        } else {
                            Expr::Identifier(call_name, Span::default())
                        }
                    } else if self.match_token(&TokenType::DoubleColon) {
                        let variant = match self.advance().kind.clone() {
                            TokenType::Identifier(v) => v,
                            _ => return Err("Expected enum variant after ::".to_string()),
                        };
                        Expr::EnumVariant(call_name, variant, Span::default())
                    } else {
                        Expr::Identifier(call_name, Span::default())
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
                        // Rust-like defaults: i32 for integers, f64 for floats
                        if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                            Some(crate::ast::ElementType::F64)
                        } else {
                            Some(crate::ast::ElementType::I32)
                        }
                    };

                    Expr::Number(num_str, el_ty, Span::default())
                }
                TokenType::StringLiteral(s) => Expr::StringLiteral(s, Span::default()),

                TokenType::Comptime => {
                    self.consume(&TokenType::LeftBrace, "Expected '{' for comptime block")?;
                    let mut stmts = Vec::new();
                    let mut ret_expr = None;
                    while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                        let stmt = self.parse_statement()?;
                        if self.check(&TokenType::RightBrace) {
                            if let Statement::ExprStmt(e, _) = stmt {
                                ret_expr = Some(Box::new(e));
                                break;
                            }
                        }
                        stmts.push(stmt);
                    }
                    self.consume(&TokenType::RightBrace, "Expected '}'")?;
                    Expr::ComptimeBlock(stmts, ret_expr, Span::default())
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
                    expr = Expr::MethodCall(Box::new(expr), ident, args, Span::default());
                } else {
                    expr = Expr::MemberAccess(Box::new(expr), ident, Span::default());
                }
            } else if self.match_token(&TokenType::LeftBracket) {
                let index = self.parse_expr()?;
                self.consume(&TokenType::RightBracket, "Expected ']'")?;
                expr = Expr::IndexAccess(Box::new(expr), Box::new(index), Span::default());
            } else {
                break;
            }
        }

        Ok(expr)
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
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
                Ok(Statement::LetDecl(
                    name,
                    is_mut,
                    type_annotation,
                    expr,
                    Span::default(),
                ))
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
                self.consume(&TokenType::RightBrace, "Expected '}' after comptime block")?;
                Ok(Statement::Comptime(stmts, Span::default()))
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
                Ok(Statement::Assert(Box::new(expr), msg, Span::default()))
            }
            TokenType::Return => {
                self.advance();
                let expr = self.parse_expr()?;
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Return(expr, Span::default()))
            }
            TokenType::Spawn => {
                self.advance();
                self.consume(&TokenType::On, "Expected 'on' after 'spawn'")?;
                self.consume(&TokenType::LeftParen, "Expected '('")?;
                let top = self.parse_topology()?;
                self.consume(&TokenType::RightParen, "Expected ')'")?;
                self.consume(&TokenType::LeftBrace, "Expected '{'")?;
                let mut stmts = Vec::new();
                while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                    stmts.push(self.parse_statement()?);
                }
                self.consume(&TokenType::RightBrace, "Expected '}'")?;
                Ok(Statement::SpawnOn(top, stmts, Span::default()))
            }
            TokenType::If => {
                self.advance();
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
                        else_block = Some(vec![self.parse_statement()?]);
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
                Ok(Statement::If(
                    Box::new(cond),
                    then_block,
                    else_block,
                    Span::default(),
                ))
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
                Ok(Statement::ForLoop(
                    iter,
                    Box::new(start),
                    Box::new(end),
                    stmts,
                    Span::default(),
                ))
            }
            _ => {
                let expr = self.parse_expr()?;
                if self.match_token(&TokenType::Equals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::Assign(expr, rhs, Span::default()))
                } else if self.match_token(&TokenType::PlusEquals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::CompoundAssign(
                        expr,
                        BinaryOp::Add,
                        rhs,
                        Span::default(),
                    ))
                } else {
                    if let Expr::UnsafeBlock(..) = &expr {
                        self.match_token(&TokenType::Semicolon); // Optional
                    } else {
                        self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    }
                    Ok(Statement::ExprStmt(expr, Span::default()))
                }
            }
        }
    }

    fn parse_generic_params(&mut self) -> Result<Vec<(String, Option<String>)>, String> {
        let mut generics = Vec::new();
        if self.match_token(&TokenType::LeftAngle) {
            while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
                let name = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err("Expected generic parameter name".to_string()),
                };
                self.generic_params.push(name.clone());
                let mut bound = None;
                if self.match_token(&TokenType::Colon) {
                    bound = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => Some(s),
                        _ => return Err("Expected trait bound identifier".to_string()),
                    };
                }
                generics.push((name, bound));
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            self.consume(
                &TokenType::RightAngle,
                "Expected '>' after generic parameters",
            )?;
        }
        Ok(generics)
    }

    pub fn parse_function(&mut self) -> Result<Function, String> {
        self.consume(&TokenType::Fn, "Expected 'fn'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected function name".to_string()),
        };

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftParen, "Expected '(' after function name")?;
        let mut params = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                let p_name = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err("Expected parameter name".to_string()),
                };
                self.consume(&TokenType::Colon, "Expected ':'")?;
                let p_type = self.parse_type()?;
                params.push((p_name, p_type));

                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
        }
        self.consume(&TokenType::RightParen, "Expected ')'")?;

        self.consume(&TokenType::Arrow, "Expected '->'")?;
        let return_type = self.parse_type()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            body.push(self.parse_statement()?);
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(Function {
            name,
            generics,
            params,
            return_type,
            body,
        })
    }

    fn parse_struct_decl(&mut self) -> Result<StructDecl, String> {
        self.consume(&TokenType::Struct, "Expected 'struct'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected struct name".to_string()),
        };

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut fields = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let f_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected field name".to_string()),
            };
            self.consume(&TokenType::Colon, "Expected ':'")?;
            let f_type = self.parse_type()?;
            fields.push((f_name, f_type));

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(StructDecl {
            name,
            generics,
            fields,
        })
    }

    fn parse_enum_decl(&mut self) -> Result<EnumDecl, String> {
        self.consume(&TokenType::Enum, "Expected 'enum'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected enum name".to_string()),
        };

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut variants = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let v_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected enum variant name".to_string()),
            };
            variants.push(v_name);

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        Ok(EnumDecl { name, variants })
    }

    fn parse_extern_block(&mut self) -> Result<Vec<ExternDecl>, String> {
        self.consume(&TokenType::Extern, "Expected 'extern'")?;

        // Optional "C" ABI string literal (we ignore it for now but parse it if it exists)
        if let TokenType::StringLiteral(s) = &self.peek().kind {
            if s == "C" {
                self.advance();
            }
        }

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut externs = Vec::new();

        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let is_safe = self.match_token(&TokenType::Safe);
            self.consume(&TokenType::Fn, "Expected 'fn'")?;
            let name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected function name".to_string()),
            };

            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let mut params = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    let p_name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err("Expected parameter name".to_string()),
                    };
                    self.consume(&TokenType::Colon, "Expected ':'")?;
                    let p_type = self.parse_type()?;
                    params.push((p_name, p_type));

                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::RightParen, "Expected ')'")?;

            self.consume(&TokenType::Arrow, "Expected '->'")?;
            let return_type = self.parse_type()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;

            externs.push(ExternDecl {
                name,
                is_safe,
                params,
                return_type,
            });
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        Ok(externs)
    }

    fn parse_trait_decl(&mut self) -> Result<TraitDecl, String> {
        self.consume(&TokenType::Trait, "Expected 'trait'")?;
        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err("Expected trait name".to_string()),
        };
        self.consume(&TokenType::LeftBrace, "Expected '{'")?;

        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.consume(&TokenType::Fn, "Expected 'fn' in trait")?;
            let method_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected method name".to_string()),
            };
            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let mut params = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    let p_name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err("Expected parameter name".to_string()),
                    };
                    self.consume(&TokenType::Colon, "Expected ':'")?;
                    let p_type = self.parse_type()?;
                    params.push((p_name, p_type));

                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
            }
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            self.consume(&TokenType::Arrow, "Expected '->'")?;
            let return_type = self.parse_type()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            methods.push((method_name, params, return_type));
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;
        Ok(TraitDecl { name, methods })
    }

    fn parse_impl_block(&mut self) -> Result<ImplBlock, String> {
        self.consume(&TokenType::Impl, "Expected 'impl'")?;

        // Either `impl Trait for Type` or `impl Type`
        let mut trait_name = None;
        let target_type;

        // Since we don't have lookahead to distinguish `impl Trait for Type` from `impl Type`,
        // if we see `Identifier` followed by `for`, it's a trait. Otherwise it's a type.
        // Wait, parse_type handles `Struct(name)`, which is an identifier!
        // We can just peek ahead.
        let parsed_type = self.parse_type()?;
        if self.check(&TokenType::For) {
            self.advance(); // consume 'for'
            if let Type::Struct(name, _) = parsed_type {
                trait_name = Some(name);
            } else {
                return Err("Expected trait name before 'for'".to_string());
            }
            target_type = self.parse_type()?;
        } else {
            target_type = parsed_type;
        }

        self.consume(&TokenType::LeftBrace, "Expected '{' after impl target")?;
        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            methods.push(self.parse_function()?);
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;
        Ok(ImplBlock {
            trait_name,
            target_type,
            methods,
        })
    }

    fn parse_import_decl(&mut self) -> Result<ImportDecl, String> {
        self.consume(&TokenType::Import, "Expected 'import'")?;
        let mut path = Vec::new();
        loop {
            let ident = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected identifier in import path".to_string()),
            };
            path.push(ident);
            if self.match_token(&TokenType::DoubleColon) {
                continue;
            } else {
                break;
            }
        }
        self.consume(&TokenType::Semicolon, "Expected ';' after import path")?;
        Ok(ImportDecl { path })
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut imports = Vec::new();
        let mut externs = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut functions = Vec::new();
        while !self.check(&TokenType::Eof) {
            if self.check(&TokenType::Import) {
                imports.push(self.parse_import_decl()?);
            } else if self.check(&TokenType::Extern) {
                externs.extend(self.parse_extern_block()?);
            } else if self.check(&TokenType::Trait) {
                traits.push(self.parse_trait_decl()?);
            } else if self.check(&TokenType::Impl) {
                impls.push(self.parse_impl_block()?);
            } else if self.check(&TokenType::Struct) {
                structs.push(self.parse_struct_decl()?);
            } else if self.check(&TokenType::Enum) {
                enums.push(self.parse_enum_decl()?);
            } else if self.check(&TokenType::Fn) {
                functions.push(self.parse_function()?);
            } else {
                return Err(format!(
                    "Unexpected token at program root: {:?}",
                    self.peek()
                ));
            }
        }
        Ok(Program {
            module_path: self.source.to_string(), // Default fallback, should be overridden by pipeline
            imports,
            externs,
            structs,
            enums,
            traits,
            impls,
            functions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use rstest::rstest;

    #[rstest]
    #[case("fn main() -> Tensor {}")]
    fn test_parse_empty_function(#[case] input: &str) {
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);
        assert_eq!(program.functions[0].name, "main");
    }

    #[test]
    fn test_parse_distributed_matmul() {
        let input = r#"
fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>, b: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_a = transfer(a, Memory::NPU_HBM);
        let local_b = transfer(b, Memory::NPU_HBM);
        let result = custom_matmul(local_a, local_b);
        return transfer(result, Memory::Host_DRAM);
    }
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens, input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);

        let func = &program.functions[0];
        assert_eq!(func.name, "distributed_matmul");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].0, "a");

        // Assert return type is Verified<Tensor>
        assert_eq!(
            func.return_type,
            Type::Verified(Box::new(Type::Tensor(ElementType::F32, vec![], None)))
        );

        // Assert body has one statement (spawn on)
        assert_eq!(func.body.len(), 1);
        if let Statement::SpawnOn(top, stmts, _) = &func.body[0] {
            assert_eq!(
                *top,
                Topology::NPU(Box::new(Expr::Number(
                    "0".to_string(),
                    Some(crate::ast::ElementType::I32),
                    Span::default()
                )))
            );
            assert_eq!(stmts.len(), 4);
        } else {
            panic!("Expected SpawnOn statement");
        }
    }

    #[test]
    fn test_parse_let_mut_with_type() {
        let input = "fn main() -> Tensor { let mut x: Tensor = Tensor([1, 2]); }";
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        let func = &program.functions[0];
        if let Statement::LetDecl(name, is_mut, ty, expr, _) = &func.body[0] {
            assert_eq!(name, "x");
            assert!(is_mut);
            assert_eq!(ty, &Some(Type::Tensor(ElementType::F32, vec![], None)));
            if let Expr::FunctionCall(func_name, args, _) = expr {
                assert_eq!(func_name, "Tensor");
                assert_eq!(args.len(), 1);
                if let Expr::Array(elements, _) = &args[0] {
                    assert_eq!(elements.len(), 2);
                } else {
                    panic!("Expected array");
                }
            } else {
                panic!("Expected function call");
            }
        } else {
            panic!("Expected LetDecl");
        }
    }

    #[test]
    fn test_parse_for_loop() {
        let input = "fn main() -> Tensor { for i in 0..10 { x = 5; } }";
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        if let Statement::ForLoop(iter, start, end, body, _) = &program.functions[0].body[0] {
            assert_eq!(iter, "i");
            assert_eq!(
                **start,
                Expr::Number(
                    "0".to_string(),
                    Some(crate::ast::ElementType::I32),
                    Span::default()
                )
            );
            assert_eq!(
                **end,
                Expr::Number(
                    "10".to_string(),
                    Some(crate::ast::ElementType::I32),
                    Span::default()
                )
            );
            assert_eq!(body.len(), 1);
            if let Statement::Assign(lhs, rhs, _) = &body[0] {
                assert_eq!(*lhs, Expr::Identifier("x".to_string(), Span::default()));
                assert_eq!(
                    *rhs,
                    Expr::Number(
                        "5".to_string(),
                        Some(crate::ast::ElementType::I32),
                        Span::default()
                    )
                );
            } else {
                panic!("Expected Assign");
            }
        } else {
            panic!("Expected ForLoop");
        }
    }

    #[test]
    fn test_parse_compound_assign() {
        let input = "fn main() -> Tensor { x[0] += y * z; }";
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        if let Statement::CompoundAssign(lhs, op, rhs, _) = &program.functions[0].body[0] {
            assert_eq!(*op, BinaryOp::Add);
            if let Expr::IndexAccess(arr, idx, _) = lhs {
                assert_eq!(**arr, Expr::Identifier("x".to_string(), Span::default()));
                assert_eq!(
                    **idx,
                    Expr::Number(
                        "0".to_string(),
                        Some(crate::ast::ElementType::I32),
                        Span::default()
                    )
                );
            } else {
                panic!("Expected IndexAccess");
            }

            if let Expr::BinaryOp(left, binop, right, _) = rhs {
                assert_eq!(*binop, BinaryOp::Mul);
                assert_eq!(**left, Expr::Identifier("y".to_string(), Span::default()));
                assert_eq!(**right, Expr::Identifier("z".to_string(), Span::default()));
            } else {
                panic!("Expected BinaryOp");
            }
        } else {
            panic!("Expected CompoundAssign");
        }
    }

    #[test]
    fn test_parse_member_and_method() {
        let input = "fn main() -> Tensor { x.shape.with_memory(Memory::NPU_HBM); }";
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        if let Statement::ExprStmt(expr, _) = &program.functions[0].body[0] {
            if let Expr::MethodCall(obj, method, args, _) = expr {
                assert_eq!(method, "with_memory");
                assert_eq!(args.len(), 1);
                if let Expr::MemberAccess(inner_obj, member, _) = &**obj {
                    assert_eq!(member, "shape");
                    assert_eq!(
                        **inner_obj,
                        Expr::Identifier("x".to_string(), Span::default())
                    );
                } else {
                    panic!("Expected MemberAccess");
                }
            } else {
                panic!("Expected MethodCall");
            }
        } else {
            panic!("Expected ExprStmt");
        }
    }

    #[test]
    fn test_parse_full_custom_matmul() {
        let input = r#"
        fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>, b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
            spawn on(Topology::NPU[0]) {
                let mut result: Tensor = Tensor([a.shape[0], b.shape[1]]).with_memory(Memory::NPU_HBM);
                for i in 0..a.shape[0] {
                    for j in 0..b.shape[1] {
                        result[i][j] = 0;
                        for k in 0..a.shape[1] {
                            result[i][j] += a[i][k] * b[k][j];
                        }
                    }
                }
                return Verified(result);
            }
        }
        "#;
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);
        let func = &program.functions[0];
        assert_eq!(func.name, "custom_matmul");
        if let Statement::SpawnOn(_, stmts, _) = &func.body[0] {
            assert_eq!(stmts.len(), 3); // Let, For, Return
        } else {
            panic!("Expected SpawnOn");
        }
    }

    #[test]
    fn test_parse_struct_and_pointers() {
        let input = r#"
        struct Config {
            value: Tensor<i32>,
            threshold: Tensor<f32>
        }

        fn update_config(c: &mut Config) -> Tensor<Bool> {
            unsafe {
                let ptr: *mut Config = &mut c;
                *ptr = Config { value: 10, threshold: 0.5 };
            }
            return c.value < 20;
        }
        "#;
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();

        assert_eq!(program.structs.len(), 1);
        assert_eq!(program.structs[0].name, "Config");
        assert_eq!(program.structs[0].fields.len(), 2);
        assert_eq!(program.structs[0].fields[0].0, "value");

        assert_eq!(program.functions.len(), 1);
        let func = &program.functions[0];
        assert_eq!(func.name, "update_config");

        // Param should be &mut Config
        let param_ty = &func.params[0].1;
        if let Type::Borrow(inner, None, true) = param_ty {
            if let Type::Struct(s, _) = &**inner {
                assert_eq!(s, "Config");
            } else {
                panic!("Expected Struct");
            }
        } else {
            panic!("Expected Borrow");
        }

        // Body should have unsafe block
        if let Statement::ExprStmt(Expr::UnsafeBlock(stmts, None, _), _) = &func.body[0] {
            assert_eq!(stmts.len(), 2);
        } else {
            panic!("Expected UnsafeBlock");
        }
    }

    #[test]
    fn test_parse_extern() {
        let input = r#"
        extern "C" {
            fn malloc(size: Tensor<i32>) -> *mut Tensor<f32>;
        }
        "#;
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();

        assert_eq!(program.externs.len(), 1);
        assert_eq!(program.externs[0].name, "malloc");
        assert_eq!(program.externs[0].params.len(), 1);
        if let Type::Pointer(inner, None, true) = &program.externs[0].return_type {
            assert_eq!(**inner, Type::Tensor(ElementType::F32, vec![], None));
        } else {
            panic!("Expected pointer return type");
        }
    }
}
