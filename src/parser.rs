use crate::ast::*;
use crate::lexer::{Token, TokenType};

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
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
            Err(format!(
                "Error at {}:{}: {}",
                self.peek().line,
                self.peek().column,
                msg
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
        if self.match_token(&TokenType::Ref) {
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
                            _ => return Err(format!("Unknown element type {}", ty_ident)),
                        };
                        match self.advance().kind {
                            TokenType::RightAngle => {}
                            _ => return Err("Expected '>' after element type".to_string()),
                        }
                    }
                    Ok(Type::Tensor(el_ty))
                }
                "Matrix" => Ok(Type::Matrix),
                _ => Err(format!("Unknown type {}", ident)),
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
                TokenType::Plus => BinaryOp::Add,
                TokenType::Star => BinaryOp::Mul,
                _ => return Err("Unknown binary operator".to_string()),
            };
            let right = self.parse_binary_expr(op_prec + 1)?;
            left = Expr::BinaryOp(Box::new(left), op, Box::new(right));
        }

        Ok(left)
    }

    fn get_operator_precedence(&self, kind: &TokenType) -> Option<u8> {
        match kind {
            TokenType::Plus => Some(10),
            TokenType::Star => Some(20),
            _ => None,
        }
    }

    fn parse_primary_expr(&mut self) -> Result<Expr, String> {
        let mut expr = if self.match_token(&TokenType::Transfer) {
            self.consume(&TokenType::LeftParen, "Expected '(' after transfer")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let mem = self.parse_memory_space()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::Transfer(Box::new(inner), mem)
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
            Expr::Array(elements)
        } else if self.check(&TokenType::Memory) {
            Expr::MemorySpace(self.parse_memory_space()?)
        } else if self.check(&TokenType::Topology) {
            Expr::Topology(self.parse_topology()?)
        } else if self.check(&TokenType::Verified) {
            self.advance();
            self.consume(&TokenType::LeftParen, "Expected '(' after Verified")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Expr::FunctionCall("Verified".to_string(), vec![inner])
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
                        Expr::FunctionCall(call_name, args)
                    } else {
                        Expr::Identifier(call_name)
                    }
                }
                TokenType::Number(s) => {
                    let n = s.parse::<f64>().map_err(|_| "Invalid number".to_string())?;
                    Expr::Number(n)
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
                    expr = Expr::MethodCall(Box::new(expr), ident, args);
                } else {
                    expr = Expr::MemberAccess(Box::new(expr), ident);
                }
            } else if self.match_token(&TokenType::LeftBracket) {
                let index = self.parse_expr()?;
                self.consume(&TokenType::RightBracket, "Expected ']'")?;
                expr = Expr::IndexAccess(Box::new(expr), Box::new(index));
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
                Ok(Statement::LetDecl(name, is_mut, type_annotation, expr))
            }
            TokenType::Return => {
                self.advance();
                let expr = self.parse_expr()?;
                self.consume(&TokenType::Semicolon, "Expected ';'")?;
                Ok(Statement::Return(expr))
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
                Ok(Statement::SpawnOn(top, stmts))
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
                ))
            }
            _ => {
                let expr = self.parse_expr()?;
                if self.match_token(&TokenType::Equals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::Assign(expr, rhs))
                } else if self.match_token(&TokenType::PlusEquals) {
                    let rhs = self.parse_expr()?;
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::CompoundAssign(expr, BinaryOp::Add, rhs))
                } else {
                    self.consume(&TokenType::Semicolon, "Expected ';'")?;
                    Ok(Statement::ExprStmt(expr))
                }
            }
        }
    }

    fn parse_function(&mut self) -> Result<Function, String> {
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

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            body.push(self.parse_statement()?);
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        Ok(Function {
            name,
            params,
            return_type,
            body,
        })
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut functions = Vec::new();
        while !self.check(&TokenType::Eof) {
            functions.push(self.parse_function()?);
        }
        Ok(Program { functions })
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
        let mut parser = Parser::new(tokens);
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
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);

        let func = &program.functions[0];
        assert_eq!(func.name, "distributed_matmul");
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.params[0].0, "a");

        // Assert return type is Verified<Tensor>
        assert_eq!(
            func.return_type,
            Type::Verified(Box::new(Type::Tensor(ElementType::F32)))
        );

        // Assert body has one statement (spawn on)
        assert_eq!(func.body.len(), 1);
        if let Statement::SpawnOn(top, stmts) = &func.body[0] {
            assert_eq!(*top, Topology::NPU(Box::new(Expr::Number(0.0))));
            assert_eq!(stmts.len(), 4);
        } else {
            panic!("Expected SpawnOn statement");
        }
    }

    #[test]
    fn test_parse_let_mut_with_type() {
        let input = "fn main() -> Tensor { let mut x: Tensor = Tensor([1, 2]); }";
        let mut parser = Parser::new(Lexer::new(input).tokenize());
        let program = parser.parse().unwrap();
        let func = &program.functions[0];
        if let Statement::LetDecl(name, is_mut, ty, expr) = &func.body[0] {
            assert_eq!(name, "x");
            assert!(is_mut);
            assert_eq!(ty, &Some(Type::Tensor(ElementType::F32)));
            if let Expr::FunctionCall(func_name, args) = expr {
                assert_eq!(func_name, "Tensor");
                assert_eq!(args.len(), 1);
                if let Expr::Array(elements) = &args[0] {
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
        let mut parser = Parser::new(Lexer::new(input).tokenize());
        let program = parser.parse().unwrap();
        if let Statement::ForLoop(iter, start, end, body) = &program.functions[0].body[0] {
            assert_eq!(iter, "i");
            assert_eq!(**start, Expr::Number(0.0));
            assert_eq!(**end, Expr::Number(10.0));
            assert_eq!(body.len(), 1);
            if let Statement::Assign(lhs, rhs) = &body[0] {
                assert_eq!(*lhs, Expr::Identifier("x".to_string()));
                assert_eq!(*rhs, Expr::Number(5.0));
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
        let mut parser = Parser::new(Lexer::new(input).tokenize());
        let program = parser.parse().unwrap();
        if let Statement::CompoundAssign(lhs, op, rhs) = &program.functions[0].body[0] {
            assert_eq!(*op, BinaryOp::Add);
            if let Expr::IndexAccess(arr, idx) = lhs {
                assert_eq!(**arr, Expr::Identifier("x".to_string()));
                assert_eq!(**idx, Expr::Number(0.0));
            } else {
                panic!("Expected IndexAccess");
            }

            if let Expr::BinaryOp(left, binop, right) = rhs {
                assert_eq!(*binop, BinaryOp::Mul);
                assert_eq!(**left, Expr::Identifier("y".to_string()));
                assert_eq!(**right, Expr::Identifier("z".to_string()));
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
        let mut parser = Parser::new(Lexer::new(input).tokenize());
        let program = parser.parse().unwrap();
        if let Statement::ExprStmt(expr) = &program.functions[0].body[0] {
            if let Expr::MethodCall(obj, method, args) = expr {
                assert_eq!(method, "with_memory");
                assert_eq!(args.len(), 1);
                if let Expr::MemberAccess(inner_obj, member) = &**obj {
                    assert_eq!(member, "shape");
                    assert_eq!(**inner_obj, Expr::Identifier("x".to_string()));
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
        let mut parser = Parser::new(Lexer::new(input).tokenize());
        let program = parser.parse().unwrap();
        assert_eq!(program.functions.len(), 1);
        let func = &program.functions[0];
        assert_eq!(func.name, "custom_matmul");
        if let Statement::SpawnOn(_, stmts) = &func.body[0] {
            assert_eq!(stmts.len(), 3); // Let, For, Return
        } else {
            panic!("Expected SpawnOn");
        }
    }
}
