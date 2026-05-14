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
            Err(format!("Error at {}:{}: {}", self.peek().line, self.peek().column, msg))
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
                "Tensor" => Ok(Type::Tensor),
                "Matrix" => Ok(Type::Matrix),
                _ => Err(format!("Unknown type {}", ident)),
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        if self.match_token(&TokenType::Transfer) {
            self.consume(&TokenType::LeftParen, "Expected '(' after transfer")?;
            let inner = self.parse_expr()?;
            self.consume(&TokenType::Comma, "Expected ','")?;
            let mem = self.parse_memory_space()?;
            self.consume(&TokenType::RightParen, "Expected ')'")?;
            Ok(Expr::Transfer(Box::new(inner), mem))
        } else {
            let token = self.advance().clone();
            match token.kind {
                TokenType::Identifier(s) => {
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
                        Ok(Expr::FunctionCall(s, args))
                    } else {
                        Ok(Expr::Identifier(s))
                    }
                }
                TokenType::Number(s) => {
                    let n = s.parse::<f64>().map_err(|_| "Invalid number".to_string())?;
                    Ok(Expr::Number(n))
                }
                _ => Err(format!("Expected expression, found {:?}", token.kind)),
            }
        }
    }

    fn parse_statement(&mut self) -> Result<Statement, String> {
        if self.match_token(&TokenType::Let) {
            let name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err("Expected identifier after let".to_string()),
            };
            self.consume(&TokenType::Equals, "Expected '='")?;
            let expr = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::LetDecl(name, expr))
        } else if self.match_token(&TokenType::Return) {
            let expr = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::Return(expr))
        } else if self.match_token(&TokenType::Spawn) {
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
        } else {
            let expr = self.parse_expr()?;
            self.consume(&TokenType::Semicolon, "Expected ';'")?;
            Ok(Statement::ExprStmt(expr))
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
        
        Ok(Function { name, params, return_type, body })
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
fn distributed_matmul(a: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let local_data = transfer(a, Memory::NPU_HBM);
        let result = custom_matmul(local_data);
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
        assert_eq!(func.params.len(), 1);
        assert_eq!(func.params[0].0, "a");
        
        // Assert return type is Verified<Tensor>
        assert_eq!(func.return_type, Type::Verified(Box::new(Type::Tensor)));
        
        // Assert body has one statement (spawn on)
        assert_eq!(func.body.len(), 1);
        if let Statement::SpawnOn(top, stmts) = &func.body[0] {
            assert_eq!(*top, Topology::NPU(Box::new(Expr::Number(0.0))));
            assert_eq!(stmts.len(), 3);
        } else {
            panic!("Expected SpawnOn statement");
        }
    }
}
