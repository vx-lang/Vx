//===- decl.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// Parser for Vx declarations, including functions, structs, impl blocks, and foreign modules.
//
//===----------------------------------------------------------------------===//

use super::*;

impl<'a> Parser<'a> {
    pub(crate) fn parse_generic_params(&mut self) -> Result<Vec<GenericParam>, String> {
        let mut generics = Vec::new();
        if self.match_token(&TokenType::LeftAngle) {
            while !self.check(&TokenType::RightAngle) && !self.check(&TokenType::Eof) {
                if self.check(&TokenType::Identifier("const".to_string())) {
                    self.advance(); // consume const
                    let name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected const parameter name")),
                    };
                    self.consume(&TokenType::Colon, "Expected ':' after const parameter name")?;
                    let ty = self.parse_type()?;
                    self.generic_params.push(name.clone());
                    generics.push(GenericParam::Const { name, ty });
                } else {
                    let name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected generic parameter name")),
                    };
                    self.generic_params.push(name.clone());
                    let mut bound = None;
                    if self.match_token(&TokenType::Colon) {
                        bound = match self.advance().kind.clone() {
                            TokenType::Identifier(s) => Some(s),
                            _ => return Err(self.error("Expected trait bound identifier")),
                        };
                    }
                    generics.push(GenericParam::Type { name, bound });
                }
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
            _ => return Err(self.error("Expected function name")),
        };

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftParen, "Expected '(' after function name")?;
        let mut params = Vec::new();
        if !self.check(&TokenType::RightParen) {
            loop {
                let p_name = match self.advance().kind.clone() {
                    TokenType::Identifier(s) => s,
                    _ => return Err(self.error("Expected parameter name")),
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

        let mut topology = Topology::Host;
        if self.match_token(&TokenType::On) {
            topology = self.parse_topology()?;
        }

        if !self.match_token(&TokenType::Arrow) {
            // In Vx, '->' is currently required for functions in parse_function
            return Err(self.error("Expected '->'"));
        }
        let return_type = self.parse_type()?;

        let mut requires = Vec::new();
        while self.match_token(&TokenType::Requires) {
            requires.push(self.parse_expr()?);
        }

        let mut ensures = Vec::new();
        while self.match_token(&TokenType::Ensures) {
            ensures.push(self.parse_expr()?);
        }

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut body = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            body.push(self.parse_statement()?);
        }

        // Transform implicit return
        if let Some(Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi,
            span,
        })) = body.last().cloned()
        {
            if !has_semi {
                let last_idx = body.len() - 1;
                body[last_idx] = Statement::Return(ReturnStmt { expr, span });
            }
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
            topology,
            return_type,
            requires,
            ensures,
            body,
        })
    }

    pub(crate) fn parse_struct_decl(&mut self) -> Result<StructDecl, String> {
        self.consume(&TokenType::Struct, "Expected 'struct'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected struct name")),
        };

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut fields = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let f_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err(self.error("Expected field name")),
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

    pub(crate) fn parse_enum_decl(&mut self) -> Result<EnumDecl, String> {
        self.consume(&TokenType::Enum, "Expected 'enum'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected enum name")),
        };

        let generics = self.parse_generic_params()?;

        self.consume(&TokenType::LeftBrace, "Expected '{'")?;
        let mut variants = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            let v_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err(self.error("Expected enum variant name")),
            };

            let mut payload = None;
            if self.match_token(&TokenType::LeftParen) {
                let mut types = Vec::new();
                if !self.check(&TokenType::RightParen) {
                    loop {
                        types.push(self.parse_type()?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                }
                self.consume(&TokenType::RightParen, "Expected ')' after enum payload")?;
                payload = Some(types);
            }

            variants.push((v_name, payload));

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightBrace, "Expected '}'")?;

        // Remove generic params from scope
        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(EnumDecl {
            name,
            generics,
            variants,
        })
    }

    pub(crate) fn parse_extern_block(&mut self) -> Result<Vec<ExternDecl>, String> {
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
                _ => return Err(self.error("Expected function name")),
            };

            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let mut params = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    let p_name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected parameter name")),
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

    pub(crate) fn parse_trait_decl(&mut self) -> Result<TraitDecl, String> {
        self.consume(&TokenType::Trait, "Expected 'trait'")?;
        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected trait name")),
        };
        let generics = self.parse_generic_params()?;
        self.consume(&TokenType::LeftBrace, "Expected '{'")?;

        let mut methods = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            self.consume(&TokenType::Fn, "Expected 'fn' in trait")?;
            let method_name = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err(self.error("Expected method name")),
            };
            self.consume(&TokenType::LeftParen, "Expected '('")?;
            let mut params = Vec::new();
            if !self.check(&TokenType::RightParen) {
                loop {
                    let p_name = match self.advance().kind.clone() {
                        TokenType::Identifier(s) => s,
                        _ => return Err(self.error("Expected parameter name")),
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

        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(TraitDecl {
            name,
            generics,
            methods,
        })
    }

    pub(crate) fn parse_impl_block(&mut self) -> Result<ImplBlock, String> {
        self.consume(&TokenType::Impl, "Expected 'impl'")?;

        let generics = self.parse_generic_params()?;

        // Either `impl Trait for Type` or `impl Type`
        let mut trait_name = None;
        let target_type;

        // Since we don't have lookahead to distinguish `impl Trait for Type` from `impl Type`,
        // if we see `Identifier` followed by `for`, it's a trait. Otherwise it's a type.
        // Note: parse_type handles `Struct(name)`, which is an identifier!
        // We can just peek ahead.
        let parsed_type = self.parse_type()?;
        if self.check(&TokenType::For) {
            self.advance(); // consume 'for'
            if let Type::Struct(name, _) = parsed_type {
                trait_name = Some(name);
            } else if let Type::GenericInstance(inner, _) = parsed_type {
                if let Type::Struct(name, _) = *inner {
                    trait_name = Some(name);
                } else {
                    return Err(self.error("Expected trait name before 'for'"));
                }
            } else {
                return Err(self.error("Expected trait name before 'for'"));
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

        for _ in 0..generics.len() {
            self.generic_params.pop();
        }

        Ok(ImplBlock {
            generics,
            trait_name,
            target_type,
            methods,
        })
    }

    pub(crate) fn parse_import_decl(&mut self) -> Result<ImportDecl, String> {
        self.consume(&TokenType::Import, "Expected 'import'")?;
        let mut path = Vec::new();
        loop {
            let ident = match self.advance().kind.clone() {
                TokenType::Identifier(s) => s,
                _ => return Err(self.error("Expected identifier in import path")),
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

    pub(crate) fn parse_macro_def(&mut self) -> Result<MacroDefDecl, String> {
        let span_start = self.peek().clone();
        self.consume(&TokenType::MacroRules, "Expected 'macro_rules!'")?;

        let name = match self.advance().kind.clone() {
            TokenType::Identifier(s) => s,
            _ => return Err(self.error("Expected macro name")),
        };

        self.consume(&TokenType::LeftBrace, "Expected '{' for macro rules body")?;

        let mut rules = Vec::new();
        while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
            // Matcher: ( ... )
            self.consume(&TokenType::LeftParen, "Expected '(' for macro matcher")?;
            let mut matcher = Vec::new();
            while !self.check(&TokenType::RightParen) && !self.check(&TokenType::Eof) {
                matcher.push(self.parse_token_tree()?);
            }
            self.consume(&TokenType::RightParen, "Expected ')' for macro matcher")?;

            self.consume(&TokenType::FatArrow, "Expected '=>' after macro matcher")?;

            // Transcriber: { ... }
            self.consume(&TokenType::LeftBrace, "Expected '{' for macro transcriber")?;
            let mut transcriber = Vec::new();
            while !self.check(&TokenType::RightBrace) && !self.check(&TokenType::Eof) {
                transcriber.push(self.parse_token_tree()?);
            }
            self.consume(&TokenType::RightBrace, "Expected '}' for macro transcriber")?;

            // Optional semicolon after a rule
            if self.check(&TokenType::Semicolon) {
                self.advance();
            }

            rules.push(MacroRule {
                matcher,
                transcriber,
            });
        }
        self.consume(
            &TokenType::RightBrace,
            "Expected '}' after macro rules body",
        )?;

        Ok(MacroDefDecl {
            name,
            rules,
            span: Span {
                line: span_start.line,
                column: span_start.column,
                length: 0,
            },
        })
    }

    pub fn parse(&mut self) -> Result<Program, String> {
        let mut imports = Vec::new();
        let mut externs = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut functions = Vec::new();
        let mut macros = Vec::new();
        while !self.check(&TokenType::Eof) {
            if self.check(&TokenType::Import) {
                imports.push(self.parse_import_decl()?);
            } else if self.check(&TokenType::MacroRules) {
                macros.push(self.parse_macro_def()?);
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
                return Err(self.error(&format!(
                    "Unexpected token at program root: {:?}",
                    self.peek().kind
                )));
            }
        }
        Ok(Program {
            module_path: self.source.to_string(), // Default fallback, should be overridden by pipeline
            imports,
            macros,
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
        if let Statement::Return(ReturnStmt {
            expr:
                Expr::SpawnOn(SpawnOnExpr {
                    top,
                    stmts,
                    ret: _,
                    span: _,
                }),
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(
                *top,
                Topology::NPU(Box::new(Expr::Number(NumberExpr {
                    value: "0".to_string(),
                    ty: Some(ElementType::I32),
                    span: Span::default()
                })))
            );
            assert_eq!(stmts.len(), 4);
        } else {
            panic!("Expected SpawnOn statement, got {:#?}", &func.body[0]);
        }
    }

    #[test]
    fn test_parse_let_mut_with_type() {
        let input = "fn main() -> Tensor { let mut x: Tensor = Tensor([1, 2]); }";
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();
        let func = &program.functions[0];
        if let Statement::LetDecl(LetDeclStmt {
            name,
            is_mut,
            ty_ann: ty,
            expr,
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(name, "x");
            assert!(is_mut);
            assert_eq!(ty, &Some(Type::Tensor(ElementType::F32, vec![], None)));
            if let Expr::FunctionCall(FunctionCallExpr {
                name: func_name,
                args,
                span: _,
            }) = expr
            {
                assert_eq!(func_name, "Tensor");
                assert_eq!(args.len(), 1);
                if let Expr::Array(ArrayExpr { elements, span: _ }) = &args[0] {
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
        if let Statement::ForLoop(ForLoopStmt {
            iter,
            iterable,
            body,
            invariants: _,
            span: _,
        }) = &program.functions[0].body[0]
        {
            assert_eq!(iter, "i");
            if let Expr::Range(RangeExpr {
                start,
                end,
                span: _,
            }) = &**iterable
            {
                assert_eq!(
                    **start,
                    Expr::Number(NumberExpr {
                        value: "0".to_string(),
                        ty: Some(ElementType::I32),
                        span: Span::default()
                    })
                );
                assert_eq!(
                    **end,
                    Expr::Number(NumberExpr {
                        value: "10".to_string(),
                        ty: Some(ElementType::I32),
                        span: Span::default()
                    })
                );
            } else {
                panic!("Expected Range expression for iterable");
            }
            assert_eq!(body.len(), 1);
            if let Statement::Assign(AssignStmt { lhs, rhs, span: _ }) = &body[0] {
                assert_eq!(
                    *lhs,
                    Expr::Identifier(IdentifierExpr {
                        name: "x".to_string(),
                        span: Span::default()
                    })
                );
                assert_eq!(
                    *rhs,
                    Expr::Number(NumberExpr {
                        value: "5".to_string(),
                        ty: Some(ElementType::I32),
                        span: Span::default()
                    })
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
        if let Statement::CompoundAssign(CompoundAssignStmt {
            lhs,
            op,
            rhs,
            span: _,
        }) = &program.functions[0].body[0]
        {
            assert_eq!(*op, BinaryOp::Add);
            if let Expr::IndexAccess(IndexAccessExpr {
                base: arr,
                index: idx,
                span: _,
            }) = lhs
            {
                assert_eq!(
                    **arr,
                    Expr::Identifier(IdentifierExpr {
                        name: "x".to_string(),
                        span: Span::default()
                    })
                );
                assert_eq!(
                    **idx,
                    Expr::Number(NumberExpr {
                        value: "0".to_string(),
                        ty: Some(ElementType::I32),
                        span: Span::default()
                    })
                );
            } else {
                panic!("Expected IndexAccess");
            }

            if let Expr::BinaryOp(BinaryOpExpr {
                lhs: left,
                op: binop,
                rhs: right,
                span: _,
            }) = rhs
            {
                assert_eq!(*binop, BinaryOp::Mul);
                assert_eq!(
                    **left,
                    Expr::Identifier(IdentifierExpr {
                        name: "y".to_string(),
                        span: Span::default()
                    })
                );
                assert_eq!(
                    **right,
                    Expr::Identifier(IdentifierExpr {
                        name: "z".to_string(),
                        span: Span::default()
                    })
                );
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
        if let Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi: _,
            span: _,
        }) = &program.functions[0].body[0]
        {
            if let Expr::MethodCall(MethodCallExpr {
                base: obj,
                method_name: method,
                args,
                span: _,
            }) = expr
            {
                assert_eq!(method, "with_memory");
                assert_eq!(args.len(), 1);
                if let Expr::MemberAccess(MemberAccessExpr {
                    base: inner_obj,
                    member,
                    struct_name: _,
                    span: _,
                }) = &**obj
                {
                    assert_eq!(member, "shape");
                    assert_eq!(
                        **inner_obj,
                        Expr::Identifier(IdentifierExpr {
                            name: "x".to_string(),
                            span: Span::default()
                        })
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
        if let Statement::Return(ReturnStmt {
            expr:
                Expr::SpawnOn(SpawnOnExpr {
                    top: _,
                    stmts,
                    ret: _,
                    span: _,
                }),
            span: _,
        }) = &func.body[0]
        {
            assert_eq!(stmts.len(), 3); // Let, For, Return
        } else {
            panic!("Expected SpawnOn, got {:#?}", &func.body[0]);
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
        if let Type::Borrow(inner, None, true, _) = param_ty {
            if let Type::Struct(s, _) = &**inner {
                assert_eq!(s, "Config");
            } else {
                panic!("Expected Struct");
            }
        } else {
            panic!("Expected Borrow");
        }

        // Body should have unsafe block
        if let Statement::ExprStmt(ExprStmtStmt {
            expr:
                Expr::UnsafeBlock(UnsafeBlockExpr {
                    stmts,
                    ret: None,
                    span: _,
                }),
            has_semi: _,
            span: _,
        }) = &func.body[0]
        {
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

    #[test]
    fn test_parse_implicit_return() {
        let input = r#"
fn stdout_write(buffer: *const u8, len: i64) -> i64 {
    vx_stdout_write(buffer, len)
}

fn stderr_write(buffer: *const u8, len: i64) -> i64 {
    vx_stderr_write(buffer, len);
}
        "#;
        let mut parser = Parser::new(Lexer::new(input).tokenize(), input);
        let program = parser.parse().unwrap();

        assert_eq!(program.functions.len(), 2);

        // stdout_write should have a Return statement (implicit return converted)
        let func1 = &program.functions[0];
        if let Statement::Return(ReturnStmt { expr, span: _ }) = &func1.body[0] {
            if let Expr::FunctionCall(FunctionCallExpr {
                name,
                args,
                span: _,
            }) = expr
            {
                assert_eq!(name, "vx_stdout_write");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected FunctionCall in implicit return");
            }
        } else {
            panic!("Expected Return statement from implicit return");
        }

        // stderr_write should have an ExprStmt with has_semicolon = true
        let func2 = &program.functions[1];
        if let Statement::ExprStmt(ExprStmtStmt {
            expr,
            has_semi: has_semicolon,
            span: _,
        }) = &func2.body[0]
        {
            assert!(*has_semicolon);
            if let Expr::FunctionCall(FunctionCallExpr {
                name,
                args,
                span: _,
            }) = expr
            {
                assert_eq!(name, "vx_stderr_write");
                assert_eq!(args.len(), 2);
            } else {
                panic!("Expected FunctionCall in ExprStmt");
            }
        } else {
            panic!("Expected ExprStmt with semicolon");
        }
    }
}
