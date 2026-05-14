import os

files = [
    "src/parser.rs",
    "src/sema.rs",
    "tests/integration_test.rs",
    "tests/compile_test.rs"
]

for file in files:
    if not os.path.exists(file):
        continue
    
    with open(file, 'r') as f:
        content = f.read()
    
    # Fix Parser::new arguments
    content = content.replace("Parser::new(Lexer::new(input).tokenize())", "Parser::new(Lexer::new(input).tokenize(), input)")
    content = content.replace("Parser::new(tokens)", "Parser::new(tokens, input)")
    content = content.replace("Parser::new(lexer.tokenize())", "Parser::new(lexer.tokenize(), input)")
    content = content.replace("Parser::new(Lexer::new(source).tokenize())", "Parser::new(Lexer::new(source).tokenize(), source)")
    
    # Additional test-specific variable names
    content = content.replace("Parser::new(tokens)", "Parser::new(tokens, \"\")") # If no input var
    content = content.replace("Parser::new(tokens, )", "Parser::new(tokens, input)")
    
    # AST specific things for parser tests
    content = content.replace("if let Statement::ExprStmt(expr) = &program.functions[0].body[0] {", "if let Statement::ExprStmt(expr, _) = &program.functions[0].body[0] {")
    content = content.replace("if let Expr::MethodCall(obj, method, args) = expr {", "if let Expr::MethodCall(obj, method, args, _) = expr {")
    content = content.replace("if let Expr::MemberAccess(inner_obj, member) = &**obj {", "if let Expr::MemberAccess(inner_obj, member, _) = &**obj {")
    content = content.replace("Statement::ExprStmt(Expr::UnsafeBlock(stmts, None))", "Statement::ExprStmt(Expr::UnsafeBlock(stmts, None, _), _)")
    
    with open(file, 'w') as f:
        f.write(content)
    
    print(f"Fixed {file}")
