import re

files = ["src/parser.rs", "src/sema.rs", "tests/integration_test.rs", "tests/compile_test.rs"]

for file in files:
    try:
        with open(file, 'r') as f:
            content = f.read()
            
        # Fix match/let patterns (adding `, _` for the span)
        content = re.sub(r'Statement::ExprStmt\(([^,]+)\)', r'Statement::ExprStmt(\1, _)', content)
        content = re.sub(r'Expr::MethodCall\(([^,]+),([^,]+),([^,]+)\)', r'Expr::MethodCall(\1, \2, \3, _)', content)
        content = re.sub(r'Expr::MemberAccess\(([^,]+),([^,]+)\)', r'Expr::MemberAccess(\1, \2, _)', content)
        content = re.sub(r'Expr::IndexAccess\(([^,]+),([^,]+)\)', r'Expr::IndexAccess(\1, \2, _)', content)
        content = re.sub(r'Statement::ForLoop\(([^,]+),([^,]+),([^,]+),([^,]+)\)', r'Statement::ForLoop(\1, \2, \3, \4, _)', content)
        content = re.sub(r'Statement::Assign\(([^,]+),([^,]+)\)', r'Statement::Assign(\1, \2, _)', content)
        content = re.sub(r'Statement::CompoundAssign\(([^,]+),([^,]+),([^,]+)\)', r'Statement::CompoundAssign(\1, \2, \3, _)', content)
        content = re.sub(r'Expr::BinaryOp\(([^,]+),([^,]+),([^,]+)\)', r'Expr::BinaryOp(\1, \2, \3, _)', content)
        content = re.sub(r'Expr::FunctionCall\(([^,]+),([^,]+)\)', r'Expr::FunctionCall(\1, \2, _)', content)
        content = re.sub(r'Expr::Array\(([^,]+)\)', r'Expr::Array(\1, _)', content)
        
        # Fix constructors (adding `, Span::default()`)
        content = re.sub(r'Expr::Number\(([^,]+)\)', r'Expr::Number(\1, Span::default())', content)
        content = re.sub(r'Expr::Identifier\("([^"]+)"\.to_string\(\)\)', r'Expr::Identifier("\1".to_string(), Span::default())', content)
        
        # Parser::new fixes (we already did this in previous script, but run again just in case)
        content = content.replace("Parser::new(tokens, \"\")", "Parser::new(tokens, input)")
        
        with open(file, 'w') as f:
            f.write(content)
            
    except Exception as e:
        pass

print("Done")
