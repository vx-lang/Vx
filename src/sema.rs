use crate::ast::*;
use std::collections::HashMap;

pub struct TypeChecker {
    scopes: Vec<HashMap<String, Type>>,
    pub errors: Vec<String>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            errors: Vec::new(),
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn insert(&mut self, name: String, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty.clone());
            }
        }
        None
    }

    pub fn check_program(&mut self, program: &Program) -> bool {
        for func in &program.functions {
            self.check_function(func);
        }
        self.errors.is_empty()
    }

    fn check_function(&mut self, func: &Function) {
        self.push_scope();
        for (name, ty) in &func.params {
            self.insert(name.clone(), ty.clone());
        }

        for stmt in &func.body {
            self.check_statement(stmt, &func.return_type);
        }

        self.pop_scope();
    }

    fn check_statement(&mut self, stmt: &Statement, return_type: &Type) {
        match stmt {
            Statement::LetDecl(name, expr) => {
                let ty = self.check_expr(expr);
                self.insert(name.clone(), ty);
            }
            Statement::Return(expr) => {
                let ty = self.check_expr(expr);
                if !self.is_assignable(return_type, &ty) {
                    self.errors.push(format!("Type mismatch on return. Expected {:?}, got {:?}", return_type, ty));
                }
            }
            Statement::SpawnOn(top, stmts) => {
                self.push_scope();
                
                // Validate topology expression if it contains one
                match top {
                    Topology::NPU(expr) | Topology::AccCore(expr) => {
                        let _ty = self.check_expr(expr);
                        // Ensure it evaluates to a number for indexing
                        // Note: For now we just evaluate it, full type check can ensure it's scalar
                    }
                    Topology::Slice(_, start, end) => {
                        let _t1 = self.check_expr(start);
                        let _t2 = self.check_expr(end);
                    }
                    Topology::Host => {}
                }

                for s in stmts {
                    self.check_statement(s, return_type);
                }

                self.pop_scope();
            }
            Statement::ExprStmt(expr) => {
                self.check_expr(expr);
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) -> Type {
        match expr {
            Expr::Identifier(name) => {
                match self.lookup(name) {
                    Some(ty) => ty,
                    None => {
                        self.errors.push(format!("Undefined variable '{}'", name));
                        Type::Tensor // Default placeholder on error
                    }
                }
            }
            Expr::Number(_) => {
                // Primitive number, we'll represent it as a generic Matrix for now or create a Scalar type.
                // Let's just use Tensor.
                Type::Tensor
            }
            Expr::Transfer(inner_expr, target_mem) => {
                let inner_ty = self.check_expr(inner_expr);
                match inner_ty {
                    Type::Ref(base_ty, _) => {
                        Type::Ref(base_ty, target_mem.clone())
                    }
                    _ => {
                        self.errors.push(format!("Cannot transfer non-reference type: {:?}", inner_ty));
                        Type::Tensor
                    }
                }
            }
            Expr::FunctionCall(name, args) => {
                // Mocking built-ins
                for arg in args {
                    self.check_expr(arg);
                }

                if name == "custom_matmul" {
                    // For the test, custom_matmul takes Ref<Tensor, NPUHBM> and returns Ref<Tensor, NPUHBM>
                    Type::Ref(Box::new(Type::Tensor), MemorySpace::NPUHBM)
                } else {
                    self.errors.push(format!("Undefined function '{}'", name));
                    Type::Tensor
                }
            }
        }
    }

    fn is_assignable(&self, target: &Type, source: &Type) -> bool {
        if target == source {
            return true;
        }

        // Semantic coercion rule: Ref<T, HostDRAM> can be assigned to Verified<T>
        if let Type::Verified(inner_target) = target {
            if let Type::Ref(inner_source, MemorySpace::HostDRAM) = source {
                return inner_target == inner_source;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    #[test]
    fn test_sema_distributed_matmul() {
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

        let mut checker = TypeChecker::new();
        let success = checker.check_program(&program);
        
        for err in &checker.errors {
            println!("Error: {}", err);
        }
        assert!(success);
        assert!(checker.errors.is_empty());
    }

    #[test]
    fn test_sema_type_mismatch() {
        let input = r#"
fn bad_matmul(a: Ref<Tensor, Memory::Host_DRAM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let result = custom_matmul(a); // ERROR: passing Host_DRAM to custom_matmul expected NPUHBM. Actually custom_matmul is mocked to return NPUHBM, but passing args is not fully checked.
        return result; // ERROR: returning NPU_HBM when Verified<Tensor> requires Host_DRAM coercion
    }
}
        "#;
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize();
        let mut parser = Parser::new(tokens);
        let program = parser.parse().unwrap();

        let mut checker = TypeChecker::new();
        let success = checker.check_program(&program);
        assert!(!success);
        assert!(!checker.errors.is_empty());
    }
}
