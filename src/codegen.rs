use crate::ast::*;

pub struct MlirGenerator {
    output: String,
    indent_level: usize,
    var_counter: usize,
}

impl Default for MlirGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MlirGenerator {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent_level: 0,
            var_counter: 0,
        }
    }

    fn push_indent(&mut self) {
        self.indent_level += 2;
    }

    fn pop_indent(&mut self) {
        if self.indent_level >= 2 {
            self.indent_level -= 2;
        }
    }

    fn write_line(&mut self, line: &str) {
        let indent = " ".repeat(self.indent_level);
        self.output.push_str(&format!("{}{}\n", indent, line));
    }

    fn next_var(&mut self) -> String {
        let var = format!("%v{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    pub fn generate(&mut self, program: &Program) -> String {
        self.write_line("module {");
        self.push_indent();

        // Hardcode external function declaration for the mock custom_matmul
        self.write_line("func.func private @custom_matmul(memref<?x?xf32, \"NPUHBM\">) -> memref<?x?xf32, \"NPUHBM\">");

        for func in &program.functions {
            self.generate_function(func);
        }

        self.pop_indent();
        self.write_line("}");
        self.output.clone()
    }

    fn lower_type(&self, ty: &Type) -> String {
        match ty {
            Type::Tensor | Type::Matrix => "memref<?x?xf32>".to_string(),
            Type::Ref(inner, mem_space) => {
                let mem_str = match mem_space {
                    MemorySpace::HostDRAM => "\"HostDRAM\"",
                    MemorySpace::NPUHBM => "\"NPUHBM\"",
                    MemorySpace::LocalSRAM => "\"LocalSRAM\"",
                };
                let inner_str = self.lower_type(inner);
                // Simple hack to inject memory space into memref
                if inner_str.starts_with("memref<") && inner_str.ends_with('>') {
                    let inner_content = &inner_str[7..inner_str.len() - 1];
                    format!("memref<{}, {}>", inner_content, mem_str)
                } else {
                    inner_str
                }
            }
            Type::Verified(inner) => self.lower_type(inner),
            Type::Pinned(inner, _) => self.lower_type(inner),
        }
    }

    fn generate_function(&mut self, func: &Function) {
        let mut params_str = Vec::new();
        for (name, ty) in &func.params {
            params_str.push(format!("%{}: {}", name, self.lower_type(ty)));
        }
        
        let ret_str = self.lower_type(&func.return_type);
        
        self.write_line(&format!("func.func @{}({}) -> {} {{", func.name, params_str.join(", "), ret_str));
        self.push_indent();

        for stmt in &func.body {
            self.generate_statement(stmt, &ret_str);
        }

        self.pop_indent();
        self.write_line("}");
    }

    fn generate_statement(&mut self, stmt: &Statement, _current_ret_ty: &str) {
        match stmt {
            Statement::LetDecl(name, expr) => {
                let (val, ty) = self.generate_expr(expr);
                // Assign to named local var (in MLIR, we should use SSA, but for readability we'll map AST names to SSA vars)
                self.write_line(&format!("%{} = \"akar.assign\"({}) : ({}) -> {}", name, val, ty, ty));
            }
            Statement::Return(expr) => {
                let (val, ty) = self.generate_expr(expr);
                self.write_line(&format!("return {} : {}", val, ty)); // Simplified, should cast to current_ret_ty if needed
            }
            Statement::SpawnOn(top, stmts) => {
                let top_str = match top {
                    Topology::NPU(_) => "NPU",
                    Topology::AccCore(_) => "AccCore",
                    _ => "Generic",
                };
                self.write_line(&format!("// --- BEGIN SPAWN ON {} ---", top_str));
                for s in stmts {
                    self.generate_statement(s, _current_ret_ty);
                }
                self.write_line("// --- END SPAWN ---");
            }
            Statement::ExprStmt(expr) => {
                self.generate_expr(expr);
            }
        }
    }

    // Returns (SSA variable name, MLIR type string)
    fn generate_expr(&mut self, expr: &Expr) -> (String, String) {
        match expr {
            Expr::Identifier(name) => {
                // In a real compiler, we'd look up the type in a symtab. 
                // For this demo, we infer it from context or return a generic.
                (format!("%{}", name), "memref<?x?xf32, \"Unknown\">".to_string()) 
            }
            Expr::Number(n) => {
                let res = self.next_var();
                self.write_line(&format!("{} = arith.constant {} : f32", res, n));
                (res, "f32".to_string())
            }
            Expr::Transfer(inner, mem_space) => {
                let (inner_val, inner_ty) = self.generate_expr(inner);
                let mem_str = match mem_space {
                    MemorySpace::HostDRAM => "\"HostDRAM\"",
                    MemorySpace::NPUHBM => "\"NPUHBM\"",
                    MemorySpace::LocalSRAM => "\"LocalSRAM\"",
                };
                
                let out_ty = if inner_ty.starts_with("memref<") && inner_ty.ends_with('>') {
                    let parts: Vec<&str> = inner_ty[7..inner_ty.len()-1].split(',').collect();
                    format!("memref<{}, {}>", parts[0], mem_str)
                } else {
                    format!("memref<?x?xf32, {}>", mem_str) // Fallback
                };

                let res = self.next_var();
                self.write_line(&format!("{} = \"akar.transfer\"({}) {{target_memory = {}}} : ({}) -> {}", res, inner_val, mem_str, inner_ty, out_ty));
                (res, out_ty)
            }
            Expr::FunctionCall(name, args) => {
                let mut arg_vals = Vec::new();
                let mut arg_tys = Vec::new();
                for arg in args {
                    let (val, ty) = self.generate_expr(arg);
                    arg_vals.push(val);
                    arg_tys.push(ty);
                }
                
                let res = self.next_var();
                let ret_ty = "memref<?x?xf32, \"NPUHBM\">"; // Hardcoded for custom_matmul demo
                
                self.write_line(&format!("{} = func.call @{}({}) : ({}) -> {}", res, name, arg_vals.join(", "), arg_tys.join(", "), ret_ty));
                (res, ret_ty.to_string())
            }
        }
    }
}
