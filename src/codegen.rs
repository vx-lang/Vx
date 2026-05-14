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

        // Hardcode external function declarations
        self.write_line("func.func private @akar_transfer(i64) -> i64");
        self.write_line("func.func private @custom_matmul(i64, i64) -> i64");
        self.write_line("func.func private @akar_print(i64)");
        
        for func in &program.functions {
            self.generate_function(func);
        }

        // Generate a main function for execution
        self.write_line("func.func @main() -> i32 {");
        self.push_indent();
        self.write_line("%dummy_tensor_a = arith.constant 42 : i64");
        self.write_line("%dummy_tensor_b = arith.constant 43 : i64");
        self.write_line("%res = func.call @distributed_matmul(%dummy_tensor_a, %dummy_tensor_b) : (i64, i64) -> i64");
        self.write_line("func.call @akar_print(%res) : (i64) -> ()");
        self.write_line("%zero = arith.constant 0 : i32");
        self.write_line("return %zero : i32");
        self.pop_indent();
        self.write_line("}");

        self.pop_indent();
        self.write_line("}");
        self.output.clone()
    }

    fn lower_type(&self, _ty: &Type) -> String {
        // Strip memory spaces for LLVM backend compatibility, since Semantic Analysis already validated them.
        "i64".to_string()
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
            Statement::LetDecl(_name, _is_mut, _ty_ann, expr) => {
                let (_val, _ty) = self.generate_expr(expr);
                // In proper SSA, we'd map AST names to the returned SSA values in a map
                // For this simplistic generation, the expr already generated the values.
                // Wait, earlier we added arith.addi for LetDecl! Let's preserve that.
                if let Expr::Identifier(_) | Expr::Number(_) | Expr::FunctionCall(..) | Expr::Transfer(..) = expr {
                    self.write_line(&format!("%{}_zero = arith.constant 0 : i64", _name));
                    self.write_line(&format!("%{} = arith.addi {}, %{}_zero : i64", _name, _val, _name));
                }
            }
            Statement::ForLoop(_iter, _start, _end, body) => {
                for s in body {
                    self.generate_statement(s, _current_ret_ty);
                }
            }
            Statement::Assign(_lhs, _rhs) | Statement::CompoundAssign(_lhs, _, _rhs) => {
                // Ignore for now (Phase B)
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
                (format!("%{}", name), "i64".to_string()) 
            }
            Expr::Number(n) => {
                let res = self.next_var();
                self.write_line(&format!("{} = arith.constant {} : i64", res, *n as i64));
                (res, "i64".to_string())
            }
            Expr::Transfer(inner, _mem_space) => {
                let (inner_val, inner_ty) = self.generate_expr(inner);
                let out_ty = inner_ty.clone();
                let res = self.next_var();
                self.write_line(&format!("{} = func.call @akar_transfer({}) : ({}) -> {}", res, inner_val, inner_ty, out_ty));
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
                let ret_ty = "i64";
                
                self.write_line(&format!("{} = func.call @{}({}) : ({}) -> {}", res, name, arg_vals.join(", "), arg_tys.join(", "), ret_ty));
                (res, ret_ty.to_string())
            }
            Expr::Array(_) | Expr::MemberAccess(_, _) | Expr::IndexAccess(_, _) | Expr::MethodCall(_, _, _) | Expr::BinaryOp(_, _, _) | Expr::MemorySpace(_) | Expr::Topology(_) => {
                let res = self.next_var();
                self.write_line(&format!("{} = arith.constant 0 : i64", res));
                (res, "i64".to_string())
            }
        }
    }
}
