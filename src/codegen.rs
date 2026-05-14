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

        self.pop_indent();
        self.write_line("}");
        self.output.clone()
    }

    fn lower_type(&self, _ty: &Type) -> String {
        // Lower Tensors to dynamic 2D memrefs for now
        "memref<?x?xf32>".to_string()
    }

    fn flatten_indices(&mut self, expr: &Expr) -> Option<(String, Vec<String>)> {
        match expr {
            Expr::IndexAccess(base, idx) => {
                let (base_name, mut indices) = self.flatten_indices(base)?;
                let (idx_val, _) = self.generate_expr(idx, "index");
                indices.push(idx_val);
                Some((base_name, indices))
            }
            Expr::Identifier(name) => Some((format!("%{}", name), Vec::new())),
            _ => None,
        }
    }

    fn generate_function(&mut self, func: &Function) {
        let mut params_str = Vec::new();
        for (name, ty) in &func.params {
            params_str.push(format!("%{}: {}", name, self.lower_type(ty)));
        }

        let ret_str = self.lower_type(&func.return_type);

        self.write_line(&format!(
            "func.func @{}({}) -> {} {{",
            func.name,
            params_str.join(", "),
            ret_str
        ));
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
                let (_val, _ty) = self.generate_expr(expr, "any");
                // For simplistic generation, we just map the returned SSA val to the var name
                // By declaring it with the same name if possible, or aliasing
                if _val.starts_with('%') {
                    // It's already an SSA value, we can create an alias or just assume
                    // the expression codegen handled it. For simplicity in our demo:
                    // If we need a new var name, we'd rename, but let's assume `generate_expr`
                    // leaves it in `_val`. If it's a Tensor alloc, we just let it be.
                }
            }
            Statement::ForLoop(iter, start, end, body) => {
                let (start_val, _) = self.generate_expr(start, "index");
                let (end_val, _) = self.generate_expr(end, "index");
                let step_val = self.next_var();
                self.write_line(&format!("{} = arith.constant 1 : index", step_val));
                self.write_line(&format!(
                    "scf.for %{} = {} to {} step {} {{",
                    iter, start_val, end_val, step_val
                ));
                self.push_indent();
                for s in body {
                    self.generate_statement(s, _current_ret_ty);
                }
                self.pop_indent();
                self.write_line("}");
            }
            Statement::Assign(lhs, rhs) => {
                if let Some((base, indices)) = self.flatten_indices(lhs) {
                    let (rhs_val, _) = self.generate_expr(rhs, "f32");
                    self.write_line(&format!(
                        "memref.store {}, {}[{}] : memref<?x?xf32>",
                        rhs_val,
                        base,
                        indices.join(", ")
                    ));
                }
            }
            Statement::CompoundAssign(lhs, BinaryOp::Add, rhs) => {
                if let Some((base, indices)) = self.flatten_indices(lhs) {
                    let load_val = self.next_var();
                    self.write_line(&format!(
                        "{} = memref.load {}[{}] : memref<?x?xf32>",
                        load_val,
                        base,
                        indices.join(", ")
                    ));
                    let (rhs_val, _) = self.generate_expr(rhs, "f32");
                    let add_val = self.next_var();
                    self.write_line(&format!(
                        "{} = arith.addf {}, {} : f32",
                        add_val, load_val, rhs_val
                    ));
                    self.write_line(&format!(
                        "memref.store {}, {}[{}] : memref<?x?xf32>",
                        add_val,
                        base,
                        indices.join(", ")
                    ));
                }
            }
            Statement::CompoundAssign(_, BinaryOp::Mul, _) => {} // Unused in custom_matmul
            Statement::Return(expr) => {
                let (val, ty) = self.generate_expr(expr, "any");
                self.write_line(&format!("return {} : {}", val, ty));
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
                self.generate_expr(expr, "any");
            }
        }
    }

    // Returns (SSA variable name, MLIR type string)
    fn generate_expr(&mut self, expr: &Expr, expected_ty: &str) -> (String, String) {
        match expr {
            Expr::Identifier(name) => (format!("%{}", name), expected_ty.to_string()),
            Expr::Number(n) => {
                let res = self.next_var();
                if expected_ty == "index" {
                    self.write_line(&format!("{} = arith.constant {} : index", res, *n as i64));
                    (res, "index".to_string())
                } else if expected_ty == "f32" {
                    self.write_line(&format!("{} = arith.constant {} : f32", res, *n));
                    (res, "f32".to_string())
                } else {
                    self.write_line(&format!("{} = arith.constant {} : i64", res, *n as i64));
                    (res, "i64".to_string())
                }
            }
            Expr::Transfer(inner, _mem_space) => {
                // Return inner transparently as no-op for now!
                self.generate_expr(inner, expected_ty)
            }
            Expr::FunctionCall(name, args) => {
                if name == "Tensor" && args.len() == 1 {
                    if let Expr::Array(dims) = &args[0] {
                        let mut dim_vals = Vec::new();
                        for dim in dims {
                            let (d_val, _) = self.generate_expr(dim, "index");
                            dim_vals.push(d_val);
                        }
                        let res = self.next_var();
                        self.write_line(&format!(
                            "{} = memref.alloc({}) : memref<?x?xf32>",
                            res,
                            dim_vals.join(", ")
                        ));
                        return (res, "memref<?x?xf32>".to_string());
                    }
                } else if name == "Verified" {
                    return self.generate_expr(&args[0], expected_ty);
                }

                let mut arg_vals = Vec::new();
                let mut arg_tys = Vec::new();
                for arg in args {
                    let (val, ty) = self.generate_expr(arg, "any");
                    arg_vals.push(val);
                    arg_tys.push(ty);
                }

                let res = self.next_var();
                let ret_ty = "memref<?x?xf32>";

                self.write_line(&format!(
                    "{} = func.call @{}({}) : ({}) -> {}",
                    res,
                    name,
                    arg_vals.join(", "),
                    arg_tys.join(", "),
                    ret_ty
                ));
                (res, ret_ty.to_string())
            }
            Expr::IndexAccess(base, idx) => {
                if let Expr::MemberAccess(inner_base, member) = &**base {
                    if member == "shape" {
                        let (base_val, _) = self.generate_expr(inner_base, "any");
                        let (idx_val, _) = self.generate_expr(idx, "index");
                        let res = self.next_var();
                        self.write_line(&format!(
                            "{} = memref.dim {}, {} : memref<?x?xf32>",
                            res, base_val, idx_val
                        ));
                        return (res, "index".to_string());
                    }
                }

                if let Some((base_name, indices)) = self.flatten_indices(expr) {
                    let res = self.next_var();
                    self.write_line(&format!(
                        "{} = memref.load {}[{}] : memref<?x?xf32>",
                        res,
                        base_name,
                        indices.join(", ")
                    ));
                    (res, "f32".to_string())
                } else {
                    ("".to_string(), "".to_string())
                }
            }
            Expr::MemberAccess(base, member) => {
                if member == "shape" {
                    // Mock shape access by returning a dummy index
                    let res = self.next_var();
                    self.write_line(&format!("{} = arith.constant 4 : index", res));
                    (res, "index".to_string())
                } else {
                    self.generate_expr(base, expected_ty)
                }
            }
            Expr::MethodCall(base, _method, _args) => {
                // e.g. .with_memory()
                self.generate_expr(base, expected_ty)
            }
            Expr::BinaryOp(lhs, op, rhs) => {
                let (lhs_val, _) = self.generate_expr(lhs, "f32");
                let (rhs_val, _) = self.generate_expr(rhs, "f32");
                let res = self.next_var();
                match op {
                    BinaryOp::Add => self.write_line(&format!(
                        "{} = arith.addf {}, {} : f32",
                        res, lhs_val, rhs_val
                    )),
                    BinaryOp::Mul => self.write_line(&format!(
                        "{} = arith.mulf {}, {} : f32",
                        res, lhs_val, rhs_val
                    )),
                }
                (res, "f32".to_string())
            }
            Expr::Array(_) | Expr::MemorySpace(_) | Expr::Topology(_) => {
                let res = self.next_var();
                self.write_line(&format!("{} = arith.constant 0 : i64", res));
                (res, "i64".to_string())
            }
        }
    }
}
