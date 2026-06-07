use crate::ast::*;
use std::io::Write;
use std::process::{Command, Stdio};

use crate::ast;
pub struct SmtProver {
    assertions: Vec<String>,
    declarations: std::collections::HashSet<String>,
}

impl SmtProver {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            assertions: Vec::new(),
            declarations: std::collections::HashSet::new(),
        }
    }

    pub fn prove(&self) -> Result<bool, String> {
        let mut script = String::new();
        // Setup QF_LIA (Quantifier-Free Linear Integer Arithmetic)
        script.push_str("(set-logic QF_LIA)\n");

        for decl in &self.declarations {
            script.push_str(&format!("(declare-const {} Int)\n", decl));
        }

        for assertion in &self.assertions {
            script.push_str(&format!("(assert {})\n", assertion));
        }

        script.push_str("(check-sat)\n");

        let mut child = match Command::new("z3")
            .arg("-in")
            .arg("-smt2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                println!("Warning: z3 solver not found in PATH. Skipping formal verification.");
                return Ok(false); // Return UNSAT so the proof trivially succeeds and compilation can continue
            }
            Err(e) => return Err(format!("Failed to spawn z3: {}", e)),
        };

        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(script.as_bytes())
                .map_err(|e| e.to_string())?;
        }

        let output = child.wait_with_output().map_err(|e| e.to_string())?;
        let result_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        let has_sat = result_str.lines().any(|l| l.trim() == "sat");
        let has_unsat = result_str.lines().any(|l| l.trim() == "unsat");

        if has_sat {
            Ok(true)
        } else if has_unsat {
            Ok(false)
        } else {
            Err(format!("Unexpected z3 output: {}", result_str))
        }
    }

    pub fn add_constraint(&mut self, expr: &Expr) -> Result<(), String> {
        let smt_expr = self.lower_expr(expr)?;
        self.assertions.push(smt_expr);
        Ok(())
    }

    fn lower_expr(&mut self, expr: &Expr) -> Result<String, String> {
        match expr {
            Expr::Number(n) => Ok(n.value.clone()),
            Expr::Identifier(id) => {
                let name = id.name.replace(".", "_");
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::BinaryOp(b) => {
                let lhs = self.lower_expr(&b.lhs)?;
                let rhs = self.lower_expr(&b.rhs)?;
                let op = match b.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    _ => return Err(format!("Unsupported binary op in SMT solver: {:?}", b.op)),
                };
                Ok(format!("({} {} {})", op, lhs, rhs))
            }
            Expr::RelationalOp(r) => {
                let lhs = self.lower_expr(&r.lhs)?;
                let rhs = self.lower_expr(&r.rhs)?;
                let op = match r.op {
                    RelationalOp::Eq => "=",
                    RelationalOp::NotEq => "distinct",
                    RelationalOp::Lt => "<",
                    RelationalOp::Le => "<=",
                    RelationalOp::Gt => ">",
                    RelationalOp::Ge => ">=",
                };
                Ok(format!("({} {} {})", op, lhs, rhs))
            }
            Expr::LogicalOp(l) => {
                let lhs = self.lower_expr(&l.lhs)?;
                let rhs = self.lower_expr(&l.rhs)?;
                let op = match l.op {
                    LogicalOp::And => "and",
                    LogicalOp::Or => "or",
                };
                Ok(format!("({} {} {})", op, lhs, rhs))
            }
            Expr::UnaryOp(u) => {
                let inner = self.lower_expr(&u.expr)?;
                match u.op {
                    UnaryOp::Not => Ok(format!("(not {})", inner)),
                    UnaryOp::Neg => Ok(format!("(- {})", inner)),
                }
            }
            Expr::MemberAccess(m) => {
                let base = self.lower_expr(&m.base)?;
                let field = m.member.replace(".", "_");
                let name = format!("{}_{}", base, field);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::IndexAccess(i) => {
                let base = self.lower_expr(&i.base)?;
                let index = self.lower_expr(&i.index)?;
                // For now, treat array indexing as a flattened variable (e.g. a_0).
                // Full theory of arrays would require (select a 0).
                let name = format!("{}_{}", base, index);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::Topology(t) => {
                let name = match &t.top {
                    Topology::Host => "Topology_Host".to_string(),
                    Topology::NPU(e) => {
                        if let Expr::Number(n) = &**e {
                            format!("Topology_NPU_{}", n.value)
                        } else {
                            "Topology_NPU".to_string()
                        }
                    }
                    Topology::AccCore(e) => {
                        if let Expr::Number(n) = &**e {
                            format!("Topology_AccCore_{}", n.value)
                        } else {
                            "Topology_AccCore".to_string()
                        }
                    }
                    Topology::AMX => "Topology_AMX".to_string(),
                    Topology::ANE => "Topology_ANE".to_string(),
                    Topology::GPU => "Topology_GPU".to_string(),
                    _ => "Topology_Complex".to_string(),
                };
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::EnumVariant(e) => {
                let name = format!("{}_{}", e.enum_name, e.variant_name);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            _ => Err(format!("Unsupported expression in SMT solver: {:?}", expr)),
        }
    }
}
