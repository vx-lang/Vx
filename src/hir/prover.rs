//===- prover.rs - Vx Compiler -------------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Formal verification prover integration. Handles symbolic execution and Z3 validation.
//
//===----------------------------------------------------------------------===//

use crate::syntax::*;
use std::io::Write;
use std::process::{Command, Stdio};

/// The type of a name in a contract, when it is an integer, and whether its range is a fact.
pub type IntTypeOf<'a> = &'a dyn Fn(&str) -> Option<(ElementType, bool)>;

pub struct SmtProver {
    assertions: Vec<String>,
    declarations: std::collections::HashSet<String>,
    /// The integer type of each declared name whose range is a fact.
    ranges: std::collections::HashMap<String, ElementType>,
}

impl SmtProver {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            assertions: Vec::new(),
            declarations: std::collections::HashSet::new(),
            ranges: std::collections::HashMap::new(),
        }
    }

    pub fn prove(&self) -> Result<bool, String> {
        let mut script = String::new();
        // Setup QF_LIA (Quantifier-Free Linear Integer Arithmetic)
        script.push_str("(set-logic QF_LIA)\n");

        for decl in &self.declarations {
            script.push_str(&format!("(declare-const {} Int)\n", decl));
            if let Some((lo, hi)) = self.ranges.get(decl).and_then(int_range) {
                script.push_str(&format!("(assert (<= {lo} {decl} {hi}))\n"));
            }
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
                // Used to return Ok(false) -- UNSAT -- so the proof "trivially succeeded" and
                // compilation continued. That made a missing solver indistinguishable from a
                // discharged obligation. These prover callers do not read VX_ALLOW_UNVERIFIED,
                // so the message must not advertise that escape hatch.
                return Err(crate::hir::solver::prover_missing_message(&e.to_string()));
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

    /// Add `expr` as a fact. `type_of` gives the integer type of a name, so that the prover knows
    /// the width of an operation on it, and its range.
    pub fn add_constraint(&mut self, expr: &Expr, type_of: IntTypeOf) -> Result<(), String> {
        let smt_expr = self.lower_expr(expr, type_of, None)?;
        self.assertions.push(smt_expr);
        Ok(())
    }

    /// `expected` is the integer type that the context gives `expr`: in a contract, which the
    /// checker does not type, `return == p.x + p.y` adds at the width of `return`.
    fn lower_expr(
        &mut self,
        expr: &Expr,
        type_of: IntTypeOf,
        expected: Option<ElementType>,
    ) -> Result<String, String> {
        match expr {
            Expr::Number(n) => Ok(n.value.to_string()),
            Expr::Identifier(id) => {
                let name = id.name.replace(".", "_");
                self.declarations.insert(name.clone());
                if let Some((ty, true)) = type_of(id.name.as_ref()) {
                    self.ranges.insert(name.clone(), ty);
                }
                Ok(name)
            }
            Expr::BinaryOp(b) => {
                let ty = b
                    .operand_ty
                    .clone()
                    .or_else(|| int_type(expr, type_of))
                    .or(expected);
                let lhs = self.lower_expr(&b.lhs, type_of, ty.clone())?;
                let rhs = self.lower_expr(&b.rhs, type_of, ty.clone())?;
                let op = match b.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    _ => return Err(format!("the `{:?}` operator", b.op)),
                };
                wrap(format!("({} {} {})", op, lhs, rhs), ty)
            }
            Expr::RelationalOp(r) => {
                let side = r
                    .operand_ty
                    .clone()
                    .or_else(|| int_type(&r.lhs, type_of))
                    .or_else(|| int_type(&r.rhs, type_of));
                let lhs = self.lower_expr(&r.lhs, type_of, side.clone())?;
                let rhs = self.lower_expr(&r.rhs, type_of, side)?;
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
                let lhs = self.lower_expr(&l.lhs, type_of, None)?;
                let rhs = self.lower_expr(&l.rhs, type_of, None)?;
                let op = match l.op {
                    LogicalOp::And => "and",
                    LogicalOp::Or => "or",
                };
                Ok(format!("({} {} {})", op, lhs, rhs))
            }
            Expr::UnaryOp(u) => {
                let inner_expected = (u.op == UnaryOp::Neg).then_some(expected.clone()).flatten();
                let inner = self.lower_expr(&u.expr, type_of, inner_expected)?;
                match u.op {
                    UnaryOp::Not => Ok(format!("(not {})", inner)),
                    // A negative literal such as `-1000` is a constant, not an operation.
                    UnaryOp::Neg if matches!(*u.expr, Expr::Number(_)) => {
                        Ok(format!("(- {inner})"))
                    }
                    UnaryOp::Neg => wrap(
                        format!("(- {})", inner),
                        int_type(&u.expr, type_of).or(expected),
                    ),
                }
            }
            Expr::MemberAccess(m) => {
                let base = self.lower_expr(&m.base, type_of, None)?;
                let field = m.member.replace(".", "_");
                let name = format!("{}_{}", base, field);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::IndexAccess(i) => {
                let base = self.lower_expr(&i.base, type_of, None)?;
                let index = self.lower_expr(&i.index, type_of, None)?;
                // For now, treat array indexing as a flattened variable (e.g. a_0).
                // Full theory of arrays would require (select a 0).
                let name = format!("{}_{}", base, index);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::Topology(t) => {
                let name = match &t.top {
                    Topology::CPU => "Topology_Host".to_string(),
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
                    Topology::GPU(e) => {
                        if let Expr::Number(n) = &**e {
                            format!("Topology_GPU_{}", n.value)
                        } else {
                            "Topology_GPU".to_string()
                        }
                    }
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
            _ => Err(format!("a `{}` expression", kind_name(expr))),
        }
    }
}

/// The integer type of `e`, from a typed literal, a name, or an operation's operand type.
fn int_type(e: &Expr, type_of: IntTypeOf) -> Option<ElementType> {
    match e {
        Expr::Number(n) => n.ty.clone(),
        Expr::Identifier(id) => type_of(id.name.as_ref()).map(|(ty, _)| ty),
        Expr::BinaryOp(b) => b
            .operand_ty
            .clone()
            .or_else(|| int_type(&b.lhs, type_of))
            .or_else(|| int_type(&b.rhs, type_of)),
        Expr::UnaryOp(u) => int_type(&u.expr, type_of),
        _ => None,
    }
}

/// `2^bits`, in decimal.
fn pow2(bits: u32) -> String {
    match bits {
        0..=127 => (1u128 << bits).to_string(),
        128 => "340282366920938463463374607431768211456".to_string(),
        _ => panic!("no integer type is wider than 128 bits"),
    }
}

/// The smallest and largest value of an integer type, as SMT terms. `usize` holds 0 to
/// `i64::MAX`. `None` for a type that is not an integer.
fn int_range(ty: &ElementType) -> Option<(String, String)> {
    let bits = ty.int_bits().filter(|_| *ty != ElementType::Bool)?;
    let max = |b: u32| {
        let p = pow2(b);
        format!("(- {p} 1)")
    };
    Some(match ty {
        ElementType::USize => ("0".to_string(), max(63)),
        t if t.is_signed_int() => (format!("(- {})", pow2(bits - 1)), max(bits - 1)),
        _ => ("0".to_string(), max(bits)),
    })
}

/// `exact`, an integer `+`, `-`, `*` or negation, with the result the machine gives. It wraps at
/// the width of the type. A `usize` result is exact: an overflow there stops the program. A float
/// stays as it is. `None` means the type is not known, and the prover cannot model the operation.
fn wrap(exact: String, ty: Option<ElementType>) -> Result<String, String> {
    let Some(ty) = ty else {
        return Err("an operation whose integer type is not known".to_string());
    };
    if ty.is_float() || ty == ElementType::USize {
        return Ok(exact);
    }
    let Some(bits) = ty.int_bits().filter(|_| ty != ElementType::Bool) else {
        return Err(format!("an operation on `{ty:?}`"));
    };
    let m = pow2(bits);
    if ty.is_signed_int() {
        let h = pow2(bits - 1);
        Ok(format!("(- (mod (+ {exact} {h}) {m}) {h})"))
    } else {
        Ok(format!("(mod {exact} {m})"))
    }
}

/// Whether the prover can model `e`: names, numbers, `+`, `-`, `*`, comparisons, `&&`, `||`, `!`,
/// negation, field and element reads of those, and calls with such arguments, whose results are
/// known through the callee's `ensures`.
pub fn is_modelled(e: &Expr) -> bool {
    match e {
        Expr::Number(_) | Expr::Identifier(_) => true,
        Expr::FunctionCall(c) => c.args.iter().all(is_modelled),
        Expr::BinaryOp(b) => {
            matches!(b.op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul)
                && is_modelled(&b.lhs)
                && is_modelled(&b.rhs)
        }
        Expr::RelationalOp(r) => is_modelled(&r.lhs) && is_modelled(&r.rhs),
        Expr::LogicalOp(l) => is_modelled(&l.lhs) && is_modelled(&l.rhs),
        Expr::UnaryOp(u) => is_modelled(&u.expr),
        Expr::MemberAccess(m) => is_modelled(&m.base),
        Expr::IndexAccess(i) => is_modelled(&i.base) && is_modelled(&i.index),
        _ => false,
    }
}

/// `e` with each name in `names` replaced by its expression: a callee's `requires` with the call's
/// arguments in place of the parameters. Only the kinds of expression `is_modelled` accepts are
/// walked; anything else is left as it is.
pub fn substitute_names(
    e: &Expr,
    names: &std::collections::HashMap<crate::symbol::Symbol, Expr>,
) -> Expr {
    let sub = |x: &Expr| Box::new(substitute_names(x, names));
    match e {
        Expr::Identifier(id) => names.get(&id.name).cloned().unwrap_or_else(|| e.clone()),
        Expr::BinaryOp(b) => {
            let mut b = b.clone();
            b.lhs = sub(&b.lhs);
            b.rhs = sub(&b.rhs);
            Expr::BinaryOp(b)
        }
        Expr::RelationalOp(r) => {
            let mut r = r.clone();
            r.lhs = sub(&r.lhs);
            r.rhs = sub(&r.rhs);
            Expr::RelationalOp(r)
        }
        Expr::LogicalOp(l) => {
            let mut l = l.clone();
            l.lhs = sub(&l.lhs);
            l.rhs = sub(&l.rhs);
            Expr::LogicalOp(l)
        }
        Expr::UnaryOp(u) => {
            let mut u = u.clone();
            u.expr = sub(&u.expr);
            Expr::UnaryOp(u)
        }
        Expr::MemberAccess(m) => {
            let mut m = m.clone();
            m.base = sub(&m.base);
            Expr::MemberAccess(m)
        }
        Expr::IndexAccess(i) => {
            let mut i = i.clone();
            i.base = sub(&i.base);
            i.index = sub(&i.index);
            Expr::IndexAccess(i)
        }
        _ => e.clone(),
    }
}

/// `e` as Vx source, for the kinds of expression `is_modelled` accepts, or `None` for any other.
pub fn to_source(e: &Expr) -> Option<String> {
    Some(match e {
        Expr::Number(n) => n.value.to_string(),
        Expr::Identifier(id) => id.name.to_string(),
        Expr::BinaryOp(b) => {
            let op = match b.op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                _ => return None,
            };
            format!("{} {op} {}", to_source(&b.lhs)?, to_source(&b.rhs)?)
        }
        Expr::RelationalOp(r) => {
            let op = match r.op {
                RelationalOp::Eq => "==",
                RelationalOp::NotEq => "!=",
                RelationalOp::Lt => "<",
                RelationalOp::Le => "<=",
                RelationalOp::Gt => ">",
                RelationalOp::Ge => ">=",
            };
            format!("{} {op} {}", to_source(&r.lhs)?, to_source(&r.rhs)?)
        }
        Expr::LogicalOp(l) => {
            let op = match l.op {
                LogicalOp::And => "&&",
                LogicalOp::Or => "||",
            };
            format!("({}) {op} ({})", to_source(&l.lhs)?, to_source(&l.rhs)?)
        }
        Expr::UnaryOp(u) => match u.op {
            UnaryOp::Not => format!("!({})", to_source(&u.expr)?),
            UnaryOp::Neg => format!("-{}", to_source(&u.expr)?),
        },
        Expr::MemberAccess(m) => format!("{}.{}", to_source(&m.base)?, m.member),
        Expr::IndexAccess(i) => format!("{}[{}]", to_source(&i.base)?, to_source(&i.index)?),
        _ => return None,
    })
}

/// The kind of expression `e` is, as its name in the syntax tree: `Grad`, `IndirectCall`.
fn kind_name(e: &Expr) -> String {
    let debug = format!("{e:?}");
    debug
        .split(['(', ' ', '{'])
        .next()
        .unwrap_or("an")
        .to_string()
}

/// `!e`.
pub fn negate(e: &Expr) -> Expr {
    Expr::UnaryOp(UnaryOpExpr {
        op: UnaryOp::Not,
        expr: Box::new(e.clone()),
        span: Span::default(),
    })
}

/// A fact that is never true, `0 == 1`: what is known where control cannot reach.
pub fn unreachable_fact() -> Expr {
    let number = |v: &str| {
        Box::new(Expr::Number(NumberExpr::new(
            v.to_string(),
            None,
            Span::default(),
        )))
    };
    Expr::RelationalOp(RelationalOpExpr {
        lhs: number("0"),
        op: RelationalOp::Eq,
        rhs: number("1"),
        span: Span::default(),
        operand_ty: None,
    })
}
