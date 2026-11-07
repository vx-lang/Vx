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

/// The integer type of an expression, and whether its range is a fact.
pub type IntTypeOf<'a> = &'a dyn Fn(&Expr) -> Option<(ElementType, bool)>;

pub struct SmtProver {
    assertions: Vec<String>,
    /// Sorted, so that one program gives z3 the same query in every run. A step limit gives the
    /// same answer only for the same query.
    declarations: std::collections::BTreeSet<String>,
    /// The integer type of each declared name whose range is a fact.
    ranges: std::collections::HashMap<String, ElementType>,
    /// Whether a fact multiplies two values that are not constants, or divides by one. Only then
    /// does z3 need nonlinear arithmetic, which is much slower.
    nonlinear: bool,
}

/// The most work z3 may do on one query. It counts steps, not seconds, so a proof that passes on
/// one machine passes on every machine with the same z3. A proof in the tests needs at most about
/// 6,000 steps, and any query at most about 250,000. A million steps took 0.5 to 2 seconds.
const Z3_STEP_LIMIT: u64 = 1_000_000;

impl SmtProver {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            assertions: Vec::new(),
            declarations: std::collections::BTreeSet::new(),
            ranges: std::collections::HashMap::new(),
            nonlinear: false,
        }
    }

    pub fn prove(&self) -> Result<bool, String> {
        let logic = if self.nonlinear { "QF_NIA" } else { "QF_LIA" };
        let mut script = format!("(set-option :rlimit {Z3_STEP_LIMIT})\n(set-logic {logic})\n");

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
        solver_result(&result_str, output.status.success())
    }

    /// Add `expr` as a fact. `type_of` gives an expression's integer type, so that the prover knows
    /// the width of an operation on it, and its range.
    pub fn add_constraint(&mut self, expr: &Expr, type_of: IntTypeOf) -> Result<(), String> {
        let smt_expr = self.lower_expr(expr, type_of, None)?;
        self.assertions.push(smt_expr);
        Ok(())
    }

    /// The SMT term of the value `expr`, for a goal that the caller builds.
    pub fn lower_term(&mut self, expr: &Expr, type_of: IntTypeOf) -> Result<String, String> {
        self.lower_expr(expr, type_of, None)
    }

    /// Adds `smt`, a term that the caller built from `lower_term`, as a fact.
    pub fn add_smt(&mut self, smt: String) {
        self.assertions.push(smt);
    }

    /// Whether `smt`, a term that the caller built from `lower_term`, is one of the facts.
    pub fn has_fact(&self, smt: &str) -> bool {
        self.assertions.iter().any(|a| a == smt)
    }

    /// `expected` supplies context for literal arithmetic. Remainder requires its own operand type.
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
                if let Some((ty, true)) = type_of(expr) {
                    self.ranges.insert(name.clone(), ty);
                }
                Ok(name)
            }
            Expr::BinaryOp(b) => {
                let ty = if b.op == BinaryOp::Rem {
                    // A comparison's result type cannot determine a remainder's signedness.
                    int_type(expr, type_of)
                } else {
                    b.operand_ty
                        .clone()
                        .or_else(|| arithmetic_context_type(expr, type_of))
                        .or(expected)
                };
                let lhs = self.lower_expr(&b.lhs, type_of, ty.clone())?;
                let rhs = self.lower_expr(&b.rhs, type_of, ty.clone())?;
                if b.op == BinaryOp::Rem {
                    let ty = ty
                        .filter(|t| t.int_bits().is_some() && *t != ElementType::Bool)
                        .ok_or("a remainder whose integer type is not known")?;
                    // Leave undefined operations unconstrained instead of assuming they cannot occur.
                    let undefined = format!("|remainder.undefined.{}|", self.declarations.len());
                    self.declarations.insert(undefined.clone());
                    // A remainder by a constant stays linear, so that z3 can use its fast logic.
                    let divisor = match smt_constant_magnitude(&rhs) {
                        Some("0") => return Ok(undefined),
                        Some(digits) => digits.to_string(),
                        None => {
                            self.nonlinear = true;
                            format!("(ite (< {rhs} 0) (- {rhs}) {rhs})")
                        }
                    };
                    let magnitude = format!("(mod (ite (< {lhs} 0) (- {lhs}) {lhs}) {divisor})");
                    let value = if ty.is_signed_int() {
                        format!("(ite (< {lhs} 0) (- {magnitude}) {magnitude})")
                    } else {
                        format!("(mod {lhs} {rhs})")
                    };
                    let invalid = if ty.is_signed_int() {
                        let (min, _) = int_range(&ty).expect("an integer remainder");
                        format!("(or (= {rhs} 0) (and (= {lhs} {min}) (= {rhs} (- 1))))")
                    } else {
                        format!("(= {rhs} 0)")
                    };
                    return Ok(format!("(ite {invalid} {undefined} {value})"));
                }
                let op = match b.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    _ => return Err(format!("the `{:?}` operator", b.op)),
                };
                if b.op == BinaryOp::Mul
                    && smt_constant_magnitude(&lhs).is_none()
                    && smt_constant_magnitude(&rhs).is_none()
                {
                    self.nonlinear = true;
                }
                wrap(format!("({} {} {})", op, lhs, rhs), ty)
            }
            Expr::RelationalOp(r) => {
                let side = r
                    .operand_ty
                    .clone()
                    .or_else(|| arithmetic_context_type(&r.lhs, type_of))
                    .or_else(|| arithmetic_context_type(&r.rhs, type_of));
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
                    UnaryOp::Neg
                        if matches!(*u.expr, Expr::Number(_))
                            && int_type(&u.expr, type_of).is_none_or(|ty| ty.is_signed_int()) =>
                    {
                        Ok(format!("(- {inner})"))
                    }
                    UnaryOp::Neg => wrap(
                        format!("(- {})", inner),
                        int_type(&u.expr, type_of).or(expected),
                    ),
                    UnaryOp::Try => Err("the `?` operator".to_string()),
                }
            }
            // A field or an element is a name of its own, `|p.x|` or `|a[0]|`. The `|` quotes it
            // in SMT, so it cannot be the name of a variable: `p_x` was both.
            Expr::MemberAccess(m) => {
                let base = self.lower_expr(&m.base, type_of, None)?;
                let name = format!("|{}.{}|", base.replace('|', ""), m.member);
                self.declarations.insert(name.clone());
                Ok(name)
            }
            Expr::IndexAccess(i) => {
                let base = self.lower_expr(&i.base, type_of, None)?;
                let index = self.lower_expr(&i.index, type_of, None)?;
                // For now, treat array indexing as a flattened variable (e.g. a_0).
                // Full theory of arrays would require (select a 0).
                let name = format!("|{}[{}]|", base.replace('|', ""), index.replace('|', ""));
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
            Expr::AsCast(c) => {
                let (from, to) = cast_int_types(c, type_of)
                    .ok_or("a cast whose integer source or destination type is not known")?;
                let value = self.lower_expr(&c.expr, type_of, Some(from.clone()))?;
                if keeps_every_value(&from, &to) {
                    Ok(value)
                } else if to == ElementType::USize {
                    // A cast to usize stops the program on a value out of range, so that cast has
                    // no value. Like a remainder by zero, it becomes a name with no facts.
                    let (lo, hi) = int_range(&to).expect("the range of usize");
                    let undefined = format!("|cast.undefined.{}|", self.declarations.len());
                    self.declarations.insert(undefined.clone());
                    Ok(format!("(ite (<= {lo} {value} {hi}) {value} {undefined})"))
                } else {
                    wrap(value, Some(to))
                }
            }
            _ => Err(format!("a `{}` expression", kind_name(expr))),
        }
    }
}

fn solver_result(result: &str, successful_exit: bool) -> Result<bool, String> {
    if !successful_exit {
        return Err(format!("z3 failed: {result}"));
    }
    match result {
        "sat" => Ok(true),
        "unsat" => Ok(false),
        "unknown" => Err(format!(
            "z3 found neither a proof nor a counterexample. It may have reached its limit of \
             {Z3_STEP_LIMIT} steps"
        )),
        _ => Err(format!("Unexpected z3 output: {result}")),
    }
}

/// The digits of `smt` when it is an integer constant, `5` or `(- 5)`, and `None` otherwise.
fn smt_constant_magnitude(smt: &str) -> Option<&str> {
    let digits = smt
        .strip_prefix("(- ")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(smt);
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(digits)
}

/// Recover an expression's own type without borrowing a comparison's operand type.
pub(crate) fn int_type(e: &Expr, type_of: IntTypeOf) -> Option<ElementType> {
    if let Some((ty, _)) = type_of(e) {
        return Some(ty);
    }
    match e {
        Expr::Number(n) => n.ty.clone(),
        Expr::BinaryOp(b) => {
            if let Some(ty) = &b.operand_ty {
                return Some(ty.clone());
            }
            let lhs = int_type(&b.lhs, type_of);
            let rhs = int_type(&b.rhs, type_of);
            if (lhs.is_none() && !untyped_literal_expression(&b.lhs))
                || (rhs.is_none() && !untyped_literal_expression(&b.rhs))
            {
                return None;
            }
            match (lhs, rhs) {
                (Some(a), Some(b)) if a != b => None,
                (None, None) => None,
                (a, b) => a.or(b),
            }
        }
        Expr::UnaryOp(u) => int_type(&u.expr, type_of),
        Expr::AsCast(c) => integer_scalar(&c.target_ty),
        _ => None,
    }
}

fn untyped_integer_literal(e: &Expr) -> bool {
    match e {
        Expr::Number(n) => n.ty.is_none() && n.value.parse::<u128>().is_ok(),
        Expr::UnaryOp(u) if u.op == UnaryOp::Neg => untyped_integer_literal(&u.expr),
        _ => false,
    }
}

fn untyped_literal_expression(e: &Expr) -> bool {
    match e {
        Expr::BinaryOp(b) if b.operand_ty.is_none() => {
            untyped_literal_expression(&b.lhs) && untyped_literal_expression(&b.rhs)
        }
        Expr::UnaryOp(u) if u.op == UnaryOp::Neg => untyped_literal_expression(&u.expr),
        _ => untyped_integer_literal(e),
    }
}

fn literal_default_type(e: &Expr) -> Option<ElementType> {
    match e {
        Expr::Number(n) if n.ty.is_none() && untyped_integer_literal(e) => {
            Some(crate::parser::expr::default_number_elem(&n.value))
        }
        Expr::UnaryOp(u) if u.op == UnaryOp::Neg => literal_default_type(&u.expr),
        Expr::BinaryOp(b) if b.operand_ty.is_none() => {
            let lhs = literal_default_type(&b.lhs)?;
            (literal_default_type(&b.rhs)? == lhs).then_some(lhs)
        }
        _ => None,
    }
}

pub(crate) fn cast_source_type(e: &Expr, type_of: IntTypeOf) -> Option<ElementType> {
    int_type(e, type_of).or_else(|| literal_default_type(e))
}

/// Apply surrounding types before defaulting literal expressions or replacing callee names.
pub(crate) fn with_integer_types(e: &Expr, type_of: IntTypeOf) -> Expr {
    with_integer_context(e, None, type_of)
}

pub(crate) fn with_integer_context(
    e: &Expr,
    expected: Option<ElementType>,
    type_of: IntTypeOf,
) -> Expr {
    fn typed(e: &Expr, expected: Option<ElementType>, type_of: IntTypeOf) -> Expr {
        let mut result = e.clone();
        match &mut result {
            Expr::Number(n) if n.ty.is_none() && untyped_integer_literal(e) => {
                n.ty = expected.or_else(|| literal_default_type(e));
            }
            Expr::BinaryOp(b) => {
                let own = int_type(e, type_of);
                let ty = own.or_else(|| {
                    untyped_literal_expression(e)
                        .then(|| expected.clone().or_else(|| literal_default_type(e)))
                        .flatten()
                });
                b.operand_ty = ty.clone();
                let child_context =
                    ty.or_else(|| (b.op != BinaryOp::Rem).then_some(expected).flatten());
                *b.lhs = typed(&b.lhs, child_context.clone(), type_of);
                *b.rhs = typed(&b.rhs, child_context, type_of);
            }
            Expr::RelationalOp(r) => {
                let ty = r
                    .operand_ty
                    .clone()
                    .or_else(|| int_type(&r.lhs, type_of))
                    .or_else(|| int_type(&r.rhs, type_of))
                    .or_else(|| literal_default_type(&r.lhs))
                    .or_else(|| literal_default_type(&r.rhs));
                *r.lhs = typed(&r.lhs, ty.clone(), type_of);
                *r.rhs = typed(&r.rhs, ty, type_of);
            }
            Expr::LogicalOp(l) => {
                *l.lhs = typed(&l.lhs, None, type_of);
                *l.rhs = typed(&l.rhs, None, type_of);
            }
            Expr::UnaryOp(u) => {
                *u.expr = typed(
                    &u.expr,
                    if u.op == UnaryOp::Neg { expected } else { None },
                    type_of,
                );
            }
            Expr::AsCast(c) => {
                // A cast's destination is not the context of its source expression.
                *c.expr = typed(&c.expr, None, type_of);
                if c.source_ty.is_none() {
                    c.source_ty = int_type(&c.expr, type_of).map(Type::Scalar);
                }
            }
            // Calls receive argument context from their declared parameters during abstraction.
            _ => {}
        }
        result
    }
    typed(e, expected, type_of)
}

/// Preserve context for older arithmetic facts whose local variables left scope.
fn arithmetic_context_type(e: &Expr, type_of: IntTypeOf) -> Option<ElementType> {
    match e {
        Expr::BinaryOp(b) if b.op != BinaryOp::Rem => b
            .operand_ty
            .clone()
            .or_else(|| arithmetic_context_type(&b.lhs, type_of))
            .or_else(|| arithmetic_context_type(&b.rhs, type_of)),
        Expr::UnaryOp(u) => arithmetic_context_type(&u.expr, type_of),
        _ => int_type(e, type_of),
    }
}

fn integer_scalar(ty: &Type) -> Option<ElementType> {
    match ty {
        Type::Scalar(e) if e.int_bits().is_some() && *e != ElementType::Bool => Some(e.clone()),
        _ => None,
    }
}

fn cast_int_types(c: &AsCastExpr, type_of: IntTypeOf) -> Option<(ElementType, ElementType)> {
    let from = match &c.source_ty {
        Some(ty) => integer_scalar(ty)?,
        None => cast_source_type(&c.expr, type_of)?,
    };
    Some((from, integer_scalar(&c.target_ty)?))
}

/// Whether each source value fits in the destination. A usize holds 0 to i64::MAX.
fn keeps_every_value(from: &ElementType, to: &ElementType) -> bool {
    let width = |e: &ElementType| match e {
        ElementType::USize => (63, false),
        e => (e.int_bits().unwrap_or(0), e.is_signed_int()),
    };
    let ((from_bits, from_signed), (to_bits, to_signed)) = (width(from), width(to));
    match (from_signed, to_signed) {
        (true, false) => false,
        (false, true) => to_bits > from_bits,
        _ => to_bits >= from_bits,
    }
}

fn can_model_integer_cast(c: &AsCastExpr) -> bool {
    integer_scalar(&c.target_ty).is_some()
        && c.source_ty
            .as_ref()
            .is_none_or(|ty| integer_scalar(ty).is_some())
}

/// `e` without the `&` or `&mut` before a place. At a call, the field `self.n` of a `&c`
/// argument is the field `c.n`.
pub fn without_borrow(e: &Expr) -> &Expr {
    match e {
        Expr::Borrow(b) => without_borrow(&b.expr),
        _ => e,
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

/// Whether the prover can model `e`: names, numbers, `+`, `-`, `*`, `%`, comparisons, `&&`, `||`, `!`,
/// negation, field and element reads of those, integer casts, and calls with such
/// arguments, or references to them, whose results are known through the callee's `ensures`.
pub fn is_modelled(e: &Expr) -> bool {
    match e {
        Expr::Number(_) | Expr::Identifier(_) => true,
        Expr::FunctionCall(c) => c.args.iter().all(|a| is_modelled(without_borrow(a))),
        Expr::AsCast(c) => can_model_integer_cast(c) && is_modelled(&c.expr),
        Expr::BinaryOp(b) => {
            matches!(
                b.op,
                BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Rem
            ) && is_modelled(&b.lhs)
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
        Expr::AsCast(c) => {
            let mut c = c.clone();
            c.expr = sub(&c.expr);
            Expr::AsCast(c)
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
                BinaryOp::Rem => "%",
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
            UnaryOp::Try => format!("{}?", to_source(&u.expr)?),
        },
        Expr::MemberAccess(m) => format!("{}.{}", to_source(&m.base)?, m.member),
        Expr::IndexAccess(i) => format!("{}[{}]", to_source(&i.base)?, to_source(&i.index)?),
        Expr::AsCast(c) => format!("{} as {}", to_source(&c.expr)?, c.target_ty),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_defaults_do_not_type_unknown_variables() {
        let literal = Expr::Number(NumberExpr::new("255".into(), None, Span::default()));
        assert_eq!(
            cast_source_type(&literal, &|_| None),
            Some(ElementType::I32)
        );
        let unknown = Expr::Identifier(IdentifierExpr::new("unknown".into(), Span::default()));
        assert_eq!(cast_source_type(&unknown, &|_| None), None);
    }

    #[test]
    fn a_remainder_needs_its_own_operand_type() {
        let unknown = Expr::Identifier(IdentifierExpr::new("unknown".into(), Span::default()));
        for divisor_type in [None, Some(ElementType::U32)] {
            let remainder = Expr::BinaryOp(BinaryOpExpr::new(
                Box::new(unknown.clone()),
                BinaryOp::Rem,
                Box::new(Expr::Number(NumberExpr::new(
                    "10".into(),
                    divisor_type.clone(),
                    Span::default(),
                ))),
                Span::default(),
            ));
            let mut prover = SmtProver::new();
            let remainder = with_integer_context(&remainder, Some(ElementType::U32), &|_| None);
            assert_eq!(
                prover.lower_expr(&remainder, &|_| None, Some(ElementType::U32)),
                Err("a remainder whose integer type is not known".to_string()),
                "unknown operand with divisor type {divisor_type:?}"
            );
        }
    }

    /// Lower `lhs op rhs` as a fact on i64 values and say whether z3 needs nonlinear arithmetic.
    fn needs_nonlinear(lhs: &str, op: BinaryOp, rhs: &str) -> bool {
        let operand = |text: &str| match text.parse::<i64>() {
            Ok(_) => Expr::Number(NumberExpr::new(
                text.into(),
                Some(ElementType::I64),
                Span::default(),
            )),
            Err(_) => Expr::Identifier(IdentifierExpr::new(text.into(), Span::default())),
        };
        let fact = Expr::RelationalOp(RelationalOpExpr::new(
            Box::new(Expr::BinaryOp(BinaryOpExpr::new(
                Box::new(operand(lhs)),
                op,
                Box::new(operand(rhs)),
                Span::default(),
            ))),
            RelationalOp::Eq,
            Box::new(operand("0")),
            Span::default(),
        ));
        let i64_names = |e: &Expr| match e {
            Expr::Identifier(_) => Some((ElementType::I64, true)),
            _ => None,
        };
        let mut prover = SmtProver::new();
        prover
            .add_constraint(&fact, &i64_names)
            .expect("a fact the prover can model");
        prover.nonlinear
    }

    #[test]
    fn only_a_product_or_remainder_of_two_unknowns_is_nonlinear() {
        assert!(!needs_nonlinear("x", BinaryOp::Mul, "3"));
        assert!(!needs_nonlinear("3", BinaryOp::Mul, "x"));
        assert!(!needs_nonlinear("x", BinaryOp::Rem, "10"));
        assert!(!needs_nonlinear("x", BinaryOp::Add, "y"));
        assert!(needs_nonlinear("x", BinaryOp::Mul, "y"));
        assert!(needs_nonlinear("x", BinaryOp::Rem, "y"));
    }

    #[test]
    fn a_solver_error_or_timeout_is_not_a_proof() {
        assert_eq!(solver_result("sat", true), Ok(true));
        assert_eq!(solver_result("unsat", true), Ok(false));
        assert!(solver_result("unknown", true).is_err());
        assert!(solver_result("(error \"unsupported arithmetic\")\nunsat", true).is_err());
        assert!(solver_result("unsat", false).is_err());
    }
}
