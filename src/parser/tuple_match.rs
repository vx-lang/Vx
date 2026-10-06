//===- tuple_match.rs - Vx Compiler -----------------------------------------===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//! A `match` on a tuple, rewritten by the parser into an `if`/`else if` chain over the tuple's
//! fields, the way `let (a, b) = e;` is rewritten into one `let` per field:
//!
//! ```text
//! match t {                      if t._0 == 0 {
//!   (0, y) => { A }                let y = t._1; A
//!   (x, _) => { B }      ==>     } else {
//! }                                let x = t._0; B
//!                                }
//! ```
//!
//! A literal in a pattern becomes a test, a name becomes a `let` at the top of the arm, and an
//! enum variant becomes a `match` on that element, which binds its payload. A failed test falls
//! back to the arms below it. The last arm must match every value, so it is the final `else` and
//! the result is a value wherever the `match` was.

use super::*;
use crate::symbol::Symbol;

/// A `match` arm's pattern: a tuple of patterns, or any other pattern.
pub(crate) enum ArmPattern {
    Tuple(Vec<ArmPattern>),
    Plain(Box<Pattern>),
}

impl<'a> Parser<'a> {
    /// A `match` arm's pattern, which may be a tuple: `(0, y)`.
    pub(crate) fn parse_arm_pattern(&mut self) -> ParseResult<'a, ArmPattern> {
        if !self.match_token(&TokenType::LeftParen) {
            return Ok(ArmPattern::Plain(Box::new(self.parse_pattern()?)));
        }
        let mut elems = Vec::new();
        while !self.check(&TokenType::RightParen) {
            elems.push(self.parse_arm_pattern()?);
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        self.consume(&TokenType::RightParen, "Expected ')' after a tuple pattern")?;
        self.tuple_struct(elems.len())?;
        Ok(ArmPattern::Tuple(elems))
    }

    /// Rewrite `match scrutinee { arms }`, where some arm is a tuple pattern, into `if`s and `match`es.
    pub(crate) fn lower_tuple_match(
        &mut self,
        scrutinee: Expr,
        arms: Vec<(ArmPattern, Vec<Statement>, Token<'a>)>,
        at_statement: bool,
    ) -> ParseResult<'a, Expr> {
        // The matched value is read once per test, so it must be one that can be read again: a
        // variable, or a tuple literal of variables and constants. Anything else goes in a
        // temporary, which needs the `match` to be a whole statement.
        let value = if is_rereadable(&scrutinee) {
            scrutinee
        } else if at_statement {
            let temp = format!("$match{}", self.tuple_lets);
            self.tuple_lets += 1;
            self.prefix_stmts.push(Statement::LetDecl(LetDeclStmt {
                name: temp.clone().into(),
                is_mut: false,
                ty_ann: None,
                expr: scrutinee,
                span: Span::default(),
            }));
            identifier(&temp)
        } else {
            return Err(self.error_at(
                &arms[0].2,
                "a `match` on a tuple used as a value must match a variable; \
                 put the value in a `let` first",
            ));
        };

        let arity = arms
            .iter()
            .find_map(|(p, _, _)| match p {
                ArmPattern::Tuple(e) => Some(e.len()),
                ArmPattern::Plain(_) => None,
            })
            .expect("a tuple match has a tuple pattern");
        // The number of elements is checked once, before the first test: by counting a tuple
        // literal here, or by `core::tuple`'s `has_N_elements`, which only that size of tuple fits.
        let size_check = match &value {
            Expr::StructInit(lit) => {
                if lit.fields.len() != arity {
                    return Err(self.error_at(
                        &arms[0].2,
                        &format!(
                            "the patterns have {arity} elements, but the tuple has {}",
                            lit.fields.len()
                        ),
                    ));
                }
                None
            }
            _ => Some(has_elements(arity, &value)),
        };
        let mut lowered = Vec::new();
        let last = arms.len() - 1;
        for (i, (pattern, body, token)) in arms.into_iter().enumerate() {
            if let ArmPattern::Tuple(e) = &pattern {
                if e.len() != arity {
                    return Err(self.error_at(
                        &token,
                        &format!(
                            "every tuple pattern in this `match` has {arity} elements; this one has {}",
                            e.len()
                        ),
                    ));
                }
            }
            let mut steps = Vec::new();
            let mut binds = Vec::new();
            self.arm_steps(&pattern, &value, &mut steps, &mut binds);
            if i < last && steps.is_empty() {
                return Err(self.error_at(
                    &token,
                    "this arm matches every value, so the arms after it can never run",
                ));
            }
            if i == last && !steps.is_empty() {
                return Err(self.error_at(
                    &token,
                    "a `match` on a tuple must end with an arm that matches every value, \
                     such as `_` or `(a, b)`",
                ));
            }
            binds.extend(body);
            lowered.push((steps, binds));
        }

        if lowered.len() == 1 {
            // One arm that matches everything: run it unconditionally, after the size check.
            let (_, body) = lowered.pop().expect("one arm");
            let cond = size_check.unwrap_or_else(|| identifier("true"));
            return Ok(Expr::If(IfExpr::new_value(cond, body.clone(), body)));
        }
        if let Some(check) = size_check {
            lowered[0].0.insert(0, Step::Test(check));
        }
        let mut tree = build(&lowered, 0);
        assert!(
            tree.len() == 1,
            "a match of two or more arms starts with a test"
        );
        let Some(Statement::ExprStmt(top)) = tree.pop() else {
            unreachable!("a test is an `if` or a `match`")
        };
        Ok(top.expr)
    }

    /// The steps `pattern` takes on `value`, in order, and the `let`s that bind its names.
    fn arm_steps(
        &mut self,
        pattern: &ArmPattern,
        value: &Expr,
        steps: &mut Vec<Step>,
        binds: &mut Vec<Statement>,
    ) {
        match pattern {
            ArmPattern::Tuple(elems) => {
                for (i, elem) in elems.iter().enumerate() {
                    self.arm_steps(elem, &element(value, i), steps, binds);
                }
            }
            ArmPattern::Plain(plain) => match plain.as_ref() {
                Pattern::Wildcard => {}
                Pattern::Literal(lit) => steps.push(Step::Test(equals(value, lit))),
                Pattern::Identifier(name) if is_bool(name) => {
                    steps.push(Step::Test(equals(value, &identifier(name.as_ref()))))
                }
                Pattern::Identifier(name) => binds.push(Statement::LetDecl(LetDeclStmt {
                    name: name.clone(),
                    is_mut: false,
                    ty_ann: None,
                    expr: value.clone(),
                    span: Span::default(),
                })),
                // The variant's `match` binds a name or `_` itself. Anything else in its payload,
                // a literal or another variant, is bound to a temporary and tested inside.
                Pattern::EnumVariant(enum_name, variant, payload) => {
                    let mut then = Vec::new();
                    let payload = payload.as_ref().map(|elems| {
                        elems
                            .iter()
                            .map(|p| match p {
                                Pattern::Wildcard => Pattern::Wildcard,
                                Pattern::Identifier(n) if !is_bool(n) => p.clone(),
                                _ => {
                                    let temp = format!("$payload{}", self.tuple_lets);
                                    self.tuple_lets += 1;
                                    self.arm_steps(
                                        &ArmPattern::Plain(Box::new(p.clone())),
                                        &identifier(&temp),
                                        &mut then,
                                        binds,
                                    );
                                    Pattern::Identifier(temp.into())
                                }
                            })
                            .collect()
                    });
                    steps.push(Step::Variant {
                        value: value.clone(),
                        enum_name: enum_name.clone(),
                        variant: variant.clone(),
                        payload,
                        then,
                    });
                }
            },
        }
    }
}

/// One step of an arm's pattern. Every step must pass for the arm to run.
#[derive(Clone)]
enum Step {
    /// This `bool` is true.
    Test(Expr),
    /// `value` is this enum variant. Its `match` binds `payload`, and `then` are the steps on
    /// what it binds.
    Variant {
        value: Expr,
        enum_name: Symbol,
        variant: Symbol,
        payload: Option<Vec<Pattern>>,
        then: Vec<Step>,
    },
}

/// The statements for arms `i..`: arm `i`'s steps, each falling back to arms `i + 1..` when it
/// fails. The fallback is written out at every step that can fail, so it is copied once per step.
fn build(arms: &[(Vec<Step>, Vec<Statement>)], i: usize) -> Vec<Statement> {
    let (steps, body) = &arms[i];
    if i == arms.len() - 1 {
        return body.clone();
    }
    build_steps(arms, i, steps, body)
}

fn build_steps(
    arms: &[(Vec<Step>, Vec<Statement>)],
    i: usize,
    steps: &[Step],
    body: &[Statement],
) -> Vec<Statement> {
    let Some(first) = steps.first() else {
        return body.to_vec();
    };
    let expr = match first {
        Step::Test(_) => {
            // Consecutive tests are one `if`, so a run of them costs one copy of the fallback.
            let n = steps
                .iter()
                .take_while(|s| matches!(s, Step::Test(_)))
                .count();
            let tests = steps[..n]
                .iter()
                .map(|s| match s {
                    Step::Test(t) => t.clone(),
                    Step::Variant { .. } => unreachable!("counted above"),
                })
                .collect();
            Expr::If(IfExpr::new_value(
                all_of(tests),
                build_steps(arms, i, &steps[n..], body),
                build(arms, i + 1),
            ))
        }
        Step::Variant {
            value,
            enum_name,
            variant,
            payload,
            then,
        } => {
            let inner: Vec<Step> = then.iter().chain(&steps[1..]).cloned().collect();
            Expr::Match(MatchExpr::new(
                Box::new(value.clone()),
                vec![
                    MatchArm {
                        pattern: Pattern::EnumVariant(
                            enum_name.clone(),
                            variant.clone(),
                            payload.clone(),
                        ),
                        body: build_steps(arms, i, &inner, body),
                    },
                    MatchArm {
                        pattern: Pattern::Wildcard,
                        body: build(arms, i + 1),
                    },
                ],
                Span::default(),
            ))
        }
    };
    // An `if` or `match` that returns on every path is a statement, not a value. The checker types
    // a `match` from the arms that end in a value, and in a `match` the parser has turned into
    // `return <match>` this one would otherwise count as a `void` value.
    let mut stmt = Statement::ExprStmt(ExprStmtStmt {
        expr,
        has_semi: false,
        span: Span::default(),
    });
    if crate::syntax::expr::statement_always_exits(&stmt) {
        let Statement::ExprStmt(e) = &mut stmt else {
            unreachable!("built above")
        };
        e.has_semi = true;
    }
    vec![stmt]
}

fn is_bool(name: &Symbol) -> bool {
    name.as_ref() == "true" || name.as_ref() == "false"
}

impl IfExpr {
    /// `if cond { then_block } else { else_block }`.
    fn new_value(cond: Expr, then_block: Vec<Statement>, else_block: Vec<Statement>) -> Self {
        Self {
            is_comptime: false,
            cond: Box::new(cond),
            then_block,
            else_block: Some(else_block),
            span: Span::default(),
        }
    }
}

/// Whether reading `e` again gives the same value with no effect: a variable, a constant, a field
/// of one, or a tuple literal of those.
fn is_rereadable(e: &Expr) -> bool {
    match e {
        Expr::Identifier(_) | Expr::Number(_) | Expr::StringLiteral(_) => true,
        Expr::MemberAccess(m) => is_rereadable(&m.base),
        Expr::StructInit(s) => {
            s.name.as_ref().starts_with("Tuple") && s.fields.iter().all(|(_, f)| is_rereadable(f))
        }
        _ => false,
    }
}

/// Element `i` of the tuple `value`: the literal's own element, or the field `_i`.
fn element(value: &Expr, i: usize) -> Expr {
    if let Expr::StructInit(s) = value {
        if s.name.as_ref().starts_with("Tuple") {
            return s.fields[i].1.clone();
        }
    }
    Expr::MemberAccess(MemberAccessExpr {
        base: Box::new(value.clone()),
        member: format!("_{i}").into(),
        struct_name: None,
        span: Span::default(),
    })
}

/// `has_N_elements(&value)`, from `core::tuple`: `true`, and a type error unless `value` is a
/// tuple of `n` elements.
pub(crate) fn has_elements(n: usize, value: &Expr) -> Expr {
    Expr::FunctionCall(FunctionCallExpr::new(
        format!("has_{n}_elements").into(),
        None,
        vec![Expr::Borrow(BorrowExpr::new(
            Box::new(value.clone()),
            false,
            Span::default(),
        ))],
        Span::default(),
    ))
}

fn identifier(name: &str) -> Expr {
    Expr::Identifier(IdentifierExpr {
        name: name.into(),
        span: Span::default(),
    })
}

fn equals(lhs: &Expr, rhs: &Expr) -> Expr {
    Expr::RelationalOp(RelationalOpExpr {
        lhs: Box::new(lhs.clone()),
        op: RelationalOp::Eq,
        rhs: Box::new(rhs.clone()),
        span: Span::default(),
        operand_ty: None,
    })
}

/// The tests joined with `&&`.
fn all_of(tests: Vec<Expr>) -> Expr {
    tests
        .into_iter()
        .reduce(|a, b| {
            Expr::LogicalOp(LogicalOpExpr::new(
                Box::new(a),
                LogicalOp::And,
                Box::new(b),
                Span::default(),
            ))
        })
        .expect("a group of tests is never empty")
}
