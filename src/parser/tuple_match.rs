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
//! A literal in a pattern becomes a test, a name becomes a `let` at the top of the arm. The last
//! arm must match every value, so it is the final `else` and the chain is a value wherever the
//! `match` was. An enum variant inside a tuple pattern is not supported yet.

use super::*;

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

    /// Rewrite `match scrutinee { arms }`, where some arm is a tuple pattern, into an `if` chain.
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
            let mut tests = Vec::new();
            let mut binds = Vec::new();
            self.arm_tests(&pattern, &value, &token, &mut tests, &mut binds)?;
            if i < last && tests.is_empty() {
                return Err(self.error_at(
                    &token,
                    "this arm matches every value, so the arms after it can never run",
                ));
            }
            if i == last && !tests.is_empty() {
                return Err(self.error_at(
                    &token,
                    "a `match` on a tuple must end with an arm that matches every value, \
                     such as `_` or `(a, b)`",
                ));
            }
            binds.extend(body);
            lowered.push((tests, binds));
        }

        if let Some(check) = size_check {
            lowered[0].0.insert(0, check);
        }

        // Build the chain from the last arm, which is the final `else`, back to the first.
        if lowered.len() == 1 {
            // One arm that matches everything: run it unconditionally, after the size check.
            let (tests, body) = lowered.pop().expect("one arm");
            let cond = if tests.is_empty() {
                identifier("true")
            } else {
                all_of(tests)
            };
            return Ok(Expr::If(IfExpr::new_value(cond, body.clone(), body)));
        }
        let (_, mut else_block) = lowered.pop().expect("a match has at least one arm");
        let (tests, then_block) = lowered.pop().expect("at least two arms");
        let mut chain = IfExpr::new_value(all_of(tests), then_block, else_block);
        while let Some((tests, then_block)) = lowered.pop() {
            else_block = vec![Statement::ExprStmt(ExprStmtStmt {
                expr: Expr::If(chain),
                has_semi: false,
                span: Span::default(),
            })];
            chain = IfExpr::new_value(all_of(tests), then_block, else_block);
        }
        Ok(Expr::If(chain))
    }

    /// The tests `pattern` puts on `value`, and the `let`s that bind its names.
    fn arm_tests(
        &self,
        pattern: &ArmPattern,
        value: &Expr,
        token: &Token<'a>,
        tests: &mut Vec<Expr>,
        binds: &mut Vec<Statement>,
    ) -> ParseResult<'a, ()> {
        match pattern {
            ArmPattern::Tuple(elems) => {
                for (i, elem) in elems.iter().enumerate() {
                    self.arm_tests(elem, &element(value, i), token, tests, binds)?;
                }
            }
            ArmPattern::Plain(plain) => match plain.as_ref() {
                Pattern::Wildcard => {}
                Pattern::Literal(lit) => tests.push(equals(value, lit)),
                Pattern::Identifier(name)
                    if name.as_ref() == "true" || name.as_ref() == "false" =>
                {
                    tests.push(equals(value, &identifier(name.as_ref())))
                }
                Pattern::Identifier(name) => binds.push(Statement::LetDecl(LetDeclStmt {
                    name: name.clone(),
                    is_mut: false,
                    ty_ann: None,
                    expr: value.clone(),
                    span: Span::default(),
                })),
                Pattern::EnumVariant(..) => {
                    return Err(self.error_at(
                        token,
                        "an enum variant inside a tuple pattern is not supported yet",
                    ))
                }
            },
        }
        Ok(())
    }
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
        .expect("an arm with no tests is the last one")
}
