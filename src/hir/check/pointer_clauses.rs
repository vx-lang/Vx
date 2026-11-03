//===- pointer_clauses.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pointer clauses: `requires` clauses about raw pointers. `valid(p, n)` says that `p` is not null,
// is aligned, and points at `n` elements of one live allocation. `initialized(p, n)` says that
// those elements hold values, and `!overlaps(p, q, n)` that the first `n` elements at `p` and at
// `q` share no byte. The body may assume them, and the code generator tells LLVM what `valid`
// says. The checker cannot prove them at a call yet, so only an `unsafe fn` may have them: each of
// its calls is in `unsafe` code, whose author keeps the promise.
// docs/discussions/borrow_facts_in_contracts.md, section 6, has the design.
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// The functions that a pointer clause uses.
const POINTER_FUNCTIONS: [&str; 3] = ["valid", "initialized", "overlaps"];

fn is_pointer_call(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall(fc) if POINTER_FUNCTIONS.contains(&fc.name.as_ref()))
}

/// Where the first use of a pointer function in `e` is. The prover does not read a clause that
/// uses one.
pub(crate) fn first_pointer_call(e: &Expr) -> Option<Span> {
    let mut found = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if found.is_none() && is_pointer_call(e) {
            found = Some(e.span());
        }
    });
    found
}

/// The length in a `valid(p, n)` clause, as far as the code generator can use it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Length {
    /// A parameter, by position.
    Param(usize),
    Constant(i64),
    /// An expression that the code generator does not compute.
    Other,
}

/// The `valid(p, n)` clauses of `f`, well formed, as the parameter of `p` and the length `n`.
pub(crate) fn valid_params(f: &Function) -> Vec<(usize, Length)> {
    if !f.is_unsafe {
        return Vec::new();
    }
    f.requires
        .iter()
        .filter_map(|e| match e {
            Expr::FunctionCall(fc) if fc.name.as_ref() == "valid" => match fc.args.as_slice() {
                [Expr::Identifier(p), n] => {
                    let ptr = pointer_param(f, p.name.as_ref())?;
                    let len = match n {
                        Expr::Identifier(id) => f
                            .params
                            .iter()
                            .position(|(name, _)| name == &id.name)
                            .map_or(Length::Other, Length::Param),
                        Expr::Number(num) => {
                            num.value.parse().map_or(Length::Other, Length::Constant)
                        }
                        _ => Length::Other,
                    };
                    Some((ptr, len))
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// The position of the raw pointer parameter `name` of `f`.
fn pointer_param(f: &Function, name: &str) -> Option<usize> {
    f.params
        .iter()
        .position(|(p, ty)| p.as_ref() == name && matches!(ty, Type::Pointer(..)))
}

impl<'a> TypeChecker<'a> {
    /// E8011 for the pointer clauses of `f`: each one is a `requires` of its own, names raw
    /// pointer parameters, and is on an `unsafe fn`.
    pub(crate) fn check_pointer_clauses(&mut self, f: &Function) {
        if self.speculating {
            return;
        }
        for e in &f.ensures {
            if let Some(at) = first_pointer_call(e) {
                self.pointer_clause_error(
                    "a pointer clause in `ensures` is not supported yet: write it as a `requires`"
                        .to_string(),
                    at,
                );
            }
        }
        for e in &f.requires {
            let Some(at) = first_pointer_call(e) else {
                continue;
            };
            if let Err(message) = self.read_pointer_clause(f, e) {
                self.pointer_clause_error(message, at);
            }
        }
    }

    /// Check one `requires` that uses a pointer function.
    fn read_pointer_clause(&self, f: &Function, e: &Expr) -> Result<(), String> {
        let (negated, call) = match e {
            Expr::UnaryOp(u) if u.op == UnaryOp::Not => (true, &*u.expr),
            e => (false, e),
        };
        let Expr::FunctionCall(fc) = call else {
            return Err(
                "write each pointer clause as a `requires` of its own, as in \
                        `requires valid(p, n)`"
                    .to_string(),
            );
        };
        if !is_pointer_call(call) {
            return Err(
                "write each pointer clause as a `requires` of its own, as in \
                        `requires valid(p, n)`"
                    .to_string(),
            );
        }
        let name = fc.name.as_ref();
        let pointers = if name == "overlaps" { 2 } else { 1 };
        let usage = if name == "overlaps" {
            "`overlaps` takes two raw pointer parameters and a length, as in `!overlaps(p, q, n)`"
                .to_string()
        } else {
            format!("`{name}` takes a raw pointer parameter and a length, as in `{name}(p, n)`")
        };
        if fc.args.len() != pointers + 1 {
            return Err(usage);
        }
        match (name, negated) {
            ("overlaps", false) => {
                return Err(
                    "only `!overlaps(p, q, n)` says something that the body can use".to_string(),
                );
            }
            (_, true) if name != "overlaps" => {
                return Err(format!("`!{name}` says nothing that the body can use"));
            }
            _ => {}
        }
        let mut seen = Vec::new();
        for arg in &fc.args[..pointers] {
            let Expr::Identifier(id) = arg else {
                return Err(usage);
            };
            let p = id.name.as_ref();
            match f.params.iter().find(|(n, _)| n.as_ref() == p) {
                None => return Err(format!("`{p}` is not a parameter of `{}`", f.name)),
                Some((_, ty)) if !matches!(ty, Type::Pointer(..)) => {
                    return Err(format!("`{name}` needs a raw pointer, but `{p}` is `{ty}`"));
                }
                Some(_) => {}
            }
            if seen.contains(&p) {
                return Err(format!(
                    "`overlaps` needs two different pointers, but both are `{p}`"
                ));
            }
            seen.push(p);
        }
        if !f.is_unsafe {
            return Err(format!(
                "`{}` has a pointer clause, so it must be an `unsafe fn` for now: the checker \
                 cannot prove the clause at a call yet, and the caller of an `unsafe fn` keeps \
                 that promise",
                f.name
            ));
        }
        Ok(())
    }

    /// E8011, once for each place and message: an instance of a generic function is checked for
    /// each type it is used with.
    fn pointer_clause_error(&mut self, message: String, at: Span) {
        if !self.borrow_clause_errors.insert((at, message.clone())) {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8011,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
        );
    }
}
