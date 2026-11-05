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
// `q` share no byte. The body may assume them. A call in safe code must prove them
// (`pointer_facts.rs`), and the author of `unsafe` code keeps the promise. The code generator
// tells LLVM what `valid` says in an `unsafe fn`. `ensures !frees(p)` says that a call frees no
// memory that it reaches through `p`, so a call keeps the facts about that memory. The checker
// checks a written `!frees(p)` against the body, and trusts one on an `extern` function.
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

/// The function of a `frees` clause.
const FREES: &str = "frees";

/// Where the first use of `frees` in `e` is. The prover does not read a clause that uses it.
pub(crate) fn first_frees_call(e: &Expr) -> Option<Span> {
    let mut found = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if found.is_none() && matches!(e, Expr::FunctionCall(fc) if fc.name.as_ref() == FREES) {
            found = Some(e.span());
        }
    });
    found
}

/// The `frees` clause `e` of the function `owner` as the position of the parameter that it names
/// and whether it has `!`, or the reason that it is not well formed.
fn read_frees_clause(
    owner: &str,
    params: &[(crate::symbol::Symbol, Type)],
    e: &Expr,
) -> Result<(usize, bool), String> {
    let usage = "write each `frees` clause as an `ensures` of its own, with one parameter, as in \
                 `ensures !frees(p)`";
    let (negated, call) = match e {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => (true, &*u.expr),
        e => (false, e),
    };
    let Expr::FunctionCall(fc) = call else {
        return Err(usage.to_string());
    };
    let [Expr::Identifier(p)] = fc.args.as_slice() else {
        return Err(usage.to_string());
    };
    if fc.name.as_ref() != FREES {
        return Err(usage.to_string());
    }
    match params.iter().position(|(name, _)| name == &p.name) {
        Some(i) => Ok((i, negated)),
        None => Err(format!("`{}` is not a parameter of `{owner}`", p.name)),
    }
}

/// The parameters that the clauses `ensures` say `!frees` for, as bits. A parameter that a
/// clause also says `frees` for gets no bit, and E8011 reports the two clauses.
pub(crate) fn not_freed_params(params: &[(crate::symbol::Symbol, Type)], ensures: &[Expr]) -> u32 {
    let clauses: Vec<(usize, bool)> = ensures
        .iter()
        .filter_map(|e| read_frees_clause("", params, e).ok())
        .collect();
    clauses
        .iter()
        .filter(|&&(i, negated)| negated && i < 32 && !clauses.contains(&(i, false)))
        .fold(0, |bits, (i, _)| bits | 1 << i)
}

/// The parameters that a well formed `frees` clause of `ensures` names, with or without `!`, as
/// bits. The walk infers no `!frees` for them, because the written clause decides.
pub(crate) fn frees_clause_params(
    params: &[(crate::symbol::Symbol, Type)],
    ensures: &[Expr],
) -> u32 {
    ensures
        .iter()
        .filter_map(|e| read_frees_clause("", params, e).ok())
        .filter(|&(i, _)| i < 32)
        .fold(0, |bits, (i, _)| bits | 1 << i)
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

/// Whether `f` has a pointer `requires`.
pub(crate) fn has_pointer_requires(f: &Function) -> bool {
    f.requires.iter().any(|e| first_pointer_call(e).is_some())
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
        let owner = self.clause_owner_name(f);
        self.check_frees_clauses(&owner, &f.params, &f.requires, &f.ensures);
        self.check_frees_promises(f);
    }

    /// E8011 for the `frees` clauses of the function `owner` with the parameters `params`: each
    /// one is an `ensures` of its own and names one parameter, and no two of them disagree.
    fn check_frees_clauses(
        &mut self,
        owner: &str,
        params: &[(crate::symbol::Symbol, Type)],
        requires: &[Expr],
        ensures: &[Expr],
    ) {
        for at in requires.iter().filter_map(first_frees_call) {
            self.pointer_clause_error(
                "`frees` says what a call does, so write it as an `ensures`, as in \
                 `ensures !frees(p)`"
                    .to_string(),
                at,
            );
        }
        let mut seen = Vec::new();
        for e in ensures {
            let Some(at) = first_frees_call(e) else {
                continue;
            };
            match read_frees_clause(owner, params, e) {
                Err(message) => self.pointer_clause_error(message, at),
                Ok((i, negated)) => {
                    if seen.contains(&(i, !negated)) {
                        let p = &params[i].0;
                        self.pointer_clause_error(
                            format!("`frees({p})` and `!frees({p})` cannot both hold"),
                            at,
                        );
                    }
                    seen.push((i, negated));
                }
            }
        }
    }

    /// E8012 for each `!frees(p)` of `f` that its body can break: a use of `p`, or of a value
    /// that the body gets from `p`, can free memory.
    fn check_frees_promises(&mut self, f: &Function) {
        let promised = not_freed_params(&f.params, &f.ensures);
        if promised == 0 || f.body.is_empty() {
            return;
        }
        let uses =
            crate::hir::frees::freeing_uses(f, &self.env.decls, &|name| self.frees_callee(name));
        let shown = self.clause_owner_name(f);
        for (i, (p, _)) in f.params.iter().enumerate() {
            if promised & (1 << i) == 0 {
                continue;
            }
            let found = match &uses {
                Some(uses) => uses[i].clone(),
                None => Some(crate::hir::frees::FreeingUse {
                    at: f.span,
                    why: "the compiler cannot follow the body".to_string(),
                }),
            };
            let Some(found) = found else {
                continue;
            };
            let message = format!(
                "`{shown}` says `!frees({p})`, but its body can free memory through `{p}`: {}",
                found.why
            );
            if self
                .borrow_clause_errors
                .insert((found.at, message.clone()))
            {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E8012,
                    message,
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&found.at)),
                );
            }
        }
    }

    /// What the `!frees` walk knows about the function `name` that a checked body calls.
    pub(crate) fn frees_callee(&self, name: &str) -> Option<crate::hir::frees::Callee> {
        let instance = self
            .mono
            .functions
            .iter()
            .find(|(f, _)| f.name.as_ref() == name)
            .map(|(f, _)| f);
        let returns = self
            .env
            .functions
            .get(name)
            .map(|f| f.0.clone())
            .or_else(|| instance.map(|f| f.return_type.clone()))
            .or_else(|| {
                self.env
                    .generic_functions
                    .get(name)
                    .map(|(f, _)| f.return_type.clone())
            })?;
        // A message names an instance of a method for its receiver, as in `Buffer::fill`.
        let shown = match instance {
            Some(f) => self.clause_owner_name(f),
            None => self.shown_callee(name),
        };
        Some(crate::hir::frees::Callee {
            not_freed: self.not_freed_by(name),
            returns,
            shown,
        })
    }

    /// The parameters that the function `name` says `!frees` for, or that the walk of its body
    /// proves, as bits. A call keeps the facts about the memory that such an argument reaches.
    pub(crate) fn not_freed_by(&self, name: &str) -> u32 {
        let name = match name.find('<') {
            Some(i) if name.ends_with('>') => &name[..i],
            _ => name,
        };
        self.mono
            .not_freed
            .get(name)
            .or_else(|| self.env.not_freed.get(name))
            .copied()
            .unwrap_or(0)
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
        Ok(())
    }

    /// E8011 for the function `name` used as a value, when it is safe and has a pointer
    /// `requires`: a call through the value proves no clause.
    pub(crate) fn check_function_value(&mut self, name: &str, at: Span) {
        if self.speculating {
            return;
        }
        let refused = self
            .env
            .syntax_functions
            .get(name)
            .copied()
            .or_else(|| {
                self.mono
                    .functions
                    .iter()
                    .find(|(f, _)| f.name.as_ref() == name)
                    .map(|(f, _)| f)
            })
            .is_some_and(|f| !f.is_unsafe && has_pointer_requires(f));
        if refused {
            self.pointer_clause_error(
                format!(
                    "`{name}` has a pointer `requires`, so it cannot be a function value yet: a \
                     call through the value would not prove the clause"
                ),
                at,
            );
        }
    }

    /// E8011 for a safe method of a trait impl with a pointer `requires`, and for a method of a
    /// trait impl with a `!frees` that the trait method does not have: a call through a trait
    /// bound reads the clauses of the trait, not of the impl.
    pub(crate) fn check_trait_impl_pointer_clauses(&mut self) {
        let mut found = Vec::new();
        for (trait_name, blocks) in &self.env.impls {
            if trait_name.as_ref() == "_inherent" {
                continue;
            }
            let signatures = self.env.traits.get(trait_name).map(|t| &t.methods);
            for block in blocks {
                for m in &block.methods {
                    let of_trait = signatures
                        .and_then(|s| s.iter().find(|s| s.name == m.name))
                        .map_or(0, |s| not_freed_params(&s.params, &s.ensures));
                    let added = not_freed_params(&m.params, &m.ensures) & !of_trait;
                    for e in &m.ensures {
                        match read_frees_clause("", &m.params, e) {
                            Ok((i, true)) if added & (1 << i) != 0 => found.push((
                                format!(
                                    "`{}` is a method of an impl of `{trait_name}`, so it cannot \
                                     add `!frees({})`: a call through the trait reads only the \
                                     clauses of the trait",
                                    m.name, m.params[i].0
                                ),
                                first_frees_call(e).unwrap_or(m.span),
                            )),
                            _ => {}
                        }
                    }
                }
                for m in block.methods.iter().filter(|m| !m.is_unsafe) {
                    if let Some(at) = m.requires.iter().find_map(first_pointer_call) {
                        found.push((
                            format!(
                                "`{}` is a method of an impl of `{trait_name}`, so it must be an \
                             `unsafe fn` to have a pointer clause: a call through the trait would \
                             not prove the clause",
                                m.name
                            ),
                            at,
                        ));
                    }
                }
            }
        }
        found.sort_by_key(|(_, at)| (at.line, at.column));
        for (message, at) in found {
            self.pointer_clause_error(message, at);
        }
    }

    /// E8011 for the `ensures` of each `extern` function: the compiler trusts them, so each one
    /// is a well formed `frees` clause.
    pub(crate) fn check_extern_clauses(&mut self) {
        let externs: Vec<&crate::syntax::ExternDecl> = self.env.externs.clone();
        for ext in externs {
            for e in &ext.ensures {
                if first_frees_call(e).is_none() {
                    self.pointer_clause_error(
                        "an `extern` function can have only a `frees` clause in its `ensures`, \
                         as in `ensures !frees(p)`"
                            .to_string(),
                        e.span(),
                    );
                }
            }
            self.check_frees_clauses(ext.name.as_ref(), &ext.params, &[], &ext.ensures);
        }
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
