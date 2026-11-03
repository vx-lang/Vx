//===- borrow_clauses.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Borrow clauses: `ensures` clauses that state borrow facts, in place of lifetime syntax.
// `points_into(return, *p)` says that the result can point into the memory under `*p`, and
// `reaches(return, *p)` also counts the memory that `*p` reaches through references. A clause
// with `!` narrows what a caller assumes, and the checker proves it against the body. A clause
// without `!` keeps the default for its parameter. The written clauses about a parameter replace
// the clauses that the return summary (provenance.rs) infers for it.
// docs/discussions/borrow_facts_in_contracts.md has the design.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::provenance::ReturnProvenance;

/// The functions that a borrow clause uses.
const BORROW_FUNCTIONS: [&str; 3] = ["keeps", "points_into", "reaches"];

fn is_borrow_call(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall(fc) if BORROW_FUNCTIONS.contains(&fc.name.as_ref()))
}

/// Whether `e` is a borrow clause, `f(..)` or `!f(..)`, well formed or not. The prover does not
/// read these.
pub(crate) fn is_borrow_clause(e: &Expr) -> bool {
    match e {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => is_borrow_call(&u.expr),
        e => is_borrow_call(e),
    }
}

/// Whether `e` uses a borrow function anywhere, also inside a larger expression. The prover does
/// not read such a clause, and the checker reports one that is not a borrow clause of its own.
pub(crate) fn mentions_borrow_function(e: &Expr) -> bool {
    first_borrow_call(e).is_some()
}

/// Where the first use of a borrow function in `e` is.
fn first_borrow_call(e: &Expr) -> Option<Span> {
    let mut found = None;
    crate::syntax::visit_expr(e, &mut |e| {
        if found.is_none() && is_borrow_call(e) {
            found = Some(e.span());
        }
    });
    found
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Relation {
    PointsInto,
    Reaches,
}

/// `points_into(return, *p)` or `reaches(return, *p)`, with or without `!`.
#[derive(Clone, Copy, Debug)]
struct ResultClause {
    relation: Relation,
    /// The position of `p` among the parameters.
    param: usize,
    negated: bool,
}

impl ResultClause {
    fn text(&self, f: &Function) -> String {
        let name = match self.relation {
            Relation::PointsInto => "points_into",
            Relation::Reaches => "reaches",
        };
        let not = if self.negated { "!" } else { "" };
        format!("{not}{name}(return, *{})", f.params[self.param].0)
    }
}

/// Read `e`, an `ensures` of `f`. `None` when `e` is not a borrow clause, and a message when it
/// is one that is not well formed. `shown` names `f` in the message.
fn read_clause(f: &Function, shown: &str, e: &Expr) -> Option<Result<ResultClause, String>> {
    let (negated, call) = match e {
        Expr::UnaryOp(u) if u.op == UnaryOp::Not => (true, &*u.expr),
        e => (false, e),
    };
    let Expr::FunctionCall(fc) = call else {
        return None;
    };
    let relation = match fc.name.as_ref() {
        "points_into" => Relation::PointsInto,
        "reaches" => Relation::Reaches,
        "keeps" => return Some(Err("`keeps` is not supported yet".to_string())),
        _ => return None,
    };
    let name = fc.name.as_ref();
    let usage = format!(
        "`{name}` takes `return` and a reference parameter after `*`, as in `{name}(return, *p)`"
    );
    let [result, place] = fc.args.as_slice() else {
        return Some(Err(usage));
    };
    let Expr::Dereference(place) = place else {
        return Some(Err(usage));
    };
    if !matches!(result, Expr::Identifier(id) if id.name.as_ref() == "return") {
        return Some(Err(usage));
    }
    let p = match &*place.expr {
        Expr::Identifier(id) => id.name.as_ref(),
        Expr::MemberAccess(_) => {
            let path = crate::hir::prover::to_source(&place.expr).unwrap_or_default();
            return Some(Err(format!(
                "a clause about a field, such as `*{path}`, is not supported yet: the checker \
                 keeps one set of borrows for a struct value"
            )));
        }
        _ => return Some(Err(usage)),
    };
    let Some(param) = f.params.iter().position(|(n, _)| n.as_ref() == p) else {
        return Some(Err(format!("`{p}` is not a parameter of `{shown}`")));
    };
    let ty = &f.params[param].1;
    if !matches!(ty, Type::Borrow { .. }) {
        return Some(Err(format!(
            "`{name}` needs a reference parameter after `*`, but `{p}` is `{ty}`"
        )));
    }
    if param >= 32 {
        return Some(Err(format!(
            "a borrow clause can name only the first 32 parameters, and `{p}` is not one of them"
        )));
    }
    Some(Ok(ResultClause {
        relation,
        param,
        negated,
    }))
}

/// The well formed clauses about the result of `f`.
fn result_clauses(f: &Function) -> Vec<ResultClause> {
    f.ensures
        .iter()
        .filter_map(|e| read_clause(f, "", e)?.ok())
        .collect()
}

/// Two clauses about one parameter that cannot both hold: a clause without `!` and a `!` clause
/// that rules it out. A result that points into `*p` also reaches it.
fn contradiction(clauses: &[ResultClause]) -> Option<(ResultClause, ResultClause)> {
    for pos in clauses.iter().filter(|c| !c.negated) {
        for neg in clauses.iter().filter(|c| c.negated && c.param == pos.param) {
            if pos.relation == Relation::PointsInto || neg.relation == Relation::Reaches {
                return Some((*pos, *neg));
            }
        }
    }
    None
}

/// The summary of `f` with its written clauses. For each parameter that a written clause names,
/// the written clauses replace the inferred ones. A clause that is not well formed adds nothing,
/// and the checker reports it.
pub(crate) fn apply_written_clauses(f: &Function, inferred: ReturnProvenance) -> ReturnProvenance {
    let clauses = result_clauses(f);
    // The bit sets hold 32 parameters, and a summary with more is `AnyParam`.
    if clauses.is_empty()
        || inferred == ReturnProvenance::NotAReference
        || f.params.len() > 32
        || contradiction(&clauses).is_some()
    {
        return inferred;
    }
    let all = u32::MAX >> (32 - f.params.len());
    let (mut into, mut held) = match inferred {
        ReturnProvenance::FromParams { into, held } => (into, held),
        _ => (all, all),
    };
    for c in &clauses {
        into |= 1 << c.param;
        held |= 1 << c.param;
    }
    for c in clauses.iter().filter(|c| c.negated) {
        into &= !(1 << c.param);
        if c.relation == Relation::Reaches {
            held &= !(1 << c.param);
        }
    }
    ReturnProvenance::FromParams { into, held }
}

impl<'a> TypeChecker<'a> {
    /// E4016 for the borrow clauses of `f`: each one is well formed, and the body keeps each `!`
    /// clause. Run after the body is checked, when its method calls are calls by name, so the
    /// summary walk can follow them.
    pub(crate) fn check_borrow_clauses(&mut self, f: &Function) {
        if self.speculating {
            return;
        }
        // An instance of a generic function is named by its template in a message.
        let shown = match f.name.split_once('$') {
            Some((template, _)) if self.env.generic_functions.contains_key(template) => template,
            _ => f.name.as_ref(),
        };
        for e in f.requires.iter().filter(|e| mentions_borrow_function(e)) {
            self.borrow_clause_error(
                "a borrow clause says what a call does, so it belongs in `ensures`, not in \
                 `requires`"
                    .to_string(),
                e.span(),
            );
        }
        let mut clauses = Vec::new();
        for e in &f.ensures {
            if !is_borrow_clause(e) {
                if let Some(at) = first_borrow_call(e) {
                    self.borrow_clause_error(
                        "write each borrow clause as an `ensures` of its own, as in \
                         `ensures !reaches(return, *p)`"
                            .to_string(),
                        at,
                    );
                }
                continue;
            }
            match read_clause(f, shown, e) {
                Some(Ok(c)) => clauses.push((c, e.span())),
                Some(Err(message)) => self.borrow_clause_error(message, e.span()),
                None => {}
            }
        }
        let Some(&(_, first)) = clauses.first() else {
            return;
        };
        if !crate::hir::provenance::can_hold_reference(&f.return_type) {
            self.borrow_clause_error(
                format!(
                    "`{shown}` returns `{}`, which holds no reference, so a clause about where its \
                     result points says nothing",
                    f.return_type
                ),
                first,
            );
            return;
        }
        let only: Vec<ResultClause> = clauses.iter().map(|(c, _)| *c).collect();
        if let Some((pos, neg)) = contradiction(&only) {
            let at = clauses
                .iter()
                .find(|(c, _)| c.negated && c.param == neg.param && c.relation == neg.relation)
                .map_or(first, |(_, s)| *s);
            self.borrow_clause_error(
                format!(
                    "`{}` and `{}` contradict each other",
                    pos.text(f),
                    neg.text(f)
                ),
                at,
            );
            return;
        }
        if !clauses.iter().any(|(c, _)| c.negated) {
            return;
        }
        // Where the body lets the result point. A call reads the summary of its callee, written
        // clauses included. So a recursive call reads the clauses of `f` itself.
        let body = {
            let instances = &self.mono.return_provenances;
            let functions = &self.env.return_provenances;
            let callee = |name: &str| instances.get(name).or_else(|| functions.get(name)).copied();
            crate::hir::provenance::infer_return_provenance(f, &self.env.decls, &callee)
        };
        for (c, at) in clauses.iter().filter(|(c, _)| c.negated) {
            let p = &f.params[c.param].0;
            let bit = 1u32 << c.param;
            let why = match body {
                // A result with no reference keeps each clause. A result that points into a
                // local is an E4005 in `f`.
                ReturnProvenance::NotAReference | ReturnProvenance::Local => continue,
                ReturnProvenance::AnyParam => format!(
                    "the checker cannot show it: it cannot follow where the result of `{shown}` \
                     points"
                ),
                ReturnProvenance::FromParams { into, .. } if into & bit != 0 => {
                    format!("the result can point into `*{p}`")
                }
                ReturnProvenance::FromParams { held, .. }
                    if c.relation == Relation::Reaches && held & bit != 0 =>
                {
                    format!(
                        "the result can point into memory that `*{p}` reaches through a reference"
                    )
                }
                ReturnProvenance::FromParams { .. } => continue,
            };
            self.borrow_clause_error(format!("`{shown}` says `{}`, but {why}", c.text(f)), *at);
        }
    }

    fn borrow_clause_error(&mut self, message: String, at: Span) {
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4016,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
        );
    }
}
