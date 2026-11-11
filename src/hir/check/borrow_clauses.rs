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
// `reaches(return, *p)` also counts the memory that `*p` reaches through references.
// `keeps(b)` says that a place outside the call can hold `b` after it, or a reference read
// through `b`. A clause with `!` narrows what a caller assumes, and the checker proves it against
// the body. A clause without `!` keeps the default for its parameter. The written clauses about
// the result replace, for each parameter they name, what the return summary (provenance.rs)
// infers for it.
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
    Keeps,
}

/// `points_into(return, *p)`, `reaches(return, *p)` or `keeps(p)`, with or without `!`.
#[derive(Clone, Copy, Debug)]
struct Clause {
    relation: Relation,
    /// The position of `p` among the parameters.
    param: usize,
    negated: bool,
}

impl Clause {
    fn text(&self, f: &Function) -> String {
        let not = if self.negated { "!" } else { "" };
        let p = &f.params[self.param].0;
        match self.relation {
            Relation::PointsInto => format!("{not}points_into(return, *{p})"),
            Relation::Reaches => format!("{not}reaches(return, *{p})"),
            Relation::Keeps => format!("{not}keeps({p})"),
        }
    }

    fn about_the_result(&self) -> bool {
        self.relation != Relation::Keeps
    }
}

/// Read `e`, an `ensures` of `f`. `None` when `e` is not a borrow clause, and a message when it
/// is one that is not well formed. `shown` names `f` in the message.
fn read_clause(f: &Function, shown: &str, e: &Expr) -> Option<Result<Clause, String>> {
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
        "keeps" => Relation::Keeps,
        _ => return None,
    };
    let name = fc.name.as_ref();
    let p = if relation == Relation::Keeps {
        let usage = "`keeps` takes one reference parameter, as in `keeps(b)`".to_string();
        match fc.args.as_slice() {
            [Expr::Identifier(id)] => id.name.as_ref(),
            _ => return Some(Err(usage)),
        }
    } else {
        let usage = format!(
            "`{name}` takes `return` and a reference parameter after `*`, as in \
             `{name}(return, *p)`"
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
        match &*place.expr {
            Expr::Identifier(id) => id.name.as_ref(),
            Expr::MemberAccess(_) => {
                let path = crate::hir::prover::to_source(&place.expr).unwrap_or_default();
                return Some(Err(format!(
                    "a clause about a field, such as `*{path}`, is not supported yet: the \
                     checker keeps one set of borrows for a struct value"
                )));
            }
            _ => return Some(Err(usage)),
        }
    };
    let Some(param) = f.params.iter().position(|(n, _)| n.as_ref() == p) else {
        return Some(Err(format!("`{p}` is not a parameter of `{shown}`")));
    };
    let ty = &f.params[param].1;
    if !matches!(ty, Type::Borrow { .. }) {
        return Some(Err(format!(
            "`{name}` needs a reference parameter, but `{p}` is `{ty}`"
        )));
    }
    if param >= 32 {
        return Some(Err(format!(
            "a borrow clause can name only the first 32 parameters, and `{p}` is not one of them"
        )));
    }
    Some(Ok(Clause {
        relation,
        param,
        negated,
    }))
}

/// The well formed borrow clauses of `f`.
fn clauses(f: &Function) -> Vec<Clause> {
    f.ensures
        .iter()
        .filter_map(|e| read_clause(f, "", e)?.ok())
        .collect()
}

/// Two clauses about one parameter that cannot both hold: a clause without `!` and a `!` clause
/// that rules it out. A result that points into `*p` also reaches it.
fn contradiction(clauses: &[Clause]) -> Option<(Clause, Clause)> {
    for pos in clauses.iter().filter(|c| !c.negated) {
        for neg in clauses.iter().filter(|c| c.negated && c.param == pos.param) {
            let rules_out = match pos.relation {
                Relation::Keeps => neg.relation == Relation::Keeps,
                Relation::PointsInto => neg.about_the_result(),
                Relation::Reaches => neg.relation == Relation::Reaches,
            };
            if rules_out {
                return Some((*pos, *neg));
            }
        }
    }
    None
}

/// `e` with each parameter name in `names` replaced: a trait's clause written with an impl's
/// names. A clause uses names, `!`, `*`, calls, fields and operators.
pub(crate) fn rename_params(
    e: &Expr,
    names: &std::collections::HashMap<crate::symbol::Symbol, crate::symbol::Symbol>,
) -> Expr {
    let sub = |x: &Expr| Box::new(rename_params(x, names));
    match e {
        Expr::Identifier(id) => match names.get(&id.name) {
            Some(to) => Expr::Identifier(crate::syntax::IdentifierExpr {
                name: to.clone(),
                span: id.span,
            }),
            None => e.clone(),
        },
        Expr::UnaryOp(u) => {
            let mut u = u.clone();
            u.expr = sub(&u.expr);
            Expr::UnaryOp(u)
        }
        Expr::Dereference(d) => {
            let mut d = d.clone();
            d.expr = sub(&d.expr);
            Expr::Dereference(d)
        }
        Expr::FunctionCall(fc) => {
            let mut fc = fc.clone();
            fc.args = fc.args.iter().map(|a| rename_params(a, names)).collect();
            Expr::FunctionCall(fc)
        }
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
        Expr::MemberAccess(m) => {
            let mut m = m.clone();
            m.base = sub(&m.base);
            Expr::MemberAccess(m)
        }
        _ => e.clone(),
    }
}

/// The parameters that `f` says `!keeps` for, as bits. A call stores no argument for them.
pub(crate) fn not_kept_params(f: &Function) -> u32 {
    let clauses = clauses(f);
    if contradiction(&clauses).is_some() {
        return 0;
    }
    clauses
        .iter()
        .filter(|c| c.negated && c.relation == Relation::Keeps)
        .fold(0, |bits, c| bits | 1 << c.param)
}

/// The `!keeps` bits of `f` that the walk proves from its body, for each reference parameter that
/// no written `keeps` clause names: the body stores it nowhere. `not_kept` gives the bits known so
/// far for each function that the body calls, and `callee` its summary.
pub(crate) fn inferred_not_kept(
    f: &Function,
    decls: &crate::hir::provenance::Decls,
    callee: &dyn Fn(&str) -> Option<ReturnProvenance>,
    not_kept: &dyn Fn(&str) -> u32,
) -> u32 {
    if f.body.is_empty() || f.params.len() > 32 {
        return 0;
    }
    let written = clauses(f)
        .iter()
        .filter(|c| c.relation == Relation::Keeps)
        .fold(0u32, |bits, c| bits | 1 << c.param);
    let candidates = f
        .params
        .iter()
        .enumerate()
        .filter(|(i, (_, ty))| matches!(ty, Type::Borrow { .. }) && written & (1 << i) == 0)
        .fold(0u32, |bits, (i, _)| bits | 1 << i);
    if candidates == 0 {
        return 0;
    }
    let body = crate::hir::provenance::walk_body(f, decls, callee, not_kept);
    if body.stored_anywhere {
        return 0;
    }
    candidates & !body.stored
}

/// The `!keeps` bits of each function in `funcs`: its written ones, `written`, and those that the
/// walk proves. A bit that the walk proves lets a caller's walk prove more, so the walk runs again
/// until nothing changes. Each pass reads only bits proved before it, so each bit stays sound.
pub(crate) fn infer_not_kept_all(
    funcs: &[&Function],
    decls: &crate::hir::provenance::Decls,
    summaries: &std::collections::HashMap<crate::symbol::Symbol, ReturnProvenance>,
    mut bits: std::collections::HashMap<crate::symbol::Symbol, u32>,
) -> std::collections::HashMap<crate::symbol::Symbol, u32> {
    for _ in 0..16 {
        let mut changed = false;
        for f in funcs {
            let inferred = {
                let callee = |name: &str| summaries.get(name).copied();
                let not_kept = |name: &str| bits.get(name).copied().unwrap_or(0);
                inferred_not_kept(f, decls, &callee, &not_kept)
            };
            let old = bits.get(&f.name).copied().unwrap_or(0);
            if old | inferred != old {
                bits.insert(f.name.clone(), old | inferred);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    bits
}

/// The summary of `f` with its written clauses. For each parameter that a written clause names,
/// the written clauses replace the inferred ones. A clause that is not well formed adds nothing,
/// and the checker reports it.
pub(crate) fn apply_written_clauses(f: &Function, inferred: ReturnProvenance) -> ReturnProvenance {
    let all_clauses = clauses(f);
    if contradiction(&all_clauses).is_some() {
        return inferred;
    }
    let clauses: Vec<Clause> = all_clauses
        .into_iter()
        .filter(Clause::about_the_result)
        .collect();
    // The bit sets hold 32 parameters, and a summary with more is `AnyParam`.
    if clauses.is_empty() || inferred == ReturnProvenance::NotAReference || f.params.len() > 32 {
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
        let shown = self.clause_owner_name(f);
        let shown = shown.as_str();
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
        if let Some(&(_, at)) = clauses.iter().find(|(c, _)| c.about_the_result()) {
            if !crate::hir::provenance::can_hold_reference(&f.return_type) {
                self.borrow_clause_error(
                    format!(
                        "`{shown}` returns `{}`, which holds no reference, so a clause about \
                         where its result points says nothing",
                        f.return_type
                    ),
                    at,
                );
                return;
            }
        }
        let only: Vec<Clause> = clauses.iter().map(|(c, _)| *c).collect();
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
        // Where the body lets the result point, and what it stores. A call reads the summary and
        // the `!keeps` clauses of its callee. So a recursive call reads the clauses of `f` itself.
        let body = {
            let instances = &self.mono.return_provenances;
            let functions = &self.env.return_provenances;
            let callee = |name: &str| instances.get(name).or_else(|| functions.get(name)).copied();
            let not_kept = |name: &str| self.not_kept_by(name);
            crate::hir::provenance::walk_body(f, &self.env.decls, &callee, &not_kept)
        };
        for (c, at) in clauses.iter().filter(|(c, _)| c.negated) {
            let p = &f.params[c.param].0;
            let bit = 1u32 << c.param;
            if c.relation == Relation::Keeps {
                let why = if body.stored & bit != 0 {
                    format!(
                        "the body can store `{p}`, or a reference read through it, where it \
                         outlives the call"
                    )
                } else if body.stored_anywhere {
                    "the checker cannot show it: the body can store a value that the checker \
                     cannot follow"
                        .to_string()
                } else {
                    continue;
                };
                self.borrow_clause_error(format!("`{shown}` says `{}`, but {why}", c.text(f)), *at);
                continue;
            }
            let why = match body.result {
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

    /// `vxc --print-borrow-clauses`: print each `!` clause that the walk proves for `f`, as
    /// `f: ensures !reaches(return, *key)`. An instance of a generic function is not printed.
    pub(crate) fn print_proved_borrow_clauses(&self, f: &Function) {
        if self.print_borrow_clauses.is_none()
            || f.file != self.print_borrow_clauses
            || f.name.contains('$')
            || f.params.len() > 32
        {
            return;
        }
        let body = {
            let instances = &self.mono.return_provenances;
            let functions = &self.env.return_provenances;
            let callee = |name: &str| instances.get(name).or_else(|| functions.get(name)).copied();
            let not_kept = |name: &str| self.not_kept_by(name);
            crate::hir::provenance::walk_body(f, &self.env.decls, &callee, &not_kept)
        };
        let references: Vec<(usize, &crate::symbol::Symbol)> = f
            .params
            .iter()
            .enumerate()
            .filter(|(_, (_, ty))| matches!(ty, Type::Borrow { .. }))
            .map(|(i, (p, _))| (i, p))
            .collect();
        let mut proved = Vec::new();
        if let ReturnProvenance::FromParams { into, held } = body.result {
            for &(i, p) in &references {
                if (into | held) & (1 << i) == 0 {
                    proved.push(format!("!reaches(return, *{p})"));
                } else if into & (1 << i) == 0 {
                    proved.push(format!("!points_into(return, *{p})"));
                }
            }
        }
        // `!keeps(p)` helps a caller only when the call gets something else that a reference can
        // be stored through.
        let stores_through: Vec<bool> = f
            .params
            .iter()
            .map(|(_, ty)| self.env.decls.can_store_through(ty, 0))
            .collect();
        for &(i, p) in &references {
            let elsewhere = (0..f.params.len()).any(|j| j != i && stores_through[j]);
            if elsewhere && !body.stored_anywhere && body.stored & (1 << i) == 0 {
                proved.push(format!("!keeps({p})"));
            }
        }
        for clause in proved {
            println!("{}: ensures {clause}", f.name);
        }
    }

    /// The parameters that the function `name` says `!keeps` for, as bits. A call stores no
    /// argument for them.
    pub(crate) fn not_kept_by(&self, name: &str) -> u32 {
        let name = match name.find('<') {
            Some(i) if name.ends_with('>') => &name[..i],
            _ => name,
        };
        self.mono
            .not_kept
            .get(name)
            .or_else(|| self.env.not_kept.get(name))
            .copied()
            .unwrap_or(0)
    }

    /// How a message names `f`. An instance of a generic function is named by its template. A
    /// method is named by its type, as in `Wrong::first`, because an impl can break a clause
    /// that its trait wrote, and the clause then points into the trait.
    pub(crate) fn clause_owner_name(&self, f: &Function) -> String {
        if let Some((template, _)) = f.name.split_once('$') {
            if self.env.generic_functions.contains_key(template) {
                return template.to_string();
            }
        }
        let owner = match f.params.first() {
            Some((name, ty)) if name.as_ref() == "self" => match ty {
                Type::Borrow { inner, .. } => inner.as_ref(),
                ty => ty,
            },
            _ => return f.name.to_string(),
        };
        // The instance of a method is named for its receiver: `Wrong$first`. A `$` or the end
        // follows the method name, so `get` does not match `get_ref`.
        let in_name = |m: &str| {
            f.name.as_ref() == m
                || f.name.ends_with(&format!("${m}"))
                || f.name.contains(&format!("${m}$"))
        };
        fn base_name(ty: &Type) -> Option<&str> {
            match ty {
                Type::GenericInstance(base, _) => base_name(base),
                Type::Struct(name, _) | Type::Enum(name, _) => Some(name.as_ref()),
                _ => None,
            }
        }
        let method = self
            .env
            .impls
            .values()
            .flatten()
            .filter(|ib| {
                base_name(&ib.target_type).is_some()
                    && base_name(&ib.target_type) == base_name(owner)
            })
            .flat_map(|ib| ib.methods.iter())
            .map(|m| m.name.as_ref())
            .filter(|m| in_name(m))
            .max_by_key(|m| m.len());
        format!("{owner}::{}", method.unwrap_or(f.name.as_ref()))
    }

    /// E4016, once for each clause, type and reason: an impl method is checked once as written
    /// and once for each call that makes its instance.
    fn borrow_clause_error(&mut self, message: String, at: Span) {
        if !self.borrow_clause_errors.insert((at, message.clone())) {
            return;
        }
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4016,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
        );
    }
}
