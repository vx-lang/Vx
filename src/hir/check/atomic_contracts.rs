//===- atomic_contracts.rs - Vx Compiler ----------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Atomic contracts: `ensures atomic(t)` on a function and `ensures : [atomic(t)]` on an `mlir!`
// block. Each one says that every access to the elements of the tensor `t` is an atomic
// operation. The race proof of an `across` loop reads the contract in place of the body: two
// atomic accesses never race. docs/discussions/mlir_declared_writes_in_across.md has the design.
//
//===----------------------------------------------------------------------===//

use crate::diagnostic::{DiagnosticCode, SourceSpan};
use crate::syntax::decl::Function;
use crate::syntax::expr::InlineMlirExpr;
use crate::syntax::stmt::Statement;
use crate::syntax::{ElementType, Expr, Span, Type};
use std::collections::{HashMap, HashSet};

/// Whether `e` is an atomic clause, well formed or not. The prover does not read these.
pub(crate) fn is_atomic_clause(e: &Expr) -> bool {
    matches!(e, Expr::FunctionCall(fc) if fc.name.as_ref() == "atomic")
}

/// The tensor that the clause `atomic(t)` names.
fn atomic_name(e: &Expr) -> Option<&str> {
    match e {
        Expr::FunctionCall(fc) if fc.name.as_ref() == "atomic" && fc.args.len() == 1 => {
            match &fc.args[0] {
                Expr::Identifier(id) => Some(id.name.as_ref()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// The parameters of `f` that its clauses name, by position.
pub(crate) fn atomic_params(f: &Function) -> Vec<usize> {
    let names: HashSet<&str> = f.ensures.iter().filter_map(atomic_name).collect();
    f.params
        .iter()
        .enumerate()
        .filter(|(_, (p, _))| names.contains(p.as_ref()))
        .map(|(i, _)| i)
        .collect()
}

/// The names that the clauses of an `mlir!` block name.
pub(crate) fn atomic_inputs(m: &InlineMlirExpr) -> HashSet<&str> {
    m.ensures.iter().filter_map(atomic_name).collect()
}

/// The tensor that an argument passes: `t` or `&t`.
pub(crate) fn passed_tensor(e: &Expr) -> Option<&str> {
    match e {
        Expr::Identifier(id) => Some(id.name.as_ref()),
        Expr::Borrow(b) if !b.is_mut => match &*b.expr {
            Expr::Identifier(id) => Some(id.name.as_ref()),
            _ => None,
        },
        _ => None,
    }
}

/// Whether `ty` is a type that an atomic clause can name: a `&` tensor of a type that has
/// atomic operations.
fn atomic_tensor(ty: &Type) -> bool {
    match ty {
        Type::Borrow {
            inner,
            is_mut: false,
            ..
        } => matches!(
            inner.as_ref(),
            Type::Tensor(
                ElementType::I32
                    | ElementType::U32
                    | ElementType::I64
                    | ElementType::U64
                    | ElementType::F32,
                ..
            )
        ),
        _ => false,
    }
}

/// The MLIR operations that read or write a memref without an atomic operation.
const PLAIN_ACCESSES: &[&str] = &[
    "memref.load",
    "memref.store",
    "memref.copy",
    "vector.load",
    "vector.store",
    "vector.transfer_read",
    "vector.transfer_write",
];

/// The MLIR operations that write a memref.
const WRITES: &[&str] = &[
    "memref.store",
    "memref.copy",
    "memref.atomic_rmw",
    "memref.generic_atomic_rmw",
    "vector.store",
    "vector.transfer_write",
];

/// Whether a line of MLIR uses `%name` as an operand of one of `ops`.
fn line_uses(line: &str, ops: &[&str], name: &str) -> bool {
    if !ops.iter().any(|op| line.contains(op)) {
        return false;
    }
    let mut rest = line;
    while let Some(at) = rest.find(name) {
        let after = rest[at + name.len()..].chars().next();
        if !after.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            return true;
        }
        rest = &rest[at + name.len()..];
    }
    false
}

impl crate::hir::TypeChecker<'_> {
    fn atomic_error(&mut self, message: String, at: Span) {
        let span = (at != Span::default()).then(|| SourceSpan::from_ast_span(&at));
        self.errors
            .error_with_code(DiagnosticCode::E8007, message, span);
    }

    /// E8007 for the atomic clauses of `f`: each one names a `&` tensor parameter, and the body
    /// uses that tensor only in atomic operations. An atomic operation is an `mlir!` block whose
    /// clauses name the tensor, or a call whose callee has a clause for that argument.
    pub(crate) fn check_atomic_contract(&mut self, f: &Function) {
        let clauses: Vec<&Expr> = f.ensures.iter().filter(|e| is_atomic_clause(e)).collect();
        if clauses.is_empty() || self.speculating {
            return;
        }
        // An instance of a generic function is named by its template in a message.
        let shown = match f.name.split_once('$') {
            Some((template, _)) if self.env.generic_functions.contains_key(template) => template,
            _ => f.name.as_ref(),
        };
        let mut names = HashSet::new();
        for clause in clauses {
            let Some(name) = atomic_name(clause) else {
                self.atomic_error(
                    format!(
                        "`atomic` takes one parameter of `{}`: the tensor whose elements it \
                         changes only with atomic operations",
                        shown
                    ),
                    clause.span(),
                );
                continue;
            };
            match f.params.iter().find(|(p, _)| p.as_ref() == name) {
                None => self.atomic_error(
                    format!("`atomic({name})` names no parameter of `{shown}`"),
                    clause.span(),
                ),
                Some((_, ty)) if !atomic_tensor(ty) => self.atomic_error(
                    format!(
                        "`atomic({name})` must name a `&` tensor of `i32`, `u32`, `i64`, `u64` \
                         or `f32`, but `{name}` is `{ty}`"
                    ),
                    clause.span(),
                ),
                Some(_) => {
                    names.insert(name.to_string());
                }
            }
        }
        if names.is_empty() {
            return;
        }
        let contracts = self.atomic_contracts();
        for (name, at) in non_atomic_uses(&f.body, &names, &contracts) {
            self.atomic_error(
                format!(
                    "`{}` says `atomic({name})`, so it may use `{name}` only in an atomic \
                     operation: an `mlir!` block with `ensures : [atomic({name})]`, or a call to a \
                     function that says `atomic` for that argument",
                    shown
                ),
                at,
            );
        }
    }

    /// E8007 for the clauses of an `mlir!` block. Each one names an input that is a tensor. The
    /// block reads and writes that tensor only with atomic operations, and it writes no other
    /// tensor input.
    pub(crate) fn check_atomic_block(&mut self, m: &InlineMlirExpr) {
        if m.ensures.is_empty() || self.speculating {
            return;
        }
        let mut named = HashSet::new();
        for clause in &m.ensures {
            let input = atomic_name(clause).and_then(|name| {
                m.inputs
                    .iter()
                    .find(|(_, e, ty)| passed_tensor(e) == Some(name) && ty.starts_with("memref"))
                    .map(|(sym, _, _)| (name, sym.as_ref().to_string()))
            });
            match input {
                Some((name, sym)) => {
                    named.insert(sym.clone());
                    if let Some(line) = m
                        .block_str
                        .lines()
                        .find(|l| line_uses(l, PLAIN_ACCESSES, &sym))
                    {
                        self.atomic_error(
                            format!(
                                "the block says `atomic({name})`, but `{}` reads or writes `{sym}` \
                                 without an atomic operation",
                                line.trim()
                            ),
                            m.span,
                        );
                    }
                }
                None => self.atomic_error(
                    "an `atomic` clause of an `mlir!` block names one of its inputs, a tensor \
                     passed by name, such as `atomic(t)` for `%t = t : memref<?xi32>`"
                        .to_string(),
                    clause.span(),
                ),
            }
        }
        for (sym, _, ty) in &m.inputs {
            let sym = sym.as_ref();
            if !ty.starts_with("memref") || named.contains(sym) {
                continue;
            }
            if let Some(line) = m.block_str.lines().find(|l| line_uses(l, WRITES, sym)) {
                self.atomic_error(
                    format!(
                        "`{}` writes `{sym}`, which the clauses of the block do not name",
                        line.trim()
                    ),
                    m.span,
                );
            }
        }
    }

    /// The parameters with an atomic clause, for each function that has one. A call to a generic
    /// function names the template before its body is checked and the instance after, so both
    /// are here.
    pub(crate) fn atomic_contracts(&self) -> HashMap<String, Vec<usize>> {
        let syntax = self.env.syntax_functions.iter().map(|(n, f)| (n, *f));
        let templates = self.env.generic_functions.iter().map(|(n, (f, _))| (n, *f));
        let instances = self.mono.functions.iter().map(|(f, _)| (&f.name, f));
        syntax
            .chain(templates)
            .chain(instances)
            .filter_map(|(name, f)| {
                let params = atomic_params(f);
                (!params.is_empty()).then(|| (name.as_ref().to_string(), params))
            })
            .collect()
    }
}

/// Each use of a name in `names` that is not an atomic operation, with where it is.
fn non_atomic_uses(
    body: &[Statement],
    names: &HashSet<String>,
    contracts: &HashMap<String, Vec<usize>>,
) -> Vec<(String, Span)> {
    // The identifiers that an atomic operation passes, by address in the tree.
    let mut atomic: HashSet<*const Expr> = HashSet::new();
    let mark = |e: &Expr, atomic: &mut HashSet<*const Expr>| match e {
        Expr::Identifier(_) => {
            atomic.insert(e as *const Expr);
        }
        Expr::Borrow(b) => {
            atomic.insert(&*b.expr as *const Expr);
        }
        _ => {}
    };
    crate::syntax::visit_exprs(body, &mut |e| match e {
        Expr::InlineMlir(m) => {
            let clauses = atomic_inputs(m);
            for (_, input, _) in &m.inputs {
                if passed_tensor(input).is_some_and(|n| clauses.contains(n)) {
                    mark(input, &mut atomic);
                }
            }
        }
        Expr::FunctionCall(fc) => {
            if let Some(params) = contracts.get(fc.name.as_ref()) {
                for &i in params {
                    if let Some(arg) = fc.args.get(i).filter(|a| passed_tensor(a).is_some()) {
                        mark(arg, &mut atomic);
                    }
                }
            }
        }
        _ => {}
    });
    let mut uses = Vec::new();
    crate::syntax::visit_exprs(body, &mut |e| {
        if let Expr::Identifier(id) = e {
            if names.contains(id.name.as_ref()) && !atomic.contains(&(e as *const Expr)) {
                uses.push((id.name.as_ref().to_string(), id.span));
            }
        }
    });
    uses
}
