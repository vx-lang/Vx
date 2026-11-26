//===- extent_names.rs - Vx Compiler ---------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `?n` in the tensor type of a parameter gives that run-time extent the name `n`
// (`docs/lang/types.md`). In the body and the contracts, `n` is a `usize` that nothing changes.
//
// The checker keeps `n` as a name, so that the prover knows what the `requires` says about it.
// It checks each read of `n` as a read of `t.extent(i)` at that place, so a move of `t`, a borrow
// of it and a closure see a read of `t`. After the check, each `n` becomes that read, so the
// code generators need nothing new. At a call, the callee's `n` is the extent of the argument.
//
//===----------------------------------------------------------------------===//

use std::collections::HashMap;

use crate::symbol::Symbol;
use crate::syntax::expr::{
    AsCastExpr, Expr, IdentifierExpr, IndexAccessExpr, MemberAccessExpr, MethodCallExpr, NumberExpr,
};
use crate::syntax::types::{ElementType, Type};
use crate::syntax::{Dim, GenericParam, Span, Statement};

/// `t.extent(axis)`, as a program writes it.
fn extent_method(tensor: &Symbol, axis: usize, span: Span) -> Expr {
    Expr::MethodCall(MethodCallExpr::new(
        Box::new(Expr::Identifier(IdentifierExpr {
            name: tensor.clone(),
            span,
        })),
        "extent".into(),
        None,
        vec![Expr::Number(NumberExpr::new(
            axis.to_string(),
            Some(ElementType::I32),
            span,
        ))],
        span,
    ))
}

/// `t.extent(axis) as usize` after the check: the read that both code generators lower.
fn extent_value(tensor: &Symbol, axis: usize, span: Span) -> Expr {
    let read = Expr::IndexAccess(IndexAccessExpr::new(
        Box::new(Expr::MemberAccess(MemberAccessExpr::new(
            Box::new(Expr::Identifier(IdentifierExpr {
                name: tensor.clone(),
                span,
            })),
            "$extent".into(),
            span,
        ))),
        Box::new(Expr::Number(NumberExpr::new(
            axis.to_string(),
            Some(ElementType::I32),
            span,
        ))),
        span,
    ));
    Expr::AsCast(AsCastExpr {
        expr: Box::new(read),
        target_ty: Type::Scalar(ElementType::USize),
        source_ty: Some(Type::Scalar(ElementType::I32)),
        span,
    })
}

/// The value of each extent name of a callee at a call, and the names that the call gives two
/// different extents. See [`crate::hir::TypeChecker::callee_extent_values`].
pub(crate) struct CalleeExtents {
    pub(crate) values: HashMap<Symbol, Expr>,
    pub(crate) mismatches: Vec<(Symbol, String, String)>,
}

impl crate::hir::TypeChecker<'_> {
    /// W1039: an extent name that a constant or a const parameter also has. An instance of a
    /// generic function gives the same warning again, which `drop_repeats` drops.
    pub(crate) fn warn_extent_names_that_hide(
        &mut self,
        params: &[(Symbol, Type)],
        generics: &[GenericParam],
        at: Span,
    ) {
        if self.speculating {
            return;
        }
        let mut seen = Vec::new();
        for (name, p, _) in crate::syntax::types::extent_names(params) {
            if seen.contains(&name) {
                continue;
            }
            seen.push(name.clone());
            let what = if self.env.consts.contains_key(&name) {
                "a constant"
            } else if generics
                .iter()
                .any(|g| matches!(g, GenericParam::Const { name: c, .. } if *c == name))
            {
                "a const parameter"
            } else {
                continue;
            };
            self.errors.warn(
                crate::diagnostic::DiagnosticCode::W1039,
                format!(
                    "`?{name}` names an extent of `{}`, so in this function `{name}` is that \
                     extent and not {what} with the name `{name}`",
                    params[p].0
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&at)),
            );
        }
    }

    /// Declares the extent names of `params` in the scope of the body: each is a `usize` that
    /// nothing changes.
    pub(crate) fn declare_extent_names(&mut self, params: &[(Symbol, Type)]) {
        let mut homes = HashMap::new();
        for (name, p, axis) in crate::syntax::types::extent_names(params) {
            if homes.contains_key(&name) {
                continue;
            }
            self.insert(name.to_string(), Type::Scalar(ElementType::USize));
            homes.insert(name, (params[p].0.clone(), axis));
        }
        self.extent_homes = homes;
    }

    /// A read of the extent name `name` at `span`, checked as `t.extent(i)`. `None` for any other
    /// name.
    pub(crate) fn check_extent_name_read(&mut self, name: &Symbol, span: Span) -> Option<Type> {
        let (tensor, axis) = self.extent_homes.get(name).cloned()?;
        self.used_vars.insert(name.clone());
        if self.is_moved(tensor.as_ref()) {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E4001,
                    format!(
                        "`{name}` is an extent of `{tensor}`, and `{tensor}` was moved before \
                         this read. Read `{name}` into a variable before the move"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                );
            }
            return Some(Type::Scalar(ElementType::USize));
        }
        let mut read = extent_method(&tensor, axis, span);
        self.check_expr_type_flag(&mut read, false);
        Some(Type::Scalar(ElementType::USize))
    }

    /// E3069: a variable of the body with the name of an extent.
    pub(crate) fn refuse_a_variable_with_an_extent_name(&mut self, name: &str) {
        let Some((tensor, _)) = self.extent_homes.get(name) else {
            return;
        };
        if !self.speculating {
            let tensor = tensor.clone();
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3069,
                format!(
                    "`{name}` is the name of an extent of `{tensor}` (`?{name}`), so no variable \
                     of the body can have this name"
                ),
                None,
            );
        }
    }

    /// E3069: an assignment to an extent name.
    pub(crate) fn refuse_an_assignment_to_an_extent_name(&mut self, name: &str, span: &Span) {
        if let Some((tensor, _)) = self.extent_homes.get(name).cloned() {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3069,
                    format!(
                        "`{name}` is an extent of `{tensor}` (`?{name}`), and nothing can change it"
                    ),
                    Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
                );
            }
        }
    }

    /// After the check of a body: each read of an extent name becomes the read of the extent.
    pub(crate) fn rewrite_extent_names(&self, body: &mut [Statement]) {
        if !self.extent_homes.is_empty() {
            crate::syntax::expr::visit_exprs_mut(body, &mut |e| self.rewrite_extent_name(e));
        }
    }

    /// [`Self::rewrite_extent_names`] for the body of a closure, which becomes a function of
    /// its own when the checker checks it.
    pub(crate) fn rewrite_extent_names_in(&self, e: &mut Expr) {
        if !self.extent_homes.is_empty() {
            crate::syntax::expr::visit_expr_mut(e, &mut |e| self.rewrite_extent_name(e));
        }
    }

    fn rewrite_extent_name(&self, e: &mut Expr) {
        if let Expr::Identifier(id) = e {
            if let Some((tensor, axis)) = self.extent_homes.get(&id.name) {
                *e = extent_value(tensor, *axis, id.span);
            }
        }
    }

    /// The extent of axis `axis` of the tensor that the argument `arg` gives: a number, or the
    /// caller's own extent name. `None` when the caller does not know it.
    fn extent_of_argument(&self, arg: &Expr, axis: usize) -> Option<Expr> {
        let arg = crate::hir::prover::without_borrow(arg);
        // The name of the caller's extent comes only from a parameter of the caller that no
        // variable hides. A type found any other way can keep a name that is no longer true,
        // as the type of a slice of the tensor can.
        if let Expr::Identifier(id) = arg {
            let hidden = self.declared_vars.iter().any(|(v, _)| *v == id.name);
            if let Some(ty) = self.borrow.current_params.get(&id.name) {
                if let (Type::Tensor(_, dims, _), false) = (ty.behind_refs(), hidden) {
                    if let Some(Dim::Dyn(Some(m))) = dims.get(axis) {
                        if self.extent_homes.contains_key(m) {
                            return Some(Expr::Identifier(IdentifierExpr {
                                name: m.clone(),
                                span: id.span,
                            }));
                        }
                    }
                }
            }
        }
        let ty = match arg {
            Expr::Identifier(id) => self.lookup(id.name.as_ref())?.0.clone(),
            other => self.check_expr_type_probe(other),
        };
        let Type::Tensor(_, dims, _) = ty.behind_refs() else {
            return None;
        };
        let literal = dims.get(axis)?.literal()?;
        Some(Expr::Number(NumberExpr::new(
            literal.to_string(),
            Some(ElementType::USize),
            arg.span(),
        )))
    }

    /// The value of each extent name of a callee with `params` at a call with `args`. A name
    /// whose value the caller does not know gets `fresh(name)`, a name that the prover knows
    /// nothing about. The value of a name is the extent of the first parameter that names it,
    /// the one that the callee's body reads.
    pub(crate) fn callee_extent_values(
        &self,
        params: &[(Symbol, Type)],
        args: &[Expr],
        fresh: &mut dyn FnMut(&Symbol) -> Expr,
    ) -> CalleeExtents {
        let mut values: HashMap<Symbol, Option<Expr>> = HashMap::new();
        let mut mismatches = Vec::new();
        for (name, p, axis) in crate::syntax::types::extent_names(params) {
            let value = args.get(p).and_then(|a| self.extent_of_argument(a, axis));
            match values.get(&name) {
                None => {
                    values.insert(name, value);
                }
                Some(first) => {
                    if let (Some(Expr::Number(a)), Some(Expr::Number(b))) = (first, &value) {
                        if a.value != b.value {
                            mismatches.push((name, a.value.to_string(), b.value.to_string()));
                        }
                    }
                }
            }
        }
        let values = values
            .into_iter()
            .map(|(name, v)| {
                let v = v.unwrap_or_else(|| fresh(&name));
                (name, v)
            })
            .collect();
        CalleeExtents { values, mismatches }
    }

    /// E3068: a call that gives one extent name of `shown` two extents that differ.
    pub(crate) fn report_extent_mismatches(
        &mut self,
        shown: &str,
        mismatches: &[(Symbol, String, String)],
        span: &Span,
    ) {
        if self.speculating {
            return;
        }
        for (name, a, b) in mismatches {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3068,
                format!(
                    "this call to '{shown}' gives `?{name}` the extents {a} and {b}, but each \
                     `?{name}` of '{shown}' is the same extent"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }
}
