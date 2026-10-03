//===- views.rs - Vx Compiler ----------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A row `q[i]` or a tensor field `h.t` is not a copy: it is a view of its owner's memory,
// and writing through one changes the other. So a view is a borrow of its owner, as `&q`
// is. Binding one records the borrow, and the existing rules then apply: the owner cannot
// be moved (E4007) or written (E4009) while the view is still used, and a view declared
// `mut` is a mutable borrow. A view also cannot outlive its owner (E4005), and cannot be
// stored in a variable that already exists or in a struct field (E4011): those hold a
// tensor of their own, and the view would replace it silently.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::env::RefProvenance;

/// What a view variable borrows: its owner, the path inside the owner, and where the
/// owner lives (`None` when the checker cannot tell).
#[derive(Clone)]
pub(crate) struct View {
    pub(crate) owner: String,
    pub(crate) path: Vec<String>,
    pub(crate) provenance: Option<RefProvenance>,
}

impl<'a> TypeChecker<'a> {
    /// The view `expr` makes, when it makes one: a field or index chain, or a view variable,
    /// whose value is a tensor. A view of a view borrows the first owner.
    pub(crate) fn view_of(&self, expr: &Expr, ty: &Type) -> Option<View> {
        if !matches!(ty, Type::Tensor(..)) {
            return None;
        }
        if let Expr::Identifier(id) = expr {
            return self.borrow.views.get(id.name.as_ref()).cloned();
        }
        if !matches!(expr, Expr::IndexAccess(_) | Expr::MemberAccess(_)) {
            return None;
        }
        let (root, path) = Self::extract_base_and_path(expr)?;
        if let Some(outer) = self.borrow.views.get(root.as_str()) {
            let mut full = outer.path.clone();
            full.extend(path);
            return Some(View {
                path: full,
                ..outer.clone()
            });
        }
        let provenance = match self.borrow.current_params.get(root.as_str()) {
            Some(pty) if Self::is_ref_type(pty) => Some(RefProvenance::External),
            Some(_) => Some(RefProvenance::Local(self.scope_depth_of(&root))),
            None => match self.lookup(&root).map(|(t, _)| t) {
                Some(t) if Self::is_ref_type(t) => {
                    self.borrow.ref_provenance.get(root.as_str()).copied()
                }
                _ => Some(RefProvenance::Local(self.scope_depth_of(&root))),
            },
        };
        Some(View {
            owner: root,
            path,
            provenance,
        })
    }

    /// `let name = expr`: when `expr` is a view, record that `name` borrows its owner,
    /// mutably when `name` is declared `mut`.
    pub(crate) fn bind_view(
        &mut self,
        name: &str,
        is_mut: bool,
        expr: &Expr,
        ty: &Type,
        span: &crate::syntax::Span,
    ) {
        if self.speculating {
            return;
        }
        let view = self.view_of(expr, ty);
        self.borrow.views.remove(name);
        let Some(view) = view else {
            return;
        };
        let through_reference = matches!(
            self.lookup(&view.owner).map(|(t, _)| t),
            Some(t) if Self::is_ref_type(t)
        );
        if is_mut && !through_reference && !self.is_mutable(&view.owner) {
            self.report_read_only(&view.owner, "a view of it cannot be declared `mut`", span);
        }
        self.check_borrow_conflicts(&view.owner, &view.path, is_mut, span);
        self.borrow.record(
            &view.owner,
            BorrowRecord {
                is_mut,
                scope_depth: self.scopes.len(),
                borrower_name: Some(name.to_string()),
                path: view.path.clone(),
            },
        );
        if is_mut {
            self.consteval_forget(&view.owner);
        }
        self.borrow.views.insert(name.into(), view);
    }

    /// `return expr` of a view of this function's own tensor: the tensor is gone once the
    /// function returns.
    pub(crate) fn check_view_return(&mut self, expr: &Expr, ty: &Type, span: &crate::syntax::Span) {
        if self.speculating {
            return;
        }
        if let Some(View {
            owner,
            provenance: Some(RefProvenance::Local(_)),
            ..
        }) = self.view_of(expr, ty)
        {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E4005,
                format!(
                    "this returns a view of `{owner}`, a tensor this function owns, which is \
                     gone once the function returns. Take `{owner}` by reference (`&`) so the \
                     view points into the caller's tensor"
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
            );
        }
    }

    /// `target = expr` where `target` is a variable or a field path (`t`, `h.t`) and `expr`
    /// a view: E4011. Writing into an element or a row, `p[i] = q[j]`, copies, and is fine.
    pub(crate) fn check_view_assign(&mut self, lhs: &Expr, rhs: &Expr, ty: &Type) {
        if !matches!(lhs, Expr::Identifier(_) | Expr::MemberAccess(_)) {
            return;
        }
        let mut place = lhs;
        while let Expr::MemberAccess(m) = place {
            place = &m.base;
        }
        if !matches!(place, Expr::Identifier(_)) {
            return;
        }
        if let Some((root, path)) = Self::extract_base_and_path(lhs) {
            let name = std::iter::once(root)
                .chain(path)
                .collect::<Vec<_>>()
                .join(".");
            self.check_view_stored(&format!("`{name}`"), rhs, ty);
        }
    }

    /// E4011 when `value` is a view and is stored in `what`, which holds a tensor of its own:
    /// a variable that already exists, or a struct field.
    pub(crate) fn check_view_stored(&mut self, what: &str, value: &Expr, ty: &Type) {
        if self.speculating {
            return;
        }
        let Some(view) = self.view_of(value, ty) else {
            return;
        };
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4011,
            format!(
                "this stores a view of `{owner}` in {what}, which holds a tensor of its own. A \
                 view shares `{owner}`'s memory and is not a copy; bind it to a new variable \
                 with `let` instead",
                owner = view.owner
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(&value.span())),
        );
    }
}
