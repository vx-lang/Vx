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

/// What a view borrows: each tensor it may point into, with the path inside it, and where
/// those tensors live (`None` when the checker cannot tell). A view chosen by an `if` or a
/// `match` may point into any of several tensors.
#[derive(Clone)]
pub(crate) struct View {
    pub(crate) owners: Vec<(String, Vec<String>)>,
    /// The tensors behind a reference among `owners`: `q` for `rq[1]` with `rq = &q`. The
    /// view keeps them borrowed too, but conflicts are checked on the reference, which
    /// already holds its own borrow of them.
    pub(crate) behind: Vec<(String, Vec<String>)>,
    pub(crate) provenance: Option<RefProvenance>,
}

impl View {
    /// The tensor to name in a message.
    pub(crate) fn owner(&self) -> &str {
        &self.owners[0].0
    }

    /// The view whichever of `views` was chosen: all their owners, and the provenance that
    /// lives shortest.
    fn join(views: Vec<View>) -> Option<View> {
        let mut views = views.into_iter();
        let mut joined = views.next()?;
        for v in views {
            for o in v.owners {
                if !joined.owners.contains(&o) {
                    joined.owners.push(o);
                }
            }
            for o in v.behind {
                if !joined.behind.contains(&o) {
                    joined.behind.push(o);
                }
            }
            joined.provenance = match (joined.provenance, v.provenance) {
                (Some(a), Some(b)) => Some(RefProvenance::join(a, b)),
                (a, b) => a.or(b),
            };
        }
        Some(joined)
    }
}

impl<'a> TypeChecker<'a> {
    /// The view `expr` makes, when it makes one, and its value is a tensor: a field or index
    /// chain, `t.reshape(..)`, a view variable, or an `if` or `match` whose value is one of
    /// these. A view of a view, or of a tensor reached through a reference, borrows the
    /// tensor itself.
    pub(crate) fn view_of(&self, expr: &Expr, ty: &Type) -> Option<View> {
        if !matches!(ty, Type::Tensor(..)) {
            return None;
        }
        match expr {
            Expr::Identifier(id) => self.borrow.views.get(id.name.as_ref()).cloned(),
            Expr::IndexAccess(_) | Expr::MemberAccess(_) => {
                let (root, path) = Self::extract_base_and_path(expr)?;
                Some(self.view_into(root, path))
            }
            Expr::MethodCall(mc) if mc.method_name.as_ref() == "reshape" => match &*mc.base {
                Expr::Identifier(id) => Some(self.view_into(id.name.to_string(), Vec::new())),
                base => self.view_of(base, ty),
            },
            Expr::If(i) => {
                let mut views = self.tail_views(&i.then_block, ty);
                views.extend(self.tail_views(i.else_block.as_deref().unwrap_or(&[]), ty));
                View::join(views)
            }
            Expr::Match(m) => View::join(
                m.arms
                    .iter()
                    .flat_map(|a| self.tail_views(&a.body, ty))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The views a block's value can be: its last expression, looked into when it is itself
    /// an `if` or a `match`. A `return` leaves the function, so it is not the block's value.
    fn tail_views(&self, stmts: &[Statement], ty: &Type) -> Vec<View> {
        match stmts.last() {
            Some(Statement::ExprStmt(e)) if !e.has_semi => {
                self.view_of(&e.expr, ty).into_iter().collect()
            }
            _ => Vec::new(),
        }
    }

    /// A view of `path` inside the variable `root`. When `root` is itself a view, or a
    /// reference, the view borrows what `root` borrows.
    fn view_into(&self, root: String, path: Vec<String>) -> View {
        if let Some(outer) = self.borrow.views.get(root.as_str()) {
            let extend = |list: &[(String, Vec<String>)]| {
                list.iter()
                    .map(|(o, p)| (o.clone(), p.iter().chain(&path).cloned().collect()))
                    .collect()
            };
            return View {
                owners: extend(&outer.owners),
                behind: extend(&outer.behind),
                provenance: outer.provenance,
            };
        }
        let root_ty = self.lookup(&root).map(|(t, _)| t.clone());
        if root_ty.as_ref().is_some_and(Self::is_ref_type) {
            let provenance = match self.borrow.current_params.get(root.as_str()) {
                Some(_) => Some(RefProvenance::External),
                None => self.borrow.ref_provenance.get(root.as_str()).copied(),
            };
            // `rq[1]` with `rq = &q` points into `q`: the view keeps `q` borrowed after
            // `rq`'s last use.
            let behind = self
                .borrow
                .borrowed_by(&root)
                .into_iter()
                .map(|(o, p)| (o, p.into_iter().chain(path.iter().cloned()).collect()))
                .collect();
            return View {
                owners: vec![(root, path)],
                behind,
                provenance,
            };
        }
        let provenance = match root_ty {
            Some(_) => Some(RefProvenance::Local(self.scope_depth_of(&root))),
            // The owner was declared in a block that has ended: the block of an `if` arm
            // whose value is this view.
            None => Some(RefProvenance::Local(self.current_scope_depth() + 1)),
        };
        View {
            owners: vec![(root, path)],
            behind: Vec::new(),
            provenance,
        }
    }

    /// `let name = expr`: when `expr` is a view, record that `name` borrows each tensor it
    /// may point into, mutably when `name` is declared `mut`. A closure that uses a view
    /// borrows what the view borrows, for as long as the closure is used.
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
        self.borrow.views.remove(name);
        if let Expr::StructInit(s) = expr {
            if s.name.starts_with("Closure_") {
                for (_, field) in &s.fields {
                    let Expr::Identifier(id) = field else {
                        continue;
                    };
                    // A captured reference: the closure holds what it borrows.
                    self.borrow
                        .copy_borrows(id.name.as_ref(), name, self.scopes.len());
                    let Some(view) = self.borrow.views.get(id.name.as_ref()).cloned() else {
                        continue;
                    };
                    let view_is_mut = self.is_mutable(id.name.as_ref());
                    self.borrow_for_view(name, &view, view_is_mut, span);
                }
                return;
            }
        }
        let Some(view) = self.view_of(expr, ty) else {
            return;
        };
        if let Some(RefProvenance::Local(depth)) = view.provenance {
            if depth > self.current_scope_depth() {
                self.report_block_escape(name, span);
            }
        }
        for (owner, _) in &view.owners {
            let through_reference = matches!(
                self.lookup(owner).map(|(t, _)| t),
                Some(t) if Self::is_ref_type(t)
            );
            if is_mut
                && !through_reference
                && self.lookup(owner).is_some()
                && !self.is_mutable(owner)
            {
                self.report_read_only(owner, "a view of it cannot be declared `mut`", span);
            }
        }
        self.borrow_for_view(name, &view, is_mut, span);
        self.borrow.views.insert(name.into(), view);
    }

    /// Record that `borrower` borrows each tensor `view` may point into.
    fn borrow_for_view(
        &mut self,
        borrower: &str,
        view: &View,
        is_mut: bool,
        span: &crate::syntax::Span,
    ) {
        for (owner, path) in &view.owners {
            self.check_borrow_conflicts(owner, path, is_mut, span);
        }
        for (owner, path) in view.owners.iter().chain(&view.behind) {
            self.borrow.record(
                owner,
                BorrowRecord {
                    is_mut,
                    scope_depth: self.scopes.len(),
                    borrower_name: Some(borrower.to_string()),
                    path: path.clone(),
                },
            );
            if is_mut {
                self.consteval_forget(owner);
            }
        }
    }

    /// `return expr` of a view of this function's own tensor: the tensor is gone once the
    /// function returns.
    pub(crate) fn check_view_return(&mut self, expr: &Expr, ty: &Type, span: &crate::syntax::Span) {
        if self.speculating {
            return;
        }
        if let Some(
            view @ View {
                provenance: Some(RefProvenance::Local(_)),
                ..
            },
        ) = self.view_of(expr, ty)
        {
            let owner = view.owner();
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
                owner = view.owner()
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(&value.span())),
        );
    }

    /// `f(q[1])` where `f` takes a tensor by value: E4011. The function would own memory
    /// that is `q`'s. A row passed to a `&Tensor` parameter is fine.
    pub(crate) fn check_views_passed_by_value(
        &mut self,
        callee: &str,
        params: &[Type],
        args: &[Expr],
    ) {
        if self.speculating {
            return;
        }
        for (param, arg) in params.iter().zip(args) {
            let Some(view) = self.view_of(arg, param) else {
                continue;
            };
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E4011,
                format!(
                    "this passes a view of `{owner}` to `{callee}`, which takes a tensor of its \
                     own. A view shares `{owner}`'s memory and is not a copy; pass a copy with \
                     `.clone()`, or make `{callee}` take the tensor by reference (`&`)",
                    owner = view.owner()
                ),
                Some(crate::diagnostic::SourceSpan::from_ast_span(&arg.span())),
            );
        }
    }
}
