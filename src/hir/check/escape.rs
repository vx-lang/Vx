//===- escape.rs - Vx Compiler ---------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A value that points into this function's stack frame must not outlive the function
// (E4005). That is a reference to a local, a value holding one, or a closure that uses a
// local -- a closure becomes a struct of what it uses, and turning it into a `ClosureK`
// keeps a pointer to a copy in the frame.
//
// `ref_provenance_of` (check/access.rs) says where a value points. This file asks at the
// two places other than `return` where a value can leave the function: a store through a
// reference, and a call given a reference the callee could store it through. Vx has no
// written lifetimes, so the call rule is conservative: it may refuse a correct program, and
// does not accept a wrong one. A value that holds a `&mut` and a reference gets the same
// rule, since any method or its `drop` can store one through the other. A store through a
// local bound to `&mut x` is a store into `x`.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use crate::hir::env::RefProvenance;

impl<'a> TypeChecker<'a> {
    /// Run after a statement has been checked, when its values have types and its closure
    /// literals have become their structs. `return` is handled in `check_return_stmt`.
    pub(crate) fn check_frame_escape(&mut self, stmt: &Statement) {
        if self.speculating {
            return;
        }
        match stmt {
            Statement::LetDecl(decl) => self.check_escape_in_calls(&decl.expr),
            Statement::Assign(AssignStmt { lhs, rhs, .. }) => {
                self.check_escape_in_calls(rhs);
                // The error points at the place written to.
                let span = lhs.span();
                let retargeted = self.retarget_on_assign(lhs, rhs);
                let held = self.check_store_escape(lhs, rhs, &span);
                if let Expr::Identifier(id) = lhs {
                    let depth = self.scope_depth_of(&id.name);
                    self.set_store_target_var(&id.name, rhs, depth);
                }
                if let Some(root) = held.or(retargeted) {
                    self.check_writes_against_holds(Some(&root), &span);
                }
            }
            Statement::CompoundAssign(CompoundAssignStmt { rhs, .. }) => {
                self.check_escape_in_calls(rhs)
            }
            Statement::Return(ReturnStmt {
                expr: Some(expr), ..
            }) => self.check_escape_in_calls(expr),
            Statement::ExprStmt(ExprStmtStmt { expr, .. }) => self.check_escape_in_calls(expr),
            Statement::ForLoop(floop) => self.check_escape_in_calls(&floop.iterable),
            Statement::Assert(a) => self.check_escape_in_calls(&a.expr),
            _ => {}
        }
    }

    /// The escape checks for `lhs = rhs`. Returns the local that now holds a value pointing
    /// into a block, if there is one.
    fn check_store_escape(
        &mut self,
        lhs: &Expr,
        rhs: &Expr,
        span: &crate::syntax::Span,
    ) -> Option<crate::symbol::Symbol> {
        // The place's type is the value's, and is known more often: a variant built
        // inside a generic instance can still read as `Unknown`.
        let stored_ty = match self.check_expr_type_probe(lhs) {
            Type::Unknown => self.check_expr_type_probe(rhs),
            known => known,
        };
        let provenance = self.ref_provenance_of(rhs);
        let local_depth = match provenance {
            Some(RefProvenance::Local(depth)) if self.type_can_hold_reference(&stored_ty) => {
                Some(depth)
            }
            _ => None,
        };
        let Some(depth) = local_depth else {
            // `r = p` makes `r` point wherever `p` does now; a stale "local" would
            // refuse a later `return r` that is fine.
            if let Expr::Identifier(id) = lhs {
                match provenance {
                    Some(p) => {
                        self.borrow.ref_provenance.insert(id.name.clone(), p);
                    }
                    None => {
                        self.borrow.ref_provenance.remove(&id.name);
                    }
                }
            }
            return None;
        };
        // `r = &x` gives the variable `r` a new value, even when `r` is itself a
        // reference; only a path through a field, an index or a `*` writes into
        // whatever the reference points at.
        let is_path = !matches!(lhs, Expr::Identifier(_));
        // `*r = v`, where `r` was bound to `&mut x`, gives `x` a new value.
        if let Some(var) = self.store_target_var_of(lhs) {
            self.hold_in_local(var.clone(), depth, false, span);
            return Some(var);
        }
        match Self::place_root_through(lhs) {
            Some((root, through_deref)) => {
                if through_deref || (is_path && self.is_reference_variable(&root)) {
                    self.report_frame_escape(
                        "this stores a value that points into this function's stack \
                         frame through a reference, where it can outlive the function",
                        span,
                    );
                } else {
                    // The local now holds it; returning that local is caught later.
                    // A field store adds to what the local already points at.
                    self.hold_in_local(root.clone(), depth, is_path, span);
                    return Some(root);
                }
            }
            None => self.report_frame_escape(
                "this stores a value that points into this function's stack frame \
                 where it can outlive the function",
                span,
            ),
        }
        None
    }

    pub(crate) fn report_frame_escape(&mut self, what: &str, span: &crate::syntax::Span) {
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4005,
            format!(
                "{what}. What it points at is gone once the function returns. Pass it in from \
                 the caller instead; for a closure, write it so it uses no variable of this \
                 function"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// Record that the local `root` now holds a value pointing into the block at `depth`.
    /// When `root` was declared in an outer block, the value would outlive the block it
    /// points into, and is refused. `adds` keeps what `root` pointed at before: a field
    /// store or a call adds a value, a plain `root = ..` replaces it.
    fn hold_in_local(
        &mut self,
        root: crate::symbol::Symbol,
        depth: usize,
        adds: bool,
        span: &crate::syntax::Span,
    ) {
        if depth > self.scope_depth_of(&root) {
            self.report_block_escape(&root, depth, span);
            return;
        }
        let mut held = RefProvenance::Local(depth);
        if adds {
            if let Some(old) = self.borrow.ref_provenance.get(&root) {
                held = RefProvenance::join(*old, held);
            }
        }
        self.borrow.ref_provenance.insert(root, held);
    }

    /// `root` is given a value pointing into the block at `depth`, which ends before `root` does.
    pub(crate) fn report_block_escape(
        &mut self,
        root: &str,
        depth: usize,
        span: &crate::syntax::Span,
    ) {
        let message = if depth == RefProvenance::STATEMENT_END {
            format!(
                "`{root}` is given a reference into a value with no name, which is dropped at \
                 the end of this statement. What it points at is gone once the statement ends. \
                 Put the value in a variable first, with `let`, so it lives as long as `{root}`"
            )
        } else {
            format!(
                "`{root}` is given a value that points at a variable of an inner block, and \
                 `{root}` outlives that block. What it points at is gone once the block ends. \
                 Declare that variable outside the block, where `{root}` is"
            )
        };
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4005,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// A call can store an argument through any mutable reference it is given whose target
    /// can hold a reference. So when an argument points into this frame, every such
    /// reference among the other arguments must point into this frame too: one into memory
    /// the caller owns is refused, and a local it points at is marked as holding the value.
    ///
    /// A method's receiver counts, since the method may take `&mut self`. An immutable
    /// reference does not: nothing can be stored through it. Blocks inside the expression
    /// are left alone; their statements are checked by themselves.
    fn check_escape_in_calls(&mut self, expr: &Expr) {
        let (args, receiver, span): (Vec<&Expr>, Option<&Expr>, _) = match expr {
            Expr::FunctionCall(c) => (c.args.iter().collect(), None, c.span),
            Expr::MethodCall(c) => (c.args.iter().collect(), Some(&*c.base), c.span),
            Expr::IndirectCall(c) => (c.args.iter().collect(), None, c.span),
            _ => {
                for sub in Self::value_parts(expr) {
                    self.check_escape_in_calls(sub);
                }
                if matches!(expr, Expr::StructInit(_) | Expr::EnumVariant(_)) {
                    // A literal may carry no position of its own; its parts do.
                    let span = std::iter::once(expr.span())
                        .chain(Self::value_parts(expr).iter().map(|e| e.span()))
                        .find(|s| *s != crate::syntax::Span::default())
                        .unwrap_or_default();
                    let held = self.ref_provenance_of(expr);
                    let target = self.store_target_of(expr);
                    self.report_write_conflict(None, held, target, &span);
                }
                return;
            }
        };

        let all: Vec<&Expr> = receiver.into_iter().chain(args.iter().copied()).collect();
        for sub in &all {
            self.check_escape_in_calls(sub);
        }
        // A method call may carry no position of its own; its receiver or first argument
        // does.
        let span = if span == crate::syntax::Span::default() {
            all.iter()
                .map(|e| match e {
                    Expr::Borrow(b) => b.expr.span(),
                    e => e.span(),
                })
                .find(|s| *s != crate::syntax::Span::default())
                .unwrap_or(span)
        } else {
            span
        };

        let mut holders: Vec<usize> = Vec::new();
        let mut depth = 0;
        for (i, arg) in all.iter().enumerate() {
            if let Some(RefProvenance::Local(d)) = self.ref_provenance_of(arg) {
                if self.type_can_hold_reference(&self.check_expr_type_probe(arg)) {
                    holders.push(i);
                    depth = depth.max(d);
                }
            }
        }
        if holders.is_empty() {
            return;
        }

        for (i, arg) in all.iter().enumerate() {
            // Only a holder held by some *other* argument can be stored through this one.
            if !holders.iter().any(|&h| h != i) {
                continue;
            }
            let is_receiver = receiver.is_some() && i == 0;
            let Some(target) = self.mutable_reference_target(arg, is_receiver) else {
                continue;
            };
            if !self.type_can_hold_reference(&target) {
                continue;
            }
            // No root means a temporary: nothing outlives the statement through it.
            if let Some((root, through_deref)) = Self::place_root_through(arg) {
                // `&mut r` borrows the variable `r`, as in `store_target_of`.
                let borrows_the_variable =
                    matches!(arg, Expr::Borrow(b) if matches!(&*b.expr, Expr::Identifier(_)));
                if through_deref || (!borrows_the_variable && self.is_reference_variable(&root)) {
                    self.report_frame_escape(
                        &format!(
                            "this call is given a value that points into this function's \
                             stack frame, and a reference through `{root}` it could store \
                             that value into, where it can outlive the function"
                        ),
                        &span,
                    );
                } else {
                    self.hold_in_local(root.clone(), depth, true, &span);
                    let others = all
                        .iter()
                        .enumerate()
                        .filter(|&(j, _)| j != i)
                        .map(|(_, e)| self.store_target_of(e));
                    if let Some(added) = Self::outermost(others.collect()) {
                        self.add_store_target(&root, added, true);
                    }
                    self.check_writes_against_holds(Some(&root), &span);
                }
            }
        }
    }

    /// Where a write through a `&mut` that `expr`'s value holds would land, when its value
    /// holds one to something that can hold a reference. See `BorrowCx::store_targets`.
    pub(crate) fn store_target_of(&self, expr: &Expr) -> Option<RefProvenance> {
        match expr {
            Expr::Borrow(b) if b.is_mut => {
                if !self.type_can_hold_reference(&self.check_expr_type_probe(&b.expr)) {
                    return None;
                }
                // No root means a temporary: nothing outlives the statement through it.
                let (root, through_deref) = Self::place_root_through(&b.expr)?;
                // `&mut r` borrows the variable `r`. `&mut *r` and `&mut r.field` write
                // through the reference `r` holds.
                let through_reference = through_deref
                    || (!matches!(&*b.expr, Expr::Identifier(_))
                        && self.is_reference_variable(&root));
                if !through_reference {
                    Some(RefProvenance::Local(self.scope_depth_of(&root)))
                } else if self
                    .borrow
                    .current_params
                    .contains_key(root.as_ref() as &str)
                {
                    Some(RefProvenance::External)
                } else {
                    // `&mut *r` writes wherever `r` does. Untracked, it may be anywhere.
                    Some(
                        self.borrow
                            .store_targets
                            .get(&root)
                            .copied()
                            .unwrap_or(RefProvenance::External),
                    )
                }
            }
            Expr::Identifier(id) => match self.borrow.current_params.get(id.name.as_ref()) {
                Some(pty) => self
                    .type_can_write_reference(pty)
                    .then_some(RefProvenance::External),
                None => self.borrow.store_targets.get(id.name.as_ref()).copied(),
            },
            Expr::StructInit(s) => Self::outermost(
                s.fields
                    .iter()
                    .map(|(_, e)| self.store_target_of(e))
                    .collect(),
            ),
            Expr::EnumVariant(v) => Self::outermost(
                v.payload
                    .iter()
                    .flatten()
                    .map(|e| self.store_target_of(e))
                    .collect(),
            ),
            Expr::MemberAccess(m) => self.store_target_of(&m.base),
            Expr::AsCast(c) => self.store_target_of(&c.expr),
            // A call's result may keep any `&mut` it was given.
            Expr::FunctionCall(c) => {
                Self::outermost(c.args.iter().map(|e| self.store_target_of(e)).collect())
            }
            Expr::MethodCall(c) => Self::outermost(
                std::iter::once(&*c.base)
                    .chain(c.args.iter())
                    .map(|e| self.store_target_of(e))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// Of several store targets, the one that outlives the others: memory the caller owns,
    /// or else the shallowest block.
    fn outermost(targets: Vec<Option<RefProvenance>>) -> Option<RefProvenance> {
        targets.into_iter().flatten().reduce(|a, b| match (a, b) {
            (RefProvenance::Local(x), RefProvenance::Local(y)) => RefProvenance::Local(x.min(y)),
            _ => RefProvenance::External,
        })
    }

    fn add_store_target(
        &mut self,
        root: &crate::symbol::Symbol,
        target: RefProvenance,
        adds: bool,
    ) {
        let old = if adds {
            self.borrow.store_targets.get(root).copied()
        } else {
            None
        };
        if let Some(t) = Self::outermost(vec![old, Some(target)]) {
            self.borrow.store_targets.insert(root.clone(), t);
        }
    }

    /// Record the variable that `name`, declared at `depth`, now points at, when `value` is
    /// `&mut` of a variable.
    pub(crate) fn set_store_target_var(
        &mut self,
        name: &crate::symbol::Symbol,
        value: &Expr,
        depth: usize,
    ) {
        match value {
            Expr::Borrow(b) if b.is_mut => match &*b.expr {
                Expr::Identifier(v) if v.name != *name => {
                    self.borrow
                        .store_target_vars
                        .insert(name.clone(), (v.name.clone(), depth));
                }
                _ => {
                    self.borrow.store_target_vars.remove(name);
                }
            },
            _ => {
                self.borrow.store_target_vars.remove(name);
            }
        }
    }

    /// The variable that `lhs` writes, when `lhs` is `*r` and `r` is a local bound to
    /// `&mut` of that variable.
    pub(crate) fn store_target_var_of(&self, lhs: &Expr) -> Option<crate::symbol::Symbol> {
        let Expr::Dereference(d) = lhs else {
            return None;
        };
        let Expr::Identifier(id) = &*d.expr else {
            return None;
        };
        // An entry made in a block that has ended is about another variable of that name.
        let (var, depth) = self.borrow.store_target_vars.get(&id.name)?;
        (*depth == self.scope_depth_of(&id.name)).then(|| var.clone())
    }

    /// For `lhs = rhs`: record the `&mut` that `rhs` puts into a local. Returns that local.
    fn retarget_on_assign(&mut self, lhs: &Expr, rhs: &Expr) -> Option<crate::symbol::Symbol> {
        if let Some(var) = self.store_target_var_of(lhs) {
            match self.store_target_of(rhs) {
                Some(target) => self.add_store_target(&var, target, false),
                None => {
                    self.borrow.store_targets.remove(&var);
                }
            }
            return Some(var);
        }
        let (root, through_deref) = Self::place_root_through(lhs)?;
        let is_path = !matches!(lhs, Expr::Identifier(_));
        if through_deref || (is_path && self.is_reference_variable(&root)) {
            return None;
        }
        match self.store_target_of(rhs) {
            Some(target) => {
                self.add_store_target(&root, target, is_path);
                Some(root)
            }
            None => {
                if !is_path {
                    self.borrow.store_targets.remove(&root);
                }
                None
            }
        }
    }

    /// A local that holds a `&mut` must not also hold a reference that the `&mut`'s target
    /// outlives: a write through the first could store the second there.
    fn check_writes_against_holds(
        &mut self,
        root: Option<&crate::symbol::Symbol>,
        span: &crate::syntax::Span,
    ) {
        let Some(root) = root else { return };
        let held = self.borrow.ref_provenance.get(root).copied();
        let target = self.borrow.store_targets.get(root).copied();
        self.report_write_conflict(Some(root), held, target, span);
    }

    fn report_write_conflict(
        &mut self,
        root: Option<&crate::symbol::Symbol>,
        held: Option<RefProvenance>,
        target: Option<RefProvenance>,
        span: &crate::syntax::Span,
    ) {
        let (Some(RefProvenance::Local(depth)), Some(target)) = (held, target) else {
            return;
        };
        let what = match root {
            Some(r) => format!("`{r}`"),
            None => "this value".to_string(),
        };
        let message = match target {
            RefProvenance::External => format!(
                "{what} holds a `&mut` to memory the caller owns and a reference to a variable \
                 of this function. A write through the `&mut` can store that reference where it \
                 outlives the function"
            ),
            RefProvenance::Local(t) if depth > t => format!(
                "{what} holds a `&mut` to a variable of an outer block and a reference to a \
                 variable of an inner block. A write through the `&mut` can store that \
                 reference where it outlives its block. Declare both variables in the same block"
            ),
            RefProvenance::Local(_) => return,
        };
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4005,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// Can a value of this type write a reference somewhere: does it hold a `&mut` to
    /// something that can hold a reference? A raw pointer is left out, as in
    /// `type_can_hold_reference`, and so is a type parameter.
    fn type_can_write_reference(&self, ty: &Type) -> bool {
        self.type_can_write_reference_in(ty, &mut std::collections::HashSet::new())
    }

    fn type_can_write_reference_in(
        &self,
        ty: &Type,
        seen: &mut std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            } => self.type_can_hold_reference(inner),
            // Nothing can be written through a shared reference.
            Type::Borrow { .. } => false,
            Type::Pinned(inner, _) | Type::Verified(inner) => {
                self.type_can_write_reference_in(inner, seen)
            }
            Type::Struct(name, _) | Type::Enum(name, _) => {
                if !seen.insert(name.to_string()) {
                    return false;
                }
                self.declared_part_types(name, &[])
                    .iter()
                    .any(|t| self.type_can_write_reference_in(t, seen))
            }
            Type::GenericInstance(base, args) => match &**base {
                Type::Struct(name, _) | Type::Enum(name, _) => {
                    if !seen.insert(format!("{name}<{args:?}>")) {
                        return false;
                    }
                    self.declared_part_types(name, args)
                        .iter()
                        .any(|t| self.type_can_write_reference_in(t, seen))
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// The type a mutable reference argument points at, when it is one: `&mut x` gives the
    /// type of `x`; a variable of type `&mut T` or `*mut T` gives `T`; a receiver is taken
    /// to be one, since the method may borrow it mutably.
    fn mutable_reference_target(&self, arg: &Expr, is_receiver: bool) -> Option<Type> {
        if let Expr::Borrow(b) = arg {
            return if b.is_mut {
                Some(self.check_expr_type_probe(&b.expr))
            } else {
                None
            };
        }
        let ty = self.check_expr_type_probe(arg);
        match &ty {
            Type::Borrow {
                inner,
                is_mut: true,
                ..
            }
            | Type::Pointer(inner, _, true) => Some((**inner).clone()),
            Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..) => None,
            _ if is_receiver => Some(ty),
            _ => None,
        }
    }

    /// The type of an expression that has already been checked, read without recording
    /// anything: no moves, no borrows, no diagnostics.
    pub(crate) fn check_expr_type_probe(&self, expr: &Expr) -> Type {
        match expr {
            Expr::Identifier(id) => self
                .lookup(id.name.as_ref())
                .map(|(t, _)| t.clone())
                .unwrap_or(Type::Unknown),
            Expr::Borrow(b) => Type::Borrow {
                inner: Box::new(self.check_expr_type_probe(&b.expr)),
                mem_space: None,
                is_mut: b.is_mut,
                region_id: self.scopes.len(),
            },
            Expr::Dereference(d) => match self.check_expr_type_probe(&d.expr) {
                Type::Borrow { inner, .. } | Type::Pointer(inner, ..) | Type::Ref(inner, _) => {
                    *inner
                }
                _ => Type::Unknown,
            },
            Expr::MemberAccess(m) => {
                let base = self.check_expr_type_probe(&m.base);
                let base = match base {
                    Type::Borrow { inner, .. } | Type::Pointer(inner, ..) | Type::Ref(inner, _) => {
                        *inner
                    }
                    other => other,
                };
                let (name, args) = match &base {
                    Type::Struct(n, _) => (n.clone(), Vec::new()),
                    Type::GenericInstance(b, a) => match &**b {
                        Type::Struct(n, _) => (n.clone(), a.clone()),
                        _ => return Type::Unknown,
                    },
                    _ => return Type::Unknown,
                };
                self.struct_field_type(&name, &args, &m.member)
                    .unwrap_or(Type::Unknown)
            }
            Expr::StructInit(s) => {
                let base: crate::symbol::Symbol =
                    s.name.split('<').next().unwrap_or(&s.name).into();
                Type::Struct(base, None)
            }
            // A call to a function the checker has resolved: its declared return type.
            Expr::FunctionCall(fc) => {
                if let Some(f) = self.env.functions.get(&fc.name) {
                    return f.0.clone();
                }
                self.mono
                    .functions
                    .iter()
                    .find(|(f, _)| f.name == fc.name)
                    .map(|(f, _)| f.return_type.clone())
                    .unwrap_or(Type::Unknown)
            }
            // Anything else: unknown, which the escape gate treats as "could hold one".
            _ => Type::Unknown,
        }
    }

    /// The declared type of `field` in struct `name`, with the struct's parameters
    /// substituted from `args`.
    fn struct_field_type(
        &self,
        name: &crate::symbol::Symbol,
        args: &[Type],
        field: &crate::symbol::Symbol,
    ) -> Option<Type> {
        let base = name.split('<').next().unwrap_or(name);
        let (generics, fields): (Vec<String>, Vec<(crate::symbol::Symbol, Type)>) =
            if let Some(s) = self.env.structs.get(base) {
                (
                    s.generics.iter().map(|g| g.name().to_string()).collect(),
                    s.fields.clone(),
                )
            } else {
                let s = self
                    .mono
                    .generated_structs
                    .iter()
                    .find(|s| s.name.as_ref() == base)?;
                (
                    s.generics.iter().map(|g| g.name().to_string()).collect(),
                    s.fields.clone(),
                )
            };
        let mapping: std::collections::HashMap<crate::symbol::Symbol, Type> = generics
            .iter()
            .zip(args.iter())
            .map(|(g, a)| (g.as_str().into(), a.clone()))
            .collect();
        fields
            .iter()
            .find(|(n, _)| n == field)
            .map(|(_, t)| t.substitute(&mapping))
    }

    /// The sub-expressions whose values make up `expr`'s value. Blocks are not entered.
    fn value_parts(expr: &Expr) -> Vec<&Expr> {
        match expr {
            Expr::StructInit(s) => s.fields.iter().map(|(_, e)| e).collect(),
            Expr::EnumVariant(v) => v.payload.iter().flatten().collect(),
            Expr::MemberAccess(m) => vec![&*m.base],
            Expr::IndexAccess(i) => vec![&*i.base],
            Expr::Borrow(b) => vec![&*b.expr],
            Expr::Dereference(d) => vec![&*d.expr],
            Expr::AsCast(c) => vec![&*c.expr],
            _ => Vec::new(),
        }
    }

    /// The variable a place expression is rooted in, and whether the path goes through a
    /// dereference on the way. Unlike `place_root`, this follows field reads too.
    fn place_root_through(expr: &Expr) -> Option<(crate::symbol::Symbol, bool)> {
        match expr {
            Expr::Identifier(id) => Some((id.name.clone(), false)),
            Expr::MemberAccess(m) => Self::place_root_through(&m.base),
            Expr::IndexAccess(i) => Self::place_root_through(&i.base),
            Expr::Borrow(b) => Self::place_root_through(&b.expr),
            Expr::Dereference(d) => Self::place_root_through(&d.expr).map(|(r, _)| (r, true)),
            _ => None,
        }
    }

    /// Is `name` a reference or pointer, so that storing through it reaches memory this
    /// function does not own?
    fn is_reference_variable(&self, name: &crate::symbol::Symbol) -> bool {
        matches!(
            self.lookup(name.as_ref()).map(|(t, _)| t),
            Some(Type::Ref(..)) | Some(Type::Borrow { .. }) | Some(Type::Pointer(..))
        )
    }
}
