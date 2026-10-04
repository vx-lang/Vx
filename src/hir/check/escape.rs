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
// does not accept a wrong one.
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
                let lhs_span = lhs.span();
                let span = &lhs_span;
                // The place's type is the value's, and is known more often: a variant built
                // inside a generic instance can still read as `Unknown`.
                let stored_ty = match self.check_expr_type_probe(lhs) {
                    Type::Unknown => self.check_expr_type_probe(rhs),
                    known => known,
                };
                let provenance = self.ref_provenance_of(rhs);
                let local_depth = match provenance {
                    Some(RefProvenance::Local(depth))
                        if self.type_can_hold_reference(&stored_ty) =>
                    {
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
                    return;
                };
                // `r = &x` gives the variable `r` a new value, even when `r` is itself a
                // reference; only a path through a field, an index or a `*` writes into
                // whatever the reference points at.
                let is_path = !matches!(lhs, Expr::Identifier(_));
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
                            self.hold_in_local(root, depth, is_path, span);
                        }
                    }
                    None => self.report_frame_escape(
                        "this stores a value that points into this function's stack frame \
                         where it can outlive the function",
                        span,
                    ),
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
            self.report_block_escape(&root, span);
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

    pub(crate) fn report_block_escape(&mut self, root: &str, span: &crate::syntax::Span) {
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4005,
            format!(
                "`{root}` is given a value that points at a variable of an inner block, and \
                 `{root}` outlives that block. What it points at is gone once the block ends. \
                 Declare that variable outside the block, where `{root}` is"
            ),
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
                if through_deref || self.is_reference_variable(&root) {
                    self.report_frame_escape(
                        &format!(
                            "this call is given a value that points into this function's \
                             stack frame, and a reference through `{root}` it could store \
                             that value into, where it can outlive the function"
                        ),
                        &span,
                    );
                } else {
                    self.hold_in_local(root, depth, true, &span);
                }
            }
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
