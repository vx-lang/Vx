//===- closure_escape.rs - Vx Compiler -------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A closure that uses a variable of the function that made it must not outlive that
// function (E4006; Rust refuses the same programs with E0373).
//
// A closure literal becomes a struct holding the variables it uses. Where a `ClosureK`
// value is wanted, code generation copies that struct into the current stack frame and keeps
// a pointer to the copy. Once the function returns, the pointer is left pointing at memory
// that is gone. A closure that uses no variable is safe: its pointer is never followed.
//
// Vx has no lifetimes, so the check is conservative. Within one function it follows a value
// that may hold such a closure through variables, struct literals, field reads and calls,
// and refuses to let it be returned, stored through a reference, or passed to a call along
// with a reference it could be stored through. It may refuse a correct program; it does not
// accept a wrong one.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::HashSet;

impl<'a> TypeChecker<'a> {
    /// Run after a statement has been checked, when its closure literals have become their
    /// structs and its variables are in scope.
    pub(crate) fn check_closure_escape(&mut self, stmt: &Statement, return_type: &Type) {
        if self.speculating {
            return;
        }
        match stmt {
            Statement::LetDecl(decl) => {
                self.check_closure_escape_in_calls(&decl.expr);
                let ty = self
                    .lookup(decl.name.as_ref())
                    .map(|(t, _)| t.clone())
                    .unwrap_or(Type::Unknown);
                if self.type_can_hold_closure(&ty) && self.may_hold_closure(&decl.expr) {
                    self.closure_borrowers.insert(decl.name.clone());
                } else {
                    // A new binding of the name holds something else now.
                    self.closure_borrowers.remove(&decl.name);
                }
            }
            Statement::Assign(AssignStmt { lhs, rhs, span }) => {
                // An assignment may carry no position of its own; the place written to does.
                let lhs_span = lhs.span();
                let span = if *span == crate::syntax::Span::default() {
                    &lhs_span
                } else {
                    span
                };
                self.check_closure_escape_in_calls(rhs);
                if !self.may_hold_closure(rhs) {
                    if let Expr::Identifier(id) = lhs {
                        self.closure_borrowers.remove(&id.name);
                    }
                    return;
                }
                match Self::closure_place_root(lhs) {
                    Some((root, through_deref)) => {
                        if through_deref || self.is_reference_variable(&root) {
                            self.report_closure_escape(
                                "this stores a closure that uses a variable of this function \
                                 through a reference, where it can outlive the function",
                                span,
                            );
                        } else {
                            self.closure_borrowers.insert(root);
                        }
                    }
                    None => self.report_closure_escape(
                        "this stores a closure that uses a variable of this function where \
                         it can outlive the function",
                        span,
                    ),
                }
            }
            Statement::CompoundAssign(CompoundAssignStmt { rhs, .. }) => {
                self.check_closure_escape_in_calls(rhs);
            }
            Statement::Return(ReturnStmt {
                expr: Some(expr),
                span,
            }) => {
                self.check_closure_escape_in_calls(expr);
                if self.type_can_hold_closure(return_type) && self.may_hold_closure(expr) {
                    self.report_closure_escape(
                        "this returns a closure that uses a variable of this function, and \
                         that variable is gone once the function returns",
                        span,
                    );
                }
            }
            Statement::ExprStmt(ExprStmtStmt { expr, .. }) => {
                self.check_closure_escape_in_calls(expr);
            }
            Statement::ForLoop(floop) => {
                self.check_closure_escape_in_calls(&floop.iterable);
            }
            _ => {}
        }
    }

    fn report_closure_escape(&mut self, what: &str, span: &crate::syntax::Span) {
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E4006,
            format!(
                "{what}. Pass the closure in from the caller instead, or write it so it uses \
                 no variable of this function"
            ),
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// A call can store an argument through any reference it is given. So when an argument
    /// may hold one of these closures, every other reference argument must point into this
    /// function: one that points out of it is refused, and a local it points at is marked.
    ///
    /// A method's receiver counts as a reference, since the method may take `&mut self`.
    /// Blocks inside the expression are left alone; their statements are checked by
    /// themselves.
    fn check_closure_escape_in_calls(&mut self, expr: &Expr) {
        let (args, receiver, span): (Vec<&Expr>, Option<&Expr>, _) = match expr {
            Expr::FunctionCall(c) => (c.args.iter().collect(), None, c.span),
            Expr::MethodCall(c) => (c.args.iter().collect(), Some(&*c.base), c.span),
            Expr::IndirectCall(c) => (c.args.iter().collect(), None, c.span),
            _ => {
                for sub in Self::value_parts(expr) {
                    self.check_closure_escape_in_calls(sub);
                }
                return;
            }
        };

        let all: Vec<&Expr> = receiver.into_iter().chain(args.iter().copied()).collect();
        for sub in &all {
            self.check_closure_escape_in_calls(sub);
        }

        let holders: Vec<usize> = (0..all.len())
            .filter(|&i| self.may_hold_closure(all[i]))
            .collect();
        if holders.is_empty() {
            return;
        }
        for (i, arg) in all.iter().enumerate() {
            let is_receiver = receiver.is_some() && i == 0;
            let reference = is_receiver
                || matches!(arg, Expr::Borrow(b) if b.is_mut)
                || Self::closure_place_root(arg)
                    .is_some_and(|(r, _)| self.is_reference_variable(&r));
            // A reference is only a way out for a closure held by some *other* argument.
            if !reference || !holders.iter().any(|&h| h != i) {
                continue;
            }
            // No root means a temporary: nothing outlives the statement through it.
            if let Some((root, through_deref)) = Self::closure_place_root(arg) {
                if through_deref || self.is_reference_variable(&root) {
                    self.report_closure_escape(
                        &format!(
                            "this call is given a closure that uses a variable of this \
                             function, and a reference through `{root}` it could store the \
                             closure into, where it can outlive the function"
                        ),
                        &span,
                    );
                } else {
                    self.closure_borrowers.insert(root);
                }
            }
        }
    }

    /// May `expr` hold a closure that uses a variable of this function? Only the value is
    /// asked about, not whether its type could hold one; callers check that separately.
    fn may_hold_closure(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Identifier(id) => self.closure_borrowers.contains(&id.name),
            Expr::StructInit(s) if self.closure_uses_variables(&s.name) => true,
            Expr::StructInit(s) => s.fields.iter().any(|(_, e)| self.may_hold_closure(e)),
            Expr::FunctionCall(c) => c.args.iter().any(|a| self.may_hold_closure(a)),
            Expr::MethodCall(c) => {
                self.may_hold_closure(&c.base) || c.args.iter().any(|a| self.may_hold_closure(a))
            }
            Expr::IndirectCall(c) => {
                self.may_hold_closure(&c.callee) || c.args.iter().any(|a| self.may_hold_closure(a))
            }
            // Anything that contains a block could produce one from inside it. Say yes if one
            // appears anywhere in it, rather than work out which part is the value.
            Expr::If(_) | Expr::Match(_) | Expr::UnsafeBlock(_) => self.mentions_held_closure(expr),
            _ => Self::value_parts(expr)
                .iter()
                .any(|e| self.may_hold_closure(e)),
        }
    }

    /// Does a closure that uses variables, or a local holding one, appear anywhere in `expr`,
    /// including inside its blocks?
    fn mentions_held_closure(&self, expr: &Expr) -> bool {
        match expr {
            Expr::If(i) => {
                self.mentions_held_closure(&i.cond)
                    || self.block_mentions_held_closure(&i.then_block)
                    || i.else_block
                        .as_ref()
                        .is_some_and(|b| self.block_mentions_held_closure(b))
            }
            Expr::Match(m) => {
                self.mentions_held_closure(&m.expr)
                    || m.arms
                        .iter()
                        .any(|a| self.block_mentions_held_closure(&a.body))
            }
            Expr::UnsafeBlock(u) => {
                self.block_mentions_held_closure(&u.stmts)
                    || u.ret
                        .as_ref()
                        .is_some_and(|r| self.mentions_held_closure(r))
            }
            _ => self.may_hold_closure(expr),
        }
    }

    fn block_mentions_held_closure(&self, stmts: &[Statement]) -> bool {
        stmts.iter().any(|s| match s {
            Statement::LetDecl(l) => self.mentions_held_closure(&l.expr),
            Statement::ExprStmt(e) => self.mentions_held_closure(&e.expr),
            Statement::Return(r) => r
                .expr
                .as_ref()
                .is_some_and(|e| self.mentions_held_closure(e)),
            Statement::Assign(a) => self.mentions_held_closure(&a.rhs),
            _ => false,
        })
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
    /// dereference on the way.
    fn closure_place_root(expr: &Expr) -> Option<(crate::symbol::Symbol, bool)> {
        match expr {
            Expr::Identifier(id) => Some((id.name.clone(), false)),
            Expr::MemberAccess(m) => Self::closure_place_root(&m.base),
            Expr::IndexAccess(i) => Self::closure_place_root(&i.base),
            Expr::Borrow(b) => Self::closure_place_root(&b.expr),
            Expr::Dereference(d) => Self::closure_place_root(&d.expr).map(|(r, _)| (r, true)),
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

    /// Is `name` the struct a closure literal became, and did that closure use any variable?
    fn closure_uses_variables(&self, name: &crate::symbol::Symbol) -> bool {
        name.starts_with("Closure_")
            && self
                .mono
                .generated_structs
                .iter()
                .any(|s| s.name == *name && !s.fields.is_empty())
    }

    /// Could a value of type `ty` hold a `ClosureK` value? A type this cannot see into, such
    /// as a type parameter, is assumed to.
    pub(crate) fn type_can_hold_closure(&self, ty: &Type) -> bool {
        self.type_can_hold_closure_in(ty, &mut HashSet::new())
    }

    fn type_can_hold_closure_in(&self, ty: &Type, seen: &mut HashSet<String>) -> bool {
        let is_closure_k = |name: &str| {
            name.strip_prefix("Closure")
                .and_then(|rest| rest.chars().next())
                .is_some_and(|c| c.is_ascii_digit())
        };
        match ty {
            Type::Struct(name, _) | Type::Enum(name, _) if is_closure_k(name) => true,
            Type::Struct(name, _) | Type::Enum(name, _) => {
                if !seen.insert(name.to_string()) {
                    return false;
                }
                self.declared_part_types(name)
                    .iter()
                    .any(|t| self.type_can_hold_closure_in(t, seen))
            }
            Type::GenericInstance(base, args) => {
                self.type_can_hold_closure_in(base, seen)
                    || args.iter().any(|a| self.type_can_hold_closure_in(a, seen))
            }
            Type::Ref(inner, _)
            | Type::Borrow { inner, .. }
            | Type::Pointer(inner, ..)
            | Type::Pinned(inner, _)
            | Type::Verified(inner) => self.type_can_hold_closure_in(inner, seen),
            Type::Closure(..) | Type::Generic(..) | Type::Unknown => true,
            _ => false,
        }
    }

    /// The field types of a struct, or the payload types of an enum, by name.
    fn declared_part_types(&self, name: &crate::symbol::Symbol) -> Vec<Type> {
        let base = name.split('<').next().unwrap_or(name);
        if let Some(s) = self.env.structs.get(base) {
            return s.fields.iter().map(|(_, t)| t.clone()).collect();
        }
        if let Some(s) = self
            .mono
            .generated_structs
            .iter()
            .find(|s| s.name.as_ref() == base)
        {
            return s.fields.iter().map(|(_, t)| t.clone()).collect();
        }
        if let Some(e) = self.env.enums.get(base) {
            return e
                .variants
                .iter()
                .flat_map(|(_, p)| p.clone().unwrap_or_default())
                .collect();
        }
        Vec::new()
    }
}
