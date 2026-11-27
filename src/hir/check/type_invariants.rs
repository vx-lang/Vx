//===- type_invariants.rs - Vx Compiler ------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Type invariants: `struct Vec<T> { .., invariant len <= capacity }`. Only the module that
// declares such a type can change its fields, so the invariant holds in each other module.
// docs/discussions/type_invariants.md has the design.
//
//===----------------------------------------------------------------------===//

use super::super::*;

/// The functions that an invariant can call, with the number of their arguments.
const POINTER_CLAUSES: [(&str, usize); 3] = [("valid", 2), ("initialized", 2), ("overlaps", 3)];

impl<'a> TypeChecker<'a> {
    /// The declaration of the struct that `ty` names, also through a reference or a raw pointer.
    fn struct_decl_of(&self, ty: &Type) -> Option<&'a StructDecl> {
        let ty = match ty {
            Type::Borrow { inner, .. } | Type::Pointer(inner, ..) | Type::Ref(inner, _) => inner,
            ty => ty,
        };
        let name = match ty {
            Type::Struct(name, _) => name,
            Type::GenericInstance(base, _) => match &**base {
                Type::Struct(name, _) => name,
                _ => return None,
            },
            _ => return None,
        };
        self.env.structs.get(name).copied()
    }

    /// The struct with an invariant that `ty` names, when another module declares it.
    fn invariant_type_of_another_module(&self, ty: &Type) -> Option<&'a StructDecl> {
        self.struct_decl_of(ty)
            .filter(|decl| !decl.invariants.is_empty() && decl.place.file != self.errors.file)
    }

    /// The file name of the module of `decl`, for a message.
    fn module_file_name(decl: &StructDecl) -> String {
        let file = decl.place.file.as_deref().unwrap_or("its module");
        let name = std::path::Path::new(file).file_name();
        name.map_or(file.to_string(), |n| n.to_string_lossy().into_owned())
    }

    /// E8014 when `place` is a field of a type with an invariant, at any depth, and another
    /// module declares the type. `change` says what changes the place, as in "this assignment
    /// changes".
    pub(crate) fn check_field_change(&mut self, place: &Expr, change: &str) {
        if self.speculating {
            return;
        }
        let mut place = place;
        loop {
            match place {
                Expr::MemberAccess(m) => {
                    let base = self.check_expr_type_probe(&m.base);
                    if let Some(decl) = self.invariant_type_of_another_module(&base) {
                        let message = format!(
                            "{change} the field `{}` of `{}`. `{}` has an invariant, so only \
                             code in `{}` can change its fields",
                            m.member,
                            decl.name,
                            decl.name,
                            Self::module_file_name(decl)
                        );
                        self.errors.error_with_code(
                            crate::diagnostic::DiagnosticCode::E8014,
                            message,
                            Some(crate::diagnostic::SourceSpan::from_ast_span(&m.span)),
                        );
                        return;
                    }
                    place = &m.base;
                }
                // An element of an array field is in the field. An element behind a reference or
                // a raw pointer is not.
                Expr::IndexAccess(i)
                    if !matches!(
                        self.check_expr_type_probe(&i.base),
                        Type::Borrow { .. } | Type::Pointer(..) | Type::Ref(..)
                    ) =>
                {
                    place = &i.base
                }
                _ => return,
            }
        }
    }

    /// Each struct with an invariant that a value of `ty` holds: the type itself, a type argument,
    /// a field or a payload at any depth, or a value behind a reference. Not a value behind a raw
    /// pointer, because a read through one is `unsafe`.
    pub(crate) fn invariant_types_held_by(&self, ty: &Type) -> Vec<&'a StructDecl> {
        let mut found = Vec::new();
        self.collect_invariant_types(ty, &mut std::collections::HashSet::new(), &mut found);
        found
    }

    fn collect_invariant_types(
        &self,
        ty: &Type,
        seen: &mut std::collections::HashSet<crate::symbol::Symbol>,
        found: &mut Vec<&'a StructDecl>,
    ) {
        match ty {
            Type::Borrow { inner, .. } | Type::Ref(inner, _) => {
                self.collect_invariant_types(inner, seen, found)
            }
            Type::GenericInstance(base, args) => {
                self.collect_invariant_types(base, seen, found);
                for a in args {
                    self.collect_invariant_types(a, seen, found);
                }
            }
            Type::Struct(name, _) | Type::Enum(name, _) if seen.insert(name.clone()) => {
                if let Some(decl) = self.env.structs.get(name).copied() {
                    if !decl.invariants.is_empty() {
                        found.push(decl);
                    }
                    for (_, t) in &decl.fields {
                        self.collect_invariant_types(t, seen, found);
                    }
                } else if let Some(decl) = self.env.enums.get(name).copied() {
                    for t in decl.variants.iter().flat_map(|(_, p)| p.iter().flatten()) {
                        self.collect_invariant_types(t, seen, found);
                    }
                }
            }
            _ => {}
        }
    }

    /// E8014 for an `mlir!` block that makes a value of a type with an invariant of another
    /// module, or a value that holds a value of that type. The block makes the value without the
    /// functions of that module, as a struct literal does. `unsafe` does not change this.
    pub(crate) fn check_mlir_block_module(&mut self, returns: &Type, span: &Span) {
        if self.speculating {
            return;
        }
        let held = self.invariant_types_held_by(returns);
        let Some(decl) = held.into_iter().find(|d| d.place.file != self.errors.file) else {
            return;
        };
        let name = &decl.name;
        let makes = match returns.to_string() == name.as_ref() {
            true => format!("this `mlir!` block makes a value of `{name}`"),
            false => format!(
                "this `mlir!` block makes a value of `{returns}`, which holds a value of `{name}`"
            ),
        };
        let message = format!(
            "{makes}. `{name}` has an invariant, so only code in `{}` can make a value of `{name}`",
            Self::module_file_name(decl)
        );
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8014,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// E8014 for a struct literal of a type with an invariant that another module declares.
    pub(crate) fn check_struct_literal_module(&mut self, name: &str, span: &Span) {
        let Some(decl) = self.env.structs.get(name).copied() else {
            return;
        };
        if self.speculating || decl.invariants.is_empty() || decl.place.file == self.errors.file {
            return;
        }
        let article = match name.chars().next() {
            Some('A' | 'E' | 'I' | 'O' | 'U' | 'a' | 'e' | 'i' | 'o' | 'u') => "an",
            _ => "a",
        };
        let message = format!(
            "this struct literal builds {article} `{name}`. `{name}` has an invariant, so only code \
             in `{}` can build it from its fields",
            Self::module_file_name(decl)
        );
        self.errors.error_with_code(
            crate::diagnostic::DiagnosticCode::E8014,
            message,
            Some(crate::diagnostic::SourceSpan::from_ast_span(span)),
        );
    }

    /// E8013 for each invariant of a struct that is not well formed.
    pub(crate) fn check_type_invariants_are_well_formed(&mut self) {
        let mut decls: Vec<&'a StructDecl> = self
            .env
            .structs
            .values()
            .copied()
            .filter(|s| !s.invariants.is_empty())
            .collect();
        decls.sort_by(|a, b| a.name.cmp(&b.name));
        for decl in decls {
            let prev_file = std::mem::replace(&mut self.errors.file, decl.place.file.clone());
            for invariant in &decl.invariants {
                if let Err((span, why)) = self.check_invariant_expr(decl, invariant) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E8013,
                        format!(
                            "this invariant of `{}` is not well formed: {why}",
                            decl.name
                        ),
                        Some(crate::diagnostic::SourceSpan::from_ast_span(&span)),
                    );
                }
            }
            self.errors.file = prev_file;
        }
    }

    /// Whether `e`, a part of an invariant of `decl`, uses only what an invariant can use.
    fn check_invariant_expr(&self, decl: &StructDecl, e: &Expr) -> Result<(), (Span, String)> {
        match e {
            Expr::Number(_) => Ok(()),
            Expr::Identifier(id) if matches!(id.name.as_ref(), "true" | "false") => Ok(()),
            Expr::Identifier(_) | Expr::MemberAccess(_) | Expr::IndexAccess(_) => {
                self.invariant_place_type(decl, e).map(|_| ())
            }
            Expr::BinaryOp(b) => {
                self.check_invariant_expr(decl, &b.lhs)?;
                self.check_invariant_expr(decl, &b.rhs)
            }
            Expr::RelationalOp(r) => {
                self.check_invariant_expr(decl, &r.lhs)?;
                self.check_invariant_expr(decl, &r.rhs)
            }
            Expr::LogicalOp(l) => {
                self.check_invariant_expr(decl, &l.lhs)?;
                self.check_invariant_expr(decl, &l.rhs)
            }
            Expr::UnaryOp(u) => self.check_invariant_expr(decl, &u.expr),
            Expr::AsCast(c) => self.check_invariant_expr(decl, &c.expr),
            Expr::FunctionCall(fc) => {
                let Some((_, arity)) = POINTER_CLAUSES
                    .iter()
                    .find(|(name, _)| *name == fc.name.as_ref())
                else {
                    return Err((
                        fc.span,
                        format!(
                            "it calls `{}`, and an invariant can call only `valid`, \
                             `initialized` and `overlaps`",
                            fc.name
                        ),
                    ));
                };
                if fc.args.len() != *arity {
                    return Err((fc.span, format!("`{}` takes {arity} arguments", fc.name)));
                }
                fc.args
                    .iter()
                    .try_for_each(|a| self.check_invariant_expr(decl, a))
            }
            Expr::MethodCall(mc) => Err((
                mc.span,
                format!(
                    "it calls the method `{}`, and an invariant can call only `valid`, \
                     `initialized` and `overlaps`",
                    mc.method_name
                ),
            )),
            _ => Err((
                e.span(),
                format!(
                    "an invariant can use the fields of `{}` at any depth, numbers, operators, \
                     `as` and the pointer clauses `valid`, `initialized` and `overlaps`",
                    decl.name
                ),
            )),
        }
    }

    /// The type of the place `e` in an invariant of `decl`: `None` for `self`, and the type of
    /// the field for a field at any depth. The parser writes each field `lo` as `self.lo`.
    fn invariant_place_type(
        &self,
        decl: &StructDecl,
        e: &Expr,
    ) -> Result<Option<Type>, (Span, String)> {
        match e {
            Expr::Identifier(id) if id.name.as_ref() == "self" => Ok(None),
            Expr::Identifier(id) => Err((
                id.span,
                format!(
                    "it names `{}`, and an invariant can name only the fields of `{}`",
                    id.name, decl.name
                ),
            )),
            Expr::MemberAccess(m) => {
                let base = self.invariant_place_type(decl, &m.base)?;
                let (owner, fields) = match &base {
                    None => (decl.name.to_string(), &decl.fields),
                    Some(ty) => match self.struct_decl_of(ty) {
                        Some(d) if !matches!(ty, Type::Borrow { .. } | Type::Pointer(..)) => {
                            (ty.to_string(), &d.fields)
                        }
                        _ => {
                            return Err((
                                m.span,
                                format!(
                                    "`{ty}` is not a struct, so it has no field `{}`",
                                    m.member
                                ),
                            ))
                        }
                    },
                };
                match fields.iter().find(|(name, _)| *name == m.member) {
                    Some((_, ty)) => Ok(Some(ty.clone())),
                    None => Err((m.span, format!("`{owner}` has no field `{}`", m.member))),
                }
            }
            Expr::IndexAccess(i) => {
                self.check_invariant_expr(decl, &i.index)?;
                match self.invariant_place_type(decl, &i.base)? {
                    Some(Type::Tensor(element, ..)) => Ok(Some(Type::Scalar(element))),
                    _ => Err((
                        i.span,
                        "an invariant can index only an array field".to_string(),
                    )),
                }
            }
            // The base of a field or an element is not a field, as in `make().a`.
            _ => {
                self.check_invariant_expr(decl, e)?;
                Err((
                    e.span(),
                    format!(
                        "it reads a part of a value that is not a field of `{}`",
                        decl.name
                    ),
                ))
            }
        }
    }
}
