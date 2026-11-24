//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A type may declare itself `Copy` only if everything it holds is `Copy` too.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Refuse `impl Copy for X` when one of `X`'s fields is a type that moves (E3031).
    ///
    /// `Copy` means a value can be used again after it has been assigned somewhere else, so
    /// every field has to survive the same treatment. A struct holding a tensor cannot: the
    /// tensor is linear because moving it is what the placement discipline is made of, and
    /// duplicating it is an explicit `transfer`. Allowing the struct to be `Copy` would be a
    /// way to duplicate the tensor inside it without saying so.
    ///
    /// This is the half of the rule that makes it safe to be opt-in. Declaring `Copy` is a
    /// promise, and adding a tensor field to a struct that made that promise fails here
    /// rather than quietly changing what assignment means.
    pub fn check_copy_impls(&mut self) {
        let Some(impls) = self.env.impls.get("Copy") else {
            return;
        };
        // The targets are collected first because reporting borrows the checker mutably.
        // An impl records its target as a struct whatever the declaration turns out to be,
        // so both tables are consulted by name. `impl<T> Copy for Twice<T>` also records the
        // type arguments of the target, and the type parameters that have a `Copy` bound.
        let targets: Vec<CopyTarget> = impls
            .iter()
            .filter_map(|ib| {
                let (name, args) = match &ib.target_type {
                    Type::Struct(name, _) | Type::Enum(name, _) => (name.clone(), Vec::new()),
                    Type::GenericInstance(base, args) => match base.as_ref() {
                        Type::Struct(name, _) | Type::Enum(name, _) => (name.clone(), args.clone()),
                        _ => return None,
                    },
                    _ => return None,
                };
                let mut params = Vec::new();
                let mut copy_params = Vec::new();
                for g in &ib.generics {
                    if let crate::syntax::decl::GenericParam::Type { name, bounds } = g {
                        params.push(name.clone());
                        if bounds.iter().any(|b| b.trait_name.as_ref() == "Copy") {
                            copy_params.push(name.clone());
                        }
                    }
                }
                Some(CopyTarget {
                    name,
                    args,
                    params,
                    copy_params,
                    place: ib.place.clone(),
                })
            })
            .collect();

        // Each error points at the `impl Copy`.
        let outside = self.errors.here();
        for target in targets {
            let name = &target.name;
            self.errors.point_at(&target.place, &outside);
            let mut generics: &[crate::syntax::decl::GenericParam] = &[];
            let mut held: Vec<(String, Type)> = Vec::new();
            if let Some(decl) = self.env.structs.get(name.as_ref()) {
                generics = &decl.generics;
                for (field, ty) in &decl.fields {
                    held.push((format!("field '{}'", field), ty.clone()));
                }
            }
            if let Some(decl) = self.env.enums.get(name.as_ref()) {
                generics = &decl.generics;
                for (variant, payload) in &decl.variants {
                    for ty in payload.iter().flatten() {
                        held.push((format!("variant '{}'", variant), ty.clone()));
                    }
                }
            }
            // The type parameters of the declaration, as the impl names them.
            let mapping: std::collections::HashMap<crate::symbol::Symbol, Type> = generics
                .iter()
                .filter_map(|g| match g {
                    crate::syntax::decl::GenericParam::Type { name, .. } => Some(name.clone()),
                    crate::syntax::decl::GenericParam::Const { .. } => None,
                })
                .zip(target.args.iter().cloned())
                .collect();
            for (what, ty) in held {
                let ty = ty.substitute(&mapping);
                // With each parameter a `Copy` type, a field that still moves needs more than a
                // bound. One that copies then needs a `Copy` bound on each parameter it holds.
                let each_param_copies = params_as_copy(&ty, &target.params);
                if each_param_copies.is_linear() && !self.is_copy(&each_param_copies) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3031,
                        format!(
                            "'{}' cannot be `Copy`: its {} has type {}, which is moved rather \
                             than copied",
                            name, what, ty
                        ),
                        None,
                    );
                } else if let Some(param) = param_held_by_value(&ty, &target.params)
                    .filter(|p| !target.copy_params.contains(p))
                {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3031,
                        format!(
                            "'{name}' cannot be `Copy` for each `{param}`: its {what} has type \
                             {ty}. Write `{param} : Copy` in the impl"
                        ),
                        None,
                    );
                }
            }
        }
        self.errors.point_back(outside);
    }
}

/// An `impl Copy`: the type, the type arguments of its target, its type parameters, and those
/// of them with a `Copy` bound.
struct CopyTarget {
    name: crate::symbol::Symbol,
    args: Vec<Type>,
    params: Vec<crate::symbol::Symbol>,
    copy_params: Vec<crate::symbol::Symbol>,
    place: crate::syntax::DeclPlace,
}

/// The first of `params` that `ty` holds by value. A reference, a pointer or a function type
/// holds no value of the parameter, so a copy of it copies none.
fn param_held_by_value<'a>(
    ty: &Type,
    params: &'a [crate::symbol::Symbol],
) -> Option<&'a crate::symbol::Symbol> {
    match ty {
        Type::Generic(name, _) | Type::Struct(name, None) => params.iter().find(|p| *p == name),
        Type::GenericInstance(_, args) => args.iter().find_map(|a| param_held_by_value(a, params)),
        Type::Verified(inner)
        | Type::Distinct(inner)
        | Type::DisjointRanges(inner)
        | Type::Pinned(inner, _) => param_held_by_value(inner, params),
        _ => None,
    }
}

/// `ty` with each of `params` replaced by `i64`, a type that is `Copy`.
fn params_as_copy(ty: &Type, params: &[crate::symbol::Symbol]) -> Type {
    match ty {
        Type::Generic(name, _) | Type::Struct(name, None) if params.contains(name) => {
            Type::Scalar(ElementType::I64)
        }
        Type::GenericInstance(base, args) => Type::GenericInstance(
            base.clone(),
            args.iter().map(|a| params_as_copy(a, params)).collect(),
        ),
        _ => ty.clone(),
    }
}
