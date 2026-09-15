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
        // so both tables are consulted by name.
        let targets: Vec<crate::symbol::Symbol> = impls
            .iter()
            .filter_map(|ib| match &ib.target_type {
                Type::Struct(name, _) | Type::Enum(name, _) => Some(name.clone()),
                _ => None,
            })
            .collect();

        for name in targets {
            let mut held: Vec<(String, Type)> = Vec::new();
            if let Some(decl) = self.env.structs.get(name.as_ref()) {
                for (field, ty) in &decl.fields {
                    held.push((format!("field '{}'", field), ty.clone()));
                }
            }
            if let Some(decl) = self.env.enums.get(name.as_ref()) {
                for (variant, payload) in &decl.variants {
                    for ty in payload.iter().flatten() {
                        held.push((format!("variant '{}'", variant), ty.clone()));
                    }
                }
            }
            for (what, ty) in held {
                if ty.is_linear() && !self.is_copy(&ty) {
                    self.errors.error_with_code(
                        crate::diagnostic::DiagnosticCode::E3031,
                        format!(
                            "'{}' cannot be `Copy`: its {} has type {}, which is moved rather \
                             than copied",
                            name, what, ty
                        ),
                        None,
                    );
                }
            }
        }
    }
}
