//===- assoc_types.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

use crate::hir::TypeChecker;

impl TypeChecker<'_> {
    /// Every impl of a trait binds every associated type that trait declares, and binds nothing
    /// else.
    ///
    /// The binding is substituted into the impl's signatures before name resolution, so by the
    /// time a body is checked there is no `Self::Item` left to complain about. A missing binding
    /// would therefore not be reported anywhere: the impl's own methods say a concrete type, and
    /// they would simply be trusted. This is the only place that sees both halves.
    pub fn check_associated_type_bindings(&mut self) {
        // Collected before reporting, which borrows the checker mutably.
        let mut missing: Vec<(crate::symbol::Symbol, crate::symbol::Symbol, String)> = Vec::new();
        let mut unknown: Vec<(crate::symbol::Symbol, crate::symbol::Symbol, String)> = Vec::new();

        for (trait_name, blocks) in &self.env.impls {
            let Some(decl) = self.env.traits.get(trait_name) else {
                continue;
            };
            for block in blocks {
                let target = block.target_type.to_string();
                for declared in &decl.assoc_types {
                    if !block
                        .assoc_bindings
                        .iter()
                        .any(|(bound, _)| bound.as_ref() == declared.as_ref())
                    {
                        missing.push((trait_name.clone(), declared.clone(), target.clone()));
                    }
                }
                for (bound, _) in &block.assoc_bindings {
                    if !decl
                        .assoc_types
                        .iter()
                        .any(|declared| declared.as_ref() == bound.as_ref())
                    {
                        unknown.push((trait_name.clone(), bound.clone(), target.clone()));
                    }
                }
            }
        }

        // `env.impls` is a hash map, so a program with two faults would report them in a
        // different order on every run without this.
        missing.sort();
        unknown.sort();

        for (trait_name, assoc, target) in missing {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3037,
                format!(
                    "`impl {trait_name} for {target}` does not bind the associated type \
                     '{assoc}'; write `type {assoc} = ..;` in the impl"
                ),
                None,
            );
        }
        for (trait_name, assoc, target) in unknown {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3038,
                format!(
                    "`impl {trait_name} for {target}` binds '{assoc}', which trait \
                     {trait_name} does not declare"
                ),
                None,
            );
        }
    }

    /// No struct field names `I::Item`. A field's type is read from the struct's own parameters
    /// wherever a value is laid out, which is in both code generators as well as here, and none
    /// of them selects an impl. Rust's `Map<I, F>` takes the closure as a parameter for the
    /// same reason, and that shape works here.
    pub fn check_no_projections_in_fields(&mut self) {
        let mut found: Vec<(String, String, String)> = Vec::new();
        for (name, decl) in &self.env.structs {
            for (field, ty) in &decl.fields {
                let mut written = Vec::new();
                crate::hir::check::projection::projections_in(ty, &mut written);
                if let Some(p) = written.first() {
                    found.push((name.to_string(), field.to_string(), p.to_string()));
                }
            }
        }
        found.sort();
        for (name, field, projection) in found {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3041,
                format!(
                    "field '{field}' of struct '{name}' names `{projection}`, which a struct field \
                     cannot yet; make the field's type a parameter of '{name}'"
                ),
                None,
            );
        }
    }
}
