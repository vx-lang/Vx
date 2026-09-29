//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A bound on a type parameter has to name a trait that exists.
//
//===----------------------------------------------------------------------===//

use super::super::*;

impl TypeChecker<'_> {
    /// Refuse a bound such as `T : NoSuchTrait` whose trait is not declared (E3042).
    ///
    /// A bound is otherwise only asked when a call binds the parameter, so a misspelled one
    /// on a function nobody calls, or on a struct, was never reported.
    pub fn check_bounds_name_a_trait(&mut self) {
        // (where the bound is written, the parameter, the trait it names)
        let mut unknown: Vec<(String, String, String)> = Vec::new();
        let traits = &self.env.traits;
        let mut collect = |owner: &str, generics: &[decl::GenericParam]| {
            for param in generics {
                if let decl::GenericParam::Type { name, bounds } = param {
                    for bound in bounds {
                        // `D : Topology` marks a topology parameter; it is a keyword, not a trait.
                        if bound.trait_name.as_ref() == "Topology" {
                            continue;
                        }
                        if !traits.contains_key(&bound.trait_name) {
                            unknown.push((
                                owner.to_string(),
                                name.to_string(),
                                bound.trait_name.to_string(),
                            ));
                        }
                    }
                }
            }
        };

        for s in self.env.structs.values() {
            collect(&format!("struct {}", s.name), &s.generics);
        }
        for e in self.env.enums.values() {
            collect(&format!("enum {}", e.name), &e.generics);
        }
        for t in self.env.traits.values() {
            collect(&format!("trait {}", t.name), &t.generics);
            for m in &t.methods {
                collect(&format!("method {}::{}", t.name, m.name), &m.generics);
            }
        }
        for block in self.env.impls.values().flatten() {
            let owner = match &block.trait_name {
                Some(t) => format!("impl {} for {}", t, block.target_type),
                None => format!("impl {}", block.target_type),
            };
            collect(&owner, &block.generics);
            let impl_params: Vec<&str> = block.generics.iter().map(|g| g.name()).collect();
            for m in &block.methods {
                // A method is also given its impl's parameters, already reported above.
                let own: Vec<decl::GenericParam> = m
                    .generics
                    .iter()
                    .filter(|g| !impl_params.contains(&g.name()))
                    .cloned()
                    .collect();
                collect(&format!("method {}::{}", block.target_type, m.name), &own);
            }
        }
        for (f, _) in self.env.generic_functions.values() {
            collect(&format!("fn {}", f.name), &f.generics);
        }

        // The tables are hash maps, and a generic method can be listed under its impl and as a
        // function, so sort for a stable order and drop the repeats.
        unknown.sort();
        unknown.dedup();
        for (owner, param, trait_name) in unknown {
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3042,
                format!(
                    "the bound `{param} : {trait_name}` on `{owner}` names no trait; \
                     no trait called '{trait_name}' is declared or imported"
                ),
                None,
            );
        }
    }
}
