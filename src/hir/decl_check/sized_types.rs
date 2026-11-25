//===- decl_check submodule - Vx Compiler -----------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A struct or an enum that holds itself by value has no finite size. A pointer, such as the
// one in a `Box`, ends the chain.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::{HashMap, HashSet};

/// One step of a chain of values held by value: the struct or enum, what it calls the part
/// (a field or a variant), and the type of that part.
struct Step {
    owner: String,
    part: String,
    ty: Type,
}

impl TypeChecker<'_> {
    /// E3068 for each struct or enum that holds itself by value, through its own fields, other
    /// structs, enum variants or generic instances such as `Option<Node>`.
    pub fn check_types_have_a_size(&mut self) {
        let mut decls: Vec<(String, &crate::syntax::DeclPlace)> = self
            .env
            .structs
            .values()
            .filter(|s| s.generics.is_empty())
            .map(|s| (s.name.to_string(), &s.place))
            .chain(
                self.env
                    .enums
                    .values()
                    .filter(|e| e.generics.is_empty())
                    .map(|e| (e.name.to_string(), &e.place)),
            )
            .collect();
        // The tables are hash maps. In source order, a cycle is reported at its first type.
        decls.sort_by(|a, b| {
            (&a.1.file, a.1.span.line, a.1.span.column, &a.0).cmp(&(
                &b.1.file,
                b.1.span.line,
                b.1.span.column,
                &b.0,
            ))
        });
        let mut reported: HashSet<String> = HashSet::new();
        let outside = self.errors.here();
        for (name, place) in decls {
            if reported.contains(&name) {
                continue;
            }
            let start = Type::Struct(name.as_str().into(), None);
            let mut chain = Vec::new();
            let mut seen = HashSet::new();
            if !self.holds_by_value(&start, &name, &mut chain, &mut seen, 0) {
                continue;
            }
            reported.extend(chain.iter().map(|s| base_name(&s.owner).to_string()));
            let steps: Vec<String> = chain
                .iter()
                .map(|s| format!("{} of `{}` holds `{}`", s.part, s.owner, s.ty))
                .collect();
            self.errors.point_at(place, &outside);
            self.errors.error_with_code(
                crate::diagnostic::DiagnosticCode::E3068,
                format!(
                    "`{name}` holds itself by value, so its size is infinite: {}. Hold the value \
                     through a pointer, for example `Box<{name}>`",
                    steps.join(", ")
                ),
                None,
            );
        }
        self.errors.point_back(outside);
    }

    /// Whether a value of type `ty` holds a `target` by value. On `true`, `chain` holds the
    /// steps from `ty` to `target`.
    fn holds_by_value(
        &self,
        ty: &Type,
        target: &str,
        chain: &mut Vec<Step>,
        seen: &mut HashSet<String>,
        depth: usize,
    ) -> bool {
        // A generic struct can name a larger instance of itself in a field, such as
        // `G<Option<T>>`, without a cycle of names. Such a chain is not followed to its end.
        if depth > 64 {
            return false;
        }
        let (name, args): (&crate::symbol::Symbol, &[Type]) = match ty {
            Type::Struct(name, _) | Type::Enum(name, _) => (name, &[]),
            Type::GenericInstance(base, args) => match base.as_ref() {
                Type::Struct(name, _) | Type::Enum(name, _) => (name, args.as_slice()),
                _ => return false,
            },
            Type::Pinned(inner, _)
            | Type::Verified(inner)
            | Type::Distinct(inner)
            | Type::DisjointRanges(inner) => {
                return self.holds_by_value(inner, target, chain, seen, depth)
            }
            // A pointer, a borrow, a tensor, a closure and a scalar do not hold a struct by value.
            _ => return false,
        };
        if depth > 0 && name.as_ref() == target && args.is_empty() {
            return true;
        }
        if !seen.insert(ty.to_string()) {
            return false;
        }
        let base = name.split('<').next().unwrap_or(name);
        let (generics, parts): (&[crate::syntax::decl::GenericParam], Vec<(String, &Type)>) =
            if let Some(s) = self.env.structs.get(base) {
                let parts = s
                    .fields
                    .iter()
                    .map(|(f, t)| (format!("field `{f}`"), t))
                    .collect();
                (&s.generics, parts)
            } else if let Some(e) = self.env.enums.get(base) {
                let parts = e
                    .variants
                    .iter()
                    .flat_map(|(v, payload)| {
                        payload
                            .iter()
                            .flatten()
                            .map(move |t| (format!("variant `{v}`"), t))
                    })
                    .collect();
                (&e.generics, parts)
            } else {
                return false;
            };
        let mapping: HashMap<crate::symbol::Symbol, Type> = generics
            .iter()
            .map(|g| crate::symbol::Symbol::from(g.name()))
            .zip(args.iter().cloned())
            .collect();
        for (part, part_ty) in parts {
            let part_ty = part_ty.substitute(&mapping);
            chain.push(Step {
                owner: ty.to_string(),
                part,
                ty: part_ty.clone(),
            });
            if self.holds_by_value(&part_ty, target, chain, seen, depth + 1) {
                return true;
            }
            chain.pop();
        }
        false
    }
}

/// `Option` for `Option<Node>`.
fn base_name(name: &str) -> &str {
    name.split('<').next().unwrap_or(name)
}
