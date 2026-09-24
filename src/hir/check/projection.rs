//===- projection.rs - Vx Compiler ------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// `I::Item`: the associated type of a generic parameter. The parser reads it as a type
// variable named `I::Item`, so the substitution that instantiates a generic already replaces
// it, provided the mapping has an entry for it. This file adds that entry: once `I` is bound,
// the impl that applies to what `I` is says what `Item` is.
//
//===----------------------------------------------------------------------===//

use super::super::*;
use std::collections::HashMap;

/// Every `P::A` written in `ty`.
pub(crate) fn projections_in(ty: &Type, out: &mut Vec<crate::symbol::Symbol>) {
    match ty {
        Type::Generic(name, _) if name.as_ref().contains("::") => out.push(name.clone()),
        Type::Ref(t, _)
        | Type::Borrow { inner: t, .. }
        | Type::Pointer(t, _, _)
        | Type::Verified(t)
        | Type::Pinned(t, _) => projections_in(t, out),
        Type::GenericInstance(base, args) => {
            projections_in(base, out);
            args.iter().for_each(|a| projections_in(a, out));
        }
        Type::Function(params, ret, _) | Type::Closure(params, ret) => {
            params.iter().for_each(|p| projections_in(p, out));
            projections_in(ret, out);
        }
        _ => {}
    }
}

impl<'a> TypeChecker<'a> {
    /// Bind the projections of an instantiation of `callee`, whose signature is `params` and
    /// `ret`, and report the ones that disagree or that nothing binds. False when one did.
    pub(crate) fn resolve_projections_for(
        &mut self,
        callee: &str,
        params: &[&Type],
        ret: &Type,
        mapping: &mut HashMap<crate::symbol::Symbol, Type>,
        span: Option<crate::diagnostic::SourceSpan>,
    ) -> bool {
        if let Err(msg) = self.bind_projections(mapping) {
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3039,
                    format!("in the call to '{callee}': {msg}"),
                    span,
                );
            }
            return false;
        }
        let mut written = Vec::new();
        params.iter().for_each(|p| projections_in(p, &mut written));
        projections_in(ret, &mut written);
        for name in written {
            if mapping.contains_key(&name) {
                continue;
            }
            let (param, assoc) = name
                .as_ref()
                .split_once("::")
                .expect("a projection is spelled `P::A`");
            // A parameter nothing bound is reported where the binding failed.
            let Some(ty) = mapping.get(param) else {
                continue;
            };
            if !self.speculating {
                self.errors.error_with_code(
                    crate::diagnostic::DiagnosticCode::E3040,
                    format!(
                        "in the call to '{callee}': `{name}` does not resolve, because no impl \
                         for `{ty}` binds an associated type named '{assoc}'"
                    ),
                    span,
                );
            }
            return false;
        }
        true
    }

    /// Does impl block `ib` apply to `ty`? The mapping it binds is left in `mapping`, with the
    /// block's own projections resolved.
    ///
    /// A projection written in the block's header, `impl<I, U> .. for Map<I, Closure1<I::Item, U>>`,
    /// is bound by unification like any other variable. That binding is a claim about `I`, so the
    /// block applies only when it agrees with what `I`'s own impl says.
    pub(crate) fn impl_applies(
        &mut self,
        ib: &decl::ImplBlock,
        ty: &Type,
        mapping: &mut HashMap<crate::symbol::Symbol, Type>,
    ) -> bool {
        self.unify_types(&ib.target_type, ty, mapping) && self.bind_projections(mapping).is_ok()
    }

    /// Add an entry for `P::A` to `mapping` for every parameter `P` it binds and every associated
    /// type `A` an impl for `P`'s type binds. An entry already there, bound by unification from an
    /// argument, has to agree; the disagreement is the error.
    pub(crate) fn bind_projections(
        &mut self,
        mapping: &mut HashMap<crate::symbol::Symbol, Type>,
    ) -> Result<(), String> {
        let mut params: Vec<(crate::symbol::Symbol, Type)> = mapping
            .iter()
            .filter(|(k, _)| !k.as_ref().contains("::"))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        for (param, ty) in params {
            for (assoc, bound) in self.associated_types_of(&ty)? {
                let key = crate::symbol::Symbol::from(format!("{param}::{assoc}").as_str());
                match mapping.get(&key) {
                    Some(existing) if *existing != bound => {
                        return Err(format!(
                            "`{key}` is {existing} here, but `{ty}` binds `{assoc}` to {bound}"
                        ));
                    }
                    Some(_) => {}
                    None => {
                        mapping.insert(key, bound);
                    }
                }
            }
        }
        Ok(())
    }

    /// What each impl that applies to `ty` binds its associated types to, in terms of `ty`.
    fn associated_types_of(
        &mut self,
        ty: &Type,
    ) -> Result<Vec<(crate::symbol::Symbol, Type)>, String> {
        // By trait name, so a report naming two of them does not change from run to run.
        let mut keyed: Vec<(&crate::symbol::Symbol, &&decl::ImplBlock)> = self
            .env
            .impls
            .iter()
            .flat_map(|(t, blocks)| blocks.iter().map(move |ib| (t, ib)))
            .filter(|(_, ib)| !ib.assoc_bindings.is_empty())
            .collect();
        keyed.sort_by(|a, b| a.0.cmp(b.0));
        let blocks: Vec<decl::ImplBlock> = keyed.into_iter().map(|(_, ib)| (*ib).clone()).collect();
        let mut out: Vec<(crate::symbol::Symbol, Type)> = Vec::new();
        for ib in blocks {
            let mut m = HashMap::new();
            if !self.impl_applies(&ib, ty, &mut m) {
                continue;
            }
            for (assoc, bound) in &ib.assoc_bindings {
                let bound = bound.substitute(&m);
                if let Some((_, other)) = out.iter().find(|(a, _)| a == assoc) {
                    if *other != bound {
                        return Err(format!(
                            "`{ty}` has two associated types named `{assoc}`, {other} and {bound}, \
                             from two traits"
                        ));
                    }
                    continue;
                }
                out.push((assoc.clone(), bound));
            }
        }
        Ok(out)
    }
}
