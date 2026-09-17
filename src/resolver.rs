//===- resolver.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Name Resolution pass for the Vx compiler.
// It maps local and global identifiers to their corresponding definitions in the AST,
// handling scope shadowing, namespace resolution, and ensuring that all referenced
// variables and functions actually exist.
//
//===----------------------------------------------------------------------===//
use crate::gid::TypeId;
use crate::hash::{compute_module_hash, DefPath};
use crate::syntax::{SymbolMap, VxModule};
use std::collections::HashMap;

/// Iterates through all parsed modules sequentially and computes their deterministic
/// 256-bit TypeId for every top-level struct, enum, and trait.
pub fn build_symbol_map(modules: &[VxModule]) -> SymbolMap {
    use rayon::prelude::*;

    modules
        .par_iter()
        .map(|module| {
            let mut module_symbols = HashMap::new();
            let module_hash = compute_module_hash(&module.module_path);

            let mut process_decl = |name: &crate::symbol::Symbol| {
                let sym_hash = DefPath::Named(name.as_ref()).compute_symbol_hash();
                let tid = TypeId::new(module_hash, sym_hash, 0, 0);
                module_symbols.insert(name.clone(), tid);
            };

            module.structs.iter().for_each(|d| process_decl(&d.name));
            module.enums.iter().for_each(|d| process_decl(&d.name));
            module.traits.iter().for_each(|d| process_decl(&d.name));

            (module.module_path.clone(), module_symbols)
        })
        .collect()
}

/// Every topology the compilation declares, gathered across its modules.
///
/// Companion to `build_symbol_map` and used the same way: computed once from `&[VxModule]`, then
/// handed to each module's `resolve_names` during the mutable walk. Owned because that walk takes
/// the array by `&mut`, and because a topology declared in a `--machine` file has to reach the
/// module that names it, which a per-module view cannot do.
pub fn collect_topologies(modules: &[VxModule]) -> Vec<crate::arch::TopologyDecl> {
    modules
        .iter()
        .flat_map(|m| m.topologies.iter().cloned())
        .collect()
}

/// Replace `Self` in a trait method's signature with the type the impl is for. A trait
/// writes `fn eq(self : Self, other : Self) -> bool`; the copy handed to `impl Eq for i32`
/// has to say `i32`, because from here on it is an ordinary method and nothing downstream
/// knows what `Self` was.
fn substitute_self(ty: &crate::syntax::Type, target: &crate::syntax::Type) -> crate::syntax::Type {
    use crate::syntax::Type;
    let is_self = |name: &crate::symbol::Symbol| name.as_ref() == "Self";
    match ty {
        Type::Struct(name, _) | Type::Generic(name, _) if is_self(name) => target.clone(),
        Type::Ref(inner, space) => {
            Type::Ref(Box::new(substitute_self(inner, target)), space.clone())
        }
        Type::Borrow {
            inner,
            mem_space,
            is_mut,
            region_id,
        } => Type::Borrow {
            inner: Box::new(substitute_self(inner, target)),
            mem_space: mem_space.clone(),
            is_mut: *is_mut,
            region_id: *region_id,
        },
        Type::Pointer(inner, space, is_mut) => Type::Pointer(
            Box::new(substitute_self(inner, target)),
            space.clone(),
            *is_mut,
        ),
        Type::Verified(inner) => Type::Verified(Box::new(substitute_self(inner, target))),
        Type::Pinned(inner, topo) => {
            Type::Pinned(Box::new(substitute_self(inner, target)), topo.clone())
        }
        Type::GenericInstance(base, args) => Type::GenericInstance(
            Box::new(substitute_self(base, target)),
            args.iter().map(|a| substitute_self(a, target)).collect(),
        ),
        // Nothing else can hold a `Self`: a scalar, a tensor and a function type are built
        // from element types and shapes rather than from nominal names.
        other => other.clone(),
    }
}

/// Give every impl of a trait the trait's default method bodies, for the methods it does
/// not write itself.
///
/// This runs over the parsed modules before anything reads them, so from here on an impl
/// block holds every method it is supposed to have and nothing downstream -- name
/// resolution, the checker, either code generator -- needs to know that defaults exist.
/// That is also why it is a rewrite rather than a fallback at method-lookup time: the
/// lookup is not the only reader, and codegen walks the impl's methods directly.
///
/// A trait's defaults are collected across every module first, because the trait and the
/// impl need not be in the same one.
pub type TraitDefaults =
    std::collections::HashMap<crate::symbol::Symbol, Vec<crate::syntax::MethodSignature>>;

/// The trait methods that carry a default body, across every module.
///
/// Global on purpose: an impl and the trait it implements need not share a module, and an
/// impl a macro produces is as entitled to the defaults as one written by hand.
pub fn collect_trait_defaults<'p>(
    programs: impl IntoIterator<Item = &'p crate::syntax::Program>,
) -> TraitDefaults {
    let mut defaults = TraitDefaults::new();
    for program in programs {
        for decl in &program.traits {
            let with_bodies: Vec<crate::syntax::MethodSignature> = decl
                .methods
                .iter()
                .filter(|m| m.default_body.is_some())
                .cloned()
                .collect();
            if !with_bodies.is_empty() {
                defaults.insert(decl.name.clone(), with_bodies);
            }
        }
    }
    defaults
}

pub fn fill_trait_defaults_in(program: &mut crate::syntax::Program, defaults: &TraitDefaults) {
    if defaults.is_empty() {
        return;
    }
    for block in &mut program.impls {
        let Some(trait_name) = block.trait_name.clone() else {
            continue;
        };
        let Some(trait_methods) = defaults.get(&trait_name) else {
            continue;
        };
        for signature in trait_methods {
            if block
                .methods
                .iter()
                .any(|m| m.name.as_ref() == signature.name.as_ref())
            {
                continue;
            }
            let body = signature
                .default_body
                .clone()
                .expect("only methods with a default body are collected");
            block.methods.push(crate::syntax::Function {
                name: signature.name.clone(),
                generics: signature.generics.clone(),
                params: signature
                    .params
                    .iter()
                    .map(|(n, t)| (n.clone(), substitute_self(t, &block.target_type)))
                    .collect(),
                topology: crate::syntax::Topology::CPU,
                return_type: substitute_self(&signature.return_type, &block.target_type),
                requires: Vec::new(),
                ensures: Vec::new(),
                where_transfers: Vec::new(),
                is_unsafe: false,
                body,
                doc_comment: None,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::Symbol;

    fn parse_module(path: &str, src: &str) -> VxModule {
        let mut lexer = crate::lexer::Lexer::new(src);
        let tokens = lexer.tokenize();
        let mut parser = crate::parser::Parser::new(&tokens, src);
        let mut program = parser.parse().expect("parse failed");
        program.module_path = path.into();
        program
    }

    /// The GID is used as intended (docs/parallel_compiler_architecture.md §2): each top-level
    /// symbol's `TypeId` is minted from *content hashes* -- word 0 = module hash, word 1 = symbol
    /// hash -- not a monotonic counter. So the same symbol name in two modules gets distinct GIDs
    /// (module isolation), the same name hashes identically (word 1), and minting is deterministic
    /// across runs. This is the property that makes cross-module identity lock-free and stable.
    #[test]
    fn gids_are_module_isolated_symbol_hashed_and_deterministic() {
        let a = parse_module("crate::a", "struct Foo { x: i32 }");
        let b = parse_module("crate::b", "struct Foo { y: i32 }");
        let map = build_symbol_map(&[a, b]);

        let a_foo = map[&Symbol::from("crate::a")][&Symbol::from("Foo")];
        let b_foo = map[&Symbol::from("crate::b")][&Symbol::from("Foo")];

        // Word 0 (module hash) isolates same-named symbols across modules, and is exactly the
        // module-path content hash.
        assert_ne!(a_foo.module_id(), b_foo.module_id());
        assert_eq!(a_foo.module_id(), compute_module_hash("crate::a"));
        assert_eq!(b_foo.module_id(), compute_module_hash("crate::b"));

        // Word 1 (symbol hash) is the content hash of the name -- identical for both `Foo`s.
        assert_eq!(a_foo.symbol_id(), b_foo.symbol_id());
        assert_eq!(
            a_foo.symbol_id(),
            DefPath::Named("Foo").compute_symbol_hash()
        );

        // Deterministic: re-minting the same module yields the identical GID (content hashes, not
        // position-dependent counters -- the incremental-stability guarantee).
        let a2 = parse_module("crate::a", "struct Foo { x: i32 }");
        let map2 = build_symbol_map(&[a2]);
        assert_eq!(map2[&Symbol::from("crate::a")][&Symbol::from("Foo")], a_foo);
    }

    /// `build_symbol_map` mints GIDs in parallel (`par_iter`); doing so must be deterministic --
    /// the same inputs produce byte-identical GID streams regardless of thread scheduling. This is
    /// the parallel-architecture promise that GIDs "mean the same thing on every thread".
    #[test]
    fn parallel_gid_minting_is_deterministic_across_threads() {
        let modules: Vec<VxModule> = (0..32)
            .map(|i| {
                parse_module(
                    &format!("crate::m{i}"),
                    "struct S { v: i32 }\nenum E { A, B }",
                )
            })
            .collect();

        let run = || {
            let map = build_symbol_map(&modules);
            let mut flat: Vec<(u64, u64)> = map
                .values()
                .flat_map(|t| t.values().map(|id| (id.module_id(), id.symbol_id())))
                .collect();
            flat.sort_unstable();
            flat
        };

        // Two independent parallel builds agree exactly.
        assert_eq!(run(), run());
    }
}
