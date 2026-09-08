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
