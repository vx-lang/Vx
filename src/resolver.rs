//===- resolver.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file implements the Name Resolution pass for the Vx compiler.
// It maps local and global identifiers to their corresponding definitions in the AST,
// handling scope shadowing, namespace resolution, and ensuring that all referenced
// variables and functions actually exist.
//
//===----------------------------------------------------------------------===//
use crate::ast::VxModule;
use crate::gid::TypeId;
use crate::hash::{compute_module_hash, DefPath};
use std::collections::HashMap;

/// A global read-only map generated during Phase 1.25.
/// Maps module_path -> (symbol_name -> TypeId)
pub type SymbolTable = HashMap<crate::symbol::Symbol, TypeId>;
pub type SymbolMap = HashMap<crate::symbol::Symbol, SymbolTable>;

/// Iterates through all parsed modules sequentially and computes their deterministic
/// 256-bit TypeId for every top-level struct, enum, and trait.
pub fn build_symbol_map(modules: &[VxModule]) -> SymbolMap {
    let mut map: SymbolMap = HashMap::new();

    for module in modules {
        let mut module_symbols = HashMap::new();
        let module_hash = compute_module_hash(&module.module_path);

        let mut process_decl = |name: &crate::symbol::Symbol| {
            // DefPath::Named can now take a borrowed &str.
            let sym_hash = DefPath::Named(name.as_ref()).compute_symbol_hash();
            let tid = TypeId::new(module_hash, sym_hash, 0, 0);
            module_symbols.insert(name.clone(), tid);
        };

        module.structs.iter().for_each(|d| process_decl(&d.name));
        module.enums.iter().for_each(|d| process_decl(&d.name));
        module.traits.iter().for_each(|d| process_decl(&d.name));

        map.insert(module.module_path.clone(), module_symbols);
    }

    map
}
