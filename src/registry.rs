//===- registry.rs - Vx Compiler -------------------------------*- Rust -*-===//
//
// Part of the Vx Project, under the BSD 3-Clause License.
// See LICENSE for license information.
// SPDX-License-Identifier: BSD-3-Clause
//
//===----------------------------------------------------------------------===//
//
// This file manages the central Module Registry for the Vx compiler.
// It handles the storage and retrieval of loaded modules, tracks dependencies to
// prevent circular imports, and maintains a unified namespace for functions and
// types across the entire project.
//
//===----------------------------------------------------------------------===//
use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use rustc_hash::FxHashMap;

use crate::gid::TypeId;

/// Structural layout definition of a nominal type.
#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub id: TypeId,
    pub name: String,
    pub size_bytes: usize,
    pub align_bytes: usize,
    /// Per-field byte offsets/sizes in declaration order (structs only; empty for
    /// enums or when the layout is not yet computable). See `crate::layout` (#199).
    pub fields: Vec<crate::layout::FieldLayout>,
    /// Dependencies represent the types embedded directly (by-value) within this type.
    /// Used to detect infinite-sized recursive structs.
    pub by_value_dependencies: Vec<TypeId>,
}

/// The signature a call site needs: the callee's GID (its stable identity) and its return type.
/// Lets the flat-HIR lowerer resolve a call `f(..)` -- identify the callee and type the result --
/// from the frozen registry it already holds, without a name->AST walk (#198).
#[derive(Debug, Clone)]
pub struct FnSig {
    pub gid: TypeId,
    pub ret_ty: crate::syntax::Type,
}

/// The globally frozen type registry for parallel compilation phases.
#[derive(Debug)]
pub struct ImmutableGlobalRegistry {
    pub layouts: FxHashMap<TypeId, TypeDefinition>,
    pub module_indices: FxHashMap<u64, FxHashMap<crate::symbol::Symbol, TypeId>>,
    /// Function signatures by name, for call resolution in the flat HIR. A name defined in more than
    /// one module (distinct GIDs) is omitted -- the name-keyed map can't disambiguate, so such a
    /// call declines rather than resolving to the wrong callee (mirrors the struct-GID policy).
    pub fn_sigs: FxHashMap<crate::symbol::Symbol, FnSig>,
    /// Method signatures keyed by `(receiver type GID, method name)`, minted from `impl` blocks. This
    /// is the GID-keyed replacement for walking borrowed AST `ImplBlock`s in `GlobalAstEnv`: method
    /// resolution (`x.exp()`) becomes a table lookup `(type-of-x GID, "exp") -> FnSig`. See
    /// `docs/discussions/implementation_plans/stdlib_decoupling_protocol.md` (#218).
    pub methods: FxHashMap<(TypeId, crate::symbol::Symbol), FnSig>,
}

impl ImmutableGlobalRegistry {
    /// Cross-module symbol lookup: resolve `symbol` *defined in* the module whose hash is
    /// `module_hash` (GID word 0) to its `TypeId`. This is the `module_indices` read path the
    /// design describes (doc §7, Step 3) — the frozen-registry counterpart of the `symbol_map`
    /// lookup `resolve_names` uses on the AST (#194). Returns `None` if the module or symbol is
    /// absent.
    pub fn resolve_in_module(
        &self,
        module_hash: u64,
        symbol: &crate::symbol::Symbol,
    ) -> Option<TypeId> {
        self.module_indices.get(&module_hash)?.get(symbol).copied()
    }

    /// Builds and validates the registry from a collection of local module thread maps.
    /// Runs a fast cycle-detection pass to ensure no infinite-sized recursive layouts exist.
    pub fn build_and_validate(definitions: Vec<TypeDefinition>) -> Result<Self, String> {
        let mut layouts: FxHashMap<TypeId, TypeDefinition> = FxHashMap::default();
        let mut module_indices: FxHashMap<u64, FxHashMap<crate::symbol::Symbol, TypeId>> =
            FxHashMap::default();

        let mut graph = DiGraph::<TypeId, ()>::new();
        let mut node_map = FxHashMap::default();

        // 1. Register all layouts and build the node map
        for def in definitions {
            let mod_id = def.id.module_id();
            // Deterministic collision guard (#195): the content hash (FNV-1a) is stable but not
            // cryptographic, so two *distinct* symbols could in principle land on the same GID.
            // Catch it here rather than silently conflate two types — an improbable but real
            // correctness bug. (Re-listing the same type is harmless: same id *and* same name.)
            if let Some(existing) = layouts.get(&def.id) {
                if existing.name != def.name {
                    return Err(format!(
                        "GID collision: distinct types '{}' and '{}' share the same 256-bit identity {:?}",
                        existing.name, def.name, def.id
                    ));
                }
            }
            module_indices
                .entry(mod_id)
                .or_default()
                .insert(def.name.clone().into(), def.id);

            let node_idx = graph.add_node(def.id);
            node_map.insert(def.id, node_idx);
            layouts.insert(def.id, def);
        }

        // 2. Add dependency edges
        for def in layouts.values() {
            let source_idx = *node_map.get(&def.id).unwrap();
            for dep in &def.by_value_dependencies {
                if let Some(target_idx) = node_map.get(dep) {
                    graph.add_edge(source_idx, *target_idx, ());
                } else {
                    return Err(format!("Unresolved by-value dependency: {:?}", dep));
                }
            }
        }

        // 3. Cycle Detection
        // `toposort` returns an error if a cycle exists in a directed graph.
        if let Err(cycle_err) = toposort(&graph, None) {
            let cyclic_type_id = graph[cycle_err.node_id()];
            if let Some(cyclic_def) = layouts.get(&cyclic_type_id) {
                return Err(format!(
                    "Infinite-sized recursive layout detected in struct '{}'. Recursive fields must be wrapped in a Box or Pointer.",
                    cyclic_def.name
                ));
            }
            return Err("Infinite-sized recursive layout detected.".to_string());
        }

        Ok(Self {
            layouts,
            module_indices,
            fn_sigs: FxHashMap::default(),
            methods: FxHashMap::default(),
        })
    }

    /// Resolve a method `recv.method(..)` to its signature by `(receiver GID, method name)` -- the
    /// registry-backed method lookup that replaces the `GlobalAstEnv` `impl`-block walk (#218).
    pub fn resolve_method(&self, recv: TypeId, method: &crate::symbol::Symbol) -> Option<&FnSig> {
        self.methods.get(&(recv, method.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gid::TypeId;

    fn make_def(name: &str, w0: u64, w1: u64, deps: Vec<TypeId>) -> TypeDefinition {
        TypeDefinition {
            id: TypeId::new(w0, w1, 0, 0),
            name: name.to_string(),
            size_bytes: 8,
            align_bytes: 8,
            fields: Vec::new(),
            by_value_dependencies: deps,
        }
    }

    #[test]
    fn test_registry_build_empty() {
        let registry = ImmutableGlobalRegistry::build_and_validate(vec![]);
        assert!(registry.is_ok());
        let reg = registry.unwrap();
        assert!(reg.layouts.is_empty());
    }

    #[test]
    fn test_registry_build_single_def() {
        let def = make_def("MyStruct", 1, 100, vec![]);
        let registry = ImmutableGlobalRegistry::build_and_validate(vec![def]);
        assert!(registry.is_ok());
        let reg = registry.unwrap();
        assert_eq!(reg.layouts.len(), 1);
        assert!(reg.layouts.contains_key(&TypeId::new(1, 100, 0, 0)));
    }

    #[test]
    fn test_registry_build_valid_chain() {
        // A depends on B (by value), no cycle
        let b = make_def("Inner", 1, 200, vec![]);
        let a = make_def("Outer", 1, 100, vec![TypeId::new(1, 200, 0, 0)]);
        let registry = ImmutableGlobalRegistry::build_and_validate(vec![a, b]);
        assert!(registry.is_ok());
        let reg = registry.unwrap();
        assert_eq!(reg.layouts.len(), 2);
    }

    #[test]
    fn test_registry_detect_cycle() {
        // A depends on B, B depends on A → cycle!
        let a = make_def("TypeA", 1, 100, vec![TypeId::new(1, 200, 0, 0)]);
        let b = make_def("TypeB", 1, 200, vec![TypeId::new(1, 100, 0, 0)]);
        let result = ImmutableGlobalRegistry::build_and_validate(vec![a, b]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Infinite-sized recursive layout"),
            "Got: {}",
            err
        );
    }

    #[test]
    fn test_registry_self_referential_cycle() {
        // A depends on itself
        let a = make_def("SelfRef", 1, 100, vec![TypeId::new(1, 100, 0, 0)]);
        let result = ImmutableGlobalRegistry::build_and_validate(vec![a]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Infinite-sized recursive layout detected in struct 'SelfRef'"),
            "Got: {}",
            err
        );
    }

    #[test]
    fn test_registry_unresolved_dependency() {
        // A depends on a TypeId that doesn't exist in the registry
        let a = make_def("Dangling", 1, 100, vec![TypeId::new(99, 999, 0, 0)]);
        let result = ImmutableGlobalRegistry::build_and_validate(vec![a]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("Unresolved by-value dependency"),
            "Got: {}",
            err
        );
    }

    #[test]
    fn test_registry_module_indices() {
        let def1 = make_def("Foo", 1, 100, vec![]);
        let def2 = make_def("Bar", 1, 200, vec![]);
        let def3 = make_def("Baz", 2, 300, vec![]);
        let reg = ImmutableGlobalRegistry::build_and_validate(vec![def1, def2, def3]).unwrap();

        // Module 1 should have Foo and Bar
        let mod1 = reg.module_indices.get(&1).unwrap();
        assert_eq!(mod1.len(), 2);
        assert!(mod1.contains_key("Foo"));
        assert!(mod1.contains_key("Bar"));

        // Module 2 should have Baz
        let mod2 = reg.module_indices.get(&2).unwrap();
        assert_eq!(mod2.len(), 1);
        assert!(mod2.contains_key("Baz"));
    }

    #[test]
    fn build_rejects_gid_collision_between_distinct_types() {
        // Two *distinct* symbols that (hypothetically) hash to the same GID must be rejected, not
        // silently conflated (#195). We simulate a content-hash collision by handing two different
        // names the same (w0, w1).
        let a = make_def("Alpha", 7, 42, vec![]);
        let b = make_def("Beta", 7, 42, vec![]); // same module + symbol hash => same GID
        let err = ImmutableGlobalRegistry::build_and_validate(vec![a, b]).unwrap_err();
        assert!(err.contains("GID collision"), "{err}");
        assert!(err.contains("Alpha") && err.contains("Beta"), "{err}");
    }

    #[test]
    fn build_allows_same_type_listed_twice() {
        // Same id *and* same name is a harmless re-listing, not a collision.
        let a = make_def("Same", 7, 42, vec![]);
        let a2 = make_def("Same", 7, 42, vec![]);
        assert!(ImmutableGlobalRegistry::build_and_validate(vec![a, a2]).is_ok());
    }

    #[test]
    fn resolve_in_module_reads_module_indices_across_modules() {
        // The cross-module lookup path (#194): a symbol defined in module 1 is found by that
        // module's hash + name, and is *not* visible under a different module's hash even when the
        // name collides.
        let m1_foo = make_def("Foo", 1, 100, vec![]);
        let m2_foo = make_def("Foo", 2, 999, vec![]); // same name, different module
        let reg = ImmutableGlobalRegistry::build_and_validate(vec![m1_foo, m2_foo]).unwrap();

        assert_eq!(
            reg.resolve_in_module(1, &crate::symbol::Symbol::from("Foo")),
            Some(TypeId::new(1, 100, 0, 0)),
            "resolves the module-1 Foo by its defining module hash"
        );
        assert_eq!(
            reg.resolve_in_module(2, &crate::symbol::Symbol::from("Foo")),
            Some(TypeId::new(2, 999, 0, 0)),
            "module isolation: module 2's Foo is a distinct GID"
        );
        assert_eq!(
            reg.resolve_in_module(3, &crate::symbol::Symbol::from("Foo")),
            None,
            "absent module -> None"
        );
        assert_eq!(
            reg.resolve_in_module(1, &crate::symbol::Symbol::from("Missing")),
            None,
            "absent symbol -> None"
        );
    }
}
