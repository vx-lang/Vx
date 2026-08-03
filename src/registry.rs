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
use rustc_hash::{FxHashMap, FxHashSet};

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

/// The signature a call site needs: the callee's GID (its stable identity), its parameter types, its
/// return type, and its return-provenance code. Lets the flat-HIR lowerer resolve a call `f(..)` --
/// identify the callee and type the result -- from the frozen registry it already holds, without a
/// name->AST walk (#198); and lets the borrow checker reborrow-track and provenance-refine an
/// *imported* call whose AST is absent (#265 step 7).
#[derive(Debug, Clone, PartialEq)]
pub struct FnSig {
    pub gid: TypeId,
    /// Parameter types in declaration order. The borrow checker needs each parameter's reference-ness
    /// to decide reborrow conflicts for a call resolved from the registry (no AST). Enriched for the
    /// cross-module borrow check; also the enrichment #219's imported-call type-check flip needs.
    pub params: Vec<crate::syntax::Type>,
    pub ret_ty: crate::syntax::Type,
    /// The return-provenance code (`hir::provenance::encode_return_provenance`): which parameter an
    /// imported reference return derives from — `0` none, `1..=4` parameter slot 0..3, `7` conservative
    /// top. Read at a cross-module call site in place of the AST-only `return_provenances` side table
    /// (empty for imports). A conservative refinement by construction, so it is never unsound (#265).
    pub ret_prov: u8,
}

/// The generic field-type information the flat path needs to resolve a member access *through a
/// (monomorphized) generic aggregate* — e.g. `self.data` on a `Vec<i32>` yields `*mut i32` by
/// substituting the instance's type arguments into the base struct's declared field types. The
/// frozen `layouts` deliberately erase a pointer field's pointee (every pointer is `FieldTy::Opaque`,
/// pointer-sized), so this carries the AST field `Type`s the substitution needs — the flat-path
/// analogue of the AST codegen's `gen.structs`. Keyed by the *base* struct's GID (`Vec<i32>`'s entry
/// sits under `Vec`'s TypeId), so two modules' same-named structs never collide (#291). (#242)
#[derive(Debug, Clone, PartialEq)]
pub struct StructFields {
    /// Generic parameter names in declaration order (`[T]` for `Vec<T>`) — the substitution keys.
    pub generics: Vec<crate::symbol::Symbol>,
    /// Each field's name and *declared* (possibly generic) AST type, in declaration order.
    pub fields: Vec<(crate::symbol::Symbol, crate::syntax::Type)>,
}

/// A data-carrying enum's declaration for the flat path: its generic parameter names and each
/// variant's name + declared payload types (empty for a payload-free variant). The flat path
/// substitutes a monomorphized instance's type arguments into the payload types to compute the
/// concrete `{ i32 tag, payload… }` layout (`Option<i32>` -> `{ i32, i32 }`). Variant order is the
/// declaration order, so a variant's index is its discriminant ordinal. (#242)
#[derive(Debug, Clone)]
pub struct EnumData {
    pub generics: Vec<crate::symbol::Symbol>,
    pub variants: Vec<(crate::symbol::Symbol, Vec<crate::syntax::Type>)>,
}

/// A function's precompiled flat-HIR body, keyed in the registry by the function's GID. Self-contained
/// so the flat codegen needs one lookup, not a join: it carries the signature (`emit_function_mlir`
/// reads `params` + `ret_ty` to emit the MLIR header) alongside the instruction + type streams. Only
/// non-generic bodies -- whose `types` are already global content-hash GIDs -- are portable across a
/// compile boundary; see `docs/discussions/implementation_plans/vxlib_bodies_and_loader.md` (#220).
#[derive(Debug, Clone)]
pub struct FnBody {
    /// The MLIR symbol / mangled emit name of the function.
    pub name: crate::symbol::Symbol,
    pub params: Vec<crate::syntax::Type>,
    pub ret_ty: crate::syntax::Type,
    pub hir: Vec<crate::hir::bytecode::HirInstruction>,
    /// The body's type stream (global GIDs), indexed by each instruction's `type_idx`.
    pub types: Vec<TypeId>,
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
    /// Precompiled flat-HIR bodies keyed by function GID -- the `body_of` backing. **Empty in a
    /// from-scratch compile** (the live pipeline keeps bodies in the per-worker streams); populated
    /// only when a registry is *deserialized from a `.vxlib` artifact*, so a downstream compile can
    /// link an imported module's bodies without its AST (#220). See
    /// `docs/discussions/implementation_plans/vxlib_bodies_and_loader.md`.
    pub bodies: FxHashMap<TypeId, FnBody>,
    /// Payload-free (C-like) enums, keyed by name, mapping to their variant names *in declaration
    /// order* — so the flat lowerer resolves a variant to its discriminant ordinal (`Color::Green` ->
    /// `1`) without walking the AST. Only enums whose every variant is payload-free are listed; a
    /// data-carrying (tagged-union) enum is omitted, so constructing/matching one declines to the AST
    /// path (#227). Populated by `build_frozen_registry`; empty when deserialized from a `.vxlib`.
    pub enum_variants: FxHashMap<crate::symbol::Symbol, Vec<crate::symbol::Symbol>>,
    /// Base struct declarations keyed by name, carrying each field's *declared* (possibly generic)
    /// AST type — the field `Type`s the frozen `layouts` erase (a pointer field becomes `Opaque`).
    /// The flat path substitutes a monomorphized instance's type arguments into these to recover a
    /// pointer field's pointee (`self.data : *mut T` → `*mut i32`), mirroring the AST codegen's
    /// `gen.structs`. Keyed by the base struct's GID — not its name — so two artifacts' same-named
    /// structs occupy distinct entries and the merge stays order-independent (#291). Resolve a bare
    /// name through `struct_fields_of` / `resolve_unique_nominal`. Populated by
    /// `build_frozen_registry` and when deserialized from a `.vxlib` (#219/#242).
    pub structs: FxHashMap<TypeId, StructFields>,
    /// Data-carrying (tagged-union) enum declarations keyed by name, carrying each variant's *declared*
    /// (possibly generic) payload types + the enum's generic parameters. A monomorphized instance
    /// (`Option<i32>`) has an instance-dependent layout (`{ i32 tag, i32 payload }`) that the frozen
    /// `layouts` can't hold (the base is generic), so the flat path substitutes the instance's type
    /// arguments into these to synthesize the concrete `{ tag, payload }` aggregate — the tagged-union
    /// analogue of `structs`. Populated by `build_frozen_registry`; empty when deserialized. (#242)
    pub enum_data: FxHashMap<crate::symbol::Symbol, EnumData>,
    /// Book-keeping for `merge_from`'s name-keyed tables: which keys came from an imported
    /// interface (own entries always win over imports) and which are poisoned by an
    /// import-vs-import conflict (the name then resolves in *neither* artifact, whatever order
    /// they merged in). Merge-session state only — never serialized (#291).
    pub merge_state: MergeState,
}

/// See [`ImmutableGlobalRegistry::merge_state`]. The `poisoned_*` tombstones are read back through
/// [`ImmutableGlobalRegistry::poisoned_reason`], so a diagnostic can distinguish "never defined"
/// from "defined twice and deliberately tombstoned" (#294).
#[derive(Debug, Default)]
pub struct MergeState {
    pub imported_fns: FxHashSet<crate::symbol::Symbol>,
    pub poisoned_fns: FxHashSet<crate::symbol::Symbol>,
    pub imported_methods: FxHashSet<(TypeId, crate::symbol::Symbol)>,
    pub poisoned_methods: FxHashSet<(TypeId, crate::symbol::Symbol)>,
    pub imported_enums: FxHashSet<crate::symbol::Symbol>,
    pub poisoned_enums: FxHashSet<crate::symbol::Symbol>,
}

/// Which table a poisoned name was tombstoned in — see [`ImmutableGlobalRegistry::poisoned_reason`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoisonKind {
    Function,
    Method,
    EnumVariant,
}

/// The one wording for "this bare name is defined in more than one imported artifact", shared by
/// the nominal (struct) decline path and the poisoned-tombstone paths so the two messages cannot
/// drift (#294). `kind` is the declaration kind as it should appear in the message ("Struct",
/// "Function", ...).
pub fn ambiguous_import_message(kind: &str, name: &dyn std::fmt::Display) -> String {
    format!(
        "{kind} '{name}' is defined in more than one imported module; the bare name cannot \
         resolve to a unique definition"
    )
}

/// Merge one *imported* name-keyed table into `dst`. An entry this compile built itself (present
/// in `dst` but not in `imported`) always wins. Among imports, a key bound to two *different*
/// values is ambiguous: the entry is removed and tombstoned in `poisoned`, so the merged result
/// is the same whatever order the artifacts arrive in — never silent first-wins (#291).
fn merge_imported_table<K, V>(
    dst: &mut FxHashMap<K, V>,
    imported: &mut FxHashSet<K>,
    poisoned: &mut FxHashSet<K>,
    src: FxHashMap<K, V>,
) where
    K: std::hash::Hash + Eq + Clone,
    V: PartialEq,
{
    for (k, v) in src {
        if poisoned.contains(&k) {
            continue;
        }
        match dst.get(&k) {
            Some(_) if !imported.contains(&k) => {} // this compile's own entry wins
            Some(existing) if *existing != v => {
                dst.remove(&k);
                imported.remove(&k);
                poisoned.insert(k);
            }
            Some(_) => {} // identical re-listing is a no-op
            None => {
                imported.insert(k.clone());
                dst.insert(k, v);
            }
        }
    }
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
            bodies: FxHashMap::default(),
            enum_variants: FxHashMap::default(),
            structs: FxHashMap::default(),
            enum_data: FxHashMap::default(),
            merge_state: MergeState::default(),
        })
    }

    /// Resolve a method `recv.method(..)` to its signature by `(receiver GID, method name)` -- the
    /// registry-backed method lookup that replaces the `GlobalAstEnv` `impl`-block walk (#218).
    pub fn resolve_method(&self, recv: TypeId, method: &crate::symbol::Symbol) -> Option<&FnSig> {
        self.methods.get(&(recv, method.clone()))
    }

    /// Resolve a *bare* nominal type name (no module qualifier) to its GID, iff it is unambiguous --
    /// defined in exactly one module, or in several that agree on the GID. A name two modules define
    /// with *distinct* GIDs returns `None`: the caller can't tell which is meant, so it is safer to
    /// decline than to attach the wrong identity. This is the registry-backed replacement for
    /// `GlobalAstEnv::struct_gids` (the same bare-name -> GID index the borrowed-AST env kept for
    /// `StructInit` annotation, with the same ambiguity policy) -- #219.
    pub fn resolve_unique_nominal(&self, name: &crate::symbol::Symbol) -> Option<TypeId> {
        let mut found: Option<TypeId> = None;
        for by_name in self.module_indices.values() {
            if let Some(&gid) = by_name.get(name) {
                match found {
                    Some(existing) if existing != gid => return None,
                    _ => found = Some(gid),
                }
            }
        }
        found
    }

    /// The precompiled flat-HIR body of the function identified by `gid`, if the registry carries one
    /// (i.e. it was deserialized from an artifact and the body was portable). `None` in a from-scratch
    /// compile, where bodies live in the per-worker streams instead (#220).
    pub fn body_of(&self, gid: TypeId) -> Option<&FnBody> {
        self.bodies.get(&gid)
    }

    /// Fold a precompiled module interface (deserialized from a `.vxlib`) into this registry, so a
    /// downstream compile resolves the imported module's types / functions / bodies without its AST
    /// (#220). `self`'s own entries win on any key collision -- the imported interface *fills in* the
    /// symbols the current compilation didn't build.
    ///
    /// The GID-keyed tables (`layouts`, `bodies`, `structs`, `module_indices`' inner maps) are
    /// content-addressed: an identical entry re-listed is a harmless no-op, and a genuine GID
    /// conflict would already have been caught by the freeze's collision guard. That argument does
    /// NOT extend to the *name*-keyed tables (`fn_sigs`, `methods`, `enum_variants`): two artifacts
    /// can bind the same name to different definitions, and no key discipline prevents it. Such an
    /// import-vs-import conflict is *poisoned* -- the name then resolves in neither artifact,
    /// mirroring `build_frozen_registry`'s intra-build ambiguity policy -- so the merged registry
    /// is identical whatever order the artifacts arrive in, never silent first-wins (#291).
    pub fn merge_from(&mut self, other: ImmutableGlobalRegistry) {
        for (id, def) in other.layouts {
            self.layouts.entry(id).or_insert(def);
        }
        for (module_hash, by_name) in other.module_indices {
            let dst = self.module_indices.entry(module_hash).or_default();
            for (name, gid) in by_name {
                dst.entry(name).or_insert(gid);
            }
        }
        merge_imported_table(
            &mut self.fn_sigs,
            &mut self.merge_state.imported_fns,
            &mut self.merge_state.poisoned_fns,
            other.fn_sigs,
        );
        merge_imported_table(
            &mut self.methods,
            &mut self.merge_state.imported_methods,
            &mut self.merge_state.poisoned_methods,
            other.methods,
        );
        for (gid, body) in other.bodies {
            self.bodies.entry(gid).or_insert(body);
        }
        merge_imported_table(
            &mut self.enum_variants,
            &mut self.merge_state.imported_enums,
            &mut self.merge_state.poisoned_enums,
            other.enum_variants,
        );
        // Imported structs' declared field types, GID-keyed (#291), so member access on an
        // imported struct resolves from the registry with no AST (#219).
        for (gid, fields) in other.structs {
            self.structs.entry(gid).or_insert(fields);
        }
    }

    /// Whether `name` was tombstoned by `merge_from`'s conflict policy — defined in more than one
    /// imported artifact and deliberately removed — and from which table. Diagnostic paths consult
    /// this *before* reporting "undefined", so a poisoned name reports the ambiguity instead of
    /// sending the reader hunting for a missing import when the cause is a duplicate one (#294).
    pub fn poisoned_reason(&self, name: &crate::symbol::Symbol) -> Option<PoisonKind> {
        if self.merge_state.poisoned_fns.contains(name) {
            return Some(PoisonKind::Function);
        }
        if self
            .merge_state
            .poisoned_methods
            .iter()
            .any(|(_, m)| m == name)
        {
            return Some(PoisonKind::Method);
        }
        if self.merge_state.poisoned_enums.contains(name) {
            return Some(PoisonKind::EnumVariant);
        }
        None
    }

    /// Whether `name` is defined in more than one module with *distinct* GIDs — exactly the case
    /// `resolve_unique_nominal` declines. Lets a diagnostic report "ambiguous across imported
    /// modules" instead of a misleading "unknown struct" when two artifacts both define the name
    /// (#291).
    pub fn is_ambiguous_nominal(&self, name: &crate::symbol::Symbol) -> bool {
        let mut found: Option<TypeId> = None;
        for by_name in self.module_indices.values() {
            if let Some(&gid) = by_name.get(name) {
                match found {
                    Some(existing) if existing != gid => return true,
                    _ => found = Some(gid),
                }
            }
        }
        false
    }

    /// The declared field types of the (base) struct a `Type` names: through the GID the type
    /// carries when resolution attached one (`Struct(_, Some(gid))`, possibly under a
    /// `GenericInstance`), else through the *unambiguous* bare name (`resolve_unique_nominal`).
    /// `None` when the type is not a nominal struct form, the bare name is defined in several
    /// modules with distinct GIDs, or the registry has no entry -- the caller declines rather than
    /// guessing between same-named structs (#291).
    pub fn struct_fields_of(&self, ty: &crate::syntax::Type) -> Option<&StructFields> {
        use crate::syntax::Type;
        let (name, attached) = match ty {
            Type::Struct(n, id) => (n, *id),
            Type::GenericInstance(base, _) => match &**base {
                Type::Struct(n, id) => (n, *id),
                _ => return None,
            },
            _ => return None,
        };
        let gid = attached.or_else(|| self.resolve_unique_nominal(name))?;
        self.structs.get(&gid)
    }
}

/// The query surface the frontend consults for anything defined *outside the current module* --
/// backed entirely by the frozen registry, never the AST. This is the "protocol" of the
/// stdlib<->compiler decoupling (`docs/discussions/implementation_plans/stdlib_decoupling_protocol.md`
/// §4): resolution is identical whether the target module was just compiled (in-memory registry) or
/// loaded from a cached artifact (a deserialized registry). Pointing the type checker's
/// imported-symbol resolution at this interface -- instead of `GlobalAstEnv`'s borrowed AST -- is what
/// lets the stdlib grow without expanding the AST / type-checker surface (#219).
///
/// The full protocol also has `resolve_trait_impl` (trait selection); that needs a `trait_impls` table
/// a later step adds (#221), so it is intentionally omitted here until its backing exists rather than
/// stubbed to always-`None`. `body_of` (a function's flat HIR body, for cross-module linking /
/// monomorphization) is backed by the `bodies` table populated on artifact deserialization (#220).
pub trait ModuleInterface {
    /// Resolve `name` *defined in* the module whose hash is `module_hash` to its GID (`module_indices`).
    fn resolve_type(&self, module_hash: u64, name: &crate::symbol::Symbol) -> Option<TypeId>;
    /// The structural layout of a nominal type by GID (`layouts`).
    fn layout_of(&self, ty: TypeId) -> Option<&TypeDefinition>;
    /// Resolve a free function by name to its signature (`fn_sigs`).
    fn resolve_fn(&self, name: &crate::symbol::Symbol) -> Option<&FnSig>;
    /// Resolve a method `recv.method(..)` by `(receiver GID, method name)` (`methods`, #218).
    fn resolve_method(&self, recv: TypeId, method: &crate::symbol::Symbol) -> Option<&FnSig>;
    /// Resolve a *bare* nominal name to its unique GID (`module_indices`), `None` if ambiguous (#219).
    fn resolve_unique_nominal(&self, name: &crate::symbol::Symbol) -> Option<TypeId>;
    /// The precompiled flat-HIR body of the function identified by `gid` (`bodies`), `None` when this
    /// registry carries no body for it (a from-scratch compile, or a non-portable body) -- #220.
    fn body_of(&self, gid: TypeId) -> Option<&FnBody>;
}

impl ModuleInterface for ImmutableGlobalRegistry {
    fn resolve_type(&self, module_hash: u64, name: &crate::symbol::Symbol) -> Option<TypeId> {
        self.resolve_in_module(module_hash, name)
    }
    fn layout_of(&self, ty: TypeId) -> Option<&TypeDefinition> {
        self.layouts.get(&ty)
    }
    fn resolve_fn(&self, name: &crate::symbol::Symbol) -> Option<&FnSig> {
        self.fn_sigs.get(name)
    }
    fn resolve_method(&self, recv: TypeId, method: &crate::symbol::Symbol) -> Option<&FnSig> {
        self.methods.get(&(recv, method.clone()))
    }
    fn resolve_unique_nominal(&self, name: &crate::symbol::Symbol) -> Option<TypeId> {
        ImmutableGlobalRegistry::resolve_unique_nominal(self, name)
    }
    fn body_of(&self, gid: TypeId) -> Option<&FnBody> {
        self.bodies.get(&gid)
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

    fn sig(w0: u64, w1: u64) -> FnSig {
        FnSig {
            gid: TypeId::new(w0, w1, 0, 0),
            params: Vec::new(),
            ret_ty: crate::syntax::Type::Scalar(crate::syntax::types::ElementType::I32),
            ret_prov: 0,
        }
    }

    fn point_fields(field_names: &[&str]) -> StructFields {
        StructFields {
            generics: Vec::new(),
            fields: field_names
                .iter()
                .map(|f| {
                    (
                        crate::symbol::Symbol::from(*f),
                        crate::syntax::Type::Scalar(crate::syntax::types::ElementType::I32),
                    )
                })
                .collect(),
        }
    }

    /// One artifact's registry for the merge tests: module `module` defines a struct `Point`
    /// (module-distinct GID + field list) and the free functions in `fns`.
    fn interface(
        module: u64,
        point_fields_list: &[&str],
        fns: &[(&str, u64)],
    ) -> ImmutableGlobalRegistry {
        let def = make_def("Point", module, 100, vec![]);
        let gid = def.id;
        let mut reg = ImmutableGlobalRegistry::build_and_validate(vec![def]).unwrap();
        reg.structs.insert(gid, point_fields(point_fields_list));
        for (name, w1) in fns {
            reg.fn_sigs
                .insert(crate::symbol::Symbol::from(*name), sig(module, *w1));
        }
        reg
    }

    /// The #291 acceptance property: merging the same set of artifacts must produce the same
    /// registry whatever order they arrive in, and an import-vs-import name conflict must be
    /// poisoned (resolves in neither artifact), never silent first-wins.
    #[test]
    fn merge_is_order_independent_and_poisons_name_conflicts() {
        let build = |order_ab: bool| {
            let a = interface(1, &["x"], &[("helper", 10), ("only_a", 11)]);
            let b = interface(2, &["x", "y"], &[("helper", 10), ("only_b", 12)]);
            let mut reg = ImmutableGlobalRegistry::build_and_validate(vec![]).unwrap();
            if order_ab {
                reg.merge_from(a);
                reg.merge_from(b);
            } else {
                reg.merge_from(b);
                reg.merge_from(a);
            }
            reg
        };
        for order_ab in [true, false] {
            let reg = build(order_ab);
            // Both same-named structs coexist under their own GIDs.
            assert_eq!(reg.structs.len(), 2, "order_ab={order_ab}");
            let gid_a = TypeId::new(1, 100, 0, 0);
            let gid_b = TypeId::new(2, 100, 0, 0);
            assert_eq!(reg.structs.get(&gid_a).unwrap().fields.len(), 1);
            assert_eq!(reg.structs.get(&gid_b).unwrap().fields.len(), 2);
            // A GID-annotated type resolves to *its* module's fields; the ambiguous bare name
            // declines rather than picking one.
            let ty_b =
                crate::syntax::Type::Struct(crate::symbol::Symbol::from("Point"), Some(gid_b));
            assert_eq!(reg.struct_fields_of(&ty_b).unwrap().fields.len(), 2);
            let ty_bare = crate::syntax::Type::Struct(crate::symbol::Symbol::from("Point"), None);
            assert!(
                reg.struct_fields_of(&ty_bare).is_none(),
                "ambiguous bare name declines"
            );
            // `helper` is bound to different GIDs by the two artifacts: poisoned in both orders.
            assert!(
                !reg.fn_sigs
                    .contains_key(&crate::symbol::Symbol::from("helper")),
                "conflicting fn name is poisoned, order_ab={order_ab}"
            );
            assert!(reg
                .fn_sigs
                .contains_key(&crate::symbol::Symbol::from("only_a")));
            assert!(reg
                .fn_sigs
                .contains_key(&crate::symbol::Symbol::from("only_b")));
            // The tombstone is queryable, so a diagnostic can say "defined twice" rather than
            // "undefined" (#294); a cleanly-resolved name is not poisoned.
            assert_eq!(
                reg.poisoned_reason(&crate::symbol::Symbol::from("helper")),
                Some(PoisonKind::Function)
            );
            assert_eq!(
                reg.poisoned_reason(&crate::symbol::Symbol::from("only_a")),
                None
            );
        }
    }

    /// This compile's own `fn_sigs` entries always win over an imported same-name signature —
    /// locals shadow imports — and are never poisoned by one (#291).
    #[test]
    fn merge_keeps_own_entries_over_imported_conflicts() {
        let mut own = ImmutableGlobalRegistry::build_and_validate(vec![]).unwrap();
        own.fn_sigs
            .insert(crate::symbol::Symbol::from("helper"), sig(9, 10));
        own.merge_from(interface(1, &["x"], &[("helper", 10)]));
        own.merge_from(interface(2, &["x", "y"], &[("helper", 10)]));
        let kept = own
            .fn_sigs
            .get(&crate::symbol::Symbol::from("helper"))
            .expect("own entry survives both imports");
        assert_eq!(kept.gid, TypeId::new(9, 10, 0, 0));
    }

    /// Re-merging the same artifact (identical entries) is a no-op, not a conflict.
    #[test]
    fn merge_tolerates_identical_relisting() {
        let mut reg = ImmutableGlobalRegistry::build_and_validate(vec![]).unwrap();
        reg.merge_from(interface(1, &["x"], &[("helper", 10)]));
        reg.merge_from(interface(1, &["x"], &[("helper", 10)]));
        assert!(reg
            .fn_sigs
            .contains_key(&crate::symbol::Symbol::from("helper")));
        assert_eq!(reg.structs.len(), 1);
    }
}
