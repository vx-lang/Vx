# Implementation Plan: Decoupling the Stdlib from the AST — a Module-Interface Protocol

> The stdlib must not be a first-class citizen of the AST. As it grows, the AST + type checker +
> codegen cannot grow with it. This doc diagnoses the current coupling, surveys the state of the art
> (precompiled module interfaces), and defines a **protocol** — grounded in Vx's *existing* GID +
> frozen-registry + flat-HIR + metadata machinery — that lets the frontend resolve stdlib
> types/methods/functions *without ever loading stdlib source or AST*.
>
> Companion to [`stdlib_design.md`](./stdlib_design.md) (what goes in the stdlib) — this doc is *how the
> compiler consumes it*. Ties into the flat-pipeline convergence epic
> [#197](https://github.com/hiraditya/Vx/issues/197).

## 0. TL;DR

- **The coupling to kill:** `GlobalAstEnv` (`src/hir/env.rs`) holds **borrowed AST** for *every* module
  — `structs: &StructDef`, `traits: &TraitDef`, `impls: Vec<&ImplBlock>`, `functions`, `externs`. The
  type checker resolves names/types/methods/trait-impls by walking those AST nodes. So every
  `import std::x` parses the module, keeps its full AST live, and the checker's logic is bound to AST
  shapes. This is O(all imported stdlib source) per compile, forever.
- **The fix is not a rewrite** — it's finishing what the convergence work started. Vx already has:
  **GIDs** (stable content-addressed identities), the **frozen registry** (a GID-keyed interface for
  `layouts` + `fn_sigs` + `module_indices`), **flat HIR streams** (AST-free function bodies), and
  **`src/metadata.rs`** (zero-copy metadata serialization). The stdlib is *already* meant to flow
  through this.
- **The protocol** = a `ModuleInterface` query API (`resolve_type` / `resolve_fn` / **`resolve_method`**
  / **`resolve_trait_impl`** / `body_of`) **backed by the frozen registry, not the AST**. The type
  checker asks the registry; it never touches stdlib AST. Two registry additions are needed: a
  **method/impl table** and **serialization of the interface** (today's `metadata.rs` serializes only
  the GID dictionary; its `ast_data` slot is an empty placeholder).
- **The precompiled stdlib** = compile the stdlib *once* to `{registry + flat HIR streams}`, serialize
  it (zero-copy, GID-keyed), and have every downstream compile memory-map it. No re-parse, no
  re-typecheck, no AST. This is Rust's `.rmeta`, Swift's `.swiftmodule`, OCaml's `.cmi` — realized on
  Vx's own substrate. **It is the same artifact the flat codegen already consumes**, so decoupling and
  convergence are one project.

## 1. The problem, precisely

`GlobalAstEnv::build(&modules)` (`src/hir/env.rs`) iterates every module and stashes **references into
the AST**:

```rust
for i in &module.impls {           // ← every impl block of every imported module
    env.impls.entry(trait_name).or_default().push(i);   // Vec<&'a ImplBlock>
}
for t in &module.traits { env.traits.insert(t.name.clone(), t); }   // &'a TraitDef
```

Method resolution (`x.exp()` → `impl Math for f32` → mangled `f32$exp`) then *searches these borrowed
AST vectors*. Consequences as the stdlib scales:

1. **Memory / time**: the entire imported stdlib is parsed and its AST kept alive for the whole
   compile — every build, no caching.
1. **Type-checker blow-up**: the checker must understand *every AST construct the stdlib uses* (traits,
   generic impls, `unsafe`, `extern`, closures…). The stdlib's surface *is* the checker's surface.
1. **No separation of compilation**: nothing is precompiled; a one-line user program pays for all of
   `std::vec` + `std::hash_map` + `std::math` it transitively imports.

This is the header-file / whole-program-source model. It does not scale.

## 2. State of the art: precompiled module interfaces

Every serious compiler decouples *consumer* from *library source* with a **serialized interface
artifact**. The consumer resolves against metadata; implementation bodies (for generics/inlining) come
from serialized IR, never source.

| System | Interface artifact | What's in it | Bodies for generics/inline |
|---|---|---|---|
| **Rust** | `.rmeta` / `.rlib` | types, trait impls, fn sigs, spans | **MIR** (monomorphized at the use site) |
| **Swift** | `.swiftmodule` (+ stable `.swiftinterface`) | serialized decls/types | SIL for `@inlinable` |
| **OCaml** | `.cmi` (interface) / `.cmx` | typed signatures | cross-module inlining info in `.cmx` |
| **C++20** | BMI / `.pcm` (vs re-parsed `.h`) | the module's exported entities | template defs in the BMI |
| **Go** | package export data (`.a`) | exported decls, types | inlinable fn bodies |
| **.NET / Java** | assembly / `.class` metadata | full type metadata (reflectable) | bytecode/IL |

**The invariant:** *the frontend resolves names, types, and trait/impl selection against a
GID-/token-keyed metadata table — not a re-parsed source AST. The implementation travels as a
lowered IR keyed by the same identities.*

Vx's version of "GID-/token-keyed metadata table" is the **frozen registry**; its "lowered IR" is the
**flat HIR stream**. Both already exist. The gap is that method resolution and the interface *dump*
still go through the AST.

## 3. Vx already has the substrate

| SOTA piece | Vx equivalent | Status |
|---|---|---|
| Stable cross-module identity (def-id / token) | **256-bit GID** (`src/gid.rs`) | ✅ done |
| Interface table (types) | registry `layouts: GID → TypeDefinition{size,align,fields}` | ✅ done |
| Interface table (free fns) | registry `fn_sigs: name → {gid, ret_ty}` | ✅ done |
| Name → def resolution | registry `module_indices: module_hash → name → GID` | ✅ done |
| Interface table (methods / trait impls) | — | ❌ **missing** |
| Lowered impl IR (the "MIR") | **flat HIR stream** `local_hir_stream` + `local_type_stream` | ✅ done |
| Zero-copy metadata serialization | `src/metadata.rs` (`VxMetadata`, `serialize_metadata_symbols`) | ⚠️ dict only; `ast_data` is an empty placeholder |
| Generic monomorphization across a boundary | deferred GIDs + generics arena (journal Entry 1) | ⚠️ exists for the flat path; not wired to a precompiled boundary |

So the decoupling is **two registry features + one serialization completion**, not a new subsystem.

## 4. The protocol: `ModuleInterface`

Define a single query surface the frontend consults for *anything not defined in the current module*.
It is **backed by the frozen registry** and is identical whether the target module was just compiled
(in-memory registry) or loaded from a cached artifact (deserialized registry) — that identity *is* the
decoupling.

```rust
trait ModuleInterface {
    // Already backed by the registry today:
    fn resolve_type(&self, module_hash: u64, name: &Symbol) -> Option<TypeId>;   // module_indices
    fn layout_of(&self, ty: TypeId) -> Option<&TypeDefinition>;                  // layouts
    fn resolve_fn(&self, name: &Symbol) -> Option<&FnSig>;                       // fn_sigs

    // NEW — the method/trait table (§5):
    fn resolve_method(&self, recv: TypeId, method: &Symbol) -> Option<&FnSig>;   // (recv, method) → impl fn GID
    fn resolve_trait_impl(&self, tr: TypeId, ty: TypeId) -> Option<ImplId>;      // trait selection

    // The impl body, for generic instantiation / inlining across the boundary:
    fn body_of(&self, func: TypeId) -> Option<(&[HirInstruction], &[TypeId])>;   // flat HIR stream
}
```

The type checker's `x.exp()` path becomes: *type-of `x` → its GID → `resolve_method(gid, "exp")` →
`FnSig`*. No `GlobalAstEnv.impls` walk, no `&ImplBlock`, no stdlib AST. "Seamless" (your word) means the
resolution code is **the same** for a user method and a stdlib method — both are a registry lookup by
`(receiver GID, method name)`.

## 5. The two registry additions

**(a) Method / trait-impl table.** Extend `ImmutableGlobalRegistry` (`src/registry.rs`), populated in
`build_frozen_registry` (`src/pipeline.rs`) right where `fn_sigs` is built:

```rust
pub methods:    FxHashMap<(TypeId /*receiver*/, Symbol /*method*/), FnSig>,  // inherent + trait methods
pub trait_impls: FxHashMap<(TypeId /*trait*/, TypeId /*type*/), ImplId>,     // trait selection / coherence
```

Minted from the modules' `impl` blocks with the same GID formula already used for `fn_sigs` (the
mangled `f32$exp` becomes the `FnSig.gid`; the *key* is `(f32-GID, "exp")`). Method resolution now lives
in the registry, not `GlobalAstEnv`.

**(b) Serialize the interface, not the AST.** `metadata.rs`'s `ast_data` field was reserved for exactly
this ("*In the future, write AST data here directly*") — but it should carry the **registry interface**,
not AST: the `layouts` / `fn_sigs` / `methods` / `trait_impls` tables + the modules' **flat HIR
streams**. GIDs are already content-addressed and the FNV-hash guard (`src/hash.rs`) versions the
format, so a stale artifact is detected, not silently misread.

## 6. The precompiled stdlib artifact

```
  stdlib/std/*.vx  ──(compile once)──▶  std.vxlib  =  { frozen registry (layouts, fn_sigs,
                                                          methods, trait_impls)
                                                        + flat HIR streams per fn }
                                                          serialized zero-copy (metadata.rs)

  user program  ──▶  mmap std.vxlib (no parse)  ──▶  ModuleInterface query  ──▶  typecheck + codegen
                                                       (links the flat HIR streams; monomorphizes
                                                        generics from them + generic-param GIDs)
```

- **No stdlib AST is ever built** in a downstream compile. The frontend resolves against the mmap'd
  registry; the backend links the flat HIR streams.
- **Generics** (`Vec<T>`, `Math for T`) travel as flat HIR + deferred/generic-param GIDs and are
  monomorphized at the use site — the machinery from journal Entry 1 (generics arena, deferred-GID
  local→global patch) is exactly this, just applied across the precompiled boundary.
- **This is the artifact the flat codegen already wants.** C2/C3 consume the frozen registry + flat
  streams to emit MLIR. So "precompiled stdlib" and "flat pipeline" are the *same* deliverable — the
  decoupling falls out of finishing convergence.

## 7. Migration path (incremental, keep-green)

1. **Add the method/impl table to the registry** (`registry.rs` + `build_frozen_registry`), minted
   from `impl` blocks. Add `resolve_method`/`resolve_trait_impl`. *Dual-run:* keep `GlobalAstEnv` as the
   oracle; assert the registry resolves methods identically (like the flat-vs-AST differential harness).
1. **Point the type checker's method/type resolution at `ModuleInterface`** (registry-backed) for
   *imported* symbols; stop stashing `&ImplBlock`/`&TraitDef` for imported modules. User-module
   resolution can follow.
1. **Complete `metadata.rs`**: serialize the registry interface + flat HIR streams (fill `ast_data` →
   `interface_data`); load a precompiled module instead of parsing its `.vx`. Precompile `stdlib/std`
   to `std.vxlib` in the build.
1. **Flat pipeline consumes the precompiled registry directly** (C2/C3): the stdlib's methods resolve
   and its bodies link with zero AST — the convergence endgame.

Each step is independently testable and reversible, and the differential-harness pattern
(flat-vs-AST, registry-vs-`GlobalAstEnv`) gives a parity gate at each one.

## 8. Payoff and non-goals

- **Payoff:** the stdlib grows without touching the AST/type-checker/codegen surface; per-compile cost
  drops to *resolve-against-metadata* (no re-parse/re-check of stdlib); this is the prerequisite for
  separate + incremental compilation; and it is the **same** registry+flat-HIR artifact the flat
  codegen consumes — decoupling and convergence become one effort.
- **Non-goals / risks to call out:** cross-boundary **generic monomorphization** is the genuinely hard
  part (as in every language above) — but Vx's deferred-GID/generics-arena machinery is designed for
  it; a **stable textual interface** (à la `.swiftinterface`, for ABI stability across compiler
  versions) is a later nicety, not needed for an in-tree stdlib; **coherence/overlap** checking for
  trait impls needs a home (the registry's collision guard is the natural place).

## 9. Open decisions

- **Is the AST frontend staying, with the registry as its import oracle — or does the frontend itself
  move to the flat/GID representation?** (This doc assumes the former near-term: keep the AST for
  *the module being compiled*, resolve *everything imported* via the registry protocol.)
- **Artifact granularity:** one `std.vxlib` for the whole stdlib, or per-module `.vxlib` (finer
  incrementality, more files)?
- **Where does trait coherence live** — checked at `build_frozen_registry` time (like the cycle
  detector) or lazily at resolution?
- **Versioning/ABI:** the FNV format hash guards accidental staleness; do we also want an explicit
  artifact version + compiler-version stamp for distributed/prebuilt stdlibs?

## Key files

- Coupling to replace: `src/hir/env.rs` (`GlobalAstEnv`, the borrowed-AST env).
- The substrate: `src/gid.rs` (GIDs), `src/registry.rs` (frozen registry — add `methods`/`trait_impls`),
  `src/pipeline.rs` (`build_frozen_registry` — mint the method table), `src/metadata.rs` (serialize the
  interface), `src/hir/flatten.rs` / `src/codegen/flat.rs` (flat HIR streams = the impl IR).
- Import resolution: `src/module_loader.rs`, `src/syntax/resolve.rs`.
