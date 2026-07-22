# `.vxlib` stage 3+4: flat-HIR bodies and the precompiled-artifact loader

Design note for the remaining two stages of the stdlib↔compiler decoupling Step 3 (#220). Stages 1–2
(the versioned `.vxlib` codec for the registry *query interface* — `module_indices`, `layouts`,
`fn_sigs`, `methods`) are landed (`3d02ece`, `e943829`). This note pins the architecture for the
**bodies** and the **loader/build** wiring *before* implementing, so the on-disk format isn't locked to
a speculative stream representation.

See also: [`stdlib_decoupling_protocol.md`](./stdlib_decoupling_protocol.md) (§5b/§6, the protocol),
[`flat_pipeline_convergence.md`](./flat_pipeline_convergence.md) (C2/C3, the flat codegen that consumes
this artifact).

## 0. TL;DR

- **Bodies are self-contained records keyed by function GID**: `FnBody { name, params, ret_ty, hir, types }`. The signature travels with the body because the flat codegen (`emit_function_mlir`) reads
  `func.params` + `func.return_type` to emit the MLIR function header — there is no AST `Function`
  downstream.
- **Only non-generic bodies are portable this stage.** Their type stream is already **global**
  content-hash GIDs (the local→global SIMD patch only rewrites *generic-instantiation* deferred GIDs).
  Generic bodies carry per-compilation deferred GIDs and need monomorphize-from-HIR — deferred, exactly
  as cross-boundary generic monomorphization is deferred everywhere else.
- **`bodies` lives on `ImmutableGlobalRegistry` but is empty in a from-scratch compile** — the live
  pipeline keeps bodies in the per-worker streams as today. `bodies` is populated only when a registry
  is **deserialized from an artifact**, so a downstream compile can `body_of` the stdlib. This sidesteps
  the freeze-ordering problem (the registry freezes at Phase 2, *before* HIR lowering at Phase 3).
- **Shared prerequisite with #219**: `FnSig`/`FnBody` must carry **parameter types**. The same
  enrichment unblocks the #219 imported-*call* type-check flip and this stage's body codegen.

## 1. What the code forces (grounding)

- `codegen/flat.rs::emit_module_mlir(funcs: &[(&Function, &[HirInstruction], &[TypeId])], registry, tensor_types)` is the flat codegen entry. Per function it takes `(&Function, hir, types)` and reads
  from the `Function` **only** `params` (each param's `Type`) and `return_type`. So a serialized body
  must reconstruct those two, plus a name for the `func.func @<name>` symbol.
- The per-function flat HIR is `LocalWorkerState::{local_hir_stream, local_type_stream}`, produced in
  Phase 3 (`type_check_phase` → `hir::flatten`). The pipeline today **does not** persist these into any
  central, GID-keyed store — they are verified (`verify_hir_stream`) and, for the type stream, harvested
  for dedup/SIMD-patch. There is no `body_of` backing yet (it was the piece #219's `ModuleInterface`
  deferred).
- **GID form.** A non-generic type reference contributes the settled global GID `resolve_names`
  attached (module+symbol content hash). A generic instantiation contributes a *deferred* GID whose
  word 2 is a per-compilation generics-arena index; Phase 5 interns it and Phase 6 (`simd_patch_phase`)
  remaps it to the global offset. Therefore a body **with no generic instantiations** has a fully
  global, directly-portable type stream; a generic one does not.
- **Freeze ordering.** `build_frozen_registry` runs at Phase 2 (the sequential freeze) from *resolved*
  modules — before any body is lowered (Phase 3). Bodies cannot be a freeze-time field.
- `module_loader.rs` is a recursive, parse-based import resolver: `load_main` parses the entry file,
  collects `imports`, and `load_import` resolves `std::math` → `stdlib/std/math.vx` and parses it. The
  artifact hook is here: resolve to a `.vxlib` and deserialize instead of parsing the `.vx`.

## 2. Stage 3 — the body store + codec

### 2.1 Data model

Add to `ImmutableGlobalRegistry`:

```rust
pub bodies: FxHashMap<TypeId /*fn GID*/, FnBody>,
```

```rust
pub struct FnBody {
    pub name: crate::symbol::Symbol,   // MLIR symbol / mangled emit name
    pub params: Vec<crate::syntax::Type>,
    pub ret_ty: crate::syntax::Type,
    pub hir: Vec<crate::hir::bytecode::HirInstruction>,
    pub types: Vec<crate::gid::TypeId>, // the body's global-GID type stream
}
```

`body_of(gid) -> Option<&FnBody>` becomes the `ModuleInterface` method #219 deferred. `params`/`ret_ty`
duplicate `fn_sigs`/`methods` slightly; that is deliberate — an artifact body is **self-contained** so
codegen needs one lookup, not a join.

`bodies` is `FxHashMap::default()` at both construction sites today (`registry.rs::build_and_validate`,
`metadata.rs::deserialize_registry_interface`) — **empty in a normal compile**. It is filled only by the
deserializer (§3) and, at artifact-emit time, by the harvester (§4.1).

### 2.2 Codec

`HirInstruction` is `#[repr(C)]` of `opcode: Opcode(#[repr(u32)]) + operand1/operand2/type_idx: u32 + imm: u64` — a fixed 24-byte record. Encode field-by-field (LE); decode `opcode` via a checked
`Opcode::from_u32(u32) -> Option<Opcode>` (a match over the contiguous `0..=30` discriminants — **no**
`transmute`, so an out-of-range tag errors, never UB). The type stream is `Vec<TypeId>` (POD, already
covered). The body section appends after `methods` in the `.vxlib` payload; format bumps to **v3**.

**Portability gate (fail-closed, matching stage 2):** a body is written only if every GID in its type
stream is global — i.e. no `is_local_deferred()` word-2 index. A generic body (or one referencing a
generic instantiation) is **skipped**, so `body_of` declines it and the downstream falls back to the
source path. `serialize_registry_interface` stays infallible; the harvester decides which bodies qualify.

### 2.3 Stage-3 test (no pipeline plumbing yet)

Lower a *real* non-generic function through `hir::flatten` against a frozen registry (as the existing
`flatten` tests do), synthesize a `FnBody` from its `(name, params, ret, local_hir_stream, local_type_stream)`, insert into `registry.bodies`, serialize → deserialize, and assert `body_of`
returns byte-identical `hir` + `types` and equal signature. This proves the codec against the true
stream form without yet touching the parallel pipeline.

## 3. Stage 4 — precompiled `std.vxlib` + loader

### 3.1 Producing the artifact

A new emit mode (`vxc --emit-interface <out.vxlib>`, or a dedicated `vx-buildstd` bin) runs the normal
frontend over `stdlib/std/*.vx` through Phase 3, then:

1. takes the frozen registry (types/layouts/sigs), and
1. **harvests bodies**: for each non-generic function/method whose flat HIR lowered completely and whose
   type stream is fully global, build a `FnBody` and collect into a `bodies` map keyed by the fn/method
   GID (the same GID `fn_sigs`/`methods` already mint).

Then `serialize_registry_interface` (extended to walk `reg.bodies`) writes the payload;
`VxMetadata::save_with_interface` writes `std.vxlib`. The `stdlib/rust_core` build already has a
`build.rs`; precompiling `std.vxlib` slots in there (or a `make`/`cargo xtask` target) so it is a build
artifact, cached and regenerated when the stdlib or the format stamp changes.

### 3.2 Consuming the artifact

- **Loader.** In `module_loader.rs::load_import`, before parsing `stdlib/std/<m>.vx`, look for a
  precompiled interface (a single `std.vxlib`, keyed by module path). If present and its format stamp
  matches, **do not parse**; record the module as "precompiled" and stash its deserialized interface.
- **Registry merge.** `build_frozen_registry` seeds from the precompiled interfaces: union their
  `layouts` / `module_indices` / `fn_sigs` / `methods` / `bodies` into the compile's frozen registry
  (the GID collision guard already rejects genuine conflicts; identical re-listing is a no-op).
  Precompiled modules contribute **no AST** — no `GlobalAstEnv` entries, the #219 endgame.
- **Type-check/codegen.** The type checker resolves imported names via `ModuleInterface` (the #219
  path, now backed by the deserialized registry). For an imported stdlib function/method the program
  *uses*, codegen pulls its `body_of(gid)` and feeds `emit_module_mlir` — no AST `Function`, since
  `FnBody` carries `name`/`params`/`ret_ty`.

### 3.3 Acceptance

A program `import std::math; … x.exp()` compiles with `std.vxlib` **loaded, not parsed** — assert the
`.vx` is never read (e.g. via a load counter / a `VX_STD_PATH` pointing only at the artifact) — and its
JIT output matches the source-compiled path. This also unblocks the softmax corpus (#217): `exp` resolves

- links from the artifact.

## 4. Decisions taken (so the format is stable)

1. **Bodies self-contained** (`name`+`params`+`ret_ty`+`hir`+`types`), keyed by fn GID. *Rejected:* a
   gid→sig join at codegen (needs a gid-keyed sig index the registry lacks).
1. **`bodies` on the registry, empty except when deserialized.** *Rejected:* a sidecar returned beside
   the registry (`body_of` belongs on `ModuleInterface`); *rejected:* populating at freeze time
   (impossible — freeze precedes lowering).
1. **Only global-GID (non-generic) bodies this stage; skip the rest.** Generic-body portability
   (monomorphize-from-HIR + generic-param GIDs) is a later step, consistent with fail-closed stage 2.
1. **`FnSig`/`FnBody` carry `params`.** Shared with the #219 flip; do it once.
1. **Format v3**, FNV stamp guards staleness; one `std.vxlib` for the whole stdlib to start
   (per-module `.vxlib` is a later granularity choice).

## 5. Work breakdown

- **3a.** `Opcode::from_u32`; `HirInstruction`/`FnBody` codec; `bodies` field + `body_of` on registry &
  `ModuleInterface`; extend serialize/deserialize (format v3); stage-3 round-trip test. *(No pipeline
  change; live compiles keep empty `bodies`.)*
- **3b.** Enrich `fn_sigs`/`methods` (or `FnBody`) with `params` at `build_frozen_registry` time.
- **4a.** Body harvester + `vxc --emit-interface`; precompile `std.vxlib` in the build.
- **4b.** Loader artifact path + registry merge; skip parsing precompiled modules.
- **4c.** Codegen links `body_of` for used imported symbols; end-to-end `import std::math` acceptance +
  softmax corpus (#217).

Each item is independently testable and keeps the suite green; the round-trip / differential parity
gates carry over from stages 1–2.
