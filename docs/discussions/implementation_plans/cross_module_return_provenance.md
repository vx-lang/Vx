# Implementation Plan: Cross-Module Return Provenance (`.vxlib` step 7)

**Status:** phase 1 (vertical slice) **landed** and phase 2 **largely landed** — a consumer compiled with `vxc --link-interface <f.vxlib>`, the library source never passed, resolves imported calls from the merged registry and borrow-checks them with cross-module provenance (accepting a reborrow of a non-aliased local, rejecting the aliased one); a scalar import also **runs** (`double(21) → 42` via `--link-interface --run`). The one piece left is a *runnable reference-provenance* demo, blocked on flat-codegen coverage of scalar references ([#273](https://github.com/hiraditya/Vx/issues/273)). See [§7](#7-phase-2-status).
**Tracks:** the "step 7" of [`borrow_checker_parameter_provenance.md`](../borrow_checker_parameter_provenance.md) §4.3 / §8.1 — serialise the per-function return-provenance summary into the module interface so the borrow check stays precise **across a compile boundary**.
**Companions:** [`stdlib_decoupling_protocol.md`](stdlib_decoupling_protocol.md) (the `ModuleInterface` protocol), [`vxlib_bodies_and_loader.md`](vxlib_bodies_and_loader.md) (the artifact codec + loader), [`../borrow_checker_architecture.md`](../borrow_checker_architecture.md) §2 (the inline slot-0 encoding).
**Issues:** [#265](https://github.com/hiraditya/Vx/issues/265) (inline encoding — landed), [#220](https://github.com/hiraditya/Vx/issues/220) / [#224](https://github.com/hiraditya/Vx/issues/224) (the `.vxlib` interface / stdlib-decoupling epic).

______________________________________________________________________

## 0. TL;DR

The per-parameter return-provenance summary (#264) and its inline 3-bit `TypeId` encoding (#265) are
implemented and give full precision **within a single compilation**. The remaining half — precision
**across** modules — needs the summary to travel in the `.vxlib` interface and be read by a downstream
compile whose copy of the callee has no AST.

We deliver this in **two sequenced phases, the first a strict subset of the second**:

1. **Vertical slice** — carry the code on `FnSig`, serialise it, and have the borrow check *read* it
   for an imported callee, proven by one end-to-end test. Load-bearing, keep-green, no convergence
   dependency.
1. **Full production load path** — generalise the slice's minimal merge entry into `load_import`
   auto-resolving a `.vxlib` (the #219/#220 endgame). The frontend consumes the interface for all
   imported symbols; imported-*body codegen* rides convergence (C2/C3).

Nothing in phase 1 is thrown away by phase 2 — see [§4](#4-why-the-slice-is-not-throwaway).

______________________________________________________________________

## 1. Why this is the last piece

`#265` put the summary in slot 0 of the return type's `TypeId` (`FAST_RETURN_PROV_MASK`), read on the
intra-compilation `verify_subtyping_bounds` path. But the *consumer* of the summary at a call site is
`hir::env::return_provenance_of`, backed by `GlobalAstEnv.return_provenances` — a map built **only from
AST function bodies** (`annotate_return_provenances` guards on `!func.body.is_empty()`). For an imported
symbol resolved from a `.vxlib`, no AST body is present, so the lookup misses and falls to the
conservative `AnyParam` (`hir/expr.rs`, the `Expr::FunctionCall` arm). Correct and sound, but maximally
imprecise for every cross-module call — exactly the annotation-free-*and*-precise property #265 is meant
to preserve.

Today this is latent, not observed: no real compile loads a `.vxlib` (`load_import` only parses `.vx`;
`merge_from` / `deserialize_registry_interface` have no non-test callers), so an `import` re-parses the
library source and the AST summary *is* populated. The precision gap opens precisely when the import
stops being parsed — which is the whole point of the `.vxlib` interface.

## 2. Where the summary must live in the interface

The interface record for a callee is `registry::FnSig`. Today:

```rust
pub struct FnSig { pub gid: TypeId, pub ret_ty: syntax::Type }
```

`gid` is the function's *identity* (`TypeId::new(module_hash, symbol_hash, 0, 0)`, word 2 = 0), **not**
the return type — and the serialised `ret_ty` is a reconstructed AST `Type` whose reference variants
carry no `TypeId` at all. So there is no return-type `TypeId` in the interface to hang a slot-0 code on.

We therefore carry the code as an explicit field on the record it travels with:

```rust
pub struct FnSig {
    pub gid: TypeId,
    pub params: Vec<syntax::Type>,   // NEW — the borrow check needs each param's ref-ness
    pub ret_ty: syntax::Type,
    pub ret_prov: u8,                // NEW — encode_return_provenance(...) of the callee
}
```

`ret_prov` holds the same 3-bit code as the slot-0 encoding (`provenance::encode_return_provenance`);
storing it on the `FnSig` is "in the interface record," not a separate side table — the cross-module
analogue of "inline in the type's identity." `params` is the enrichment #219 also needs ("do it once"):
the reborrow-conflict check needs each parameter's reference-ness, which `FnSig` does not carry today.

## 3. The vertical slice (phase 1)

Goal: a consumer that calls `pick(a: &Map, b: &Map) -> &i32 { return &b.slot; }` — resolved with only
the *interface* present, **no imported body or AST summary** — borrow-checks

```rust
let r = pick(&x, &y);
insert(&mut y, 99);   // E4003/E4004 — r aliases y        (must reject)
insert(&mut x, 99);   // OK          — r does not alias x  (must accept)
```

with the per-parameter precision coming solely from the code carried in the deserialised interface.

### Steps — landed (phase 1)

| # | Change | Where |
|---|--------|-------|
| 1 | Add `params` + `ret_prov` to `FnSig`; compute `ret_prov = encode_return_provenance(compute_return_provenance(f))` from the AST at freeze; fill `params` from the signature. All 5 `FnSig` construction sites updated (free fns / externs / methods in `build_frozen_registry`; the two in the deserializer). An `extern` — opaque, no body — takes the conservative top. | `src/registry.rs`, `src/pipeline.rs` |
| 2 | Serialise/deserialise `params` (via `write_type`/`read_type`) and `ret_prov` (one byte) in the `fn_sigs` **and** `methods` sections (shared `encode_sig_record`/`read_sig_record`). Bump `VXLIB_FORMAT_TAG` `v3 → v4` so a stale artifact is rejected by the FNV guard, not misread. | `src/metadata.rs` |
| 3 | At the persist decision, on an AST-summary *miss* (`return_provenances` has no entry — an import) read the callee's `FnSig.ret_prov` from the frozen registry and gate `persists(i)` with `provenance::inline_prov_includes(code, i)` instead of the conservative `AnyParam`. A summary *hit* (any in-compilation function) is untouched, so a local definition always wins. | `src/hir/expr.rs` |
| 4 | Tests: (a) codec round-trip asserts `params`/`ret_prov` parity and that `pick`'s code is `2` (derives from slot 1); (b) a checker-level end-to-end test — freeze a lib registry, round-trip it through the codec, check a consumer whose imported signatures are visible but body-stripped (so the summary misses), and assert **1** borrow error with the interface vs **2** without it (the conservative baseline). | `src/metadata.rs`, `src/pipeline.rs` tests |

### Deferred to phase 2 (the #219 flip + a real driver entry)

- **Registry-backed signature resolution.** `resolve_callee_ref_signature` (and the general call
  type-check) resolving an imported callee whose *signature* is present only in the registry — the
  #219 "retire the borrowed-AST env for imports" flip. Phase 1 feeds the signature through the env
  (the imports are still parsed, or modelled body-stripped), so this is inert until phase 2 exercises
  it — it is deliberately **not** added yet (it would be carried-but-unread).
- **A driver `--link-interface` / `load_import` auto-resolution.** A real `vxc` "no source parsed"
  run needs the signature-resolution flip above to even type-check the imported call, so the CLI entry
  composes with phase 2 rather than standing alone.

### Soundness

The read is a **conservative refinement** by construction (`encode_return_provenance`, proven by
`inline_code_conservatively_refines_the_summary`): a missing/again-conservative code is `7` = "any
parameter", i.e. today's behaviour. So an incomplete or absent `ret_prov` is only ever a precision
loss, never unsound — the same guarantee that governs the intra-compilation path.

## 4. Why the slice is not throwaway

Every artefact of phase 1 is consumed verbatim by phase 2 (the full `load_import` → `.vxlib` path):

- `FnSig.params` / `FnSig.ret_prov` and their codec — the interface record the full path serialises and
  reads; unchanged.
- The borrow-check read via `inline_prov_includes` on an AST-summary miss — unchanged; it does not care
  *how* the registry came to hold the callee (a test merge today, `load_import` tomorrow).

Phase 2 adds, on top and without reworking the above: registry-backed signature resolution for imports
(the #219 flip — needed so an imported call type-checks at all with no AST), a driver entry
(`--link-interface`, then `load_import` auto-detecting a sibling `.vxlib`) that `merge_from`s the
interface into the session registry, and imported-*body* codegen — the last of which is
**convergence-gated** (the production `vxc` path is still AST codegen; linking AST-free artifact bodies
needs the flat codegen as production, C2/C3). Phase 1's provenance consumption needs none of that: it
reads a byte off a `FnSig` the registry already holds.

## 5. Non-goals

- **Not** the full stdlib decoupling (#224) — that epic has open design decisions (numeric trait tower,
  error model, artifact granularity) needing a human call, and imported-body codegen gated on
  convergence. This plan is only the borrow-check/provenance vertical.
- **No** location sensitivity, no change to the intra-compilation `verify_subtyping_bounds` math — the
  slot-0 encoding and its 9-bit region stand as in #265.
- **No** generic cross-boundary monomorphization changes — a non-portable (deferred-GID) body is still
  skipped by the codec; its `ret_prov` still serialises on the `FnSig`, so the *summary* crosses even
  when the *body* does not.

## 6. Key files

- `src/registry.rs` — `FnSig`, `ImmutableGlobalRegistry`, `ModuleInterface`, `merge_from`.
- `src/pipeline.rs` — `build_frozen_registry`, `emit_module_interface`, `harvest_bodies`.
- `src/metadata.rs` — `serialize_registry_interface` / `deserialize_registry_interface`, `VXLIB_FORMAT_TAG`.
- `src/hir/expr.rs` — the `Expr::FunctionCall` persist decision, `resolve_callee_ref_signature`.
- `src/hir/provenance.rs` — `encode_return_provenance`, `inline_prov_includes`.
- `src/driver.rs` — `--emit-interface`, the new `--link-interface`.

## 7. Phase 2 status — what landed

Phase 2 was taken as far as the flat-codegen coverage allows, in keep-green milestones:

- **M1 — registry-backed frontend resolution (landed).** An imported call with no AST resolves from
  the merged registry: the type-check falls back to `FnSig` at the `E2002` site (arg-check vs
  `params`, result `ret_ty`), and `resolve_callee_ref_signature` resolves the same `FnSig` for the
  borrow reborrow-tracking; the persist decision then applies `ret_prov`. Both fallbacks are
  **additive** — inert when the registry is empty (every normal compile), so no dual-run gate is
  needed and nothing regresses. Proven by `borrow_check_resolves_imported_call_from_registry_only`
  (the callee exists *only* in the registry) — accepts a reborrow of the non-aliased local, rejects
  the aliased one.

- **M1d — the `--link-interface` driver flag (landed).** `vxc --link-interface <f.vxlib>` deserialises
  the interface and builds the session `with_registry(merged)` for the frontend, and merges it into
  `build_flat_module`'s registry for codegen. Verified through the real driver: a consumer compiled
  with only the app + the artifact (the library source never on the command line) borrow-checks the
  cross-module provenance correctly.

- **M2 — runnable, scalar import (landed).** `vxc --link-interface mathlib.vxlib app.vx --run` JITs
  `double(21) → 42`, the library never parsed — `build_flat_module` merges the interface and appends
  each referenced imported body (from `body_of`) as a signature-only `Function` to the emit list.
  Proven by `driver_link_interface_runs_a_scalar_import`.

- **Clean decline (landed).** The AST codegen path cannot link an imported body (it has no AST for it),
  so under `--link-interface` a flat decline is a **clean error**, never an AST-fallback ICE. Guard:
  `driver_link_interface_declines_cleanly_outside_flat_subset`.

- **M3 — runnable *reference*-provenance demo (blocked, [#273](https://github.com/hiraditya/Vx/issues/273)).**
  The remaining goal — a consumer that *compiles + runs* a program the conservative rule would reject,
  because the imported reference return derives from one specific parameter — needs the flat path to
  lower scalar references (`&x` on a local, a reference return, a `*r` load). That pattern is outside
  the flat subset today (and the AST path fails it too), so only *codegen* of the reference case is
  missing. The **frontend already does the full job**: with `--link-interface`, the consumer
  type-checks and borrow-checks against the `.vxlib` with per-parameter precision (accept the
  non-aliased mutable reborrow, reject the aliased one) — the provenance crosses and is enforced; it
  just can't be *run* until #273 lands.

- **`load_import` auto-detecting a sibling `.vxlib`** (landed, #219). An `import mathlib;` now resolves
  to `mathlib.vxlib` where the source would be (`resolve_artifact_path`), and its interface bytes are
  collected (`ModuleLoader.loaded_interfaces`) and merged into both the frontend session and the flat
  codegen — with the module's source never parsed, and no `--link-interface` flag. `--link-interface`
  and any number of auto-loaded imports compose (the driver merges the whole set). Proven by
  `driver_import_auto_resolves_a_vxlib_artifact` (`import mathlib;` + a sibling `.vxlib`, source
  deleted, JITs to 42). A `.vxlib` is preferred over `.vx` when both exist; staleness (rebuild on a
  newer source) is a follow-up.

- **Imported struct field access + construction off the AST env** (landed, #219). `registry.structs`
  (`StructFields` — the un-erased declared field types) is now serialized into the `.vxlib` (format
  tag `v5`) and carried by `merge_from`; both `check_memberaccess_expr` (reading `p.x`) and
  `check_structinit_expr` (building `Point { .. }`, incl. missing/extra/mismatched-field checking) fall
  back to it when `env.structs` misses, so an imported struct types the same way a local one does — a
  consumer can construct one and read its fields with no source. Proven runnable by
  `driver_import_uses_a_struct_from_a_vxlib` (`origin() -> Point`, `p.x` → 7) and
  `driver_import_constructs_a_struct_from_a_vxlib` (`let p = Point { x: 5, .. }`, `p.x` → 5). A `let p: Point = ..` annotation also works (the type flows from the literal).

### Still deferred

- **Bare imported-nominal type classification.** `resolve_parsed_type` (`src/hir/env.rs`, on the
  registry-less `GlobalAstEnv`) still demotes a bare imported nominal in a *type position* to
  `Generic` — harmless where the type flows from a value (construction, annotation-with-initializer),
  but a `let p: Point;` with no initializer, or an imported struct as a bare parameter/return
  annotation in *consumer* code, would need a registry check here.
- **`.vxlib` staleness** — a `.vxlib` is preferred over `.vx` unconditionally; rebuild-on-newer-source
  is unimplemented.
- **Imported-body codegen for the full language** — convergence-gated (C2/C3); the flat path links what
  it can lower, which is why a scalar/struct-by-value import runs and a reference one waits on #273.
