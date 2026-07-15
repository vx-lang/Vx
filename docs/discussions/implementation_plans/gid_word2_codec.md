# Implementation Plan: Unify GID Word-2 Encoding Behind One Codec

> Fixes [#193](https://github.com/hiraditya/Vx/issues/193) (GID word 2 double-booked). Design
> rationale: [`../parallel_pipeline_design_review.md`](../parallel_pipeline_design_review.md) (H1).
> **Gated on the S1 decision** — do this only if we are converging the flat-array pipeline toward
> production; if the flat pipeline stays a research track, this drops in priority below the M1 hash
> fix.

## Goal

Word 2 of the 256-bit `TypeId` currently has three inline meanings — fast-path lifetime bitfield,
lifetime slow-path arena index, generic deferred arena index — signalled by **two disagreeing**
flags (`ESCAPE_HATCH_MASK` on word 2 vs `LOCAL_DEFERRED_BIT` on word 3) and read with **two index
conventions** (masked vs raw). Unify on:

- **One is-index bit:** `ESCAPE_HATCH_MASK` (word 2 bit 63). Clear ⇒ inline lifetime bitfield; set ⇒
  word 2 is an arena index.
- **Word-3 flags select arena + scope:** `IS_GENERIC_INST_FLAG` (generics vs slow-meta arena),
  `LOCAL_DEFERRED_BIT` (worker-local vs global index).
- **One codec** in `gid.rs` that is the *only* place that reads/writes word 2, so the three
  interpretations cannot re-diverge.

Invariant established: **word 2 is always exactly one of {inline lifetime bitfield, one arena
index}** — never two things. A type that is *both* borrowed and generic rides the slow-path
`UnboundedFunctionMetadata` arena, which already carries `type_arguments` **and** `lifetime_regions`.

## Current state (grounded)

| Fact | Evidence |
|---|---|
| fast-path lifetime bitfield, escape-hatch clear | `hir/expr.rs::lower_to_type_id`, `borrow.rs::verify_subtyping_bounds` |
| lifetime slow-path: `word2 = ESCAPE_HATCH_MASK \| index`, read `word2 & INDEX_MASK` | `session.rs::resolve_lifetime`, `gid.rs::lifetime_context`, `gid.rs::test_slow_path_arena` |
| generic deferred: `word2 = offset_index` (escape-hatch **clear**), read `word2 as usize` (raw) | `pipeline.rs::mint_deferred_generic`, `pipeline.rs::simd_patch_phase` |
| verifier masks (`word2 & INDEX_MASK`), simd-patch does not | `parallel_architecture_verifier.rs::verify_phase_3_isolation` vs `pipeline.rs::simd_patch_phase` |
| `try_set_fast_param` param 3 writes bits 48–63 — variance top bit == escape-hatch bit | `gid.rs::try_set_fast_param` |

## Status

**W1–W4 landed** (`cf9d5c4` codec; `a947b60` W2/W3/W4). Word 2 is now read/written *only* through
`classify_word2` / `set_arena_index`, across all three touch points (`mint_deferred_generic`,
`simd_patch_phase`, `resolve_lifetime`). Behaviour-preserving: full suite green (274 lib + 49
integration). **W5 is deferred** — it only matters once borrowed generics are actually emitted, which
they are not yet; the invariant (word 2 holds exactly one thing) already forces the composite when
they are. #193 is fixed for the current surface; keep it open only if we want W5 tracked there.

## Milestones

- **W1 — The codec (no behaviour change yet). ✅ `cf9d5c4`.** In `gid.rs`, add:

  ```
  enum Arena { Generics, SlowMeta }
  enum Scope { Local, Global }
  enum Word2 {
      FastLifetime(u64),                 // the inline 4×16-bit bitfield
      Index { index: u64, arena: Arena, scope: Scope },
  }
  impl TypeId {
      fn classify_word2(&self) -> Word2 { ... }             // reads bit 63 + word-3 flags
      fn with_arena_index(index, arena, scope) -> TypeId { ... }  // sets bit 63 + flags
  }
  ```

  Route on `ESCAPE_HATCH_MASK`; when set, read `word2 & INDEX_MASK` and pick `arena`/`scope` from
  `IS_GENERIC_INST_FLAG` / `LOCAL_DEFERRED_BIT`. Unit-test the round-trip matrix here (fast-lifetime,
  slow-lifetime local/global, generic local/global). *Foundational; nothing else changes yet.*

- **W2 — Reserve bit 63 in the fast path. ✅ `a947b60`.** Fix `try_set_fast_param` so the 4th param cannot write
  bit 63 (cap its variance field to 3 bits, or cap fast params at 3.9 — pick and document). Assert
  in a test that a maxed-out 4-param fast GID has bit 63 == 0. *Removes the bonus latent collision.*

- **W3 — Route the generic deferred path through the codec. ✅ `a947b60`.** `mint_deferred_generic` →
  `TypeId::with_arena_index(offset, Generics, Local)` (now sets the escape-hatch bit).
  `simd_patch_phase` → `classify_word2`, remap `Local`→`Global` index, write back via
  `with_arena_index(global, Generics, Global)`. Extraction becomes `word2 & INDEX_MASK` everywhere.
  Verify `verify_phase_3_isolation` and `simd_patch_phase` now agree. *This is the actual bug fix.*

- **W4 — `resolve_lifetime` routes by arena. ✅ `a947b60`.** Use `classify_word2`: `Index{arena: Generics, ..}` →
  the generics arena; `Index{arena: SlowMeta, scope}` → local/global slow-path arena;
  `FastLifetime(bits)` → fast path. Regression test: a generic deferred GID is **not** misread as a
  lifetime bitfield (the H1 symptom).

- **W5 — Borrowed-generic → slow-path composite (design closure). ⏸ deferred (no borrowed generics emitted yet).** When a type needs both a generic
  entry and lifetime info, `emit_type_gid` routes it to the `UnboundedFunctionMetadata` slow-path
  arena (type_arguments + lifetime_regions) instead of the lean generics arena. Test:
  `&'a List<i32>` produces one `SlowMeta` GID whose arena entry carries both. *Closes the "word 2
  can't hold two things" gap; can trail W3/W4 if borrowed generics aren't yet emitted.*

## Sequencing & risk

W1→W2→W3→W4 is the critical path and is self-contained (the codec + the two producers/consumers). It
is **behaviour-preserving for the current test suite** (generic and lifetime GIDs still round-trip;
the only observable change is that a generic GID now sets bit 63 and is read consistently). W5 is
additive and only matters once borrowed generics are emitted. Low risk: all five touch points are
already covered by (or get) unit tests, and the flat pipeline is test-only, so there is no
production-codegen exposure yet.

## Out of scope

The broader convergence (populating `local_hir_stream`, switching `vxc` to `compile_pipeline`) is
separate; this plan only makes word 2 coherent so that work can mix generic and lifetime GIDs
safely. The M1 hash issue and M2 order-determinism (design review) are tracked separately.
