# Design Review: The Parallel / GID Architecture

> A critical review of [`parallel_compiler_architecture.md`](../parallel_compiler_architecture.md)
> against the implementation, done *before* investing in the remaining convergence work
> (`local_hir_stream`, path convergence). Goal: surface design flaws that would bite that work.
> Findings are grounded in specific code; severity is about impact on the planned work + correctness
>
> - paper claims.

## Summary

The infrastructure is sound and the *idea* (content-hash GIDs, zero-lock epoch session, flat
arrays) is coherent. But there are **two high-severity design flaws** that must be resolved before
converging the type stream, borrow check, and codegen, plus **two medium** issues (a correctness/
claims gap in the hash, and unproven order-determinism), and **one strategic** question (the payoff
needs a codegen rewrite). None are fatal; all are fixable. Details below.

______________________________________________________________________

## H1 — Word 2 is double-booked with *contradictory* encodings (High)

Word 2 of the GID is claimed to be "Generic Context Hash / **Lifetime** Bitfield … dual purpose
based on the entity type" (doc §2, Word 2). But the two uses have **incompatible bit formats**, and
their decoders disagree:

- **Generic instantiation (deferred):** `mint_deferred_generic` (`pipeline.rs`) sets
  `word2 = offset_index` (a *plain* small index, escape-hatch bit **clear**) and flags it with
  `LOCAL_DEFERRED_BIT` in **word 3**. `simd_patch_phase` decodes it as `w2 as usize` — **no mask**.
- **Lifetime slow path:** `resolve_lifetime`/`lifetime_context` (`session.rs`, `gid.rs`) route on
  `word2 & ESCAPE_HATCH_MASK` (bit 63) and decode `index = word2 & INDEX_MASK`. The canonical
  slow-path GID (see `gid.rs::test_slow_path_arena`) is `word2 = ESCAPE_HATCH_MASK | index`.
- **Lifetime fast path:** `lower_to_type_id` packs `(region, variance)` into word 2's low 16 bits
  (escape-hatch clear).

The contradictions:

1. `simd_patch_phase` requires the deferred index to have the escape-hatch bit **clear** (it reads
   `w2` raw); `resolve_lifetime` requires it **set** to recognize a slow-path index. A single GID
   cannot satisfy both.
1. A deferred-generic GID (escape-hatch clear) fed to `resolve_lifetime` is misread as a **fast-path
   lifetime bitfield** — the offset index is interpreted as region/variance bits. Silent garbage.
1. `verify_phase_3_isolation` extracts `index = word2 & INDEX_MASK` (masks bit 63); `simd_patch`
   does not mask. So the verifier and the patch would disagree the moment the escape-hatch bit is
   used.

This is **latent today** only because the type stream (generic GIDs) and the borrow check (lifetime
GIDs) never share a GID. It **blocks** the planned convergence: a borrowed generic `&'a List<i32>`
needs *both* a generic-arena index and a lifetime bitfield in word 2, which is impossible; and
running the borrow check over the type stream would hit misread #2.

**Fix direction.** Give word 2 (or word 3) a small explicit **kind tag** (fast-lifetime /
slow-lifetime / generic-deferred / settled) that all decoders agree on, and one consistent
slow-path signal (pick either `ESCAPE_HATCH_MASK` *or* `LOCAL_DEFERRED_BIT`, not both with different
meanings). Or split the concerns: a lifetime/variance word and a generic-context word cannot share
64 bits if a type is both borrowed and generic.

______________________________________________________________________

## H2 — Cross-module identity is not actually resolved (High)

The GID's whole reason to exist is **cross-module identity**: word 0 = module hash, so a type
`ModuleA::User` referenced from module B resolves to A's GID via `module_indices` (doc §7, Step 3).
That resolution is **not implemented**:

- `Type::resolve_names` (`src/syntax/resolve.rs:23`) resolves a nominal type only against the
  **current module's** symbol table: `mod_syms.and_then(|m| m.get(&name))`. The full `symbol_map`
  is threaded through every arm but **never queried** for the nominal case.
- So a reference to a type defined in another module gets `id = None` (unless the current module
  happens to define a same-named type, which would then *wrongly* shadow it).
- The frozen registry's `module_indices` (built in Entry 2) is therefore **never read** — nothing
  does the namespace→module-hash→symbol lookup the design describes.

Consequences: the flat type stream and the registry only carry *intra-module* identity; the central
cross-crate/cross-module story (zero-swizzle metadata, `module_indices` lookup) is aspirational.
Any work that assumes cross-module GIDs (metadata serialization, cross-module monomorphization
routing in Phase 7) rests on this.

**Fix direction.** Implement namespace resolution in `resolve_names`: split a qualified path,
resolve the leading segment to a module hash (imports/`use` aliases), then look up the symbol in
that module's `SymbolTable` (or the registry's `module_indices`). Until then, the "cross-module"
parts of §7/Phase 7 are not real.

______________________________________________________________________

## M1 — The hash is non-cryptographic; "collisions mathematically impossible" is false (Medium)

`compute_module_hash` / `compute_symbol_hash` (`src/hash.rs`) use **`FxHasher`** (rustc_hash) — a
fast, **non-cryptographic** hash (the code comment even says "In a production environment, this
might use SipHash or xxHash"). The doc claims word 0/1 form "a composite 128-bit … **cryptographic**
hash, rendering collisions **mathematically impossible**." Three problems:

1. FxHash collisions are cheap to find; a GID collision **conflates two distinct types**. Not
   "impossible" — just unlikely by accident.
1. Word 0 and word 1 are compared **per word** (`module_id()` = w0, `symbol_id()` = w1); there is no
   128-bit composite check. Within a module, w0 is fixed, so distinctness rests on the **64-bit**
   symbol hash alone — the "128-bit" framing doesn't apply intra-module.
1. FxHash is **not guaranteed stable across `rustc_hash` versions** — contradicting the design's
   cross-toolchain-stability claim, and inconsistent with `arch.rs`, which hand-rolled FNV-1a
   *specifically* to avoid unstable std hashing (`fnv_dispatch_id`, "not `DefaultHasher`, whose
   algorithm may change").

**Fix direction.** Either soften the paper claim to "content-hash with negligible collision
probability," or switch to a stable, keyed 128-bit hash (SipHash-1-3 / BLAKE3-truncated) and compare
`(w0, w1)` as a unit. At minimum, make the hash algorithm one the project controls (like `arch.rs`
already does) so IDs are reproducible across toolchains.

______________________________________________________________________

## M2 — Stream determinism is proven as a *set*, not an *order* (Medium)

`compile_pipeline_gid_stream_is_deterministic` **sorts** before comparing, so it proves the *set* of
GIDs is stable, not the *order*. Downstream consumers are index-addressed (the metadata dictionary;
`HirInstruction.type_idx` indexes `LOCAL_TYPE_STREAM`), so **order matters** for reproducible
output. `rayon`'s `collect()` preserves input order, so it is *likely* deterministic — but nothing
asserts it. **Fix:** add an order-sensitive determinism assertion (don't sort), or document that
order is not part of the contract.

______________________________________________________________________

## S1 — The payoff requires a codegen rewrite; the flat pipeline is currently vestigial (Strategic)

`vxc` runs the **sequential AST driver** (`driver.rs::execute` → `run_codegen`, MLIR from the AST).
The flat GID/HIR streams only pay off if **codegen consumes them** (the doc's Phase 7: O(1) array
codegen, SIMD patch). Converging means rewriting codegen to consume `local_hir_stream` — which is
also not populated (needs an instruction-selection pass). So the entire flat-array pipeline is, for
now, **exercised only by tests**, not by real compilation.

This is the decision to make before more investment: either **commit to the codegen rewrite** (large,
multi-stage: HIR lowering → flat-stream codegen → switch `vxc` to `compile_pipeline`), or **treat the
flat pipeline as a research artifact** and keep the AST path as production (in which case H1/H2/M1
are about the *research claims*, not the shipping compiler). Building `local_hir_stream` without a
consumer just grows more vestigial scaffolding.

______________________________________________________________________

## L1 — Registry layout is incomplete (Low)

`build_frozen_registry` sets `size_bytes = align_bytes = 0` (not computed), and `layouts` is only
used for cycle detection, never for actual layout/ABI. Fine for now; noted so "the registry has
layouts" isn't overread.

______________________________________________________________________

## Recommendation

Before taking up `local_hir_stream` / path convergence:

1. **Decide S1** — are we converging to the flat pipeline as production, or is it a research track?
   This gates how much the rest matters.
1. If converging: **fix H1 first** (a coherent word-2/word-3 kind discipline) — it is a prerequisite
   for any code that mixes generic and lifetime GIDs, which the HIR stream will.
1. **Fix H2** (cross-module resolution) — otherwise the flat streams/registry carry only
   intra-module identity and the cross-module claims are unfounded.
1. **M1** is cheap and high-value for paper integrity — fix the hash or the claim.
1. **M2** is a one-line test change.
