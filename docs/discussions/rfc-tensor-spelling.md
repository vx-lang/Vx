# RFC: One `Tensor` spelling with static rank (retire `DynTensor`)

**Status:** Landed on `main` — `830fc911` (`Dim`), `ae80cc0d` (`?` per dimension, `DynTensor` removed), `a08aa299` (`extent(i)`, `.shape` gone), `c327e04a` (corpus rewritten, `DynTensor` a fix-it). See "What landed" at the end.
**Target:** Before the first public tag
**Scope:** Surface syntax, type identity, lowering rank. No bounds, no proofs, no admission changes.
**Builds on:** `rfc-unified-tensor-vx-review.md` (findings against `main` at `85342711`). File references below are to that commit and that review.

______________________________________________________________________

## 0. For the implementing agent

This RFC is deliberately small. Its job is to land the **only breaking spelling change** that the later bounded-extents work needs, so that everything after launch can be additive. Anything not listed in Section 3 is out of scope here and belongs to the bounded-extents RFC — do not pull it in.

Before writing code, read the review. Most Phase 0 questions are already answered there; Section 8 lists the few that remain.

Gates that must hold at every checkpoint (these exist today):

- All tests pass.
- `codegen_determinism.rs` (3× at default threads) and `pipeline_scale_test.rs` (1 vs 4 threads) pass byte-for-byte.
- The lock lint at `.github/workflows/ci.yml:38` passes; the exempted-atomics count does not grow.
- `flat_corpus_sweep.rs`: `KNOWN_DECLINES` does not grow. It may shrink.

Gates that this RFC adds are in Section 6 and must exist before the tag.

______________________________________________________________________

## 1. Summary

Retire `DynTensor<T>`. A tensor type is written

```
Tensor<T, [d0, d1, ...], Memory::X>
```

where each `di` is either a static extent or `?`. **Rank is the length of the list and is always static.** Widening from a static extent to `?` stays implicit, exactly as `is_assignable` treats `Tensor` → `DynTensor` today. `extent(i)` becomes the typed way to read a runtime extent; `.shape` on a tensor warns. The memory space enters the flat path's type identity.

After this lands, adding a bounded extent state (`?<=B`) later is a new value in an existing position, not a rename — no user code breaks.

______________________________________________________________________

## 2. Why this and why now

**Renames break users; new dimension states do not.** `DynTensor<T>` → `Tensor<T, [?, ?]>` is a rename. Once the repo is public, every rename is a migration someone else has to do. `?<=B` later is a new state in a position that already exists.

**Rank is hardcoded to 2 today (Vx#404).** `DynTensor` lowers to `[?, ?]` on the flat path (`src/hir/flatten.rs:87-95`) and to `memref<?x?xT>` on the oracle (`src/codegen/lower/expr.rs:2116`). A rank-1 value that reaches a `DynTensor` position is lowered as rank 2. That is a miscompile, not a type error. Putting rank in the type converts every such site into a compile error — which the migration in Section 5 will surface.

**The flat identity drops the memory space.** `tensor_gid` (`src/hir/flatten.rs:48`) hashes only the element type and shape. Two tensors differing only in placement share a GID on the shipping default path. This is a correctness gap independent of tensor spelling and belongs before the tag regardless.

______________________________________________________________________

## 3. Design

### 3.1 Syntax

Element first, dims second, placement third — the order the parser already uses (`src/parser/types.rs:338-392`).

```
Tensor<f32, [2, 3]>                     // static, as today
Tensor<f32, []>                         // rank 0, as today
Tensor<f32, [?, ?]>                     // what DynTensor<f32> means today, rank stated
Tensor<f32, [?]>                        // rank 1, dynamic extent — expressible for the first time
Tensor<f16, [?, 4096], Memory::HBM>     // mixed: dim 0 dynamic, dim 1 static
```

- Add a `?` token to the lexer. It is valid only inside a tensor dims list.
- The dims list is **required in every type position** (parameters, returns, fields, annotations, impl receiver patterns). `Tensor<T>` with no list no longer parses as a type. The constructor `Tensor<T>([...])` is unchanged (Section 3.6).
- `DynTensor` becomes a parse error with a fix-it: *"`DynTensor<T>` is spelled `Tensor<T, [?, ?]>`; state the rank."* Not an alias. Nothing in the public tree ships the old spelling.

### 3.2 Per-dimension extent state and subtyping

Each dimension is `Static(n)` or `Unbounded`. Per dimension, `Static(n) ⊑ Unbounded`. Two tensor types are in the subtype relation iff same rank, same element type, same memory space, and every dimension is. Rank mismatch is an error, never a coercion.

**Widening is implicit**, as today: `is_assignable` (`src/hir/expr.rs:496-530`) generalizes from "shaped `Tensor` at a `DynTensor` position" to "per-dimension `⊑`". The flattener keeps minting the `memref.cast` (`forget_extents_for_param` / `forget_extents_for_return`, `src/hir/flatten.rs:2992-3011`; `cast_memref`, `src/codegen/flat/emit/arith.rs:315`). This is structural subtyping decided from the two types alone — it changes no bits and involves no scope-wide search — so it is compatible with the frontend's no-implicit-conversion rule, which targets representation-changing conversions.

`as` between tensor types is **not** added here. (An explicit spelling may come with the bounded-extents RFC; it is not needed for this one.)

### 3.3 Method resolution

Impl receiver patterns state rank:

```
impl<T> Tensor<T, [?, ?]> { fn fill(...) }      // was: impl<T> DynTensor<T>
impl<T: Float, const N: i32, const M: i32> Tensor<T, [N, M]> { ... }   // unchanged (stdlib/std/tensor.vx:219)
```

In a pattern, `?` matches any extent; a static extent or `const` parameter matches as today. The dims-less wildcard match (`src/hir/env.rs:661-675`) is retired. A rank-1 receiver calling a method declared on `[?, ?]` is a type error — this is the Vx#404 fix becoming visible.

### 3.4 Identity

`tensor_gid` builds a formatted string and content-hashes it. Extend the string, nothing else:

- one token per dimension: the integer for `Static(n)`, a fixed token (e.g. `?`) for `Unbounded`;
- the **resolved** memory space — never the `stated: Stated` field (`src/syntax/types.rs:206-218`), which records how the source spelled it and would give one type two identities.

No new identity mechanism, no counter, no table. The existing content hash is the digest.

### 3.5 Lowering

Rank N from the dims list on both paths. `Tensor<T, [?, 3]>` → `memref<?x3xT>`; `Tensor<T, [?]>` → `memref<?xT>`. Remove the rank-2 assumptions at `src/hir/flatten.rs:87-95` and `src/codegen/lower/expr.rs:2116`. memref already supports mixed static/dynamic dims; no attribute is needed.

### 3.6 Construction

`Tensor<T>([e0, e1, ...])`, `::new([...])`, `::uninit([...])` (`src/hir/check/calls.rs:1202-1256`) keep taking the shape as a runtime argument. The result type is derived per position: a literal (or an expression the checker already folds) gives `Static(n)`; anything else gives `?`. Rank is the argument count. The result is then assignable to an annotated type by `⊑`. Phase 0 confirms what the constructor's result type is today so that this is a generalization, not a change.

### 3.7 `extent(i)` and `.shape`

Add `t.extent(i)` as the typed accessor for a runtime extent; it lowers to `TensorDim` (opcode 44), which exists on the flat path. `i` must be a literal less than the rank (compile error otherwise).

`.shape` on a tensor value emits a deprecation warning pointing at `extent(i)` and is otherwise unchanged. Removal is in the bounded-extents RFC. (`.shape` is untyped today — `src/hir/check/access.rs:458` — and a `DynTensor`'s shape is unreadable by design, Vx#399; `extent(i)` is the replacement for both.)

### 3.8 Admission: unchanged

A runtime extent still skips the capacity check with W1029 (`src/hir/check/transfer.rs:278`). Promoting that to an error is bounded-extents work and requires the stdlib to have bounds first. Not here.

### 3.9 Explicitly out of scope

Bounds (`?<=B`), `narrow` / `narrow_checked`, W1029 promotion, top-level `const`, `config`, dimension-expression folding on the flat path (`tensor_dim_string`, `src/hir/flatten.rs:162` — a bridge item that is needed before bounds, not before this), bounded-generic stdlib signatures, ownership changes (`Tensor` is linear with `NEEDS_DROP` and stays so), struct-field layout, FFI.

______________________________________________________________________

## 4. Frontend constraints

- Identity remains a pure function of content: the per-dimension tags and the memory space go into the hashed string. Nothing arrival-ordered.
- No new global or thread-local state. The lint applies.
- No new opcodes. `TensorDim` and `Cast` exist; the widening cast is already emitted by the flattener.
- Per-function lowering stays atomic; a construct the flat path cannot handle declines to the oracle as today. `KNOWN_DECLINES` may not grow.
- The determinism gates are unchanged and must pass at every phase.

______________________________________________________________________

## 5. Migration

The review counts 40 `.vx` files spelling `DynTensor`, 18 of which also use `transfer` or `spawn`.

1. **Mechanical pass.** Every `DynTensor<X>` today is rank 2 by construction, so `DynTensor<X>` → `Tensor<X, [?, ?]>` and `impl<T> DynTensor<T>` → `impl<T> Tensor<T, [?, ?]>` is a textual rewrite. Do it in one commit.
1. **Run the checker.** Every rank error it reports is a real latent miscompile — a rank-1 value that was being lowered as rank 2. Each needs a human decision: the value is genuinely rank 1 and the callee needs a `[?]` impl, or the site was wrong. Fix these one commit per file so the diff is reviewable. Do not paper over them with `[?, ?]`.
1. **`.shape` sites.** Leave them; they warn. Record the count in the phase report.
1. **The 18 transfer/spawn files.** Nothing changes for them here — W1029 stays a warning. They are the bounded-extents RFC's migration list.

______________________________________________________________________

## 6. Gates to add before the tag

The review found that three numbers the launch materials call "measured" live only in walkthrough docs. They become tests:

| Claim | Gate | Hardware |
| --- | --- | --- |
| 90-cell admission matrix: 67 admit / 23 reject / 0 errors | integration test over the 15 configs × 6 SKU files | none — compile-time only |
| 64/64 token ids identical to llama2.c | test; CPU variant in CI; GPU variants behind a feature flag | CPU in CI; A100/H100 when present |
| KV transfer 442,368 B on 2×H100 | test that asserts the transfer size against the compile-time figure; skips without 2 GPUs | 2×GPU |
| TSan zero races, positive control fires | `scripts/tsan.sh` in the repo (the command from the deck plus the control); run on the bare-metal box before the tag; CI optional | — |
| byte-identical at 1/8/32/48 threads | keep 1 vs 4 in CI; add `scripts/determinism-ladder.sh` for the bare-metal box | — |
| flat path coverage | `flat_corpus_sweep.rs` already gates; additionally record `examples/llama.vx`'s status (reference path today) in the sweep output so the README claim is checked by the same test | none |

A gate that skips without hardware must still *exist* and must run to completion when the hardware is present.

______________________________________________________________________

## 7. Launch-material consistency (outside the tree)

- README: the diagnostic family is E6001–**E6018** (`src/diagnostic.rs:270-340`), not E6014. State that `llama.vx` compiles via the reference path today and that the flat path covers 95 of 130 corpus programs; do not let the frontend scaling number attach to the demo.
- Deck, slide 9: `Ref<T, Memory::NPU_HBM>` no longer parses; write `Tensor<T, [..], Memory::NPU_HBM>`. Slide 10's `Tensor<f16>([8, 64])` is the constructor and stays valid. Slide 14: one clause — tensor shape subtyping is structural, decided from the two types alone, and pre-existing.
- Nowhere claim bounded dynamism at the language level. It is Vx#245, open. Claim it when the bounded-extents RFC lands.

______________________________________________________________________

## 8. Phases

### Phase 0 — remaining questions (short)

1. What is the constructor's result type today for literal vs non-literal shape arguments?
1. Can two tensors that differ only in placement reach the flat path in the same compile? Write the test either way; if yes, confirm whether it miscompiles before the identity fix.
1. Enumerate the 40 `DynTensor` sites with the rank each is *intended* to have.
1. Enumerate `.shape` sites.

**Checkpoint:** written report.

### Phase 1 — lexer and parser

`?` token; dims list accepts it; dims list required in type positions; `DynTensor` parse error with fix-it.

### Phase 2 — types, assignability, resolution, identity

Per-dim state; rank; `is_assignable` per-dim; impl-pattern matching with rank; `tensor_gid` string extended with per-dim tags and resolved memory space.

**Checkpoint:** all gates; the Phase 0 placement test passes.

### Phase 3 — lowering

Rank N on both paths.

**Checkpoint:** differential tests (`flat_codegen_differential.rs`) for a rank-1 and a mixed-dim tensor; determinism gates.

### Phase 4 — `extent(i)`, `.shape` warning

### Phase 5 — migration (Section 5)

**Checkpoint:** no `DynTensor` in the tree; every checker-found rank error resolved by decision, not by `[?, ?]`; `KNOWN_DECLINES` did not grow.

### Phase 6 — gates (Section 6)

### Phase 7 — docs, README, deck notes (Section 7)

______________________________________________________________________

## 9. Tests

- Parse: each form in 3.1; `Tensor<T>` without dims in a type position is an error; `DynTensor` is an error with the fix-it text.
- Subtyping: every per-dim case, positive and negative; rank mismatch is an error at each site (argument, assignment, return, field, receiver).
- Identity: two tensors differing only in placement have different GIDs; `stated` variations of the same placement have the same GID.
- Lowering: rank 1, rank 3, mixed static/`?` — flat vs oracle differential.
- `extent(i)`: in range, out of range (error), non-literal `i` (error).
- Migration: a rank-1 value into a `[?, ?]` method is a type error (regression for Vx#404).
- Section 6 gates.

______________________________________________________________________

## 10. What landed

Sections 3.1–3.7 as written, with the decisions the review asked for:

- A dimension is `Dim::Static(Expr)` or `Dim::Dyn` (review §1.1). `?` mangles as `_`.
- Section 3.4's identity change was not made: the placement-twin test passes without it
  (review §1.3), so the memory space stays out of `tensor_gid`.
- Two impls matching one receiver resolve by specificity (review §1.4), pinned by
  `tests/frontend/pass/impl_most_specific_pattern_wins.vx`; a `const` dimension refuses
  a `?` argument.
- The dims-list requirement was already the tree's behavior; Phase 1 was the `?` token
  and the fix-it.

Two deviations to know about:

- A `[?, ?]` value does not narrow to a static binding implicitly. The RFC's `⊑` says so,
  and the one corpus program that relied on it (`kernel_kind_matmul.vx`) now spells its
  accumulator `[?, ?]`. Until the bounded-extents RFC adds `narrow`, the only spelling
  for that narrowing is annotating the binding with `?`.
- The checker still builds an empty dimension list internally for a shape it does not
  know yet (a matmul result before its shape is settled, `operators.rs`), and
  assignability keeps its old leniency for that case. Rank is otherwise never coerced.

Migration step 2 found one latent miscompile, not zero: `traits.vx` passed a rank-0
tensor to an impl for `[?, ?]`, and codegen bitcast an f32 into a rank-2 memref. Every
corpus `DynTensor` was rank 2, as predicted. Writing the fixtures found a second,
unrelated hazard: a user function named `sum` is dispatched as the AST path's slice
reduction (Vx#457).
