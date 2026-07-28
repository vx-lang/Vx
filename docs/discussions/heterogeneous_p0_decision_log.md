# Heterogeneous P0 work — decision log

**Author:** autonomous session (Claude), continuing from the `#242` flat-path convergence work.
**Context:** the user asked me to evaluate `heterogeneous_target_gap_analysis.md`, audit its claims,
and then proceed on the reasonable near-term work while they were away — making my own calls and
documenting them here for later review.

## The audit (done first, at the user's request)

I re-ran the checked-in compiler against the document's own reproductions. **Every executable claim
held, most verbatim.** Confirmed: §4.2 multi-hop routing (TPU 2-hop / B200 3-hop-to-TMEM), §4.3
granule offsets (0 / 65536 @ 4096), §4.4 E6009/E6010/E6007 (byte-exact), §4.5 const-generic
capacity check, §4.6 the z3 seam counterexample (verbatim, behind `--verify-seams`), and the bugs
B1–B4 and the §7 static gaps (no async keywords; no FP8; `Type::Tensor` has 3 fields; the
`element_bits(I4)=4` vs `scalar_size_align(I4)=1` table disagreement is real). Two minor loose spots
in the doc, not errors: E6009 dominates E6010 when a *single* tile already overflows (E6010 needs
tiles that individually fit but collectively don't), and the E6004 seam proof is `--verify-seams`-
gated (the doc does say so). Not verified (outside a compiler run): hardware facts, z3 timings, the
NVPTX effort estimate.

**Conclusion:** the analysis is trustworthy as a planning basis.

## The decision

Do the **P0 ship-blocker cluster** from §10, in this order, and *not* the big P2 execution work
(NVPTX backend / grid-roles) which the document itself correctly defers to after release:

1. **P0-4 — warn on unverified dynamic shapes** (§9.4). Smallest; pure integrity. First, as a quick
   banked win.
1. **P0-1 — carry subspace metadata on the flat path** (§9.1). The biggest-value item and the one
   squarely in my area (I just spent the session widening `flat.rs`, which *grew* this bug's blast
   radius: `flat_used` 87→96, and the default flat path silently drops the §4.3 scratchpad-allocation
   flagship). Ship-blocker.
1. **P0-3 — use-site seam proof for declared spaces** (§9.3). The best small *topology* item; reuses
   the `pending_transfer_relaxed` machinery the built-in intrinsics already use. High demo value.

**Why this and not "topology/spawn" directly:** the audit confirmed that "topology/spawn" is not one
task — it is spread across P0-2/P0-3/P1-3/P2-1/P2-2/P2-3. The high-value, low-risk topology work
right now is these P0s; the execution model below `spawn` (P2-2) is the largest single piece and needs
the NVPTX backend (P2-1) under it first. Both are correctly post-release.

**Deferred deliberately** (documented so it isn't lost): P0-2 async transfer tokens is the pivotal
mid-size item (unlocks the perf story, shares the QF_LIA/QF_BV solver with P1-1 bounded shapes) — a
good "next big thing" but bigger than this P0 cluster and worth its own focused session. The P0-4
opt-out (`unverified` on a `MemoryDecl`, alongside `overcommit`) is left as a follow-up; the warning
itself is the ship-blocker fix.

## Progress

_(updated as each item lands; newest first)_

- **P0-4 — DONE.** Made the silent early-return in `TypeChecker::check_capacity` loud: a new **W1029**
  fires when the destination space declares a `capacity` but the tensor shape is non-literal, naming
  the first dynamic dimension (`dimension 0 is the runtime value 'n'`). No false positives on const
  generics — `check_function` skips generic templates, so only monomorphized (literal-dim) bodies reach
  the check. Multi-hop staging emits one W1029 per capacity-bounded hop (consistent with E6009/E6010,
  which check each hop). New test `tests/warnings/pass/w1029_dynamic_shape_unverified.vx`. The
  `unverified` opt-out on `MemoryDecl` (doc §9.4) is left as a follow-up; the warning is the fix.
  Files: `src/diagnostic.rs` (W1029), `src/hir/expr.rs::check_capacity`.

- **P0-1** — _next_. Port the subspace attribute block from the AST path
  ([`src/codegen/lower/tensors.rs`](../../src/codegen/lower/tensors.rs)) into the flat `Opcode::Transfer`
  arm ([`src/codegen/flat.rs:1730`](../../src/codegen/flat.rs#L1730)) so `granule/offset/slots/scope/ space/within/capacity` survive on the default path. Approach under evaluation: compute the metadata in
  the flat *lowerer* and carry it as a side table (like `tensor_types`/`agg_layouts`), matching the
  flat path's "lowerer computes, emitter emits" split; the offset assignment must match the AST bump
  allocator so both paths emit byte-identical attrs.
