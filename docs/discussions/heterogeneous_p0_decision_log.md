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

- **P0-1 — DONE.** The flat `Opcode::Transfer` arm now re-attaches the full scheduling attribute set
  (`space`/`within`/`granule`/`capacity`/`scope` + a bump-allocated `offset`/`slots`), **byte-identical
  to the AST path** (verified against `--legacy-codegen` on `subspace_schedule.vx`: two 128×128 f32
  tiles into SMEM land at `offset 0` then `offset 65536`, 4 slots each).

  Final approach (settled on emitter-side, the doc's §9.1 suggestion, rather than the lowerer-side
  side-table I'd first sketched): the frozen registry carries no memory decls, so `build_flat_module`
  builds a `SubspaceInfo` per declared space from the per-compilation env, keyed by dispatch id (the
  identity a `Transfer` carries in its `imm`), and threads them into `emit_module_mlir` →
  `EmitCtx.subspaces`. The emitter resolves `ins.imm` → descriptor, sizes the tile from `ctx.tensors`
  via a new `static_tile_bytes` (the flat analogue of `static_tensor_bytes`), and bump-allocates with a
  per-function `subspace_offsets` map mirroring `MeliorGenerator::subspace_offsets`. MLIR sorts
  attributes on print, so matching the *set* yields identical output regardless of emission order.

  Files: `src/codegen/flat.rs` (`SubspaceInfo`, `EmitCtx.subspaces`, `static_tile_bytes`, the Transfer
  arm, `emit_module_mlir` signature), `src/driver.rs` (build + thread the descriptors),
  `tests/integration_test/flat_codegen_differential.rs` (`flat_module_mlir` helper +
  `flat_carries_subspace_scheduling_metadata` test). Corpus sweep unchanged (flat_used 96, 0 new
  miscompiles — the attrs are runtime-inert metadata); full suite green.

  Note: the middle_end `subspace_*.vx` tests run through `MeliorGenerator` (the AST path) in the
  harness, so they never exercised the flat path — hence the dedicated flat-path test above rather than
  relying on those. A pre-existing, unrelated diff remains: the flat path emits a *static* memref
  (`memref<128x128xf32>`) where legacy emits dynamic (`memref<?x?xf32>`); the scheduling attrs — the
  subject of P0-1 — are identical.

- **P0-3 — DONE.** A transfer over a *declared* `relaxed` edge now gets the same per-buffer use-site
  proof as the `*_relaxed` intrinsics: `E6004` + a z3 counterexample under `--verify-seams`, while
  `W1027` stays as the always-on declaration-time smell. Added `TransferCostGraph::is_relaxed_edge(from, to)` (scans the topology descriptors' `TransferEdge`s for a `!sync` edge — the cost graph's
  `transfer_edges` drop the sync flag); `check_transfer_expr` ORs it into the `relaxed` flag passed to
  `run_seam_hop`, so a relaxed hop anywhere in a staged multi-hop route taints that hop (each single hop
  re-enters the check). No new syntax. New test `tests/frontend/fail/topology_declared_relaxed_seam.vx`
  (`// VERIFY-SEAMS`, expects E6004). Files: `src/arch.rs`, `src/hir/expr.rs::check_transfer_expr`. Full
  suite green.

## Status: P0 ship-blocker cluster complete

All four P0s from §10 landed and committed, each green through the full pre-commit suite: P0-4
(`5e10637a`), P0-1 (`fa056b4e`), P0-3 (this commit). B2/P0-3 and B1/P0-1 and B3/P0-4 from the gap
analysis are closed. **P0-2 (async transfer tokens) is the remaining P0 and is deliberately left for a
focused session** — it is the pivotal mid-size item (a linear `Token<T, Mem>` from `transfer_async`,
`wait` consuming it, the seam engine proving the wait wasn't skipped), unlocks the perf story, and
shares the solver with P1-1 bounded shapes; it warrants its own runway rather than being rushed at the
tail of this one.

Reasonable next steps, in rough priority: **P0-2** (async tokens — the last P0), then **P1-5** (the
`mlir!` escape hatch — cheapest item by unblocked surface area, design already written), or **P1-4a**
(collapse `ElementType` to a descriptor table — a pure refactor that could ship in P0 and makes every
later numeric format one row instead of ten edits). The audit confirmed all three are real and
well-scoped.
