# Memory Crossing Implementation Plan: how a multi-hop transfer is priced

This plan covers `crossing:`, a new field on a `Memory` declaration, and `composition`, a new field
in the `--diagnostics-json` record. Together they fix a case where the compiler's predicted transfer
cost was wrong by about 67%.

Phase 1 is **done** (commit `3cc0c88f`). Phases 2 and 3 are not started, and Phase 2 needs a
decision from a human because it changes numbers we have promised not to change quietly.

Tracking issue: [vx-review#26](https://github.com/hiraditya/vx-review/issues/26).

______________________________________________________________________

## 1. The problem

When a program moves a tile from HBM into SMEM, the data passes through L2 on the way. That is two
steps:

```
HBM  ──step 1──>  L2  ──step 2──>  SMEM
```

How long does the whole trip take? There are only two sensible answers:

1. **Add** the two step times together.
2. Take the **slower** of the two steps and ignore the other.

They differ by roughly 2x, so picking the wrong one is not a rounding error. Until this change the
compiler always added.

### Why "add" is sometimes the right answer

Some GPUs move data using ordinary instructions: *load from L2 into a register*, then *store from
the register into SMEM*. Those two instructions really do happen one after the other. The thread
issues the load, waits for it, then issues the store.

Think of carrying boxes by hand. You walk to the truck, pick up a box, walk to the shelf, put it
down. The two walks add up.

### Why "take the slower step" is sometimes the right answer

Other GPUs have a copy engine — on an NVIDIA H100 this is the `cp.async` instruction — that streams
data straight from global memory into SMEM without ever stopping in a register. That is a pipeline.
While byte 1 is being written into SMEM, byte 2 is already being read out of L2. The steps overlap.

Think of a bucket brigade. Everyone in the line works at the same time, and the line moves at the
speed of the slowest person. Adding up each person's time would give a wildly wrong answer.

### What the hardware actually does

We measured both, on two different machines. `utils/memalg/compose.py` scores the two rules against
real numbers:

| machine | route | how the hardware moves it | rule that fits | error |
| ------- | ----- | ------------------------- | -------------- | ----- |
| H100 | `HBM->L2->SMEM` | copy engine (`cp.async`) | slower step | -13.9% |
| H100 | `L1->REG->SMEM` | load then store | add | -9.3% |
| M4 | `L2->REG->SMEM` | load then store | add | +13.2% |
| M4 | `HBM->REG->SMEM` | load then store | add | +2.7% |

Four routes, two vendors, two instruction sets. **The right rule depends on how the hardware moves
the data, not on which machine it is.** Apple's family 9 GPUs have no copy engine at all, so every
route there is load-then-store.

### Why the compiler could not get this right before

A machine file says `within: Memory::L2`, which means "SMEM lives inside L2". That tells the
compiler the two spaces are nested. It does **not** tell the compiler whether crossing that boundary
is a hardware copy engine or a pair of instructions.

So no single rule could be right:

- keep adding → about 67% too slow on copy-engine routes
- switch to "slower step" → about 49% too fast on load-then-store routes

Same mistake, pointed in opposite directions. The fix is to let the machine file say which one it
is.

______________________________________________________________________

## 2. The design

### `crossing:` — an input, written by a human

A new optional field on a `Memory` declaration:

```
Memory SMEM {
  within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc, crossing: streamed
}
```

- `crossing: sequenced` — the hardware uses instructions, one after another. **Add** the steps.
- `crossing: streamed` — the hardware has a copy engine that pipelines. **Take the slowest step.**

### Why it goes on the destination space

What matters is how data gets **in**. An H100's SMEM can be filled by `cp.async`. An Apple GPU's
threadgroup memory cannot, because no such engine exists there, so it is always load-then-store.
This is a fact about the receiving end.

A test pins this down: `crossing_is_read_from_the_destination_not_the_source` checks that marking
SMEM as `streamed` does **not** change the price of a walk that merely *starts* at SMEM.

### `composition` — an output, written by the compiler

The `--diagnostics-json` record now reports which rule was used:

```json
{"path": ["L2", "SMEM"], "derived_cost": 11534337, "composition": "bottleneck"}
```

It is `"sum"`, `"bottleneck"`, or `null`. It is `null` for a host link, which is a single step and
therefore composes nothing.

This exists because we save predictions to files and score them against real measurements weeks
later. If a saved number does not say whether it came from adding or from taking the maximum, it
cannot be checked afterwards — and the two answers differ by about 2x.

### Why the two fields use different words

The input says `streamed` / `sequenced`, which describes **the hardware**. The output says
`bottleneck` / `sum`, which describes **the arithmetic that followed from it**. Keeping the two
vocabularies separate means a reader of the JSON is told what was computed, not asked to re-derive
it from a hardware claim.

### Worked example: H100, 1 MiB from L2 into SMEM

| | picoseconds |
| --- | --- |
| SMEM's own speed costs | 4,137,374 |
| L2's speed costs (after splitting across its 132 SMs) | 11,534,337 |
| **`sequenced`** — add them | **15,671,711** |
| **`streamed`** — take the slower | **11,534,337** |

The `streamed` answer is just L2's term, because L2 is the slower of the two.

______________________________________________________________________

## 3. Phase 1 — the mechanism (DONE, commit `3cc0c88f`)

### 3.1 Front end

- `src/syntax/decl.rs`: added the `Crossing` enum (`Sequenced`, `Streamed`) and a `crossing` field
  on `MemoryDecl`.
- `src/parser/decl.rs`: parse `crossing: streamed` and `crossing: sequenced`. Any other word is a
  parse error naming the two valid choices.

### 3.2 Cost model

- `src/hir/memory.rs`: `derived_transfer_cost` reads the destination space's `crossing` and then
  combines its per-step terms with either `+` (sequenced) or `max` (streamed). Both the
  cycle-denominated path and the picosecond path go through the same rule.

### 3.3 Diagnostics

- `src/hir/env.rs`: added `composition` to `StagingRoute`.
- `src/hir/check/transfer.rs`: fills it in, but only for containment routes — a declared host link
  has one step and composes nothing.
- `src/diagnostics_json.rs`: serialises it as `"sum"` / `"bottleneck"` / `null`.

### 3.4 The default is `sequenced`, and this is the important part

`sequenced` is exactly what the compiler did before this change. So a machine file that says nothing
about `crossing:` gets exactly its old answer.

This was checked rather than assumed. All 240 saved predictions were regenerated and compared
against a regeneration from just before the change:

```
cells compared          : 240
predictions that moved  :   0
```

Getting the new behaviour requires editing a machine file on purpose, so the change appears in a
diff instead of quietly moving every number at once.

### 3.5 One honest caveat about the saved predictions

The JSON record gained a field, so it is no longer *byte-identical* to the saved files, even though
every value is the same. The check for this is stronger than a plain diff: after removing the new
`composition` field from the regenerated records, all 240 compare **equal** to the pre-change run.
Zero values moved; the record grew a column.

Anyone re-verifying the frozen predictions needs to know this, because the verification is a byte
comparison.

### 3.6 Tests

In `src/hir/memory.rs`:

- `sequenced_is_the_default_and_sums` — the default is unchanged behaviour.
- `streamed_takes_the_slowest_leg_alone` — 128 cycles, not 160.
- `crossing_is_read_from_the_destination_not_the_source` — a walk *out of* a streamed space still
  adds.
- `streamed_applies_across_a_multi_hop_walk` — three spaces, 224 vs 128.

In `src/diagnostics_json.rs`: `accept_record_carries_route_and_edge_costs` now also asserts the
`composition` field is present.

Full suite: 455 library tests, 197 integration tests. `vx-format` leaves `crossing:` unchanged.

______________________________________________________________________

## 4. Phase 2 — declare it on the real machine files (NOT STARTED, needs a decision)

No file in `fleet/` was changed in Phase 1. That was deliberate: changing one moves saved
predictions, and those are under a freeze protocol that exists so a model cannot be quietly tuned
after seeing the measurements.

### What should change, on current evidence

- `fleet/h100-sxm.vx` — SMEM should get `crossing: streamed` (it has `cp.async`).
- `fleet/a100-40.vx`, `fleet/a100-80.vx` — same, `cp.async` exists from sm_80 onward.
- `fleet/h200.vx`, `fleet/b200.vx`, `fleet/node-*.vx` — same family, same engine.
- `fleet/m4-uma.vx` — stays `sequenced`. Apple family 9 has no such engine, and we measured this.
- `fleet/mi300x.vx` — **unknown**. Nobody has measured an AMD part in this campaign. Leaving it
  `sequenced` is the honest default because that is what the file already implied.

### Steps

1. Decide whether to re-take the freeze or to record a dated exception. This is a human decision,
   not a code change.
2. Edit each machine file, adding a `spec:` comment naming the evidence for the choice, the same way
   every other figure in those files carries its source.
3. Regenerate the saved predictions and record exactly which cells moved and by how much.
4. Write the dated note the freeze protocol requires, explaining that the change came from a
   measurement rather than from fitting.

### Verification

- Re-run `utils/memalg/compare_m1.py` against the H100 measurements. The `L2->SMEM` residual should
  improve; if it gets worse, the `streamed` claim for that part is wrong and should be reverted.
- The Apple numbers should not move at all, because that file stays `sequenced`.

______________________________________________________________________

## 5. Phase 3 — the pipeline fill term (NOT STARTED, blocked on data)

"Take the slowest step" under-predicts the H100's copy-engine route by 14%. That is the direction a
pipeline model *should* be wrong in: a bucket brigade still takes a moment to fill up before it
reaches full speed, and the current rule ignores that.

The likely refinement is:

```
cost = bytes / (speed of slowest step) + fill
```

where `fill` is a small fixed cost per route.

**This is blocked, and should stay blocked.** Calibrating `fill` needs at least two copy-engine
routes to measure against, and the fleet currently provides exactly one: every Apple route is
load-then-store, and the H100 has a single scorable copy-engine composite. Fitting a constant to one
data point would produce a number that matches that point and predicts nothing, which is the failure
mode the whole measurement campaign is set up to avoid.

Unblocking it needs either a second copy-engine route on the H100, or a powerset run on another
NVIDIA part.

______________________________________________________________________

## 6. Open questions

- **AMD.** `fleet/mi300x.vx` describes a part nobody has measured. AMD GPUs have `buffer_load_dword
  lds`, which is a copy-engine-like path, so `streamed` may well be correct there — but nothing in
  this campaign has tested it.
- **Is the destination always the right place to ask?** It is right for every case measured so far.
  A machine where the *source* gates the transfer would break this, and none has turned up yet.
- **Mixed routes.** A three-space walk currently gets one rule for the whole thing, taken from the
  final destination. If a real machine streams one step and sequences the next, the model cannot say
  so. No such machine has appeared, so this is recorded rather than fixed.
