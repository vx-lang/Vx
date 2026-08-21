# Memory Crossing Implementation Plan: how a multi-hop transfer is priced

> **Read [`memory_algebra.md`](memory_algebra.md) first** for what the memory algebra is for and
> where this fits. In particular §5 (capability / choice / cost) explains why the `crossing:`
> declarations this plan's Phase 2 proposed were pre-registered, measured on an A100, and
> **falsified** — the mechanism below stays, the fleet declarations do not.
>
> The route-kind law in §1 is also weaker than it looks: the single route that fit "slowest leg"
> has a leg that later turned out to be measuring the wrong thing, and no probe anywhere exercised
> a copy engine. Treat the two-regime claim as open.

This plan covers `crossing:`, a new field on a `Memory` declaration, and `composition`, a new field
in the `--diagnostics-json` record. Together they fix a case where the compiler's predicted transfer
cost was wrong by about 67% (measured against real hardware, on the three-step
`HBM->L2->SMEM` route on an H100 — see §1 for where that number comes from, and §2 for
why the smaller worked example gives a different figure).

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
1. Take the **slower** of the two steps and ignore the other.

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
real numbers. The "error" column is each rule's predicted cost compared against the **measured**
cost — not one rule compared against the other:

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

### Where the attribute is read from — and a flaw found in review

Phase 1 reads `crossing:` from the **destination** space, on the reasoning that what matters is how
data gets *in*: an H100's SMEM can be filled by `cp.async`, an Apple GPU's threadgroup memory cannot
because no such engine exists there. A test pins that down —
`crossing_is_read_from_the_destination_not_the_source` checks that marking SMEM as `streamed` does
not change the price of a walk that merely *starts* at SMEM.

**That reasoning is not sufficient, and the table in §1 is its own counterexample.** Look at the
first two rows: `HBM->L2->SMEM` streams, `L1->REG->SMEM` sequences — and **both end at SMEM**. One
attribute read from the final destination and applied to the whole walk cannot give two different
answers for two routes that share a destination.

Nothing is broken today, because no machine file declares `REG` or `L1`, so the compiler never
prices the second route. But it is a trap waiting for Phase 2: M1's vertical calibration
(SMEM→registers) needs those spaces declared, and at that point marking H100 SMEM as `streamed`
would flip the load/store route to "slower step" and make it wrong by about 50%.

### The fix: a register round-trip always adds

The evidence already contains the rule. Look at what separates the four measured routes: **whether
the walk passes through registers.**

| route | through REG? | rule that fits |
| ----- | ------------ | -------------- |
| `HBM->L2->SMEM` (H100) | no | slower step |
| `L1->REG->SMEM` (H100) | yes | add |
| `L2->REG->SMEM` (M4) | yes | add |
| `HBM->REG->SMEM` (M4) | yes | add |

A register round-trip *is* a load followed by a store. There is no mechanism by which it could
overlap, because the second instruction needs the value the first produced. So:

> **If a route passes through a register-class space, its steps add — always, whatever any
> `crossing:` says. `crossing: streamed` only decides routes that bypass registers.**

This is a structural property, not a fitted parameter, and it explains all four rows.

**Implemented** (see §3.3). The language already had the vocabulary to say "this space is
registers": `scope: thread`, since a register file is private to one thread. `derived_transfer_cost`
checks for one on the path and forces addition. No new syntax, no new declaration.

One refinement the data forced: the test is **strictly between the endpoints**, not "anywhere on the
path". A route that *ends* at a register is a load, and the hardware does stream that — on an H100
`HBM->REG` measures 18.8 B/cyc, close to its slowest leg's 17.7 and nowhere near the 10.1 that
adding the legs would predict. Only a register the data is loaded into and then stored back out of
is a staging point.

The general version — a `crossing:` per hop, with a walk that streams one step and sequences the
next — is in §6 as an open question. The register rule is the smallest version the current data
supports, and it should not be widened past the evidence.

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

**Do not check the headline against this example.** The two are different measurements of different
things, and they give different numbers on purpose:

- **35.9%** is what this table shows: `sequenced` against `streamed`, two *predictions* for the same
  two-step route, one model against another.
- **67%** is the figure in the summary: `sequenced` against what the **hardware actually did**, on
  the three-step `HBM->L2->SMEM` route, which has an extra term this example does not.

A reader who divides 15,671,711 by 11,534,337, gets 1.36, and concludes the headline is inflated has
compared a model-versus-model ratio with a model-versus-hardware error.

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
- **The destination-only rule alone is not sufficient** — see §2 — so it is now guarded by §3.3.

### 3.3 The register rule (DONE, this commit)

`derived_transfer_cost` now checks whether any space **strictly between the endpoints** declares
`scope: thread`. If one does, the terms are added regardless of what `crossing:` says.

Two details that are not arbitrary:

- **Endpoints are excluded**, because a route ending at a register is a load, which streams. The
  H100 numbers above are the evidence.
- **Membership, not position.** `route_spaces` emits the source chain, then the destination chain,
  then possibly the common ancestor, so the vector is not in traversal order and "the middle one"
  is not a well-defined index. Testing identity against the two endpoints is order-independent and
  is what a staging point actually means.

It changes nothing today: no fleet file declares `scope: thread`, so the guard never fires, and
regenerating the 240 frozen cells against the run from just before it gives **0 moved**. A test
asserts that no fleet-style hierarchy declares a thread-scoped space, so if one ever does, the
frozen cells get re-checked rather than silently moving.

### 3.4 Diagnostics

- `src/hir/env.rs`: added `composition` to `StagingRoute`.
- `src/hir/check/transfer.rs`: fills it in, but only for containment routes — a declared host link
  has one step and composes nothing.
- `src/diagnostics_json.rs`: serialises it as `"sum"` / `"bottleneck"` / `null`.

### 3.5 The default is `sequenced`, and this is the important part

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

### 3.6 One honest caveat about the saved predictions

The JSON record gained a field, so it is no longer *byte-identical* to the saved files, even though
every value is the same. The check for this is stronger than a plain diff: after removing the new
`composition` field from the regenerated records, all 240 compare **equal** to the pre-change run.
Zero values moved; the record grew a column.

Anyone re-verifying the frozen predictions needs to know this, because the verification is a byte
comparison.

### 3.7 Tests

In `src/hir/memory.rs`:

- `a_register_in_the_middle_forces_addition` — a staging register defeats `crossing: streamed`.
- `a_register_at_an_endpoint_does_not_force_addition` — a load still streams.
- `the_register_rule_does_not_disturb_a_walk_without_registers` — the guard is inert where it does
  not apply, which is what stops it silently undoing `crossing:` everywhere.
- `no_fleet_style_hierarchy_declares_a_thread_scoped_space` — records *why* the rule moves nothing
  today, and fails loudly if that stops being true.
- `sequenced_is_the_default_and_sums` — the default is unchanged behaviour.
- `streamed_takes_the_slowest_leg_alone` — 128 cycles, not 160.
- `crossing_is_read_from_the_destination_not_the_source` — a walk *out of* a streamed space still
  adds.
- `streamed_applies_across_a_multi_hop_walk` — three spaces, 224 vs 128.

In `src/diagnostics_json.rs`: `accept_record_carries_route_and_edge_costs` now also asserts the
`composition` field is present.

Full suite: 459 library tests, 197 integration tests. `vx-format` leaves `crossing:` unchanged.

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

0. ~~**Settle the register rule from §2 first.**~~ **Done** — see §3.3. Declaring `streamed` on
   H100 SMEM is now safe from the flaw the review found: even once registers are declared, a walk
   that stages through them adds regardless of the attribute.
1. Decide whether to re-take the freeze or to record a dated exception. This is a human decision,
   not a code change. Tracked at
   [vx-review#27](https://github.com/hiraditya/vx-review/issues/27).
1. Edit each machine file, adding a `spec:` comment naming the evidence for the choice, the same way
   every other figure in those files carries its source. The comment must say whether the choice was
   **measured** on that part or **derived from the ISA** — they are different kinds of claim and only
   one of them is evidence.
1. Regenerate the saved predictions and record exactly which cells moved and by how much. Keep the
   before/after table.
1. Write the dated note the freeze protocol requires — and see the honesty note below about what it
   has to admit.

### What the dated note has to admit

Re-running `compare_m1.py` on the H100 to check that `streamed` improves the residual is **scoring
against the data that selected the rule**. It is training data. It is worth doing as a sanity check,
but it is not evidence, and the note should say so in plain words rather than presenting a smaller
residual as a result.

The genuinely pre-registered move is the other files. Nobody has measured an A100, H200 or B200 in
this campaign, so `crossing: streamed` on those parts is a **prediction derived from the ISA**
(`cp.async` exists from sm_80). Written down and dated now, it is scored when those parts are
eventually measured — a real prediction with a real way to be wrong. That converts what would
otherwise be a freeze exception into a new pre-registration.

### Verification

- Re-run `utils/memalg/compare_m1.py` against the H100 measurements, understanding it as a sanity
  check on training data, not as a result. If the `L2->SMEM` residual gets *worse*, something is
  wrong with the mechanism itself, because the rule was chosen from this exact data.
- The Apple numbers should not move at all, because that file stays `sequenced`. This is the real
  check in this phase: an unchanged file whose predictions move means the edit leaked.

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

### A middle path worth recording

There is something useful short of a transferable constant. M1 sweeps sizes from 4 KiB to 1 GiB, so
on the *one* copy-engine route we have, `bytes/bw + fill` can be fitted **per route** and then
checked on sizes held out of the fit. That does not give a `fill` that transfers to another edge,
but it does give a **bound on how large fill can be** — and a bound is a real result where a fitted
constant would not be.

It also interacts with α (see below): both are fixed per-transfer costs, and a fit that does not
separate them will silently roll the edge's startup latency into the route's fill term.

______________________________________________________________________

## 6. Hardening, and open questions

### Hardening worth doing alongside Phase 2

- **Refuse `crossing:` where it cannot mean anything.** A space that is never the destination of a
  containment hop can carry the attribute today and it will silently do nothing. Rejecting it is the
  same "refuse what you do not understand" rule as E6012 and E6014.
- **Make the composite guard permanent.** `compose.py` already refuses to score a route whose
  measured cost beats its own slowest leg, because a composite cannot outrun a link it crosses. That
  check belongs in `compare_m1.py` too — it is what catches a mislabelled seam, and it found the
  `HBM->L2` defect in vx-review#22 without being told to look.

### Open questions

- **α is missing, and it is the neighbour of `fill`.** The cost model has no fixed per-transfer term
  at all: every edge is `bytes/bandwidth`, a line through the origin. Phase 3's `fill` and a
  per-edge α are the same shape of quantity, and adding either without the other will absorb one
  into the other. Tracked at
  [vx-review#28](https://github.com/hiraditya/vx-review/issues/28).
- **AMD.** `fleet/mi300x.vx` describes a part nobody has measured. AMD GPUs have `buffer_load_dword lds`, which is a copy-engine-like path, so `streamed` may well be correct there — but nothing in
  this campaign has tested it.
- **Mixed routes.** A walk currently gets one rule for the whole thing. The register rule in §2 is
  the first crack in that: it says one *kind* of hop always adds regardless. A machine that streams
  one step and sequences the next, with neither being a register hop, would need a per-hop
  `crossing:` and a composition rule over mixed hops. No such machine has appeared, so this stays
  recorded rather than built.
- **Is the destination the right place to ask at all?** With the register rule doing the
  discriminating, `crossing:` carries less weight than Phase 1 assumed. If registers were declared
  as spaces, the two machines would have genuinely different *routes* — Apple's threadgroup memory
  reachable only via a register hop, the H100's reachable directly — and "through REG ⇒ add" would
  fall out of the topology with no attribute at all. `crossing:` would still be needed for the
  remaining question (do the non-register hops overlap), but the design is worth revisiting once
  registers are declared.
