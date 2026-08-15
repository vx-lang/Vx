# The memory algebra: what it is, why it exists, and what we have learned

This is the design document for Vx's model of data movement. It is written for someone who has
never seen the memory algebra before, and it is honest about the parts that turned out to be wrong,
because most of what we know came from being wrong in a measurable way.

Tracking: [vx-review#23](https://github.com/hiraditya/vx-review/issues/23) (the calibration),
[Vx#352](https://github.com/hiraditya/Vx/issues/352) (making placement real in emitted code).

______________________________________________________________________

## 1. The problem

**Model the data traffic of a heterogeneous system well enough to refuse a bad placement before you
rent the machine.**

Not "estimate performance". Refuse. If a program's working set does not fit in the memory it is
placed in, or a tensor is read from a device that cannot see it, the compiler should say so, on a
laptop, before any money is spent.

That is worth doing because data movement is where the time goes. Training runs sit around 40%
model-FLOPs utilisation; decode is far worse, in single digits, and it is worse *because* it is
bandwidth-bound rather than compute-bound. The arithmetic units are idle waiting for memory. So a
model of traffic is a model of the dominant term.

### Two constraints that make it tractable

Both are real properties of the workload, not simplifying assumptions we apologise for.

**The graph is acyclic.** Within one decode step, data flows forward. Autoregression does feed back
across steps, but what a future iteration consumes is *state* — the KV cache, the weights — and
state has a home that does not move in steady state. So the cyclic part is handled by pinning, not
by modelling a cycle. (Modelling cycles properly is not hard once the acyclic model is right; it is
just not the first problem.)

**The graph is small.** A machine graph is a dozen nodes. A transformer is a chain of ~100 layers.
Chain program graph × tree machine graph is dynamic programming in O(layers × devices²) —
microseconds. The literature reaches for ILP and heuristics because the general case (multiway cut,
N ≥ 3) is NP-hard; at the scale that actually ships, you never enter that regime. **Exact optimal
placement is affordable.** That is a stronger claim than another approximation, and it follows from
the problem being smaller than people assume.

______________________________________________________________________

## 2. Where this sits next to SYCL, Kokkos and Descend

Four things a heterogeneous system could make static. Only the first is solved in the field:

| what is typed | who has it |
| --- | --- |
| memory region **within** a device (registers, scratchpad, DRAM) | OpenCL, SYCL, Kokkos |
| **which physical device** a value lives on | nobody, in C++ |
| **where a function may run** | nobody — macros approximate it |
| **cost of moving** between them | nobody |

SYCL has address spaces in the type system, inherited from OpenCL, and Kokkos has memory spaces as
template parameters with real compile-time checking. Neither types *which* device. Two
`malloc_device` allocations on two GPUs are both `float *`, freely assignable, and reading one from
the other's queue compiles silently.

The reason is not oversight. Kokkos proves it: `SpaceAccessibility<...>::accessible` is a
`constexpr` boolean, and the specialisation for `false` — the case the compiler has *already
proven* illegal — emits a runtime `Kokkos::abort`. It cannot do better, because "where this code
runs" is not part of a C++ function's type. A library can only type what a template parameter can
carry.

Vx is a language, so it can carry it. `Pinned(T, Space)` puts placement in the type and `spawn
on(Topology::X)` makes the execution site static. The error below is the SYCL bug, refused at
compile time:

```
Cross-topology access error: 'tile' (type: Pinned(Tensor(F32,[32,16]), Custom("SMEM")))
is unreachable from GPU(0): no transfer path exists
```

The nearest relative is **Descend** (PLDI 2024) — a language, memory spaces in reference types,
dereference checked per execution context. It has rows 1 and 3 and **no capacities, no costs, no
machine files, no admission**. Vx's claim is the whole stack: rows 1–4 plus refusal.

______________________________________________________________________

## 3. Two graphs, never mixed

**The machine graph.** Nodes are memory spaces (a GPU's HBM, its SMEM, host DRAM, a peer's HBM).
Edges are physical links. Nodes carry capacity; edges carry bandwidth. This is `fleet/*.vx`, and it
exists today.

**The program graph.** Nodes are compute steps. Edges are data, weighted in bytes: this tensor,
produced here, consumed there. The compiler has this implicitly and **does not yet emit it as an
object**, which is the main thing standing between us and an optimiser.

A **placement** maps program-graph data onto machine-graph nodes. The algebra prices the movement a
placement induces; an optimiser would choose the placement. Different jobs — and the second
consumes the first as a black box.

______________________________________________________________________

## 4. Four layers

| layer | question | model | status |
| --- | --- | --- | --- |
| 0 | one edge | `α + bytes/β` | **α is not implemented**; we compute `bytes/β` |
| 1 | one route (multi-hop) | how the legs compose | mechanism landed, law unsettled |
| 2 | one placement | Σ route-cost, subject to capacity | admission works (E6009/E6010) |
| 3 | many transfers at once | contention | measured once; no term in the model |

Layer 0's missing α is not a rounding error. On an A100, `bytes/β` alone is wrong by **−98.4%** at
4 KiB — and 4 KiB is where activations and decode steps live. See §7.

______________________________________________________________________

## 5. Capability, choice, cost — three facts, not one

This is the lesson that cost the most to learn, so it gets its own section.

For any movement there are three separate facts:

1. **Capability** — what the machine *can* do. `cp.async` exists from sm_80. A property of the
   hardware. Belongs in the machine file.
2. **Choice** — what this particular movement *does*. A property of the plan.
3. **Cost** — follows from both.

We collapsed 1 and 2. `crossing: streamed` was declared on a `Memory` space, which made "this
transfer uses the copy engine" into a claim about the silicon. It was pre-registered on four
unmeasured NVIDIA parts and **falsified within hours** on a held-out A100: sum fit better than
bottleneck at every size (−71.8% against −79.3% at 4 KiB).

It failed because the premise was about capability while the measurement was about choice. Neither
the instrument nor Vx emits `cp.async` — `grep` across the lowering finds none — so every route
anyone measured was a load followed by a store. The declaration described something nothing did.

**The rule:** a machine file states capability. A plan states choice. Cost is derived. Do not let a
declaration in one place assert a fact from another.

______________________________________________________________________

## 6. Where a topology says how a movement happens

The extension point that makes this future-proof: **a topology declares not only which edges exist
and what they cost, but how a movement across them is performed — that is, what code is emitted.**

The same declaration then determines the mechanism *and* its cost, so the two cannot drift. That is
precisely the failure §5 describes, closed by construction rather than by discipline.

It also means hardware we have never seen is a new topology file rather than a compiler change. A
part with a copy engine lowers a fill one way; a part without lowers it to load-then-store; a part
with something nobody has invented yet lowers it to that. Vx does not need to have anticipated it.

There is precedent: the plugin ABI (`vx_plugin_dispatch_async`, with CoreML and MPS backends) is
already an extension point for **compute**. This is the analogue for **movement**.

The correctness contract a lowering must satisfy is written up separately in
[`custom_transfer_contract.md`](custom_transfer_contract.md) — ten constraints, four of them
mechanically checkable, one discharged by the seam verifier that already exists, and the rest
declarations the compiler then relies on. Performance is deliberately not constrained.

Open questions, and they are real:

- **How is the lowering named?** A strategy chosen from a registry the compiler ships, or something
  the machine file supplies? A registry is safer and less expressive; anything more requires the
  file to carry code, which is a much larger commitment.
- **What happens to the safety properties?** Rows 1–3 of §2 are only as sound as the emitted
  movement. An extension point that can emit anything can emit something that violates the
  placement guarantees the type system just proved.
- **Capability still has to be declarable separately**, because the compiler must be able to refuse
  a plan the machine cannot perform — before choosing a lowering for it.

______________________________________________________________________

## 7. Where the model is wrong, measured

Everything here is a residual against a real machine. The instruments and results live in
`utils/memalg/` and `vx-review/memory-algebra-paper/measurements/`.

### Layer 0: no α

Fitting `α + bytes/β` on an A100 host seam from **only the two extreme sizes** gives α = 9.79 µs,
β = 25.5 GB/s (declared 31.5). That predicts all eight held-out intermediate sizes to within ±8%,
most within ±2%. The compiler's `bytes/β` is off by −98.4% at 4 KiB and −0.0% at 1 GiB.

The error is entirely a missing constant. Adding it is not free: route selection currently
minimises per-byte cost, which is only valid because every edge is a line through the origin. With
α the cheapest route depends on transfer size. (vx-review#28)

### Layer 1: the composition law is unsettled

Summing the legs over-predicts by **+66.9%** on the H100's `HBM->L2->SMEM`, and under-predicts by
−9.3% on `L1->REG->SMEM`. That looked like two regimes selected by mechanism. It is now doubtful:
the one route that fit "slowest leg" has a leg (`HBM->L2`) that we later established is
**measuring the wrong thing**, and no probe anywhere used a copy engine. Everything cleanly measured
is consistent with legs adding.

What survives: a **register staging point always adds**, because a load followed by a store cannot
overlap — the second instruction needs the first's value. That holds on both vendors and is
structural rather than fitted.

### Layer 2: capacity boundaries are slightly wrong

On an A100: declared capacity is 0.9% larger than the device reports, and the largest single
allocation is 1.16 GiB smaller than declared. The allocation granule is **2 MiB** and no fleet file
declares one.

A pre-registered prediction that *did* hold: the machine files declare SMEM at the per-SM figure
(167,936 B) while a block can only opt into 166,912 B — a window exactly **one granule** wide where
the model admits what the hardware refuses.

### Layer 3: contention is a queue, not a division

On an M4, across 1 to 64 co-resident streams, aggregate bandwidth is flat at 105.5–106.2 GB/s and
**no individual flow degrades**. The obvious patch — `bandwidth ÷ users` — predicts all K flows
completing together at `K·N/β`. Measured, flow 1 completes at `N/β` and flow k at `(k+1)·N/β`.
They queue.

All three models agree on makespan and disagree completely on when any particular tensor arrives —
which is the number a placement needs.

### The instruments were wrong more often than the model

**Five defects, all the same shape**: a row measured a cache instead of the memory its label named.
Launch overhead swamping the on-die seams; grid-stride re-reads served from L1; `HBM->L2` measuring
an L2-resident read; four store edges pinned at the store-issue port; six of ten host-seam points
reporting traffic *above* the hardware peak.

Every one was caught by the same check: **a number that cannot physically be true.** A composite
cannot beat its own slowest leg. Nothing leaving DRAM can exceed DRAM bandwidth.

This is a result, not an embarrassment. Measuring the machine honestly is harder than modelling it,
and a large share of what looked like model error was instrument error. Any model calibrated
without that check is calibrated against noise.

______________________________________________________________________

## 8. Provenance: three classes, not two

Every figure in a machine file must say where it came from, because the audit depends on being able
to be wrong:

- **`spec:`** — from a vendor document. A claim. Scored.
- **`measured:`** — from an instrument, with a log reference. A claim. Scored.
- **`policy:`** — chosen to steer routing. **Not a claim about the machine.** Excluded from every
  error table and every frozen cell.

The third is needed because costs are also the natural knob for influencing which route the
compiler picks. That is legitimate, but the moment a number is chosen to produce an outcome it
stops being falsifiable, and nothing in the file distinguishes it from one that describes reality.
Without the third class, six months of calibration can turn out to have been scoring numbers
somebody tuned.

______________________________________________________________________

## 9. Predict first, measure second

Predictions are frozen (`memalg-freeze-2026-08-07`) before any hardware is rented, harvested by
`utils/memalg/freeze.py` straight from `--diagnostics-json` so no number is hand-copied. Changing a
frozen cell needs a dated note saying what moved and why.

It works. The `crossing: streamed` prediction was committed, scored on a held-out SKU hours later,
falsified, and reverted — the whole cycle in a day, and much cheaper than finding it in a paper.

Two disciplines worth keeping:

- **Re-scoring against the data that selected a rule is not evidence.** It is a sanity check on
  training data and the note should say so.
- **A held-out SKU is the test.** Seven of eight fleet parts have never been measured. That is the
  asset, and it is spent the moment a rule is tuned against them.

______________________________________________________________________

## 10. What is open

Roughly in dependency order:

1. **Emit the program graph** — the byte-weighted DAG of what moves, as an object. Without it, a
   plan cannot be named, priced or compared, and "optimisation" stays hand-waving. This is now the
   load-bearing item.
2. **Make SMEM placement real** (Vx#352). Today a tile placed in SMEM is checked and priced but
   never becomes code — the emitted kernel has no shared memory at all. Half the algebra prices
   transfers that do not happen.
3. **Add α** (vx-review#28), and rework route selection for size-dependent choice.
4. **Settle the composition law** once a copy engine is actually exercised.
5. **A contention term**, if the queue result holds on more than one part.
6. **The transfer-lowering extension point** (§6), which is what makes 4 and 5 answerable per
   machine rather than globally.

An optimiser over the program graph (min-cut for two devices, DP for a chain) is future work. It
consumes a truthful graph; it does not belong in the work that establishes truthfulness.
