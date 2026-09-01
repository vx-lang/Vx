# The memory algebra: what it is, why it exists, and what we have learned

This is the design document for Vx's model of data movement. It is written for someone who has
never seen the memory algebra before, and it is honest about the parts that turned out to be wrong,
because most of what we know came from being wrong in a measurable way.

Tracking: [vx-review#23](https://github.com/hiraditya/vx-review/issues/23) (the calibration),
[Vx#352](https://github.com/hiraditya/Vx/issues/352) (making placement real in emitted code),
[Vx#353](https://github.com/hiraditya/Vx/issues/353) (the transfer-lowering extension point and
derived traffic).

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

Vx is a language, so it can carry it. `Pinned(T, Space)` puts placement in the type and `spawn on(Topology::X)` makes the execution site static. The error below is the SYCL bug, refused at
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
produced here, consumed there. The compiler now emits the **weights** — `--diagnostics-json`
carries derived traffic per transfer route and per `spawn` region, split read from written per
space and broken down per buffer (§6). It does **not yet emit the graph as an object**: the
records say how many bytes each site moves, not which site's bytes another site consumes. That
missing structure is the main thing standing between us and an optimiser.

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
1. **Choice** — what this particular movement *does*. A property of the plan.
1. **Cost** — follows from both.

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

The extension point that makes this future-proof: **a topology declares which edges exist, the
bandwidths at their endpoints, and how a movement across them is performed — that is, what code is
emitted.**

Traffic — and hence cost — is then *derived* from that code, so mechanism and cost cannot drift.
That is precisely the failure §5 describes, closed by construction rather than by discipline. (An
edge stops needing a declared cost at all; the bandwidths are properties of the machine, which is
what a machine file was always for.)

It also means hardware we have never seen is a new topology file rather than a compiler change. A
part with a copy engine lowers a fill one way; a part without lowers it to load-then-store; a part
with something nobody has invented yet lowers it to that. Vx does not need to have anticipated it.

There is precedent: the plugin ABI (`vx_plugin_dispatch_async`, with CoreML and MPS backends) is
already an extension point for **compute**. This is the analogue for **movement**.

The correctness contract a lowering must satisfy is written up separately in
[`custom_transfer_contract.md`](custom_transfer_contract.md) — ten constraints: five read off the
lowering body (it is Vx code our own front end compiles), one discharged by the seam verifier that
already exists, one remaining declaration (runtime failure modes), a conformance suite for the
two properties only the hardware can witness, and one that dissolves (cost honesty: traffic is
derived from the body, so the cost cannot disagree with the code). Performance is deliberately not
constrained.

Three questions were open here. The contract document settles them in design, and they are now
**built** (Vx#353, stages A1–A4):

- **How is the lowering named?** The file carries Vx code — `impl Transfer<Memory::A, Memory::B> for Topology::X`
  — compiled and checked by our own front end like any other Vx. The trust boundary is the small
  primitive set the body may use (`raw::load/store/barrier/...`), not the file.
- **What happens to the safety properties?** They survive because the body cannot be opaque: it is
  composed of indexed primitives whose bounds and space obligations the compiler discharges, so an
  `impl Transfer` cannot emit a movement that violates the placement guarantees the type system
  proved. This holds by the contract's construction, not by trusting authors.
- **Capability is declarable separately, and it gates the primitives.** `raw::async_copy` is a
  compile error in a lowering for a part whose machine file declares no copy engine — the
  capability/choice split of §5, enforced.

### What is built

- **A1–A2**: the `impl Transfer<A, B> for Topology::X` declaration, and the `raw::` primitive
  floor with its obligations discharged against z3 (`src/hir/check/raw.rs`).
- **A3**: a user lowering *moves the bytes*. The witness is structural rather than numeric — a
  transfer is a move, not a conversion, so the computed answer cannot tell two lowerings apart.
  The corpus fixture says `raw::barrier()` twice and its device image carries **two `bar.sync`**
  where the builtin's carries one. Emission is sm-scoped destinations only for now; every other
  edge falls back to the builtin, which now carries the barrier it was missing.
- **A4**: traffic *derived* from code, per space, reaching a consumer through
  `--diagnostics-json` — both per transfer route and per `spawn` region, the latter broken down
  per buffer. This is where this section's central claim — cost derived, not declared — stops
  being a plan.

### One word for three questions, now split

`Transfer` used to name the reachability predicate *and* the implicit-movement trait, while the
edge lowering went by the lowercase keyword `transfer` and was keyed on the edge alone. That last
part is why the fleet directory this whole design exists for was unbuildable: an Ampere part and a
Hopper part both declare `Memory::L2 -> Memory::SMEM`, and only one of them could implement it.
The names are now split by question:

| spelling | question | keyed on |
| --- | --- | --- |
| `where Reachable<S, D>` | *may* these two topologies exchange data at all? | a pair of topologies |
| `impl Relocatable for T` | may a value of this type move implicitly across a boundary? | a user type |
| `impl Transfer<Memory::A, Memory::B> for Topology::X` | *how* does this machine move bytes across this edge? | (from, to, **machine**) |

The machine in that third key is load-bearing in both directions. Without it a lowering could not
name the part it was written for; with it, two obligations become checkable that had to be skipped
before — that the named topology exists in this compilation, and that it declares the edge being
implemented. It also stops one machine borrowing another's body, and one machine's declared copy
engine arming every machine's lowerings. Both of those were live and are described in the contract
document.

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

### The same check, pointed at the compiler

Once traffic is *derived* rather than declared (§6), the compiler becomes an instrument, and it
inherits the failure mode. A review of everything A1–A4 emitted found three ways a placed tensor
could slip past the region counter — a dereference, a placed field of an unplaced struct, a
call-returned binding — and each published `traffic: [], exact: true`. **An exact zero over a
kernel that demonstrably reads device memory is a number that cannot be true**, and it is worse
than a missing one: absent says so, while zero is indistinguishable in a harvested record from a
measurement of nothing.

The rule that falls out is the one the machine-file provenance classes of §8 already encode, now
applied to derived figures: anything the counter cannot weigh exactly produces **no traffic at
all, with a reason recorded beside it**. Never a guess, and never a zero standing in for one.

The same review found three miscompiles on the emission side, all reachable from `vxc` on programs
the checker accepted with zero diagnostics. Details are in
[`custom_transfer_contract.md`](custom_transfer_contract.md); the backlog it left open is
[Vx#358](https://github.com/hiraditya/Vx/issues/358). The reason it found what the test suite did
not is that it ran against a different bar: not *does this pass*, but **is there a test that goes
red if this behaviour is disabled**. Several guards had none, and one had been hard-disabled in the
working tree while the suite stayed green.

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

Two items this document listed as open have landed, both of them load-bearing:

- **SMEM placement is real.** A tile placed in SMEM used to be checked and priced but never became
  code, so half the algebra priced transfers that did not happen. An sm-scoped placement now
  lowers to genuine `.shared` storage, pinned by a `bar.sync` count in the device image rather
  than by a comment. Vx#352 stays open for the rest of its scope — the four things blocking
  `cp.async`.
- **The transfer-lowering extension point exists** (§6, Vx#353 A1–A4) — declaration, primitive
  floor, emission, and derived traffic.

Roughly in dependency order, what is left:

1. **Emit the program graph** — the byte-weighted DAG of what moves, as an object. This is the
   remaining load-bearing item, and it is now *half* done: A4 publishes the weights — per route
   and per `spawn` region, per space, per buffer — but as flat records, not as a graph. The
   missing half is the edges: which region produced the bytes another consumes. Without them a
   plan still cannot be named, priced or compared, and "optimisation" stays hand-waving.
1. **Add α** (vx-review#28), and rework route selection for size-dependent choice.
1. **Settle the composition law** once a copy engine is actually exercised. Nothing in the tree
   drives one yet: `raw::async_copy` is declared and gated but lowers to a synchronous element
   copy, so the capability half of §5 is enforced while the choice half remains unexercised.
1. **A contention term**, if the queue result holds on more than one part.
1. **Widen lowering emission past sm-scoped destinations**, edge kind by edge kind. Until then the
   extension point is real on exactly one edge shape.
1. **Settle topology identity before the parallel pipeline needs it.** Making topology equality
   compare index *values* rather than AST nodes (Vx#355) fixed the symptom and exposed the shape
   of the problem: five consumers in the compiler derive "which topology is this" independently,
   with different fidelity each time, and an identity that crosses a worker boundary has to be
   canonical, context-free, schedule-deterministic and totally ordered all at once. Written up in
   [`discussions/brainstorming/topology_identity_parallel_compilation.md`](discussions/brainstorming/topology_identity_parallel_compilation.md).
   The symbolic tier — `GPU[i]` compared against `GPU[j]` where `i == j` only at run time — is
   parked deliberately; it is a real gap and not the one blocking anything.
1. **Close the coverage backlog** ([Vx#358](https://github.com/hiraditya/Vx/issues/358)). Several
   shipped guards have no test that goes red when the guard is disabled, which is the condition
   under which one of them was found already disabled.

An optimiser over the program graph (min-cut for two devices, DP for a chain) is future work. It
consumes a truthful graph; it does not belong in the work that establishes truthfulness.

______________________________________________________________________

## 11. Where a value lives: one fact, two projections

A tensor's placement was expressible five ways, and they did not agree with each other.

| spelling | position | what it carried | now |
| -------- | -------- | --------------- | --- |
| `Pinned<T, Topology::X>` | type | a device | kept |
| `Ref<T, Memory::X>` | type | a space | removed for tensors |
| `Tensor<f32, [4, 4], Topology::X>` | type | a device | kept, and takes a space too |
| `.with_memory(Memory::X)` | expression | a space, via a wrapper | removed |
| `transfer(a, Memory::X)` | expression | a move, not an annotation | kept |

Four of the five say the same kind of thing in two different vocabularies — device or space — and
nothing reconciled them. `is_type_accessible` had to, at each of its 39 call sites, and it did so by
fabricating a `Ref` over a mock `f32` from a `Pinned` and recursing into itself. That function is the
shape of the problem: the type did not carry enough to answer where a value was, so a graph query
answered it instead, per use.

### They are not alternatives

Every topology has a default space, and a space is not anywhere in particular without a device that
holds it. "SMEM" alone is not a location; *this SM's* SMEM is. So device and space are two
projections of one fact, and a placement carries both:

```rust
struct Placement { topology: Topology, space: MemorySpace }
```

The surface lets either be written and derives the other:

| written | topology | space |
| ------- | -------- | ----- |
| `Topology::GPU` | as written | `default_memory_for(GPU)` |
| `Memory::GPU_HBM` | the space's owner | as written |
| neither | unplaced | unplaced |

Deriving in the second direction is what did not exist. `default_memory_for` has always answered
which space a device holds; nothing answered which device holds a space.

### Why the owner is stated rather than derived

Two derivations look plausible and both are wrong.

**Inverting the descriptor table.** Each topology kind declares a `default_space`, so the inverse
looks like a lookup. It is not a function: `CPU_DRAM` is the default of CPU, AMX, CpuAvx512 and
CpuNeon, and `NPU_HBM` of NPU, ANE and Slice. The two most-used spaces in the language would both be
ambiguous.

**Following `within:`.** A declared space names its parent, so the root of the chain looks like the
device. But `within:` is containment in the cost hierarchy, and every chain roots at host memory —
`Memory GPU_HBM { within: Memory::CPU_DRAM }` is how a GPU's memory is declared. Following it would
make GPU memory host-owned.

So ownership is a fact about the space, written once: a table for the built-ins, and for a declared
space, the topology whose `memory:` names it. Exactly one. None is an error, and so is more than
one — a space held by two devices is ambiguous in a way choosing the first would hide.

### What follows from carrying both

- `is_type_accessible` reads the device and the space off the type rather than reconstructing one
  from the other.
- A signature and a body resolve identically, because nothing consults the enclosing region. That
  matters: `fn f(t : Tensor<f32, [4, 4], Memory::SMEM>)` has no enclosing region, and neither does a
  buffer declared before the `spawn on` that fills it — which is how every placed destination in the
  corpus is written.
- `.with_memory(..)` and `Ref<T, Memory>` have nothing left to express that the placement slot does
  not, so five spellings become three.
- A memory space stops being a pseudo-value. `.with_memory(Memory::X)` was the one construction that
  legitimately took a space as an argument, so while it existed the checker could not refuse
  `Memory::X` in expression position at all. Removing it let that diagnostic finally be raised where
  the expression is typed, which is the only place that reaches a bare `let` or a `return`.

### Status

Built. The tensor type carries a `Placement` of a device and a space; both directions are derived
in name resolution, which is the first pass with program-wide scope — a declared topology's
`memory:` may arrive from a `--machine` file the module never mentions. The placement slot accepts
either spelling, `.with_memory(..)` and `Ref<T, Memory::X>` are gone, and `is_type_accessible` reads
the space off the type rather than fabricating a `Ref` over a mock `f32`. Five spellings are three.

`Placement` records which projection the source wrote. That is not bookkeeping: neither direction is
injective enough for the completion to guess. A device holding a non-default space and a declared
topology whose memory is not its like-named space produce indistinguishable pairs, so a heuristic
over `(topology, space)` gets one of them wrong. Equality ignores the flag, which is the point.

One thing above is still a claim rather than a rule: a space **no** topology declares falls back to
the like-named device rather than being refused. `owning_topology_in` computes the error; what is
missing is a pass with both a type walk and a diagnostic channel to report it from. Tracked
separately, together with the undeclared-topology fallback it has to stay consistent with.
