# What a custom `transfer` lowering must guarantee

A topology may supply its own code for moving data across one of its edges (see
[`memory_algebra.md`](memory_algebra.md) §6). That is how Vx stays useful on hardware nobody has
seen yet: a new part is a new topology file, not a compiler change.

The cost of that flexibility is that the safety properties Vx advertises are only as good as the
emitted movement. The type system proves a tile is in SMEM; if the lowering did not put it there,
the proof is about nothing. So a lowering is not just code — it is code plus a contract.

**We constrain correctness only.** How fast a lowering is, how many threads it uses, whether it
picks the best instruction — none of our business. A slow lowering is a bad lowering and the cost
model will say so. An *incorrect* lowering silently invalidates a compile-time guarantee, which is
a different category of problem.

______________________________________________________________________

## The shape of the thing

For `let y = transfer(x, Memory::B)` where `x : Pinned(T, A)`, a lowering emits code that moves
`x`'s bytes from `A` to `B` and yields `y : Pinned(T, B)`.

A sketch — the syntax is not settled, but the shape is. `raw::*` is the primitive set defined
later in this document:

```
impl transfer Memory::L2 -> Memory::SMEM {
  fn move(src: &Tile<f32>, dst: &mut Tile<f32>) {
    let n = raw::extent(src);
    let mut i = raw::lane();
    loop {
      if i >= n { break; }
      raw::store(dst, i, raw::load(src, i));
      i = i + raw::lanes();
    }
    raw::barrier();
  }
}
```

The signature carries one contract from the call site: the compiler allocated `dst` with `src`'s
shape, so `raw::extent(src) == raw::extent(dst)` is a fact the prover may assume. That is what
makes the `store`'s bound provable from a loop guard that only mentions `n = extent(src)`.

An earlier draft had the topology *declare* its lowering's properties (`effect: copy`,
`sync: barrier`, `overhead: 0 B`). Those fields are gone, and their absence is the point: the body
is Vx code we compile, so every one of them is **read off the body** rather than asserted beside
it. `effect` is `copy` because the body fills a distinct `dst` rather than returning a view of
`src` (C4); `sync` is `barrier` because the body ends in one (C3); `overhead` is zero because the
body allocates nothing (C5). A declaration can drift from the code it describes; a derivation cannot. Only C7's runtime
failure conditions still need declaring, because "the driver refused the allocation" is not
visible in any AST.

______________________________________________________________________

## The constraints

### C1 — Placement. The bytes end up in B.

After the lowering runs, `y`'s storage is in space `B`. This is what the result type asserts, and
every accessibility and capacity check downstream is built on it.

**Mechanically checked.** The emitted operations carry a memory space; it must match `B`'s. If a
lowering claims to place into SMEM and emits a global allocation, that is a compile error, not a
runtime surprise.

### C2 — Value preservation. `y` reads the same as `x`.

Element for element, same values, same indices. A transfer is a move, not a conversion. No layout
change, no precision change, no reordering that a program can observe.

**Conformance-tested.** Not checkable in general, so each (topology, edge, lowering) triple must
pass a test that writes a known pattern, transfers it, and reads it back. The pattern must vary
along every axis — a lowering that transposes or drops a dimension passes a constant-fill test.

### C3 — Visibility. Every reader observes the complete transfer.

If the movement is cooperative (many threads each copying part) or asynchronous (a copy engine), a
reader must not be able to see a partially-filled destination. The lowering emits the barrier or
the completion wait; it does not assume the consumer will.

**Discharged by the seam verifier.** `src/hir/seam.rs` already models exactly this: a transfer is
`Sync` when the lowering carries a release/acquire or a DMA completion wait, and
`Relaxed { published }` otherwise, and z3 is asked whether a stale read is reachable. Whether a
lowering is `Sync` or `Relaxed` is read off its body — a trailing barrier or completion wait is a
syntactic fact — and the existing obligation does the rest.

> **This is not hypothetical.** The SMEM lowering added in `dee50a69` emits an allocation and a
> copy into shared memory with **no `gpu.barrier`**. It is not a live bug only because kernels are
> currently single-threaded; the moment a kernel has more than one thread, a reader can observe a
> half-filled tile. The first lowering we wrote violated C3 on its first day.

### C4 — An alias is not a copy, and the checker sees which is which.

A lowering may return a *view* of the source rather than a copy. On unified memory that is the
whole point: the M4's host "transfer" costs nothing measurable (106.0 GB/s against a 106.3 GB/s
control, ratio 0.997) because there is no copy — the GPU reads the buffer the CPU wrote.

But then `x` and `y` are the same memory, and a write through one is visible through the other.
The borrow checker treats them as distinct values.

**Derived, then relied on.** Whether the body returns a view of the source or fills a distinct
destination is read off the body, and the borrow checker consumes that fact — the aliasing rules
then apply across the transfer. The derivation removes what would otherwise be the worst failure
mode in this list: a lowering that aliases while claiming to copy cannot be written, because
nothing is claimed — the checker reads the code.

The two shapes are different signatures. A copy fills a `dst` the call site allocated; an alias
returns a view of `src` and allocates nothing. The settled syntax must keep the two forms
distinct, because a copy-shaped body has no way to express an alias.

### C5 — Capacity honesty. Do not use more of B than was admitted.

Admission (E6010) checked the granule-rounded working set against `B`'s declared capacity. A
lowering that needs scratch — a staging buffer, double-buffering for a pipelined copy — consumes
capacity nobody accounted for, and E6010 is then proving something false.

**Derived, then checked.** Scratch the body allocates is visible in the body, and the compiler
adds it to the resident set at admission time. Undeclarable rather than undeclared: a lowering
cannot hide an allocation from the front end that compiles it.

### C6 — Only touch spaces the executing topology can see.

If the lowering stages through a third space, that space must be declared and listed in the
topology's `visible:`. A lowering that reaches into an undeclared space defeats the cross-topology
check that is Vx's headline safety property.

**Mechanically checked.** Walk the spaces the emitted operations touch against the topology's
visibility list.

### C7 — Totality, or declared failure conditions.

If admission accepted the program, the lowering must succeed — or it must declare the conditions
under which it can fail at run time (peer access unavailable, allocation refused).

This is the one that protects "refuse before you rent". A lowering that can fail for reasons
admission never checked makes the refusal weaker than advertised, and the failure shows up on the
rented machine, which is the exact outcome the whole design exists to avoid.

**Declared.** An undeclared runtime failure mode is a contract violation even if the code is
otherwise correct.

### C8 — Determinism.

Same source bytes in, same destination bytes out, independent of thread count, scheduling, or how
the work was divided. A lowering whose result depends on which thread got there first fails C2
intermittently, which is worse than failing it always.

**Conformance-tested**, by running the same transfer under different launch geometries.

### C9 — No effect outside `y`.

The lowering does not modify `x`, and does not touch program-visible state other than the
destination. Scratch is fine — it is accounted under C5 — provided it is not observable afterwards.

Exception: when the lowering is an alias (C4), `x` and `y` are the same storage by construction.

### C10 — The declared cost describes the emitted code.

The `: 128 B/cyc` on the edge must be the cost of what the lowering actually does, not of what the
hardware could do if asked differently.

**Not mechanically checkable** — and it is the reason the mechanism and the cost belong in **one
declaration**. Splitting them is what produced the `crossing: streamed` failure: a cost was
declared for a copy engine while the emitted code was a load followed by a store, and it took a
held-out A100 to notice. One declaration cannot drift from itself.

The calibration campaign is the backstop. A lowering whose cost is a fiction shows up as a residual
that does not close.

**Superseded below.** When the lowering is Vx code — the design this document settles on — no edge
cost is declared at all: traffic is derived from the body, and only the time model remains a claim.
See "Which means cost should be derived, not declared". C10 stays in the list as the property the
derivation *guarantees*, not as something an author still upholds by hand.

______________________________________________________________________

## How each one is enforced

| | constraint | how |
| --- | --- | --- |
| C1 | placement | read off the body — the emitted operations carry their memory space |
| C2 | value preservation | conformance test, varying along every axis |
| C3 | visibility | **seam verifier** (`src/hir/seam.rs`), fed the body's sync facts |
| C4 | aliasing | read off the body — a view is visibly a view; the borrow checker consumes it |
| C5 | capacity overhead | read off the body — allocations are in the AST; admission adds them |
| C6 | space visibility | read off the body — spaces touched, walked against `visible:` |
| C7 | failure modes | **declared** — the one thing no AST can show |
| C8 | determinism | by construction from the primitives' guarantees; the conformance run validates those guarantees on the part |
| C9 | no side effects | read off the body — every write is visible |
| C10 | cost honesty | superseded — traffic is derived from the body; the time model is what calibration scores |

Five are read off the body (C1, C4, C5, C6, C9), one reuses a verifier that exists (C3), one
remains a declaration (C7), and two need the hardware (C2 and C8 — the conformance suite). C10
dissolves: a cost computed from the code cannot disagree with the code. Nothing here needs new
proof machinery.

______________________________________________________________________

## What we deliberately do not constrain

- **Performance.** A lowering may be slow. The cost model will say so, and that is the correct
  place for it to show up.
- **Mechanism.** Copy engine, load/store loop, DMA, something that does not exist yet.
- **Parallelism.** How many threads participate, and how the work is split.
- **Internal asynchrony.** A lowering may be async internally as long as C3 holds at its boundary.

______________________________________________________________________

## The lowering is Vx code, so almost none of it is opaque

An earlier draft of this document worried that letting a machine file carry real code puts
arbitrary code inside the trust boundary, and that C1 and C6 stop being mechanically checkable as a
result. **That is wrong**, and getting it wrong pointed at a worse design.

A lowering is written as `impl transfer Memory::A -> Memory::B { ... }` in **Vx**. Our own front
end parses it, type-checks it and lowers it. There is no foreign object code and no plugin
boundary — it is more Vx, subject to every check Vx already performs. So:

- **C1 (placement)** — we emitted the operations, so we can read their memory space.
- **C6 (space visibility)** — we have the AST; walking the spaces the body touches is a traversal.
- **C9 (no side effects)** — every write in the body is visible to us.
- **C3 (sync)** — whether the body contains a barrier or a completion wait is a syntactic fact,
  and it is exactly the input `seam.rs` already wants.
- **C4 (aliasing)** — whether the body returns a view of the source or a fresh allocation is
  something the checker can see rather than something the author asserts.
- **C5 (overhead)** — scratch the body allocates is in the AST, so admission can account for it.

The trust boundary is not the file. It is the **primitive set** a lowering needs and ordinary code
does not: raw address arithmetic, barriers, and whatever asynchronous-copy intrinsic a part
exposes. `transfer` cannot be implemented in terms of `transfer`, so the bottom of the stack has to
be primitive.

That is the Rust model, and Vx already has the pieces. `in_unsafe_block` tracking exists,
`requires`/`ensures` exist as syntax, and `SmtProver` discharges them against z3. A lowering is
safe code over a small unsafe primitive set, with the obligations in this document attached as
`requires`/`ensures` clauses on the primitives. Nothing new is needed except naming the primitives.

## Which means cost should be derived, not declared

This is the larger consequence, and it supersedes C10 rather than merely satisfying it.

If the lowering is code we compile, we can **count what it moves**. Tile shapes are static, so loop
bounds are static, so the number of loads and stores against each space is a compile-time quantity.
That is *traffic*, exactly, with no declaration involved.

The split then falls out cleanly:

- **Traffic** — derived from the lowering. Exact, mechanical, no way to drift from the code.
- **Time** — modelled from traffic plus the machine's declared bandwidths, α, and composition.
  This is where the uncertainty lives, and where the calibration campaign belongs.

An edge no longer needs a declared cost at all; it needs declared *bandwidths on its endpoints*,
which is a property of the machine and always was.

Two things this buys that a declared edge cost cannot:

1. **C10 becomes structural.** The cost cannot describe something other than the emitted code,
   because it is computed from the emitted code.
2. **It sees plan-level waste.** The shipped flash-attention kernel re-reads K and V from global
   memory on *every* query iteration. No edge cost can express that — the per-hop rate is identical
   either way — but a traffic count reads it straight off the loop structure. That is precisely the
   class of inefficiency the whole exercise is aimed at, and a declared-cost model is blind to it.

The limit is honest and narrow: derived traffic needs static bounds. A data-dependent loop needs
either a bound or a declaration, and a lowering that has one should say so.

## The primitive set

`transfer` cannot be implemented in terms of `transfer`, so a lowering needs a floor to stand on.
This names the floor: the operations an `impl transfer` body may use that ordinary Vx code may
not, each with the obligation the compiler discharges at the call site (`requires`) and the
guarantee downstream proofs rely on (`ensures`). Both are existing Vx syntax, and the prover that
discharges them against z3 exists (`src/hir/prover.rs`). What is new is only this vocabulary.

**The design decision that matters: indexed, not addressed.** Every primitive takes a typed tile
and an element index. None takes an address, and no primitive produces one. Address arithmetic is
exactly what would make C1 and C6 unprovable — a pointer can point anywhere, so the compiler could
no longer read which space an access touches or whether it stays in bounds. Indexed access on a
typed tile keeps both facts checkable at every call site, and a copy loop needs nothing more.

### The eight primitives

| primitive | `requires` | `ensures` |
| --- | --- | --- |
| `raw::extent(t)` | — | the element count of `t`; pure |
| `raw::lane()` | — | result `< raw::lanes()`; the same value on every call within one activation |
| `raw::lanes()` | — | result `>= 1`; the same value on every lane |
| `raw::load(t, i)` | `i < raw::extent(t)`; `t`'s space readable by the executing topology | the value at index `i`; writes nothing |
| `raw::store(t, i, v)` | `i < raw::extent(t)`; `t`'s space writable by the executing topology; `t` held by `&mut` | afterwards `t[i] == v`, and no other element changed |
| `raw::barrier()` | reached by every lane — a barrier under divergent control flow is rejected outright | every store issued before it, by any lane, is visible to every load after it |
| `raw::async_copy(dst, src, i)` | both index bounds; both space obligations; **the machine file declares a copy engine** | the copy is *initiated*; `dst[i]` is unspecified until a matching `raw::async_wait` |
| `raw::async_wait()` | — | every `async_copy` this lane initiated has completed (cross-lane visibility still needs `raw::barrier`) |

`lane`/`lanes` returning the same value everywhere they are asked is what makes a lowering
deterministic by construction — *given* the primitives honour their `ensures` on the part, which
is exactly what the C8 conformance run validates. The work split is a pure function of lane
identity, not of who arrived first.

The async pair is two primitives on purpose. "`dst[i]` is unspecified until the wait" is a fact z3
can use: a lowering that issues copies and forgets the wait is rejected by the same seam
obligation (`src/hir/seam.rs`) that catches a missing barrier — the transfer function is
`Relaxed { published }`, a stale read is reachable, REJECT. The failure mode is proven away, not
tested away.

### How each obligation is discharged

- **Bounds** (`i < extent`): the prover. Tile shapes are static, so loop bounds are static and the
  comparison is arithmetic over known quantities. Where it cannot be proven, the call is rejected —
  unless the author wraps it in `unsafe`, which records the asserted obligation and is surfaced
  the way an unverified `spec:` figure is.
- **Spaces** (readable/writable by the executing topology): a table lookup against `visible:` and
  `scope:`. Mechanical; no prover involved. This is C1 and C6.
- **Exclusivity** (`t` held by `&mut`): the borrow checker, unchanged — a lowering body is
  ordinary Vx to it.
- **Capability** (`async_copy` needs a copy engine): a lookup in the machine file — see the next
  section.
- **Barrier uniformity**: a control-flow check, and conservative — a barrier some lanes can skip
  deadlocks on every real part, so it is rejected rather than warned about.
- **Async discipline**: the seam verifier, as above. This is C3.

### Capability gates the primitives

`raw::async_copy` is legal only in a lowering for a part whose machine file declares the copy
engine. Capability stays in the machine file, choice stays in the lowering, and using a primitive
the part does not have is a compile error rather than a runtime surprise. This is the
capability/choice split of `memory_algebra.md` §5 **enforced** rather than merely documented — the
mistake that produced the falsified `crossing: streamed` prediction becomes unwritable.

### What each primitive contributes to derived traffic

| primitive | traffic |
| --- | --- |
| `load` | `sizeof(T)` read against `src`'s space |
| `store` | `sizeof(T)` written against `dst`'s space |
| `async_copy` | one chunk read from `src`'s space, written into `dst`'s space |
| everything else | none |

With static bounds the counts are exact at compile time — this is where "cost is derived, not
declared" cashes out.

### Deliberately absent

- **Addresses, pointer arithmetic, space casts.** Their absence is what keeps C1 and C6 provable.
- **Type reinterpretation.** A transfer is a move, not a conversion (C2).
- **Atomics.** A copy does not need them, and every primitive added grows what `seam.rs` must
  model. They can earn a place with a use case; they do not get one in advance.
- **Unbounded loops.** A loop whose bound the prover cannot see needs a declared bound, or the
  lowering is rejected — the same restriction that keeps derived traffic computable.

______________________________________________________________________

## Open questions

- **Inside the primitive set:** the async chunk granularity (sm_80's `cp.async` moves 4, 8 or 16
  bytes per lane, and that constraint belongs in the machine file next to the capability it
  refines), and whether atomics ever earn a place.
- **Who runs the conformance suite, and when?** Ideally the compiler refuses a lowering that has
  never passed one. That needs the suite to be part of the topology's declaration, not a separate
  process someone remembers to run.
- **Can a lowering be checked without the hardware?** C1, C3, C4, C5, C6 and C9 yes — they are
  properties of the code. C2 and C8 need the part. So a topology for hardware nobody has rented
  has a lowering whose *value preservation* is unverified, and that should be as visible as an
  unverified `spec:` figure.
