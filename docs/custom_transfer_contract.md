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

A sketch of how a topology might declare one — not settled syntax, but the fields the contract
needs to name:

```
transfer Memory::L2 -> Memory::SMEM : 128 B/cyc {
  lowering: cooperative_copy,   // which lowering; from a registry the compiler ships
  effect:   copy,               // `copy` or `alias`            -- C4
  sync:     barrier,            // `barrier`, `async_wait`, `none` -- C3
  overhead: 0 B,                // extra destination space used  -- C5
}
```

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
`Relaxed { published }` otherwise, and z3 is asked whether a stale read is reachable. A custom
lowering declares which it is, and the existing obligation does the rest.

> **This is not hypothetical.** The SMEM lowering added in `dee50a69` emits an allocation and a
> copy into shared memory with **no `gpu.barrier`**. It is not a live bug only because kernels are
> currently single-threaded; the moment a kernel has more than one thread, a reader can observe a
> half-filled tile. The first lowering we wrote violated C3 on its first day.

### C4 — Aliasing must be declared.

A lowering may return a *view* of the source rather than a copy. On unified memory that is the
whole point: the M4's host "transfer" costs nothing measurable (106.0 GB/s against a 106.3 GB/s
control, ratio 0.997) because there is no copy — the GPU reads the buffer the CPU wrote.

But then `x` and `y` are the same memory, and a write through one is visible through the other.
The borrow checker treats them as distinct values.

**Declared, then relied on.** A lowering is `effect: copy` or `effect: alias`. `alias` tells the
checker the two values share storage, so the aliasing rules apply across the transfer. A lowering
that aliases while declaring `copy` breaks the borrow checker's assumptions silently, which is the
worst failure mode in this list.

### C5 — Capacity honesty. Do not use more of B than was admitted.

Admission (E6010) checked the granule-rounded working set against `B`'s declared capacity. A
lowering that needs scratch — a staging buffer, double-buffering for a pipelined copy — consumes
capacity nobody accounted for, and E6010 is then proving something false.

**Declared, then checked.** `overhead:` is added to the resident set at admission time. A lowering
with undeclared overhead is a lowering that can overflow a space the compiler just certified.

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
destination. Scratch is fine if it is declared under C5 and not observable afterwards.

Exception: when `effect: alias`, `x` and `y` are the same storage by construction, and C4 governs.

### C10 — The declared cost describes the emitted code.

The `: 128 B/cyc` on the edge must be the cost of what the lowering actually does, not of what the
hardware could do if asked differently.

**Not mechanically checkable** — and it is the reason the mechanism and the cost belong in **one
declaration**. Splitting them is what produced the `crossing: streamed` failure: a cost was
declared for a copy engine while the emitted code was a load followed by a store, and it took a
held-out A100 to notice. One declaration cannot drift from itself.

The calibration campaign is the backstop. A lowering whose cost is a fiction shows up as a residual
that does not close.

______________________________________________________________________

## How each one is enforced

| | constraint | how |
| --- | --- | --- |
| C1 | placement | compiler checks the emitted memory space |
| C2 | value preservation | conformance test, varying along every axis |
| C3 | visibility | **seam verifier** (`src/hir/seam.rs`), already built |
| C4 | aliasing | declared; borrow checker consumes it |
| C5 | capacity overhead | declared; admission adds it |
| C6 | space visibility | compiler walks emitted ops against `visible:` |
| C7 | failure modes | declared |
| C8 | determinism | conformance test across launch geometries |
| C9 | no side effects | conformance test |
| C10 | cost honesty | one declaration, plus the calibration |

Four are mechanical, one reuses a verifier that exists, four are declarations the compiler then
relies on, and the rest are a conformance suite. Nothing here needs new proof machinery.

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

## Open questions

- **Which primitives?** The unsafe set a lowering may use — raw addressing, barriers, async-copy
  intrinsics — needs naming, and each needs its `requires`/`ensures`. This is the whole remaining
  design surface, and it is much smaller than "how do we sandbox arbitrary code".
- **Who runs the conformance suite, and when?** Ideally the compiler refuses a lowering that has
  never passed one. That needs the suite to be part of the topology's declaration, not a separate
  process someone remembers to run.
- **Can a lowering be checked without the hardware?** C1, C3, C4, C6 and C9 yes — they are
  properties of the code. C2 and C8 need the part. So a topology for hardware nobody has rented
  has a lowering whose *value preservation* is unverified, and that should be as visible as an
  unverified `spec:` figure.
