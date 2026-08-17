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

`raw::*` is the primitive set defined later in this document:

```
impl Transfer<Memory::L2, Memory::SMEM> for Topology::Ampere {
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

### Why the `for Topology::X` clause is required

This document's first sentence says a *topology* supplies the code for one of *its* edges. The
syntax did not say so for a while: a lowering named an edge and nothing else, and was keyed on
that pair across the whole compilation. So the shape the fleet directory exists to describe was
unbuildable. An Ampere part fills shared memory with `cp.async` and a Hopper part drives the same
`Memory::L2 -> Memory::SMEM` edge differently; both machine files declare that edge for
themselves, but only one of them could implement it, and writing the second was a hard error.

The edge clause always lived inside a `Topology { ... }` block. Only the `impl` had escaped its
machine, and naming the machine puts it back. Two obligations become checkable as a result, both
of which had to be skipped while a lowering named no machine:

- the topology it names is one this compilation actually declares;
- that topology declares the edge being implemented, so there is a movement to implement.

The second was carried on the AST as a comment for as long as the clause was missing — the edge
should be "matched against the topology's declared edges by sema (not yet wired)". It could not
be wired. The declaration did not say which topology to look at.

Selection at a transfer site prefers the topology in force there. It cannot *require* it, because
a host edge is driven from the host: `transfer(a, Memory::GPU_HBM)` in `main` runs with the CPU
active while implementing an edge that belongs to the device. So a single unambiguous candidate is
taken as well, and only a real ambiguity — several machines implementing this edge, none of them
the one we are on — is refused.

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

> **This was not hypothetical.** The SMEM lowering added in `dee50a69` emitted an allocation and a
> copy into shared memory with **no `gpu.barrier`**. It was not a live bug only because kernels are
> currently single-threaded; the moment a kernel has more than one thread, a reader can observe a
> half-filled tile. The first lowering we wrote violated C3 on its first day — and it stayed
> violated for the two stages it took to build the machinery that could say so. Closed in A3
> (see below); the barrier is now pinned by a `bar.sync` count in `device_image_test.rs`, because
> a fix nothing asserts is a fix waiting to be deleted.

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

A lowering is written as `impl Transfer<Memory::A, Memory::B> for Topology::X { ... }` in
**Vx**. Our own front
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
This names the floor: the operations an `impl Transfer` body may use that ordinary Vx code may
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
| `raw::async_copy(dst, src, i)` | both index bounds; both space obligations; **the machine file declares a copy engine** | the copy is *initiated*; `dst[i]` is unspecified until a matching `raw::async_wait`, and `src` must not be written until then (the engine is still reading it) |
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

### What A2 landed (2026-08-16, Vx#353)

The slice above is implemented in `src/hir/check/raw.rs`. The decisions the sketch left open
were settled as follows:

- **Indices and extents are `i64`.** `raw::extent`/`lane`/`lanes` return `i64`, so
  `for i in 0..raw::extent(t)` binds an `i64` induction variable; a bare literal index
  (`raw::load(t, 0)`) types `i32` by default (#240) and both are accepted — the prover does not
  care, and demanding a cast would prove nothing.
- **Bounds run through `prove_expr`**, the same refutation loop `Verified<T>` uses. Two facts
  make the common shapes close without hand-written invariants: a range loop now records
  `a <= i && i < b` for its induction variable (dropped if the body reassigns it), and
  `raw::extent(t)` folds to its literal element count inside recorded facts and obligations —
  the prover cannot lower a function call, so the fold is what connects `let n = raw::extent(src)`
  to a bound written against `n`. The obligation is `0 <= i < extent`, both ends. Note the
  prover fails **open** when z3 is missing (`src/hir/prover.rs`): the obligation is a proof on a
  complete toolchain and a recorded assumption otherwise.
- **The conservative barrier rule is positional**: `raw::barrier()` is legal only as a top-level
  statement of the body (E6019). A barrier in a uniform-trip-count loop is real hardware practice
  and could be admitted later by a lane-taint analysis; until someone needs it, top-level-only is
  the rule that cannot admit a divergent barrier.
- **The copy engine is a transfer-edge marker**: `transfer Memory::L2 -> Memory::SMEM
  copy_engine` in the machine file (the same trailing-marker position as `relaxed`/`sync`).
  A capability is a property of a link, and the edge is the link. `fleet/a100-80.vx`,
  `fleet/h100-sxm.vx` and `fleet/node-8gpu.vx` declare it on their SMEM hops (`cp.async`).
- **Async discipline is a forward walk that models every exit** (E6021): branches join by union,
  a `return` snapshots its in-flight state for the end-of-body leak check, and a `break` or
  `continue` feeds the enclosing loop's joins — a lane-guarded early return carrying an unwaited
  copy was the review's first counterexample, and exits are where such bugs live. Reading or
  writing a destination with an outstanding copy is refused, as is writing the *source* of one
  (the engine is still reading it), as is any path out of the body with outstanding copies.
  Copies may stay outstanding *across* loop iterations — one copy per iteration and a
  single wait after the loop is exactly the commit-group pattern the engine exists for; loop
  bodies are walked twice so a read at the top of iteration N+1 still meets the copy issued at
  the bottom of iteration N. A body
  that publishes (stores or async-copies) on a synchronizing edge must end with
  `raw::barrier()` — the syntactic form of the seam obligation: without it the transfer is
  `Relaxed { published }` and the stale read of `hir/seam.rs` is reachable.
- **Space visibility** (E6022) checks both edge endpoints against every declaring topology's
  `visible:` list. The finer `scope:`-based writability question (can a device-scoped engine
  write an sm-scoped space?) is noted, not yet modelled.
- **Primitives cannot appear inside closures** (E6017), refused at typing — the one place that
  reliably sees closure bodies, since they are lifted out before the whole-body scan runs. A
  generic instantiated *from* a lowering body is ordinary code and the namespace does not
  resolve inside it either.
- **A publishing lowering on a synchronizing edge has one exit**, at the bottom, past the
  barrier (E6021): an early `return` would leave a path that publishes unsynchronized, and if
  only some lanes take it, the rest wait at a barrier the exited lanes never reach. The final
  `return`'s own expression must not publish — it runs after the barrier.
- **Tiles are the lowering's parameters, nothing else** (E6017). A local alias (`let d2 = dst`)
  carries the same tensor under a different name, and a different name is invisible to the
  name-keyed async walk — the review read an in-flight destination through exactly that alias.
- **Prover facts are shadow-aware**: rebinding a name (a shadowing `let`, a loop induction
  variable) neutralizes every recorded fact about it. Before that, a stale outer `i < 2` next
  to an inner `i < 8` forged an out-of-bounds "proof". Range facts with bounds the prover
  cannot lower (a call) are not recorded at all, so they cannot poison later proofs with
  warnings. Wrapping the body in a top-level `unsafe { }` block is transparent to the
  barrier-position and trailing-grade rules — unsafe is the documented bounds escape, not
  control flow.
- **`raw::` is a reserved call namespace everywhere**, so a user struct named `raw` cannot
  supply static methods; the diagnostic says so directly instead of "undefined method".
- **Imported modules' lowerings are checked like the main module's**, with diagnostics kept:
  a library's ordinary functions were validated when the library compiled, but a lowering's
  edge, capability, and space obligations resolve against the *importing* compilation's
  machine file, which the library never saw. `E6022` exempts `CPU_DRAM` endpoints — the host
  side of a host link is reachable by construction and no shipped fleet file lists it as
  `visible:`.
- Still open after A2: emission of the body (A3, below), derived traffic counts, the C7
  failure-mode declaration syntax, and running the whole-body pass on the parallel-pipeline
  schedule (today it runs on the driver path that `vxc` uses; the pipeline schedule
  type-checks bodies but skips the whole-body pass).

### What A3 landed (2026-08-16, Vx#353)

A user-supplied lowering now moves the bytes. The witness is structural: the corpus fixture
`tests/backend/pass/custom_topology_user_lowering.vx` says `raw::barrier()` twice, and its
device image carries **two `bar.sync`** where the builtin's carries one. A transfer is a
move, not a conversion (C2), so the computed answer cannot tell the two apart — only the
shape of the emitted code can.

- **The builtin's missing barrier is closed.** `TransferOpLowering`'s shared-memory branch
  now emits `gpu.barrier` after its copy. That was the C3 gap named in this document: a copy
  into shared memory that no lane waits on. It was latent only because kernels launch one
  thread today.
- **Inlined, never called.** The lowering body is spliced into the transfer site with its
  `(src, dst)` parameters bound to the source value and the placed tile. A `func.call` inside
  a kernel region would compile fine on the host and silently cost the kernel its device twin
  — the device-readiness walk allows `arith/cf/gpu/math/memref/scf` and excludes `func`, so a
  call means no PTX, no image, and a fallback nobody asked for.
- **The site keeps its allocation half.** The `vx.transfer` op still carries the space,
  granule, slot offset, and capacity descriptors, and still becomes the shared-space alloca
  that gets promoted to real `.shared` storage. A `user_lowered` attribute tells the C++ side
  to skip only the copy and the barrier. The part that took hardware debugging to get right
  is untouched by design.
- **Primitive lowerings**: `load`/`store` become `memref.load`/`memref.store`, with the
  contract's flat element index delinearized row-major against the tile's static dims;
  `extent` becomes the constant element count; `barrier` becomes `gpu.barrier`; `lane` and
  `lanes` become `gpu.thread_id`/`gpu.block_dim` rather than the constants 0 and 1 — correct
  under any launch geometry, and today's one-thread launches read exactly 0 and 1 anyway.
  `async_copy` lowers to a synchronous element copy and `async_wait` to nothing: the
  capability is declared and gated (E6020), but no engine is driven until the `nvgpu` route
  exists. The fallback preserves the contract's semantics; it does not preserve its
  performance, which is the honest state to be in.
- **The host runs the same body.** The CPU fallback compiles the same kernel region, where
  the thread id is 0, the block holds one thread, and a barrier over one thread orders
  nothing — so the host stage folds those three ops to exactly that after the device image
  has been taken. That is what lets a machine with no GPU still execute the fixture and check
  the answer.
- **Static shapes, on both paths.** The AST path used to type a placed tile as `?x?`, and a
  dynamic shared tile cannot become a `.shared` global — the image it produced carried
  shared-typed instructions against local storage and faulted on an A100. The transfer's
  result type is now static whenever the source tensor's dims are literals, so that path
  materialises real storage too. Genuinely dynamic tiles are still refused.
- **The flat path declines** a transfer with a user lowering and hands the module to the AST
  path, which stays the oracle — the same arrangement every other unsupported construct uses.
- **Emission is sm-scoped destinations only, for now.** That is the one edge kind where the
  site becomes an allocation the body fills in place. On every other edge the builtin still
  moves the bytes — the plugin's device copy, or the host alloc-and-copy — so an inlined
  body would run *beside* it: the same bytes moved twice, and on a real GPU a host store
  through a device pointer (reproduced on three edge kinds in review). A lowering declared
  for a non-sm edge warns and falls back to the builtin. Widening this means teaching the
  other lowering branches to stand aside the way the sm branch does, edge kind by edge kind.
- **What the splice cost, and what paid for it.** Inlining someone else's body into a
  function's scope is where this stage's real defects lived, not in the primitives. The
  generator's name maps are flat — no scoping — so the first version leaked a body-local
  `let` into the caller (silently wrong answer), let the body's loop variable clobber the
  site's (out-of-bounds store), and left a stale alloca flag behind (an ICE). It also bound
  only the first two parameters, so a third resolved against whatever the caller had under
  that name and wrote into a buffer the lowering never named. The splice now snapshots and
  restores the whole environment, and a lowering must take exactly two statically-shaped
  tile parameters. Every one of those was found by adversarial review with a running
  compiler, not by reading the code.
- **Declared shape must be the shape moved** (E6023). A lowering is selected by edge, so
  nothing else ties its `&Tensor<f32, [2,2]>` to the tile at the site, and the primitives
  read their extents from the declaration. A too-small declaration copied part of the tile
  and read the rest back uninitialised — disclosing stack contents; a too-large one stored
  past the end and segfaulted. Refused now, with the honest message that the two shapes must
  agree. Making a lowering generic over shape is future work; it wants the shape to be a
  parameter, which is the same unsettled signature question.
- **A `return` before the end is refused on every edge** (E6021), not only on publishing
  bodies over synchronizing edges as A2 had it. The body is inlined: a non-final return
  returns from whatever function performs the transfer. On a `relaxed` edge this escaped
  every check and made `main` return 7 without running the rest of the program.

#### Known limitation: a barrier is only as uniform as its site

`E6019` refuses a `raw::barrier()` under control flow *in the lowering body*, because a
barrier some lanes skip deadlocks. The splice then pastes that body wherever the transfer
is — so a `transfer(x, Memory::SMEM)` inside an `if` puts the barrier under the site's
condition, and the positional rule cannot see it. The builtin lowering has exactly the same
exposure now that it emits a barrier.

This is latent for the same reason the original C3 gap was: kernels launch one thread today
(#251). It becomes real the moment they do not, and the fix wants uniformity analysis —
knowing whether a branch condition is the same for every lane — which is worth building
alongside the parallelism it protects, not before. Recorded here rather than left for
someone to rediscover, which is what the C3 gap earned by being written down.

- Still open after A3: uniformity analysis for the barrier-at-the-site case above, the
  `nvgpu` route for real asynchronous copies, derived traffic counts in `--diagnostics-json`
  (A4), the C7 failure-mode declaration syntax, a settled signature convention for lowering
  methods (today: exactly one method taking exactly two statically-shaped tiles), `raw::`
  opcodes on the flat path, and the parallel-pipeline schedule parity noted above.

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

**Implemented in A4** (2026-08-16). Each `raw::` call contributes the table's bytes multiplied by
the product of its enclosing static `for`-range trip counts; `raw::extent(t)` resolves from the
declared tile shape, since at a transfer site the lowering's own parameters are not in scope. A
branch contributes the per-space **maximum** of its arms, not their sum — one execution takes one
arm — and the route is then marked `traffic_exact: false`. Anything the counter cannot weigh
exactly (an unbounded `loop`, a `break`, a non-literal bound) produces **no traffic at all**, with
a reason recorded beside it.

That last rule is the load-bearing one. A count that quietly assumed one lane, or reported zero
for a body it did not understand, would be indistinguishable in a harvested record from a
measurement — the same reason `derived_cost` is null rather than 0 when nothing declares a
bandwidth. The cooperative lane-strided copy sketched above is exactly such a body: correct code
whose trip count is a launch-time fact, so its traffic is honestly absent.

The counts carry bytes and nothing else — no bandwidth, no time, no opinion about how long they
take. What they cost is the time model's separate and calibrated claim; traffic is the input it
consumes, not a prediction it produces. They reach a consumer through `--diagnostics-json`, per
route, split read from written per space (a space's read and write bandwidths are different
figures, and a plan that re-reads a tile is wasteful in a way a combined total cannot show).

The first thing this makes visible is *plan-level* waste. A lowering that reads its source twice
and averages — value-preserving, so C2 holds and the answer is bit-identical — reports twice the
reads against an unchanged tile size. No declared edge cost can distinguish that program from the
faithful copy, because the waste is not in the edge.

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
