# Vx Operational Semantics & Memory Model

This document specifies the runtime evaluation, spatial computing execution model, and memory rules of the Vx language.

## 1. Spatial Computing Model

Vx does not assume a single, flat instruction stream executing on a monolithic CPU. The language intrinsically models distributed, heterogeneous topologies (e.g., CPU, GPU, NPU, AccCore).

### 1.1 The `spawn on(Topology)` Operation

The `spawn on(Topology)` statement transitions the execution context from the current hardware unit to the specified hardware topology.

**Operational Rule:**
Let $E[ \\text{spawn on}(\\tau) { B } ]$ be a program state evaluated on an active hardware unit $\\rho$.

1. The runtime checks if the topology $\\tau$ is reachable from $\\rho$.
1. The runtime reserves resources on $\\tau$.
1. Execution of the block $B$ is enqueued on $\\tau$.
1. The spawning unit $\\rho$ continues asynchronously unless a data dependency explicitly synchronizes the contexts.
1. All variables captured within $B$ that are not in a shared or transferred memory space relative to $\\tau$ will result in a compiler error.
1. **Index scope rule**: If $\\tau$ contains an index expression (e.g., `NPU[i]` or `NPU[0..4]`), the index expression is evaluated in the **calling** scope $\\rho$, not in $\\tau$. This ensures loop variables and other outer-scope identifiers are accessible.

### 1.2 The `unroll across(Topology)` Operation

`unroll across` is a spatial loop parallelizer. It takes a physical or logical topology slice (e.g., `Topology::NPU[0..4]`) and spatially duplicates the inner block $B$.

**Operational Rule:**
`unroll across(Topology::Unit[A..B]) { |id| B }` evaluates to parallel `spawn on` instances:
`spawn on(Topology::Unit[A]) { let id = A; B } || ... || spawn on(Topology::Unit[B-1]) { let id = B-1; B }`.

______________________________________________________________________

## 2. Memory Model & Transfers

Standard languages employ a flat memory model (`*mut T`). Vx uses a partitioned address space memory model via `Ref<T, MemorySpace>`.

### 2.1 Spatial Isolation

If variable $v$ is of type `Ref<T, Memory::CPU_DRAM>` and execution context is `Topology::NPU[0]`, attempting to dereference $v$ triggers a **spatial fault** (caught statically at compile-time by the Semantic Analyzer).

### 2.2 The `transfer` Primitive

Data movement across distinct memory hierarchies is explicit.
`let local_data = transfer(host_data, Memory::NPUHBM);`

**Operational Rule:**

1. Given source reference $S$ residing in space $M_1$.
1. Evaluates to a target reference $T$ in space $M_2$.
1. The runtime schedules an asynchronous or synchronous DMA transfer from $M_1$ to $M_2$.
1. Linearity: If $S$ represents uniquely owned data, the `transfer` consumes $S$. Future use of $S$ in space $M_1$ is invalid.

### 2.3 Declared Memory Spaces & Sub-space Scheduling

A `Memory` declaration gives a space real semantics beyond a bare name — its place in the hierarchy, its size, and how it is allocated:

```
Memory SMEM {
  within:    Memory::GPU_HBM,   // containment: SMEM ⊂ GPU_HBM
  capacity:  228 KB,            // the space is bounded
  bandwidth: 128 B/cyc,         // feeds the roofline transfer cost
  granule:   16 KB,             // allocation granularity (a TMEM/SMEM sub-scratchpad)
  scope:     sm,                // the execution level the space is private to
}
```

**Placement checking.** A statically-shaped tensor placed into a space (by `transfer`) must fit:

- *Per-tile* (`E6009`): a single tile larger than `capacity` is rejected.
- *Cumulative* (`E6010`): the **working set** — the sum over all tiles placed in the space — must fit `capacity`. When the space declares a `granule`, each tile is rounded up to the granule first (a tile smaller than a granule still occupies a whole one), so the quantity checked is the granule-rounded working set. Declaring the space `overcommit` downgrades `E6010` to a warning (`W1028`): the programmer asserts the tiles do not all coexist.
- *Coherence*: `within:` forms a containment tree; a sub-space may not exceed its parent's capacity (`E6007`) nor widen its `scope` below the parent (`E6011`); a `within:` cycle is rejected (`E6006`).

**Sub-space reachability.** A sub-space (SMEM/TMEM) has no hardware transfer edges of its own; a `transfer` **into or between** sub-spaces is reachable through their enclosing spaces — each endpoint is resolved to itself-or-a-`within:`-ancestor, and a sibling→sibling move (e.g. TMEM→SMEM) meets at their nearest common ancestor and is costed on both legs. A value already resident in a sub-space is recognized as *starting there* when re-transferred.

**Sub-space scheduling.** For a space with a `granule`, the compiler assigns each placed tile a concrete position via a granule-rounded bump allocator — a byte `offset` and a `slots` (granule) count within the space — and preserves the full sub-space descriptor together with this assignment as **metadata on the IR** (`vx.transfer`).

**Design choices** (why it takes this shape):

- *Metadata, not the memref type.* The sub-space identity and schedule ride as op-attributes, not as a memref memory-space / address space. This keeps the description available to later passes and a device backend with **zero effect on the lowering** — a typed memory space would break the CPU-fallback address computation. Promoting the identity into the memref type is deferred to a real device backend.
- *Scoped to granule'd sub-spaces.* Scheduling and granule-rounding apply only to spaces that declare a `granule` (the sub-scratchpads — TMEM/SMEM); device-global spaces (HBM/DRAM) are unaffected.
- *One budget, two granularities.* The capacity check and the scheduler share a single notion of the working set; the granule-rounded sum is the allocation-accurate refinement of the raw sum, and `overcommit` relaxes both.

> Full design: `docs/discussions/implementation_plans/first_class_memory_spaces.md` (declaring + checking) and `subspace_scheduling.md` (the metadata + scheduler).

______________________________________________________________________

## 3. Typestates and Hardware Fallibility

Because computations map to physical hardware, availability is non-deterministic.

### 3.1 `Verified<T>` vs `Pinned<T, Topology>`

- `Verified<T>`: A lazily evaluated computation graph. The compiler has verified memory and type safety, but the physical location is deferred to the runtime cost model.
- `Pinned<T, Topology>`: An eagerly bound computation locked to a specific physical unit.

### 3.2 Monadic Transitions (`HardwareState`)

The transition `Verified<T> -> try_pin(T) -> HardwareState<Pinned<T>, Verified<T>>` acts as a failible monad. The runtime attempts to acquire the lock on the Topology. If successful, it returns `Available(Pinned)`, else `Saturated(Verified)`.

______________________________________________________________________

## 4. Compile-Time Evaluation (`comptime`)

Vx supports deterministic ahead-of-time evaluation via `comptime` blocks.

- Execution within a `comptime` block happens during the Semantic Analysis compiler pass.
- Operations inside `comptime` are guaranteed to have zero runtime overhead.
- Used predominantly for layout calculation, array sizing, and topological assertion (`assert(...)`).

**Operational Rule:**
If $E[\\text{comptime} { B }]$ evaluates to $v$, the AST is strictly replaced by the literal or reduced expression $v$ before lowering to the intermediate representation (MLIR).

______________________________________________________________________

## 5. Automatic Differentiation

Vx provides semantic primitives for program transformations related to calculus.

- `grad(f)`: Yields a new function $f'$ which computes the gradients of $f$.
- `vjp(f)`: Vector-Jacobian Product. Propagates gradients backwards (Reverse-Mode AD).
- `jvp(f)`: Jacobian-Vector Product. Propagates tangents forwards (Forward-Mode AD).

**Operational Rule:**
When $g = \\text{grad}(f)$ is evaluated, the compiler generates an adjoint computation graph for $f$, tracing linear data consumption (as verified by the borrow checker). Linear variables are consumed exactly once in the forward pass and uniquely referenced in the reverse pass.
