# Vx Operational Semantics & Memory Model

This document specifies the runtime evaluation, spatial computing execution model, and memory rules of the Vx language.

## 1. Spatial Computing Model

Vx does not assume a single, flat instruction stream executing on a monolithic CPU. The language intrinsically models distributed, heterogeneous topologies (e.g., CPU, GPU, NPU, AccCore).

### 1.1 The `spawn on(Topology)` Operation

The `spawn on(Topology)` expression transitions the execution context from the current hardware unit to the specified hardware topology.

**Operational Rule:**
Let $E[ \\text{spawn on}(\\tau) { B } ]$ be a program state evaluated on an active hardware unit $\\rho$.

1. The runtime checks if the topology $\\tau$ is reachable from $\\rho$.
1. The runtime reserves resources on $\\tau$.
1. The block $B$ runs on $\\tau$ and produces a value $v$.
1. The spawning unit $\\rho$ continues with $v$, which lives on $\\tau$ (type `Pinned<T, τ>`). Everything after the spawn observes $B$ as complete. The runtime may overlap $B$ with later work on $\\rho$ as long as $\\rho$ cannot tell the difference; see [`../spawn_on.md`](../spawn_on.md).
1. All variables captured within $B$ that are not in a shared or transferred memory space relative to $\\tau$ will result in a compiler error.
1. **Index scope rule**: If $\\tau$ contains an index expression (e.g., `NPU[i]` or `NPU[0..4]`), the index expression is evaluated in the **calling** scope $\\rho$, not in $\\tau$. This ensures loop variables and other outer-scope identifiers are accessible.

### 1.2 The `unroll across(Topology)` Operation

`unroll across` is a spatial loop parallelizer. It takes a physical or logical topology slice (e.g., `Topology::NPU[0..4]`) and spatially duplicates the inner block $B$.

**Operational Rule:**
`unroll across(Topology::Unit[A..B]) { |id| B }` evaluates to parallel `spawn on` instances:
`spawn on(Topology::Unit[A]) { let id = A; B } || ... || spawn on(Topology::Unit[B-1]) { let id = B-1; B }`.

______________________________________________________________________

## 2. Memory Model & Transfers

Standard languages employ a flat memory model (`*mut T`). Vx uses a partitioned address space memory model: a value's memory space is part of its type. For a tensor that is the placement slot, `Tensor<f32, [4, 4], Memory::CPU_DRAM>`, which may equally be written as the device that holds the space, `Topology::NPU[0]`. `Ref<T, MemorySpace>` says the same thing for values that are not tensors.

### 2.1 Spatial Isolation

If variable $v$ is of type `Tensor<f32, [4, 4], Memory::CPU_DRAM>` and execution context is `Topology::NPU[0]`, attempting to dereference $v$ triggers a **spatial fault** (caught statically at compile-time by the Semantic Analyzer).

### 2.2 The `transfer` Primitive

Data movement across distinct memory hierarchies is explicit.
`let local_data = transfer(host_data, Memory::NPUHBM);`

**Operational Rule:**

1. Given source reference $S$ residing in space $M_1$.
1. Evaluates to a target reference $T$ in space $M_2$.
1. The runtime schedules an asynchronous or synchronous DMA transfer from $M_1$ to $M_2$.
1. Linearity: If $S$ represents uniquely owned data, the `transfer` consumes $S$. Future use of $S$ in space $M_1$ is invalid.
1. Borrowed source: If $S$ is a borrowed tensor (`&` or `&mut`), the `transfer` copies it. $T$ is a new tensor that the caller owns, and the owner of $S$ keeps $S$. Over a `: shared` edge, the compiler may skip the copy of a `&` parameter when every use of $T$ only reads it. The program gives the same values either way.

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

> Which spaces exist at all is decided by the machine and host models a compilation is given: see
> [`hosts_and_machines.md`](hosts_and_machines.md). No host is assumed — a program that stages
> through `Memory::CPU_DRAM` while a machine model is in force must name one (`E6014`), because
> `--machine` describes an accelerator and says nothing about the machine it hangs off.

### 2.4 Data races

A data race is two accesses to one place from two tasks at the same time. At least one of them writes, and at least one of them is not atomic.

- Safe code has no data races. The borrow checker and the `Send` and `Sync` rules refuse a program that can have one.
- A data race in `unsafe` code is undefined behavior. The compiler may assume that it does not occur, so the program can give any result.

Tasks that must write one place share an atomic, such as `AtomicI32` in `core::sync::atomic`. The rules are S5, S6 and U1 to U4 in [`cpu_tasks_send_sync_and_shared_atomics.md`](../discussions/cpu_tasks_send_sync_and_shared_atomics.md).

### 2.5 Atomic orders

Each operation on an atomic takes an `AtomicOrdering`. The orders are those of C++ and Rust, with one exception for `Relaxed` reads.

| Order | Meaning |
|---|---|
| `Relaxed` | Only the operation itself is atomic. It orders no other access. The compiler makes a `Relaxed` read an `Acquire` read (see below). |
| `Release` | For a store, or the write of a read-modify-write. The accesses before it in its task happen before the accesses after an `Acquire` read that reads its value. |
| `Acquire` | For a load, or the read of a read-modify-write. It is the other half of `Release`. |
| `AcqRel` | `Acquire` for the read and `Release` for the write of a read-modify-write. |
| `SeqCst` | `AcqRel`, and all tasks see the `SeqCst` operations in one order. |

**A `Relaxed` read is an `Acquire` read.** C++ and Rust allow load buffering. In it, a relaxed read takes its value from a write that comes later in its own thread, through a cycle with another thread. In such a cycle a value can come "out of thin air", and no known proof method rules it out. The Vx memory model forbids the cycle, as RC11 does. AArch64 and PTX allow it. So the compiler makes these reads `Acquire` reads, in host code and in device code:

- a `Relaxed` load,
- the read of a `Relaxed` read-modify-write (a `Release` one becomes `AcqRel`),
- both the success and the failure read of a `Relaxed` compare-exchange.

A `Relaxed` store stays relaxed. On x86 the change costs nothing, because each load there is already an acquire load. On AArch64 and on GPUs, a `Relaxed` read costs as much as an `Acquire` read. Metal atomics on the Apple GPU take only `memory_order_relaxed`, so there the read stays relaxed and an acquire fence follows it.

**`-fweak-relaxed-atomics`** keeps `Relaxed` reads relaxed, for the whole program. A program compiled with it may show load buffering (`formal/memory_model/tests/load_buffering_relaxed.litmus`). The default is `-fno-weak-relaxed-atomics`. When a command has both, the last one wins.

**Do not rely on a `Relaxed` read for order.** The compiler adds the `Acquire` only to rule out values out of thin air. A program that needs an access to happen before another must ask for `Acquire` and `Release`. Such a program is correct with and without the flag. The proofs in `formal/memory_model` also treat `rlx` as relaxed, so they hold in both cases.

**ThreadSanitizer cannot find a missing `Acquire` on a `Relaxed` read** in a default compile, because the read is an `Acquire` read in the program. To check that code does not depend on it, compile with both options: `vxc -fweak-relaxed-atomics --sanitize=thread`.

Other rules:

- An order that Rust panics on fails an `assert`. These are `Release` or `AcqRel` for a load, and `Acquire` or `AcqRel` for a store. They are also `Release` or `AcqRel` as the failure order of a compare-exchange, and `Relaxed` for a fence.
- `compare_exchange` may run with a stronger order than the one asked for. `compare_exchange_weak` is `compare_exchange`, and it never fails spuriously.
- On a GPU, an atomic has a scope. Two atomics race unless the scope of each one includes the thread of the other, as in the PTX model.

The model is `formal/memory_model/vx.cat`. [`formal/memory_model/README.md`](../../formal/memory_model/README.md) has its litmus tests and Lean proofs.

### 2.6 Drops

When its owner goes away, a value is *dropped*: the drop releases what the value owns. A moved value is not dropped, because its new owner drops it. An assignment drops the old value of the variable. The checker writes each drop into the program, so both code generators drop a value at the same point.

- A value that only holds memory, such as a tensor or a `Vec` of numbers, drops after its last use.
- A value whose type implements `Drop` waits for the end of its block. The values of one block drop in reverse order of declaration. The `drop` method of a value runs first, and then its fields drop in the order of their declaration. An enum drops the payload of the variant that it holds.
- `core::mem::drop(x)` drops `x` at once.

**A pattern that takes a value owns all of it.** At the end of its scope, the pattern drops the names that it binds. Then it drops the parts that it does not name, such as a `_` part.

- A `match` arm takes the value when its pattern moves a part out of it. `match t { Two::Pair(a, _) => .. }` moves all of `t` into the arm, so code after the `match` cannot use `t`. At its end, the arm drops `a` and then the `_` part.
- A `match` in which no arm moves a part, such as `match t { Two::Pair(_, _) => .. }`, leaves `t` as it is. Then `t` drops at the end of its block.
- Each arm of a `match` on a value with no name, such as `match make() { .. }`, takes the value.
- In a `let`, `_` is a name like any other, and its scope is the rest of the block. `let _ = e;` drops the value at the same point as `let _x = e;`. `let _ = a;` moves `a`.

<!-- vx-doctest: run -->

```rust
import core::mem;

struct Noisy {
    id : i32,
}

impl Drop for Noisy {
    fn drop(self : &mut Noisy) -> void {
        println!("drop {}", self.id);
    }
}

enum Two {
    Pair(Noisy, Noisy),
    Nothing,
}

fn main() -> i32 {
    let t = Two::Pair(Noisy { id : 1 }, Noisy { id : 2 });
    match t {
        Two::Pair(a, _) => {
            println!("the arm uses {}", a.id);
        }
        Two::Nothing => {}
    }
    let _ = Noisy { id : 3 };
    let _x = Noisy { id : 4 };
    drop(Noisy { id : 5 });
    println!("end");
    return 0;
}
```

It prints:

```text
the arm uses 1
drop 1
drop 2
drop 5
end
drop 4
drop 3
```

**Rust decides by the place that a pattern reads.** In Rust, `_` never moves a value. So the time of a drop depends on whether the pattern reads a variable or a value with no name:

- `match t { Pair(a, _) => .. }` leaves the `_` part in `t`. The `_` part drops at the end of the block of `t`.
- `match make() { Pair(a, _) => .. }` drops the `_` part at the end of the `match`.
- `let _ = make();` drops the value at the end of the statement, and `let _ = a;` moves nothing.

Thus `let _ = m.lock();` releases a lock at once, and `let _g = m.lock();` holds it to the end of the block. `rustc` refuses the first form by default, with the `let_underscore_lock` lint. In Vx, the pattern and its scope decide when a value drops. A `_` part and a named part drop at the same point.

[`drop_semantics.md`](../implementation_plans/drop_semantics.md) has the design of drops.

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
- An `assert` over a placement query (`x.topology() == Some(Topology::T)`, see the type system document) is decided by the checker: a true one is removed, a false one fails the compile (E8002).

**Operational Rule:**
If $E[\\text{comptime} { B }]$ evaluates to $v$, the AST is strictly replaced by the literal or reduced expression $v$ before lowering to the intermediate representation (MLIR).

A block in statement position produces no value and is removed outright. Neither back end
ever sees a `comptime` block, because the replacement happens in the checker.

**The block must be able to run, and must leave nothing behind.** These follow from the rule
above rather than adding to it, and each is refused rather than quietly lowered as run-time
code, which is what used to happen:

- A block the evaluator cannot finish is an error (E3033). There is no half-running it: a
  block that was emitted as ordinary code instead is how a loop or a call the evaluator
  skipped stayed invisible, and a build could pass while proving nothing.
- A block that writes to a variable declared outside it is an error (E3033). The block
  disappears, so the write would disappear with it.
- A block inside another block is an error (E3034). The outer block already runs while
  compiling, so the inner one asks for nothing more.
- A block may read a value known only when the program runs, when neither its value nor its
  effects depend on that value.
- The compiler decides, while compiling, every check that the program makes while it runs: an
  `assert`, and an integer `/` or `%`. The division stops the program on a divisor of zero,
  and on the minimum value divided by -1. A check left for run time would never run, because
  the block disappears.
  - A false `assert` on a path that always runs is an error (E8002).
  - Any other check that the compiler cannot show to pass is an error (E3033). This is a check
    that fails on a path that runs only for some run-time values, such as one branch of an
    `if` on a run-time value. It is also a check that depends on a run-time value.
- An integer result outside its type has no compile-time value, so a block whose value depends
  on it is an error (E3033). At run time the program wraps the result and does not stop, so a
  block whose value does not depend on it still folds.
- The evaluator runs at most 1,000,000 loop iterations in a block, in all, and at most 256
  nested calls. More is an error (E8005, E8004).

**Comptime lambdas.** `comptime { .. }` is also Vx's block-expression form, so it is how a
closure with more than one statement is written. Such a lambda is a function that runs while
compiling — nearer C++'s `consteval` than its `constexpr`:

- Its body is **not** folded where the lambda is written. Its parameters have no values yet.
- Every **call** to it must fold. A call whose arguments are not all known while compiling is
  an error; there is no falling back to a run-time call, which is where this is stricter than
  `constexpr`.
- Once every call has folded, nothing refers to the lambda, so its binding and its generated
  body are both dropped and it leaves no trace. A lambda nobody calls is an ordinary unused
  variable.

A closure body starts a fresh compile-time context, so a lambda whose body is a `comptime`
block, written inside another such block, is two functions rather than a nested block.

The lineage is Zig's `comptime` rather than C++'s `constexpr`. A `constexpr` function only
*may* be evaluated while compiling and silently becomes an ordinary call when it cannot be,
which is the reason C++ later grew `consteval`. Vx refuses instead, so a block that did not
run while compiling can never be mistaken for one that did.

______________________________________________________________________

## 5. Automatic Differentiation

Vx provides semantic primitives for program transformations related to calculus.

- `grad(f)`: Yields a new function $f'$ which computes the gradients of $f$.
- `vjp(f)`: Vector-Jacobian Product. Propagates gradients backwards (Reverse-Mode AD).
- `jvp(f)`: Jacobian-Vector Product. Propagates tangents forwards (Forward-Mode AD).

**Operational Rule:**
When $g = \\text{grad}(f)$ is evaluated, the compiler generates an adjoint computation graph for $f$, tracing linear data consumption (as verified by the borrow checker). Linear variables are consumed exactly once in the forward pass and uniquely referenced in the reverse pass.
