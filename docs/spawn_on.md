# `spawn on`: sequential meaning, overlapped execution

> **Status.** Decided 2026-09-25. The language rules in §2 are what the checker and the
> lowering implement today. The overlap in §3 is designed and only partly built: the `cuda`
> runtime returns once a launch is queued, the other runtimes block inside dispatch, and the
> compiler never emits a wait. §6 lists that work.
>
> An earlier version of this file proposed `spawn on` returning `Future<Pinned<T, τ>>` with an
> explicit `.await`. §4 says why that was rejected. The old text is in git history.

## 1. The question

`spawn on(τ) { B }` runs the block `B` on topology `τ` and gives the result back as
`Pinned<T, τ>`. Should the caller wait for `B` to finish, or continue at once and join later?

The worry behind "continue at once" is real. One host thread feeding eight NPUs cannot sit
idle on `NPU[0]` while `NPU[1]` to `NPU[7]` have nothing to do. The question is whether the
*language* has to say "async" for the *machine* to get that overlap. It does not, and saying
it would cost more than it gives.

## 2. What the language promises

A Vx program means what it says, in the order it says it. `spawn on` follows that rule.

1. `B` is type-checked and runs on `τ`. A device index such as `NPU[i]` is evaluated in the
   caller's scope.
1. The result is a value that lives on `τ`, typed `Pinned<T, τ>`. There is no handle type and
   no future type.
1. A tensor the region hands out moves to the function around it, which frees it. A variable
   from outside the region keeps its own type. A region on a device that creates the tensor it
   hands out has it created before the region, in `τ`'s memory, since the host cannot free what
   a device allocates; when that is not possible, the region gets E6030.
1. Every statement after the spawn sees `B` as finished. If `B` wrote a tensor the host can
   see, the host reads the written value. Reading the result on another topology needs the
   same visibility or `transfer` as any other placed value.
1. A region is scope structure. Tiles allocated inside it are released at its end, reads
   inside it keep outer tiles alive until its end (E6010), and a function's capacity summary
   takes the max across its regions.

This is the model the operational, denotational and axiomatic treatments in
[`semantics/`](semantics/) already write down: the body runs to a value `v`, and the host
continues with `Pinned(v, τ)`.

## 3. What the implementation may do

The runtime may overlap `B` with the host's later work as long as the host cannot tell the
difference. A CPU reorders instructions under the same rule, and CUDA streams work the same
way: a launch returns at once, and the first read of the result waits.

Concretely:

- **Dispatch returns a future id at once.** `vx_plugin_dispatch_async` already promises this
  in [`vx_hardware_runtime.h`](../include/vx_hardware_runtime.h) ("Returns a Future/Event ID
  immediately"), and every backend breaks the promise today.
- **The compiler emits `vx_plugin_await_future` at the first host use of anything the region
  produced or wrote.** The checker already decides every such use, so the hook points exist:
  - `transfer(r, Memory::X)` of the result: the wait goes before the copy.
  - A direct read the checker allowed because the space is visible from the host or declared
    `managed: cached` (unified memory): the wait goes before the read. `print(r)` is one of
    these.
  - A use of `r` inside a later region on the same device: no host wait. The runtime queues
    the second kernel behind the first, in stream order.
  - A use of `r` inside a region on another device: the transfer between them carries a
    device-side event wait. Still no host wait.
- **Release.** `vx_plugin_release_future` when the value goes out of scope.
- **Conservative default.** When the compiler cannot tie a host read to the region that made
  the value — the region wrote through memory it shares with the host and the recorded region
  traffic does not name it — it joins at region exit. The answer stays right; only the overlap
  is lost.
- **CPU regions stay inlined.** There is nothing to dispatch to. Host work that runs at the
  same time is a `tasks` block
  ([`waiting_tasks_channels_and_task_errors.md`](discussions/waiting_tasks_channels_and_task_errors.md)),
  not a `spawn on`.

Why the wait points are complete: the host cannot compute with a `Pinned<T, τ>` (E3004),
cannot read a tensor in an `explicit` space without `transfer` (E6003), and every allowed read
goes through one checker path. The set of wait points is the set of allowed reads. There is no
way for the host to touch a device result the compiler did not see.

## 4. Why not `Future` and `.await`

The rejected design: `let f: Future<Pinned<T, τ>> = spawn on(τ) { ... };` then
`f.await.to_host()`, with a compile error for a future dropped without an await.

1. **It fixes a runtime problem with syntax.** The stall the design worried about is the
   `cudaDeviceSynchronize` at the end of dispatch in `runtime/cuda_dispatch.cpp` and the
   `waitUntilCompleted` in `runtime/npu_dispatch.mm`. Moving those into `await_future` and
   inserting the wait at first use gives the fan-out with no new token.
1. **`.await` repeats `transfer`.** Every device result would read `.await.to_host()`, and the
   `.await` alone gives a value the host still cannot use. One boundary is enough.
1. **It gives up scope structure.** Tile release at region end, E6010, and the capacity max
   rule all rely on the region being joined at exit.
   [`cross_call_capacity_implementation_plan.md`](cross_call_capacity_implementation_plan.md)
   prices the alternative: an outstanding task becomes a second edge kind whose operator is
   `+` across concurrent siblings. The backward pass of `spawn on` gets harder in the same way.
1. **It ties placement to concurrency.** X10's `at(p)` and Chapel's `on` are place shifts, and
   concurrency has its own word in both (`async`, `begin`). With a future, "run there and
   wait" needs `.await` noise, and "run two things on the host at once" has no spelling.
1. **It does not produce overlap by itself.** `let a = spawn on(NPU[0]) {..}.await;` followed
   by `let b = spawn on(NPU[1]) {..}.await;` runs the two in series. The as-if model overlaps
   them without the programmer doing anything.
1. **The dropped-future error is unneeded.** In the as-if model an unused result is dead code,
   and the region's effects still join before anything that depends on them.

## 5. Examples

Fan-out across two devices. This compiles today. It runs in series today and will overlap
once §3 lands, with no change to the program:

```vx
fn main() -> i32 {
    let a = spawn on(Topology::NPU[0]) { 40 };
    let b = spawn on(Topology::NPU[1]) { 2 };
    let ha = transfer(a, Memory::CPU_DRAM);
    let hb = transfer(b, Memory::CPU_DRAM);
    print(ha);
    print(hb);
    return 0;
}
```

Both launches enqueue. The first `transfer` waits for `NPU[0]`; `NPU[1]` is already running.

A host region with a side effect. It must print `2.5` under any implementation, and it does,
because a CPU region is inlined where it stands:

```vx
fn main() -> i32 {
    let mut a = Tensor<f32, [1, 1]>::new();
    a[0][0] = 1.5;
    spawn on(Topology::CPU) {
        a[0][0] = a[0][0] + 1.0;
    }
    print(a[0][0]);
    return 0;
}
```

The same program with `Topology::NPU[0]` and `a` in `Memory::NPU_HBM` is the unified-memory
case: the host read `a[0][0]` is allowed by visibility, so it is where the wait goes.

A loop `for i in 0..8 { spawn on(Topology::NPU[i]) { .. } }` warns W1030 today and targets
device 0, because the index has to be a compile-time constant. That gap is separate from this
decision.

## 6. What is built and what remains

| Piece | Today | Target |
| --- | --- | --- |
| Result type `Pinned<T, τ>`, index in caller scope, region joined at exit | built | unchanged |
| CPU region | inlined in place | unchanged |
| Runtime dispatch (`cuda`, `npu`, host) | `cuda` returns once the launch is queued (#25); the others block before returning | returns a future id |
| `vx_plugin_await_future` | `cuda` waits for every kernel that a GPU started; a no-op in the other backends | blocks on that id |
| Wait insertion in the compiler | none | at first host use (§3) |
| Value of a device spawn | the kernel stores it in a host slot and the host loads it after the launch; such a kernel gets no GPU device image and runs on the host | the slot staged to device memory, so the GPU twin can run |
| Runtime device index `NPU[i]` | W1030, device 0 | separate issue |
| `for i in a..b across <level>` (#252) | §8 | lane loops fuse, and per-lane arrays go in registers |

The `cuda` backend puts every launch and copy on the default stream, so the GPU runs them in
program order, and a copy to the host waits for the kernels before it. A fault shows at the next
wait: a copy to the host, a free, or the end of the program. That message names the last kernel
that the GPU started. `VX_CUDA_SYNC=1` waits after each launch, so that the message names the
kernel that faulted.

Order of work: the value stub first, since any model needs it; then non-blocking dispatch plus
a real `await_future` in the runtimes, with the wait emitted at region exit so nothing
observable changes and the plumbing is proven; then move the wait to first use. Issue #25
tracks the runtime wiring.

## 7. Documents this settles

- [`lang/semantics.md`](lang/semantics.md) §1.1, rules 3 and 4, now say what §2 says.
- [`lang/formal_semantics.md`](lang/formal_semantics.md) §3.1: the task handle in E-SPAWN is
  the runtime's future id. It never becomes a language value.
- [`lang/abi.md`](lang/abi.md) §3: a CPU region is inlined; the `async.execute` lowering is
  gone.
- [`scalable_plugin_system.md`](scalable_plugin_system.md): its `.await` sketches predate this
  decision.
- The status table in the top-level `README.md` and the note in [`tutorial.md`](tutorial.md).

Precedent: X10 `at`, Chapel `on`, CUDA streams, and PyTorch's eager mode, which is
asynchronous under the hood while the program reads as sequential.

## 8. The launch shape (#252)

A region on a GPU gets its launch shape from its loops:

<!-- vx-doctest: skip (needs a GPU topology) -->

```vx
spawn on(Topology::Gpu) {
  for row in 0..rows across blocks {
    for t in 0..256 across lanes { .. }
  }
}
```

- Each loop of blocks is a grid axis, and the lane and subgroup loops give the lanes of a
  block. The host computes each bound before the region. The levels come from the topology's
  `levels:` in the fleet file, and the NVIDIA files give `blocks`, `subgroups` and `lanes`.
- Only `nvptx64` launches the levels. On another arch, `-fauto-schedule-levels` runs each
  `across` loop as a plain `for` that carries its level as metadata (`fleet/README.md`).
- E6032 checks the place and the size of each `across` loop. It also checks that no pass
  touches what another pass writes (L4), and that every lane reaches the end of each lane loop
  (L5, L8, L10).
- An atomic operation races with no other one. It is a call to a function that says
  `ensures atomic(t)`, or an `mlir!` block with `ensures : [atomic(t)]`. Every access to `t` in
  the loop must then be atomic (#2139, `discussions/mlir_declared_writes_in_across.md`).
  `core::sync::atomic` has these functions for an integer or float element of a tensor in any
  memory, such as `atomic_fetch_add_u32(&counts, i, 1u32, AtomicOrdering::Relaxed)`.
- A call to a function that says `ensures writes_only()` accesses no memory, so it races with
  nothing. `scoped_fence(order, FenceScope::Device)` is one.
- Each lane loop ends with a barrier. Lane 0 runs the code between lane loops that writes
  memory.
- `lanes::sum`, `lanes::max` and their `_unordered` versions combine a tensor over the lanes of
  a block or a subgroup, and every lane gets the result (`docs/lane_reductions.md`).
- Code after the nest of blocks runs once, after every pass, in the same kernel. Each pass ends
  with an atomic add to a count that the compiler makes, between two fences for the device. The
  block of the pass that sees the last count runs the code as the body of a block, so it may hold
  lane loops. A nest with no passes runs no such code, and a nest with two axes of blocks cannot
  have it yet (`discussions/last_arriver_in_across_loops.md`).
- Delete each `across` and the program means the same. The CPU copy of the region runs one
  block of one lane, and a loop steps over the passes of the others.
- A size is not a kernel argument. The launch passes each size that is not a number after the
  kernel's arguments, and the payload names the sizes in `grid=` and `block=`.
- `--emit-ptx` gives the shape in its `.json` file (`lang/abi.md` §3.5).
- The rules L1 to L10 and N1 to N4 are in the design on
  [#252](https://github.com/vx-lang/Vx/issues/252#issuecomment-6321577866).
