# `spawn on`: sequential meaning, overlapped execution

> **Status.** Decided 2026-09-25. The language rules in §2 are what the checker and the
> lowering implement today. The overlap in §3 is designed and still to be built: every runtime
> blocks inside dispatch, and the compiler never emits a wait. §6 lists that work.
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
- **CPU regions stay inlined.** There is nothing to dispatch to. If the host should ever run
  two regions at the same time, that is a separate keyword; `unroll across` is reserved for
  that shape.

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
| Runtime dispatch (`cuda`, `npu`, host) | blocks before returning | returns a future id |
| `vx_plugin_await_future` | no-op in every backend | blocks on that id |
| Wait insertion in the compiler | none | at first host use (§3) |
| Value of a device spawn | the `vx.launch` result is replaced by `0` or `undef` in `LaunchOpLowering` | the kernel's result, through the future |
| Runtime device index `NPU[i]` | W1030, device 0 | separate issue |

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
