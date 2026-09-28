# Freeing buffers on every device

> **Status: proposal, not implemented.** This note compares two ways to decide when a buffer is
> freed, and says which workloads each one serves. It is for issues #642 and #340, and touches
> #495 and #821.

## The problem

A Vx program allocates buffers in several places: host memory, a GPU's memory, an NPU's memory,
a remote worker's memory. Each buffer has to be freed exactly once, after its last use, by the
same backend that allocated it. Free too early and the program reads freed memory; free too late
and a loop runs the device out of memory; never free and the program leaks.

Two separate questions hide in "when is it freed":

1. **When is the buffer dead?** This is a question about the program: where is its last use?
1. **How is the free carried out?** This is a question about the device: a host `free`, a
   `cudaFree`, a call to a remote worker, and whether it may run before the device has finished
   the work that reads the buffer.

This note keeps the two apart. Question 2 has one answer (below). Question 1 has two good
answers, and they suit different workloads.

## What Vx does today

- **Host programs:** `vx-free-heap-buffers` runs MLIR's `buffer-deallocation-pipeline` and frees
  each heap buffer after its last use (#826). Small buffers that stay inside their function are on
  the stack instead (#820).
- **Programs that place data** on a device or in a memory space are skipped as a whole module.
  Of the 41 test programs in `tests/backend/pass` and `tests/optimizations/pass` that the pass
  skips, 40 are skipped because a buffer goes through `builtin.unrealized_conversion_cast` into an
  LLVM descriptor, made by the lowering of `vx.launch` and `vx.transfer`. By then the analysis
  cannot tell whether the callee only reads the buffer or keeps it.
- **Device buffers** are freed by the transfer lowering itself (`TransferToPluginLowering` in
  `src/dialect/VxLowering.cpp`), which calls `vx_plugin_free` at a place picked from where the
  uses are and which blocks dominate the exits. Its own comments say this is wrong in both
  directions in some cases (#340).
- **Kernel launches wait.** The CUDA plugin calls `cudaDeviceSynchronize` after every
  `cuLaunchKernel`, so a free placed right after a launch is safe today. The plugin interface
  already has futures (`vx_plugin_release_future`), so this will not stay true.

## How the free is carried out: in the device's queue

CUDA programmers moved from `cudaFree`, which waits for the whole device, to `cudaFreeAsync(p, stream)` (CUDA 11.2). A stream is a queue of device work. `cudaFreeAsync` does not free when it
is called; it adds the free to the queue, so it runs after the work already queued. The host
never waits, and the memory goes back to a pool, so allocating in a loop is cheap. MLIR's
`gpu.dealloc` with async tokens is the same idea. PyTorch, RAPIDS RMM and SYCL buffers all build
on it.

Vx should do the same, whichever design below decides the point of death:

- The compiler writes one free op, placed at the buffer's last use in program order.
- The lowering for each place turns it into the right call: `free` for host memory, a queued
  `vx_plugin_free` for a device, a message for a remote worker.
- A buffer used by more than one device or queue is freed only after each of them signals it is
  done. The compiler inserts this wait. In PyTorch the programmer must call
  `tensor.record_stream(s)` by hand, and forgetting it is a well-known use-after-free.
  `two_gpu_devices.vx` already has this shape.

With frees in the queue, placing the free after the last use is correct even when launches stop
waiting. The hand-written frees in the transfer lowering then go away.

## Design A: the frontend decides, from the program's own lifetimes

The frontend already tracks lifetimes for borrows (`BorrowCx::live_borrows` in
`src/hir/borrow_cx.rs`). A tensor bound to a variable gets a drop at the end of its life, as a
Rust value does. Moving it into a struct, returning it, or pushing it into a `Vec` moves the
ownership, and the new owner drops it later. This is the same work #495 asks for `Vec`, which has
no drop today (`TYPE_NEEDS_DROP` exists and is never set).

**Strengths**

- It sees values the programmer named, however they travel: through struct fields, returns,
  `Vec`s, and between devices. These are exactly the cases the MLIR analysis cannot follow.
- It can explain itself in the programmer's terms: "`w` lives until line 40 because the loop on
  line 32 reads it."
- The programmer can predict where memory is released, and can shorten a lifetime on purpose.

**Weaknesses**

- It cannot see buffers the compiler makes after the frontend: the result of an elementwise op,
  of a `map`, a transpose, tiling scratch. These never have a name in the source.
- It needs drop for tensors and `Vec` in the language before it frees anything. The rules are
  settled (see "Decisions" below); the work is building them.
- It frees at the end of a variable's life in the source, which can be later than the last use
  in the optimized code.

## Design B: MLIR decides, after the `vx` ops say what they do to memory

Give `vx.launch`, `vx.transfer` and the other `vx` ops real memory effects: which operands they
read, which they write, and that they keep no pointer. Then run `buffer-deallocation-pipeline`
before `vx-to-llvm` lowers them into LLVM calls and casts, while the ops are still visible.

**Strengths**

- It frees every buffer the compiler made, named or not, including ones created by lowering.
- It works on the optimized code: after fusion removes a buffer there is nothing to free, and the
  free lands at the real last use, which can be earlier than the end of the variable's life.
  That lowers peak memory.
- Most of it exists already: host programs use it today (#826), and the change is effects on the
  `vx` ops plus running the pass earlier. #821 wants the same effects for its checked rule.

**Weaknesses**

- When ownership is unclear it copies the buffer (`bufferization.clone`) rather than guess. For a
  host array that is cheap; for a large device tensor it is a surprise copy across a bus.
- It cannot follow a buffer stored into a struct field or a `Vec`; those still need a rule.
- It analyses the whole module and gives no reason the programmer can read when a buffer lives
  longer than expected.

## Which workloads each design serves

| Workload | Better served by | Why |
|---|---|---|
| Inference: weights moved to a device once and read by every step of a loop | A | A named value with a long life; the free belongs after the loop, and the programmer should be able to see that |
| Training: activations kept from the forward pass for the backward pass | A | Lifetimes span phases of the program and pass through structs |
| Tensors returned from functions, kept in structs or `Vec`s | A | Ownership moves; B cannot follow a struct field (#828 was this bug on the stack side) |
| Pipelines across devices: a GPU hands a tensor to another GPU or an NPU | A | A move between devices is an ownership change the language can state |
| Devices with little memory (NPU, ANE) where the programmer plans residency | A | The free has to be where the programmer expects it |
| Elementwise and `map`-heavy numeric code | B | Most buffers are compiler temporaries with no name |
| Small helper functions called in hot loops (the per-call leak in #642) | B | Short-lived buffers, freed right after last use, already working on the host |
| Tiled or fused kernels with scratch buffers | B | Scratch is made by lowering, after the frontend is done |
| Code where peak memory matters more than predictability | B | It frees at the last use in the optimized code |

## Recommendation: both, with one owner per buffer

Neither design covers every workload, and they split along a clear line: **the frontend owns
buffers the programmer named; MLIR owns buffers the compiler made.** Each allocation is owned by
exactly one of them. A buffer that the frontend owns is marked when it is created, and the MLIR
analysis leaves it alone; everything unmarked is MLIR's. A buffer freed twice or never freed is
then a compiler bug with one place to look, and deserves an `assert`.

Suggested order, each step useful on its own:

1. **Memory effects on the `vx` ops.** Both designs need them, and #821's checked rule becomes
   possible.
1. **One free op, lowered per place, into the device's queue.** Replaces the hand-written frees
   in the transfer lowering, fixes the early and late frees of #340, and stays correct once
   launches stop waiting.
1. **Design B for programs that place data.** Removes the whole-module skip; temporaries in
   those programs stop leaking.
1. **Design A for named values**, together with drop for `Vec` (#495). This step needs the
   most new work in the frontend, so it goes last.

## Decisions

- **`let b = a;` moves the buffer**, as in Rust, unless the type's `Copy` implementation does
  something custom. There is no reference count, so an assignment costs nothing.
- **The programmer can end a lifetime early** with `drop(t)`, as in Rust, and can ask the
  compiler why a buffer is still alive.

## Open questions

- How does a remote worker report that its queue has finished with a buffer, so a free on
  another device can wait for it?
