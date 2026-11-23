# Pending borrows for kernels that start early

Issues: #3677 (this design), #3672 (starting a kernel early on an H100), #3675 (the first part:
a wait at the start of each kernel).

On an H100 or a later NVIDIA GPU, a kernel can start while the kernel before it on the stream
still runs. This note gives a kernel a way to say which of its tensors the kernel before it may
still use. The kernel can then read its other tensors before it waits.

## The problem: a borrow does not order two kernels

The borrow checker reads a program in program order:

```vx
rmsnorm(&mut h, &x);      // writes h
matvec(&mut y, &w, &h);   // reads w and h
```

The `&mut h` of `rmsnorm` ends when the call returns. A normal launch keeps this order on the
GPU, because `matvec` starts after `rmsnorm` ends. A kernel that starts early can run while
`rmsnorm` still writes `h`. So the order that the borrows rely on holds only after the wait in
`matvec` (`griddepcontrol.wait`).

#3675 puts the wait before the first read or write of global memory, so no program can see the
overlap. Only the launch overlaps, and the gain is small. On one H200, a decode step of glm-vx
took 1269.0 us, against 1284.5 us with normal launches. That is about 0.5 us for each pair of
kernels that can overlap. To gain more, a kernel must do work before its wait, for example read
its weights. `matvec` can read `w` before the wait, but not `h`. Its signature cannot tell them
apart.

## The memory model

- The end of a kernel releases all its writes. The wait of the next kernel acquires them. So the
  end of the kernel before happens before each operation after the wait.
- Before the wait, the two kernels have no order, as two threads with no synchronization. So the
  code before the wait must have no data race with the kernel before:
  - It reads only tensors that the kernel before does not write.
  - It writes only tensors that the kernel before does not read or write.
- `griddepcontrol.launch_dependents` is not a release. It only lets the GPU schedule the next
  kernel. Each kernel runs it after its own wait. So the kernel two launches back ended before
  the next kernel starts, and only the kernel just before matters.
- This assumes that a kernel sees the writes of each kernel that ended before its launch,
  although it did not wait for that kernel. The CUDA documentation does not say this. #3719 is
  a litmus test for it on an H100 or an H200, and a Lean model of the argument.

## The type

`Pending<&T>` is a reference to a tensor that the kernel before may still write.
`Pending<&mut T>` is a reference to a tensor that the kernel before may still read or write.

- A `Pending` has no index. So a read or a write of an element before the wait is a type error,
  and the borrow checker needs no new rule.
- `.wait()` waits for the kernel before, and gives the `&T` or the `&mut T`. Each lane that calls
  it waits. A second call costs little.
- A caller makes one with `Pending(&h)` or `Pending(&mut h)`. Vx has no implicit conversions, so
  `&h` alone does not become a `Pending`. `Pending(..)` is safe, because it allows less than the
  reference. It also shows the reader of the call which tensors order the two kernels.
- Before the wait, a kernel can read `.len()` and the shape of a `Pending`. The sizes come
  with the launch, as kernel parameters (`docs/lang/abi.md`), so the kernel before does not
  write them.
- A kernel can pass a `Pending` to a function whose parameter is a `Pending`. A struct cannot
  hold a `Pending`.
- A `Pending` is a parameter of a function, as `Distinct` is. So the claim is in the signature.
  `&Distinct<T>` has the claim inside the borrow, because it is about the values. `Pending` is
  around the borrow, because it says when the reference can be used.

A sketch, with a tile of weights that loads while the kernel before ends:

```vx
pub fn matvec(y : &mut Tensor<f32, [?], Memory::HBM>, w : &Tensor<f32, [?, 256], Memory::HBM>,
h : Pending<&Tensor<f32, [256], Memory::HBM>>, rows : i32) -> void {
  spawn on(Topology::Device) {
    for b in 0..rows / 64 across blocks {
      let mut tile = unsafe { Tensor<f32, [64, 256], Memory::SMEM>::uninit() };
      let copy = lanes::copy_async(&mut tile, &w, [64 * b, 0]);  // w: no wait
      let h = h.wait();                                          // h: after the wait
      copy.wait();
      // The lanes multiply tile by h here.
    }
  }
  return;
}
```

## What the compiler does

- The wait goes where the program calls `.wait()`. The compiler does not move it out of a loop,
  as it moves the wait of #3675. In the sketch, a wait above the loop would end the overlap
  with `copy_async`. `launch_dependents` follows each wait.
- A return that no wait comes before gets a wait, as in #3675. A kernel must not end before the
  kernel before it.
- A `Pending` parameter keeps both safeguards of #3675. It does not get `readonly`, and its
  pointer is an operand of each wait. Without them, LLVM can join a load below the wait with a
  load of the same address above it, also through `ld.global.nc`.
- The `&` parameters that are not `Pending` keep `noalias readonly`, so their loads keep the
  read-only cache (`ld.global.nc`). This is safe only because the launch check makes sure that
  the kernel before does not write them.
- The payload and the manifest mark each `Pending` parameter. The manifest already gives
  `"mutable"` for each parameter.
- A kernel with no `Pending` parameter works as in #3675: with `VX_GPU_PDL=1`, all its tensors
  are pending, and the wait comes before the first access.

## Who chooses an early launch

The launcher chooses for each launch. Take a kernel A, and a kernel B that is the next launch on
the same stream. The launcher knows the tensors of both by address. B starts early only when
this is true for each tensor of B that is not `Pending`:

- If B reads it, A does not write any byte of it.
- If B writes it, A does not read or write any byte of it.

Otherwise B launches normally. B also launches normally after any other operation on the
stream, such as a copy or a kernel that is not from Vx. For A, an `&` to atomics counts as a
write.

- `runtime/cuda_dispatch.cpp` keeps the tensors of the last launch on each stream.
- glm-vx's launcher makes the same check from the manifest.

A check at run time sees the real addresses, so it is exact. It works for calls in loops and
across functions, and for a launch from Python with no `unsafe` promise. It costs a few compares
for each launch, and a launch costs microseconds. Under a CUDA graph, it runs once, at capture.
A check in the compiler, from the borrows of the two calls, would see only two calls in one
body.

## The rules for a parallel compiler

[`language_rules_for_a_parallel_compiler.md`](../language_rules_for_a_parallel_compiler.md)
holds:

- **A signature is the whole contract.** `Pending` is in the signature. The check of a body
  needs only that body and the signatures.
- **No implicit conversions.** The caller writes `Pending(..)`.
- **A body declares nothing global.** The choice of launch is at run time, so the code of a
  kernel never depends on its callers.

## What it does not cover

- A release before the end of the kernel before. On an H100, a wait is for the whole kernel, and
  nothing marks one tensor as ready. A kernel that needs part of `h` early needs flags in memory.
- A prefetch. `lanes::prefetch` (#3672) loads no value, so it needs no `Pending`. A prefetch
  before the wait is always safe.
- More than one GPU.

## Order of work

1. `lanes::prefetch`, before the wait. It shows how much the work before the wait is worth.
1. `Pending` in the type checker: the type, `.wait()` and `Pending(..)`.
1. Code generation: the waits, `readonly` on the other `&`, the payload and the manifest.
1. The launch check, in `cuda_dispatch.cpp` and in glm-vx's launcher.
1. glm-vx times its decode step on one H200 and on 8.

The method stays `.wait()`, as on a `Token` and on the handle of `copy_async`. Each one waits
until something is ready.

Open questions: the name of the type (`Pending`, `AfterWait`), and whether `VX_GPU_PDL` and
`VX_CUDA_PDL` stay switches after step 5.
