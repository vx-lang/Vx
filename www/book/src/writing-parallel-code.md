# Writing parallel code

Vx has two kinds of parallel code. Tasks do different work at the same time, and
[Tasks and channels](tasks.md) explains them. An `across` loop does the same work on each element
of a tensor, on the blocks and lanes of a GPU, and [Parallel code on a GPU](gpu.md) explains it.
This chapter explains how to choose between the tools, and how to check that a parallel program is
correct. It is for programmers who come from Rust with tokio, from Go, from C with threads, or from
CUDA.

## Start with the program in order

Write the program without tasks first, and keep it. The version in order is the reference: run
both versions on the same input, and compare the results. A parallel program that is fast and
wrong is easy to write. A comparison finds the problem in the first run.

For an `across` loop, the version in order is the same program without the word `across`.

## Choose the shape of the work

Most parallel programs have one of four shapes:

- **The same work on each element.** Each pass of a loop works on its own elements of a tensor.
  Write it as an `across` loop, as [Loops across blocks and lanes](#loops-across-blocks-and-lanes)
  shows. This shape needs no tasks.
- **Split the data.** Each task takes one part of the input and gives one result. The tasks
  borrow the input from the function around them, and `join()` gives each result. The first
  program of [Tasks and channels](tasks.md) has this shape.
- **Workers take jobs.** When jobs take different times, an even split leaves some tasks idle.
  Give each worker the next job number from an atomic counter instead.
- **A pipeline.** Each stage is a task, and a channel connects each stage to the next one. A full
  channel makes a fast stage wait for a slow one.

In this program, four workers take job numbers from a counter. Each worker sends a result with
its job number on a channel (`book_workers_take_jobs_from_a_counter.vx`):

```rust
{{#include ../../../tests/backend/pass/book_workers_take_jobs_from_a_counter.vx:program}}
```

It prints:

```text
squares 0 1 4 9 16 25 36 49 64 81
each job ran once: true
```

The results arrive in any order. The job number puts each result in its place, so the output is
the same in each run.

## Choose the tool

| You need | Use |
| --- | --- |
| The same work on each element of a tensor | an `across` loop in a `spawn on` region |
| A sum, a maximum or a sort over the lanes of a block | `lanes::sum`, `lanes::max`, `lanes::sort` |
| Many passes that add to the same element of a tensor | `atomic_fetch_add_u32` and the other atomics on elements |
| A copy into a tile while the lanes work | `lanes::copy_async` |
| A faster `exp` or `tanh` in a GPU kernel | `fast_exp`, `fast_tanh` ([Exact and fast math](exact-and-fast-math.md)) |
| Work on several devices at the same time | a task for each device, with a `spawn on` in each task |
| The result of one task | `join()` on its handle |
| Values from many tasks | a channel |
| A counter or a flag that tasks share | an atomic |
| A change to more than one value at a time | a `Mutex` |
| Values from more than one channel | a `Selector` |
| A value that lives longer than its `tasks` block | an `Arc` |

Use a channel when you are not sure. A channel moves each value to one owner, so two tasks never
change the same data. A task can borrow the locals of the function around it, so most programs
need no `Arc`.

## Loops across blocks and lanes

A word after the range of a loop, `across blocks` or `across lanes(n)`, makes its passes run at the
same time on a GPU. A block runs on one part of the chip, and a lane is one thread of a block. This
program doubles each element of a tensor, and adds the number of its row, `y`
(`book_gpu_blocks_and_lanes.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_blocks_and_lanes.vx:program}}
```

It prints `corner 2, last row 65`. Give the compiler a machine file with a GPU:

```sh
vxc blocks.vx --machine "$HOME/.vx/current/fleet/h100-sxm.vx" --host default
```

On a computer with no GPU, each kernel runs on the threads of the CPU, and the result is the same.

An `across` loop is easier to get right than tasks:

- The compiler checks that two passes cannot write the same element. When they can, it gives the
  error E6032. So an `across` loop has no data race for a test to find.
- The meaning of the program does not depend on `across`. Remove each `across`, and the loops run
  one pass at a time with the same result.
- The lanes of a block can share a tile in `Memory::SMEM`. The end of each lane loop waits for all
  the lanes of the block, so the next lane loop can read what any lane wrote.

[Parallel code on a GPU](gpu.md) explains the rules of `across`, tiles, the `lanes::` routines,
atomics on elements, and copies in the background.

## Keep the result independent of the order

The order in which tasks run changes from run to run. Do not let the result depend on it:

- Give each result a number, and put it in its place, as the program above does.
- When jobs can conflict, run each job on a copy of the data from before the jobs started. Then
  go through the jobs in order, and run again each job that read data an earlier job changed.
  Parallel blockchain executors work this way. The starter project
  [parallel-batch](https://github.com/vx-lang/Vx-starters/tree/main/parallel-batch) shows it on
  a batch of transfers.
- Do not add floating-point numbers in the order that they arrive. The sum can change in the last
  bits. Store the numbers by job, and add them in job order.
- In an `across` loop, `lanes::sum` adds in the same order each time. `lanes::sum_unordered` lets the compiler
  choose the order, so the last bits of its sum can differ from `lanes::sum`.

## Choose a memory order

- A counter that only gives out numbers needs `Relaxed`.
- To give data to another task through a flag, store the flag with `Release`, and load it with
  `Acquire`. [Memory orders](tasks.md#memory-orders) has a program that does this.
- Use `SeqCst` only when two or more flags must be seen in one order by every task.

By default, Vx compiles a `Relaxed` load as an `Acquire` load. With `-fweak-relaxed-atomics`, the
load stays `Relaxed`. So a program that needs `Acquire` but asks for `Relaxed` can work on Vx and
fail elsewhere. Ask for the order that the program needs.

## Waits that do not end

- A `recv()` loop ends only after the program drops each `Sender`. Drop the `Sender` that made the
  clones, as the program above does with `drop(tx)`.
- `join()` and `lock()` cannot be cancelled. Do not hold a `Mutex` while a task waits for a channel.
- When each task waits and nothing can wake one, the runtime stops the program with
  `vx: deadlock: ...`. It does not hang.

## Test it

Run each test in these ways:

- With the default runtime, which gives each task an OS thread.
- With `VX_TASK_RUNTIME=shared`, which runs the tasks on a few OS threads.
- With `VX_TASK_SEED=1`, `2` and more. The tasks then run on one thread, and the seed chooses the
  order. The same seed gives the same order, so a failure happens again in the next run.
- With `vxc --sanitize=thread`, which finds data races through raw pointers that the compiler
  cannot see.

For an `across` loop, also run the program with `CUDA_VISIBLE_DEVICES=`. Then it runs the CPU copy
of each kernel, and the result must be the same as on the GPU.

## Coming from tokio

| Rust | Vx |
| --- | --- |
| `std::thread::scope`, `tokio::spawn` | a `tasks` block and `spawn` |
| a multi-thread tokio runtime | `VX_TASK_RUNTIME=shared`, with no change to the program |
| `async fn` and `.await` | a normal function: `recv()`, `join()` and `lock()` wait in place |
| `tokio::sync::mpsc` | `std::sync::channel` |
| `tokio::select!` | `Selector` |
| `AtomicU64` with `Ordering::Relaxed` | `AtomicU64` with `AtomicOrdering::Relaxed` |

Vx has no `async`. [Why Vx has no `async`](tasks.md#why-vx-has-no-async) explains this.

## Coming from CUDA

| CUDA | Vx |
| --- | --- |
| a kernel and the code that starts it | a `spawn on(Topology::Device)` region with `across` loops |
| `blockIdx.x` | the variable of a loop `across blocks` |
| `threadIdx.x` | the variable of a loop `across lanes(n)` |
| `__shared__` memory | a tensor in `Memory::SMEM` |
| `__syncthreads()` | the end of a lane loop |
| `atomicAdd` | `atomic_fetch_add_u32`, and the same for other types |
| a block reduction, such as `cub::BlockReduce` | `lanes::sum`, `lanes::max` |
| `cp.async` or TMA | `lanes::copy_async` |
| `expf`, `tanhf` with `--use_fast_math` | `fast_exp`, `fast_tanh` |
| `cudaMemcpy` | `transfer(x, Memory::HBM)` and `transfer(x, Memory::CPU_DRAM)` |

The compiler generates PTX for an NVIDIA GPU. An Apple GPU runs block and lane loops, shared memory,
and atomics on `i32` and `u32`.

## Calling parallel Vx from C or Rust

A `pub fn` can start tasks, and C or Rust can call it as any other `pub fn`. The program that
calls it starts no threads. Link the object with the Vx runtime library, `libvx_std_core`, which
runs the tasks:

```sh
vxc work.vx --action emit-obj -o work.o
cc host.c work.o -L"$HOME/.vx/current/lib" -lvx_std_core -Wl,-rpath,"$HOME/.vx/current/lib" -lpthread -lm -o host
```

[Calling Vx from C](unsafe-and-ffi.md#calling-vx-from-c) says which types a `pub fn` can take.
