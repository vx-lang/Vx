# Writing parallel code

[Tasks and channels](tasks.md) explains each tool for concurrency. This chapter explains how to
choose between them, and how to check that a parallel program is correct. It is for programmers
who come from Rust with tokio, from Go, or from C with threads.

## Start with the program in order

Write the program without tasks first, and keep it. The version in order is the reference: run
both versions on the same input, and compare the results. A parallel program that is fast and
wrong is easy to write. A comparison finds the problem in the first run.

## Choose the shape of the work

Most parallel programs have one of three shapes:

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
| The result of one task | `join()` on its handle |
| Values from many tasks | a channel |
| A counter or a flag that tasks share | an atomic |
| A change to more than one value at a time | a `Mutex` |
| Values from more than one channel | a `Selector` |
| A value that lives longer than its `tasks` block | an `Arc` |

Use a channel when you are not sure. A channel moves each value to one owner, so two tasks never
change the same data. A task can borrow the locals of the function around it, so most programs
need no `Arc`.

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

## Choose a memory order

- A counter that only gives out numbers needs `Relaxed`.
- To give data to another task through a flag, store the flag with `Release`, and load it with
  `Acquire`. [Memory orders](tasks.md#memory-orders) has a program that does this.
- Use `SeqCst` only when two or more flags must be seen in one order by every task.

By default, Vx compiles a `Relaxed` load as an `Acquire` load. With `-fweak-relaxed-atomics`, the
load stays `Relaxed`. So a program that needs `Acquire` but asks for `Relaxed` can work on Vx and
fail elsewhere. Ask for the order that the program needs.

## Waits that do not end

- A `recv()` loop ends only after the program drops each `Sender`. Drop the `Sender` that made the clones,
  as the program above does with `drop(tx)`.
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

## Calling parallel Vx from C or Rust

A `pub fn` can start tasks, and C or Rust can call it as any other `pub fn`. The program that
calls it starts no threads. Link the object with the Vx runtime library, `libvx_std_core`, which
runs the tasks:

```sh
vxc work.vx --action emit-obj -o work.o
cc host.c work.o -L"$HOME/.vx/current/lib" -lvx_std_core -Wl,-rpath,"$HOME/.vx/current/lib" -lpthread -lm -o host
```

[Calling Vx from C](unsafe-and-ffi.md#calling-vx-from-c) says which types a `pub fn` can take.

## On a GPU

Code on a GPU uses blocks and lanes, not tasks. [Parallel code on a GPU](gpu.md) explains them.
