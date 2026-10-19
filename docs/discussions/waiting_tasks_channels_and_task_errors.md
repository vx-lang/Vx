# Tasks that wait, channels, and errors from tasks

Status: agreed. Issues: #1865, #1864, #1893. Tracking: #1870. Builds on
`cpu_tasks_send_sync_and_shared_atomics.md` (T1 to T8, S1 to S6) and the memory model (C0 to C7).

Vx is for programs that run on a fleet of mixed hardware. Its concurrency must be better than
Rust's, not equal to it. Scoped tasks already are: a task can borrow, and no lifetimes are
needed. Three gaps remain, and this note closes them:

1. A task that waits, for example for I/O, holds an OS thread (#1865).
1. Tasks cannot send values to each other (#1864).
1. A task cannot fail except by `panic`, and the error messages are not designed (#1893).

Most of the work is in the runtime, not in the compiler. Section 6 lists it.

## 1. Tasks that wait, with no `async` (#1865)

**Rust's weakness.** An `async fn` has a color: a plain function cannot call one, and a caller of
one must be `async` too. Whether a future is `Send` comes from its body, which breaks the rule
that a signature is the whole contract. A borrow across an `.await` is the hardest case for
Rust's borrow checker.

**W1. No `async` and no `await`.** A task calls a function that waits, such as `recv`, as a normal
call. A function that calls `recv` looks the same as any other function.

**W2. Two runtimes.** The language is the same for both. Only the cost of a task changes.

- **The first runtime** gives each task its own OS thread, as the scoped tasks note decided. A
  wait blocks the thread of the task. A task never moves to another thread.
- **The second runtime** lets many tasks share a few OS threads. A wait parks the task: the runtime
  keeps the stack of the task and runs another ready task on the same thread. Go and Java virtual
  threads work this way. W3, W8 and W9 are the parts that it needs, and they come together.

**W3. Stacks (second runtime).** Go and Java move a stack when it grows. They can, because their
runtime knows each pointer into the stack. Vx cannot: a task borrows locals of its parent (T5),
and raw pointers exist. So each task that can park gets a fixed stack from a pool, with a guard
page below it. The compiler emits stack probes, so that a large frame cannot jump over the guard
page. A task that cannot reach a wait does not need its own stack: it runs on the stack of its
worker thread. An analysis of the call graph finds these tasks. It runs after the freeze and only
changes the code that the compiler generates, as inlining does.

**W4. Wait points.** A task waits only at a call to a runtime function that waits on the host:

- `send`, `recv` and `select` on a channel
- `join()`
- `lock` of a `Mutex` (M7), and `wait` of a `Condvar`
- `sleep` and timers
- an I/O call, after the runtime has its I/O event loop (section 6)
- the host wait for a device that the compiler inserts (`vx_plugin_await_future`, `spawn_on.md`
  section 3)

Between two wait points, a task runs without a break. So the borrow checker needs no new rule: a
borrow across a wait is a borrow across a call. A wait inside device code, such as #244's `wait`
in a kernel, is not a wait point: device code never parks.

**W5. Which functions show that they wait.** A function that waits returns a `Result` whose error
can be `Cancelled` (F4). So its own signature shows that it waits, but its callers' signatures do
not: a caller can handle the `Err` and return a plain value. With `async`, every caller must be
`async` too. So Vx has no function colors, for the same reason that Rust's `Result` is not a color.
A device body that calls a host wait gets the compile error of F8.

**W6. Serial elision ends with channels.** T7 lets each task run to its end at its `spawn`. With
channels or a `Mutex`, that run can wait forever: a task waits for a value that a later task
sends. So the compiler can lower a task as a call at its `spawn` only until channels exist. After
that, a task needs its own thread (first runtime) or a stack switch (second runtime).

**W7. The promise of T7 grows by one clause.** A task finishes if each of its waits can continue.
The runtime owns this promise: it runs each ready task at some time. A runtime with fewer OS
threads than tasks needs the monitor of W8 for this, because a blocked thread runs no other task.
The first runtime needs nothing more. The promise still does not say that two tasks run at the
same time, so a spin loop between tasks can still hang.

**W8. A foreign call that blocks (second runtime)** holds its OS thread. A monitor thread watches
the progress of each worker. When a worker makes no progress for a time, the monitor starts
another worker for the ready tasks. This costs nothing on each `extern` call, unlike Go's wrapper
around each cgo call. It also helps a worker that runs a long loop with no wait point.

**W9. Tasks that stay on their thread (second runtime).** A task that parks can continue on
another OS thread. Some state belongs to a thread: CUDA's current device, the owner of a pthread
mutex, and thread-local storage in foreign code. Also, LLVM can keep the address of a
thread-local across a call, and the park call changes the thread. So:

- Vx code has no thread-locals.
- A task stays on its OS thread while a foreign frame is on its stack. Then a wait blocks the OS
  thread and does not park. Java pins a virtual thread in the same case.
- A foreign frame can be below a Vx frame only through a callback. So the runtime counts the
  entries to the Vx functions that foreign code can call: definitions with a foreign ABI, and
  function pointers given to foreign code. An ordinary `extern` call costs nothing.
- The park code of the runtime is foreign code too, so it does not count. It must keep no
  thread-local address across the switch.
- The runtime does not keep state per thread. Each dispatch names its device.

## 2. How a task takes a local

The scoped tasks note does not say when a task moves a local and when it borrows it.

**K1. The use decides.** A task moves a local if its body moves it, and borrows it otherwise, as a
Rust closure does.

**K2. A local that ends first is moved.** A task moves each local that is not `Copy` and whose
scope ends before the join of the task. A borrow of that local can never be valid, so the rule
has no other choice. This covers a local of one loop pass, such as the `tx` in the example of
section 3.

**K3.** To move a local that the body only borrows, the body writes `let t = tx;`. Vx needs no
`spawn move`. K1 and K2 read only the body of the task, so they are local facts.

## 3. Channels (#1864)

<!-- vx-doctest: skip (proposed syntax) -->

```vx
let total = tasks {
  let (tx, rx) = channel<i64>(16);
  for part in parts {
    let tx = tx.clone();
    spawn { tx.send(sum(part)) };
  }
  drop(tx);
  let mut total = 0;
  loop {
    match rx.recv() {
      Result<i64, RecvError>::Ok(v) => { total = total + v; }
      Result<i64, RecvError>::Err(_) => { break; }
    }
  }
  Result<i64, SendError<i64>>::Ok(total)
};
```

Each task has the type `Result<void, SendError<i64>>`, so the block has the type
`Result<i64, SendError<i64>>` (F1). The `tx` of each pass ends before the join, so each task moves
it (K2). A task that borrowed the outer `tx` would make `drop(tx)` a compile error (T5).

**M1. The type.** `channel<T : Send>(n)` returns a `Sender<T>` and a `Receiver<T>` with room for `n`
values, and `n` is at least 1. Every channel has a limit, so a fast sender cannot fill the memory.
The first version has one receiver and many senders: `Sender<T>` has `clone`, and `Receiver<T>`
does not. Both are `Send` when `T` is `Send`. Neither is `Sync`.

**M2. The signatures.**

- `send(self : &Sender<T>, v : T) -> Result<void, SendError<T>>`. It waits when the channel is full.
  When the program dropped the `Receiver`, or the task is cancelled, it gives `v` back in the error.
- `recv(self : &Receiver<T>) -> Result<T, RecvError>`. It waits when the channel is empty.
  `RecvError` is `Closed` when the program dropped every `Sender` and the channel is empty, and
  `Cancelled` in a cancelled task.

So a sender to a dropped receiver does not wait forever, and the join at the end of the block
ends. In the example, a task whose receiver is gone fails, and so does the block (F1). That is the
right result: no code reads the values.

**M3. `select`.** A `select` closure for each branch fails the borrow checker in the usual case:
two branches both change `total`. Go has syntax for this, and Rust has a macro. Vx uses the form of
crossbeam's `Select`: the program adds each operation, `wait()` returns the index of the one that
can continue, and the program then does that operation. No closures, no new syntax. When more than
one operation can continue, `select` starts its search at the index after the one that it chose
last time (round robin). So a branch that can continue at each of `k` calls in a row, for `k`
branches, is chosen in one of them. A random start would give only a probability.

**M4. The memory model.** The Go memory model has the rules, and they fit C2:

- A `send` happens before the `recv` that gets its value. It is a release, and the `recv` is an
  acquire.
- For a channel with room for `n` values, the `k`-th `recv` happens before the `(k + n)`-th `send`
  finishes.
- The drop of the last `Sender` happens before a `recv` that returns `Closed`.
- An `unlock` of a `Mutex` happens before the next `lock` of it.

The Lean model (#1867) adds these as edges, and proves DRF-SC with channels and locks, as
`Tasks.lean` does for fork and join. The edges hold only if the library code gives them. So the
library code also needs the model check or litmus tests that #1864 asks for.

**M5. A value in a channel holds no reference.** Vx has no written lifetimes. So the escape
check (E4005) cannot see that a reference sent by a task outlives the local that it points to.
The compiler rejects a `channel<T>` whose `T` holds a `&` or a `&mut`, from the frozen layout.
For the same reason, `Task<T>` is not `Send`, so it cannot leave its `tasks` block through a
channel (T3).

**M6. Channel ends stay on the host.** A `spawn on` body for a device runs on the device
(`spawn_on.md`). A channel end there would make the device poll memory that the host writes. A GPU
gives no promise that a block of threads makes progress, so a kernel that waits for the host while
the host waits for the kernel is a deadlock. It would also be a second data path next to
`transfer`, with no memory-space type. So a device body that uses a channel end gets the compile
error of F8. `spawn on(CPU)` runs on the host (`spawn_on.md` section 3), so its body can use one.
A channel of `Tensor<.., Memory::X>` moves the handle and copies no data. The memory-space type,
not "holds no pointer", decides what can go between targets. A channel between machines can come
later, with serialization.

**M7. `Mutex`.** #1864 also asks for a `Mutex`. In the second runtime, a lock on the futex system
call blocks the OS thread, and a task that waits for a lock whose owner is parked on that thread
never continues. So `lock` parks the task (W4), and the lock belongs to a task, not to a thread
(W9). `lock` cannot be cancelled (F4). It has the same model check as the channel.

**M8. `Arc`.** #1864 also asks for `Arc`. A task borrows the locals of its parent (T5), so most
programs need no reference count. `Arc` waits until a program needs data that lives longer than
its `tasks` block.

## 4. Errors from tasks (#1893)

**F1. A task that fails, and a task that returns data.** A `spawn` body whose type is core's
`Result<T, E>` says that its failure is the failure of the block. Then `join()` returns that
`Result`, and the `tasks` block must have a type `Result<U, E>` with the same `E`. Tasks with
different error types need one enum that the program writes. A task that returns another type
returns data. So to collect the results of the requests that worked, or to try another source
after a failure, the task returns its own enum. The compiler knows `Result` by name, as it knows
`SharedCell`. `panic` still stops the program (T6), so Vx needs no unwinding.

**F2. `?`.** Vx has no `?` today. This design adds one:

- `x?` returns the `Err` of `x` unchanged. The type of the error must be the `E` of the target. It
  does not convert the error, because Vx has no implicit conversions. `.map_err(..)` converts it.
- The target is the innermost function, `spawn` body, or `tasks` block. Rust's `?` does the same in
  closures, async blocks and `try` blocks.
- In the code of a `tasks` block, an `Err` from `?` is a failure of the block, as an `Err` from a
  task is (F3). An `Err` that is the last value of the block is also a failure.
- In a `spawn` body, `return` leaves the task. A `break` or `continue` that targets a loop outside
  the body is a compile error.

So passing `Cancelled` up costs more in Vx than in Rust: Rust's `?` converts it with `From`. A
function with its own error enum needs a `Cancelled` case in it, or a `.map_err(..)` at each wait.

**F3. One failure cancels the scope.** When a task of a `tasks` block, or the code of the block,
fails:

- The runtime marks each task of the block as cancelled, and also the code of the block.
- A `tasks` block inside a cancelled task is cancelled too.
- The mark stays: each later wait that can be cancelled fails at once.
- The value of the block is the first failure, whatever its last expression gives. So the block
  does not lose the error of a task that the program did not join (T2). A later `Cancelled` does
  not hide the real error, also when the code of the block passes it up with `?`.

The example above shows why. If one task fails, the others stop, their `Sender`s drop, `recv`
returns `Err`, and the loop ends with a partial sum. The block then returns the error, not the
sum.

Trio's nurseries work this way. Swift's `withThrowingTaskGroup` does not: an error from a child
does not cancel the other children, and an error that nothing waits for is lost. Only Swift's
discarding task groups cancel. Rust has no structured cancellation. Cancellation that only the
program starts would fail without a sign: after one shard fails, the other shards keep their
devices busy until they finish.

**F4. Which waits can be cancelled.** `send`, `recv`, `select`, `sleep` and I/O can be cancelled.
They return their error with a `Cancelled` case. `join()`, `lock` and the waits for a device
cannot be cancelled. A cancelled parent must join its children anyway (T3), so a cancelled `join`
gives nothing. A wait for a device holds a token that must be used once (#244), and an error would
lose it. A loop with no wait point does not see the cancel. `tasks::cancelled()` lets it check.

**What a cancel promises.** In a cancelled task, the next wait that can be cancelled returns `Err`.
The task stops only if each function between that wait and the task body passes the error up.
The compiler cannot force that. The mark stays (F3), so a function that drops the error gets
`Cancelled` again at its next wait. The bad case is a loop that waits again after a `Cancelled`:
each wait fails at once, and the loop never ends. Dropping a `Cancelled` is not the problem: the
`Err(_) => { break; }` of the example is right. So:

- The program must use the `Result` of each wait that can be cancelled.
- The compiler warns when an arm that matches `Cancelled` can reach the same wait again without
  leaving the loop. An arm matches `Cancelled` by name, by `_`, or by `Err(_)`. It is a warning,
  not an error, because the check cannot follow every call.
- F9 reports the exact case at run time.

**F5. I/O that still owns a buffer.** A cancelled I/O wait does not return until the kernel
finished the operation or the runtime revoked it. Otherwise the drops of the task free a buffer,
and the kernel then writes into freed memory. For io_uring, the runtime sends a cancel and then
waits for its completion. A copy by a device has no such case, because its wait cannot be
cancelled (F4).

**F6. Code that must finish.** A `drop` runs with cancellation off, so a flush, a close, or a free
in stream order (`freeing_buffers_on_every_device.md`) can wait. Trio calls this a shielded scope.
For example, a slot of a KV cache must go back to its pool when a task is cancelled. A plain
`pool_tx.send(slot)` gets `Cancelled`, and the slot is lost. The same `send` in the `drop` of a
guard for the slot works.

- A wait in a `drop` must have a time limit. Cancellation cannot stop it, so a wait for a peer
  that is gone would never end.
- A `tasks` block inside a `drop` does not get the cancel of the task outside it.

**F7. Leaving a `tasks` block early.** `return` or `break` out of the code of a `tasks` block
cancels the tasks of the block and then joins them (T3). Then the exit happens. `?` is a failure
of the block (F2).

**F8. Compile errors.** Each error names what a programmer wrote, not internal types:

- A race: the two tasks, the variable, and the line of each access, and the `join` where the
  first access ends.
- A value that is not `Send`: the field path that makes it so, for example "`Job` is not `Send`
  because `Job.cache.buf` is a `*mut u8`". Rust gives a chain of future types here.
- A `Task<T>` that leaves its `tasks` block (T3): the line where it leaves.
- A host wait in a device body, such as `recv` (W5, M6): the call, and the `spawn on` that puts the
  body on the device.

**F9. Run-time reports.**

- A panic in a task gives the line of its `spawn`, and the `spawn` lines of each parent task.
- A deadlock: the runtime stops the program when every task waits, and nothing can wake one. A
  task in a foreign call (W8), work on a device, a timer, or I/O in progress can wake one. Go
  already lists each goroutine, what it waits for and where, and the line that started it. Vx
  also names the channel or lock, with the line that made it, and prints the cycle of waits.
- A task that gets `Cancelled` from many waits in a row: the line of the wait (F4).

## 5. Admission

`cross_call_capacity_implementation_plan.md` uses the max rule for calls in sequence. It says that
tasks that run at the same time make the rule a sum across them. A `tasks` block of `spawn on`
tasks makes that real. So:

- A channel with room for `n` values holds `n` values of `T` in host memory. When `T` is a tensor
  handle, the channel also keeps up to `n` tensors alive in their memory space.
- A `spawn` in a loop adds the peak of each task, so the count of passes multiplies it.
- Each task that can park has a stack (W3) in host memory.
- When `n` or the count is known only at run time, the verdict is "unknown". That is the true
  answer. The program gets a verdict with an `invariant` clause on the `for` loop that bounds the
  count. `hir/memory.rs` already reads these clauses.

## 6. The runtime work

For the first runtime:

- OS threads (#1863), one for each task.
- A dispatch runtime that is safe from many threads. Two tasks on two threads already race on it.
  For example, `cublas_handle()` in `runtime/cuda_dispatch.cpp` keeps its handles in a `static` map
  with no lock.
- The runtime `Mutex` and `Condvar` (M7).

For the second runtime, all together:

- A pool of fixed stacks with guard pages, the stack switch, and stack probes in the generated
  code (W3).
- The monitor thread (W8) and the rules of W9.
- A test mode with one thread and a fixed seed, which gives the same run each time. It is a tool
  for tests: it cannot run a program that blocks in a foreign call.

After that:

- An I/O event loop on epoll or kqueue, with a thread pool or io_uring for files. `std/net.vx`
  then calls it, not the blocking `extern` calls that it uses today. Until then, network I/O blocks
  its OS thread.

## The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | No colors and no `async fn` (W1). A mark in a signature does not spread to callers (W5). The analysis of W3 only changes generated code. |
| Every layout is known at the freeze | `Sender<T>` and `Receiver<T>` are library structs. The checks of M5 and M6 use the frozen layout. |
| A generic closes at its definition | `T : Send` is a bound. |
| A body declares nothing global | `Select` is a library type, and `tasks::cancelled()` is a call. K1 and K2 read only the body of the task. |
| No implicit conversions | `?` does not convert the error (F2). |

## Decisions

We chose these on 2026-10-19, from the recommendations of a review.

1. **The promise of T7.** W7 adds "if each of its waits can continue" to the decision of
   2026-10-18, and the runtime owns it. Chosen: yes.
1. **The runtime for the first steps.** One OS thread for each task, or a shared runtime with the
   monitor from the start. Chosen: one OS thread for each task (W2). It needs no stack switch,
   no monitor and no rules for W9. The second runtime then comes as one unit.
1. **Cancellation.** Automatic on the first failure (F3), or only when the program calls
   `cancel()`. Chosen: automatic. Its cost is that a task that must not fail the block
   returns its own enum, not `Result` (F1).
1. **`?`.** Add a `?` that does not convert, with the target rule of F2, or let the program write
   each `match`. Chosen: add it.
1. **A dropped `Cancelled`.** Chosen: the warning of F4 for an arm that can reach the same wait
   again, and the run-time report of F9.
1. **Code that must finish.** Only `drop` runs with cancellation off (F6), or also a block that the
   program marks. Chosen: `drop` only, with a time limit on each wait in it.
1. **How a task takes a local.** The use decides, and K2 moves a local that ends first, or a new
   `spawn move`. Chosen: K1 to K3, with no new syntax.

The review also settled two earlier questions: `select` starts its search round robin (M3), and
the memory-space type decides what can go between targets (M6).

## The work, in order

1. The first runtime: OS threads, one for each task, and the safe dispatch runtime. Then `tasks`,
   `spawn`, K1 to K3, F1 with the first-failure rule of F3, and `?` (F2).
1. Channels, `Select` and `Mutex` (M1 to M3, M7). A wait blocks the thread of the task. The Lean
   edges (M4) and the model check of the library code.
1. Cancellation (F3 to F7) and the run-time reports (F9).
1. The second runtime, as one unit: the stack pool and the switch, the monitor (W8), W9, and the
   test mode.
1. The I/O event loop, and `std/net.vx` on it.
1. Admission (section 5), F8 with the borrow checker of tasks, and M6 with `spawn on`.
