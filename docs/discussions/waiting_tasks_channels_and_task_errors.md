# Tasks that wait, channels, and errors from tasks

Status: proposed. Issues: #1865, #1864, #1893. Tracking: #1870. Builds on
`cpu_tasks_send_sync_and_shared_atomics.md` (T1 to T8, S1 to S6) and the memory model (C0 to C7).

Vx is for programs that run on a fleet of mixed hardware. Its concurrency must be better than
Rust's, not equal to it. Scoped tasks already are: a task can borrow, and no lifetimes are
needed. Three gaps remain, and this note closes them:

1. A task that waits, for example for I/O, holds an OS thread (#1865).
1. Tasks cannot send values to each other (#1864).
1. A task cannot fail except by `panic`, and the error messages are not designed (#1893).

Most of the work is in the runtime, not in the compiler. Section 5 lists it.

## 1. Tasks that wait, with no `async` (#1865)

**Rust's weakness.** An `async fn` has a color: a plain function cannot call one, and a caller of
one must be `async` too. Whether a future is `Send` comes from its body, which breaks the rule
that a signature is the whole contract. A borrow across an `.await` is the hardest case for
Rust's borrow checker.

**W1. No `async` and no `await`.** A task calls a function that waits, such as `recv`, as a normal
call. A function that calls `recv` looks the same as any other function.

**W2. A wait parks the task.** The runtime keeps the stack of the task and runs another ready task
on the same OS thread. Go and Java virtual threads work this way. So many tasks can wait on a few
OS threads. T7 already allows this, because it promises only that a task finishes.

**W3. Stacks.** Go and Java move a stack when it grows. They can, because their runtime knows
each pointer into the stack. Vx cannot: a task borrows locals of its parent (T5), and raw pointers
exist. So each task that can park gets a fixed stack from a pool, with a guard page below it. The
compiler emits stack probes, so that a large frame cannot jump over the guard page. A task that
cannot reach a wait does not need its own stack: it runs on the stack of its worker thread. An
analysis of the call graph finds these tasks. It runs after the freeze and only changes the code
that the compiler generates, as inlining does.

**W4. Wait points.** A task parks only at a call to a runtime function that waits on the host:

- `send`, `recv` and `select` on a channel
- `join()`
- `lock` of a `Mutex` (M7), and `wait` of a `Condvar`
- `sleep` and timers
- an I/O call, after the runtime has its I/O event loop (section 5)
- the host wait for a device that the compiler inserts (`vx_plugin_await_future`, `spawn_on.md`
  section 3)

Between two wait points, a task runs without a break. So the borrow checker needs no new rule: a
borrow across a wait is a borrow across a call. A wait inside device code, such as #244's `wait`
in a kernel, is not a wait point: device code never parks.

**W5. Which functions show that they wait.** A function that waits returns a `Result` whose error
can be `Cancelled` (F4). So its own signature shows that it waits, but its callers' signatures do
not: a caller can handle the `Err` and return a plain value. With `async`, every caller must be
`async` too. So Vx has no function colors, for the same reason that Rust's `Result` is not a color.
A GPU target has no host wait functions in its library, so a GPU body that calls one gets
"function not found".

**W6. This changes T7's serial elision.** T7 lets each task run to its end at its `spawn`. With
channels, that run can wait forever: a task waits for a value that a later task sends. So the run
on one thread changes to this: a task runs at its `spawn` until it parks, and then the parent
continues. One OS thread is still a valid runtime, but it needs the stack switch of W3.

**W7. The promise of T7 grows by one clause.** A task finishes if each of its waits can continue.
So the runtime must run a parked task again when its wait can continue. The promise still does
not say that two tasks run at the same time, so a spin loop between tasks can still hang.

**W8. A foreign call that blocks** holds its OS thread. A monitor thread watches the progress of
each worker. When a worker makes no progress for a time, the monitor starts another worker for
the ready tasks. This costs nothing on each `extern` call, unlike Go's wrapper around each cgo
call. It also helps a worker that runs a long loop with no wait point.

**W9. Tasks that stay on their thread.** A task that parks can continue on another OS thread. Some
state belongs to a thread: CUDA's current device, the owner of a pthread mutex, and thread-local
storage in foreign code. Also, LLVM can keep the address of a thread-local across a call, and the
park call changes the thread. So:

- Vx code has no thread-locals.
- A task stays on its OS thread while a foreign frame is on its stack. Then a wait blocks the OS
  thread and does not park. Java pins a virtual thread in the same case.
- The runtime does not keep state per thread. Each dispatch names its device. The dispatch
  runtime must be safe to call from many threads (section 5).

## 2. Channels (#1864)

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
  total
};
```

Each task moves its own `tx` into itself (T5). A task that only borrowed `tx` would keep the
parent from `drop(tx)` until the join, and the loop would never end.

**M1. The type.** `channel<T : Send>(n)` returns a `Sender<T>` and a `Receiver<T>` with room for `n`
values, and `n` is at least 1. Every channel has a limit, so a fast sender cannot fill the memory.
The first version has one receiver and many senders: `Sender<T>` has `clone`, and `Receiver<T>`
does not. Both are `Send` when `T` is `Send`. Neither is `Sync`.

**M2. The signatures.**

- `send(self : &Sender<T>, v : T) -> Result<void, SendError<T>>`. It parks when the channel is full.
  When the program dropped the `Receiver`, or the task is cancelled, it gives `v` back in the error.
- `recv(self : &Receiver<T>) -> Result<T, RecvError>`. It parks when the channel is empty.
  `RecvError` is `Closed` when the program dropped every `Sender` and the channel is empty, and
  `Cancelled` in a cancelled task.

So a sender to a dropped receiver does not wait forever, and the join at the end of the block
ends.

**M3. `select`.** A `select` closure for each branch fails the borrow checker in the usual case:
two branches both change `total`. Go has syntax for this, and Rust has a macro. Vx uses the form of
crossbeam's `Select`: the program adds each operation, `wait()` returns the index of the one that
can continue, and the program then does that operation. No closures, no new syntax. When more than
one operation can continue, `select` starts its search at a new index each time, so no channel
waits forever. A test mode with one thread and a fixed seed gives the same run each time.

**M4. The memory model.** The Go memory model has the rules, and they fit C2:

- A `send` happens before the `recv` that gets its value. It is a release, and the `recv` is an
  acquire.
- For a channel with room for `n` values, the `k`-th `recv` happens before the `(k + n)`-th `send`
  finishes.
- The drop of the last `Sender` happens before a `recv` that returns `Closed`.

The Lean model (#1867) adds these as edges, and proves DRF-SC with channels, as `Tasks.lean` does
for fork and join. The edges hold only if the library code gives them. So the library channel also
needs the model check or litmus tests that #1864 asks for. Rust has no formal model of its
channels.

**M5. A value in a channel holds no reference.** Vx has no written lifetimes. So the escape
check (E4005) cannot see that a reference sent by a task outlives the local that it points to.
The compiler rejects a `channel<T>` whose `T` holds a `&` or a `&mut`, from the frozen layout.
For the same reason, `Task<T>` is not `Send`, so it cannot leave its `tasks` block through a
channel (T3).

**M6. Channel ends stay on the host.** A `spawn on` body runs on the device (`spawn_on.md`). A
channel end there would make the device poll memory that the host writes. A GPU gives no promise
that a block of threads makes progress, so a kernel that waits for the host while the host waits
for the kernel is a deadlock. It would also be a second data path next to `transfer`, with no
memory-space type. So a `spawn on` body that uses a channel end is a compile error. A channel of
`Tensor<.., Memory::X>` moves the handle and copies no data. The memory-space type, not "holds no
pointer", decides what can go between targets. A channel between machines can come later, with
serialization.

**M7. `Mutex`.** #1864 also asks for a `Mutex`. A lock on the futex system call blocks the OS
thread. On one thread, a task that waits for a lock whose owner is parked on that thread then
never continues. So `lock` parks the task (W4), and the lock belongs to a task, not to a thread
(W9). `lock` cannot be cancelled (F3). It has the same model check as the channel.

## 3. Errors from tasks (#1893)

**F1. A task can return an error.** A `spawn` body can have the type `Result<T, E>`. Then `join()`
returns that `Result`, and the `tasks` block must have a type `Result<U, E>` with the same `E`.
Tasks with different error types need one enum that the program writes. `panic` still stops the
program (T6), so Vx needs no unwinding.

**F2. `?`.** Vx has no `?` today. This design needs a `?` that returns the `Err` unchanged: the type
of the error must be the `E` of the function or block. It does not convert the error, because Vx
has no implicit conversions. `.map_err(..)` converts it.

So passing `Cancelled` up costs more in Vx than in Rust: Rust's `?` converts it with `From`. A
function with its own error enum needs a `Cancelled` case in it, or a `.map_err(..)` at each wait.

**F3. One failure cancels the scope.** When a task returns `Err`:

- The runtime marks each other task of the same `tasks` block as cancelled, and also the code of
  the block itself.
- A `tasks` block inside a cancelled task is cancelled too.
- The mark stays: each later wait that can be cancelled fails at once.
- The value of the block is the first error, whatever its last expression gives. So the block does
  not lose the error of a task that the program did not join (T2). A sibling's `Cancelled` does not
  hide the real error.

The example above shows why. If one task fails, the others stop, their `Sender`s drop, `recv`
returns `Err`, and the loop ends with a partial sum. The block then returns the error, not the
sum.

Trio's nurseries work this way. Swift's `withThrowingTaskGroup` does not: an error from a child
does not cancel the other children, and an error that nothing waits for is lost. Only Swift's
discarding task groups cancel. Rust has no structured cancellation.

**F4. Which waits can be cancelled.** `send`, `recv`, `select`, `sleep` and I/O can be cancelled.
They return their error with a `Cancelled` case. `join()`, `lock` and the waits for a device
cannot be cancelled. A cancelled parent must join its children anyway (T3), so a cancelled `join`
gives nothing. A wait for a device holds a token that must be used once (#244), and an error would
lose it. A loop with no wait point does not see the cancel. `tasks::cancelled()` lets it check.

**What a cancel promises.** In a cancelled task, the next wait that can be cancelled returns `Err`.
The task stops only if each function between that wait and the task body passes the error up.
The compiler cannot force that. The mark stays (F3), so a function that drops the error gets
`Cancelled` again at its next wait. The bad case is a loop that retries on any `Err`: each wait
fails at once, and the loop never ends. So the program must use the `Result` of each wait that
can be cancelled, and the compiler warns when a `match` arm drops a `Cancelled`.

**F5. Hardware that still owns a buffer.** A cancelled I/O or DMA wait does not return until the
device finished the operation or the runtime revoked it. Otherwise the drops of the task free a buffer, and the kernel or the
device then writes into freed memory. For io_uring, the runtime sends a cancel and then waits for
its completion.

**F6. Code that must finish.** A `drop` runs with cancellation off, so a flush, a close, or a free
in stream order (`freeing_buffers_on_every_device.md`) can wait. Trio calls this a shielded scope.

**F7. Leaving a `tasks` block early.** `return`, `break` or `?` out of a `tasks` block cancels the
tasks of the block and then joins them (T3). Then the exit happens.

**F8. Compile errors.** Each error names what a programmer wrote, not internal types:

- A race: the two tasks, the variable, and the line of each access, and the `join` where the
  first access ends.
- A value that is not `Send`: the field path that makes it so, for example "`Job` is not `Send`
  because `Job.cache.buf` is a `*mut u8`". Rust gives a chain of future types here.
- A `Task<T>` that leaves its `tasks` block (T3): the line where it leaves.

**F9. Run-time reports.**

- A panic in a task gives the line of its `spawn`, and the `spawn` lines of each parent task.
- A deadlock: when every task waits and no I/O is in progress, the runtime stops the program. Go
  already lists each goroutine, what it waits for and where, and the line that started it. Vx
  also names the channel or lock, with the line that made it, and prints the cycle of waits.

## 4. Admission

`cross_call_capacity_implementation_plan.md` uses the max rule for calls in sequence. It says that
tasks that run at the same time make the rule a sum across them. A `tasks` block of `spawn on`
tasks makes that real. So:

- A channel with room for `n` values of `T` holds `n` times the size of `T` in the memory space
  of `T`.
- A `spawn` in a loop adds the peak of each task, so the count of passes multiplies it.
- When `n` or the count is known only at run time, the verdict is "unknown". That is the true
  answer, and the program can bound the count to get a verdict.

## 5. The runtime work

- A pool of fixed stacks with guard pages, the stack switch, and stack probes in the generated
  code (W3).
- OS threads (#1863), and the monitor thread (W8).
- A dispatch runtime that is safe from many threads. For example, `cublas_handle()` in
  `runtime/cuda_dispatch.cpp` keeps its handles in a `static` map with no lock.
- An I/O event loop on epoll or kqueue, with a thread pool or io_uring for files. `std/net.vx`
  then calls it, not the blocking `extern` calls that it uses today. Until then, network I/O blocks
  its OS thread, and only W8 helps.
- The runtime `Mutex` and `Condvar` (M7).

## The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | No colors and no `async fn` (W1). A mark in a signature does not spread to callers (W5). The analysis of W3 only changes generated code. |
| Every layout is known at the freeze | `Sender<T>` and `Receiver<T>` are library structs. The checks of M5 and M6 use the frozen layout. |
| A generic closes at its definition | `T : Send` is a bound. |
| A body declares nothing global | `Select` is a library type, and `tasks::cancelled()` is a call. |
| No implicit conversions | `?` does not convert the error (F2). |

## Decisions

1. **The promise of T7.** W7 adds "if each of its waits can continue" to the decision of
   2026-10-18. Without it, a channel between two tasks can hang on one thread.
1. **Cancellation.** Automatic on the first `Err` (F3), or only when the program calls
   `cancel()`. Automatic is the recommendation: it is the safe default.
1. **`?`.** Add a `?` that does not convert (F2), or let the program write each `match`. The `?` is
   the recommendation: F7 and most error code need it.
1. **A dropped `Cancelled`.** A warning when a `match` arm drops it (F4), or an error. The warning
   is the recommendation: some code, such as a cleanup loop, drops it on purpose.
1. **Code that must finish.** Only `drop` runs with cancellation off (F6), or also a block that the
   program marks. The recommendation is `drop` only at first, and a block when a program needs
   one.

The review of 2026-10-19 settled two earlier questions: `select` starts at a new index each time
(M3), and the memory-space type decides what can go between targets (M6).

## The work, in order

1. The stack pool and the stack switch on one thread, and OS threads (#1863). W6 needs the switch.
1. `tasks` and `spawn` with W6, F1 and `?` (F2).
1. Channels, `Select` and `Mutex` (M1 to M3, M7), with the Lean edges (M4) and the model check of
   the library code.
1. Many OS threads: the monitor thread (W8), the rules of W9, and the safe dispatch runtime.
1. Cancellation, early exit and the run-time reports (F3 to F7, F9).
1. The I/O event loop, and `std/net.vx` on it.
1. Admission (section 4), F8 with the borrow checker of tasks, and M6 with `spawn on`.
