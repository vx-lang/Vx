# Tasks that wait, channels, and errors from tasks

Status: agreed, except decision 9 (who owns a task's error, #1937). Issues:
#1865, #1864, #1893. Tracking: #1870.
Builds on `cpu_tasks_send_sync_and_shared_atomics.md` (T1 to T8, S1 to S6) and the memory model
(C0 to C7).

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

**Stack size.** A task stack is as large as a thread stack on the platform, because foreign code
runs on it (W8). On a 64-bit machine this costs address space, not memory: the kernel commits a
page only when the task touches it. Segmented stacks save only address space, and they slow calls
and calls into C, so Rust and Go both gave them up. The limits are elsewhere:

- Each stack and its guard page are two kernel mappings, and Linux allows 65,530 by default
  (`vm.max_map_count`). So about 30,000 tasks can park at once, fewer with shadow stacks. A
  program that needs more raises the limit.
- A page that a task touched stays committed when its stack goes back to the pool. The pool gives
  the pages back with `madvise`.
- The pool marks each stack `MADV_NOHUGEPAGE`, so that a huge page cannot commit 2 MiB for a stack
  that uses a few KiB.
- A parked task costs at least one page: 4 KiB on x86-64, and 16 KiB on Apple silicon.
- Under strict overcommit (`vm.overcommit_memory=2`), each stack counts at its full size.

**What a parked task holds.** A value that only holds memory is freed after its last use
(`drop_semantics.md`). So a buffer whose last use comes before a wait is freed before the task
parks. A Rust future keeps such a value across each `.await` until the end of its scope. The size
of a frame does not shrink this way: it is fixed when the function starts. LLVM lets two stack
slots share memory only when the code marks where the life of each one ends
(`llvm.lifetime.end`), and vxc emits no such marks today. Marks at each drop can let Vx frames
share more than C++ and Rust frames, whose marks are at the end of a scope.

**W4. Wait points.** A task waits only at a call to a runtime function that waits on the host:

- `send` and `recv` on a channel, and `select` (M3)
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

**K4. A copy never goes stale.** A task copies a `Copy` value that it only reads, at its `spawn`
(T4). Until the task's join, or the end of its `tasks` block, the code around the task may not
write that variable, as if the task borrowed it, but the variable may end first. So the task
always sees the variable's current value. `for i in 0..n { spawn { work(i) }; }` is valid.
`spawn { work(x) }; x += 1;` is an error: write `let x0 = x;` before the `spawn`, or share the
value through an atomic or a channel. A closure follows the same rule until its last use (#1941).

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

**M3. `select`.** Decision 8 chose a library function, not the block expression below (#2384):
`select(rs : &Vec<Receiver<T>>)` gives the first value from any channel in the list, with the index
of its channel, or `Closed` when each channel is closed and empty. It waits on `recv` only, and the
channels have one element type. A program that waits on channels of different types sends an enum
of its own. A `select` sleeps on one word of the runtime, which each event on any channel changes.
The rest of this section is the design of the block expression, which can still come later for
`send` and `sleep` arms.

The block expression waits on several operations, and runs the arm of the first one that can
continue. It is a block expression, as `tasks` is, not a library type.

<!-- vx-doctest: skip (proposed syntax) -->

```vx
// Add the values from `a` and `b` until both close. Stop when no value comes for `limit` seconds.
let mut total = 0;
let mut a_open = true;
let mut b_open = true;
while a_open || b_open {
  select {
    let r = a.recv() if a_open => {
      match r {
        Result<i64, RecvError>::Ok(v) => { total = total + v; }
        Result<i64, RecvError>::Err(_) => { a_open = false; }
      }
    }
    let r = b.recv() if b_open => {
      match r {
        Result<i64, RecvError>::Ok(v) => { total = total + v; }
        Result<i64, RecvError>::Err(_) => { b_open = false; }
      }
    }
    sleep(limit) => { break; }
    cancelled => { break; }
  }
}
```

- **The arms.** Each arm is one operation, an optional guard, `=>` and a block. The operations are
  `recv`, `send` and `sleep`. `let r = op` binds the result of the operation, with the type of the
  plain call. The program must use it, as it must use the `Result` of a plain wait (F4). A `sleep`
  arm binds nothing.
- **Entry.** The `select` evaluates the guards, and then the operands of each arm that takes part,
  in source order (E1). An arm whose guard is false takes no part. Go uses a nil channel for this.
  If no arm takes part, the `select` waits only for a cancel. Vx's `match` has no guards today, so
  the guard is new syntax.
- **One operation commits.** If some operations can continue, the `select` commits one of them. It
  starts its search at the arm after the one that it chose last time (round robin). So an arm that
  can continue at each of `k` passes in a row, for `k` arms, is chosen in one of them. If none can
  continue, it registers each operation and waits. The first one that can continue commits, and the `select`
  takes back the other registrations. So exactly one operation has an effect.
- **A `send` that loses keeps its value.** The value of a `send` arm must be a place, such as a
  local or a field, not a temporary. It moves out of the place only when its arm commits. After the
  `select`, the checker treats the place as moved only on the path of that arm.
- **The borrow checker treats a `select` as a `match`.** One arm runs, so two arms can both change
  `total`. Each operand is borrowed from the entry to the start of the arm that runs, so an arm can
  drop or move a channel end. `break` and `continue` in an arm go to the loop around the `select`,
  as in a `match` arm. In Go, a `break` in a `select` leaves only the `select`.
- **Cancellation.** A `select` must have a `cancelled` arm. In a cancelled task, no operation
  commits, and that arm runs. So the result that an arm binds never holds `Cancelled`, and the
  warning of F4 looks only at the `cancelled` arm. The `Err(_)` arms of the example get no warning.
- **Closed channels.** A `recv` on a closed channel can always continue: it returns `Closed`. A
  `send` to a dropped receiver can always continue: it gives its value back. A loop that keeps such
  an arm selects it on every pass, so the program turns the arm off with its guard, as the example
  does.
- **The memory model.** A `select` that commits an operation has the edges of that operation (M4).
  A `select` that commits none has no channel edge.
- **The runtime.** The wait records are in the frame of the `select`, and they do not move until it
  returns. The compiler takes them back on each exit. The records do not depend on `T`, so one
  runtime type serves channels of every element type. In the first runtime, the wait blocks the
  thread of the task, with the same protocol.

**Why `select` is syntax.** Rust's `Select` for its standard channels had an `unsafe` `add`, because
a handle in the set must not move, and neither it nor `select!` became stable. crossbeam's `Select`
fixed this with two features that Vx does not have. `Select<'a>` borrows each channel with a written
lifetime, and `&dyn SelectHandle` puts channels of different element types in one set. Vx has no
written lifetimes, and no type with dynamic dispatch (`generics_design.md`). crossbeam also checks
at run time that the program completes the operation that was chosen. A closure for each arm fails
the borrow checker in the usual case, where two arms both change `total`. This is the reason that
`tasks` is syntax (decision 1 of the scoped tasks note), and it holds here too. As syntax, the
compiler owns the wait records, so they cannot move. It writes the typed code of each arm, so the
set needs no dynamic dispatch. It completes the chosen operation itself.

**What `select` is not.** In Concurrent ML, an operation is a value of type `event`. A program
combines events with `choose` and `wrap`, and a library can return an event that its caller selects
on. Vx's `select` is closer to Go's: each arm is written in place. A library that its caller must
wait on together with other things gives the caller a channel. A first-class event would be a value
that holds borrows of its channels, and Vx has no written lifetimes to tie it to them. CML's
`withNack` shows a cost that comes later. An operation that has already started when its arm loses,
such as a read on io_uring, must be cancelled, and the runtime must wait for that (F5). So the
first version allows only operations that a losing arm can take back with no effect.

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
`Result<T, E>` says that its failure is the failure of the block. Then `join()` returns its `T`.
When the task failed, `join()` does not return: the block ends with its failure, as `?` ends it
(F7). So `join()` never gives an error to the code of the block (F3a). The `tasks` block must have
a type `Result<U, E>` with the same `E`. Tasks with different error types need one enum that the
program writes. A task that returns another type returns data. So to collect the results of the
requests that worked, or to try another source after a failure, the task returns its own enum.
The compiler knows `Result` by name, as it knows `SharedCell`. `panic` still stops the program
(T6), so Vx needs no unwinding.

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

**F3a. Who owns a failure.** The block owns its first failure: that `Err` moves into the value of
the block, once. The block drops each later failure, with cancellation off (F6): at the join of its
task, at the end of the block, or at the `?` that raised it. So the block returns or drops each
error exactly once. Nothing copies an error, and `E` needs no `Clone`. This matters because an
error can own a resource: a `SendError<T>` gives back the value that it could not send (M2), such
as a tensor.

`join()` gives no error to the code of the block (F1). If it gave the `Err` by value, the code
could drop it, and the block would then have no first failure to return (#1937). Code that must
look at the error of one task makes that task return its own enum, as data.

**F4. Which waits can be cancelled.** `send`, `recv`, `sleep` and I/O can be cancelled. They
return their error with a `Cancelled` case. A cancelled `send` that already has its place in the
ring of the channel puts the number of that place in a skip set of the runtime. The `recv` of that
place passes it, so the place does not stop the later ones (#2417). A `select` runs its `cancelled` arm (M3). `join()`,
`lock` and the waits for a device cannot be cancelled. A cancelled parent must join its children
anyway (T3), so a cancelled `join` gives nothing. A wait for a device holds a token that must be
used once (#244), and an error would lose it. A loop with no wait point does not see the cancel.
`tasks::cancelled()` lets it check.

**What a cancel promises.** In a cancelled task, the next wait that can be cancelled returns `Err`.
The task stops only if each function between that wait and the task body passes the error up.
The compiler cannot force that. The mark stays (F3), so a function that drops the error gets
`Cancelled` again at its next wait. The bad case is a loop that waits again after a `Cancelled`:
each wait fails at once, and the loop never ends. Dropping a `Cancelled` is not the problem: the
`Err(_) => { break; }` of the example is right. So:

- The program must use the `Result` of each wait that can be cancelled.
- The compiler warns when an arm that matches `Cancelled` can reach the same wait again without
  leaving the loop (W1035). An arm matches `Cancelled` by `_`, by `Err(_)`, or by a name that
  starts with `_`. An arm that gives the error a used name can check it. An arm that changes the
  condition of its `while` can end the loop. A `while` whose body changes its condition in a
  statement of its own ends after some passes, and so does a `for` loop. It is a warning, not an
  error, because the check cannot follow every call.
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
  that is gone would never end. `send_for` and `recv_for` take one, and `wait_while_until` is
  the wait under them. A task of the shared runtime parks for such a wait, and a timer thread
  wakes it at the time limit. The compiler does not check this yet.
- A `tasks` block inside a `drop` does not get the cancel of the task outside it.

**F7. Leaving a `tasks` block early.** `return` or `break` out of the code of a `tasks` block
cancels the tasks of the block and then joins them (T3). Then the exit happens. `?` is a failure
of the block (F2). A `join()` of a task that failed leaves the block the same way (F1).

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
- A shadow stack in each pool entry, which the switch moves from its first version (section 7).
- `llvm.lifetime.end` at each drop, so that the slots of a frame can share memory (W3).
- The monitor thread (W8) and the rules of W9.
- A test mode with one thread and a fixed seed, which gives the same run each time. It is a tool
  for tests: it cannot run a program that blocks in a foreign call.

After that:

- An I/O event loop on epoll or kqueue, with a thread pool or io_uring for files. `std/net.vx`
  then calls it, not the blocking `extern` calls that it uses today. Until then, network I/O blocks
  its OS thread.

## 7. Security

Notes from a review of 2026-10-19. They are not decisions yet, except where a rule above already
says the same thing.

**What safe code gets.**

- No data races (S6). The proof holds only if the library code gives the edges of M4. That code
  is the trusted base (see below).
- No reference that outlives its local across tasks: T3, M5, and `Task<T>` is not `Send`.
- No use after free from cancelled I/O (F5). Async Rust with io_uring needs owned buffers for this
  reason: a dropped future cannot wait for the kernel.
- A cancel loses no data. A cancelled `send` gives its value back (M2), and a `drop` runs with
  cancellation off (F6). Rust's futures that are not safe to cancel in `select!` have no
  counterpart.
- No unwinding (T6). No code sees state that a panic left half-changed, a `Mutex` needs no
  poisoning, and the binary has no unwind tables.
- Memory with a limit: each channel has one (M1), and admission (section 5) gives a verdict when
  the counts are known.
- A stack overflow faults on the guard page (W3) and does not change memory.

**Where the guarantees stop.**

- `unsafe` and `extern` code is the trusted base: the channel, `Mutex` and atomics, each
  `unsafe impl Send` or `Sync`, the C++ runtime, and the vendor libraries. Today `std/net.vx` is
  all `extern` calls. One list of each `unsafe impl` and each runtime entry point would give the
  surface for an audit.
- Availability: a panic stops the process, and so does a deadlock report (F9). So one tenant must
  have its own process, not its own task.
- The monitor (W8) starts workers with no limit when tasks block in foreign calls. It needs a
  limit.
- A stack from the pool (W3) still holds the data of the last task. Safe Vx cannot read it,
  because each `let` must have a value. `unsafe` code and foreign code can. When a stack goes back
  to the pool, `madvise(MADV_DONTNEED)` on its used part clears it and gives the memory back.
- A stale current device can send work into the context of another partition. So "each dispatch
  names its device" (W9) is also a security rule.

**Shadow stacks and CFI.**

1. **A shadow stack for each task stack.** This is the item that costs the most to add later, so
   the first stack switch must have it.
   - Each pool entry of W3 holds a task stack and a shadow stack, each with its own guard page.
   - The switch moves the shadow-stack pointer with the stack pointer, with the restore tokens of
     the architecture. It does this from the first version, also while the feature is off.
   - Linux has user-space shadow stacks on x86 since 6.6, and arm64's Guarded Control Stack since
     6.13. `map_shadow_stack` makes the extra stacks for the pool.
1. **No unwinding keeps the shadow stack in step.** Each frame leaves by a normal return, also on a
   cancel. So Vx must not get a fast cancel that jumps between stacks, or `longjmp`.
1. **Forward-edge CFI from the freeze.** Vx has no overloading, and the freeze knows each signature.
   In safe Vx, an indirect call goes through a closure type or a foreign function pointer. So the
   compiler knows each function whose address the program takes, and its type. It can put an
   IBT or BTI landing pad only on those functions and on the task entry, and a type-hash check
   at each indirect call. LLVM has this check (KCFI). vxc emits neither landing pads nor CFI
   today.
1. **One library can turn it off.** A process gets shadow stacks only when the executable and each
   library that it loads are marked as compatible. The Rust core, `libcuda` and cuBLAS decide it
   for Vx. Check that the NVIDIA libraries have the mark before a doc promises this. The compiler
   and the runtime must still be ready, so that Vx is never the reason a process runs without it.

## The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | No colors and no `async fn` (W1). A mark in a signature does not spread to callers (W5). The analysis of W3 only changes generated code. |
| Every layout is known at the freeze | `Sender<T>` and `Receiver<T>` are library structs. The checks of M5 and M6 use the frozen layout. |
| A generic closes at its definition | `T : Send` is a bound. |
| A body declares nothing global | `select` is an expression, as `tasks` is, and `tasks::cancelled()` is a call. K1 and K2 read only the body of the task. |
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
1. **`select`.** A block expression (M3), or a library type. A library type needs written lifetimes
   and dynamic dispatch, and Vx has neither. Chosen on 2026-10-31: a library function over a list
   of receivers of one type (#2384). It needs neither feature, because the list owns the
   `Receiver`s and they all have one type. It has no `send`, `sleep` or `cancelled` arm yet.
1. **Who owns a task's error (#1937).** `join()` returns `T` and leaves the block on a failure,
   and the block owns its first failure (F1, F3a). One other choice: `join()` gives a
   `Result<T, &E>`. The code can read it but not pass it up with `?`, so each join needs a `match`.
   Another: `join()` gives the `Err` by value, and the code must pass it up on every path. That
   needs a flow check, and it can give the block a second error that is not the first. Open. The
   recommendation is the first.

The review also settled two earlier questions: `select` starts its search round robin (M3), and
the memory-space type decides what can go between targets (M6).

## The work, in order

1. The first runtime: OS threads, one for each task, and the safe dispatch runtime. Then `tasks`,
   `spawn`, K1 to K3, F1 with the first-failure rule of F3, and `?` (F2).
1. Channels, `select` and `Mutex` (M1 to M3, M7). A wait blocks the thread of the task. The Lean
   edges (M4) and the model check of the library code.
1. Cancellation (F3 to F7) and the run-time reports (F9).
1. The second runtime, as one unit: the stack pool and the switch, the monitor (W8), W9, and the
   test mode.
1. The I/O event loop, and `std/net.vx` on it.
1. Admission (section 5), F8 with the borrow checker of tasks, and M6 with `spawn on`.
