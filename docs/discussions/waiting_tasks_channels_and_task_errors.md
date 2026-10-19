# Tasks that wait, channels, and errors from tasks

Status: proposed. Issues: #1865, #1864, #1893. Tracking: #1870. Builds on
`cpu_tasks_send_sync_and_shared_atomics.md` (T1 to T8, S1 to S6) and the memory model (C0 to C7).

Vx is for programs that run on a fleet of mixed hardware. Its concurrency must be better than
Rust's, not equal to it. Scoped tasks already are: a task can borrow, and no lifetimes are
needed. Three gaps remain, and this note closes them:

1. A task that waits, for example for I/O, holds an OS thread (#1865).
1. Tasks cannot send values to each other (#1864).
1. A task cannot fail except by `panic`, and the error messages are not designed (#1893).

## 1. Tasks that wait, with no `async` (#1865)

**Rust's weakness.** An `async fn` has a color: a plain function cannot call one. Whether a future
is `Send` comes from its body, which breaks the rule that a signature is the whole contract. A
borrow across an `.await` is the hardest case for Rust's borrow checker.

**W1. No `async` and no `await`.** A task calls a function that waits, such as `recv` on a socket,
as a normal call. The function looks the same as any other function, so Vx has no function colors.

**W2. A wait parks the task.** The runtime saves the stack of the task and runs another ready task
on the same OS thread. Go and Java virtual threads work this way. So many tasks can wait on a few
OS threads. T7 already allows this, because it promises only that a task finishes.

**W3. Wait points.** A task parks only at a call to a runtime function that waits: a channel
operation, an I/O call, `join()`, or the `wait` of a GPU transfer (#244). Between two wait points,
a task runs without a break. So the borrow checker needs no new rule: a borrow across a wait is a
borrow across a call.

**W4. Heterogeneous targets.** The library of a target has the functions that the target supports.
A GPU target has no socket functions. So a GPU task that calls one gets a "function not found"
error. No signature carries a "this function waits" mark.

**W5. A foreign call that blocks** holds its OS thread. The runtime then starts another OS thread
for the other tasks, as Go does.

**W6. This changes T7's serial elision.** T7 lets each task run to its end at its `spawn`. With
channels, that run can wait forever: a task waits for a value that a later task sends. So the run
on one thread changes to this: a task runs at its `spawn` until it parks, and then the parent
continues. One OS thread is still a valid runtime.

**W7. The promise of T7 grows by one clause.** A task finishes if each of its waits can continue.
So the runtime must run a parked task again when its wait can continue. The promise still does
not say that two tasks run at the same time, so a spin loop between tasks can still hang.

## 2. Channels (#1864)

<!-- vx-doctest: skip (proposed syntax) -->

```vx
let total = tasks {
  let (tx, rx) = channel<i64>(16);
  for part in parts {
    let tx = tx.clone();
    spawn { tx.send(sum(part)); };
  }
  drop(tx);
  let mut total = 0;
  while let Some(v) = rx.recv() { total = total + v; }
  total
};
```

**M1. The type.** `channel<T : Send>(n)` returns a `Sender<T>` and a `Receiver<T>` with room for `n`
values. Every channel has a limit, so a fast sender cannot fill the memory. `send` parks when the
channel is full, and `recv` parks when it is empty (W2). `Sender<T>` and `Receiver<T>` are `Send`
when `T` is `Send`. Neither is `Sync`: each task gets its own `Sender` from `clone`.

**M2. The end.** `recv` returns `None` when the program dropped every `Sender` and the channel is empty. So a
receiver loop ends without a special message.

**M3. `select`.** `select` waits on several channels, and runs the first branch that can continue.
It is a library function over a list of operations, not new syntax.

**M4. The memory model.** The Go memory model has the rules, and they fit C2:

- A `send` happens before the `recv` that gets its value. It is a release, and the `recv` is an
  acquire.
- For a channel with room for `n` values, the `k`-th `recv` happens before the `(k + n)`-th `send`
  finishes.
- The drop of the last `Sender` happens before a `recv` that returns `None`.

The Lean model (#1867) adds these as edges, and proves DRF-SC with channels, as `Tasks.lean` does
for fork and join. Rust has no formal model of its channels.

**M5. A value in a channel holds no reference.** Vx has no written lifetimes. So the escape
check (E4005) cannot see that a reference sent by a task outlives the local that it points to.
The compiler rejects a `channel<T>` whose `T` holds a `&` or a `&mut`, from the frozen layout.
For the same reason, `Task<T>` is not `Send`, so it cannot leave its `tasks` block through a
channel (T3).

**M6. Channels between targets.** A task with `spawn on(T)` can hold a channel end. Then one API
moves data between the CPU and a device. A value that goes to another target must have the same
layout on both targets. So the first version accepts only a `T` that is `Send` and holds no
pointer. A channel between machines can use the same types later, with serialization.

## 3. Errors from tasks (#1893)

**F1. A task can return an error.** A `spawn` body can have the type `Result<T, E>`. Then `join()`
returns that `Result`, and the `tasks` block must have a type `Result<U, E>` with the same `E`.
`E` must have a `From<Cancelled>` (F3). `panic` still stops the program (T6), so Vx needs no
unwinding.

**F2. One failure cancels the sibling tasks.** When a task returns `Err`, the runtime marks each
other task of the same `tasks` block as cancelled. The value of the block is then that first
error, whatever its last expression gives. So the block does not lose the error of a task that the
program did not join (T2). The program also does not see `Cancelled` from a sibling in place of
the real error. Swift task groups and Trio nurseries work this
way. Rust has no structured cancellation: a dropped future stops at an `.await`, and its `Drop`
cannot wait.

**F3. A cancelled task stops at its next wait point.** Each operation that waits (W3) returns
`Err(Cancelled)` in a cancelled task. The task returns with `?`, which converts it to `E`, and its drops run as on any
return. So cancellation needs no unwinding, and no code stops in the middle of a function. A loop
with no wait point does not see the cancel. `tasks::cancelled()` lets such a loop check.

**F4. Compile errors.** Each error names what a programmer wrote, not internal types:

- A race: the two tasks, the variable, and the line of each access, and the `join` where the
  first access ends.
- A value that is not `Send`: the field path that makes it so, for example "`Job` is not `Send`
  because `Job.cache.buf` is a `*mut u8`". Rust gives a chain of future types here.
- A `Task<T>` that leaves its `tasks` block (T3): the line where it leaves.

**F5. Run-time reports.**

- A panic in a task gives the line of its `spawn`, and the `spawn` lines of each parent task.
- A deadlock: when every task waits and no I/O is in progress, the runtime stops the program. It
  lists each task, its `spawn` line, and the channel or `join` that it waits for. Go reports only
  "all goroutines are asleep".

## The parallel compiler

| Rule | This design |
| --- | --- |
| A signature is the whole contract | No colors and no `async fn` (W1). Whether a function waits is not a fact about it (W4). |
| Every layout is known at the freeze | `Sender<T>` and `Receiver<T>` are library structs. The checks of M5 and M6 use the frozen layout. |
| A generic closes at its definition | `T : Send` is a bound. |
| A body declares nothing global | `select` is a library function, and `tasks::cancelled()` is a call. |
| A signature is the whole contract (F1) | The type of a `tasks` block comes from the block itself, not from a function that it calls. |

## Decisions

1. **The promise of T7.** W7 adds "if each of its waits can continue" to the decision of
   2026-10-18. Without it, a channel between two tasks can hang on one thread.
1. **Cancellation.** Automatic on the first `Err` (F2), or only when the program calls
   `cancel()`. Automatic is the recommendation: it is the safe default, and a task that must finish
   can ignore `Err(Cancelled)`.
1. **The choice in `select`.** When two branches can continue, take the first one in the list, or
   take one at random, as Go does. The first one gives the same run each time. A random one
   prevents starvation.
1. **Values between targets (M6).** Only values with no pointers at first, or also `Vec<T>` and
   `String` with a deep copy.

## The work, in order

1. W6 in the first lowering of `tasks`, and F1. Both need only one thread.
1. Channels (M1 to M3), with the Lean edges (M4).
1. Parking on a few OS threads (W2, W5), after OS threads (#1863).
1. Cancellation (F2, F3) and the run-time reports (F5).
1. F4 with the borrow checker of tasks, and M6 with `spawn on`.
