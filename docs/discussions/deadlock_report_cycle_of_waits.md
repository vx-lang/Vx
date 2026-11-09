# The cycle of waits in a deadlock report

Issue #2932, part of F9 in `waiting_tasks_channels_and_task_errors.md`.

## The problem

When each task waits and nothing can wake one, the runtime stops the program
(`stdlib/rust_core/src/ffi/wait.rs`). The report lists each wait: its line, the line that made its
channel or lock, and the `spawn` lines of its task. In a program with many tasks, the programmer
must find the cycle in that list. The report should print the cycle:

```
  a task waits in `recv` at main.vx:12 on channel 1, which only the task from the `spawn` at main.vx:20 sends on
  that task waits in `lock` at main.vx:22 on lock 1, which the task from the `spawn` at main.vx:9 holds
```

To print an edge "wait W waits for task T", the runtime must know which task can end W.

| Wait | Who can end it | Does the runtime know it now? | |---|---|---| | `join` of a task | that
task | Yes. The note has the site of the task. | | `lock` | the task that holds the lock | No. But
`lock` can record the holder when it takes the lock. | | `recv` on channel c | each task that holds
a `Sender` of c | No. | | `send` on channel c | the task that holds the `Receiver` of c | No. | |
`select` | the senders of each of its channels | No. |

A lock and a join are easy. A channel end is the hard case. It is a plain value. It moves into a
task at a `spawn`, into a struct, into a function, and through another channel (`Sender<T>` is
`Send` and `Owned`). `clone` makes more `Sender`s. The runtime sees none of these moves. It sees
only the count of live `Sender`s in the word `SENDERS` of the channel.

"The tasks that sent on c before" is not an answer. In
`a_deadlock_report_names_two_tasks_that_wait_for_each_other.vx`, both tasks wait in `recv` before
either one sends.

## Options

### A. Locks and joins only

`lock` records the site of the task that takes the lock, and the drop of its `MutexGuard` clears it. The report prints a
cycle when every edge in it is a join or a lock. A cycle through a channel stays a plain list.

- Exact for what it covers, and small: a change to `wait.rs` and `mutex.vx` only.
- It does not help the usual deadlock, which goes through channels. The two-task test above gets no
  cycle.

### B. The ends that each `spawn` captures

The checker knows the Vx type of each value that a `spawn` captures. It already walks the fields of
those types for `Send` (`thread_safety.rs`). The walk can also collect each channel end and each
lock in them (`HostOnly`, #3087). At the `spawn`, the code passes the runtime the channel words of
each captured end. The runtime keeps "task T holds a `Sender` of c" until T ends. Locks work as in
A.

- It covers the usual program: ends made in `main` and moved into tasks. The two-task test gets its
  cycle.
- It does not follow an end that moves after the `spawn`: through a channel, into a task that a task
  starts, or into a `Vec`. Then the runtime has a wrong holder or none. The report must then print
  only what it knows. It must never print a cycle that is not real, so it prints an edge only when
  the count in `SENDERS` matches the number of `Sender`s that it can place.
- Work: the checker, both code generators, the lowering of `vx.spawn`, and the runtime.

### C. Follow each move of an end

Each move of a value that holds an end tells the runtime: a `spawn`, a `send` and a `recv` of such a
value, and a drop. A generic `send<T>` knows at compile time whether `T` holds an end, so the cost
is zero for other `T`.

- Exact, except for an end that unsafe code copies.
- The most work. It adds a runtime call to each such move. A move into a struct, or into a function
  in the same task, needs no call, because the holder stays the same task.

### D. No holders: group the waits by channel

Print, for each channel, the waits on it and the count of live `Sender`s:

```
  channel 1 (made at main.vx:43): 1 task waits in `recv`, no task waits in `send`, 1 `Sender` is alive
```

- Small, and always true.
- It is not a cycle. The programmer still finds the holder of the live `Sender`.

## Recommendation

B, with A for locks and D as the fallback for each channel that B cannot place. B gives the cycle
for the common program, and it never prints an edge that it cannot prove. C can come later, on the
same report, if real programs move ends after a `spawn` often.

## Tests

- Two tasks that wait in `recv` for each other: the report prints the cycle (B).
- Two tasks that each hold one lock and wait for the other: the report prints the cycle (A).
- A `Sender` sent through a channel to a task: the report prints the counts of D for that channel,
  and no cycle.
