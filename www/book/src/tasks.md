# Tasks and channels

A Vx program can do several things at the same time on the CPU. It starts each one as a task. This
chapter shows how to start tasks, how they share data, and how they send values to each other.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## A `tasks` block and `spawn`

A `tasks` block holds tasks. `spawn { .. }` starts a task, and the task runs on a thread of its
own. The block ends only after each of its tasks ends. So a task can borrow the locals of the
code around it: the locals live longer than the task.

`spawn` gives a handle to the task. `join()` waits for the task and gives the value of its last
expression. The value of the block is the value of its last expression too.

This program adds the numbers 1 to 100 with two tasks. Each task borrows the vector `v` from `main`
and adds one half of it (`book_tasks_split_the_work.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tasks_split_the_work.vx:program}}
```

It prints `total 5050`.

## A task that fails

A task fails when its value is the `Err` of a `Result`. Then its `join()` leaves the block, and that
`Err` is the value of the block. A `join()` of a task that did not fail gives the value inside the
`Ok`. So the code of the block works with plain values, and the block gives the first failure
(`book_tasks_join_a_failure.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tasks_join_a_failure.vx:program}}
```

It prints:

```text
read 1 and 2: Ok 30
read 1 and 4: Err 4
```

The block must join each task that can fail, in a statement of its own that always runs. Otherwise
the compiler gives the error E3060. A task that fails also cancels the other tasks of its block, so
that they do not wait for nothing. The section on channels below says what a cancel does.

## What a task can take

A task can take a value from the code around it in two ways:

- It moves the value. Then the type of the value must be `Send`.
- It borrows the value, while the code around it can use the value too. Then the type must be
  `Sync`, because two threads can read it at the same time.

The compiler finds both traits from the fields of a type. A number, a `bool` and a struct of them
are `Send` and `Sync`. A raw pointer is neither, so a type with a raw pointer field must say what
it allows with `unsafe impl`. The library does this for `Vec<T>`, the atomics and the types below.

For example, two tasks cannot borrow one `Receiver` of a channel, because two `recv` calls at the
same time would take the same value. The compiler says:

```text
Error[E3055]: the task reads 'rx', which the code around the task can read at the same time, so
its type 'Receiver<i64>' must be `Sync`: field 'words' of 'Receiver<i64>' has type
'*mut AtomicI64', and '*mut AtomicI64' is a raw pointer, which is not `Sync`
```

The rule depends only on the types, never on the body of a function. So the compiler can check each
function on its own, and in parallel.

## Atomics that tasks share

An atomic is a number that many tasks can change at the same time, from `core::sync::atomic`. Each
task here borrows the same counter and adds 1 to it 10000 times (`book_tasks_share_an_atomic.vx`):

```rust
{{#include ../../../tests/backend/pass/book_tasks_share_an_atomic.vx:program}}
```

It prints `count 40000`. A plain `i64` in place of the atomic does not compile: a task cannot
write a value that another task can read at the same time.

## Channels

A channel moves values from tasks that send to one task that receives. `channel<T>(n)` makes a
channel with room for `n` values, and gives a `Sender<T>` and a `Receiver<T>`:

- `send(v)` puts `v` in the channel. It waits while the channel is full.
- `recv()` takes the next value. It waits while the channel is empty. It gives `Closed` when the
  program dropped each `Sender` and the channel holds no value.
- Clone the `Sender` to send from many tasks. A channel has one `Receiver`.

The values of one sender arrive in the order that it sent them. The type `T` must be `Send`, and it
must not hold a reference: a value in a channel can outlive the data that a reference points to.

In this program, four tasks send squares, and the code of the block receives them
(`book_channel_from_workers.vx`):

```rust
{{#include ../../../tests/backend/pass/book_channel_from_workers.vx:program}}
```

It prints `got 40 values, sum 20540`.

### Cancellation

When the code of a `tasks` block leaves it early, with `return` or `break`, or when a task fails,
the block cancels its tasks. A wait that can be cancelled then gives an error at once: `recv()`
gives `Cancelled`, and `send(v)` gives `v` back with `cancelled` set. The task should then end. The
compiler warns about a loop that waits again after a `Cancelled` (W1035), because each wait in a
cancelled task gives `Cancelled` at once, and such a loop never ends.

A `join()` and a `lock()` cannot be cancelled.

## `select`

`select(&receivers)` waits on a list of receivers, and takes the first value from any of them. It
gives the value and the index of its channel in the list. It gives `Closed` when each channel is
closed and empty. A channel that always has a value does not hide the others
(`book_select_from_two_channels.vx`):

```rust
{{#include ../../../tests/backend/pass/book_select_from_two_channels.vx:program}}
```

It prints `from 0: 3 values, from 1: 3 values`.

## `Mutex`

A `Mutex<T>` guards a value that many tasks change. `lock()` waits until no other task holds the
lock, and gives a guard. `get()` on the guard gives a pointer to the value. The drop of the guard
unlocks the `Mutex` (`book_mutex_guards_a_total.vx`):

```rust
{{#include ../../../tests/backend/pass/book_mutex_guards_a_total.vx:program}}
```

It prints `total 20000`. Use an atomic for one number, and a `Mutex` when a change touches more
than one number.

## Why Vx has no `async`

In Vx, a function that waits is a normal function. `recv`, `join`, `lock` and a network read wait
for an event, and the task continues after it. A function that calls them is a normal function too.
So Vx has no `async` and no `await`.

In a language with `async`, a function that waits must be `async`, and so must each function that
calls it. Such a function has a "color", and a program must keep the two colors apart. Vx does not
have that problem. A borrow that lives across a wait is a borrow across a call, so the borrow
checker needs no new rule.

The runtime runs each task on an OS thread of its own today. A later runtime will run many tasks on
a few OS threads. The language is the same for both.

## Compared with Rust and Go

| | Vx | Rust | Go |
| --------------------------- | ----------------------------------- | ------------------------------------ | -------------------------------- |
| Start a task | `spawn` in a `tasks` block | `thread::scope` and `spawn` | `go f()` |
| A task borrows locals | Yes, the block outlives its tasks | Yes, in `thread::scope` | No borrows, a race is possible |
| What can move to a task | `Send`, from the types | `Send`, from the types | Anything |
| A task that fails | Its `join` leaves the block | `join` gives a `Result` | Each program writes it |
| Cancel the other tasks | When the block leaves or a task fails | Each program writes it | `context`, by hand |
| Functions that wait | Normal functions | `async fn`, or a blocked thread | Normal functions |
| Wait on many channels | `select` over a list of receivers | A crate, such as `crossbeam` | `select` statement |
