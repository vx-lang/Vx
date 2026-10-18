# Scoped tasks, Send and Sync, and shared atomics on the CPU

Status: proposal. Issues: #1860, #1861, #1862. Tracking: #1870. Memory model: C0 to C7 in
`borrow_check_sequence_points_for_calls.md`.

These three designs depend on each other. A task needs a rule for what it may take (#1861), and
two tasks can share an atomic only through a type that a shared reference can write through
(#1862). So this note gives all three, and the decisions they need are at the end.

## 1. Scoped tasks (#1860)

<!-- vx-doctest: skip (proposed syntax) -->

```vx
let total = tasks {
  let left = spawn { sum(&a, 0, n / 2) };
  let right = spawn { sum(&a, n / 2, n) };
  left.join() + right.join()
};
```

**T1. Syntax.** `tasks { .. }` is a block expression. `spawn { .. }`, without `on`, starts a task.
It is valid only in a `tasks` block, also inside an `if` or a loop of that block. It is not valid
in a function or closure that the block calls. Outside a `tasks` block it is a compile error.

**T2. The result.** `spawn { e }` has the type `Task<T>`, where `T` is the type of `e`.
`h.join()` takes `h` by value, waits for the task, and returns its `T`. The end of the `tasks` block
joins each task that the program did not join, and drops its result.

**T3. The join comes from the program structure.** When a `tasks` block ends, every task that it
started is finished. A `Task<T>` cannot leave its `tasks` block: the program may not return it,
store it in a variable declared outside the block, or give it to a call that could store it. This
is the rule that closures already have (E4005).

**T4. Fork and join.** The fork is the `spawn`. It evaluates the captured values by E1 and is a
release (C2). The join is `join()`, or the end of the `tasks` block. It is an acquire. Between
the fork and the join, the task and the code after the `spawn` run at the same time.

**T5. Borrows.** The checker checks a task body as a block. Each access of the task lasts from the
fork to the join (C3). So:

- two tasks may both read a local, or both hold `&a`.
- a task that writes a local, or holds `&mut` to it, excludes every other access to it until the
  join. That includes the code of the `tasks` block after the `spawn`.
- a task in a loop is a new part on each pass. A `&mut` that it takes conflicts with the task of
  the next pass, so it is an error. A value that each pass moves into its own task is valid.
- a task may move a value of the function into itself. The function cannot use it again.

**T6. A panic in a task stops the program**, as `panic` does today. Vx does not unwind, so
`join()` never returns a failure.

**T7. What runs a task.** The language promises that a task finishes, not that it runs at the same
time as another task. One valid run is: each task runs to its end at its `spawn` (C1, serial
elision). So a task must not wait for another task except through `join()`. A spin loop on an
atomic that another task sets may never end. This is a hang, not undefined behavior. The runtime
of #1863 can start with one OS thread for each task, and change to a pool later, with no change
to the language.

**T8. `spawn on`.** Outside a `tasks` block, `spawn on(T) { .. }` keeps its meaning: one task
on `T`, joined at once (C1). In a `tasks` block, `spawn on(T) { .. }` is a task on `T`, and
`join()` returns its `Pinned` value. Then `spawn` means "start work", and `on` says where.

## 2. Which values a task may take (#1861)

Two models exist. Swift checks regions: a value that goes to a task is not used again by the
sender (SE-0414). Swift needs it because a Swift class is a shared reference. Vx has no type like
that: a value that is not `Copy` moves, and the borrow checker already stops the sender from
using it. So Vx needs the rest of Rust's rule, not regions.

**S1. Meaning.** A `Send` value may move into a task. A `Sync` value may be used through `&T` by
two tasks at once.

**S2. The compiler derives both from the fields.** A number and a `bool` are `Send` and
`Sync`. A struct, enum, tuple or array is `Send` when every field is `Send`, and `Sync` when every
field is `Sync`. A raw pointer `*const T` or `*mut T` is neither.

**S3. A type with a raw pointer says so itself.** `Vec`, `Box`, `String` and `HashMap` hold raw
pointers. Each one opts in, with a new item form:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
unsafe impl<T : Send> Send for Vec<T> {}
unsafe impl<T : Sync> Sync for Vec<T> {}
```

`unsafe` says that the compiler cannot check the claim. Vx needs no opt-out (`impl !Send`): a type
that must not cross tasks holds a raw pointer, and does not opt in.

**S4. Closures.** A closure is `Send` when every value that it captures is `Send`. The body that
writes the closure knows its captures, so this is a local fact. A closure type in a signature
(`|i32| -> i32`) is neither `Send` nor `Sync`, because the signature does not say what it
captures. This keeps the rule that a signature is the whole contract.

**S5. What `spawn` checks.** Each value that the task moves into itself is `Send`. Each local that
the task reads through a shared borrow is `Sync`. Each local that the task writes, or borrows
`&mut`, is `Send`.

**S6. Data races.** Safe code has no data races, because of A1, C3 and S5. A data race in `unsafe`
code is undefined behavior (C7). This is the only case in which a race has no meaning.

## 3. A type that a shared reference can write through (#1862)

**U1. `UnsafeCell<T>`** in `core::cell`. The compiler knows it by name, as it knows `Copy`.

<!-- vx-doctest: skip (proposed syntax) -->

```vx
struct UnsafeCell<T> { v : T }
impl<T> UnsafeCell<T> {
  fn new(v : T) -> UnsafeCell<T> { .. }
  fn get(self : &UnsafeCell<T>) -> *mut T { .. }   // the only way to write through `&`
  fn get_mut(self : &mut UnsafeCell<T>) -> &mut T { .. }
}
```

**U2. Why the compiler must know it.** The flat generator marks each `&T` parameter
`llvm.readonly` (`src/codegen/flat.rs`). An atomic store through `&Self` would then break the
optimizer's assumptions. A `&T` whose `T` contains an `UnsafeCell`, in a field at any depth, does
not get `llvm.readonly`. Whether a type contains one is a layout fact, and the freeze knows it.

**U3.** A write through the pointer from `get` needs `unsafe`. A safe type wraps it.
`UnsafeCell<T>` is `Send` when `T` is, and it is not `Sync`.

**U4. Atomics.** Each atomic holds its value in an `UnsafeCell`, and each method that writes takes
`self : &Self`. Each atomic opts in to `Sync`:

<!-- vx-doctest: skip (proposed syntax) -->

```vx
struct AtomicI32 { v : UnsafeCell<i32> }
unsafe impl Sync for AtomicI32 {}
```

`get_mut` keeps `&mut self`. So a plain access to an atomic is exclusive, and no program has a race
between an atomic access and a plain one. #1868 is not needed while this holds.

U1 to U4 do not need threads. They can come first, and the atomic tests still run on one thread.

## The parallel compiler

`docs/language_rules_for_a_parallel_compiler.md`:

| Rule | This design |
| --- | --- |
| A signature is the whole contract | No signature gets a fact from a body. An erased closure type is not `Send` (S4). Vx has no `async fn` or `-> impl Trait`, which leak `Send` in Rust. |
| Every layout is known at the freeze | `Send`, `Sync` and "contains an `UnsafeCell`" come from the fields and from top-level `unsafe impl` items. |
| A generic closes at its definition | `T : Send` is a bound like any other. `Task<T>` is a generic instantiation, the allowed exception. |
| A body declares nothing global | `tasks` and `spawn` are expressions. `unsafe impl` is valid only at the top level. |

## Decisions

1. **Syntax.** `tasks { spawn { .. } }`, as above. A library function cannot do it: Rust's
   `thread::scope` needs lifetimes to tie a task to its scope, and Vx has none. Other spellings
   (`scope`, `parallel`) are possible. `tasks` adds one keyword.
1. **What runs a task.** The language promise of T7 (a task finishes, and nothing more), with one
   OS thread for each task at first. The other choice promises one thread for each task. That
   makes spin loops between tasks valid, but a pool can then never replace the threads.
1. **`Send` and `Sync`.** Rust's rule, derived from fields (S2), with `unsafe impl` to opt in. The
   other choice is Swift's regions.
1. **The cell's name.** `UnsafeCell`, as in Rust, which a Vx programmer probably already knows.

## For the Lean model (#1867)

The fork is `spawn`, a release. The join is `join()` or the end of the `tasks` block, an acquire.
The parent runs between the fork and the join. A task returns one value.

## The work, in order

1. `UnsafeCell`, the `llvm.readonly` change, and atomics that write through `&Self` (U1 to U4).
1. `Send` and `Sync` derived from fields, and `unsafe impl` (S1 to S5).
1. `tasks` and `spawn`, parsed and borrow checked (T1 to T5). The first lowering runs each task at
   its `spawn`, which T7 allows. So programs and tests can come before threads.
1. OS threads (#1863), and the Lean fork and join (#1867).
