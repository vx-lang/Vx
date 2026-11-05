# Closures that change captured locals, `Duration`, and `within` as a function

Proposed on 2026-11-05, for #2760. It follows F6a in
[`waiting_tasks_channels_and_task_errors.md`](waiting_tasks_channels_and_task_errors.md).

#2760 decided that `within` becomes a library function that takes a closure:

```vx
let sent = within(Duration::from_millis(50), || -> Result<void, Flush> {
  self.file.flush()?;
  self.pool_tx.send(self.slot)?;
  return Result<void, Flush>::Ok(());
});
```

This note says what the language and the library need for that, in the order of the PRs.

## The problem today

- **A write to a captured local is lost, with no message.** A closure copies each scalar that it
  names into its environment, `Closure_N`. Each call copies the environment into new locals
  (`check_closure_expr` in `src/hir/check/control.rs`). So this program prints `0`:

  ```vx
  let mut total = 0;
  let add_one = || -> void {
    total = total + 1;
  };
  add_one();
  print(total);
  ```

  W1034 warns only about a write *outside* the closure.

- **A write through a captured reference works.** `self.x = 1` in a closure in a method with
  `self : &mut S` changes `self`, because the closure copies the reference.

- **A closure moves each struct and tensor that it names**, also when it only reads it.

- **A `spawn` body already does what a closure should do** (K1 to K4). It gets the address of each
  local that it writes, and the checker records a borrow of the local until the join.

- **A closure literal cannot be passed to a `|| -> T` parameter of a generic function.** The literal
  needs `as || -> f64`, and type deduction has no rule for a closure type (`unify_types_internal` in
  `src/hir/env.rs`).

## 1. How a closure takes a local (C1 to C4)

These are the rules K1 to K4 of tasks, for closures:

- **C1. The use decides.** A closure moves a local if its body moves it. It holds `&mut` of a local
  that its body writes, takes `&mut` of, or calls a `&mut self` method on. It copies a `Copy` local
  that it only reads (C4). It holds `&` of any other local that it only reads.
- **C2. The borrow lasts until the last use of the closure.** For `let c = || ..`, that is the last
  use of `c`. For a literal written as an argument, it is the end of the call. While a closure holds
  `&mut x`, the code around it cannot read or write `x`, as with any `&mut`.
- **C3.** To move a local that the body only uses, the body writes `let t = x;`, as in K3.
- **C4. A copy never goes stale.** This is K4 and W1034 as they are today.

What changes for existing programs:

- A write in the body changes the local. Today it changes a copy.
- The closure borrows a struct that the body only reads, and does not move it. So the code after the
  closure's last use can use it again.

What stays:

- **E4005 does not change.** It already treats each closure with a capture as a value that points
  into the frame of its function.
- **Tasks.** A closure type such as `|| -> T` is not `Send`, so a callee cannot give one to a task.
  A call of a closure `c` takes `&mut c` (`Closure_N_call(&mut c, ..)`), so two tasks cannot call
  one closure at once.
- **The parallel compiler.** C1 reads only the body of the closure, inside the function that writes
  it. Nothing global depends on it.

**The generated code.** The field of `Closure_N` for a written local has the type `&mut T`, and `&T`
for a borrowed one. In `Closure_N_call`, each use of that local becomes `*_env.x`. The checker
already finds each use of a captured name (`access.rs`), so it does this rewrite, and both code
generators see plain structs as today.

## 2. A closure literal as an argument

- **A literal takes the closure type of its parameter**, as a number literal takes its type from the
  place that uses it. So `within(d, || -> i64 { .. })` needs no `as`. A closure in a variable still
  needs `as`, so the rule "no implicit conversions" stays.
- **Type deduction gets a rule for closure types.** `T` in `body : || -> T` comes from the type that
  the literal returns.

## 3. `Duration` and `Instant`

`core::time::Duration` exists, with the API of Rust's. The library starts to use it:

| Today | Proposed | | --- | --- | | `std::time::now() -> f32` | `Instant::now()`, with
`elapsed() -> Duration` and `Instant - Instant`. An `Instant` holds the nanoseconds of the monotonic
clock, `vx_clock_ns`. | | `sleep(seconds : f32)` | `sleep(d : Duration)` | |
`send_for(v, seconds : f64)`, `recv_for(seconds : f64)` | `send_for(v, d : Duration)`,
`recv_for(d : Duration)` | | `deadline_after(seconds : f64) -> i64` | `Instant::now() + d` | |
`wait_while_until(word, value, deadline : i64, ..)` | `deadline : Instant` | |
`Drop::wait_limit(&self) -> f64`, 10.0 by default | `-> Duration`, `Duration::from_secs(10)` by
default |

- `core::ops` then imports `core::time`, and `core::time` imports `core::ops`. A program that uses
  `Duration` compiles with this cycle. The PR must also pass the test that checks each
  standard-library module on its own.
- The runtime takes nanoseconds as an `i64`: `vx_cancel_off`, `vx_within_start`, and the deadlines.
- `VX_DROP_WAIT_SECONDS` stays a number of seconds, because an environment variable has no type.
- `now()` is in 23 files of tests, examples and the book. The PR changes them all.

## 4. `within` as a function

```vx
/// Runs `body`. Each wait in it that can be cancelled gives up `limit` after the call starts,
/// or at the deadline of a scope around it when that is earlier.
fn within<T>(limit : Duration, body : || -> T) -> T {
  unsafe { vx_within_start(limit.as_nanos() as i64) };
  let value = body();
  unsafe { vx_within_end() };
  return value;
}
```

- It is in `std::sync::wait`, beside `cancelled()`.
- **A `return` in the body ends the closure**, and not the function around `within`. The body gives
  a value or a `Result`, and `?` in it gives the `Err` to the code that called `within`.
- **W1036 does not warn about a wait in a closure literal written as the `body` of `within`.** The
  checker counts that literal in `within_depth`. A closure in a variable gets the warning, because
  the checker cannot see that the variable goes to `within`.
- **A panic in the body** is as in the block of F6a. This design changes nothing there.
- **What goes:** the keyword `within`, the `within` flag of the block, its hidden owner
  `$within{n}`, and the generated `vx::within_start` and `vx::within_end` calls and their opcodes.
  The tests and the book change to the function.

## The PRs, in order

1. **C1 to C3**, with tests: a write in the body changes the local, a read borrows a struct, and
   each conflict with the borrow is an error. This fixes the lost write on its own.
1. **A closure literal as an argument**, and type deduction for closure types.
1. **`Duration` and `Instant`** in the library and the runtime (section 3).
1. **`within` as a function**, and the removal of the keyword (section 4).

## Open questions

1. **Should a struct that the body only reads be borrowed?** C1 says yes, as K1 does for tasks. It
   changes when the program drops such a struct: at the end of its own scope, not with the closure.
   The other choice keeps the move, and borrows only what the body writes.
1. **Should `now() -> f32` stay for a release, beside `Instant::now()`?** This note removes it,
   because an `f32` loses milliseconds after a few hours.
