# Contracts and verification

Most languages let you write down what a function expects only in a comment. Vx lets you write it
in the signature, where the compiler can act on it.

There are four pieces: `requires`, `ensures`, `invariant`, and `assert`.

## requires and ensures

`requires` states what must be true when the function is called. `ensures` states what will be true
when it returns.

```rust
fn halve(x : i32) -> i32
    requires x > 0
    ensures return > 0
{
    return x;
}

fn main() -> i32 {
    return halve(4) - 4;
}
```

Inside `ensures`, the word `return` means the value the function is about to return.

A function may carry more than one of each. They are read as a list, all of which must hold:

```rust
fn clamp_positive(x : i32, hi : i32) -> i32
    requires x > 0
    requires hi > x
    ensures return > 0
{
    return x;
}

fn main() -> i32 {
    return clamp_positive(1, 2) - 1;
}
```

The condition may be written bare or in parentheses — `requires x > 0` and `requires (x > 0)` are
the same.

## invariant

An `invariant` on a loop states something that is true on every turn.

```rust
fn main() -> i32 {
    let limit : i32 = 3;
    let mut i : i32 = 0;
    loop invariant(limit > 0) {
        if i >= limit {
            break;
        }
        i = i + 1;
    }
    return 0;
}
```

Like `requires` and `ensures`, an invariant may be written bare or in parentheses:
`invariant limit > 0` and `invariant(limit > 0)` are equivalent. An invariant about `i` itself, such
as `i >= 0`, cannot be proven yet: see "Not yet" below.

## What checks these

Conditions are discharged by an SMT solver — z3 — at compile time. If the prover cannot show a
condition holds, compilation fails with a diagnostic in the `E8xxx` range rather than the program
being allowed through.

This means two things worth understanding:

- **z3 must be installed** for contracts to be checked. The build instructions in
  [Building from source](building.md) cover it.
- **The prover can fail to prove something true.** A condition it cannot discharge is reported, not
  assumed. If you hit that, the usual fix is to state an intermediate fact the prover needs, rather
  than to remove the contract.

## What the prover knows

The prover checks each function on its own, from the facts it has at each point in the body.

**Where facts come from:**

- the function's own `requires`;
- `let x = e;`, which gives `x == e` (a `let mut` gives nothing, since `x` may change);
- `assert(c, ..)`, which gives `c` after the `assert`;
- the condition of an `if`: the `then` branch knows it is true, the `else` branch knows it is false;
- a loop's `invariant`, inside the loop, and `a <= i && i < b` inside `for i in a..b`;
- what a called function promises in its `ensures` (see below).

**Facts follow the paths.** What a branch learns ends with the branch. When a branch ends early, by
`return`, `break`, `continue`, `panic`, `abort` or a call to a function that never returns
(`-> !`), the code after the `if` is reached only through the other branch, so it keeps what that
branch knew:

```rust
fn positive(x : i32) -> i32
    ensures return > 0
{
    if x <= 0 {
        return 1;
    }
    return x;
}

fn main() -> i32 {
    return positive(3) - 3;
}
```

At `return x`, the prover knows `x > 0`, because the only way there is past the `if`.

**`ensures` is checked at each `return`,** from what is known there, with `return` standing for that
value. An error (`E8001`) points at the `return` that fails.

**Calls.** At a call, each `requires` of the called function must follow from what is known there,
with the call's arguments in place of its parameters. If it does not, the call is an error
(`E8006`). The call's result is then known through the called function's `ensures`:

```rust
fn halve(x : i32) -> i32
    requires x > 0
    ensures return > 0
{
    return x;
}

fn twice_halved(x : i32) -> i32
    requires x > 0
    ensures return > 0
{
    return halve(halve(x));
}

fn main() -> i32 {
    return twice_halved(4) - 4;
}
```

The inner `halve(x)` meets `x > 0` from `twice_halved`'s own `requires`. The outer call meets it
from the inner call's `ensures`. A function with no `ensures` promises nothing, so the prover knows
nothing about what it returns.

**What it can reason about:** whole numbers with `+`, `-` and `*`, comparisons, `&&`, `||` and `!`,
fields and elements of those, and calls. A condition that uses anything else, such as division, is
ignored, and the compiler warns that it was: "the prover ignores a condition here".

**Not yet:**

- A `match` arm does not know which pattern matched
  ([Vx#1352](https://github.com/vx-lang/Vx/issues/1352)).
- After a loop, nothing is known about what happened in it: not its `invariant`, and not the
  condition that ended it. And since a `let mut` gives no fact, an `invariant` about a loop counter
  cannot be proven when the loop starts ([Vx#1353](https://github.com/vx-lang/Vx/issues/1353)).

## assert

`assert` is the run-time counterpart. It takes a condition and an optional message:

```rust
fn main() -> i32 {
    let x : i32 = 1;
    assert(x == 1, "x should be one");
    return 0;
}
```

If the condition is false at run time, the program stops and reports the message.

Use `assert` for what you cannot state statically. Use `requires` and `ensures` where you can, so
the error arrives at compile time instead.

## Verified

`Verified<T>` is a type that carries the fact that a value has been checked. A plain `T` and a
`Verified<T>` are different types, so a function that demands a checked value cannot be handed an
unchecked one by mistake.

```rust
fn main() -> i32 {
    let x : i32 = 1;
    let v = Verified(x);
    return 0;
}
```

This is the same idea as `requires`, moved into the type: rather than every function re-stating the
condition, one function establishes it and the type carries it onwards.

## Diagnostics

Contract failures report under their own codes. The full list is in the
[diagnostic index](error-index.md); the `E8xxx` group is the prover.

## Where to next

- [Control flow](control-flow.md) — where `invariant` attaches
- [Ownership and borrowing](ownership.md) — the other thing checked at compile time
- [Unsafe and FFI](unsafe-and-ffi.md) — what the checks do *not* cover
