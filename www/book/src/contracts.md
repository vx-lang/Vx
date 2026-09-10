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
the same. (`invariant` is stricter; see below.)

## invariant

An `invariant` on a loop states something that is true on every turn.

```rust
fn main() -> i32 {
    let mut i : i32 = 0;
    loop invariant(i >= 0) {
        if i >= 3 {
            break;
        }
        i = i + 1;
    }
    return 0;
}
```

Unlike `requires` and `ensures`, `invariant` **requires** its parentheses. `invariant i >= 0` is a
parse error. This is an inconsistency rather than a design decision, tracked as
[Vx#501](https://github.com/vx-lang/Vx/issues/501).

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
