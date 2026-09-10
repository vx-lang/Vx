# Unsafe and FFI

Vx checks a lot at compile time: ownership, borrowing, memory placement, contracts. Some things
cannot be checked — a pointer that came from C, a hardware register, a cast the compiler has no way
to justify. `unsafe` is where you take responsibility for those.

`unsafe` does not switch the checks off. It permits a small, specific set of operations that are
otherwise refused.

## unsafe blocks

An `unsafe` block is an expression. It can produce a value.

```rust
fn main() -> i32 {
    let x : i32 = 7;
    let p : *const i32 = &x;
    let v = unsafe { *p };
    return v - 7;
}
```

Reading through a raw pointer — `*p` — is the operation that needs the block. So does indexing
through one, or reading a field through one.

## Raw pointers

Two kinds, matching the two kinds of reference:

| Type | Meaning |
| --- | --- |
| `*const T` | A raw pointer you may read through. |
| `*mut T` | A raw pointer you may read and write through. |

Unlike `&T` and `&mut T`, raw pointers are not tracked by the borrow checker. Nothing stops two
`*mut T` pointing at the same value. That is exactly why dereferencing one needs `unsafe`.

## unsafe functions

A function whose *body* is not the dangerous part — but whose *contract* is — should be marked
`unsafe` itself. A caller then has to opt in.

```rust
unsafe fn read_at(p : *const i32) -> i32 {
    return unsafe { *p };
}

fn main() -> i32 {
    return 0;
}
```

Unsafe-ness is part of the function's type. An `unsafe fn` cannot be passed where a safe function
is expected, so the obligation cannot be lost by storing the function in a variable.

The rule for when to mark a function `unsafe`: if a caller can make it misbehave by passing
something the compiler cannot check — a dangling pointer, a wrong length — it is `unsafe`. If the
function checks everything itself, it is not, even if its body uses `unsafe` internally.

## Calling C

An `extern` block declares functions that exist outside Vx.

```rust
extern "C" {
    safe fn abs(x : i32) -> i32;
}

fn main() -> i32 {
    return abs(0);
}
```

By default an `extern` function is unsafe to call — the compiler knows nothing about what is on the
other side. `safe` marks one that genuinely is safe to call with any arguments its types allow, so
callers do not need an `unsafe` block.

`abs` qualifies: every `i32` is a valid input and it cannot misuse memory. A function taking a
pointer and a length would not qualify, because passing a mismatched pair breaks it.

Use `safe` sparingly. It is a promise the compiler cannot verify, and it is the one place in an
`extern` block where a mistake is silent.

## What unsafe does not unlock

`unsafe` covers operations the type system cannot justify. It does **not** override placement.

A pointer into device memory still cannot be dereferenced from the host inside an `unsafe` block —
that is not an unchecked operation, it is a wrong one, and it stays a compile error. See
[Topologies and memory](heterogeneous.md).

## Where to next

- [Ownership and borrowing](ownership.md) — the checks `unsafe` steps around
- [Topologies and memory](heterogeneous.md) — the checks it does not
- [Standard library reference](stdlib-reference.md) — which `std` functions are `unsafe`, and why
