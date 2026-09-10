# Control flow

Vx has four ways to branch or repeat: `if`, `loop`, `for`, and `match`.

## if and else

```rust
fn classify(x : i32) -> i32 {
    if x < 0 {
        return 0;
    } else if x == 0 {
        return 1;
    } else {
        return 2;
    }
}

fn main() -> i32 {
    return classify(5) - 2;
}
```

Braces are always required. There is no single-statement form.

An `if` whose branches all return is a statement, not a value. An `if` used in value position has
to produce a value on every path:

```rust
fn main() -> i32 {
    let x : i32 = 3;
    let label = if x > 2 { 1 } else { 0 };
    return label - 1;
}
```

## loop

`loop` repeats until something breaks out of it.

```rust
fn main() -> i32 {
    let mut i : i32 = 0;
    loop {
        if i >= 3 {
            break;
        }
        i = i + 1;
    }
    return i - 3;
}
```

`continue` skips to the next turn of the loop:

```rust
fn main() -> i32 {
    let mut seen : i32 = 0;
    let mut i : i32 = 0;
    loop {
        i = i + 1;
        if i < 3 {
            continue;
        }
        seen = seen + 1;
        if i >= 5 {
            break;
        }
    }
    return seen - 3;
}
```

### There is no while loop

`while` is not a keyword in Vx. Writing `while i < n { ... }` does not produce a "no such loop"
message — `while` and `i` both lex as ordinary identifiers, and you get a confusing parse error
about a missing `;`.

Write the same thing with `loop` and a guard:

```rust
fn main() -> i32 {
    let mut i : i32 = 0;
    loop {
        if i >= 4 {
            break;
        }
        i = i + 1;
    }
    return i - 4;
}
```

Whether `while` should exist is [Vx#506](https://github.com/vx-lang/Vx/issues/506).

### Loop invariants

A `loop` can carry an `invariant`: a condition that must hold on every turn. The prover checks it.

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

The parentheses around the condition are **required** here, unlike `requires` and `ensures` on a
function, which take theirs optionally. That inconsistency is not deliberate — it is
[Vx#501](https://github.com/vx-lang/Vx/issues/501).

## for

`for` walks a range or anything that implements `Iterator`.

```rust
fn main() -> i32 {
    let mut total : i32 = 0;
    for i in 0..4 {
        total = total + i;
    }
    return total - 6;
}
```

`0..4` counts from 0 up to but not including 4, so that loop adds `0 + 1 + 2 + 3`.

A `for` loop can carry an `invariant` in the same way a `loop` can.

## match

`match` compares a value against patterns, in order, and runs the first arm that fits.

```rust
fn main() -> i32 {
    let x : i32 = 1;
    match x {
        0 => { return 1; }
        _ => { return 0; }
    }
}
```

`_` matches anything. It is usually the last arm.

`match` is most useful with an `enum`, where each arm handles one variant:

```rust
enum Colour {
    Red,
    Green,
}

fn main() -> i32 {
    let c = Colour::Red;
    match c {
        Colour::Red => { return 0; }
        Colour::Green => { return 1; }
    }
}
```

## Where to next

- [Ownership and borrowing](ownership.md) — who owns a value, and who may look at it
- [Contracts and verification](contracts.md) — `requires`, `ensures`, `assert`
- [Generics and traits](generics.md) — writing code once for many types
