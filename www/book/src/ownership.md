# Ownership and borrowing

Vx has no garbage collector and no mandatory reference counting. Memory is managed by ownership,
checked at compile time, in the same family as Rust's model.

## Owning and borrowing

A binding owns its value. Passing it to a function by value moves it, and the original binding is
no longer usable:

```rust
let v = make_buffer();
consume(v);      // v is moved
// reading v here is an error
```

Borrow instead of moving to keep the original alive:

```rust
fn total(v : &Vec<i32>) -> i32 { /* ... */ }
fn append(v : &mut Vec<i32>, x : i32) { /* ... */ }
```

`&T` is a shared borrow, `&mut T` is an exclusive one. The usual rule applies: any number of shared
borrows, or exactly one exclusive borrow, never both at once. The checker tracks variance and
regions, so a borrow cannot outlive what it points into.

## Linear values

Some values are *linear*: they must be consumed exactly once, and the checker enforces it. Device
buffers are the motivating case. A buffer that has been handed off to an accelerator has left your
control, and reading it again is a use-after-move — reported as a compile error rather than as
corrupted data.

This is stronger than an ordinary move check. A linear value cannot be quietly dropped either,
because dropping a device allocation without releasing it is a leak the runtime cannot detect for
you.

## Boxing

Recursive types must be boxed. `Box<T>` is a heap allocation with a single owner:

```rust
import std::box;

struct Node {
    value: i32,
    next: Box<Node>,
}
```

The requirement is not an oversight — it is what lets the compiler give every nominal type a size
without solving a fixpoint across module boundaries, which in turn is what lets the frontend compile
modules in parallel with no shared state.

## Raw pointers

`*const T` and `*mut T` are raw pointers, and they opt out of all of the above. Because of that,
the operations that can go wrong with them require `unsafe`:

- dereferencing one
- indexing through one
- reading a field through one
- calling an `unsafe fn`

A function that takes a caller's raw pointer and dereferences it is itself declared `unsafe fn`, so
the obligation is visible in the signature rather than buried in the body.

```rust
unsafe fn read_first(p : *const i32) -> i32 {
    return *p;
}

fn main() -> i32 {
    let x : i32 = 42;
    unsafe {
        return read_first(&x);
    }
}
```

Keep `unsafe` blocks small. The point of the annotation is that the region a human has to verify by
hand is written down and searchable.
