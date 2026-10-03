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

## The rules the checker enforces

Each rule below is a compile error, with the code the error index lists it under.

**A moved value cannot be used (E4001).** Assigning a value, or passing it by value, moves it
unless its type is `Copy`; numbers are. Giving the variable a new value makes it usable again. A
`Vec` is not tracked this way yet (#495).

```rust
let a = S { v : 1 };
let b = a;          // a is moved
print(a.v);         // E4001
```

Only one branch of an `if` or arm of a `match` runs, so each may move the same value. After the
`if`, the value counts as moved if any branch that reaches the next line moved it.

Inside a loop, a value moved in the body is used again on the next pass, so it is refused at the
loop unless the body gives it a new value first. A `continue` goes round again as well. A move
just before `break` is fine; the value then counts as moved after the loop.

```rust
let a = S { v : 1 };
while more() {      // E4001: a is moved on every pass
    take(a);
}
```

**A value cannot move while it is borrowed (E4007).** If a borrow of `a` is used after `a` moves,
the borrow would point at a value that has gone.

```rust
let r = &a;
let b = a;          // E4007: r is used below
print(r.v);
```

**A reference cannot give away what it points at (E4008).** `let b = *r;` would move the value out
of `r`, which only borrows it. Copy or clone it instead; for a number, `*r` is a copy and is fine.

**A variable without `mut` is read-only (E4010).** As in Rust, `let x = ..` and a parameter `x : T`
cannot be assigned, nor can a field or element of `x`; `x` cannot be borrowed `&mut`; and a method
that takes `&mut self` cannot be called on it. Write `let mut x = ..`, or `mut x : T` for a
parameter (`mut self : T` for a method that takes `self` by value). A `&mut` held in a variable
without `mut` can still be written through, since that changes what it points at, not the variable.

```rust
let x = 1;
x = 2;              // E4010
let mut y = 1;
y = 2;              // fine
let r = &mut y;
*r = 3;             // fine: writes y, not r
```

**Shared or exclusive, never both (E4002, E4003, E4004).** While a `&mut` borrow of `x` is still
going to be used, `x` cannot be read, written or borrowed again. While a `&` borrow is still going
to be used, `x` cannot be borrowed `&mut`. Two borrows passed to one call count as alive together.

```rust
let m = &mut x;
print(x);           // E4002: m is used below
*m = 2;
```

**A borrow ends at its last use.** It does not last to the end of the block, so this is fine:

```rust
let m = &mut x;
*m = 2;
print(x);           // m is not used again
```

**Different fields are borrowed separately.** `&mut p.a` and `&mut p.b` can be alive together;
`&mut p.a` twice, or `&p.a` while `&mut p` is alive, cannot.

**A reborrow borrows the reference (E4002, E4004).** `let n = &mut *m;` borrows `m` for as long as
`n` is used. Until then `m` cannot be read, written or reborrowed `&mut` again. After `n`'s last use,
`m` works as before.

```rust
let m = &mut x;
let n = &mut *m;
*m = 4;             // E4002: n is used below
*n = 5;
```

**A borrowed variable cannot be assigned (E4009).** While a `&` borrow of `x`, or of a field or
element of it, is still going to be used, that part of `x` cannot be given a new value. A borrow
passed only to a call, as in `v.cmp(&best)`, ends with the call.

```rust
let r = &x;
x = 2;              // E4009: r is used below
print(*r);
```

**Only a `&mut` can be written through (E4006).** `*r = v`, `r.f = v` and `r[i] = v` need `r` to be
a `&mut`. So does a `&mut` field reached through a `&`. A `&` cannot be turned into a `&mut` either:
`&mut *r`, or calling a method that takes `&mut self` through `r`, needs `r` to be a `&mut`.

**A reference cannot outlive what it points at (E4005).** A reference to a local, or a value or
closure holding one, cannot be returned, stored through a reference, or passed to a call that could
store it. Inside a function, a variable declared outside a block cannot be given a reference to a
variable declared inside it.

## Views of a tensor

A row `q[i]`, a field `h.t` that holds a tensor, and a view of either are *views*: they share
their owner's memory instead of copying it. Writing `r[0]` through a row `r = q[1]` changes
`q[1][0]`, and the other way round. So a view is a borrow of its owner, exactly as `&q` is, and the
rules above apply to it.

```rust
let mut q = Tensor<f32, [2, 4]>::new();
let mut r = q[1];   // borrows q mutably, since r is `mut`
r[0] = 5.0;         // writes q[1][0]
print(q[1][0]);     // fine: r is not used again
```

- A view declared `mut` is a `&mut` borrow, and its owner must be `mut` (E4010). Any other view
  is a `&` borrow, so two of them can be used together.
- The owner cannot be moved (E4007), assigned (E4009) or, under a `mut` view, read (E4002) while
  the view is still going to be used. As with `&`, the borrow ends at the view's last use.
- A view of a tensor the function owns, a local or a parameter taken by value, cannot be
  returned (E4005). A view of a tensor taken by reference (`q : &Tensor<..>`) can be: it points
  into the caller's tensor.
- A view cannot be stored where a tensor of its own is held: a variable that already exists
  (`keep = q[1]`) or a struct field (`h.t = q[1]`, `Holder { t : q[1] }`) (E4011). It would
  replace that tensor with a window onto another. Bind it to a new variable with `let` instead.
  Writing a row into another row, `p[0] = q[1]`, copies the elements and is fine.
- A view cannot be passed to a function that takes a tensor by value (E4011): the function
  would own memory that is the owner's. Pass a copy, `f(q[1].clone())`, or make the function
  take the tensor by reference, `f(&q[1])`. `t.clone()` makes a new tensor of `t`'s type with
  a copy of its elements.

**Why views are checked this strictly.** Vx is moving to freeing memory automatically
(`docs/implementation_plans/drop_semantics.md`). A tensor that only holds memory will be freed
after the last use of the tensor *and of every view of it*, and a type with a `Drop` of its own
at the end of its block. Freeing at the last use is only safe if the checker sees every view, so
a free that comes too early is a borrow checker bug, not a rule for the programmer to remember.
The checker tells a view from an owner by where the value came from, since both have the type
`Tensor`; a separate view type (#400) may replace this later.

**Not checked yet.** These views are still invisible to the checker, or refused when they need
not be:

| Case | Issue |
| --- | --- |
| A view chosen by an `if` or a `match` used as a value | #1056 |
| `t.reshape(..)`, which is a view of `t` | #1057 |
| A view taken through a reference variable, `rq[1]` with `rq = &q` | #1058 |
| A closure that uses a view | #1059 |
| Two `mut` rows of different indices, refused because indices are not told apart | #1060 |

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
