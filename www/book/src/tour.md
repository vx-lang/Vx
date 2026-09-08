# A tour of Vx

This chapter covers the ordinary parts of the language — everything you would need to write a
command-line program, with no accelerator in sight. If you have written Rust, most of this will look
familiar; the differences are called out where they matter.

## Values and types

```rust
let x : i32 = 21;      // annotated
let y = 21;            // inferred from context
let mut count = 0;     // mutable
```

Bindings are immutable unless you write `mut`.

The primitive types are the ones you would expect: `i8` `i16` `i32` `i64`, `u8` `u16` `u32` `u64`,
`f16` `f32` `f64`, and `bool`.

**There are no implicit numeric conversions.** An `i32` does not become an `i64` because the context
wants one; you write the conversion. Integer literals infer to the type the context requires, so
`let n : i64 = 5;` is fine, but mixing two differently-typed values in one expression is an error.
This is deliberate — silent widening is a common source of both bugs and unintended performance
cliffs.

## Functions

```rust
fn add(a : i32, b : i32) -> i32 {
    return a + b;
}
```

Parameter types and the return type are both mandatory — there is no inference for either, and a
function with no useful result is written `-> void`. A function may end with a bare expression
instead of `return`, as in Rust.

## Control flow

```rust
if x > 10 {
    // ...
} else if x == 10 {
    // ...
} else {
    // ...
}

for i in 0..n {
    // ...
}

loop {
    // forever, until you break
    if done { break; }
}
```

**There is no `while`.** It is not a keyword, and writing one is a parse error. The two loops are
`for` over a range and bare `loop` with an explicit `break`.

`0..n` is a half-open range: it includes `0` and excludes `n`.

Returning from inside every branch of an `if`/`else` does not yet satisfy the return checker — see
[the note in Your first program](first-program.md). Until that is fixed, assign to a `mut` binding
and return once at the end of the function.

## Structs

```rust
struct Point {
    x: i32,
    y: i32,
}

fn main() -> i32 {
    let p = Point { x: 3, y: 4 };
    let a = p.x;
    return a;
}
```

Methods go in an `impl` block, and the receiver is written out in full — there is no implicit
`self`:

```rust
impl Point {
    fn magnitude_squared(self: &Point) -> i32 {
        return self.x * self.x + self.y * self.y;
    }

    fn translate(self: &mut Point, dx: i32, dy: i32) -> void {
        self.x += dx;
        self.y += dy;
    }
}
```

A return type is never optional. A function that produces no useful result returns `void`.

Note that a `void` function cannot exit early: a bare `return;` does not parse. Structure such a
function so control reaches the end.

`&Point` borrows immutably, `&mut Point` mutably. A method with no `self` parameter is an associated
function, called as `Point::make(...)`.

## Enums and pattern matching

Enums carry data:

```rust
enum Result {
    Ok(i32),
    Err(i32),
}

enum Color {
    Red, Green, Blue,
}
```

Construct a variant with `::`, and take it apart with `match`:

```rust
fn unwrap_or(r: Result, default: i32) -> i32 {
    let mut out = default;
    match r {
        Result::Ok(val) => { out = val; },
        Result::Err(code) => { out = -code; },
    }
    return out;
}
```

A data-carrying enum is laid out as a tag plus a payload.

> **Assign in the arms; do not use `match` as a value.** The form above — each arm assigning to a
> `mut` binding, with a single `return` after — is correct. A `match` used directly as the value of
> a function or a `let` currently produces the wrong answer with no error and no warning:
>
> ```rust
> fn pick(x : i32) -> i32 {
>     match x { 0 => { 7 }, _ => { 9 } }   // pick(0) evaluates to 0, not 7
> }
> ```
>
> A `match` whose arms each `return` is fine. Only the value position is affected.

## Arrays and tensors

An array literal is a tensor:

```rust
let a : Tensor<f32, [4]> = [ 1.0, 2.0, 3.0, 4.0 ];
let first = a[0];
```

The shape is part of the type. `Tensor<f32, [4]>` has four elements, known at compile time.
A `?` marks a dimension that is only known at runtime:

```rust
fn matmul(a : Tensor<f32, [?, ?]>, b : Tensor<f32, [?, ?]>) -> Tensor<f32, [?, ?]> {
    let mut result : Tensor<f32, [?, ?]> =
        Tensor<f32, [?, ?]>::uninit([a.extent(0), b.extent(1)]);

    for i in 0..a.extent(0) {
        for j in 0..b.extent(1) {
            result[i][j] = 0.0;
            for k in 0..a.extent(1) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}
```

`.extent(n)` reads the size of dimension `n`. Shapes that *are* known statically get checked
statically — a matmul whose inner dimensions disagree is a compile error rather than a runtime one.

## Collections

The standard library ships the usual containers. `Vec<T>` is a growable array:

```rust
import std::vec;

fn main() -> i32 {
    let mut v = Vec<i32>::new();
    v.push(10);
    v.push(32);
    return v.get(0) + v.get(1);
}
```

Also available: `HashMap`, `HashSet`, `Option`, `Result`, `String`, `Box`, and iterator adaptors.
See [the standard library](tooling.md#the-standard-library) for the full list.

## Modules

One file is one module. `import` pulls another in:

```rust
import std::vec;
import std::io;
import graph::traversal;
```

`std::` resolves against the standard library shipped with your toolchain. Anything else resolves
against the library search path and then the current directory.

## Unsafe

Raw pointers exist, and the operations that can go wrong with them require `unsafe`:

```rust
extern "C" {
    fn vx_vec_new_i32() -> *mut i8;
    fn vx_vec_push_i32(vec : *mut i8, val : i32) -> i32;
}

fn main() -> i32 {
    unsafe {
        let v = vx_vec_new_i32();
        vx_vec_push_i32(v, 42);
    }
    return 0;
}
```

Dereferencing a raw pointer, indexing through one, reading a field through one, and calling an
`unsafe fn` all require an `unsafe` block. A function that takes a caller's raw pointer and
dereferences it is itself `unsafe fn`, so the obligation is visible in its signature rather than
buried in its body.

`extern "C"` blocks declare foreign functions. Vx's C ABI interop is zero-overhead: there is no
marshalling layer.

## Comptime

`if comptime` selects a branch at compile time. The branches not taken are pruned before semantic
analysis, so they may refer to things that do not exist on the target being compiled for:

```rust
let mut val = 0;

if comptime Topology::Current == Topology::CPU {
    val = 1;
} else if comptime Topology::Current == Topology::CPU_AVX512 {
    val = 2;
} else {
    val = 3;
}
```

`Topology::Current` is the topology the current region compiles for. This is how one source file
carries code for several targets without the dead paths having to typecheck against all of them.

`if` is also an expression:

```rust
return if val == 1 { 0 } else { 1 };
```

Const generics let a value appear in a type — `Tensor<f32, [N]>` for a `const N : i32` — and are
resolved by monomorphization.

## What is next

- [Ownership and borrowing](ownership.md) — the memory model.
- [Generics and traits](generics.md) — abstraction without runtime cost.
- [Topologies and memory](heterogeneous.md) — the reason Vx exists.
