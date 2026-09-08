# Generics and traits

Generics in Vx are monomorphized: each instantiation becomes its own concrete function or type at
compile time. There is no boxing, no vtable and no dynamic dispatch unless you ask for it.

## Generic types and functions

```rust
struct Pair<T> {
    first: T,
    second: T,
}

enum Maybe<T> {
    Just(T),
    Nothing,
}
```

Instantiate by naming the argument:

```rust
fn main() -> i32 {
    let m = Maybe<i32>::Just(42);

    let mut result = 0;
    match m {
        Maybe<i32>::Just(val) => { result = val; },
        Maybe<i32>::Nothing => { result = 0; },
    }
    return result;
}
```

## Generic impls

An `impl` block can be generic, or specific to one instantiation:

```rust
impl<T> Pair<T> {
    fn first(self: &Pair<T>) -> T {
        return self.first;
    }
}

impl Pair<i32> {
    fn sum(self: &Pair<i32>) -> i32 {
        return self.first + self.second;
    }
}
```

When several impls could apply, the most specific one wins.

## Bounds

Constrain a parameter with `:`:

```rust
impl<T : Float> Tensor<T, [?, ?]> {
    // ...
}
```

## Traits

Traits describe shared behaviour, and are implemented with `impl ... for`:

```rust
impl<T> Iterator<VecIter<T>, T> for VecIter<T> {
    // ...
}
```

That is how `Vec` participates in `for` loops and in the iterator adaptors — `map` and friends are
ordinary generic functions over the `Iterator` trait rather than compiler magic.

## Const generics

A *value* can be a type parameter, not only a type. This is what makes statically-shaped tensors
work:

```rust
fn dot<const N : i32>(a : Tensor<f32, [N]>, b : Tensor<f32, [N]>) -> f32 {
    let mut acc = 0.0;
    for i in 0..N {
        acc += a[i] * b[i];
    }
    return acc;
}
```

Because `N` is part of the type, passing two tensors of different lengths to `dot` is a compile
error rather than a runtime check you forgot to write. Each distinct `N` monomorphizes to its own
function, so the loop bound is a constant the optimizer can see.

## How this stays fast to compile

Every symbol, nominal type and monomorphized instantiation is identified by a flat 256-bit
identifier rather than by a pointer into a shared tree. Combined with a nominal type system and
mandatory boxing for recursive types, that decouples modules from one another: the frontend
resolves and checks them in parallel across cores, with no query engine, no locks and no shared
mutable state.

The observable consequence is that the same source produces byte-identical MLIR whether it is
compiled serially or in parallel — which is asserted in the test suite rather than assumed.
