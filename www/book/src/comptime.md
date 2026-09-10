# Compile-time evaluation

`comptime` marks work that happens while the program is being compiled, not while it runs.

## comptime blocks

A `comptime` block runs during compilation.

```rust
fn main() -> i32 {
    comptime {
        let size : i32 = 4 * 4;
    }
    return 0;
}
```

Everything inside has to be knowable at compile time. A `comptime` block cannot read a run-time
value, call into C, or touch a device.

## comptime conditions

`if comptime` chooses a branch at compile time. The branch not taken is not compiled.

```rust
fn main() -> i32 {
    if comptime 1 < 2 {
        return 0;
    }
    return 1;
}
```

This is different from an ordinary `if` with a constant condition. An ordinary `if` is compiled in
full and then possibly folded by the optimiser; `if comptime` decides before code generation, so
the untaken branch does not have to compile at all. That matters when a branch is only valid for
some types or some hardware.

## Const generic parameters

A generic parameter can be a value rather than a type, written with `const`:

```rust
fn buffer_size<const N : i32>() -> i32 {
    return N;
}

fn main() -> i32 {
    return buffer_size<4>() - 4;
}
```

The value is fixed when the function is instantiated, so it can be used where a compile-time
constant is required — most usefully in a tensor's shape.

## Where this is used

Compile-time evaluation is what lets a shape be part of a type. `Tensor<f32, [4, 4]>` needs `4` to
be known while type checking, not while running, and const generic parameters are how a function
can be generic over a shape without giving up that knowledge:

```rust
fn main() -> i32 {
    let t : Tensor<f32, [2, 2]> = Tensor<f32, [2, 2]>();
    return 0;
}
```

Because the shape is in the type, a matrix multiply whose dimensions do not line up is a compile
error rather than a run-time crash — the same argument as [contracts](contracts.md), applied to
dimensions.

## Where to next

- [Generics and traits](generics.md) — the rest of the generic system
- [Contracts and verification](contracts.md) — other things checked before the program runs
