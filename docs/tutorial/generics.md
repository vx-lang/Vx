# Vx Tutorial: Using Generics

Vx supports generic programming, allowing you to write highly reusable, statically typed functions and data structures that work across multiple data types without sacrificing performance.

## The Problem

In systems programming, you frequently write operations that are structurally identical but operate on different precision levels (e.g., `f32` vs `i32` tensors). Without generics, you would have to duplicate code:

```vx
fn process_f32(val: Tensor<f32>) -> Tensor<f32> {
    return val;
}

fn process_i32(val: Tensor<i32>) -> Tensor<i32> {
    return val;
}
```

## Writing a Generic Function

In Vx, you can define a generic function using the angle-bracket `<T>` syntax. `T` serves as a placeholder for any concrete type.

```vx
// A generic function that takes any type T and returns T
fn process<T>(val: T) -> T {
    return val;
}
```

## Implicit Type Deduction

When calling a generic function, you **do not** need to explicitly specify the type. The Vx compiler is smart enough to deduce the type of `T` based on the arguments you pass in.

```vx
fn main() {
    let t1: Tensor<f32> = 1.0;
    let t2: Tensor<i32> = 2;

    // The compiler automatically infers T = Tensor<f32>
    let res1 = process(t1);

    // The compiler automatically infers T = Tensor<i32>
    let res2 = process(t2);
}
```

## Zero-Cost Abstractions

When you compile the code above, Vx performs **Monomorphization**. It looks at the calls you made and automatically generates specialized, highly optimized versions of the `process` function behind the scenes.

If you inspect the compiled MLIR output, you will see exactly two distinct functions generated:
- `@process_TensorF32`
- `@process_TensorI32`

This means that using generics incurs **zero runtime overhead**. There is no dynamic dispatch, no vtables, and no type-checking occurring when your program runs. It performs exactly as fast as if you had hand-written the specialized `f32` and `i32` functions yourself!
