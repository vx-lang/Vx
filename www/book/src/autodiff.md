# Automatic differentiation

Vx can differentiate a function you wrote, at compile time. There is no tape, no graph built at run
time, and no separate framework — the derivative is generated from the function's own code.

Three forms, matching the three things people usually want.

## grad

`grad` gives the derivative of a function with respect to its input.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let d = grad(square, 3.0);
    return 0;
}
```

`grad(square, 3.0)` is the derivative of `square` evaluated at `3.0`. For `x * x` that is `2 * x`,
so `6.0`.

## vjp — reverse mode

A vector-Jacobian product. This is what backpropagation computes, and it is the efficient choice
when a function has many inputs and few outputs — the usual shape of a loss function.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let g = vjp(square, 2.0, 1.0);
    return 0;
}
```

The arguments are the function, the point to evaluate at, and the seed — the vector to multiply the
Jacobian by, which for a scalar loss is `1.0`.

## jvp — forward mode

A Jacobian-vector product. The efficient choice in the opposite case: few inputs, many outputs.

```rust
fn square(x : f32) -> f32 {
    return x * x;
}

fn main() -> i32 {
    let d = jvp(square, 2.0, 1.0);
    return 0;
}
```

Same arguments, but the seed is a direction in the *input* space, and the result is how the outputs
move in that direction.

## Choosing between them

| You have | Use |
| --- | --- |
| Many inputs, one output (a loss) | `vjp` |
| One input, many outputs | `jvp` |
| A scalar function of a scalar | `grad` |

The cost of `vjp` scales with the number of outputs; the cost of `jvp` scales with the number of
inputs. That is the whole reason both exist.

## How it works

Differentiation is done by [Enzyme](https://enzyme.mit.edu/), which differentiates LLVM IR. Because
it works on the IR rather than on source, it differentiates through the optimiser's view of your
code, including calls into other functions.

Enzyme has to be present when the compiler is built — [Building from source](building.md) covers
installing it.

## Limits worth knowing

**A function must be differentiable to be differentiated.** Differentiating something with a
discrete result — an integer comparison, a branch on equality — is not meaningful. Vx does not yet
reject every such case: `grad` of a discrete-valued function is currently accepted rather than
refused, which is [Vx#503](https://github.com/vx-lang/Vx/issues/503). Until that is fixed, the
compiler will not stop you asking for a derivative that does not exist.

## Where to next

- [Compile-time evaluation](comptime.md) — the other thing that happens before the program runs
- [Topologies and memory](heterogeneous.md) — running the result on an accelerator
