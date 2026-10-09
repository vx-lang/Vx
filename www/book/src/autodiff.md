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

## Larger examples

These use `f64`, with `exp`, `ln_1p` and `cos` from `core::num`. Pass the point as an `f64`
variable: a bare literal such as `0.5` is read as an `f32`, and `grad` then refuses it
([#1446](https://github.com/vx-lang/Vx/issues/1446)).

### Through calls to other functions

```rust
import core::num;

fn log_one_plus(t : f64) -> f64 {
    return t.ln_1p();
}

// softplus(x) = ln(1 + e^x), a smooth version of max(0, x).
fn softplus(x : f64) -> f64 {
    return log_one_plus(x.exp());
}

fn sigmoid(x : f64) -> f64 {
    return 1.0 / (1.0 + (-x).exp());
}

fn main() -> i32 {
    let x : f64 = 0.5;
    print(grad(softplus, x));
    print!(" ");
    print(sigmoid(x));
    return 0;
}
```

The derivative of softplus is the sigmoid, and the program prints both:
`0.6224593312018545 0.6224593312018546`. The derivative goes through `exp` and then through
`log_one_plus` by the chain rule, and neither function had to be marked. The last digit differs
because the two numbers are computed in different ways.

### Branches and loops

The derivative follows whichever branches and loop iterations the function takes at that point.

```rust
// The Huber loss: quadratic near zero, linear further out.
fn huber(x : f64) -> f64 {
    if x < -1.0 {
        return -x - 0.5;
    }
    if x > 1.0 {
        return x - 0.5;
    }
    return 0.5 * x * x;
}

fn main() -> i32 {
    let near : f64 = 0.25;
    let far : f64 = 3.0;
    print(grad(huber, near));
    print!(" ");
    print(grad(huber, far));
    print!(" ");
    print(grad(huber, -far));
    return 0;
}
```

This prints `0.25 1 -1`, the slope of the piece each point falls in.

```rust
// x raised to the n, by multiplying n times.
fn power(x : f64, n : i32) -> f64 {
    let mut result : f64 = 1.0;
    for i in 0..n {
        result = result * x;
    }
    return result;
}

fn main() -> i32 {
    // The derivative of x^5 is 5 x^4, which is 80 at x = 2.
    let x : f64 = 2.0;
    print(grad(power, x, 5));
    return 0;
}
```

This prints `80`. The count `n` comes after `x`, so it is passed through unchanged and the loop
runs five times.

### Partial derivatives

`grad` moves the first parameter and holds the others fixed. To differentiate in another
parameter, write a function that takes it first.

```rust
fn f(x : f64, y : f64) -> f64 {
    return x * x * y;
}

// f with y as its first parameter, so grad moves y.
fn f_in_y(y : f64, x : f64) -> f64 {
    return f(x, y);
}

fn main() -> i32 {
    let x : f64 = 3.0;
    let y : f64 = 2.0;
    print(grad(f, x, y));
    print!(" ");
    print(grad(f_in_y, y, x));
    return 0;
}
```

This prints `12 9`: the derivative in `x` is `2xy`, and in `y` it is `x * x`.

### Using the derivative in a loop

A derivative is an ordinary function call, so a program can call it as often as it needs.
Newton's method uses it to find where a function is zero:

```rust
import core::num;

fn gap(x : f64) -> f64 {
    return x.cos() - x;
}

fn main() -> i32 {
    // Newton's method for the x where cos x = x.
    let mut x : f64 = 1.0;
    for step in 0..6 {
        x = x - gap(x) / grad(gap, x);
    }
    print(x);
    return 0;
}
```

This prints `0.7390851332151607`, the number whose cosine is itself.

Gradient descent uses it to fit a model. Here the model is a line through the origin, `y = w x`,
and the loss measures how far it is from three points:

```rust
// How far the line y = w x is from three measured points, squared and added up.
fn loss(w : f64) -> f64 {
    let e1 = w * 1.0 - 2.0;
    let e2 = w * 2.0 - 4.1;
    let e3 = w * 3.0 - 5.9;
    return e1 * e1 + e2 * e2 + e3 * e3;
}

fn main() -> i32 {
    let mut w : f64 = 0.0;
    for step in 0..50 {
        w = w - 0.02 * grad(loss, w);
    }
    print(w);
    return 0;
}
```

This prints `1.9928571428571429`, the best slope for those points (27.9 / 14).

## How it works

Differentiation is done by [Enzyme](https://enzyme.mit.edu/), which differentiates LLVM IR. Because
it works on the IR rather than on source, it differentiates through the optimiser's view of your
code, including calls into other functions.

Enzyme has to be present when the compiler is built — [Building from source](building.md) covers
installing it.

## Limits worth knowing

**A function must be differentiable to be differentiated.** A derivative needs both ends
continuous, and the compiler checks both at the call:

- the **result**. An `i32`, a `bool`, a tensor of integers — these take separated values, so
  between any two of them there is no limit to take.
- the **value it is taken with respect to**, which is the first parameter, for the same reason.

```
Error: Function 'discrete_func' cannot be differentiated because it returns the discrete type i32
```

A *later* parameter may be discrete. A function of an `f32` that also takes an index or a loop
count is an ordinary thing to differentiate, and only the first argument is the one being moved.

## Where to next

- [Compile-time evaluation](comptime.md) — the other thing that happens before the program runs
- [Topologies and memory](heterogeneous.md) — running the result on an accelerator
