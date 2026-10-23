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

The block's value, and every check in it, has to be knowable at compile time. A `comptime`
block cannot call into C or touch a device. It may read a value that is known only when the
program runs, but only where nothing the block produces depends on it. [A comptime block must
run](#a-comptime-block-must-run) has the rules.

Arithmetic gives the same answers it would at run time. Two integers divide as integers, so
`7 / 2` is `3`. An integer result outside its type, such as an `i32` above 2147483647, has no
compile-time value at all, rather than a wrapped one. A block whose value or whose `assert`
depends on such a result is an error.

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

## Fixed-size arrays

Compile-time code can build a fixed-size array of scalars, read its elements, and write them
back. The length is fixed when the array is built:

```rust
fn main() -> i32 {
    comptime {
        let mut a : Tensor<i64, [3]> = [ 3i64, 1i64, 2i64 ];
        a[0] = 9i64;
        assert(a[0] == 9, "the write happened during compilation");
    }
    return 0;
}
```

Because both the array and the index are known while compiling, an index past the end is a
compile error rather than a bad read at run time:

```rust
let a : Tensor<i64, [3]> = [ 3i64, 1i64, 2i64 ];
let x = a[5];   // error[E8003]: index 5 is out of range: this array has 3 elements
```

An index the compiler cannot work out, such as the result of a call, leaves it unable to follow
the write. When that happens the array stops having a compile-time value entirely, rather than
keeping the value it had before the write. A later `assert` on that array cannot be decided, so
it is an error (E3033).

Only scalars can go in such an array today. Structs, arrays of arrays, and anything that grows
are not compile-time values yet.

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
    let t : Tensor<f32, [2, 2]> = Tensor<f32, [2, 2]>::new();
    return 0;
}
```

Because the shape is in the type, a matrix multiply whose dimensions do not line up is a compile
error rather than a run-time crash — the same argument as [contracts](contracts.md), applied to
dimensions.

## A comptime block must run

A `comptime` block is not a hint. It runs while compiling and leaves nothing behind, and the
compiler holds you to both halves:

- If the evaluator cannot finish the block, that is an error, not a quiet fall back to running the
  code at run time.
- A block may not write to anything declared outside it. The block disappears, so the write would
  have to disappear with it.
- A block may not sit inside another one. The outer block already runs while compiling.
- The loops in a block, and in the functions it calls, may run 1,000,000 times in all. More is an
  error (E8005).

The program makes some checks while it runs. An `assert` stops it when its condition is false.
An integer `/` or `%` stops it when it divides by zero, or when it divides the minimum value by
-1. The block disappears, so a check in it would disappear too. The compiler makes each check
while compiling instead.

A false `assert` is an error (E8002):

```rust
fn main() -> i32 {
    comptime {
        let mut v : i64 = 1i64;
        v = v + 1i64;
        assert(v == 1i64, "v is 2");   // error[E8002]: Comptime assert failed: v is 2
    }
    return 0;
}
```

A check that depends on a value known only when the program runs, such as `argc`, is an error
(E3033). The compiler cannot decide it:

```rust
fn main(argc : i32) -> i32 {
    comptime {
        assert(argc > 0, "at least one argument");   // error[E3033]
    }
    return 0;
}
```

A check on a path that runs only for some run-time values, such as one branch of an `if` on
`argc`, is an error (E3033) when it fails on that path. The path may never run, but the
compiler cannot know that:

```rust
fn main(argc : i32) -> i32 {
    let x = comptime {
        if argc > 1 {
            let _q = 1i64 / 0i64;   // error[E3033]: this division stops the program
        };
        3
    };
    return x;
}
```

An integer overflow does not stop the program, which wraps the result. So a block whose value
does not depend on an overflow still runs.

This is the same choice Zig makes with its `comptime`, and it is why C++ grew `consteval` alongside
`constexpr`: a `constexpr` function only *may* be evaluated while compiling, and when it cannot be,
it silently becomes an ordinary call. You find out by reading the disassembly. Here you find out by
the compiler refusing.

## Lambdas that run while compiling

A closure whose body is a `comptime` block is a function that runs while compiling:

```rust
fn main() -> i32 {
    let base = 10;
    let twice = || comptime {
        let k = 2;
        base * k
    };
    return twice();
}
```

The body is not worked out where the lambda is written — its parameters have no values yet. It is
worked out when it is called, so `twice()` is `20`, and the multiplication never reaches the
generated code.

A lambda that takes its arguments and captures nothing is dropped entirely once its calls have
folded, leaving a single constant. One that captures a variable, as `twice` captures `base` above,
still has its call emitted today; only its body has folded. Removing that call as well is not done
yet.

The consequence is worth stating plainly. A lambda like this cannot be called with a value that is
only known at run time — that call is an error rather than a run-time call. If you want a closure
that runs at run time, do not give it a `comptime` body.

## Functions that run while compiling

A function whose whole body is `return comptime { .. };` is the named form of such a lambda. Each
call runs while compiling and is replaced by its value, and the function itself is not compiled into
the program:

```rust
fn twice(x : i64) -> i64 {
    return comptime { x * 2i64 };
}

fn main() -> i32 {
    let t = twice(21i64);   // the program holds 42
    return 0;
}
```

As with the lambda, a call with an argument that is only known at run time is an error. A function
without parameters, which names an expression, is also worked out where it is written, so a mistake
in it is reported even when nothing calls it.

Inside a `comptime` block a string literal is a byte buffer the block can write through a pointer,
and a pointer result becomes a string constant in the program. `core::num` uses this for
`decimal_c_str`, which writes a number's digits while compiling:

```rust
import core::num;

fn main() -> i32 {
    let n : i64 = -42i64;
    print!(n.decimal_c_str());   // the program holds "-42"; nothing is worked out at run time
    return 0;
}
```

A `comptime` block may put at most 1 MiB in such buffers. Compile-time integers are `i64`, so a
`u64` from 2^63 up cannot be worked out yet.

## Where to next

- [Generics and traits](generics.md) — the rest of the generic system
- [Contracts and verification](contracts.md) — other things checked before the program runs
