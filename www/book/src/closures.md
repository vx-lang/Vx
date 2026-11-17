# Closures and function values

A *closure* is a function without a name that you write inside an expression. It can use the
local variables of the code around it. A *function value* is a function that you keep in a
variable or give to another function. This chapter shows how to write closures, how they hold the
locals they use, and how to pass closures and named functions to other functions.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/backend/pass/` with the name that its section gives.

## Writing a closure

A closure starts with its parameters between two `|` characters. Each parameter has a type. A
short closure has one expression as its body:

```vx
let double = |x : i64| x * 2;
```

A longer closure has a block as its body. Then you write the return type after `->`, as for a
function. A closure without parameters starts with `||`. You call a closure as you call a function:
`double(4i64)`. Write the type of an integer argument, as in `4i64`. A call of a closure does not
convert the literal `4` to an `i64` yet ([Vx#3234](https://github.com/vx-lang/Vx/issues/3234)).

When a closure uses a local of the code around it, the closure *captures* the local. The compiler
looks at what the body does with the local, and chooses one of three ways:

- The body only reads a number or a `bool`. The closure keeps a copy of the value.
- The body changes the local. The closure borrows the local as `&mut`, so each call changes the
  local itself, not a copy.
- The body only reads a struct. The closure borrows the struct as `&`.

A borrow lasts until the last use of the closure, and the rules of
[Ownership and borrowing](ownership.md) apply to it. This program shows the two forms and the three
ways (`book_closures_capture.vx`):

```rust
{{#include ../../../tests/backend/pass/book_closures_capture.vx:program}}
```

It prints `double 8 add_k 14`, then `total 12`, then `peek 3 count 3`. The closure `add` changed
`total` twice. `main` reads `c.count` after `peek`, because `peek` only borrowed `c`.

The compiler checks each way:

- A closure keeps the value it copied. If the program writes the local after it makes the closure
  and then calls the closure, the closure does not see the write. The compiler gives the warning
  W1034. A later release makes this an error.
- While a closure holds a local as `&mut`, the code around it cannot read or write that local
  before the last use of the closure (E4002).
- A closure whose body moves a struct, for example by passing it by value to a function, takes the
  struct. After that, the code around the closure cannot use the struct (E4001).

Inside a closure, `?` needs the return type written:
`|x : i32| -> Result<i32, E> { .. }`. [Vx for Rust programmers](for-rust-programmers.md) says
more about `?`.

A closure that does no work while the program runs, but runs while compiling, has a `comptime`
body. [Compile-time evaluation](comptime.md) describes it.

## Passing a closure to a function

A *closure type* names the parameters and the result of a closure: `|i64| -> i64` takes an `i64`
and returns an `i64`. `|| -> i64` takes nothing. A function parameter with a closure type accepts a
closure, and the function calls it as `f(x)`.

A closure that you write directly as an argument gets the parameter's type. A closure that you keep
in a variable needs `as` and the closure type: `call_twice(next as || -> i64)`. A generic function
can use its type parameters in the closure type, for example `f : |T| -> T`.

The standard library takes closures too. `Option::map`, `Result::map` and the iterator methods such
as `fold` and `any` take a `Closure1<T, U>` (one argument) or a `Closure2<A, B, U>` (two
arguments). These types are in `core::ops`. You can give such a method a closure literal or a
closure in a variable, with no `as` (`book_closures_as_arguments.vx`):

```rust
{{#include ../../../tests/backend/pass/book_closures_as_arguments.vx:program}}
```

It prints `twice 13`, then `apply 21`, then `calls 3 count 2`, then `map 15`. The closure `next`
changed `count` on each of its two calls. The borrow of `count` ended with the call to
`call_twice`, so `main` can read `count` after it.

A function cannot return a closure that uses a local. The closure would point into the function's
stack frame, and the frame is gone after the return (E4005). For the same reason, a function cannot
store such a closure through a reference that it gets. A closure that uses no local can be
returned:

```vx
fn make() -> || -> i32 {
  let c = || 40 + 2;
  return c as || -> i32;
}
```

### Calling a `Closure1` in your own function

There is no call syntax for a `Closure1` value yet. A `Closure1` is a struct with two fields: `env`
points at the locals that the closure captured, and `func` is the code. To call it, you take `func`
and give it `env` as the first argument:

```vx
import core::ops;

fn apply_c1(f : Closure1<i64, i64>, x : i64) -> i64 {
  let p = f.func;
  return p(f.env, x);
}
```

The standard library calls its closures this way. In your own code, a closure type such as
`|i64| -> i64` is shorter and safer, because `env` is a raw pointer.

## Function pointers

A named function is a value too. Its type is a *function pointer type*, written with `fn`:
`fn(i64) -> i64` takes an `i64` and returns an `i64`. You give the function by its name, with no
parentheses. A variable or a struct field can hold it, and a function parameter can take it
(`book_closures_function_pointers.vx`):

```rust
{{#include ../../../tests/backend/pass/book_closures_function_pointers.vx:program}}
```

It prints `apply 25`, `g 8`, `step 6`, `hello` and `run 7`, each on its own line. `run` is generic,
and `T` is `void` for `say_hello`, a function that returns nothing.

To call a function pointer in a struct field, first copy the field into a variable, as `op` above
does. A call such as `(s.op)(3i64)` does not compile yet
([Vx#3234](https://github.com/vx-lang/Vx/issues/3234)).

Function pointer types and closure types are different types. A function that takes
`fn(i64) -> i64` does not accept a closure, and a function that takes `|i64| -> i64` or a
`Closure1` does not accept a named function (E3003). A function pointer holds no captured locals,
so a closure cannot fit in one.

### `unsafe fn` values

An `unsafe fn` has the type `unsafe fn(..) -> ..`, so its contract stays with the value. You can
call it through a variable only in an `unsafe` block (E5001). You cannot give it to a parameter of a safe
function pointer type:

```vx
unsafe fn raw_double(x : i64) -> i64 {
  return x * 2;
}

fn apply(f : fn(i64) -> i64, x : i64) -> i64 {
  return f(x);
}

fn main() -> i32 {
  // Error E3003: expected fn(i64) -> i64, got unsafe fn(i64) -> i64
  let y = apply(raw_double, 4);
  return 0;
}
```

If a safe function accepted it, any safe code could call the `unsafe fn` through that function,
without an `unsafe` block. [Unsafe and FFI](unsafe-and-ffi.md) explains `unsafe fn`.

## Closures and tasks

Do not wait for a task inside a closure yet. A wait inside a closure blocks the thread that runs
it, so other tasks cannot use that thread ([Vx#3212](https://github.com/vx-lang/Vx/issues/3212)). A `spawn` in a closure is an error
(E3056). [Tasks and channels](tasks.md) shows where a `spawn` can go.
