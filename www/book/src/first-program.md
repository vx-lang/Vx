# Your first program

## Hello, 42

Every Vx program starts at `main`, which returns an `i32`.

```rust
fn main() -> i32 {
    let x : i32 = 21;
    return x * 2;
}
```

Save that as `hello.vx` and run it:

```bash
vxc --run hello.vx
```

```
[JIT] Translating to LLVM IR...
[JIT] Optimizing LLVM IR (-O0)...
[JIT] Compiling to native object (-O0)...
[JIT] Linking native executable...
[JIT] Executing native binary...
[JIT] Program exited with code: 42
```

`--run` compiles the program and executes it immediately, then propagates the program's own exit
code. **A non-zero exit status from `vxc --run` is your program's return value, not a compiler
failure** — this program genuinely exits 42.

## Printing

`print` takes a value; `print!` takes a literal. Neither needs an import.

```rust
fn main() -> i32 {
    print!("the answer is ");
    print(42);
    print!("\n");
    return 0;
}
```

## Compiling ahead of time

The JIT is convenient for iterating. For anything you intend to keep, compile to a native
executable:

```bash
vxc -c hello.vx -o hello.o
```

`vxc --help` lists the other actions — `--emit-mlir` and `--emit-llvm` are the two you will reach
for most when you want to see what the compiler did with your code.

## Something with a shape to it

Types annotate a binding with `:`, `let mut` makes it mutable, and `for` ranges with `..`:

```rust
fn sum_to(n : i32) -> i32 {
    let mut total : i32 = 0;
    for i in 0..n {
        total += i;
    }
    return total;
}

fn classify(x : i32) -> i32 {
    if x > 10 {
        return 1;
    } else if x == 10 {
        return 2;
    } else {
        return 3;
    }
}

fn main() -> i32 {
    print(sum_to(10));
    print!("\n");
    print(classify(15));
    print!("\n");
    return 0;
}
```

> **If you forget a `return`, the error is not friendly yet.** A function that falls off the end
> without returning is currently caught late, by the MLIR verifier, which reports something like
> `block with no terminator, has %0 = "arith.addi"(...)` and no source location. It means a return
> is missing. Check that every path through the function returns a value.

## Arrays and tensors

An array literal is a tensor, and indexing reads an element back:

```rust
fn main() -> i32 {
    let a : Tensor<f32, [4]> = [ 1.0, 2.0, 3.0, 4.0 ];
    print(a[0]);
    print!(" ");
    print(a[3]);
    return 0;
}
```

`Tensor<f32, [4]>` is a tensor of four `f32` with its shape known at compile time. A `?` stands in
for a dimension that is not — `Tensor<f32, [?, ?]>` is a matrix whose extents are runtime values,
which you read with `.extent(0)` and `.extent(1)`.

## Structs and methods

```rust
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn magnitude_squared(self: &Point) -> i32 {
        return self.x * self.x + self.y * self.y;
    }
}

fn main() -> i32 {
    let p = Point { x: 3, y: 4 };
    return p.magnitude_squared();
}
```

The receiver is written out in full: `self: &Point` borrows it, `self: &mut Point` borrows it
mutably. There is no implicit `self`.

## Where to go next

You now have enough to write ordinary programs. Two directions from here:

- [A tour of Vx](tour.md) covers the rest of the language — generics, enums, pattern matching,
  ownership — none of which involves an accelerator.
- [Topologies and memory](heterogeneous.md) is the part that makes Vx different from every other
  systems language: placing data in a named memory space and having the compiler check it.
