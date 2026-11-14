# Unsafe and FFI

Vx checks a lot at compile time: ownership, borrowing, memory placement, contracts. Some things
cannot be checked — a pointer that came from C, a hardware register, a cast the compiler has no way
to justify. `unsafe` is where you take responsibility for those.

`unsafe` does not switch the checks off. It permits a small, specific set of operations that are
otherwise refused.

## unsafe blocks

An `unsafe` block is an expression. It can produce a value.

```rust
fn main() -> i32 {
    let x : i32 = 7;
    let p : *const i32 = &x;
    let v = unsafe { *p };
    return v - 7;
}
```

Reading through a raw pointer — `*p` — is the operation that needs the block. So does indexing
through one, or reading a field through one.

## Raw pointers

Two kinds, matching the two kinds of reference:

| Type | Meaning |
| --- | --- |
| `*const T` | A raw pointer you may read through. |
| `*mut T` | A raw pointer you may read and write through. |

Unlike `&T` and `&mut T`, raw pointers are not tracked by the borrow checker. Nothing stops two
`*mut T` pointing at the same value. That is exactly why dereferencing one needs `unsafe`.

A raw pointer can name a memory, as in `*mut f32 in Memory::GPU_HBM`. The functions of `core::ptr`
that move a pointer keep its memory: `offset`, `add` and `sub` and their `_mut` forms take a pointer
in any memory `M` and give a pointer in `M`. `read` and `write` take only a host pointer.

## unsafe functions

A function whose *body* is not the dangerous part — but whose *contract* is — should be marked
`unsafe` itself. A caller then has to opt in.

```rust
unsafe fn read_at(p : *const i32) -> i32 {
    return unsafe { *p };
}

fn main() -> i32 {
    return 0;
}
```

Unsafe-ness is part of the function's type. An `unsafe fn` cannot be passed where a safe function
is expected, so the obligation cannot be lost by storing the function in a variable.

The rule for when to mark a function `unsafe`: if a caller can make it misbehave by passing
something the compiler cannot check — a dangling pointer, a wrong length — it is `unsafe`. If the
function checks everything itself, it is not, even if its body uses `unsafe` internally.

## Pointer clauses

An `unsafe fn` can state what its caller must give it as `requires` clauses about its raw
pointers:

```rust
unsafe fn sum(p : *mut i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    let mut i = 0i64;
    while i < n {
        total = total + unsafe { p[i] };
        i = i + 1i64;
    }
    return total;
}

fn main() -> i32 {
    return 0;
}
```

- `valid(p, n)`: `p` is not null, is aligned, and points at `n` elements of one live allocation.
  The elements do not have to hold values yet.
- `initialized(p, n)`: each of the first `n` elements at `p` holds a value.
- `!overlaps(p, q, n)`: the first `n` elements at `p` and at `q` share no byte.

At a call in safe code, the compiler proves each clause from what it knows there: the clauses
of the caller, a pointer made from `&x` for a local variable `x`, and the pointer functions
`offset`, `add` and `sub`. A write `p[k] = x` or `*p = x` in `unsafe` code gives that one element
a value, so after `p[0] = 7i64` the compiler knows `initialized(p, 1i64)`. Two facts whose ranges
touch prove a clause together, so after `p[0] = 7i64; p[1] = 8i64;` it knows
`initialized(p, 2i64)`. A call that gets a raw pointer can free its memory, so after the call
the compiler knows nothing about memory that the caller does not own. When such code ended a
fact that the call needs, the error has a note with its line. In a loop, such code anywhere in
the body ends the facts at the start of the body, because an earlier pass can run it. A call in
`unsafe` code needs no proof: its author keeps the promise. A safe function with a pointer
clause cannot be a function value yet, because a call through the value would not prove the
clause.

`ensures !frees(p)` says that a call frees no memory that it reaches through `p`. Then a call
keeps the facts about the memory of its argument, and the compiler checks the body against the
clause. The compiler also finds `!frees(p)` itself when the body only reads and writes through
`p`, moves `p` with `offset`, compares it, and gives it only to functions with `!frees`. So
`split` compiles:

```rust
import core::ptr;

fn sum(p : *const i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    for i in 0i64..n {
        total = total + unsafe { p[i] };
    }
    return total;
}

fn split(p : *const i64, n : i64, k : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
    requires 0i64 <= k
    requires k <= n
{
    return sum(p, k) + sum(offset(p, k), n - k);
}

fn main() -> i32 {
    return 0;
}
```

`ensures frees(p)` says that the call can free that memory, so the compiler does not find
`!frees(p)` for it. An `extern` function can say `!frees(p)` after its return type, and the
compiler trusts it. It can also say a clause about numbers, such as `ensures 0i64 <= return`.
Each call then knows the clause, and `vxc --audit` lists it.

A function can also promise a pointer clause in an `ensures`, where `return` is the result. A
caller gets each such clause as a fact when the call returns. The body proves each one at each
`return` and at its end, after the storage of its local variables ends. So a function cannot
promise a pointer to one of its local variables. In an `unsafe` block, `assume e;` adds the fact
`e` without a proof, and it generates no code. Use it for a fact that the compiler cannot see,
such as the size of the memory that `alloc` gives. A wrong `assume` is undefined behavior.

```rust
import core::ptr;
import std::alloc;

fn alloc_i64(n : i64) -> *mut i64
    requires n > 0i64
    ensures valid(return, n)
{
    let p : *mut i64 = alloc(n * 8i64, 16i64);
    if is_null_mut(p) {
        panic("out of memory");
    }
    unsafe {
        assume valid(p, n);
    }
    return p;
}

fn first(p : *const i64) -> i64
    requires valid(p, 1i64)
    requires initialized(p, 1i64)
{
    return unsafe { p[0] };
}

fn main() -> i32 {
    let p = alloc_i64(1i64);
    unsafe {
        p[0] = 7i64;
    }
    let x = first(p as *const i64);
    let bytes : *mut i8 = p;
    dealloc(bytes, 16i64);
    return (x - 7i64) as i32;
}
```

The methods of `Vec` state such clauses. `as_mut_ptr` ensures `valid` and `initialized` for the
`len` elements of the vector, and `len` ensures that it returns that count. The methods that only
read the vector have `!frees(self)`. So safe code can give the pointer to `sum`. A `push` can move
the buffer, so after a `push` the pointer has no facts.

```rust
import core::ptr;
import std::vec;

fn sum(p : *const i64, n : i64) -> i64
    requires valid(p, n)
    requires initialized(p, n)
{
    let mut total = 0i64;
    for i in 0i64..n {
        total = total + unsafe { p[i] };
    }
    return total;
}

fn total(v : &Vec<i64>) -> i64 {
    let p = v.as_mut_ptr();
    return sum(p as *const i64, v.len() as i64);
}

fn main() -> i32 {
    let mut v = Vec<i64>::new();
    v.push(2i64);
    v.push(3i64);
    return (total(&v) - 5i64) as i32;
}
```

A vector that a function makes with `Vec<i64>::new()` owns only new memory. So a `push` on it, or
its drop, keeps the facts about a pointer parameter. That stops at a pointer or a reference into
the vector, and at a loop or a closure that uses it. It also stops at a call that can put memory of
the caller into the vector.

From a `spawn` until its `join()` or the end of its `tasks` block, the task can free memory, so
no fact about other memory starts in that time.

For `requires valid(p, n)`, the compiler tells LLVM that `p` is not null and aligned, and that
`n` elements at `p` can be read. For `requires !overlaps(p, q, n)`, it tells LLVM that an access
`p[i]` does not alias an access `q[i]`, where `i` is the variable of a `for i in 0..n` loop. After
each call of a function with `ensures valid(return, n)`, it tells LLVM that `n` elements at the
result can be read. A clause that does not hold is undefined behavior, as a wrong `*p` is.

`vxc --audit program.vx` prints each fact that the compiler trusts without a proof, one line each:

- each `assume`
- each pointer clause that a call in `unsafe` code trusts, because the compiler cannot prove it
- each `ensures` of an `extern` function
- each pointer `requires` of a `pub fn`, because a call from C proves nothing

Each line starts with the file, the line and the column, and the lines are in that order. So a
review can compare the audit of two versions of a program.

## Calling C

An `extern` block declares functions that exist outside Vx.

```rust
extern "C" {
    safe fn abs(x : i32) -> i32;
}

fn main() -> i32 {
    return abs(0);
}
```

By default an `extern` function is unsafe to call — the compiler knows nothing about what is on the
other side. `safe` marks one that genuinely is safe to call with any arguments its types allow, so
callers do not need an `unsafe` block.

`abs` qualifies: every `i32` is a valid input and it cannot misuse memory. A function taking a
pointer and a length would not qualify, because passing a mismatched pair breaks it.

Use `safe` sparingly. It is a promise the compiler cannot verify, and it is the one place in an
`extern` block where a mistake is silent.

## Calling Vx from C

An object file exports only the functions marked `pub`. Every other function the program defines
is hidden, so C cannot call it.

```rust
pub fn add_one(x : i32) -> i32 {
    return x + 1;
}
```

```sh
vxc add_one.vx --action emit-obj -o add_one.o
cc -shared -o libadd_one.so add_one.o
```

C, or Python's `ctypes`, can now load `libadd_one.so` and call `add_one`. Give a `pub` function
parameters and a result that C can pass: numbers, `bool` and raw pointers. Only a top-level
function that is not generic can be `pub` for now.

## Inline MLIR

The compiler turns a Vx program into MLIR, a set of small languages for compilers. Each of these
languages is a *dialect*: `arith` has integer and float arithmetic, `math` has functions such as
`ctpop` (count the bits that are 1), and `llvm` has the operations of LLVM. An `mlir!` block puts
MLIR text directly into a Vx function. Use it for one operation that Vx has no syntax for.

```vx
mlir!(inputs : (%a = x : i32), returns : i32, dialects : ["math"]) {
  %r = math.ctpop %a : i32
  macro.yield %r : i32
}
```

- `inputs` gives each Vx value to the block. `%a = x : i32` names the value of `x` as `%a` in the
  MLIR, with the MLIR type `i32`.
- `returns` gives the Vx type of the value of the block. `macro.yield` gives that value. A block
  that gives no value says `returns : void`.
- `dialects` lists each dialect whose operations the block uses.

This program has two functions whose bodies are `mlir!` blocks
(`book_unsafe_ffi_inline_mlir.vx`):

```rust
{{#include ../../../tests/backend/pass/book_unsafe_ffi_inline_mlir.vx:program}}
```

It prints:

```text
ones in 255: 8
ones in 1024: 1
larger of 3 and 7: 7
```

The `dialects` list must agree with the body. If `count_ones` listed `"arith"` and not `"math"`,
the compiler would give the error E8010: the block uses `math.ctpop`, but the list does not name
`math`. A name in the list that is not a dialect is also E8010.

The compiler does not check what the MLIR operations do. So treat each `mlir!` block as code that
the compiler trusts, as you treat an `unsafe` block. The standard library uses `mlir!` in this way.
`core::ptr` moves a raw pointer and compares two raw pointers with one LLVM operation each.

The compiler assumes one more thing about a block: a block that gets a raw pointer can free the
memory behind it. So a function that says `ensures !frees(p)` (see [Pointer
clauses](#pointer-clauses)) cannot give `p` to an `mlir!` block (E8012). `clobbers : [n]` tells the compiler that the block writes the variable `n`.

## What unsafe does not unlock

`unsafe` covers operations the type system cannot justify. It does **not** override placement.

A pointer into device memory still cannot be dereferenced from the host inside an `unsafe` block —
that is not an unchecked operation, it is a wrong one, and it stays a compile error. See
[Topologies and memory](heterogeneous.md).

## Where to next

- [Ownership and borrowing](ownership.md) — the checks `unsafe` steps around
- [Topologies and memory](heterogeneous.md) — the checks it does not
- [Standard library reference](stdlib-reference.md) — which `std` functions are `unsafe`, and why
