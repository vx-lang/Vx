# Loop optimizations

`vxc` compiles a program to MLIR, and then to LLVM. MLIR has passes that fuse, tile and unroll
loops, but they work only on loops that the compiler writes as `affine.for`. This chapter explains
which loops get that form, what the passes do with them, and how to check a loop of your own.

The loop passes run at `-O1` and above. The default is `-O0`.

## What each loop becomes

The compiler writes each loop in one of two forms:

- An `affine.for`: a loop with a start, an end and a step of 1. MLIR's loop passes work on this
  form.
- Branches between blocks: a block that tests the condition, a block for the body, and a jump back.
  MLIR's loop passes do not see these loops. LLVM still optimizes them.

| Loop | Form |
| --- | --- |
| `for i in a..b` over `i32` or `i64`, and nothing leaves the loop | `affine.for` |
| The same loop with an `if` or a `match` in its body | `affine.for`, but fusion skips it |
| A nested loop with bounds from before the outer loop, or with the outer loop's variable, as in `for j in i..n` | `affine.for` |
| A nested loop whose bound the outer loop computes, as in `let m = i + 1;` before `for j in 0..m` | branches |
| A range over `usize`, `u32` or another unsigned type | branches |
| A loop that its own `break`, `continue`, `return` or `?` leaves | branches |
| `while`, `loop`, and `for x in v.iter()` | branches |
| Any loop in a `spawn` region, `across` loops too | branches |

A `break` or a `continue` in a nested loop leaves only the nested loop. So the loop around it can
still be an `affine.for`. But a nested loop in the branch form puts branches in the outer body, as an
`if` does, and fusion skips the outer loop.

The code that runs a `spawn` region on a GPU or on host threads works on loops in the branch form.
So these loops keep it.

## Tensor operations

An operation on a whole tensor is not a loop in your code, but it becomes one. The compiler writes
it as a `linalg` operation:

| Operation | MLIR |
| --- | --- |
| `a + b`, `a - b`, `a * b` and `a / b` on tensors of rank 2 or more | `linalg.add`, `linalg.sub`, `linalg.mul` and `linalg.div` |
| `t.map(..)`, `t.reduce(..)` and `-t` | `linalg.generic` |
| `Tensor::fill` | `linalg.fill` |
| `a @ b` | `linalg.matmul` |

Many methods of `std::tensor`, such as `fill` and `compare`, use `linalg` operations too.
Arithmetic on a 1-D float tensor is different: it becomes one `vector` operation over the whole
tensor.

At `-O1` and above, the `linalg` operations become `affine.for` loops. They can then fuse with each
other and with your loops.

## What the passes do

At `-O1` and above, the compiler runs these passes on each function, in this order:

1. **Inlining.** The compiler copies small functions into the functions that call them. The loops
   of a small function can then fuse with the loops around the call.
1. **`linalg` to loops.** Each `linalg` operation becomes `affine.for` loops.
1. **Affine accesses.** In a loop body without branches, each load and store of a tensor element
   becomes an affine load or store. The passes then know which elements a loop reads and writes.
1. **Fusion and tiling.** When a loop reads what the loop before it wrote, the two loops become one.
   The value stays in a register, and the program goes through the tensor once instead of twice.
   Tiling splits a loop into parts that fit in the cache.
1. **Unrolling.** Each iteration does the work of a few iterations, so the loop jumps back less
   often.
1. **Scalar replacement.** A load of a value that the same iteration stored becomes that value.
1. **Lowering.** The affine loops become `scf.for` loops, and then branches. LLVM then optimizes
   every loop at the same level, in either form.

Fusion and tiling run only in a function where no two tensors can share memory. Each tensor must be
one that the function makes, a global, or a `&` or `&mut` parameter. The compiler skips a function
that reads a tensor through a view, or that turns a tensor into a pointer. A row `t[i]` of a 2-D
tensor is a view. So a loop over `t[i][j]` does not fuse yet, but it is still unrolled.

## An example

These two loops each go through 64 elements:

```rust
fn add_then_square(a: &Tensor<f32, [64]>, b: &mut Tensor<f32, [64]>, c: &mut Tensor<f32, [64]>) -> i32 {
    for i in 0..64 {
        b[i] = a[i] + 1.0;
    }
    for i in 0..64 {
        c[i] = b[i] * b[i];
    }
    return 0;
}
```

`vxc --emit-mlir` shows two `affine.for` loops. `vxc -O2 --emit-mlir` shows the MLIR after the loop
passes, and there is one loop:

```text
scf.for %arg3 = %c0 to %c64 step %c1 {
  %0 = arith.addi %arg3, %c1 : index
  scf.for %arg4 = %arg3 to %0 step %c1 {
    %1 = memref.load %arg0[%arg4] : memref<64xf32>
    %2 = arith.addf %1, %cst : f32
    memref.store %2, %arg1[%arg4] : memref<64xf32>
    %3 = arith.mulf %2, %2 : f32
    memref.store %3, %arg2[%arg4] : memref<64xf32>
  }
}
```

The square uses `%2`, the sum, and does not load `b[i]` again. The inner loop of one iteration comes
from tiling.

## Check a loop

- `vxc --emit-mlir program.vx` prints the MLIR that the compiler writes, before the loop passes.
  Look for `affine.for` in the function.
- `vxc -O2 --emit-mlir program.vx` prints the MLIR after the loop passes, before LLVM. Two loops that
  fused are one loop there.

## Loops that MLIR can optimize

- Use an `i32` or `i64` range for a counted loop. A `usize` range keeps the branch form, and it also
  checks each step for overflow.
- Compute each bound before the loop.
- Keep `break`, `continue`, `return` and `?` out of a loop that should fuse. An `if` in the body also
  stops fusion.
- Pass the tensors of a function with hot loops as `&` and `&mut` parameters.

## Known problems

- At `-O1` and above, fusion can delete the writes to a tensor when a later loop in the branch form
  reads that tensor. The program then reads memory that nothing wrote
  ([#3395](https://github.com/vx-lang/Vx/issues/3395)).
- The older code generator (`--legacy-codegen`) writes every loop as branches.

[Loop optimizations in the frontend and in MLIR](https://github.com/vx-lang/Vx/blob/main/docs/discussions/loop_optimizations_in_the_frontend_and_in_mlir.md)
gives the design and its limits.
