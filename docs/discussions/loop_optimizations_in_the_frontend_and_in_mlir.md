# Loop optimizations in the frontend and in MLIR

Issues: #2042 (affine loops), #252 (`across` loops).

## The question

Some loop optimizations need facts that only the language has. Others need only the code. This
document decides where each one goes, and how the frontend can give the MLIR loop passes more
loops to work on.

## The rule

- The frontend does a rewrite that needs a fact from the language: that two passes do not race,
  which lane owns which data, or what an `across` loop means.
- MLIR and LLVM do a rewrite that a compiler can prove from the code alone: CSE, hoisting,
  fusion and tiling of affine loops, SROA, scheduling and register allocation.

The frontend has the facts at the source level. After lowering, the same facts are spread over
stack slots and index arithmetic, and a pass must prove them again, often more weakly.

| Optimization | Where | Why | | --- | --- | --- | | Fuse adjacent lane loops and remove the
barrier between them | frontend | Only the race proof (L4) knows that lane `i` reads only what
lane `i` wrote. | | Keep a per-lane array in registers | frontend and LLVM | The frontend gives
the array to one lane and unrolls the loops over it. LLVM's SROA moves it into registers. | |
Fuse, tile and vectorize counted CPU loops | MLIR, from `affine.for` | The affine passes do this
already, but they never see our loops. | | CSE, hoisting, scheduling, register allocation | MLIR
and LLVM | These need nothing from the language. |

## Loops today

The flat code generator (`lower_for_loop` in `src/hir/flatten.rs`) lowers `for i in 0..64` as
branches between blocks. The loop variable and the bound are in stack slots:

```mlir
  memref.store %c0_i32, %alloca[] : memref<i32>
  cf.br ^bb1
^bb1:
  %0 = memref.load %alloca[] : memref<i32>
  %2 = arith.cmpi slt, %0, %1 : i32
  cf.cond_br %2, ^bb2, ^bb4
^bb2:
  ...
```

MLIR's loop passes work on `affine.for` and `scf.for`, so they see none of these loops. Today
the only affine loops come from `linalg` ops, through `convert-linalg-to-affine-loops` in the
`-O1` pipeline (`src/driver.rs`).

## What `affine.for` needs

- The step is a positive constant.
- Each bound is affine in constants, values defined outside every affine loop, and the indices
  of the loops around it. A value made inside an affine loop is not a valid bound, even a cast
  of a valid one. `mlir-opt` refuses it with "operand cannot be used as a symbol".
- The body is one block. A body with branches can go in an `scf.execute_region`, but then the
  fusion pass skips that loop.
- No `break`, `continue`, `return` or `?` leaves the body.
- The passes see a load or store only as `affine.load` or `affine.store`.
  `affine-raise-from-memref` turns a `memref.load` into one when its index is affine.

A quick test with `mlir-opt` (LLVM 22) showed that this is enough. In two loops of the form
below, `canonicalize` removes the two casts, `affine-raise-from-memref` raises the accesses, and
`affine-loop-fusion` fuses the loops into one:

```mlir
affine.for %iv = 0 to 64 {
  %i = arith.index_cast %iv : index to i32
  %k = arith.index_cast %i : i32 to index
  %x = memref.load %a[%k] : memref<64xf32>
  ...
  memref.store %y, %b[%k] : memref<64xf32>
}
```

So the frontend does not need to emit `affine.load`. It emits the loop, and the passes do the
rest.

## Which Vx loops qualify

| Loop | Affine? | | --- | --- | | `for i in a..b` with `a` and `b` known before the loop, a
signed range, and no early exit | yes | | the same, inside another qualifying loop, with bounds
that are constants, outer values or an outer loop variable | yes | | `while`, `loop`, and
`for x in v.iter()` | no | | a loop that a `break`, `continue`, `return` or `?` leaves | no | |
`across blocks` | no: the step is `grid_dim`, and a loop that starts at `block_id` and steps by
`grid_dim` has no affine form, because affine maps divide only by constants | | `across lanes`
as emitted now | no: the step is `min(w, block_dim)`. On the device the step is always `w`. If
the CPU copy got its own form, the device loop would qualify. | | a loop in a `spawn` region, or
a loop that the parallel pipeline owns | no: the GPU and host-worker lowerings read these loops
in their branch form |

A loop that qualifies but indexes with data (`t[idx[i]]`) is still a correct `affine.for`. Its
accesses stay `memref.load`, and the passes leave them alone.

## The plan

1. **Pick the loops in the frontend.** `lower_for_loop` emits an affine loop when the loop
   qualifies (see the table). It decides from the syntax, and from the place where the code
   computes each bound. Otherwise it emits the branch form, as it does today.
1. **Two new instructions.** `AffineFor` opens the loop with the two bounds, and `AffineForEnd`
   closes it. The loop variable is the cast of the index to the range type, bound like any other
   local. If the body has branches, the loop holds an `scf.execute_region`.
1. **Bounds.** The loop holds a constant bound as a number. Any other bound is cast to `index`
   just before the outermost open affine loop, so the cast is a valid symbol. A bound that is
   the variable of an enclosing affine loop uses that loop's index directly.
1. **The pipeline.** `vx-raise-loop-accesses` runs before `vx-fuse-and-tile-loops` at `-O1` and
   above. It runs `affine-raise-from-memref`, and then turns back to `memref` each access that
   does not reach an `affine.for` through regions of one block. At `-O0`, `lower-affine` turns
   the loops back into branches.
1. **GPU kernels.** A function that a kernel calls can hold an affine loop. The device check
   accepts the affine dialect, and the GPU module pipeline lowers it.
1. **Measure.** Count the loops in `tests/` and in the book that become affine. Count how many
   of them are in functions that `vx-fuse-and-tile-loops` skips.

## Limits

- `vx-fuse-and-tile-loops` skips a function where two buffers can alias, for example one read
  through a `reinterpret_cast` view. Many tensor accesses use a view. To fuse those loops, the
  frontend must pass alias facts to MLIR. That is later work.
- The affine passes do not understand `gpu.barrier`. Lane-loop fusion stays in the frontend.
- `affine-scalrep` assumes that no branches run between the accesses it sees. The upstream
  `affine-raise-from-memref` also raises an access in a region of several blocks. With the two
  together, `affine-scalrep` reused a load across a store in a branch, and
  `core_libm_logs_bf16.vx` printed wrong hashes at `-O2`. So only an access in a loop without
  branches stays affine.
- A bound that the loop around it computes, such as `a.extent(1)` in a matmul, keeps the inner
  loop in branch form. Hoisting a bound that has no side effects would make more loops qualify.
- The AST code generator keeps its own loops. It stays a debugging oracle, so it does not need
  the same speed.

## Later work

- Fuse adjacent lane loops in the `across` lowering, using the race proof's facts (#252).
- Give each per-lane array its own stack space in its lane loop, and unroll the loops over it.
  Check in the PTX that the array stays in registers.
- Affine bounds of the form `i + 1` and `2 * i`.
- Pass `noalias` facts for views, so that fusion runs in more functions.
