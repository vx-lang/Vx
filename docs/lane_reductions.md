# Lane reductions

A lane reduction combines the elements of a tensor over the lanes of a block or a subgroup,
and gives every lane the result (#260).

<!-- vx-doctest: skip (needs a GPU topology) -->

```vx
for row in 0..rows across blocks {
  let mut part = unsafe { Tensor<f32, [1024], Memory::SMEM>::uninit() };
  for t in 0..1024 across lanes {
    part[t] = x[row][t] * x[row][t];
  }
  let ss = lanes::sum(&part, 1024);
  for t in 0..1024 across lanes {
    y[row][t] = x[row][t] / ss.sqrt();
  }
}
```

## The routines

| Routine | Result |
|---|---|
| `lanes::sum(&t, n)` | the sum, in the order below |
| `lanes::max(&t, n)` | the larger value, in the order below |
| `lanes::sum_unordered(&t, n)` | the sum, in an order that the compiler chooses |
| `lanes::max_unordered(&t, n)` | the larger value, in an order that the compiler chooses |
| `lanes::argmax(&values, &indices, n)` | the index of the larger value, in the order below |
| `lanes::argmax_unordered(&values, &indices, n)` | the index of the larger value, in an order that the compiler chooses |

- `t` is a rank-1 tensor of `f32` with a size `N` that is a number, in any memory. `&rows[w]`,
  one row of a rank-2 tensor, is one too.
- For the argmax, `indices` is a rank-1 tensor of `i32` with the size of `values`. A pair with
  an index below 0 is empty: it never wins. The result is an `i32`, and -1 when no pair is
  full.
- `n` is an `i32`. The routine combines `t[0]` to `t[n - 1]`. An `n` above `N` counts as `N`,
  and one below 0 counts as 0.
- Every lane gets the same bits. So the result is uniform: it can be the condition of an `if`
  around a lane loop.
- `lanes::` is a reserved prefix, as `raw::` is. The compiler types and lowers the calls.

## Where a call can be

A call is the whole value of a `let` in the code of a level: the code of a body of
`across blocks` or `across subgroups`, outside its lane loops. That is where a lane loop can be,
and every lane must get to the call, as to a lane loop (L5, L8, L10). E6032 refuses a call in
any other place.

- In the code of a block, `N` is at most 1024. To launch, the lanes of the block are a multiple
  of 32, and at least `N`.
- In the code of a subgroup, `N` is at most 32, and a subgroup has 32 lanes.

The routine is a function of `t` and `n` alone, so a program still means the same without each
`across`.

## The order

`a ⊕ b` is `a + b` for a sum, and `if b > a { b } else { a }` for the larger value. So a NaN in
`b` is dropped. For the argmax, `a ⊕ b` is `b` when `b` is full, and `a` is empty, or `b` has the
larger value, or the same value and a lower index. Otherwise it is `a`.

1. `v[i]` is `t[i]` for `i < n`. Past `n`, it is the pad: `0.0` for a sum, `t[0]` for the
   larger value, or negative infinity when `n` is 0, and an empty pair for the argmax.
1. **A group of 32:** for each offset in 16, 8, 4, 2, 1, `a[i] = a[i] ⊕ a[i + offset]` for each
   `i` below the offset. The result is `a[0]`. On NVIDIA this is `shfl.sync.down`. A sum uses
   `shfl.sync.bfly`: each lane adds the same two values as lane 0, only maybe in the other
   order, and `a + b` has the bits of `b + a` there.
1. **In the code of a subgroup**, the result is the group over `v[0..32]`.
1. **In the code of a block**, `p[w]` is the group over `v[32w..32w + 32]`, for each `w` below
   `W = (n + 31) / 32`. The result is the group over `q`, where `q[i]` is `p[i]` for `i < W`,
   and otherwise `0.0` for a sum, `p[0]` for the larger value, and an empty pair for the
   argmax. A block always takes both steps.

This is the order of the hand-written CUDA kernels of glm-vx, so their results stay the same to
the last bit. The argmax is the router's. The unordered routines may give a sum that differs in
the last bits. Today they use the same tree on the GPU, and combine from left to right on the
CPU.

## Lowering

The code generators emit `vx.lane_reduce`, or `vx.lane_argmax` for the argmax. Each copy of a
kernel expands it, and the argmax shuffles a value and an index at each step:

- **GPU, block:** a load for each lane, 5 shuffles, the result of each warp in 32 elements of
  `.shared` memory, a barrier, 5 more shuffles, a broadcast from lane 0, and a barrier, so that a
  loop can use the scratch again. A sum needs no broadcast.
- **GPU, subgroup:** 5 shuffles and a broadcast, with no barrier. A sum needs no broadcast.
- **A tensor in registers:** a lane loop from 0 over all the lanes of its level can write
  `part[t]`, or `part[w][t]` in the code of subgroup `w`. Then only lane `t` writes that element.
  When only lane reductions read `part`, the GPU kernel keeps each lane's element in a register,
  and `part` takes no memory. The larger value in the code of a block also needs a count that is
  a multiple of 32, because its pad is the element of lane 0.
- **CPU:** loops that compute the same tree. The CPU copy of a kernel and
  `-fauto-schedule-levels` use them. So does the AST code generator (`--legacy-codegen`), which
  runs a kernel on one lane.

## Sorts

`lanes::sort(&mut t, n)` sorts `t[0]` to `t[n - 1]` from the smallest value up, and
`lanes::sort_descending(&mut t, n)` from the largest value down. The other elements of `t` do not
change. A sort gives no value, so a call is a statement of its own, in the places that a lane
reduction can be.

- `t` is a rank-1 tensor of `i64`, with a size `N` that is a power of two. To sort other values,
  put each value in the high bits of an `i64` key, and its index in the low bits.
- `n` is an `i32`, with the limits of a lane reduction.
- In the code of a block, the lanes of the block are a power of two and at least 32. In the code
  of a subgroup, a subgroup has 32 lanes. Each lane holds at most 8 elements, so `N` is at most
  8 times the lanes.

A sort of integers has one result, so every copy of a kernel gives the same bits. Each copy uses
a bitonic sorting network. On the GPU, each lane holds its elements in registers. A step of the
network with a distance below 32 uses `shfl.sync.bfly`, and a step with a distance of at least
the lanes stays in the registers of a lane. Only the steps between them use shared memory and
barriers: 20 of the 66 steps for 2048 elements on 1024 lanes. The CPU runs the same network in
loops.

## Copies

`lanes::copy_async(&mut tile, &x, [i, j])` starts a copy of the tile of `x` that starts at
`[i, j]` into `tile`, and gives a token. `t.wait()` takes the token and waits until the copy
ends. Between the two, the lanes can work on another tile, so the next copy runs while the lanes
use the last one:

<!-- vx-doctest: skip (needs a GPU topology) -->

```vx
let mut tiles = unsafe { Tensor<f32, [2, 64, 64], Memory::SMEM>::uninit() };
for k in 0..steps / 2 {
  let even = lanes::copy_async(&mut tiles[0], x, [0, k * 128]);
  let odd = lanes::copy_async(&mut tiles[1], x, [0, k * 128 + 64]);
  even.wait();
  // The lanes read tiles[0] here, while the copy into tiles[1] goes on.
  odd.wait();
  // The lanes read tiles[1] here.
}
```

- `tile` is a tensor in shared memory with a size that is a number, such as a row of a rank-3
  tensor. Its elements have 4 or 8 bytes. `x` has the same element type and the same number of
  dimensions. The place is a list of integers, one for each dimension.
- An element of the tile past the end of `x`, or before its start, is zero.
- The token holds `tile` as `&mut` until its wait, so the lanes cannot read the tile before the
  copy ends. Two tokens can hold two different rows of one tensor.
- A token must be waited (E6036).
- The copy is the whole value of a `let`, and the wait a statement of its own, in the code of a
  loop of blocks or subgroups. In the code of a subgroup, the lanes of the subgroup copy the
  tile, and the tile is usually the row of the subgroup, such as `&mut tiles[w]`.

On the GPU, each lane starts a `cp.async` for its elements of the tile, and the copies of one
call are one group. The token is the number of groups that the lane started before it. The wait
lets the groups that started after the token go on (`cp.async.wait_group`), and then every lane
of the block or the subgroup waits. The CPU copies the tile at the call, and the wait does
nothing.

## Not yet

- `f64` and the integer types.
