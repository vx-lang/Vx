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
   `i` below the offset. The result is `a[0]`. On NVIDIA this is `shfl.sync.down`.
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
  loop can use the scratch again.
- **GPU, subgroup:** 5 shuffles and a broadcast, with no barrier.
- **CPU:** loops that compute the same tree. The CPU copy of a kernel and
  `-fauto-schedule-levels` use them. So does the AST code generator (`--legacy-codegen`), which
  runs a kernel on one lane.

## Not yet

- `f64` and the integer types.
- A form that takes each lane's value straight from the lane loop, with no tensor between them.
