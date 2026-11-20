# Parallel code on a GPU

A GPU runs one piece of code on many threads at the same time. Vx writes that code as plain loops
in a `spawn on` region. A word after the range of a loop, `across blocks` or `across lanes`, says
which level of the GPU runs its passes. The compiler checks that two passes cannot write the same
element, so the loops give the same result on the GPU as they do one pass at a time.

Each program in this chapter is also a test of the compiler. You can find it under
`tests/optimizations/pass/` with the name that its section gives. The programs use the machine
file of an NVIDIA H100, so run them with this command:

```sh
vxc program.vx --machine "$HOME/.vx/current/fleet/h100-sxm.vx" --host default
```

[Machine files](machine-files.md#using-one) says where the install puts these files. On a computer
with no GPU, the runtime prints `no CUDA device` and runs each kernel on the CPU. The result is the
same.

## Blocks and lanes

A GPU has two levels of threads. A block is a group of threads that runs on one part of the chip,
and a lane is one thread in a block. The lanes of a block can share a fast memory, and wait for each
other. Two blocks cannot.

`for y in 0..64 across blocks` runs each pass in a block of its own. `for x in 0..300 across lanes(256)` runs the passes on 256 lanes, so some lanes run two passes. This program doubles each
element of a tensor in GPU memory, and adds the number of its row, `y`
(`book_gpu_blocks_and_lanes.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_blocks_and_lanes.vx:program}}
```

It prints `corner 2, last row 65`.

The meaning of the program does not depend on `across`. Remove each `across`, and the loops run one
pass at a time with the same result. The words only tell the compiler where to run the passes.

The compiler checks these rules, and gives the error E6032 when a rule is broken:

- An `across` loop is in a `spawn on` region for a device that has blocks and lanes, such as an
  NVIDIA or an Apple GPU. [Topologies and memory](heterogeneous.md) explains `spawn on`.
- The block loop is directly in the region. A lane loop is in the block loop.
- A block has at most 1024 lanes. The program writes their number as a constant, as in
  `lanes(256)`.
- A `break`, `continue` or `return` cannot leave an `across` loop, because each lane must get to
  the end of the loop.
- At most three block loops can be in each other, and at most three lane loops, because the grid
  and the block of a GPU have three dimensions.

## Two passes cannot write the same element

The passes of a block loop run at the same time, in any order. So each write in the loop must use
the variable of the loop in its index, so that each pass writes other elements. In `grid[y][x]`,
`y` is different in each pass, so no two passes write the same element. A write to `grid[0][x]`
does not compile:

```text
Error[E6032]: two passes of this block loop may write the same element of `grid`: an index of the
write must fix `y`
```

One block loop can cover a grid of tiles, with `b / n` and `b % n` as the row and the column of
the tile. In `c[(b / n) * 64 + i][(b % n) * 64 + j]`, the first index fixes `b / n` and the second
fixes `b % n`, so the two together fix `b`. `n` must be a number above zero, or a factor of the
end of the loop, as in `for b in 0..m * n across blocks`. When `n` is zero, that loop has no passes.

An index can also use `i / n` and `i % n`, where `i` is the variable of a loop in the block loop,
with the same rule for `n`. With `i` below 4 and `j` below 32, `(i / 2) * 32 + j` is at most 63.
So `c[b * 64 + (i / 2) * 32 + j]` fixes `b`.

The parts can also be parts of parts, with numbers that each divide the next. The result of a
tensor core is an example: in a block of 128 lanes, lane `l` holds row `16 * (l / 32) + (l % 32) / 4` and the row 8 below it, at columns `2 * (l % 4)` and one more. The compiler cuts `l` into the
digits `l % 4`, `(l % 32) / 4` and `l / 32`. It writes each part as a sum of these digits, so
`l / 4` is `(l % 32) / 4 + 8 * (l / 32)`. The indices must fix every digit. Here the row fixes the
two upper digits, and the column fixes `l % 4`, so the writes fix `l`.

Two loops that come one after the other, such as a `for j in 0..2` in each of two `if` blocks,
count as one loop when their ranges are the same numbers. So a write in one loop and a write in
the other can use the same column `col + 8 * j`.

The same check refuses a read of an element that another pass writes, a write to a variable from
outside the loop, `println!`, and a call that can write through its argument.

## Rows from a tensor of indices

An index can also come from a tensor, as in `y[idx[b]]`. The compiler cannot see the values in
`idx`, so two passes may get the same row, and the write is E6032. A mixture-of-experts layer in
a language model needs this write: the kernel of each expert writes its results to the rows of its
tokens.

When no two elements of `idx` are equal, say so in its type:
`Distinct<Tensor<i32, [?], Memory::HBM>>`. Then `idx[b]` fixes `b`, as `b` does.

`Distinct::check(t)` makes a `Distinct` tensor. It reads the values of `t` on the CPU, and returns
`Some` with a `Distinct` tensor when no two are equal, and `None` when two are. A program that
matches on the result imports `core::option`. In this program, the rows 5, 1 and 6 pass the check
and move to GPU memory. Then block `b` adds row `b` of `x` to row `rows[b]` of `y`
(`book_gpu_rows_from_a_distinct_tensor.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_rows_from_a_distinct_tensor.vx:program}}
```

It prints `column 0: 1 3 1 1 1 2 4 1` and the same line for column 63.

`Distinct::check(t)` takes `t` by value: a tensor with one dimension of `i32`, `i64`, `u32` or
`u64`. It sorts a copy of the values and compares each value with the next, so it costs one sort
on the CPU. When two values are equal, it returns `None` and drops `t`. It cannot read a tensor in
GPU memory, so check the tensor first, and then `transfer` the `Distinct` tensor. A `transfer` of
a `Distinct` tensor gives a `Distinct` tensor in the new memory.

When the program made the values itself and knows that no two are equal, `unsafe { Distinct(t) }`
makes a `Distinct` tensor with no check. It takes a tensor of integers with one dimension, and it
moves `t`, so no other name can change the values. If two elements are equal, two passes write the
same element. That is a data race, and Vx makes no promise about the result. This is why
`Distinct(t)` needs `unsafe`.

A function of the library can make the claim for its caller. `nn_moe_group` in `std::nn` groups
the token-expert pairs of a mixture-of-experts layer by expert, on the GPU. It returns the grouped
order as a `Distinct` tensor. Its `unsafe { Distinct(order) }` is inside the library, so the caller
needs no check and no `unsafe`.

The compiler adds one fact, that no two elements are equal, and checks everything else as before:

- A `Distinct` tensor is read only. A write to one of its elements is E4006, because a write could
  make two elements equal. `rows.len()` gives its length.
- Only a parameter of the function counts. A `Distinct` local or field, or a parameter whose name
  the function binds again, does not fix `b`. So write the kernel in a function that takes the
  tensor as a parameter, as `add_to_rows` does.
- `rows[e]` fixes what `e` fixes, so `rows[b / 2]` does not fix `b`. A read of `y[b]`, or of
  `y[rows[b + 1]]`, in the same loop is still E6032, because another pass may write that row.
- The other rules for an index still apply. `(rows[b] as i64) * 4 + (l as i64)`, with `l` below 4,
  fixes `b`, as `b * 4 + l` does. With `i64` elements, `rows[b] * 4` can wrap, so two different
  elements can give the same index, and the write is E6032.

A `Distinct` tensor does not help when two passes add to the same row. In a mixture-of-experts
layer, a token goes to several experts, and their results add into the row of the token. Use
[atomics](#atomics-on-the-elements-of-a-tensor) for that, or a separate pass.

## Rows from a table of ranges

A block can also write a range of rows that a table gives it. The grouped matrix product of a
mixture-of-experts layer is an example: a first kernel cuts the rows of each expert into tiles,
and block `t` of a second kernel writes the rows of tile `t`. Distinct starts do not keep two
blocks apart, because two ranges can overlap.

When no two ranges overlap, say so in the type of the table:
`DisjointRanges<Tensor<i32, [?, 2], Memory::HBM>>`. Row `t` holds range `t`: the start in
`tiles[t][0]` and the end, one past the last row, in `tiles[t][1]`. Then a row `tiles[t][0] + r`
fixes `t`, inside an `if` that keeps it below `tiles[t][1]`
(`blocks_write_the_rows_of_disjoint_ranges.vx`):

```rust
{{#include ../../../tests/optimizations/pass/blocks_write_the_rows_of_disjoint_ranges.vx:program}}
```

The rules:

- The index is `tiles[t][0]` plus a value that is never below zero, such as a lane index.
- The `if` tests that index against `tiles[t][1]`, with the same `t`: `row < end`, `end > row`,
  `row <= end - 1`, also as one side of `&&`. Only the writes in its `then` block count.
- The index can be part of a larger one, as in `(start + r) * 2 + c` with `c` below 2.
- The sum must not wrap. With `i32` ranges, cast the start and the lane index to `i64`, as above.
  An `i32` sum `tiles[t][0] + r` is E6032.
- A table is read only. A write to it is E4006, because a write could make two ranges overlap.
  `tiles.len()` gives the number of ranges.
- As for `Distinct`, only a parameter of the function counts.

`DisjointRanges::check(t)` makes a table, as `Distinct::check` makes a `Distinct` tensor. It takes
`t` by value: a tensor of shape `[?, 2]` of `i32`, `i64`, `u32` or `u64`, on the CPU. It returns
`None` when two ranges overlap or a range ends before it starts. An empty range `[s, s)` overlaps a
range that has `s` inside it, but not a range that starts or ends at `s`. The check sorts a copy of
the ranges, so it costs one sort on the CPU.

`unsafe { DisjointRanges(t) }` makes a table with no check. It takes a tensor of integers of shape
`[?, 2]`, and it moves `t`. If two ranges overlap, two blocks write the same row. That is a data
race, and Vx makes no promise about the result.

## A tile in shared memory

`Memory::SMEM` is the fast memory of a block. A tensor in it, a tile, lives as long as the code of
the block that makes it. The code of a block is the code in the block loop and outside its lane
loops. The end of each lane loop waits for all the lanes of the block. So the next lane loop
can read what any lane wrote. Code between two lane loops runs once for each block.

`lanes::sum(&part, n)` adds the first `n` elements of a tile, with all the lanes of the block. This
program adds each row of a matrix in a block of its own (`book_gpu_row_sums_in_shared_memory.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_row_sums_in_shared_memory.vx:program}}
```

It prints `row 0: 256, row 7: 2048`.

Here each lane reads only the element that it wrote, so the compiler keeps `part` in registers.
This also works through several lane loops. When every access to a tile is `t[..][l]`, or
`t[w][..][l]` in subgroup `w`, with `l` the index of a lane loop over all the lanes, the compiler
keeps each lane's elements of `t` in its registers.

The library has other routines of this kind. `lanes::max` gives the largest element, and
`lanes::argmax` its index. Write each call as the whole value of a `let`, in the code of a block.
`lanes::sum_unordered` lets the compiler choose the order of the adds, so the last bits of the sum
can differ from `lanes::sum`. `lanes::sort(&mut t, n)` sorts the first `n` elements of a tile of
`i64`, whose size is a power of two.

## Atomics on the elements of a tensor

Many passes can add to the same element with an atomic operation from `core::sync::atomic`. Each
lane here adds 1 to the bin of its value (`book_gpu_histogram_with_atomics.vx`):

```rust
{{#include ../../../tests/optimizations/pass/book_gpu_histogram_with_atomics.vx:program}}
```

It prints `bin 0: 128, bin 7: 128`.

The atomic operations are `atomic_load`, `atomic_store`, `atomic_swap`, `atomic_fetch_add`,
`atomic_fetch_sub`, `atomic_fetch_max`, `atomic_fetch_min` and `atomic_compare_and_swap`. Each name
ends with the type of the element, such as `atomic_fetch_add_u32`. Each one also takes a memory
order, such as `AtomicOrdering::Relaxed`, as the atomics of [Tasks and channels](tasks.md) do.

When a loop changes a tensor with atomics, each access to that tensor in the loop must be atomic:

```text
Error[E6032]: the passes of this block loop change `bins` with atomic operations, so every access
to it in the loop must be one
```

`scoped_fence(order, FenceScope::Block)` orders the memory operations of the lanes of one block.
`FenceScope::Device` orders them for the whole GPU, and `FenceScope::System` also for the CPU.

On an Apple GPU, the atomics work on `i32` and `u32` with `Relaxed` only.

## Copies in the background

`lanes::copy_async(&mut tile, &x, [i, j])` starts a copy of a part of `x` into `tile`, and gives a
token. The part starts at the element `[i, j]` of `x`, and has the shape of `tile`. `.wait()` on
the token waits until the copy ends.
Between the two, the lanes can work on another tile. So with two tiles, the next copy runs while
the lanes use the last one:

```rust
let mut tiles = unsafe { Tensor<f32, [2, 64, 64], Memory::SMEM>::uninit() };
for k in 0..steps / 2 {
  let even = lanes::copy_async(&mut tiles[0], &x, [0, k * 128]);
  let odd = lanes::copy_async(&mut tiles[1], &x, [0, k * 128 + 64]);
  even.wait();
  // The lanes read tiles[0] here, while the copy into tiles[1] goes on.
  odd.wait();
  // The lanes read tiles[1] here.
}
```

The token holds the borrow of the tile. So a read of the tile before its `.wait()` is a borrow
error, and a token with no `.wait()` gets the error E6036. On an H100 the copy uses TMA, and on an
older NVIDIA GPU `cp.async`. Today the copy does not work on tiles of `f64` or of integers.
`tests/optimizations/pass/lane_copies_on_the_cpu.vx` is a complete program.

`lanes::copy_rows_async(&mut tile, &x, &ids, [first, col])` copies rows that a list of ids
chooses, such as the cached rows of the tokens that an attention step reads. Row `r` of `tile` gets
row `ids[first + r]` of `x`, from column `col`. A row whose id is below 0 or not a row of `x`, or
past the end of `ids`, is zeros. `tile` and `x` have 2 dimensions, and `ids` is a tensor of `i32`
with 1 dimension. The token works as the one of `lanes::copy_async`, and the copy uses `cp.async`.
`tests/optimizations/pass/lane_copies_of_rows_by_id.vx` is a complete program.

`lanes::copy(&mut dst, &src, [i, j])` copies a tile and ends before it returns. When `dst` is a
tile in shared memory, the copy goes into it, as `lanes::copy_async` and `.wait()` do. Otherwise
`src` is the tile, and the copy goes into the part of `dst` that starts at `[i, j]`. Past the end
of `dst`, it writes nothing. So the subgroups of a block can store their tiles of a result:

```rust
for w in 0..4 across subgroups {
  lanes::copy(c, &acc[w], [bm * 128 + w * 32, bn * 128]);
}
```

Two blocks or two subgroups must not write the same element of `c` (E6032).
`tests/optimizations/pass/lane_copy_into_and_out_of_a_tile.vx` is a complete program.

`lanes::fill(&mut tile, v)` sets each element of a tile in shared memory to `v`. Every lane of the
block or the subgroup sets a part of the tile, so it is faster than a loop on one lane.

`lanes::scale_rows(&mut tile, &scale)` multiplies row `i` of a tile in shared memory by `scale[i]`.
Flash attention does this to its output sums between two matmuls. The sums of a tensor-core
matmul can then stay in registers.

## Math functions in a kernel

`x.exp()` and `x.tanh()` are exact, and on a GPU they compute in `f64`, which is slow. In a kernel
where a small difference in the last bits does not matter, `x.fast_exp()` and `x.fast_tanh()` use
a few `f32` instructions. [Exact and fast math](exact-and-fast-math.md) explains when to use which.

## What runs where

- An NVIDIA GPU runs each feature of this chapter. The compiler generates PTX.
- An Apple GPU runs block and lane loops, shared memory, and atomics on `i32` and `u32`.
- Each kernel also has a copy for the CPU, which runs when there is no GPU. The threads of the host
  share the blocks. `VX_HOST_THREADS=n` sets their number.
