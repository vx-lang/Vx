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

The same check refuses a read of an element that another pass writes, a write to a variable from
outside the loop, `println!`, and a call that can write through its argument.

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

## Math functions in a kernel

`x.exp()` and `x.tanh()` are exact, and on a GPU they compute in `f64`, which is slow. In a kernel
where a small difference in the last bits does not matter, `x.fast_exp()` and `x.fast_tanh()` use
a few `f32` instructions. [Exact and fast math](exact-and-fast-math.md) explains when to use which.

## What runs where

- An NVIDIA GPU runs each feature of this chapter. The compiler generates PTX.
- An Apple GPU runs block and lane loops, shared memory, and atomics on `i32` and `u32`.
- Each kernel also has a copy for the CPU, which runs when there is no GPU. The threads of the host
  share the blocks. `VX_HOST_THREADS=n` sets their number.
