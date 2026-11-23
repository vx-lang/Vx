# GEMM on an NVIDIA GPU: Vx against cuBLAS

Eight versions of C = A @ B, with A and B in bf16 and the sums in f32, on the tensor cores. Each
version changes one thing from an earlier one, so the timings show what each change is worth.
`cublas_gemm.cu` computes the same product with cuBLAS, from the same inputs.

From v4 on, `lanes::fill` sets the sums to zero and `lanes::copy` stores them into C, so the sums
of each subgroup stay in registers. v1 to v3 store the sums into a C with 6 dimensions, which
`lanes::copy` cannot do. So their sums stay in shared memory, and the compiler warns (W1037).

| Version | What it changes |
|---|---|
| `v1_lanes_copy_the_tiles.vx` | A 64 x 64 tile of C for each block. The lanes copy A and B into shared memory one element at a time. |
| `v2_steps_of_64.vx` | As v1, with K going 64 at a time, not 32. |
| `v3_tile_128x64.vx` | As v1, with a 128 x 64 tile and 8 subgroups. |
| `v4_one_block_loop.vx` | As v1, with C as a plain matrix and one block loop over the tiles. The sums stay in registers. |
| `v5_copy_async.vx` | As v4, with `lanes::copy_async` for the tiles of A and B. |
| `v6_two_steps_a_pass.vx` | As v5, with two steps of K in each pass, so one copy runs while the subgroups use the other tile. |
| `v7_tile_128x128_4_subgroups.vx` | As v5, with a 128 x 128 tile and 4 subgroups of 32 x 128. |
| `v8_tile_128x128_8_subgroups.vx` | As v5, with a 128 x 128 tile and 8 subgroups of 64 x 32. |

## Running

On a machine with an NVIDIA GPU and CUDA, from the repository root:

```
cargo build --release
sh benchmarks/gpu_gemm/run.sh results
VERSIONS="v5 v8" sh benchmarks/gpu_gemm/run.sh results
```

The script times each version at 8192 x 3072 x 768 and at 4096 x 4096 x 4096. Each line gives the
median of 50 launches and how many of 64 sampled entries of C are wrong. The inputs are small
integers, so every sum is exact and a right answer matches bit for bit. The test that compiles
every benchmark compiles these files too, but does not run them.

## Results

TFLOP/s for `main` of 2026-11-15 on an A100 80GB PCIe and of 2026-11-14 on an RTX A6000, both with
CUDA 12.8. Every run was exact. These runs were before v4 to v8 kept the sums in registers, so
the numbers of v4 to v8 are for their sums in shared memory.

| | A100, 8192 x 3072 x 768 | A100, 4096 x 4096 x 4096 | A6000, 8192 x 3072 x 768 | A6000, 4096 x 4096 x 4096 |
|---|---|---|---|---|
| cuBLAS | 214.2 | 260.1 | 91.8 | 114.8 |
| v1 | 11.6 | 8.8 | 9.2 | 9.7 |
| v2 | 11.7 | 10.1 | 8.1 | 8.8 |
| v3 | 11.0 | 8.4 | 8.5 | 9.2 |
| v4 | 14.8 | 10.3 | 8.1 | 8.7 |
| v5 | 26.3 | 27.0 | 24.0 | 24.6 |
| v6 | 13.4 | 13.5 | 13.1 | 13.9 |
| v7 | 44.4 | 47.3 | 28.5 | 31.3 |
| v8 | 39.0 | 38.9 | | |

v8 was added after the A6000 runs. #3400 tracks the work on the gap to cuBLAS.
