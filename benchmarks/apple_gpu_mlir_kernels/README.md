# llama.cpp's Metal kernels as `mlir!` blocks on the Apple GPU

Two kernels of the prompt phase of a mixture-of-experts model, written as one `mlir!` block with
`threads : 128` each. Every thread of a threadgroup runs the block, so the sums of the simdgroup
matrices stay in registers through a loop with barriers in it (`docs/spawn_on.md`). The designs
are those of llama.cpp's Metal kernels (MIT License, Copyright (c) 2023-2026 The ggml authors).
The vx-lang/qwen-vx session wrote them for its Qwen3-Coder-30B-A3B app.

| File | What it times |
|---|---|
| `expert_product_q4.vx` | The expert product with 4-bit weights (Q4_0): 4096 tokens in tiles of 32, 128 experts of 1536 x 2048. It times the same design as a Vx kernel and as an `mlir!` block, and prints the largest difference between the two outputs. |
| `flash_attention.vx` | Flash attention for a chunk of 512 tokens at positions 0, 1536 and 7680. It has 32 query heads, 4 heads of K and V, and 128 elements a head. It compares 16 sampled rows with attention computed on the CPU. |

## Running

On a Mac with Apple silicon, from the repository root:

```
cargo build --release
. ./config.local
target/release/vxc benchmarks/apple_gpu_mlir_kernels/expert_product_q4.vx --host default --machine fleet/m4-uma.vx
target/release/vxc benchmarks/apple_gpu_mlir_kernels/flash_attention.vx --host default --machine fleet/m4-uma.vx
```

Each time is the wall time of 10 calls, divided by 10. The test that compiles every benchmark
compiles these files too, but does not run them.

## Results

On an M4 MacBook Air (10 cores), `main` of 2026-11-16:

| Kernel | Vx kernel | `mlir!` block |
|---|---|---|
| Expert product | 11.8 ms | 9.8 ms, the same bits as the Vx kernel |
| Flash attention, chunk at 0 | | 1.26 ms |
| Flash attention, chunk at 1536 | | 7.6 ms |
| Flash attention, chunk at 7680 | | 33.6 ms |

The sampled rows of the flash attention were within 0.00014 of the CPU. Both round the query to
f16, and only the GPU rounds the probabilities to f16.

The qwen-vx session measured the same kernels in its harness:

- The expert product took 9.2 ms when written by hand in Metal, and llama.cpp's own kernel took 8.2
  to 8.5 ms.
- The flash attention was 2.5 to 2.7 times faster than qwen-vx's attention in Vx at the same
  three positions.
- In the whole app, a prompt of 8,000 tokens took 4.67 ms a token with both blocks, against 6.98 ms
  with Vx kernels and 4.23 ms for llama.cpp (vx-lang/qwen-vx#48).

#3557 asks for one more of llama.cpp's hints, a SIMD-group barrier with no memory order between
the loads and the multiplies. In a hand-written version of the expert product, it saved 0.13 ms
of 9.0.
