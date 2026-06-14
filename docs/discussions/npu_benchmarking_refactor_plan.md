# Refactor ANE Benchmarks to Test Individual Attention Components

Currently, the `ane_decode_bench.vx` file tests monolithic sequences of nested loops inside two giant functions (`prefill` and `decode`). Because Vx implements tensors as strict linear types, passing `Tensor<f32>` into smaller functions consumes them unless we drop down to raw `*mut f32` pointers.

Instead of passing pointers around, the most robust way to evaluate the individual components of attention natively on the NPU (without hitting LLVM dead-code-elimination snags) is to write standalone benchmark functions for each distinct computational boundary within the transformer layer.

## Proposed Changes

We will refactor `benchmarks/ane_decode_bench.vx` and introduce the following standalone benchmark functions based on the LLaMA2-15M shapes (dim=288, hidden_dim=768, head_size=48).

### 1. `benchmark_attention_proj()`

- Tests the $O(N^2)$ projections used in `W_Q`, `W_K`, `W_V`, and `W_O`.
- **Dimensions**: $[1, 288]$ multiplied by $[288, 288]$.

### 2. `benchmark_ffn_up_gate()`

- Tests the expansion projection inside the SwiGLU FFN (`W_1` and `W_3`).
- **Dimensions**: $[1, 288]$ multiplied by $[288, 768]$.

### 3. `benchmark_ffn_down()`

- Tests the contraction projection back to the embedding dimension (`W_2`).
- **Dimensions**: $[1, 768]$ multiplied by $[768, 288]$.

### 4. `benchmark_attention_scores()`

- Tests the dynamically shaped $Q \\cdot K^T$ and $Scores \\cdot V$ self-attention matrices.
- We will benchmark this against a simulated context length of $T=1024$.
- **Dimensions**: $[1, 1024]$ multiplied by $[1024, 48]$.

## Open Questions

> [!IMPORTANT]
> The Apple Neural Engine is predominantly a matrix-multiplication co-processor. Do you want me to also include raw element-wise benchmarks (like `RoPE`, `Softmax`, and `RMSNorm`) in this file, or should we strictly keep this benchmarking suite isolated to the $O(N^3)$ matmul bottlenecks that we are offloading via `spawn on(Topology::NPU[0])`?

### [MODIFY] \[ane_decode_bench.vx\](file:///Users/adityak/go/Vx/benchmarks/ane_decode_bench.vx)

- Delete the monolithic `benchmark_decode_step_decode` and `benchmark_decode_step_prefill` functions.
- Insert the 4 new component-specific benchmark functions.
- Update `main()` to invoke and print the elapsed times of all 4 sequentially.

## Verification Plan

### Automated Tests

- Run `cargo run --release --bin vxc -- benchmarks/ane_decode_bench.vx -O3 --run` to verify that all 4 components compile, dispatch, verify their numerical accuracy against the CPU fallback, and log their dispatch timings cleanly.
