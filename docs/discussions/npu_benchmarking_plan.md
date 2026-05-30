# CPU/NPU Fusion Benchmark Implementation Plan

We will add two new end-to-end performance tests to measure the overhead of kernel fusion.

## Proposed Changes

### `tests/backend/pass/cpu_fusion_overhead.vx`
Will benchmark CPU execution overhead:
1. `cpu_unfused`: Three separate loops calculating standard Matmul -> Bias Add -> ReLU into temporary intermediate `Tensor<f32>` allocations.
2. `cpu_fused`: A single outer loop calculating the Matmul dot product locally, instantly adding the bias, applying the ReLU scalar check, and writing only once to the destination `Tensor<f32>`.
3. The test will allocate size `128x128` matrices, invoke `vx_get_time()` before and after both versions (repeating them 10x for stability), print the elapsed times via `print_f32`, and assert that the outputs of the fused and unfused passes are mathematically identical.

### `tests/backend/pass/npu_fusion_overhead.vx`
Will benchmark NPU execution and memory overhead:
1. `npu_unfused`: Transfers data to `NPU_HBM`. Dispatches three separate `spawn on(Topology::NPU[0])` blocks (Matmul, Bias Add, ReLU) simulating separate kernel launches and massive DRAM trips.
2. `npu_fused`: Dispatches a single `spawn on(Topology::NPU[0])` kernel executing the fused inner loop simulating an optimized SRAM-bound operator.
3. Will measure elapsed time (using `vx_get_time()`) from host perspective to demonstrate how multi-kernel dispatching overhead scales.
4. Will run FileCheck (`// RUN: vxc ...`) expectations so we can assert on the unified `vx.spawn` mapping emitted.
