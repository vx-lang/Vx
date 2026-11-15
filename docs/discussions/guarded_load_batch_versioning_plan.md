# Plan: Guarded Load Batch Versioning on NVIDIA Hopper (Issue #3403)

## 1. Context and Problem

On NVIDIA Hopper (`sm_90`), SASS provides 7 predicate registers (`P0` through `P6`).
When a kernel issues eight guarded memory access steps in a loop:
- Each step $k \in \{0..7\}$ tests a condition of the form `c + k * 32 < cols`.
- In PTX, each condition generates a `setp` instruction that writes to a predicate register.
- To keep all 16 loads in flight simultaneously, `ptxas` needs 8 live predicate registers.
- Because only 7 predicate registers exist in hardware, `ptxas` runs out of predicate registers at step 7.
- To free a register, `ptxas` must execute instructions that consume an earlier predicate (such as `FMUL` and `FADD` from step 0).
- These math instructions depend on the result of the load from step 0.
- The GPU core stalls while waiting for the memory round trip of load 0 before it can issue the loads of step 7.
- This serializes the batch into two memory round trips instead of one.

## 2. Proposed Solution

When $N$ guarded memory access steps test monotonic affine indices against a common upper bound:
$$\text{guard}_k: \quad c + \text{offset}_k < \text{bound}, \quad \text{where } 0 \le \text{offset}_0 < \text{offset}_1 < \dots < \text{offset}_{N-1}$$

Because the stride is positive, the last condition strictly implies all earlier conditions:
$$(c + \text{offset}_{N-1} < \text{bound}) \implies (c + \text{offset}_k < \text{bound}) \quad \forall k < N-1$$

We transform the candidate sequence into two paths:
1. Fast Path (Batch in Range):
   - Guarded by the single condition `c + offset_{N-1} < bound`.
   - All $N$ load steps execute unconditionally without predicates or branches.
   - All loads are issued back-to-back into the memory pipeline.
   - Zero predicate registers are used for load guarding.
2. Slow Path (Fallback / Remainder):
   - Taken when `c + offset_{N-1} < bound` is false.
   - Runs the original individually guarded steps to handle edge elements safely.

## 3. Location in the Compiler Pipeline

The transformation lives in `src/dialect/VxLowering.cpp`:
- Inside `ConvertVxToStandardPass::materializeGpuKernels` via helper `versionGuardedLoadBatches(gpu::GPUFuncOp func)`.
- This runs directly on `gpu.func` operations in the GPU module before lowerings to LLVM and NVVM.
- A compiler flag `-fguarded-load-versioning` / `-fno-guarded-load-versioning` in `src/driver.rs` allows toggling the optimization.

## 4. Safety Invariants

The optimization must verify all of the following rules before transforming a candidate sequence:
1. Common Upper Bound and Base Expression:
   - All guards in the sequence must compare against the exact same bound SSA value.
   - The base address expression must be identical across steps.
   - If guards compare against different bounds (e.g. `cols` vs `cols2` in `neg_bound`), the sequence must be rejected.
2. Strictly Monotonic Positive Offsets:
   - Offsets must be compile-time constants or statically provable positive increments.
   - Negative or non-monotonic offsets invalidate the implication and must be rejected.
3. No Intervening Aliasing Writes:
   - No memory store that may alias the loads can occur between steps in the batch (as in `neg_store`).
   - If an intervening write exists, the batch must not be hoisted.
4. Memory Safety:
   - The fast path is reached only when the maximum index is proven in bounds.
   - Therefore, unconditional loads in the fast path cannot trigger out-of-bounds page faults.
5. Minimum Batch Threshold:
   - The transform applies when a batch has at least 4 steps (and especially $\ge 7$ steps where Hopper predicate exhaustion occurs).

## 5. Verification Plan

1. Positive LIT Test:
   - Run `vxc test_issue3403.vx --host default --emit-ptx`.
   - Check with FileCheck:
     - One single `setp` on the last step condition.
     - 16 unguarded `ld.global.nc.b32` loads with no `setp` between them.
2. Negative LIT Tests:
   - Verify `inspect_sass_neg.vx`:
     - `neg_bound`: no versioning when bounds differ.
     - `neg_store`: no versioning across intervening stores.
     - Confirm `diff on.ptx off.ptx` is empty.
3. Runtime and Memory Safety Test:
   - Run `inspect_sass.py` on Modal H100 with `compute-sanitizer --tool memcheck` across irregular column counts `[1, 31, 225, 255, 256, 257, 4099]`.
   - Verify all numerical outputs match the reference implementation within `1e-4`.

