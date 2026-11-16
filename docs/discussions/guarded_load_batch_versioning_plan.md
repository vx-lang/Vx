# Plan: Guarded Load Batch Versioning on NVIDIA Hopper (Issue #3403)

## 1. Context and Problem

On NVIDIA Hopper (`sm_90`), SASS provides 7 predicate registers (`P0` through `P6`).
When a kernel issues eight guarded memory access steps in a loop:

- Each step $k \\in {0..7}$ tests a condition of the form `c + k * 32 < cols`.
- In PTX, each condition generates a `setp` instruction that writes to a predicate register.
- To keep all 16 loads in flight simultaneously, `ptxas` needs 8 live predicate registers.
- Because only 7 predicate registers exist in hardware, `ptxas` runs out of predicate registers at step 7.
- To free a register, `ptxas` must execute instructions that consume an earlier predicate (such as `FMUL` and `FADD` from step 0).
- These math instructions depend on the result of the load from step 0.
- The GPU core stalls while waiting for the memory round trip of load 0 before it can issue the loads of step 7.
- This serializes the batch into two memory round trips instead of one.

## 2. Proposed Solution

When $N$ guarded memory access steps test monotonic affine indices against a common upper bound:
$$\\text{guard}\_k: \\quad c + \\text{offset}\_k < \\text{bound}, \\quad \\text{where } 0 \\le \\text{offset}\_0 < \\text{offset}_1 < \\dots < \\text{offset}_{N-1}$$

Because the stride is positive, the last condition strictly implies all earlier conditions:
$$(c + \\text{offset}\_{N-1} < \\text{bound}) \\implies (c + \\text{offset}\_k < \\text{bound}) \\quad \\forall k < N-1$$

We transform the candidate sequence into two paths:

1. Fast Path (Batch in Range):
   - Guarded by the single condition `c + offset_{N-1} < bound`.
   - All $N$ load steps execute unconditionally without predicates or branches.
   - All loads are issued back-to-back into the memory pipeline.
   - Zero predicate registers are used for load guarding.
1. Slow Path (Fallback / Remainder):
   - Taken when `c + offset_{N-1} < bound` is false.
   - Runs the original individually guarded steps to handle edge elements safely.

## 3. Location in the Compiler Pipeline

The transformation lives in `src/dialect/VxLowering.cpp`:

- Inside `ConvertVxToStandardPass::materializeGpuKernels` via helper `versionGuardedLoadBatches(gpu::GPUFuncOp func)`.
- Restricted to Hopper targets (`sm_90+`) on the NVPTX pipeline (`pipeline == DevicePipeline::Nvptx && chipHasHopperPredicatePressure(...)`), avoiding non-Hopper chips and other GPU pipelines (such as Metal).
- This runs directly on `gpu.func` operations in the GPU module before lowerings to LLVM and NVVM.
- A compiler flag `-fguarded-load-versioning` / `-fno-guarded-load-versioning` in `src/driver.rs` allows toggling the optimization.

## 4. Safety Invariants

The optimization must verify all of the following rules before transforming a candidate sequence:

1. Common Upper Bound and Base Expression:
   - All guards in the sequence must compare against the exact same bound SSA value.
   - The base address expression must be identical across steps.
   - If guards compare against different bounds (e.g. `cols` vs `cols2` in `neg_bound`), the sequence must be rejected.
1. Strictly Monotonic Positive Offsets:
   - Offsets must be compile-time constants or statically provable positive increments.
   - Negative, decreasing, or non-monotonic offsets (e.g. `7 - k < 4` in `neg_decreasing`) invalidate the implication and must be rejected.
1. No Intervening Aliasing Writes:
   - No memory store that may alias the loads can occur between steps in the batch (as in `neg_store`).
   - If an intervening write exists, the batch must not be hoisted.
1. Memory Safety:
   - The fast path is reached only when the maximum index is proven in bounds.
   - Therefore, unconditional loads in the fast path cannot trigger out-of-bounds page faults.
1. Minimum Batch Threshold:
   - The transform applies when a batch has at least 8 steps (where Hopper predicate exhaustion occurs due to the 7 hardware predicate registers $P_0..P_6$).

## 5. Verification Plan

1. Positive LIT Test:
   - Run `vxc tests/optimizations/pass/guarded_load_batch_versioning.vx --host default --machine fleet/h100-sxm.vx --emit-ptx`.
   - Check with FileCheck:
     - Hoisted `setp` on the maximum offset (`c + 224 < cols`).
     - 16 unguarded `ld.global.nc.b32` loads with no `setp` between them (`CHECK-NOT: setp`).
     - `-fno-guarded-load-versioning` retains individual `setp` checks.
1. Negative LIT Tests:
   - Run `vxc tests/optimizations/pass/guarded_load_batch_versioning_negative.vx --host default --machine fleet/h100-sxm.vx --emit-ptx`.
   - Check with FileCheck:
     - `neg_decreasing`: decreasing guard (`7 - k < 4`) is not versioned and stores `4.0` (`1082130432`).
     - `neg_bound`: no versioning when bounds differ (`cols` vs `cols2`).
     - `neg_store`: no versioning across intervening tensor stores.
