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

Because the stride is positive, when arithmetic does not overflow, the maximum condition strictly implies all earlier conditions:

$$(c + \\text{offset}\_{N-1} < \\text{bound}) \\implies (c + \\text{offset}\_k < \\text{bound}) \\quad \\forall k < N-1$$

In 32-bit signed arithmetic, $c + \\text{offset}_{N-1}$ can wrap around to negative numbers if $c + \\text{offset}_{N-1} > \\text{INT32_MAX}$.
To prevent integer wrap-around from triggering the fast path on out-of-bounds indices, the optimization emits a combined check:

$$\\text{versionCond} = (c + \\text{offset}_{N-1} \\ge c) \\land (c + \\text{offset}_{N-1} < \\text{bound})$$

We transform the candidate sequence into two paths:

1. Fast Path (Batch in Range):
   - Guarded by the combined non-wrapping in-bound condition.
   - All $N$ load steps execute unconditionally without predicates or branches.
   - All loads are issued back-to-back into the memory pipeline.
   - Zero predicate registers are used for load guarding.
1. Slow Path (Fallback / Remainder):
   - Taken when either integer overflow occurs or $c + \\text{offset}\_{N-1} \\ge \\text{bound}$.
   - Runs the original individually guarded steps to handle edge elements safely.

## 3. Location in the Compiler Pipeline

The transformation lives in `src/dialect/VxLowering.cpp`:

- Inside `ConvertVxToStandardPass::materializeGpuKernels` via helper `versionGuardedLoadBatches(gpu::GPUFuncOp func)`.
- Restricted to Hopper targets (`sm_90+`) on the NVPTX pipeline (`pipeline == DevicePipeline::Nvptx && chipHasHopperPredicatePressure(...)`), avoiding non-Hopper chips (such as Ampere `sm_80`) and non-NVPTX pipelines (such as Metal).
- This runs directly on `gpu.func` operations in the GPU module before lowerings to LLVM and NVVM.
- A compiler flag `-fguarded-load-versioning` / `-fno-guarded-load-versioning` in `src/driver.rs` allows toggling the optimization.

## 4. Safety Invariants

The optimization must verify all of the following rules before transforming a candidate sequence:

1. Straight-Line Diamond Control Flow:
   - Each guarded step is a control-flow diamond: a header block, a `then` block containing memory loads, an optional `else` block containing passthrough operations, and a join destination block.
   - The comparison operation (`arith::CmpIOp`) must reside in the header block immediately preceding the conditional branch (`cf::CondBranchOp`). If instructions or intermediate blocks exist between the comparison and branch, the diamond is rejected.
1. Common Upper Bound and Equivalent Base:
   - All guards in the sequence must compare against the exact same bound SSA value using signed less-than (`arith::CmpIPredicate::slt`). Unsigned comparisons are rejected.
   - The base SSA values must be equivalent. The pass accepts identical SSA values, base values loaded from identical indices of the same stack alloca (`memref.alloca`) without intervening stores, and non-constant `arith::AddIOp` bases with offset 0.
1. Strictly Monotonic Non-Negative Offsets:
   - Offsets must be compile-time constants satisfying $0 \\le \\text{offset}\_0 < \\text{offset}_1 < \\dots < \\text{offset}_{N-1}$.
   - Negative, decreasing, or non-monotonic offsets invalidate the implication and are rejected.
1. Non-Wrapping Arithmetic Guard:
   - The fast path is reached only when $c + \\text{offset}_{N-1} \\ge c$ and $c + \\text{offset}_{N-1} < \\text{bound}$ both hold.
   - This prevents out-of-bounds access when large values of $c$ wrap around to negative numbers in signed 32-bit arithmetic.
1. No Intervening Mutating Stores:
   - No store operation to the base alloca or to external non-alloca memory may occur within any block of the candidate batch.
1. SSA Dominance and No Escaping Values:
   - Every operation defined inside the batch blocks must have all its uses within the batch blocks. If any value escapes to blocks outside the batch (such as intermediate sum reductions), the batch is rejected to avoid SSA dominance errors.
1. Minimum Batch Threshold:
   - The transform applies when a batch has at least 8 steps (where Hopper predicate exhaustion occurs due to the 7 hardware predicate registers $P_0..P_6$).

## 5. Verification Suite

1. Positive LIT Test (`tests/optimizations/pass/guarded_load_batch_versioning.vx`):
   - Multi-threaded kernel matching the structure of Issue #3403 with `for g in 0..1 across blocks` and `for lane in 0..32 across lanes(32)`.
   - Verified with FileCheck on `fleet/h100-sxm.vx`:
     - Combined non-wrapping in-bound condition check.
     - 16 back-to-back loads with `CHECK-NOT: setp` between all consecutive loads.
     - `-fno-guarded-load-versioning` retains all 8 individual `setp.ge.s32` checks.
1. Negative LIT Test (`tests/optimizations/pass/guarded_load_batch_versioning_negative.vx`):
   - `neg_decreasing_offsets`: decreasing guard offsets (from `c + 224` down to `c`) are rejected by monotonicity checks and remain unversioned.
   - `neg_bound`: differing bounds (`cols` vs `cols2`) reject the batch and remain unversioned.
   - `neg_store`: intervening memory store across loads rejects the batch and remains unversioned.
   - `neg_u32_compare`: unsigned comparisons (`ult`) are rejected and remain unversioned.
   - `neg_partial_sum`: intermediate computations used outside the batch are rejected to preserve SSA dominance and remain unversioned.
   - `neg_scale_between`: instructions between comparison and branch reject diamond structure and remain unversioned.
   - `neg_if_between`: control flow between comparison and branch rejects diamond structure and remain unversioned.
   - `kernel_a100`: targeting Ampere `sm_80` (`--machine fleet/a100-80.vx`) leaves the kernel unversioned.
