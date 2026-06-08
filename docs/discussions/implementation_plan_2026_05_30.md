# Introduce MLIR Optimization Pipeline into Compiler Driver

This plan proposes an architectural enhancement to `vxc` by removing the disjoint MLIR pass executions and replacing them with a fully native, customizable MLIR Pass Pipeline exposed directly to the user via `-O` flags.

## Goal Description

The current compiler driver operates across multiple loosely coupled boundaries for MLIR generation and lowering:

1. It relies on `--disable-optimizations` instead of standard compiler optimization flags.
1. It handles Vx-to-Standard lowering and Vx-to-LLVM lowering through C++ hooks (`addVxLoweringPass`), but defers final LLVM dialect conversions to an external process (`mlir-opt` shell invocation in `src/jit.rs`).
1. It entirely skips standard MLIR optimizations (like `canonicalize`, `cse`, and `affine-loop-tile`) during standard `JIT` execution unless explicitly piped via external scripts.

We will replace this by registering the custom Vx passes globally in MLIR and utilizing `melior::utility::parse_pass_pipeline` natively from Rust. We will also expose standard `-O0` through `-O3` flags.

## User Review Required

> [!IMPORTANT]
> The default execution behavior for `cargo run --bin vxc` will change to apply full `-O3` equivalent optimizations unless `-O0` is passed. This aligns with your request for maximum optimizations under `-O`.

## Completed: Native MLIR Pass Pipeline Wiring

The native MLIR optimization pipeline via `melior::utility::register_all_passes` has been fully wired. We've fixed the `opt_level` integration so `-O0`, `-O1`, and `-O3` correctly construct the `func.func(..)` wrapping for function passes and register all upstream LLVM/SCF/Affine passes natively. All 12 compile_test targets succeed natively without shelling out to `vx-opt`.

## Next Step: Fusion Overhead Benchmarks

We will create a new test `tests/benchmarks/fusion_matmul_bias_relu.vx` (or similar) to measure the performance delta between fused and unfused loops.

### Proposed Changes

#### [NEW] `tests/benchmarks/cpu_fusion.vx`

This will compute `ReLU(Matmul(A, B) + Bias)`.

1. **Unfused variant**: Separate `scf.for` nest for matmul, a second for bias, and a third for ReLU.
1. **Fused variant**: A single combined `scf.for` nest that computes the dot-product, adds the bias, applies the ReLU, and stores the final result.
1. We will wrap these in `vx_get_time()` to measure execution latency.

#### [NEW] `tests/benchmarks/npu_fusion.vx`

Same logic, but leveraging `spawn on(Topology::NPU[0])` or equivalent NPU lowering paths to measure how the NPU backend handles loop fusion compared to separated dispatch.

## Open Questions

> [!IMPORTANT]
> How large should the matrices be for the benchmarking? e.g. 128x128, 512x512? Should we emit performance numbers via standard output (like `print_f32`)?

### 1. Dialect & Pass Registration (C++)

#### [MODIFY] `src/dialect/VxLowering.cpp`

- Expose a new C linkage hook `extern "C" void registerVxPassesC()` that safely invokes `mlir::vx::registerVxPasses()`. This makes our custom `convert-vx-to-standard` and `vx-to-llvm` passes discoverable by MLIR's generic string pass pipeline parser.

### 2. Driver Options & Pass Construction (Rust)

#### [MODIFY] `src/driver.rs`

- Remove `disable_optimizations: bool` from `DriverOptions`.
- Add `#[arg(short = 'O', num_args = 0..=1, default_missing_value = "3", default_value_t = 0)] pub opt_level: u8` to enable `-O0`, `-O1`, `-O2`, `-O3`, or just `-O`.
- Create a `get_optimization_pipeline(opt_level: u8, llvm_lower: bool) -> String` function that returns the pass pipeline.
  - `-O0`: `convert-vx-to-standard`
  - `-O>0`: Includes `canonicalize`, `cse`, `affine-loop-fusion`, `affine-loop-tile`, `affine-loop-unroll`, `affine-scalar-replacement`, and `lower-affine`.
  - If `llvm_lower = true`, it automatically appends `vx-to-llvm`, `convert-scf-to-cf`, and `finalize-memref-to-llvm` to bring the module all the way to LLVM IR ready status.
- Update `Action::RunJit`, `Action::EmitMlir`, and `Action::EmitLlvm` to parse the constructed string using `melior::utility::parse_pass_pipeline` and execute the passes *in-process*, producing a fully lowered MLIR module.

#### [MODIFY] `src/codegen/mod.rs`

- Add `fn registerVxPassesC();` to the `extern "C"` block.
- Add a safe wrapper `pub fn register_vx_passes()` that executes this C function.
- We will no longer need `lower_to_llvm` because the native pass pipeline string directly lowers the `melior::Module` seamlessly. We will safely deprecate or refactor it.

### 3. Cleanup JIT Execution (Rust)

#### [MODIFY] `src/jit.rs`

- Remove the external `mlir-opt` process invocation (`Command::new("...mlir-opt")`) entirely.
- The `execute_mlir` function will now expect the provided `mlir_src` to *already* be in the LLVM dialect. It will pass the MLIR straight to `mlir-translate` and `lli`, massively simplifying the function.

## Verification Plan

### Automated Tests

- `cargo test --test compile_test` will run across all our file-check assertions. Because `FileCheck` explicitly sets `-O0` in its `RUN` lines, tests will assert the unoptimized frontend structures correctly, while the `JIT` runs in tests will fully exercise the `-O3` path natively.

### Manual Verification

- We will re-run the `cpu_fusion_overhead.vx` and `npu_fusion_overhead.vx` manually using `cargo run --bin vxc` to observe how MLIR's `-O` affects the CPU and NPU latency compared to the previous run.
