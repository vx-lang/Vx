# Rewriting `llama2.c` in Vx

The ultimate demonstration of the Vx programming language is rewriting Andrej Karpathy's `llama2.c` natively in `.vx`. This will prove Vx's viability as a systems language capable of high-performance ML inference, bare-metal memory management, and hardware dispatch.

However, to support `llama2.c`, the Vx compiler first requires significant infrastructure upgrades. `llama2.c` relies heavily on raw pointers, structs, file I/O, and math intrinsics.

## User Review Required

> [!WARNING]
> Porting `llama2.c` is a massive undertaking. The Vx compiler currently lacks support for pointer arithmetic, struct field access in MLIR codegen, String literals, and File I/O.
>
> I propose we break this down into milestones. We should start with **Milestone 1: Compiler Infrastructure**. Do you agree with this phased approach, or is there a specific kernel (like `matmul`) you want to hack together first?

## Proposed Roadmap

### Milestone 1: Compiler Core Upgrades (Current Focus)
Before writing the `llama2.vx` application, we must upgrade the `vxc` compiler:
1. **Scalar Arithmetic**: Fix the type system and MLIR lowering so `f32` and `i32` can be correctly operated on natively (without falling back to `memref` tensors).
2. **Struct Code Generation**: Fully implement MLIR codegen for struct instantiation and field accesses (e.g., `config.dim`, `state.x`).
3. **Pointers & Memory Allocation**: Implement `malloc`, `free`, and native `*mut f32` array indexing in MLIR.
4. **Math Intrinsics**: Expose `sqrtf` and `expf` via `extern` or the MLIR `math` dialect.
5. **Strings & File I/O**: Add basic support for C-style strings (`*const i8`) and FFI for `fopen`/`fread`.

### Milestone 2: Mathematical Kernels
Once the compiler supports basic C-like systems programming, we will implement the core LLaMA operations natively in `llama2.vx`:
- `rmsnorm(o: *mut f32, x: *mut f32, weight: *mut f32, size: i32)`
- `softmax(x: *mut f32, size: i32)`
- `matmul(xout: *mut f32, x: *mut f32, w: *mut f32, n: i32, d: i32)`

### Milestone 3: Hardware Dispatch Integration
Instead of running `matmul` entirely on the CPU, we will wrap the `matmul` calls in Vx's `spawn on(Topology::NPU)` block to automatically offload token generation to the Apple Silicon AMX hardware dispatcher we just integrated in v2.0!

### Milestone 4: End-to-End Inference Loop
Finally, we will implement the tokenizer, `.bin` file loading, and the autoregressive token generation loop to produce text.

## Verification Plan

1. **Milestone 1**: Write individual `.vx` integration tests in `tests/backend/pass/` for struct accesses, pointers, and scalar math.
2. **Milestone 2**: Write unit tests comparing our native Vx kernels against `llama2.c` baseline outputs for numerical precision.
3. **Milestone 4**: Execute the full `llama2.vx` binary on a small model (e.g., `stories15M.bin`) and verify coherent text generation.
