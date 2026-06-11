# Attention Rewrite Tasks

- `[x]` Implement native standard library math methods in Vx (e.g. `f32.sqrt()`, `f32.exp()`).
- `[x]` Cleanup `examples/llama.vx` to use these native math functions instead of unsafe FFI calls.
- `[x]` Fix hardcoded dimension assumptions in `examples/llama.vx` by allowing proper dynamic parsing via parameter injection.
- `[x]` Rewrite attention loop in `examples/llama.vx` to implement Grouped Query Attention (GQA) and FlashAttention natively.
- `[x]` Clean up tests and benchmarks inside `tests/backend/pass/` to remove `unsafe` math function invocations.
- `[x]` Ensure successful end-to-end compilation and execution via MLIR/LLI.
- `[x]` Create walkthrough artifact summarizing the rewrite.
