# Refactor Raw FFI Bindings in Stdlib

Based on the `SKILLS-stdlib.md` workflow, I have investigated the `stdlib` directory and identified several areas where raw `extern "C"` bindings bypass the type system and pollute the code. We need to centralize these and build safe public wrappers.

## Proposed Changes

### Centralize Memory Allocation (`std::alloc`)

Multiple files in `stdlib/std` manually import `malloc`, `realloc`, and `free` from C. We should create a dedicated, centralized module for memory allocation.

#### [NEW] `stdlib/std/alloc.vx`

Create a native standard library module to serve as the unified source for raw memory operations. This will contain the `extern "C"` block for `malloc`, `realloc`, and `free`, and provide safe wrappers (or at least centralize the unsafe pointers).

```vx
extern "C" {
  fn malloc(size: i64) -> *mut i8;
  fn realloc(ptr: *mut i8, size: i64) -> *mut i8;
  fn free(ptr: *mut i8) -> i32;
}

// Since memory operations inherently return pointers, 
// we expose the raw functions, but limit the `extern "C"` usage to this single file.
fn vx_malloc(size: i64) -> *mut i8 { return unsafe { malloc(size) }; }
fn vx_realloc(ptr: *mut i8, size: i64) -> *mut i8 { return unsafe { realloc(ptr, size) }; }
fn vx_free(ptr: *mut i8) -> i32 { return unsafe { free(ptr) }; }
```

#### [MODIFY] `stdlib/std/box.vx`

- Remove `extern "C"` block for `malloc` and `free`.
- Add `import std::alloc;`.
- Update usages to `alloc::vx_malloc` and `alloc::vx_free`.

#### [MODIFY] `stdlib/std/option.vx`

- Remove `extern "C"` block for `malloc`.
- Add `import std::alloc;`.
- Update usage to `alloc::vx_malloc`.

#### [MODIFY] `stdlib/std/vec.vx`

- Remove `extern "C"` block for `malloc`, `realloc`, and `free`.
- Add `import std::alloc;`.
- Update usages to `alloc::vx_malloc`, `alloc::vx_realloc`, and `alloc::vx_free`.

______________________________________________________________________

### Add Safe Wrappers to Benchmark Utilities

The utilities under `stdlib/benchmarks/` have raw `extern "C"` without safe native wrappers. We need to encapsulate these.

#### [MODIFY] `stdlib/benchmarks/benchmark.vx`

- Add safe wrappers around `start_benchmark` and `end_benchmark`.

```vx
extern "C" {
  fn start_benchmark() -> Tensor<f32>;
  fn end_benchmark() -> Tensor<i32>;
}

fn start() -> Tensor<f32> { return unsafe { start_benchmark() }; }
fn end() -> Tensor<i32> { return unsafe { end_benchmark() }; }
```

#### [MODIFY] `stdlib/benchmarks/tracing.vx`

- Add safe wrappers around `trace_start` and `trace_end`.

```vx
extern "C" {
  fn trace_start() -> Tensor<i32>;
  fn trace_end() -> Tensor<i32>;
}

fn start() -> Tensor<i32> { return unsafe { trace_start() }; }
fn end() -> Tensor<i32> { return unsafe { trace_end() }; }
```

______________________________________________________________________

### Migrate Downstream User Code

The `malloc` duplication isn't just in the stdlib, but spans across user tests and benchmarks.
To follow step 6 ("Migrate User Code to the Safe API"), we will update the following to remove local FFI `malloc`/`free` and use `std::alloc`:

#### [MODIFY] `tests/middle_end/pass/ffi.vx`

#### [MODIFY] `tests/middle_end/pass/pointers.vx`

#### [MODIFY] `tests/backend/pass/pointer_arithmetic.vx`

#### [MODIFY] `tests/backend/pass/ffi_fs.vx`

#### [MODIFY] `tests/backend/pass/llama2_math.vx`

*(Note: We will only migrate those that are simple substitutions to ensure they continue to test the desired compiler behaviour. For example, `ffi.vx` explicitly tests FFI bindings, so we might want to keep its local `extern` or rename it).*

## User Review Required

> [!IMPORTANT]
> The skill mentions: "Write a strongly-typed, native public function (e.g. `pub fn now() -> f64`)". Since `malloc` inherently deals with raw pointers, does it make sense to expose `vx_malloc` as `fn vx_malloc(size: i64) -> *mut i8` from `std::alloc`? Alternatively, we could keep the `unsafe` FFI restricted internally and just centralize the `extern "C"` without public safe wrappers, since raw allocation is inherently unsafe. Let me know which approach you prefer for `alloc`.

## Verification Plan

1. Run all unit tests (`cargo test`).
1. Build and run all backend integration tests using `vxc` to ensure we didn't break memory allocation in `Vec`, `Box`, or `Option`.
1. Verify the `benchmarks/` continue to compile properly with the new `std::alloc` and benchmark wrappers.
