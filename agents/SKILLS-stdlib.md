## Skill: Refactoring Raw FFI Bindings into Native Standard Library Modules

When migrating raw C-bindings (like Math routines or OS I/O) into the native Vx standard library, follow this systematic workflow:

### 1. Identify and Centralize External Dependencies
Locate raw `extern "C"` definitions scattered across user scripts, tests, or benchmarks (e.g., `vx_get_time`, `math_exp`). These manual definitions pollute user code and bypass the type system.

### 2. Create the Native Stdlib Module
Establish a dedicated `.vx` file in the `stdlib/std/` directory (e.g., `time.vx`, `math.vx`) to serve as the unified, authoritative source for these operations.

### 3. Encapsulate the FFI Binding
Move the `extern "C"` declaration inside the new stdlib module. By centralizing the raw binding, you prevent users from having to guess the C-ABI signatures.

### 4. Build a Safe Public Wrapper
Write a strongly-typed, native public function (e.g., `pub fn now() -> f64`) that abstracts the underlying FFI call. 
- *Why:* This scopes any potential `unsafe` behavior entirely inside the standard library, guaranteeing that user-space code remains completely safe and verified.

### 5. Audit the Host Runtime Shims (`rt.rs`)
Check the underlying Rust/C runtime (`stdlib/rust_core/src/ffi/rt.rs`) to ensure it perfectly mirrors the expected `#[no_mangle] pub extern "C"` signature.
- *Pro-Tip (Math Libraries):* For standard mathematical operations (e.g., `exp`, `sin`, `cos`, `sqrt`), do not manually rebuild custom Rust shims. Instead, map the Vx compiler's FFI definitions directly to the host OS's native `libm` implementations for zero-overhead execution.

### 6. Migrate User Code to the Safe API
Update all downstream scripts, tests, and benchmarks to:
1. `import` the new native module (e.g., `import std::time;`).
2. Remove their local `extern` declarations.
3. Replace manual, unsafe FFI calls with the new safe abstractions (e.g., `time::now()`).

### 7. Purge Redundant Mocks
Aggressively delete any deprecated C-shims, unused print mocks (e.g., `vx_print`), and legacy FFI rust modules that are no longer necessary after mapping directly to the OS or the new safe abstractions.
