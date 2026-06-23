# Vx ABI and Calling Convention

This document defines the Application Binary Interface (ABI) policy for the
**Vx** compiler: how Vx code passes arguments and returns values at the machine
level, both for ordinary functions and for offloaded `spawn on(...)` kernels.

## 1. Policy: follow the platform C ABI

Vx does **not** define its own register-allocation or argument-passing rules.
Vx lowers to MLIR and then to LLVM IR, and defers all low-level calling
convention decisions to the target's standard C ABI — the same convention a C
compiler uses on that platform:

| Target triple | Standard C ABI |
| ---------------------- | --------------------------------------- |
| `x86_64-*-linux`, `*-darwin` | System V AMD64 psABI |
| `aarch64-apple-darwin` | AAPCS64 (Apple variant) |
| `aarch64-*-linux` | AAPCS64 |

These are different documents, but they agree on the rules that matter to Vx:

- **Integers and pointers** are passed in general-purpose registers (then spill
  to the stack once the GP registers are exhausted).
- **Floating-point scalars (`f32`/`f64`)** are passed in **floating-point /
  vector registers** (xmm on x86-64, v-registers on arm64), in their *own*
  register pool, independent of the GP registers.
- **Small aggregates** are passed in registers field-by-field; **large
  aggregates** are passed by hidden pointer, and large return values use an
  implicit `sret` out-parameter.

> System V is an umbrella spec: a generic ABI (gABI) plus per-architecture
> supplements (psABIs); the IA-64/"Itanium" *processor* psABI is itself a member
> of this family. The "Itanium ABI" used by Clang/GCC is a separate, C++-level
> concern (name mangling, vtables, exception tables) layered *on top of* the C
> ABI; it does not affect register placement and is not relevant here. Vx uses
> its own symbol mangling and does not adopt the Itanium C++ ABI.

**Rule of thumb:** if you are writing or debugging an ABI boundary in Vx, the
answer is "whatever `clang` does for the same C signature on this target." Do not
invent a Vx-specific convention.

## 2. `extern` / FFI functions

Functions declared in an `extern { ... }` block use the platform C ABI directly.
A Vx type maps to its natural C counterpart:

| Vx type | C ABI lowering |
| -------------- | --------------------------------------- |
| `i8..i64`, `u8..u64` | corresponding integer, GP register |
| `f32`, `f64` | float/double, **FP register** |
| `*const T`, `*mut T` | pointer, GP register |
| `Tensor`, memref-backed values | passed as an MLIR memref descriptor (see §3) |

Because Vx follows the platform convention, calling a C library function or
being called from C requires no shims as long as the declared signature matches.

## 3. `spawn on(...)` kernel dispatch ABI

`spawn on(Topology::...)` outlines its body into a kernel function. For CPU
topologies this becomes an `async.execute` region. For accelerator topologies
(NPU / ANE / GPU) the body is outlined into a function `vx_npu_kernel_N` carrying
the `llvm.emit_c_interface` attribute, so MLIR additionally emits a C-interface
wrapper `_mlir_ciface_vx_npu_kernel_N`. The captured values become the kernel's
parameters, and memref-backed captures are passed as pointers to their memref
descriptor structs (consistent with §2).

The runtime (`runtime/npu_dispatch.mm`) receives the captures as a `void**`
array (`device_args`) and must invoke the kernel.

### 3.1 Known limitation (current state)

The current dispatcher calls the kernel through a fixed signature:

```c
typedef void (*KernelFuncPtr)(void*, void*, void*, void*, void*, void*, void*, void*);
kernel(device_args[0], ..., device_args[7]);
```

This forces **every** argument into a general-purpose register. That violates
the policy in §1 for any kernel that has a by-value **float** parameter: per the
platform ABI an `f32`/`f64` must travel in an FP register, so it (a) reads
garbage and (b) — because it does not consume a GP register — shifts every
following pointer argument into the wrong register, producing a wild pointer and
a `SIGSEGV` inside the kernel.

- Kernels whose C-interface is entirely integer/pointer work
  (e.g. `tests/backend/pass/ane_matmul.vx`).
- Kernels with an `f32`/`f64` scalar capture crash
  (e.g. `tests/backend/pass/npu_fusion_overhead.vx`, `llama2.vx`).

An earlier change packed integer captures by value (via `inttoptr`) in
`LaunchOpLowering`; that fixed only the integer instance of this same class of
bug. The float case remains broken.

### 3.2 Target design

The dispatch boundary must honor the platform C ABI like every other call. Two
acceptable implementations, in order of preference:

1. **Packed `void**` ABI.** Have the compiler emit a per-kernel wrapper that
   takes the single `void** args` array, loads each argument from `args[i]`
   **with its real type** (so a float is loaded as a float and ends up in an FP
   register when the typed kernel is called), and calls the typed kernel. The
   runtime then simply calls `wrapper(device_args)` through `void (*)(void**)`.
   Because the compiler knows every type, the platform ABI is reconstructed
   correctly. This also lets us delete the integer `inttoptr` special case and
   the "allocate at least 8 slots" workaround in `LaunchOpLowering`.

1. **libffi.** The compiler emits a type-descriptor array alongside
   `device_args`; the runtime uses `ffi_prep_cif`/`ffi_call` to perform the call
   with correct per-argument types. Simpler compiler change, but adds a libffi
   dependency.

Until one of these lands, the MLIR verifier is left disabled in
`src/codegen/mod.rs` and the float-capturing NPU tests are expected to crash at
run time. See the `TODO(npu-abi)` note in `runtime/npu_dispatch.mm`.

## 4. Related

- `docs/npu_hardware_dispatch.md` — the `_mlir_ciface` dispatch mechanism.
- `docs/spawn_on.md` — `spawn on(...)` semantics and outlining.
- `src/dialect/VxLowering.cpp` — `SpawnOpLowering` / `LaunchOpLowering`.
- `runtime/npu_dispatch.mm` — the runtime dispatcher.
