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

The runtime (`runtime/npu_dispatch.mm`) receives the captures and invokes the
kernel through libffi so that the platform calling convention is honored.

### 3.1 The dispatch contract

`LaunchOpLowering` lowers a `vx.launch` to a call:

```c
vx_plugin_dispatch_async(name, payload_size, device_args, arg_tags, num_args);
```

- `device_args[i]` is a pointer to the value of the i-th C-interface argument:
  - scalar param → pointer to the scalar;
  - memref param → pointer to the (pointer-to-descriptor).
- `arg_tags[i]` is the argument's ABI type tag
  (`0=ptr, 1=i1, 2=i8, 3=i16, 4=i32, 5=i64, 6=f32, 7=f64`; see `abiTagForType`
  in `src/dialect/VxLowering.cpp` and `vx_abi_ffi_type` in
  `runtime/npu_dispatch.mm` — keep the two in sync).

The runtime builds an `ffi_cif` from the tags and `ffi_call`s
`_mlir_ciface_<name>`. libffi places each argument in the correct GP/FP register
or stack slot, so this is correct for **all** argument types, including by-value
`f32`/`f64`.

### 3.2 Why the naive dispatch was wrong

The previous dispatcher called the kernel through a fixed
`void (*)(void*, …)` signature, forcing every argument into a general-purpose
register. For a by-value `f32`/`f64` that violates the policy in §1: per the
platform ABI a float must travel in an FP register, so it (a) read garbage and
(b) — because it did not consume a GP register — shifted every following pointer
argument into the wrong register, producing a wild pointer and a `SIGSEGV`. An
even earlier version packed only integers by value (via `inttoptr`), which fixed
just the integer instance of the same problem. The libffi path supersedes both
and removes those workarounds (`inttoptr` packing, the "allocate at least 8
slots" hack). `tests/backend/pass/npu_float_scalar.vx` exercises a float-capturing
kernel end to end.

## 4. Related

- `docs/npu_hardware_dispatch.md` — the `_mlir_ciface` dispatch mechanism.
- `docs/spawn_on.md` — `spawn on(...)` semantics and outlining.
- `src/dialect/VxLowering.cpp` — `SpawnOpLowering` / `LaunchOpLowering`.
- `runtime/npu_dispatch.mm` — the runtime dispatcher.
