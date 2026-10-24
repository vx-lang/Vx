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
| `*mut T in Memory::X` | the same pointer; the placement is erased at the C boundary |
| `Tensor`, memref-backed values | passed as an MLIR memref descriptor (see §3) |

Because Vx follows the platform convention, calling a C library function or
being called from C requires no shims as long as the declared signature matches.

A tensor cannot appear in an `extern` signature (E3022): its descriptor expands to
seven C arguments no C source writes. A C library that takes a device buffer takes
it as a plain pointer, and that is what a pointer into device memory becomes at the
call. `t.as_mut_ptr()` on a tensor placed in `Memory::GPU_HBM` has type
`*mut f32 in Memory::GPU_HBM`; passed to a parameter of that type it is the raw
device address, the same `float *` cuBLAS expects. The placement is checked by the
compiler and never reaches C: a host pointer passed where a device pointer is
declared is E6029, and a host read through the device pointer is E6003.

### 2.1 Opaque-pointer ownership lifecycle (the Rust core contract)

Vx's standard library is backed by a Rust crate (`stdlib/rust_core`, built as
`libvx_std_core.a`) exposing ~100 `extern "C"` entry points. Rust generics cannot
cross a C ABI, so every non-scalar type crosses as an **opaque `*mut c_void`**,
and ownership is transferred explicitly. This is a normative contract: every new
stdlib FFI binding must follow it, because the Vx side has no way to observe a
Rust `Drop` and the Rust side has no way to observe a Vx scope exit.

**Constructors** allocate on the Rust heap and release ownership to Vx:

```rust
#[no_mangle]
pub extern "C" fn vx_vec_new_i32() -> *mut c_void {
    Box::into_raw(Box::new(Vec::<i32>::new())) as *mut c_void
}
```

**Borrowing operations** reconstruct a reference and must *not* take ownership —
no `Box::from_raw`, or the buffer is freed while Vx still holds the pointer:

```rust
#[no_mangle]
pub extern "C" fn vx_vec_len_i32(ptr: *mut c_void) -> usize {
    if ptr.is_null() { return 0; }
    let v = unsafe { &*(ptr as *const Vec<i32>) };
    v.len()
}
```

**Destructors** reclaim ownership exactly once; Rust's `Drop` then runs:

```rust
#[no_mangle]
pub extern "C" fn vx_vec_drop_i32(ptr: *mut c_void) {
    if ptr.is_null() { return; }
    drop(unsafe { Box::from_raw(ptr as *mut Vec<i32>) });
}
```

Three rules follow, and violating any of them is a use-after-free or a leak that
neither language's checker will catch:

1. **`Box::from_raw` exactly once per `Box::into_raw`.** A borrowing accessor that
   uses `from_raw` frees the value at the end of the call.
1. **Null-check every pointer parameter.** Vx may pass a null for an
   uninitialised handle; Rust must not dereference it.
1. **Monomorphise per concrete type.** Symbol names carry the instantiation
   (`vx_vec_push_i32`, `vx_hash_map_insert_i32_f32`), because C has no generics.

The collections surface is generated from these rules by the
`instantiate_vec_ffi!` / `instantiate_hash_map_ffi!` macros in
[`stdlib/rust_core/src/ffi/macros.rs`](../../stdlib/rust_core/src/ffi/macros.rs) —
prefer extending a macro over hand-writing a shim.

> [!NOTE]
> This lifecycle governs the **stdlib FFI boundary only**. It is unrelated to
> `transfer(x, Memory::X)`, which is a compiler-level memory-space move lowered to
> the `vx.transfer` op (`memref.alloc` + `memref.copy`), not an FFI call. See
> [`docs/topology_representation.md`](../topology_representation.md).

## 3. `spawn on(...)` kernel dispatch ABI

For `Topology::CPU` the body of `spawn on` is inlined where it stands; there is
nothing to dispatch to. For accelerator topologies
(NPU / ANE / GPU) the body is outlined into a function `vx_npu_kernel_N` carrying
the `llvm.emit_c_interface` attribute, so MLIR additionally emits a C-interface
wrapper, which Vx exports as `_vx_ciface_vx_npu_kernel_N` (see §4). The captured values become the kernel's
parameters, and memref-backed captures are passed as pointers to their memref
descriptor structs (consistent with §2).

The kernel takes the parameters of its function first, in the order that the
function declares them. The other captured values come after them, in the order
of their first use in the region. So an edit to the region body does not change
the order of the function's parameters.

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
`_vx_ciface_<name>`. libffi places each argument in the correct GP/FP register
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

### 3.3 libffi dependency

The runtime uses libffi, which ships with the platform — we do **not** vendor or
build it. Two pieces come from libffi, not from Vx:

- **The `ffi_type_*` primitives** (`ffi_type_sint8`, `ffi_type_float`,
  `ffi_type_pointer`, …) that `vx_abi_ffi_type` returns are not functions; they
  are predefined global objects of type `struct ffi_type` (size/alignment/class
  descriptors). They are only *declared* `extern` in `<ffi/ffi.h>`; their storage
  is *defined* inside the system libffi.
- **The call machinery** `ffi_prep_cif` / `ffi_call`.

On macOS the header is `<ffi/ffi.h>` (under the SDK's `usr/include/ffi/`), and
the symbols are resolved at link time through the SDK stub `usr/lib/libffi.tbd`
(its symbol table lists `_ffi_type_sint8`, `_ffi_type_float`, …) and at run time
from the system libffi in the dyld shared cache.

Because only the header is implicit, the library must be linked explicitly with
`-lffi`. This is wired in two places; forgetting either yields an undefined
symbol such as `_ffi_type_sint8` even though compilation succeeds:

- `build.rs` — `-lffi` on the shared-runtime link plus
  `cargo:rustc-link-lib=dylib=ffi` for the static archive linked into `vxc`.
- `src/jit.rs` — `-lffi` on the JIT executable link.

### 3.4 Linux and other non-macOS targets

`spawn on(Topology::NPU/ANE/GPU)` targets Apple Silicon, and the dispatch runtime
(`runtime/npu_dispatch.mm`) is an Objective-C++ file that links Metal, CoreML and
Accelerate. The whole thing — runtime compilation, the libffi link, and the
`cargo:rustc-link-lib=dylib=ffi` directive — is gated behind
`cfg!(target_os = "macos")` in `build.rs`, and the JIT executable link adds
`-lffi` only on macOS (`src/jit.rs`). So on Linux/Ubuntu:

- None of the NPU dispatch path or libffi is compiled or linked; a build does
  **not** require `libffi-dev`. Accelerator `spawn on(...)` is simply not
  available there yet — only the CPU (`async.execute`) path applies.
- If the dispatch path is ever ported to Linux, note that the header is plain
  `<ffi.h>` (from `libffi-dev`), not macOS's `<ffi/ffi.h>`, and `-lffi` must stay
  gated to targets that actually link the runtime.

### 3.5 GPU kernels that `--emit-ptx` writes

`vxc --emit-ptx prog.vx -o kernels.ptx` writes the GPU kernels of a program to
one PTX file, for a program that loads and launches them itself (for example
with `cuModuleLoadData` and `cuLaunchKernel`). It also writes `kernels.json`
beside the PTX. Without `-o`, the files are `prog.ptx` and `prog.json` beside
the source.

A GPU kernel is a `spawn on` region whose topology has `arch: nvptx64` and
that can run on the GPU. Its PTX entry has the name of the function that holds
the region:

```vx
pub fn add(out: &mut Tensor<f32, [?], Memory::HBM>, a: &Tensor<f32, [?], Memory::HBM>,
           b: &Tensor<f32, [?], Memory::HBM>, n: i32) -> void {
  spawn on(Topology::Device) { ... }   // .entry add
  return;
}
```

The name comes from the signature and not from the body, because a body
declares nothing global (`docs/language_rules_for_a_parallel_compiler.md`).
`--emit-ptx` refuses these kernels:

- a function that holds two GPU kernels: put each kernel in a function of its own;
- a kernel that uses a value that is not a parameter of its function;
- a function whose name PTX cannot spell: use only letters, digits and `_`.

The kernel's parameters are the function's parameters that the region uses, in
the order that the function declares them. Each takes `words` PTX parameters:

| Vx parameter | PTX parameters |
| --- | --- |
| a scalar | 1, of the scalar's width |
| a tensor of rank r | 3 + 2r `.u64`: the base pointer, the aligned pointer, the offset, r sizes, r strides |

The kernel reads elements through the aligned pointer. For a tensor that the
host allocated, give the same pointer twice, offset 0, its sizes, and strides
for row-major order.

The `.json` file gives `target` (the `chip:` of the machine file, or
`VX_GPU_CHIP`, `sm_80` by default) and, for each kernel, its `name` and its
`params` in launch order. A parameter
has `name` and `type` as the Vx source writes them, `mutable` (the type is
`&mut`), `kind` (`scalar` or `tensor`), `element` (the MLIR element type),
`rank` for a tensor, and `words`:

```json
{ "name": "n", "type": "i32", "mutable": false, "kind": "scalar", "element": "i32", "words": 1 }
```

An MLIR integer type has no sign, so `element` is `i16` for both `u16` and `i16`. Take the sign
from `type`.

A kernel whose shared tensors need more than 48 KiB keeps them in dynamic
shared memory, and its entry also gives `dynamic_shared_bytes`. A program that
launches it calls `cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, n)` once, and passes `n` as
the `sharedMemBytes` of `cuLaunchKernel`. Vx's own launch does both.

A kernel from a region with `across` loops also has `launch`: `grid`, one entry for each loop
of blocks, `block` for the lanes of a block, and `"grid_loops": true`. An entry is a number,
`{"param": "rows"}` for the value of a parameter, or `{"computed": "(rows + 3) / 4"}` for a size
that the host computes from parameters and numbers. A size is not a parameter of the kernel, so
a launcher computes it and launches that shape. A loop of blocks steps over the blocks of the
grid, so a smaller grid is also correct. The kernel needs exactly the lanes of `block`.

With no `launch`, the program that launches the kernel chooses the grid and the block.

A kernel with code after its nest of blocks, or with the code of a group, counts the passes that
ended. Its last parameter is that count, `arrivals`, of `kind` `arrivals`: a tensor of `u32` with
`elements` elements, one for each group. `elements` is 1 for code after the whole nest, and the
size of the first axis of blocks for the code of a group, such as `{"param": "heads"}`. The
caller zeroes the buffer before the first launch. The last pass of each group sets its element
back to 0, so later launches, such as the replays of a CUDA graph, use the same buffer with no
new zeroing. Two launches that run at the same time need two buffers.

### 3.6 Calling a Vx function that launches GPU kernels

A host program that is not Vx, for example Python with `ctypes`, can call a `pub fn` from a
library that `vxc --action emit-obj` builds. The function launches its GPU regions through the
CUDA backend (`runtime/cuda_dispatch.cpp`). These rules hold for such a call:

- **Parameters.** A `pub fn` takes the C signature of its MLIR function. A tensor parameter of
  rank r is 3 + 2r values: the allocated pointer, the aligned pointer, the offset, r sizes and r
  strides (the `words` of §3.5). A scalar is one value. For example,
  `glm_vx_spawn_scale(out, x, alpha, n)` with two rank-1 tensors is
  `(ptr, ptr, i64, i64, i64, ptr, ptr, i64, i64, i64, f32, i32)`.
- **Device memory.** A tensor in a GPU space, such as `&Tensor<f32, [?], Memory::HBM>`, can
  point at memory that is already on the GPU. The launch gives the descriptor to the kernel as
  it is, with no copy.
- **Context.** Vx uses the primary context of each GPU, the one that the CUDA runtime API uses.
  Make it current (`cuDevicePrimaryCtxRetain`, then `cuCtxSetCurrent`) and allocate in it. Vx
  stops the program with a message when it finds another context current.
- **Stream.** By default, each Vx launch and copy goes on the legacy default stream, and a
  launch does not wait for the GPU (#25). Work of the host program on the legacy default stream,
  on the per-thread default stream, or on a blocking stream is ordered with Vx's. Work on a
  stream made with `cudaStreamNonBlocking` is not. `VX_CUDA_SYNC=1` makes Vx wait after each
  launch.
- **A stream of the host program.** `void vx_cuda_set_stream(void *stream)` sets a `CUstream`
  for the calling thread, and null gives back the legacy default stream. `vx_cuda_get_stream()`
  returns it, so a host program can put back the stream it had before. Then Vx's kernels, cuBLAS
  calls and copies go on that stream. A wait (before a copy to host memory, at the end of a region)
  waits for that stream only. Vx allocates its temporaries with `cudaMallocAsync` on the stream,
  and frees each with `cudaFreeAsync` on the stream it came from, with no wait. A build without
  CUDA has both calls, and they do nothing.
- **CUDA graphs.** A host program can record a call into a graph with `cudaStreamBeginCapture`
  on its stream. A function whose tensors are all in device memory and whose results stay there
  records cleanly. While the stream records, Vx stops the program with a message at anything a
  graph cannot hold: a wait, a copy from or to host memory, a free of memory that needs a wait,
  `VX_CUDA_SYNC=1` and `VX_TIME_KERNEL`. The first call of a function loads its kernel image, and
  the first GEMM on a GPU creates a cuBLAS handle with a workspace of its own. Vx stops when
  either happens in a recording, so call the function once before the recording.
- **Launch shape.** The function computes the sizes that its `across` loops give and launches
  that shape (§3.5 and `docs/spawn_on.md` §8).
- **No GPU.** When no GPU takes a launch, the region runs on the CPU. A function that only GPU
  code calls stops the program there with a message.
- **Optimization level.** The device image follows `-O`. A library built at `-O 3` can hold
  different PTX from `--emit-ptx` at the default level, so build both at the same level.

## 4. Symbol names

- A function the program defines keeps its name, and is `hidden` and `dso_local`.
  The executable does not export it, so a program's `fn write` does not replace
  libc's `write` for the runtime library.
- A `pub fn` keeps its name and is exported, so a shared library built from the
  program gives it to C. Only a top-level function that is not generic can be
  `pub`. A `pub fn write` replaces libc's `write`, as any exported C function does.
- Apart from `main` and `pub` functions, every symbol Vx exports starts with
  `_vx_`. The kernel entry
  points `_vx_ciface_<kernel>` are the only ones today. The dispatch runtime
  finds them with `dlsym`.
- `main` keeps its C name, because the C runtime calls it.
- A function in an `extern` block keeps its C name and C linkage.

## 5. Related

- `docs/npu_hardware_dispatch.md` — the C-interface dispatch mechanism.
- `docs/spawn_on.md` — `spawn on(...)` semantics and outlining.
- `src/dialect/VxLowering.cpp` — `SpawnOpLowering` / `LaunchOpLowering`.
- `runtime/npu_dispatch.mm` — the runtime dispatcher.
