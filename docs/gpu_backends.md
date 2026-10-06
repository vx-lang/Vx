# GPU backends

*What a device backend is in Vx, which ones exist, and how a new one gets in.*

This is for contributors who want to support a new device, and for users
who want to know what runs where. Related:
[`adding_a_topology.md`](adding_a_topology.md) (declaring the hardware),
[`spawn_on.md`](spawn_on.md) (the execution model),
[`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md) (building and testing).

______________________________________________________________________

## What a backend is

A backend is three pieces. The NVIDIA backend is the worked example for each.

1. **A machine file.** The hardware is declared in a `.vx` file: its memory
   spaces and their capacities, the transfer edges between them, the `arch:`
   it executes, and `dtypes:` — the element types it supports. No compiler
   change is needed for this part; see [`adding_a_topology.md`](adding_a_topology.md)
   and the examples in [`fleet/`](../fleet/).
1. **A device image compiler.** `spawn on(Topology::X) { ... }` is outlined
   into a kernel, and eligible kernels are cloned into a standard MLIR
   `gpu.module`, which no vendor owns. From there, `deviceImageOf()` in
   `src/dialect/VxLowering.cpp` runs the NVVM passes and produces PTX. A new
   backend adds a sibling of that function producing its own image format
   (for example SPIR-V; see "SYCL first" below for the route).
1. **A runtime dispatch library.** Compiled programs call a small C interface
   (`include/vx_hardware_runtime.h`): allocate and copy in, launch a kernel,
   wait, copy out, free. `runtime/cuda_dispatch.cpp` is the CUDA
   implementation; a new backend adds its own file next to it and an arm in
   `build.rs`. The argument marshalling in `runtime/vx_kernel_launch.h` is
   vendor-free, unit-tested on the host, and meant to be reused.

## What a backend does not do

The language, the type checker, and the placement rules (`transfer`, memory
spaces, capacity checks) are the same for every backend. Two decisions of
record:

- **Matmul goes to a vendor library** (cuBLAS today; oneMKL would be the
  Intel parallel). A hand-tiled GEMM loses to the vendor one, so generated
  kernels are for everything else.
- **A device limitation is a compile error, never a silent change.** A device
  that lacks f64 declares `dtypes:` without it, and placing an f64 tensor
  there is error E6026 before anything runs (`fleet/m4-uma.vx` already does
  this for a GPU without f64). We do not demote f64 to f32 behind the
  programmer's back.

## Status

| Target | Route | Status |
|---|---|---|
| CPU (x86-64, arm64) | native, through LLVM | working, tested in CI |
| NVIDIA | `gpu` dialect → NVVM → PTX; cuBLAS for matmul | working for placed kernels; completion tracked in #251 |
| Apple GPU / ANE | library routing (MPS, CoreML) | working for the routed patterns |
| Intel XPU | kernel SPIR-V, SYCL runtime, oneMKL for matmul | the first community backend, offered by a contributor; planning in #1137 |
| Vulkan | shader SPIR-V | later: the shader model fights a language with raw pointers, and there is no vendor BLAS — see below |
| AMD | ROCDL | machine file exists (`fleet/mi300x.vx`); no backend yet. Likely the cheapest backend after the shared work — ROCDL sits close to the NVVM path, hipBLASLt parallels cuBLAS — and a maintainer priority when hardware access appears |
| TPU | — | not planned: there is no native route, since the vendor compiler stack is closed. Emitting StableHLO for a PJRT plugin is a public route, but a different kind of backend than the three pieces above |

A "working" row is true of a build made for that target: `build.rs` builds
one dispatch library (shared-work item 5), so a compiler built with CUDA
cannot run programs on an Apple GPU, and one built on a Mac cannot run them
on an NVIDIA GPU.

**SYCL first.** The first community backend targets Intel GPUs through the
SYCL runtime. Vx still compiles each kernel itself, to SPIR-V; the dispatch
library loads that SPIR-V into a SYCL kernel bundle and launches it on a
`sycl::queue`. SYCL is the runtime and not a compile target: Vx's checker
already does the job SYCL's C++ layer does, so Vx does not generate SYCL C++.
The runtime is chosen over calling Level Zero directly because oneMKL, the
vendor library that matmul goes to (see above), takes a `sycl::queue`: a backend that calls oneMKL
uses SYCL either way. Level Zero is what SYCL runs on underneath.

Kernel SPIR-V and shader SPIR-V are different dialects, and the difference
decides which runtimes a kernel can go to:

- **SYCL (and Level Zero under it)** takes kernel-flavor SPIR-V (the OpenCL
  flavor): `Physical64` addressing and kernel parameters that are plain
  scalars or plain pointers into device memory. The parameters the NVIDIA
  path passes are already only those: per tensor, two pointers, an offset,
  sizes and strides, each its own parameter. So `runtime/vx_kernel_launch.h` carries over.
- **Vulkan** takes shader-flavor SPIR-V. It has no raw pointers unless the
  device supports `VK_KHR_buffer_device_address`, so a Vulkan backend here
  should require that extension outright — the alternative is descriptor
  sets, which means a rework of the whole argument convention. Vulkan also
  has no vendor BLAS, so the matmul rule has no answer there yet. What Vulkan
  buys is reach (cards from every vendor) and Mesa's lavapipe, a software
  Vulkan device that installs on a plain CI runner, so Vulkan kernels could
  actually *execute* in CI.

A Vulkan runtime can come later under the same device-image machinery.

**The route to kernel SPIR-V.** `convert-gpu-to-llvm-spv{use-64bit-index=true}`
lowers a Vx kernel to the LLVM dialect with SPIR-V calling conventions; with
`llvm.target_triple = "spirv64-unknown-unknown"`, `llc -mtriple=spirv64`
turns it into SPIR-V, as text by default, so a test can FileCheck it. This
reaches a module for a real Vx kernel today. The other two routes,
`convert-gpu-to-spirv` and the XeVM target, have not been run against a Vx
kernel end to end; the planning issue (#1137) records what blocks the first (it
expects structured control flow where Vx has a CFG, and replaces a memref
with a runtime array that drops its shape and strides), and keeps the full
recipe for the route above.

## Support tiers

CI runs on Ubuntu with no GPU, so "merged" has to mean something a machine
without the hardware can check. The tiers, modeled on Rust's target tiers:

- **Tier 1 — CPU paths.** CI runs the tests; a regression blocks the merge.
- **Tier 2 — NVIDIA.** Maintainer-owned. CI proves it builds; correctness is
  shown by parity runs on hardware before any release or public claim. A
  scheduled, non-blocking run on rented hardware would catch regressions
  between releases; until one exists, the pre-release run is the only
  hardware signal.
- **Tier 3 — community backends.** Must build in CI with no vendor SDK
  installed: the device-image half compiles and its output is checked with
  FileCheck, and the vendor-free runtime pieces have unit tests in
  `tests/runtime/`. Correctness is shown by parity runs the hardware owner
  records in each PR, with the test marked `REQUIRES: gpu`. A named owner is
  a requirement for merging; a backend that loses its owner is marked
  unmaintained here, not reverted. A Tier 3 regression never blocks `main`,
  and that includes its CI build job: the job is a non-required check, and
  when a change on `main` breaks it, the owner has until the next release
  (or thirty days, whichever comes first) to fix it before the job is
  disabled and the row here marked unmaintained.
  To compile without the SDK, vendor the open headers (Khronos publishes
  both the Vulkan and the Level Zero headers) or load the driver with
  `dlopen`, so the file builds everywhere and only running needs the device.
  SYCL is the exception: its headers are C++ and belong to one
  implementation, so the SYCL dispatch library is built with oneAPI's
  compiler, which CI installs for that job. How that job is set up is
  settled in the planning issue before the runtime PR.

## Shared work before a second backend

These are the places where the compiler currently assumes NVIDIA is the only
device target. Each gets its own issue; they are vendor-neutral, so they keep
their value whichever backend lands first.

1. **Kernels all land in one module, for one arch.** `materializeGpuKernels`
   in `src/dialect/VxLowering.cpp` only clones kernels whose `arch:` is
   `nvptx64`, and clones them all into the single `gpu.module @vx_kernels`.
   Kernels need to be grouped into one `gpu.module` per target arch, each
   module carrying an attribute that names its target (`#nvvm.target`, or
   the SPIR-V route's `llvm.target_triple`), and `deviceImageOf()` becomes a choice keyed on that
   attribute rather than a function that always runs the NVVM passes.
1. **The dispatch payload cannot carry a binary image.** The payload is a
   sequence of NUL-terminated `key=value` entries, which works because PTX is
   text; `deviceImageOf()` rejects images containing a NUL byte. SPIR-V is
   binary, and since nothing here is frozen yet, the fix is to change the
   format — a length-prefixed image section — rather than base64-encode
   binary into a text format. Alongside the image, the payload should name
   what it carries: a version key (`abi=1`), the image format (`format=ptx`
   or `format=spirv`), and the kernel entry point, so a dispatch library
   that sees something it does not know refuses instead of guessing. The
   payload also crosses the remote-worker wire verbatim
   (`runtime/vx_wire.h` already frames it by length), so the version key
   inside the payload is what lets an older worker refuse rather than
   misread. The image is checked today by searching the `--emit-llvm` text
   (`tests/integration_test/device_image_test.rs`); once it is a binary
   section, that test has to decode the section, or it stops covering the
   image.
1. **Address spaces are mapped for NVVM only.** `AddressSpace` in
   `src/arch.rs` knows the NVPTX numbering, and several places in
   `VxLowering.cpp` compare the shared-memory address space to the integer 3
   directly. The lowering should carry the symbolic
   `#gpu.address_space<workgroup>` instead, and each target's type converter
   maps it to its own number (3 for NVVM, the `Workgroup` storage class for
   SPIR-V). The same goes for memory a kernel keeps for itself, such as the
   loop bounds it stores in stack slots: it has to be marked
   `#gpu.address_space<private>`. The SPIR-V route puts memory with no address
   space in device memory (`CrossWorkgroup`), and a stack slot there is
   invalid SPIR-V ("Storage class must match result type storage class").
1. **Scalar element types escape the `dtypes:` check.** E6026 covers tensors
   that are placed or transferred. An f64 *scalar* inside a `spawn` body
   passes the checker today and would only fail on the device, and the same
   goes for vectors and the small float formats. The body of a `spawn` has
   to be checked against the target topology's `dtypes:` list, for every
   element type it uses.
1. **One dispatch library per build.** `build.rs` builds exactly one runtime
   dispatch library and programs load that one (`VX_DISPATCH_LIB` overrides
   it by hand). Running two device kinds from one program needs a routing
   registry: each `vx_plugin_*` call carries or implies a topology id, and
   the registry maps that id to the backend that owns the device. This is
   the item that keeps Vx a heterogeneous language rather than a portable
   single-target compiler, and the pattern exists already: the remote path
   routes a dispatch by topology through a manifest
   (`runtime/vx_remote_routing.h`); local routing is the same idea without
   the socket.
1. **A conformance test set.** A vendor-neutral set of placed-kernel programs
   whose results are compared against the CPU path.
   `tests/backend/pass/placed_kernel_four_operands.vx` is the reference
   shape. The set must include a matmul routed through the backend's vendor
   library (oneMKL for Intel), compared against the host product — that
   route is how the dominant operation runs, so it cannot be the one path
   without a conformance case. Once dispatch routing (the item above) lands,
   the set also gains a program that places kernels on two device kinds in
   one run.

## The bar for being listed as working

The conformance set passes on real hardware, with the runs recorded in the
PR. The bar is the CPU path's result: same program, same numbers. Once
dispatch routing exists, the two-device program in the set is part of this
bar — a backend that only works alone keeps Vx from being what it is for.

## How to start

1. **Talk first.** Open an issue (or use the one you have) and agree on the
   slicing before writing much code.
1. **Machine file.** Declare the device per
   [`adding_a_topology.md`](adding_a_topology.md). This needs no compiler
   change and immediately exercises the placement checks, including
   `dtypes:`.
1. **Device image compiler.** The sibling of `deviceImageOf()`. This half is
   testable in CI with FileCheck, so it can merge before any runtime exists.
1. **Runtime dispatch library.** Implement the `vx_plugin_*` entry points,
   reuse `runtime/vx_kernel_launch.h`, add the `build.rs` arm. For SYCL, the
   entry points are a C wrapper around the C++ runtime.

Keep PRs small and in that order; dependent PRs go in a GitHub stacked PR.
Signing the CLA (`docs/CLA.md`) is checked by CI on the first PR.

One thing about parallelism so it does not surprise you: a kernel body runs
parallel only when the compiler can prove the outer `for` loop safe to split
across threads (the proof is conservative and syntactic), and it uses one
grid dimension today. #251 shows what completing a backend looks like for
NVIDIA.
