# Spike: one GPU backend for the Apple GPU and WebGPU

*Decision record for #2691 ("Spike: portable GPU backend"), part of the vx-cam laptop proposal
(#2688). 2026-11-03. The Metal backend that followed is #2677.*

## Question

Which route gets Vx kernels onto the Apple GPU (Metal) and into a browser (WebGPU) from one
backend? The proposal listed three routes: A (SPIR-V, then Naga), B (a direct WGSL emitter) and
C (a direct MSL emitter). This spike adds D (Metal's own LLVM-based format, AIR).

## Answer

Build route C now, and build B from the same code for the browser. Route C ran two real Vx
kernels on the M4 GPU. With Metal's safe math, the output was bit-equal to the CPU result.
Routes B and C share all the hard work. That work turns a Vx kernel into structured code with
flat buffer parameters, and existing MLIR passes do most of it. Route A adds two translators
whose acceptance of this SPIR-V is not known. Route D could not run here, and it gives nothing
for the browser.

## Setup

- MacBook Air M4, 10-core GPU, macOS 27.0.1, Homebrew LLVM and MLIR 22.1.4.
- `vxc` and `vx-opt` from a debug build of main of Nov 1. The spike branch is at 2ad21238.
- Two kernels, in [`portable_gpu_backend_spike/`](portable_gpu_backend_spike/):
  - `blur.vx`: a separable 9-tap box blur of a 1280 x 720 f32 plane. It has two `across`
    kernels, one block per row (or column) and 256 lanes.
  - `gemm.vx`: a 1024 x 1024 matrix product. Each block makes 256 elements of one row of `c`.
    It keeps 16 elements of `a`, a 16 x 256 tile of `b` and 256 sums in shared memory.
- The kernel source is the `gpu.module` that `materializeGpuKernels` makes, before
  `deviceImageOf` runs. Today only an NVIDIA machine file gives a kernel its `levels:`, so the
  spike compiled with `--machine fleet/h100-sxm.vx` and ran `vx-opt --convert-vx-to-standard`.
  The module has no NVIDIA operations. Shared memory still uses the NVVM address space number 3.
- Other builds ran on the machine at the same time (load average 15 to 26). When the host waited
  for each frame, the GPU time of one blur frame varied from 0.42 ms to 1.8 ms. The batched time
  (100 frames in one command buffer) was stable. A steady video stream behaves like the batch.

## The common first half

Routes A, B and C need the same four steps. All four worked on both kernels.

1. `mem2reg` turned every scalar stack slot (`memref.alloca`) into an SSA value. This removes
   the stack-slot half of item 4 of "Shared work before a second backend" in
   [`gpu_backends.md`](../gpu_backends.md).
1. `lift-cf-to-scf` turned the CFG into `scf.while` and `scf.if`. Neither kernel kept a
   branch. The pass only visits `func.func`, so the spike renamed each `gpu.func` while the pass
   ran. A real pass calls `transformCFGToSCF` on the `gpu.func` itself.
1. Each tensor parameter `memref<?x?xf32>` became a flat `memref<?xf32>`, and its row length
   became a scalar parameter. The NVIDIA launch already passes these sizes as separate values.
1. A load or store through a row view (a unit-stride `memref.reinterpret_cast`) became a load or
   store on the flat parameter, at the offset plus the index. Neither `fold-memref-alias-ops`
   nor `expand-strided-metadata` does this today. This is the view half of item 4.

Steps 3 and 4 were Python text rewrites in the spike (`prep_kernel.py`). In the backend they are
one MLIR pass.

## The routes

### A. SPIR-V for shaders, then Naga

- `convert-gpu-to-spirv` refused the kernel as Vx gives it. It refused `memref.dim` first,
  because the SPIR-V runtime array has no shape. After that fix, it refused
  `memref.reinterpret_cast`. After steps 1 to 4, the conversion worked, and `mlir-translate`
  wrote a 4.7 KB SPIR-V module (`route_a.sh`).
- Each scalar parameter became its own storage buffer. The blur used 6 storage buffers, 2 for
  tensors and 4 for scalars. WebGPU allows 8 for each shader stage by default, so a kernel with
  three rank-2 tensors and three scalars goes over. The scalars must share one uniform block.
- The spike stopped there. Naga is not on this machine, and the session's permission rules
  refused `cargo install naga-cli`. `spirv-val` is not on this machine either. So it is not
  known if Naga accepts this module.
- On a Mac, this route also ends as MSL text that Metal compiles at run time. That is where
  route C ends, with two more translators before it.

### B. A direct WGSL emitter

- `emit_wgsl.py` is the MSL printer with different spelling. It uses the same walk over the
  structured kernel and has about 50 lines of differences. It printed WGSL for the blur.
- It was not checked and not run. No Naga, Tint or browser harness was available.
- The main open point is WGSL's uniformity rule for `workgroupBarrier()`. In these kernels,
  the barrier is in a loop whose bounds depend only on `workgroup_id` and uniform scalars.

### C. A direct MSL emitter (ran)

- `emit_msl.py` (about 250 lines) prints MSL from the structured kernel. The harness compiles
  the text at run time with `newLibraryWithSource`, which needs no Xcode.
- Blur, with `MTLMathModeSafe`: all 921,600 pixels were bit-equal to a C copy of the algorithm.
  The checksum was equal to the CPU copy of the same Vx kernels (`blur_cpu_ref.vx`).
- Blur, with Metal's default fast math: 151,733 pixels differed, by at most 1.8e-7.
- GEMM, with safe math: all 1,048,576 elements were bit-equal to a C product with the same order
  of additions and no fused multiply-add.
- The NVVM shared memory global (address space 3) became a `threadgroup` array. The backend must
  use the symbolic address space, as item 3 of the shared work says.

### D. Metal AIR through LLVM (did not run)

- AIR needs Apple's Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`). It is not
  installed here, and the spike left the download to the maintainer.
- Even with the toolchain, Vx would need to write what Apple's tools write: AIR metadata for each
  argument and builtin, and a `metallib` container. Metal.jl takes this route. As far as the
  spike could tell from its public sources, it also converts its bitcode to an older LLVM format
  before Apple's tools read it. The spike did not check this in detail.
- What D gives: it keeps the CFG and raw pointers, it needs no structurizer, and it needs no
  compile at run time. What it costs: it works only on a Mac, the format is not documented, and
  it does nothing for the browser.

## Numbers

| Route | How far it got | Output check | Blur, GPU ms per frame | Notes |
| --- | --- | --- | --- | --- |
| A | a SPIR-V module | not checked (no Naga, no `spirv-val`) | not measured | 6 storage buffers for the blur |
| B | WGSL text | not checked | not measured | same walk as C |
| C | ran on the M4 GPU | bit-equal (safe math) | 0.42 to 0.51 batched | see the run-time costs below |
| D | did not run | none | not measured | needs the Metal Toolchain |

- Route C, blur: 0.42 ms per frame at best when the host waited for each frame. The worst was
  1.8 ms, when the GPU clocked down between frames.
- Route C, GEMM: 30.8 ms to 36.4 ms, which is 59 to 70 GFLOP/s. This simple kernel keeps its sums
  in shared memory. Matmul goes to MPS by Vx's policy anyway.
- Compile at run time: 100 ms the first time with fast math, and about 1 s the first time with
  safe math. Metal caches the result, and later runs took about 1 ms.
- The by-hand route C (`route_c.sh`, from Vx source to MSL) took 0.28 s for the blur.
- No copy: a page-aligned host allocation wrapped with `newBufferWithBytesNoCopy` kept its
  address. The blur ran at the same speed, and the output stayed bit-equal.
- Chrome and Safari with WebGPU: not measured.

## Recommendation for the backend (#2692)

1. Add `DevicePipeline::Metal` next to `Nvptx` in `pipelineForArch`
   (`src/dialect/VxLowering.cpp`), for `arch: applegpu`. Give `fleet/m4-uma.vx` `levels:`, so
   that the checker accepts `across` loops there.
1. Build steps 1 to 4 as one pipeline for that arm. Steps 1 and 2 are existing MLIR passes, and
   steps 3 and 4 are one new pass.
1. Write the MSL printer in C++ over the structured form: `scf`, `arith`, `math`, `memref` on flat
   buffers, and the `gpu` ids and barrier. An op it does not know is a compile error that names
   the op.
1. Compile with safe math. A Vx program must give the same bits on the CPU and the GPU. The
   NVVM path already forces a correctly rounded square root for the same reason.
1. Put the WGSL printer next to it in the browser work (#2697).
1. Do not build route A now. Naga can later check the WGSL output in CI.
1. Do not build route D now. Look at it again in #2699 if the compile at run time or a
   missing feature (simdgroup matrices) needs it.

## What this means for the browser build (#2697)

- The WGSL printer reuses the walk, so the browser build does not need a second backend.
- WGSL has no 64-bit integers, and `f16` is optional. A kernel that needs `i64` arithmetic or
  more than 2^31 elements must be a compile error for a WebGPU topology, like E6026 for `dtypes:`.
- The default limit of 8 storage buffers for each stage means the scalars go in one uniform
  block. The compiler counts the tensors and refuses a kernel that goes over the limit.
- In the browser, WebGPU cannot map a storage buffer without a copy. A frame crosses with
  `writeBuffer` or `importExternalTexture`, and the placement report must show those copies.

## What this means for the argument convention (`runtime/vx_kernel_launch.h`)

- NVIDIA: each tensor passes two pointers, an offset, its sizes and its strides, each as its own
  kernel parameter.
- Metal and WebGPU: each tensor is one buffer binding, from the aligned pointer. The offset goes
  in `setBuffer:offset:` or into the index. All sizes, strides and scalars go in one constant
  block (`setBytes`, up to 4 KB on Metal, or a uniform buffer on WebGPU). The host code in
  `vx_kernel_launch.h` can build that block from the same descriptor. The kernel side is new.
- The payload can carry MSL and WGSL as they are, because both are text. A binary image (SPIR-V
  or a metallib) needs the length-prefixed image section of shared-work item 2.
- Unified memory: a tensor in a page-aligned allocation (16 KiB pages, length rounded up) can be
  wrapped with `newBufferWithBytesNoCopy`. Then the handoff of #2690 is a wait and not a
  copy.

## How to run it again

From the repository root, with `config.local` set up (`VX_CONFIG`, `VXC` and `VX_OPT` can point
somewhere else):

```sh
d=docs/discussions/portable_gpu_backend_spike
$d/route_c.sh $d/blur.vx /tmp/blur.metal /tmp/blur      # and gemm.vx
$d/route_a.sh /tmp/blur/final.mlir /tmp/blur.spv
python3 $d/emit_wgsl.py /tmp/blur/final.mlir /tmp/blur.wgsl
clang -O2 -fobjc-arc $d/harness.m -framework Metal -framework Foundation -o /tmp/harness
VX_SAFE_MATH=1 /tmp/harness /tmp/blur.metal vx_npu_kernel_0 vx_npu_kernel_1
```

`VX_NOCOPY=1` makes the harness wrap a host allocation as the input buffer.
`gemm_harness.m` runs the GEMM in the same way. `blur_cpu_ref.vx` prints the checksum of the CPU
copy of the blur.
