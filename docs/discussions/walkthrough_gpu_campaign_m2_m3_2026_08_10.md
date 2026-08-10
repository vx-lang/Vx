# M2/M3: making a placed Llama actually reach the GPU, and what it cost

Session of 2026-08-09 into 2026-08-10, continuing
[gpu_disaggregated_inference.md](implementation_plans/gpu_disaggregated_inference.md)
(#319, #321). [M1](walkthrough_gpu_campaign_m1_2026_08_09.md) built the CUDA dispatch
backend and everything about it testable without a GPU. This session made a real Llama
program routable to it, gave the compiler a vocabulary for *hosts* as distinct from
*machines*, and then rented an A100 and found out what the whole thing was worth.

The short version: token-for-token parity through cuBLAS on real hardware, 3.4x against
our own CPU path, and the 3.4x is the least interesting number in the session because it
is bounded by work that never left the host.

## The gap M1 left: placement was real, execution was not

M1 ended with a plugin that could recognise a GEMM and run it on a GPU. It could not
receive one from an actual program, for three separate reasons that only became visible
when a program was written against it.

**`vx.transfer` planned a movement it did not perform.** It lowered to `memref.alloc` +
`memref.copy` — a host allocation. Every placement analysis, every capacity check, every
E6009 was therefore metadata: correct metadata, describing a movement that did not
happen. `cca92e63` routes it through the plugin ABI when the target space is device
memory, so a `transfer` finally means something physical.

**`scope` was declared and ignored.** The lowering decides device-ness by asking the
machine model for the target space's `scope:` — `device`, `sm`, `cta`. That attribute
was reaching the MLIR only sometimes. Root cause, found after a while chasing the
lowering: the AST codegen path merged a module's structs and enums but not its memories
and topologies, so a declaration in an imported module simply was not there. Fixed in
`fe89f30c`. Worth recording because the symptom (an attribute intermittently absent)
pointed nowhere near the cause (module merging).

**`vx_plugin_free` freed with `free()`.** On a host backend that is right; on CUDA it
hands a device pointer to the C library allocator. The same commit made every backend
delegate to the vendor path that allocated, and split the CPU backends into
`x86_dispatch.cpp` and `arm64_dispatch.cpp` over a shared `host_dispatch_common.h`
parameterised by name and alignment — the user's suggestion, and it removed a class of
"which backend am I actually running" confusion for the rest of the campaign.

## The linear-type detour, and being wrong about it

To write Llama's projections as tensors, `@` had to be usable on a weight matrix read
once per token. It consumed its operands (#335), so the second read was a
use-after-move.

The first fix (`910b7da4`) made `@` non-consuming. This was wrong, and the user said so
directly: *"linear type was in my mind from day one... we can elide copies later in
MLIR"*, and *"I'm glad that we are fixing it right now because once we break out of this
linear type system, going back is very difficult."*

Non-consuming `@` does not weaken one operator, it puts a hole in the discipline: a
value that was moved-from is observable again, and every later analysis that assumed
linearity is quietly unsound. The replacement (`0920216b`) is borrowing — `&a @ &b`
reads without consuming, `a @ b` still moves — which stays inside the system instead of
stepping out of it.

The lesson is not about matmul. A type system's guarantees are only as strong as the
one operator willing to break them, and the cheap fix was the one that broke them.

## Three pieces of surface the program needed

- **`tensor_view_2d`** (`a867dad3`) — a rank-2 view over foreign memory that copies
  nothing. Llama's weights are `*mut f32` offsets into an mmap'd blob; without a
  zero-copy view they could not become tensors at all, and with a copying one the
  staging cost would have swamped the point.
- **`matmul_into`** (`56d36386`) — fills a buffer it was handed rather than allocating
  one. This is the first real producer of `outkind=buffer`; until now every matmul
  published through a slot, so the buffer branch in both backends and in
  `vx_dispatch_plan.h` was code that had never run in anger and only looked exercised
  because the decode tests construct that case by hand.
- **A device index on `Topology::GPU`** (`b8dce73d`, #331) — `GPU[0]` and `GPU[1]` as
  distinct dispatch ids, carried to the plugin as `topo=N`. Written for M4; it turned
  out to be the thing M4 was mostly waiting on.

With those, `llama2.vx`'s seven projections became `linalg.matmul` regions the compiler
classifies and a plugin routes (`4b102c7e`), and the weights are staged once before the
first token rather than per dispatch (`1b5af545`) — the difference between a resident
model and moving `wq` across the bus 6xN times for an N-token generation.

## Hosts, machines, and the flag that was three opinions

Running against a fleet exposed a nomenclature problem that turned into a real design
change (#342).

There was `--machine` (the accelerator) and `--target` (the triple). The user asked for
`--host`, and articulated why: *"we do not assume any host for any program"* — a program
that never stages through a host should not need one, and one that does should be made
to say so. That is E6014, and `--host default` means native compilation.

Then the sharper observation: *"we should remove `target`. it is super confusing. just
found out that both gcc and clang have suffered from this nomenclature. host is the
'driving' device, everything else is 'machine'."*

They were right, and the bug proves it. `--target` was a third opinion that could
disagree with the machine a program was admitted against — and did: `--target nvptx64`
stamped `nvptx64-nvidia-cuda` onto a module containing `main`. The host *is* the target;
`main`, the dispatch calls, and the outlined kernels' C interfaces are all code for it.
So the flag is gone (`41e4e76a`) and the triple is derived from the host's declared
`arch:` (`ba390069`), with every fleet file now stating the ISA it executes
(`a07f419a`) — because *"how would anyone know the ISA otherwise"*.

Cross-compilation falls out rather than being added: the machine running the compiler is
never consulted, only the machine that was declared.

Two smaller corrections in the same thread, both the user's, both the same principle —
a model should not contain what the hardware does not have:

- *"how come X86 has SMEM?"* → *"if something does not exist, it should not be there. it
  should be error IMO."*
- *"Dont worry about declaring host memory capacity as host has virtual memory."* A
  device's HBM is a wall; a host's DRAM is a working set, and declaring a limit there
  would refuse programs that run.

## A test that asserted nothing

While adding FileCheck coverage I noticed the guard that catches vacuous tests only
rejected `CHECK-*` prefixes, so a test using a custom prefix could assert nothing and
pass. Widening it to derive prefixes from the RUN lines exposed a pre-existing one:
`cpu/npu_fusion_overhead.vx` carried `LLVM-LABEL: module {` against LLVM IR that has no
such line. It had never checked anything. Removed, filed as #338.

Worth stating plainly because the same shape recurred twice more this session: an
instrument that reports success without measuring is worse than no instrument, and the
only defence is a negative control.

## The A100

Two rentals. The first died immediately.

**`matmul_ane` segfaulted.** The FFN projections were placed on `Topology::ANE`, which
the CUDA plugin does not route, so they ran on the host through libffi — and dereferenced
pointers into device memory. Invisible on every host build, where "device memory" is host
memory and the code is merely slow. The fix is one line (delegate to `matmul`); the
lesson is that a fallback path which is *correct* on one machine can be *fatal* on
another, and nothing in the type system said so.

Results, stories15M, 1000 tokens, greedy:

| configuration | wall clock |
|---|---|
| all 7 projections on the A100 | 20,721 ms |
| 4 of 7 on the A100 (FFN on host) | 41,243 ms |
| same binary, device hidden | 69,663 ms |

64/64 token parity with llama2.c, through cuBLAS. Routing confirmed by
`24000 [Vx CUDA] GEMM 288x1x288 f32 -> buffer` and
`1000 [Vx CUDA] GEMM 32000x1x288 f32 -> buffer`.

### Why only 3.4x

The user asked, and the answer is Amdahl, not cuBLAS.

Both figures include ~2.4 s of JIT compilation, so the runtime comparison is 67.3 s
against 18.3 s — 3.7x. If the GPU makes matmul time approximately free, that 18.3 s *is*
the non-matmul work, so matmul was ~73% of host runtime and everything else ~27%. A 27%
remainder caps the speedup at 1/0.27 = 3.7x. We got 3.7x.

**The GEMMs are no longer the bottleneck.** What remains is attention, softmax, RMSNorm,
RoPE, sampling and the KV-cache writes — none of them `linalg.matmul`, so none classify,
so all run on the host. Lifting the ceiling is #251 (kernel emission), not a faster GEMM.

A second factor is hidden behind that ceiling and would dominate the moment #251 lands:
only the *weights* are resident. Each of ~43,000 dispatches does `cudaMalloc` for the
activation, H2D, `cudaMalloc` for the output, the GEMM, a full `cudaDeviceSynchronize`,
D2H, and two `cudaFree`s. A 288x288 matvec is ~166 KFLOP and should take single-digit
microseconds; two synchronising allocator calls and a device-wide sync are plausibly
50-100x that.

**What this run does and does not claim.** It claims correctness on real silicon through
a completely different arithmetic path, and residency working. The 3.4x is against our
own `-O0` loop nest, not a tuned CPU implementation, and it is ceilinged by unaccelerated
work rather than by anything about the GEMMs. It should not be put next to vLLM.

## Process notes worth keeping

**`git add -A` swept the user's uncommitted edits into `8b7922e2`.** They asked to
preserve rather than discard them (*"can we 'stash' the changes you didnt do"*), which is
`stash@{0}`. Explicit pathspecs only, from then on.

**#343 was filed wrongly and closed as invalid.** I reported that admitted configurations
carried no margin data; they had since #285, and my harness was reading the wrong JSON
field. Checking the artefact before filing against it would have cost a minute.

## State at the end of M3

- A placed Llama runs on an A100 with token parity, weights resident, seven projections
  on the device.
- The admission matrix runs one program against every SKU and prints byte-precise
  verdicts in both directions.
- `--host` / `--machine` are the whole vocabulary; the triple is derived, never flagged.
- Open: #251 (kernel emission, the real ceiling), #344 (strided views of a placed
  tensor), #320/#333 (dtypes that matter for serving), #332 (AOT), #339, #340.

M4 — two devices — is [the next walkthrough](walkthrough_gpu_campaign_m4_2026_08_10.md).
