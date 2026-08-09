# M1: a CUDA backend, written and tested before renting a GPU

Session of 2026-08-09, continuing the campaign in
[gpu_disaggregated_inference.md](implementation_plans/gpu_disaggregated_inference.md)
(#321). M0 ended with the compiler working on Linux and Llama matching llama2.c token
for token. This session built the CUDA dispatch backend, and — more usefully — built
everything about it that can be checked without a GPU, then checked it.

## The premise

The A100 is the expensive resource, so the question shaping the whole session was: what
fraction of "run a matmul on a GPU" can be made true on a laptop? The answer turned out
to be nearly all of it, and the parts that could be tested on the CPU are exactly the
parts that were wrong.

## What a plugin has to know, and what it was being told

Routing `c = a @ b` to cuBLAS requires four facts about the arguments: that the kernel
is a GEMM, which argument is A and which is B, where the result goes, and the type and
shape of each. M0 established the first (`kind=matmul`) and the last (element type and
rank in the ABI tag). This session finished the rest and found that two of them were
carrying less information than they appeared to.

**Slots.** The result of `c = a @ b` is not written into a buffer the kernel was handed.
The kernel allocates it and stores the descriptor through a captured slot — MLIR
`memref<memref<?x?xf32>>` — which the caller then loads. A plugin standing in for the
kernel must do the same: allocate, and publish a descriptor where the kernel would have
stored one. Described verbatim, though, that slot is a rank-0 memref with no scalar
element type, which is byte-identical to the encoding for an opaque pointer. The fact
that made the argument writable was being dropped.

Bit 24 of the tag now says "slot", and the element type and rank describe what it holds
rather than the slot itself (e5d94139). The kind byte is untouched, so the calling
convention and every existing consumer are unaffected.

**Roles resolved for parameters and silently not for locals.** This is the one that
would have cost GPU time. `matmulRolesOf` looked up each matmul operand among the
region's captures. A function parameter is captured as a buffer, so that works. A local
tensor lives in a slot, and the region *loads* the descriptor out of it before use — so
the captured value is the slot and the operand is the load, and the lookup returned -1.
Roles were therefore absent on every ordinary program.

The failure had no symptom. A plugin would read `kind=matmul`, find no roles, decline to
route, and run the outlined loop nest: correct numbers, never on the GPU, no diagnostic.
The existing test used parameters and passed. Following the load fixes it, and
`kernel_roles_local.vx` now pins the local form against `kernel_kind_matmul.vx`'s
parameter form.

It was found by writing the acceptance test — `gpu_matmul_roles.vx`, which uses locals
like ordinary code does — and dumping what the compiler emitted for it before assuming
anything.

## The decode, deliberately without CUDA in it

`runtime/vx_dispatch_plan.h` turns a dispatch into a GEMM plan and contains no vendor
API at all, so the decision can be tested anywhere. This is the code that decides which
operand a GPU reads and which it overwrites; a mistake produces a plausible matrix of
wrong numbers rather than a failure, which is the worst way to learn something on a
rented machine.

It refuses rather than assuming, on every axis — unclassified kernel, partial role list,
role index past the end, two roles naming one argument, mismatched element types, wrong
rank, non-contiguous rows, disagreeing inner dimensions, a result buffer of the wrong
extent, an `outkind` the tags contradict. Every refusal falls back to running the
outlined kernel as written, so strictness costs performance and never correctness.

`tests/runtime/gemm_plan_test.cpp` builds descriptors by hand in the shape the compiler
actually emits and checks the decode field by field, including that square operands
follow the stated roles rather than argument order. Writing the payload blobs out by
hand pins the wire format against the producer. It found two defects while being
written: a trailing comma parsed as a valid role list, and a repeated role silently
overwriting an earlier one.

## The backend

`runtime/cuda_dispatch.cpp` stages operands with `cudaMemcpy2D`, calls `cublasSgemm`,
`cublasDgemm` or `cublasGemmEx` (f16 and bf16 accumulating in f32), and writes the
result back — into the buffer for `outkind=buffer`, or as a freshly allocated buffer
plus a published descriptor for `outkind=slot`. cuBLAS is column-major and a row-major
buffer read column-major is its own transpose, so computing C^T = B^T A^T — swap the
operands, swap m and n, no transpose flags — leaves C correct with no repacking.

`build.rs` selects it wherever a CUDA toolkit is installed. Building it needs no GPU, and
a binary built with it runs correctly on a machine without one: `cuda_available()` says
so once on stderr and every kernel takes the host path.

It compiles clean under `-Wall -Wextra` against CUDA 13.2 on the x86 build box, and the
decode tests pass there too.

## The acceptance test is backend-independent

`tests/backend/pass/gpu_matmul_roles.vx` expects `38 56 83 128 9 7 10` whether the host
loop nest or cuBLAS computed it. Its operands are chosen so that a plugin which confuses
the roles is caught: a 2x3 by 3x4 case where a swap does not even conform, and a 3x3 by
a 3x3 diagonal where every assignment conforms and only the compiler's statement
distinguishes them — A·B gives 9, 7, 10 where B·A gives 3, 21, 10 and (A·B)^T gives
7, 9, 10.

## The Apple path stops guessing too

The ANE dispatcher took every memref argument, kept the last three, and called them
`[result, a, b]`. That convention is unfalsifiable for square operands, and the slot bit
made it actively dangerous: with slots now described as f32/rank-2, its layout check
started passing on arguments it would then have misread. It now uses the same decoder.
The 4x4 check that remains is a real constraint — the CoreML primitive is compiled for
that shape — rather than a heuristic standing in for information the compiler had all
along.

The three copies of the ABI-tag-to-libffi mapping, each under a comment asking that the
copies be kept in sync, became one (`runtime/vx_host_call.h`).

## Found, filed, not fixed

- **#331** — dispatch carries no device index, so a plugin cannot tell `Topology::GPU[0]`
  from `GPU[1]`. That is the whole of disaggregation, and the fix is one more payload
  entry. Nothing single-GPU needs it.
- **#332** — `vxc -c` fails the LLVM verifier on any program with a placed matmul
  (`DISubprogram attached to more than one function`). Found while planning how to run
  on a pod without a toolchain: compiling the object on the build box and shipping only
  that is the clean answer, and it does not work today. Every backend test JITs, so the
  AOT path has no coverage beyond trivial programs.

## Deployment

`scripts/make_gpu_bundle.sh` packs the compiler binary, four runtime files and the
programs to run into one archive; `scripts/setup_gpu_pod.sh` prepares the pod and builds
the dispatch library against *its* CUDA. No git history, no compiler source, and the
directory is removed after the run. `vxc` on Linux links LLVM statically and needs only
libc, libstdc++, libz and libzstd, so the binary itself travels. The one thing the pod
must install is LLVM's command-line tools, because JIT execution shells out to
`mlir-translate` and `clang++` — which is what #332 would remove the need for.

`VX_DISPATCH_LIB` overrides the dispatch library baked in at build time, since a
compiler copied onto another machine must be pointed at a backend built there.

## What the A100 is actually for

Everything above is checked. What is not, and cannot be without hardware:

1. cuBLAS produces the same numbers as the host path for `gpu_matmul_roles.vx`.
1. The staging and writeback work against real device memory.
1. Whether the per-dispatch H2D/D2H shows up as expected — it will be slower than the
   CPU at these sizes, which is why M2 (residency) exists and why no performance number
   is published before it lands.
