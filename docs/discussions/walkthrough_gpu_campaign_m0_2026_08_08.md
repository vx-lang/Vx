# Walkthrough: M0 of the GPU/disaggregation campaign (2026-08-08)

Session journal for #321. Covers getting Vx to run placed programs on Linux,
making Llama numerically correct, extending the dispatch ABI, and greening the
Linux test baseline. Written as a record of what was *learned*, not only what
was changed — several documented facts turned out to be wrong, and how they were
wrong is the transferable part.

Commits, in order: 32899aaa, 5e83aa6a, 997d3d87, 9546c5e9, aa8f1c64, b02479ce,
ef9032f8, e5954840, be776772, 37794df0.

Issues filed: #321 (umbrella), #322, #323, #324, #325, #326. Corrected: #320.

## 1. The starting picture, and what it got wrong

Three surveys of the tree established the split cleanly: the *compile-time* half
of disaggregation is real and tested — placement, capacity admission
(E6009/E6010), routed and costed transfers, and a prefill/decode modelling
precedent in `tests/frontend/pass/rubin_disaggregated.vx` — while the *execution*
half is host-CPU only.

#319 had scoped the single-GPU work as "a runtime plugin plus a program". That
framing was right about the CUDA work and wrong about the platform underneath it:

**There was no provider for `vx_plugin_dispatch_async` on Linux at all.**
`build.rs` compiled the dispatcher only on macOS, so any program containing a
non-CPU `spawn on` failed to *link* off macOS. Nobody had noticed because the
placed programs in the corpus are compile-only tests.

## 2. Four defects between here and a running placed program

In order of discovery, on a rented-then-provisioned x86 Linux box:

1. **No dispatch shim.** `runtime/host_dispatch.cpp` now implements the whole
   `vx_plugin_*` ABI portably, running outlined kernels on the host through
   libffi. It is deliberately the file the CUDA plugin *specializes* rather than
   a throwaway: M1 starts from working C++ instead of Objective-C++.

1. **`jit.rs` hardcoded a Homebrew `llvm-config` path.** Resolves through PATH
   now, which is what `build.rs` already did.

1. **`-rdynamic`.** The dispatcher finds outlined kernels with
   `dlsym(RTLD_DEFAULT, ...)`. Mach-O exports those symbols; ELF does not unless
   asked. Without it the lookup failed at run time, after a clean build and link.

1. **The `@malloc_N` bug**, which is the interesting one. An unused `extern`
   declaration lands as `func.func private @malloc`. `finalize-memref-to-llvm`
   needs a malloc for `memref.alloc`, cannot reuse that one because it looks for
   an `llvm.func`, creates its own, and the symbol table uniques the name. One
   dangling `@malloc_0`, `@malloc_1`, ... per allocation site, ~21 of them,
   none resolvable at link.

   Merely `import std::vec` was enough to trigger it. **This is why no Llama
   program had ever linked.** Fixed by running `symbol-dce` before memref
   finalization — and note it had to go in *both* pipelines, because the driver
   builds its own, separate from the one in `codegen/mod.rs`.

   The general lesson: a compile-only test cannot catch a link failure, and the
   corpus was full of compile-only tests.

## 3. Llama: fluent, and wrong in four ways

With linking fixed, `tests/backend/pass/llama2.vx` executed for the first time in
its life and produced fluent English that degenerated into "She was so happy."
repeated to the token limit. Four defects, each independently fatal:

1. **The attention softmax was not a softmax.** `val - max_val.exp()` parses as
   `val - exp(max_val)` — the method call binds tighter than the subtraction. The
   correctly-written `softmax()` helper sits in the same file, never called from
   the attention path.

1. **RoPE used the global index** rather than the index within the head, so the
   frequency ladder — which should restart every head — ran off the end for every
   head past the first. `examples/llama.vx` and `llama2_v2.vx` compute the
   modulo correctly; this file did not.

1. **Attention scale hardcoded** to `1/sqrt(32)` with the comment `256/8 = 32`.
   stories15M is dim=288, n_heads=6, so head_size is 48.

1. **RMSNorm divided by a hardcoded 256.0** against an actual dim of 288, in both
   the per-layer and final calls. Its comment claimed Vx had no `as f32` cast; it
   has had one since #240, and `main()` was already using it a few lines away.

Defects 3 and 4 share a cause worth naming: **constants transcribed for a model
the code was no longer running.**

### Establishing parity, and why from BOS

Fixed, the program matches Karpathy's llama2.c **token for token — 64/64
identical ids** on the same weights with greedy sampling.

The comparison starts from BOS rather than from a prompt *by necessity*.
`vx_encode_prompt` tokenizes character-by-character with no BPE merge loop, while
llama2.c merges pairs by score, so the prompts tokenize differently and no
prompted comparison could ever match regardless of the transformer. Starting from
BOS isolates the forward pass from the tokenizer. Filed as #323.

A patched llama2.c tracer lives on the build box (`~/ref/`), emitting per-step
token ids and first-step logits, and able to take explicit token IDs (`-T`) to
bypass BPE when prompted parity is wanted later.

## 4. Two documented blockers that dissolved, and one that did not

**#320 was wrong.** It reported f16/bf16/i8 as declaration-only and unable to
execute, and put the dtype work *ahead of cuBLAS wiring* on the grounds that
serving in f32 costs 2x the memory bandwidth. Measured: f16 scalar arithmetic,
f16 tensor store/load, and a full f16 `@` matmul all execute and give right
answers. Only `print()` of a bare narrow scalar failed — a diagnostics gap. The
2x penalty was never being paid by the compute path.

**But not entirely.** Probing further while filing follow-ups showed that
indexing a `Tensor<f16>` yields `f32`, so a hand-rolled half-precision
accumulate loop cannot be written at all (#324). So f16 *compute* works while
f16 *authoring* does not — narrower than #320 claimed, wider than my correction
on it implied. Both statements needed care.

**`llama2_v2.vx` is not the serving base** #319 recommended: it prints pointers
instead of text, because printing a `String` emits a compile-time warning and
then silently falls through to `print_i32` (#323).

The pattern across all three: reported from reading, not from running. The
three-line probe is cheap.

## 5. The real M1 prerequisite

Starting `cuda_dispatch.cpp` stopped at the design stage. `abiTagForType`
described every memref as tag `0` plus an opaque descriptor pointer — element
type, rank and shape all erased. cuBLAS picks its kernel by dtype and needs
M/N/K, so library routing was **unwritable** against that boundary.
`npu_dispatch.h` only appears to cope by hardcoding `float *`, rank 2, and a 4x4
comparison against a descriptor whose layout it has already assumed.

This also made #319's "route by the outlined kernel's structure" impossible as
written: *the structure was not in the message.*

ef9032f8 packs element type and rank into unused bytes of the tag, leaving the
low byte — and therefore the calling convention — untouched. Shapes are
deliberately not transmitted: they already live in the descriptor, and rank is
what makes them readable.

It closed a latent misread as well as unblocking work: the Apple path
reinterpreted *any* tag-0 argument through `MemRef2D`, including opaque pointers
(also tag 0) and f16 buffers (which do execute). The layout is now checked before
the cast, with argument *selection* deliberately left alone so the only
behavioural change is declining to the CPU fallback instead of reading whatever
the bytes happened to be.

Next step is #325: recognise the op in the compiler rather than guessing at
shapes, because shape conformance cannot disambiguate square matrices and
getting operands backwards yields a plausible matrix of wrong numbers.

## 6. Tests, and a test that tested nothing

Everything above had been verified by hand-run probes — evidence that something
worked once, not that the suite would notice it breaking. Four tests now cover
it (e5954840), one per claim.

The ABI test was first written under `middle_end/` **and passed while checking
nothing**. Caught by negative control: corrupting the expected value still
passed. Two independent reasons — that runner reads only plain `// CHECK:` and
silently drops FileCheck's other directives, and it inspects the module *before*
the lowering pipeline, so a constant materialized by `vx-to-llvm` could never
appear there.

Both were worth fixing rather than working around. The test moved to
`optimizations/`, where the RUN line goes through real FileCheck; and
`run_middle_end_test` now *rejects* a `CHECK-` directive it does not implement
and says where such a test belongs. **Always run the negative control.**

## 7. The Linux baseline was red for reasons unrelated to Linux

Running the full suite on Linux for the first time: 5 failures on clean `main`.
My first hypothesis — the in-flight memalg work, which the build tree carried —
was **wrong**; a clean-main tree failed identically.

- **Four were one missing package.** Seam verification shells out to the z3
  *binary* (`Command::new("z3")`), and `libz3-dev` does not provide it. Every
  failing case involved a `relaxed` transfer edge. The symptoms — a missing
  W1027, an expected-failure that passes — read exactly like compiler
  regressions.
- **One is genuinely macOS-only.** `topology_dispatch.vx` checks for
  `{plugin = "Apple_NPE_v1"}`, emitted only where the Apple plugin is registered
  (`#[cfg(target_os = "macos")]`). Now `// REQUIRES: macos`, with the
  optimizations runner honouring the gate the other two runners already did.

Both platforms now report **183 passed, 0 failed**. This matters more than it
sounds: the campaign builds and runs on Linux, and a red baseline cannot detect a
regression. `scripts/setup_linux.sh` records the provisioning so the next box
does not repeat the z3 detour.

## 8. Working practices worth keeping

- **The index holds pre-staged work.** Committing without a pathspec swept
  unrelated staged files into a commit, twice (once via `--amend`). Always
  `git commit -- <paths>`, and never `--amend` here.
- **Branches can move under a long session.** An ABI commit landed on a branch
  created mid-session; cherry-picked to `main` and the branch restored with
  `git branch -f`. Check `git branch --show-current` before committing.
- **Build on the trusted box, ship artifacts.** Sources do not go to rented
  hosts; the tarball is built from `git ls-files`, and macOS `tar` needs
  `COPYFILE_DISABLE=1` or it smuggles `._*` AppleDouble files that the test
  harness then tries to parse as tests.

## 9. State at the end of the session

Done: M0 complete except promoting the Llama to an executing CI test, which
needs a per-test environment mechanism (the `EXPECT:` path JITs in-process and
cannot set `LLAMA_TOKENS_CONFIG` the way a `// RUN:` line can). The dispatch ABI
carries dtype and rank. Both platforms green.

Next, none of it needing a GPU: #325 (compiler-side op recognition), then
`cuda_dispatch.cpp`. The A100 is worth renting when the transfer path
(`cudaMalloc`/`cudaMemcpy`) is buildable, since that is the first thing that
genuinely needs silicon. The build box already has the CUDA 13.2 toolkit with
driver stubs, so the plugin links there without one.
