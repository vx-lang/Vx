# A tensor library for an inference pipeline

**Status:** proposal, 2026-10-02; §3 revised the same day. Tracking issue: Vx#992.
**Scope:** `std::tensor` and the compiler work it needs, up to running a Llama-class model end to
end. Tokenizers and serving are outside this plan except where they constrain the library.

______________________________________________________________________

## 0. Summary

The target is the tensor library an industrial inference pipeline needs: the one that would let
someone write a Llama 3.2 1B decoder in Vx and compare it with llama.cpp on the same machine, for
correctness (perplexity) and speed (tokens per second).

`std::tensor` today is 1,417 lines and 36 methods, all on rank-1 and rank-2 tensors. Most methods
are hand-written vector loops with a fixed width (16 lanes for `f32`, 8 for `f64`). Only `fill`,
`assign`, `copy` and `compare` are written in linalg.

The decisions this document makes:

1. **Library operations are written in Vx, with a few tensor operations the compiler lowers to
   linalg.** `map` is already lowered this way; `zip`, `reduce`, `reduce_axis` and broadcasting
   arithmetic are added (§3). A method written with them is type-checked and borrow-checked, and
   works for every element type and rank without a copy per type. `mlir!` stays as the escape
   hatch for what these cannot say, such as a hardware intrinsic.
1. **The compiler makes linalg fast.** Tiling, vectorization and fusion are compiler passes, not
   code in each method. A hand-written kernel is allowed only where a benchmark shows the
   generated version falling short (§6).
1. **The linalg the compiler emits works on tensor values, and buffers are assigned afterwards.**
   MLIR's fusion works best before bufferization. Vx emits linalg on memrefs today, which is why
   fusing it needed an alias guard (#984). Moving to tensor values and one-shot bufferization is
   the largest compiler change here (§4).
1. **The first target is Llama 3.2 1B on an Apple Silicon CPU.** It is the hardware we have, and
   the ANE dispatch path already exists for the later device phase.
1. **Every operation is tested against PyTorch or NumPy values**, with a tolerance per element
   type (§7).

## 1. What exists today

Measured on main at 781b8c54.

| Area | State |
|---|---|
| Ranks | Rank-3 and rank-4 tensors compile and index correctly. Every library method is rank 1 or rank 2, written twice. |
| Element types | `f32`, `f64`, `f16`, `bf16` in the library. `i8`, `f8e4m3`, `f8e5m2` and `f4e2m1` exist in the type checker but have no library methods. |
| Methods | Element-wise in place, reductions (`sum`, `dot`, `mean`, `norm`, `max`, `min`, `argmax`, `argmin`, `variance`, `std_dev`), reductions along one axis of a rank-2 tensor, `softmax_inplace`, `fill`/`copy`/`compare`, views from a pointer and slices. |
| Built into the compiler | `@` (matrix multiply) and indexing. |
| Optimization | At `-O1` and above, linalg is fused with the loops around it, for static shapes only (#984). Otherwise it is lowered to loops and LLVM vectorizes what it can. The hand-written vector loops are not fused with anything. |
| Measured | A `linalg.generic` reduction with `fastmath<reassoc>` on its addition runs within 2% (`sum`) and 6% (`dot`) of the hand-written vector loops, and about 9x faster than an in-order loop (`benchmarks/stdlib/reduce`, #1033, Apple Silicon). Without `reassoc` it is as slow as the loop: floating-point addition may not be reordered unless the program allows it. |
| Devices | ANE dispatch for some matmul shapes; host loops split across worker threads. |

Open issues this plan depends on: #400 (views), #406 (layout), #429 (placement in the type),
#245 (bounded dynamic shapes), #404 and #328 (rank bugs), #466 (lost 3-D/4-D coverage), #924
(reductions and element-wise work already planned), #951 (moving slice arithmetic out of the
compiler).

## 2. What an inference pipeline needs

### 2.1 The tensor type

- Any rank, with strides. Slice, transpose, broadcast and reshape return views that do not copy.
- Owned tensors and borrowed views as separate types, with Rust-style lifetimes (#400).
- Element types: `f32`, `f16` and `bf16` for compute; `f8` and packed `i8`/`i4` with per-group
  scales for weights; `i32`/`i64` for indices; `bool` for masks.
- Dynamic sizes with an upper bound. Batch size and sequence length change at run time, and the
  KV cache has a fixed maximum (#245).
- Placement in the type, so a CPU tensor cannot be passed where a GPU tensor is expected (#429).

### 2.2 Operations

In the order a transformer decoder uses them:

| Area | Operations |
|---|---|
| Matrix multiply | GEMM, batched GEMM, GEMV; bias or activation fused after it; `f16`/`bf16` inputs accumulated in `f32`; quantized matmul that dequantizes `i4`/`i8` weights inside the loop. |
| Attention | Scaled dot product with a causal mask; tiled attention with an online softmax (the flash-attention scheme); grouped-query attention; RoPE; reading keys and values from a paged KV cache. |
| Normalization and activations | RMSNorm, LayerNorm, SiLU, GELU, SwiGLU, softmax and log-softmax that do not overflow. |
| Element-wise and reductions | Arithmetic with broadcasting; reductions along any axis. |
| Data movement | reshape, permute, concat, split, slice, pad, gather (embedding lookup), scatter (writing new keys and values into the cache). |
| Sampling | argmax, top-k, top-p, temperature, using `std::rand`. |
| Later | conv2d and pooling, for vision models. |

### 2.3 Memory

- A memory planner. Tensor lifetimes are known at compile time, so buffers whose lifetimes do not
  overlap can share storage.
- Operations that write in place, and a scratch allocator for temporaries.
- A KV cache made of fixed-size pages, so sequences of different lengths share one pool.
- Weights memory-mapped from safetensors or GGUF files, without copying.

## 3. Tensor operations the compiler lowers

The compiler cannot find a fast reduction in an ordinary loop. `for i in 0..n { s += a[i]; }` adds
in order, and floating-point addition gives a different answer in another order, so the compiler
may not vectorize it. Recognizing the pattern does not change that: permission to reorder has to
come from the program. Turning general loops into linalg is also hard in itself, because of
aliasing, early exits and side effects.

So the language gets a few operations whose meaning includes that permission, and the code
generator lowers each one to a `linalg.generic`, as it already does for `map`:

| Operation | Meaning | Lowers to |
|---|---|---|
| `t.map(f)` (exists) | `f` applied to every element | a parallel `linalg.generic` |
| `a.zip(b)` | pairs of elements at the same position, for `map` or `reduce` | one `linalg.generic` with two inputs |
| `t.reduce(init, f)` | every element combined with `f`, in any order | a reduction `linalg.generic`, with `fastmath<reassoc>` on floating-point arithmetic in `f` |
| `t.reduce_axis(axis, init, f)` | the same along one axis | a `linalg.generic` with one reduction dimension |
| `a + b` etc. on tensors | element-wise, with broadcasting | a parallel `linalg.generic` |

Written with them, `sum` and `dot` are one line each, for every element type and rank. The
`[..]` (a tensor of any rank) is proposed in phase 0 and does not exist yet:

<!-- vx-doctest: skip -- proposed syntax, not yet compilable -->

```vx
fn sum(self : &Tensor<T, [..]>) -> T {
  return self.reduce(0 as T, | acc, x | acc + x);
}
fn dot(self : &Tensor<T, [..]>, other : &Tensor<T, [..]>) -> T {
  return self.zip(other).reduce(0 as T, | acc, (x, y) | acc + x * y);
}
```

`reduce` may combine elements in any order, so `f` has to give the same answer in any order:
addition, multiplication, `max` and `min` do, subtraction does not. For floating-point values the
answer can differ in its last bits from an in-order sum, which is the same trade C++ makes with
`std::reduce`.

**Done when:** `sum` and `dot` written with `reduce` run within about 10% of today's hand-written
vector loops (the linalg versions in `benchmarks/stdlib/reduce` already do), and a chain of three
element-wise operations becomes one loop.

## 4. The compiler pipeline

These operations only pay off if the compiler turns linalg into fast code. Today it lowers linalg
to loops, and LLVM vectorizes what it can.

The pipeline this plan needs:

1. The operations in §3 produce linalg on tensor values (`tensor<?x?xf32>`), not memrefs.
1. After inlining, `linalg-fuse-elementwise-ops` merges chains of element-wise operations, and
   tile-and-fuse (through the transform dialect) tiles matmul and attention and fuses their
   producers and consumers into the tiles.
1. Tiles are vectorized with `transform.structured.vectorize`, with vector sizes picked per target
   instead of written in the library.
1. One-shot bufferization assigns buffers, reusing them in place where it can.
1. The existing lowering to LLVM runs on the result.

Fusion on tensor values does not have the alias problem #984 had to guard against, because tensor
values cannot alias.

**Done when:** the measurements in §3 hold at every rank, and with shapes known only at run time.

## 5. Phases

Each phase gets its own issue under Vx#992 when work on it starts.

| Phase | Work | Done when |
|---|---|---|
| **0. Foundations** | One generic method body for every rank instead of separate rank-1 and rank-2 copies; strided views (#400); broadcasting rules; explicit layout (#406). Fix #404 and #328; restore 3-D and 4-D coverage (#466). | One method body works for ranks 1 to 4, and slicing and transposing do not copy. |
| **1. Tensor operations and the pipeline** | §3 and §4: `zip`, `reduce`, `reduce_axis` and broadcasting arithmetic lowered to linalg, then the tensor-value pipeline. Rewrite #924's methods with them. | The §3 and §4 measurements. |
| **2. One transformer layer in f32** | matmul and batched matmul, RMSNorm, RoPE, SiLU/SwiGLU, softmax, attention with a causal mask, gather, permute, concat. | One Llama decoder layer matches PyTorch within 1e-5. |
| **3. Low precision and quantization** | `f16`/`bf16` with `f32` accumulation; weights stored as `i8`/`i4` with group scales, dequantized inside the matmul. | Perplexity of a 4-bit model is within 1% of llama.cpp at the same quantization. |
| **4. Inference runtime** | safetensors/GGUF loader, paged KV cache, memory planner, sampling, batching. A tokenizer, outside this library. | Llama 3.2 1B generates text end to end; tokens per second reported against llama.cpp on the same machine. |
| **5. Devices** | Multi-threaded CPU, then GPU (Metal) and ANE, with placement in the type (#429, #344, #350). | The same model code runs on each device. |

Phases 0 and 1 are the slow, uncertain part. Once the pipeline exists, phases 2 to 4 are mostly
library code.

## 6. Hand-written kernels

A method may use a hand-written `mlir!` kernel (explicit `vector` ops, a fixed tile size) only when:

1. a benchmark in `benchmarks/stdlib` shows the version written with §3's operations more than
   10% slower on the target machine, and
1. that version stays in the tree as the reference the kernel is tested against.

Today's vector loops for `sum`, `dot`, `max`, `min`, `variance` and `std_dev` are the first to be
rewritten with `reduce` and measured against.

## 7. Testing

- Every operation is checked against values computed with PyTorch or NumPy, at several shapes
  including ones that are not a multiple of the vector width.
- Tolerances are set per element type: tight for `f32`, looser for `f16`, `bf16` and quantized
  types.
- An option makes reductions add in a fixed order, so results can be reproduced exactly.
- Benchmarks compare against llama.cpp and Apple Accelerate on the same machine.
- One end-to-end test runs the full model and checks perplexity on a fixed text.

## 8. Open questions

1. How the tensor-value linalg of §4 meets code that still uses memrefs: Vx references are memrefs
   today, so a write through a `&mut` has to come back out as a write to that buffer.
1. Whether `reduce` should refuse a combining function that is not associative, such as
   subtraction, or only document the rule.
1. How bounded dynamic shapes (#245) are written in the type, and how a method states the bound it
   needs.
1. Which quantization formats to support first: GGUF's `Q4_K` and `Q8_0` match llama.cpp, while
   per-group `i4` with `f16` scales matches GPTQ and AWQ checkpoints.
