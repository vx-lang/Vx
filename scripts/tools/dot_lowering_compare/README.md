# What `dot()` compiles to, and what Vx#382 would change

> **Measured follow-up:** the register argument this example was built to
> demonstrate does **not** survive `ptxas` — see [BENCH.md](BENCH.md) and
> [#382 (comment)](https://github.com/hiraditya/Vx/issues/382#issuecomment-5366292395).
> Physical allocation is 34 registers (wide) vs 44 (chunked), zero spill in both;
> the 257 below is PTX *virtual* register numbering, not pressure. The
> chain-depth and FMA differences are real, and worth ~1.09x at low occupancy
> only. Kept as-is because the compile-side comparison is still the clearest way
> to see what the two lowerings differ in.

A runnable, self-contained demonstration of the problem behind
[#382](https://github.com/hiraditya/Vx/issues/382). No GPU required — it needs
only `mlir-opt` (`source config.local`).

```
./compare.sh          # defaults to sm_80
```

## The two kernels

`two_dots.mlir` holds two `gpu.func`s that compute **exactly the same thing** —
`o[i] = dot(q[i], k[i])` over 64-wide f32 rows. The only difference is the shape
of the reduction.

**`@dot_wide`** — what `src/codegen/flat.rs:2235` emits today. Load each whole row
as one `vector<64xf32>`, multiply, reduce:

```mlir
%a = vector.load %q[%i, %c0] {alignment = 16 : i64} : memref<128x64xf32>, vector<64xf32>
%b = vector.load %k[%i, %c0] {alignment = 16 : i64} : memref<128x64xf32>, vector<64xf32>
%p = arith.mulf %a, %b : vector<64xf32>
%s = vector.reduction <add>, %p : vector<64xf32> into f32
```

**`@dot_chunked`** — what #382 proposes. Walk the row in `vector<8xf32>` chunks,
accumulate with `vector.fma`, reduce the 8-lane accumulator once at the end:

```mlir
%acc = scf.for %j = %c0 to %c64 step %c8 iter_args(%a = %z) -> (vector<8xf32>) {
  %x = vector.load %q[%i, %j] {alignment = 16 : i64} : memref<128x64xf32>, vector<8xf32>
  %y = vector.load %k[%i, %j] {alignment = 16 : i64} : memref<128x64xf32>, vector<8xf32>
  %n = vector.fma %x, %y, %a : vector<8xf32>
  scf.yield %n : vector<8xf32>
}
%s = vector.reduction <add>, %acc : vector<8xf32> into f32
```

## What it prints

Both go through `--gpu-lower-to-nvvm-pipeline`, the same pipeline `deviceImageOf`
uses in `src/dialect/VxLowering.cpp`, then the script counts what matters:

```
                         dot_wide dot_chunked
ld.global.v4                   32          4
mul.rn.f32                     64          0
add.rn.f32                     64          8
fma.rn.f32                      0          8
b32 registers                 257         33
dependent FP chain             65          9
```

## How to read that

**`b32 registers`: 257 vs 33.** The wide form issues all 32 vector loads *before*
the first multiply — check the order yourself in `dot_wide.ptx` — so 128 loaded
values are live at once. For one scalar result. This is what pins the split-K
attention kernel to the 255-register architectural ceiling with spill.

**`dependent FP chain`: 65 vs 9.** This is the one that matters most, and it is
not about registers at all. Look at the adds in `dot_wide.ptx`:

```
add.rn.f32 %r198, %r197, 0f00000000;
add.rn.f32 %r199, %r198, %r196;     // consumes the previous add
add.rn.f32 %r200, %r199, %r195;     // and again, 64 deep
```

`vector.reduction <add>` lowers to a **strictly sequential** sum, because f32
addition is not reassociable without fast-math. That is a 64-long dependency
chain of ~4-cycle instructions — hundreds of cycles no scheduler can hide. The
chunked form's `vector<8xf32>` accumulator is 8 *independent* chains, so the
depth drops to 8 plus a final 8-lane reduce.

**`fma.rn.f32`: 0 vs 8.** Same strict-ordering reason: the wide form cannot
contract its separate `mul` and `add` into `fma`. Chunked accumulation makes
the contraction explicit, halving the arithmetic instruction count.

The script computes the chain by walking `add`/`mul`/`fma` operands and taking
the longest def-use path, so it measures the real thing rather than trusting
instruction counts.

## Why this retro-explains the #378 ILP results

The campaign measured that **two** softmax chains beat one (1.497 → 1.087 ms) and
that **four** were ~3× slower. Both now have one explanation: two independent
64-deep chains interleave and hide each other's latency, while four accumulator
rows exhaust the registers. Extra chains were buying chain-parallelism with the
resource that was already scarce. Chunking buys the same parallelism *inside* a
single dot while **lowering** register pressure instead of raising it.

## A gotcha this example found

The `{alignment = 16 : i64}` on every `vector.load` is load-bearing. Without it,
MLIR assumes the element's natural 4-byte alignment, LLVM cannot form 128-bit
loads, and the row scalarizes into **128 separate `ld.global.b32`** — no v4 loads
at all. `flat.rs` gets this right via `vector_align_attr`; it is the same class of
bug as the `.shared` alignment fault fixed earlier in the #378 campaign.

## Not covered here

`ptxas -v` gives *physical* register allocation and spill bytes, which is what
#382's acceptance criteria are written against. It needs a CUDA toolchain, so it
is not part of this script. On a CUDA box:

```
ptxas -arch=sm_80 -v dot_wide.ptx -o /dev/null
ptxas -arch=sm_80 -v dot_chunked.ptx -o /dev/null
```

The counts above are PTX *virtual* registers — strong evidence for the mechanism,
but the 255-plus-spill figure quoted in #382 comes from `ptxas -v` on the real
split-K kernel, not from this toy.
