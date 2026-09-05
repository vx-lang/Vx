# Flash attention on Apple silicon

Two files, making two different claims.

| file | what runs where |
|---|---|
| `flash_attention_split.vx` | every matmul on the Neural Engine, everything else on the CPU |
| `flash_attention.vx` | the whole algorithm in Vx, placed on the ANE, executed on the host shim |

## flash_attention_split.vx — two devices, both named

Each matmul of the flash inner loop is `spawn on(Topology::NPU[0])` and lands on
the Neural Engine; the online softmax is `spawn on(Topology::CPU)`. The split is
measured rather than chosen: CoreML puts a 512-square f16 matmul on the ANE and
a 512x512 softmax on the CPU, and a softmax only reaches the ANE at around 16M
elements or inside a matmul's own graph. `scripts/tools/ane_device_check.py` pins all
of that.

With a head dimension of 512 and a 512-key tile both matmuls are exactly 512
cubed, which is the shape the ANE route takes -- the algorithm was tiled to meet
the hardware, not the other way round.

```
source config.local
VX_DISPATCH_VERBOSE=1 ./target/debug/vxc \
    benchmarks/flash_attention_ane/flash_attention_split.vx --action run-jit
```

Four dispatches (two tiles, two matmuls each) and a max absolute error of 2.3e-5
against a host reference over four query rows.

Verified by breaking it: staging Kt untransposed moves the error to 4.1e-2 and
trips the assert.

# The Vx-native version, executed on the host

`flash_attention.vx` is fused attention with the algorithm in the program text,
placed on `Topology::ANE`. There is no builtin standing in for the kernel body
and no vendor model looked up behind the call: the compiler compiles what the
file says.

## Running it

```
source config.local
./target/debug/vxc benchmarks/flash_attention_ane/flash_attention.vx --action run-jit
```

The program checks itself. A materialized reference attention over the same
inputs runs on the host, and the two are compared elementwise, so the run fails
rather than printing a plausible number when the kernel is wrong.

## What it claims

Placement is declared and checked. Q, K and V move into `NPU_HBM`; a region on
`ANE` may read them because the topology says it sees that space; the result is
transferred home before the host reads it. Delete a staging transfer and the
compiler refuses the program — `tests/frontend/fail/ane_prefill_decode_missing_transfer.vx`
is that case.

The arithmetic runs on the host. Vx has no Neural Engine code generator, so
`spawn on(Topology::ANE)` states where the work belongs and the CPU shim
executes it. The timing the program prints is the host's, and it is there to
compare the two kernels against each other rather than to describe Neural Engine
throughput.

## The algorithm

Online softmax over K/V tiles of 32, with the running max, normalizer and output
accumulator carried across tiles. The `sq x sk` score matrix is never
materialized — only one tile of 32 scores exists at a time — which is what the
fusion is for and why this cannot be decomposed into a matmul/softmax/matmul
chain.

Rescaling is conditional: the `exp(m - m_new)` correction is applied only when
the running max grows past `tau = 8`. Exact either way, since a max that has not
grown by `tau` cannot overflow the exponential.

`exp_poly` is a software exponential — range-reduce to `2^n * 2^f`, degree-4
polynomial for the fraction — so the kernel calls no libm. It is declared
`on Topology::ANE`, which makes it a device-side helper: the host reference
cannot call it, and the compiler says so with E6001 if you try. The reference
uses libm instead, so the comparison checks the fusion and the polynomial at
once.

Q/K/V are f16 and the accumulator is f32, rounded once on store. That is what
flash attention does everywhere: the accumulator is where the precision is worth
paying for, and rounding once at the end beats rounding every tile.

## Sabotage-checked

The comparison was verified by breaking the kernel, not by watching it pass.
Dropping the scale from the score, and reading the wrong row of V, each fail the
run. Both are real bugs that a zero-filled input would have hidden, which is why
the operands are pseudo-random.
