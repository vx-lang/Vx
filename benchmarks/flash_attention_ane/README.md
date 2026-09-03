# Flash attention on Apple silicon, written in Vx

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
flash attention does everywhere, and it is currently also the only mixed form
the flat emitter handles: an f16 row slice times a scalar emits MLIR that does
not parse.

## Sabotage-checked

The comparison was verified by breaking the kernel, not by watching it pass.
Dropping the scale from the score, and reading the wrong row of V, each fail the
run. Both are real bugs that a zero-filled input would have hidden, which is why
the operands are pseudo-random.
