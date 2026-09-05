# Timing the two lowerings on real hardware

`compare.sh` shows what the two forms *compile to*. This shows what they *cost*,
which turned out to matter: the static numbers predicted a win that measurement
did not find. See
[#382 (comment)](https://github.com/hiraditya/Vx/issues/382#issuecomment-5366292395).

`bench.mlir` holds grid-strided versions of the same two kernels (one thread per
row, many rows per thread — how the flat path actually launches). `driver.cu`
loads the PTX through the CUDA driver API, runs both over identical inputs,
checks they agree, and times them.

On a CUDA box:

```bash
mlir-opt bench.mlir \
  --gpu-lower-to-nvvm-pipeline="cubin-chip=sm_80 cubin-features=+ptx76 cubin-format=isa" \
  -o nvvm.mlir
# extract the PTX blob from nvvm.mlir into bench.ptx (see compare.sh for the decode)
ptxas -arch=sm_80 -v bench.ptx -o /dev/null     # physical registers + spill
nvcc -O3 -arch=sm_80 driver.cu -o driver -lcuda
./driver <N-rows> <threads-per-block> <blocks>
```

## What it found on an A100-SXM4-80GB

`ptxas`: wide **42** registers, chunked **44**, zero spill in both. The chunked
form uses *more* — the opposite of what the virtual-register count suggested.

Size sweep at 432x256 (plenty of warps): **0.98-1.00x**, i.e. nothing. The dot is
memory-bound at 0.125 flop/byte and a GPU hides the dependency chain with
occupancy.

Occupancy sweep at 108 blocks (1/SM), where latency cannot be hidden:

| launch | wide | chunked | speedup |
|---|---|---|---|
| 108 x 32 | 1.069 ms | 1.005 ms | 1.064x |
| 108 x 64 | 0.642 ms | 0.598 ms | 1.072x |
| 108 x 128 | 0.428 ms | 0.392 ms | 1.092x |
| 108 x 256 | 0.336 ms | 0.338 ms | 0.994x |

So the chain does cost something, but only when there are too few warps to cover
it — and it disappears at bandwidth saturation (1599 GB/s is about A100 HBM peak).

`max|diff|` between the two outputs is 3.26e-09 over 1M dots: the reassociation
you would expect from changing summation order, and nothing more.
