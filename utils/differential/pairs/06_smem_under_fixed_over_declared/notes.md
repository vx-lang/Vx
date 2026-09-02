# A 64 KiB tile the part has room for and CUDA will not declare statically

## Status: the CUDA half has not been run

The Vx half is verified. The CUDA half is written and has **not** been executed — the rented A100 was
released before this pair existed. Nothing below claims an observed CUDA result, and the row should
stay out of any table until it has one.

## The mistake, or rather the absence of one

This pair contains no mistake. It is the case where the fixed constant and the declared model
disagree, and the *program is fine*.

128 x 128 f32 = 65536 bytes. An A100 has 164 KiB of shared memory per SM, which
`fleet/a100-80.vx` declares. The tile fits the part with room to spare.

## Vx

Admits it. Compiles clean against `fleet/a100-80.vx`, no diagnostic:

```
$ vxc --host default --machine fleet/a100-80.vx vx.vx --action emit-mlir -o /dev/null
$ echo $?
0
```

## CUDA

Cannot say it statically. `nvcc` refuses any static `__shared__` array over 48 KiB whatever part it
is compiling for -- pair 05 has that refusal verbatim, `0xc000 max`, checked at `-arch=sm_80`. To use
shared memory the hardware already has, the program must switch to `extern __shared__` and opt in
with `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.

`cuda.cu` here is that rewrite. **Expected** to compile and run and print 8128; not yet observed.

## Why this pair matters more than pair 05

Pair 05 on its own reads as "CUDA catches something Vx also catches", which is true and dull. This
pair is the same ceiling seen from the other side, and it inverts the usual direction of the suite:
**Vx admits a correct program that CUDA's front end refuses.**

The difference is not strictness. It is what each toolchain checks against. `nvcc` checks a constant
compiled into it; Vx checks the number the machine model declares for the part in hand. When the
constant and the part disagree, the constant wins in CUDA, and the program is rewritten to work
around a limit the hardware does not have.

Run 05 and 06 next to each other or neither. Alone, 05 flatters CUDA and 06 flatters Vx; together
they say the accurate thing, which is that a fixed ceiling is wrong in both directions and a
declared one is not.

## To finish it

On an A100-80: `NVCC=/usr/local/cuda/bin/nvcc ./run.sh --cuda-only 06`. If it prints
`out=8128.000000` and exits 0, the hardware has the room and pair 05's refusal was about a compiler
constant. If the launch fails instead, this pair is wrong and should be deleted rather than
explained.
