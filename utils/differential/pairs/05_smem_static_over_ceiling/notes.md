# A shared-memory tile over the on-chip ceiling

## The mistake

A tile staged into on-chip scratchpad that the scratchpad cannot hold.

## Vx

`tests/frontend/fail/flash_attention_tile_too_big.vx`. A 128x64 f32 query tile is 32768 bytes,
staged into a `Local_SRAM` declared at 16 KiB, and E6009 refuses it with both numbers.

## CUDA

**Rejected at compile time.** A static `__shared__` array over the 48 KiB per-block limit is a
front-end error, not a run-time one. Going past that ceiling requires the dynamic
`extern __shared__` form together with an opt-in `cudaFuncSetAttribute` call, and `nvcc` will not
let the static spelling through.

## Why a case CUDA catches belongs in the suite

Because the honest claim is narrower than "CUDA misses placement errors", and a table where every
row goes one way is a table someone selected. CUDA has a real static check here, and it is a good
one.

What is different is the bound each toolchain checks against. `nvcc` checks a fixed architectural
constant -- the 48 KiB static limit built into the compiler. Vx checks the capacity **the machine
model declares**, which is 164 KiB on `fleet/a100-40.vx`, 16 KiB in the test above, and whatever the
part actually has on any other file. A program that fits 48 KiB and does not fit the scratchpad of
the part it will run on is accepted by `nvcc` and refused here.

So the distinction this pair draws is not "who catches it" but "against which number". That is a
smaller claim than pairs 01, 02 and 04 support, and it is the true one for this case.

## A stronger variant, not built

The sharper version of this pair would sit between the two ceilings: a tile under 48 KiB, over the
declared scratchpad of a specific part. `nvcc` would accept it and the kernel would launch; Vx would
refuse it against that part's model. That needs a machine file whose SMEM is below 48 KiB and a
kernel written to use it, which is more than this pass of the suite covers. Worth building if the
evaluation leans on this row.
