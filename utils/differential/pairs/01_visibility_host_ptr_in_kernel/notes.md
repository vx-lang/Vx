# A device kernel reads host memory that nothing transferred

## The mistake

A value is produced on the host and read inside a device region, with no copy between the two.

## Vx

`tests/frontend/fail/memory_algebra.vx`. Refused at compile time, E6003, naming three things: where
the value lives, what the topology can see, and the transfer that would fix it.

```
'host_data' lives in CPU_DRAM, NPU[0] sees only [NPU_HBM]; insert an explicit transfer to NPU_HBM
```

The check is a set-membership test against the topology's declared `visible:` list. No solver is
involved.

## CUDA

Compiles. `const float *` is a pointer, and which memory it addresses is not part of its type, so
there is nothing for the front end to object to. The failure arrives when the kernel dereferences it.

## Why this one is first

It is the smallest case where the two toolchains disagree about *when* the error is available. Both
find it. Vx finds it before the machine is rented; CUDA finds it after the kernel launches, and the
error surfaces on the `cudaDeviceSynchronize` rather than at the launch, which is its own small
lesson about where a fault becomes visible.

## Reading the result

`sync=` carries the verdict. A non-zero exit of 1 means the kernel faulted, which is the expected
outcome. Exit 2 would mean it ran and returned the wrong number, which is worse and worth capturing
if it happens on some driver or card.
