# Host code reads device memory with no copy back

## The mistake

The reverse of pair 01. A value is produced on the device and read on the host, with no transfer
back.

## Vx

`tests/frontend/fail/host_reads_gpu_memory.vx`, added with this suite because the corpus covered the
host-to-device direction and not this one.

```
Error: Cross-topology access error: Cannot access Pinned type on GPU(..) from CPU
```

Two things about that diagnostic are worth stating in any write-up that quotes it, because a
reviewer who runs the compiler will see them:

- **It carries no error code.** It prints as a bare `Error:`, not `Error[E6003]`, so it does not
  appear under any code in a taxonomy. This is the same class of gap as E6002, which is declared and
  never emitted.
- **It leaks internal formatting.** The topology is printed via its `Debug` impl, so the message
  contains `GPU(Number(NumberExpr { value: "0", ty: None, span: Span { .. } }))` rather than
  `Topology::GPU[0]`.

The rule underneath is right and the refusal is correct. The presentation is not, and the `CHECK`
lines in the test deliberately pin only the stable half of the message.

## CUDA

Compiles. `float *` says nothing about which memory it addresses, so the host dereference is
ordinary code. Expected outcome is a SIGSEGV rather than a CUDA error, since the fault is taken by
the host and never reaches the driver: the runner records the signal.

## Why the device is a GPU and not an NPU

The built-in CPU descriptor's visibility is `[CPU_DRAM, NPU_HBM]`. On an Apple part the host and the
NPU share one physical memory, so a host read of NPU_HBM is legal and correct, and the compiler
allows it. Using an NPU here would have produced a Vx program that compiles, and a pairing that
proves nothing. A discrete GPU's memory is not in that list, which is what makes this a mistake.
