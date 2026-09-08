# Introduction

Vx (pronounced *"vee-ex"*) is a systems programming language for machines that are no longer a
single processor. A modern node is a CPU, one or more GPUs, sometimes a neural accelerator, and a
memory hierarchy with half a dozen distinct spaces in it — each with its own capacity, its own
bandwidth, and its own rules about who is allowed to read it.

Most languages treat that hardware as *infrastructure*. You write the computation, and a large
runtime library decides where it lands. The result is that a whole class of mistake — reading device
memory from the host, exceeding the capacity of a scratchpad, using a buffer whose copy has not
landed yet — is discovered at runtime, if you are lucky, and silently tolerated if you are not.

Vx takes the opposite position:

> Heterogeneity belongs in the type system, not in the runtime.

A pointer into accelerator memory has a different type from a pointer into host DRAM. Crossing
between them requires an explicit `transfer()`. A host thread dereferencing a device pointer is a
compile error with a source span, not a segfault in production.

## What that buys you

The compiler front-loads into type checking a set of bugs that normally surface much later:

| Check | What it rules out |
| --- | --- |
| Address-space typing | Dereferencing a device pointer from the host |
| Capacity admission | A placement whose working set cannot fit the space it targets |
| Seam contracts | Reading a buffer whose asynchronous transfer has not been made visible |
| Linear types | Use-after-move of a consumed buffer |
| Borrow checking | Aliasing and lifetime errors, with region tracking |
| Topology reachability | A transfer between spaces with no declared path between them |

Capacity admission is worth dwelling on, because it is the one with no equivalent elsewhere. Vx
reads a **machine file** describing a real part — its memory hierarchy, capacities, bandwidths and
interconnect — and checks your placements against it *before a binary exists*. An allocation that
cannot fit in the scratchpad you assigned it to is a compile error, not an out-of-memory at training
step 1200.

## Who this is for

Vx is aimed at the layer underneath the machine-learning stack: runtimes, kernels, schedulers,
inference engines, and the systems code that has to be correct across several kinds of silicon at
once.

It is deliberately not aimed at exploratory work. PyTorch users mutate a model mid-loop, print a
tensor's shape, branch on it and carry on. In Vx — ahead-of-time compiled, statically regioned —
that same dynamism takes real effort.

**Vx is the right language for the thing that must be correct and fast across ten kinds of silicon.
It is not the right language for the thing you are still figuring out.**

## How to read this book

If you want to run something in the next ten minutes, go to [Install Vx](getting-started.md) and
then [Your first program](first-program.md).

If you want to know whether the language is worth your time before installing anything, read
[A tour of Vx](tour.md), which covers the whole language with no accelerator involved, and then
[Topologies and memory](heterogeneous.md), which is the part that is actually different.

If you are evaluating Vx for a real system, [Machine files](machine-files.md) is the chapter that
will tell you fastest whether the model matches your hardware.

## Project status

Vx is young and under active development. The language, the type checker and the MLIR optimization
pipeline are all still moving. Expect sharp edges, expect syntax to change, and please
[report what you hit](https://github.com/vx-lang/Vx/issues) — early bug reports are the most useful
thing you can contribute right now.
