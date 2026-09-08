# Topologies and memory

This is the chapter that makes Vx different from other systems languages. Everything up to here you
could have done in Rust or C++; none of it needed a new language.

## The two vocabularies

Vx describes a machine with two kinds of declaration, and they answer different questions:

- **`Memory`** — *where data lives.* `Memory::CPU_DRAM`, `Memory::NPU_HBM`, a scratchpad, a cache
  level. Each has a capacity, a bandwidth, and a scope.
- **`Topology`** — *where code runs.* `Topology::CPU`, `Topology::NPU[0]`, `Topology::GPU`. A
  topology is bound to the memory it can address.

A tensor's type records which memory space it is in. A region of code records which topology it runs
on. The checker's job is to make sure those two agree everywhere.

## Placement and transfer

Moving data between memory spaces is explicit:

```rust
let a = transfer(a_host, Memory::NPU_HBM);
let b = transfer(b_host, Memory::NPU_HBM);
```

`transfer` yields a value whose type says it lives in `NPU_HBM`. The original is still typed as
living where it was.

The explicitness is the point, and it is required **even when the hardware boundary costs nothing**.
On Apple's unified memory the CPU and the GPU address the same physical DRAM, so the copy compiles
away to nearly nothing — and you still write it. The reason is that data locality should be provable
by reading the source, not by profiling the binary. A `transfer` you cannot find in the text is a
transfer you cannot reason about.

Two things the compiler checks here:

- **Reachability** — there must be a declared path between the two spaces. A transfer between
  memories with no route between them is an error, not a runtime hang.
- **Admission** — the destination must have room. This is checked against the
  [machine file](machine-files.md), before any binary exists.

## Running code somewhere else

`spawn on` runs a block on a named topology:

```rust
fn main() -> i32 {
    let mut a_host : Tensor<f32, [4, 4]> = /* ... */;
    let mut b_host : Tensor<f32, [4, 4]> = /* ... */;
    let mut result_host : Tensor<f32, [4, 4]> = /* ... */;

    let a = transfer(a_host, Memory::NPU_HBM);
    let b = transfer(b_host, Memory::NPU_HBM);
    let mut result = transfer(result_host, Memory::NPU_HBM);

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    print(result);
    return 0;
}
```

The block is outlined into a kernel and handed to the dispatcher for that topology. On Apple Silicon
that means CoreML and the neural engine; on an NVIDIA box it means PTX. The source does not change
between the two — the machine file does.

Every value the block touches must already live in a memory the target topology can address. That is
why the three `transfer` calls come first. Skip one, and the error names the value and the space it
is in rather than crashing inside a vendor runtime.

## What gets rejected

The point of putting placement in the type system is the errors you get for free.

**Dereferencing a device pointer from the host.** A `Pinned<T, NPU_SRAM>` that escapes into a host
expression is a type error with a source span. This is the error that motivates the whole design: in
C++ with CUDA it is a segfault, and in Python it is a silent wrong answer.

**Working set overflow.** A tile you place in a scratchpad that cannot hold it is rejected at compile
time, with the required and available figures in the diagnostic. See
[machine files](machine-files.md).

**Use-after-move.** Buffers are linear values. Consuming one and then reading it again is an error.

**Unvisible transfers.** An asynchronous `transfer` whose completion has not been made visible before
the buffer is read is a *seam* violation. Turn the check on with `--verify-seams`; it discharges the
obligation with an SMT solver, and needs `z3` on your `PATH`.

## Verified values

`Verified<T>` marks a value whose computation carried its proof obligations all the way through. A
function returning `Verified<Tensor>` is asserting that the placement, capacity and visibility
conditions on the path that produced it were all discharged, not merely unchecked.

```rust
fn custom_matmul(a: Ref<Tensor, Memory::NPU_HBM>,
                 b: Ref<Tensor, Memory::NPU_HBM>) -> Verified<Tensor> {
    spawn on(Topology::NPU[0]) {
        let c = a * b;
        return Verified(c);
    }
}
```

## Choosing a machine at compile time

The same program compiles against different hardware by swapping the machine file:

```bash
vxc --machine fleet/h100-sxm.vx program.vx -o program
vxc --machine fleet/m4-uma.vx  program.vx -o program
```

You can ask what the compiler concluded, as JSON, rather than reading it out of diagnostics:

```bash
vxc --machine fleet/h100-sxm.vx program.vx --diagnostics-json out.json
```

That record carries every diagnostic with its structured fields — a capacity rejection includes the
space, the amount required, the amount available and the margin — plus the staging routes and
per-edge costs that an admitted program resolved to.

## Next

[Machine files](machine-files.md) covers how a real part gets described, and what the compiler
derives from that description.
