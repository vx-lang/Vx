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
    let mut host : Tensor<f32, [4, 4]> = unsafe { Tensor<f32, [4, 4]>::uninit() };
    for i in 0..4 {
        for j in 0..4 {
            host[i][j] = 1.0;
        }
    }

    let mut device = transfer(host, Memory::NPU_HBM);

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                device[i][j] += 1.0;
            }
        }
    }

    return 0;
}
```

`Tensor<f32, [4, 4]>::uninit()` takes no arguments: the shape is already part of the type. Only the
dynamic form needs extents passed — `Tensor<f32, [?, ?]>::uninit([rows, cols])`.

The block is outlined into a kernel and handed to the dispatcher for that topology. On Apple Silicon
that means CoreML and the neural engine; on an NVIDIA box it means PTX. The source does not change
between the two — the machine file does.

Every value the block touches must already live in a memory the target topology can address. That is
why the `transfer` comes first. Skip it, and the error names the value and the space it is in rather
than crashing inside a vendor runtime.

The block also calls no helper function. A function declared without a topology belongs to the host,
and calling it from inside a device region is a compile error:

```
Error[E6001]: Type error: Function 'f' requires topology 'CPU', but is called from 'ANE'
```

That is the address-space rule doing its job. Code meant for a device is written in the region, or
in a function declared for that topology.

`spawn on` is a statement. To start a copy and do other work before it ends, use
`transfer_async(x, M)`. It gives a token, and `.wait()` on the token gives the placed tensor:

```rust
let first = transfer_async(next_tile, Memory::W);
// Other work here.
let tile = first.wait();
```

A program must wait for each token. A token that the program drops with no `.wait()` gets the
error E6036. Today the copy ends before `transfer_async` returns, so `.wait()` only gives the
tensor. A later runtime will run the copy in the background, and the program will not change.

A `spawn on` can also be in a task. So a `tasks` block can give work to several devices, one task
for each device (see [Tasks and channels](tasks.md)).

## Functions for any topology

The last section said that a device region cannot call a plain function. A plain function
belongs to the host. Often you want one helper for both: a clamp, a sum, a small formula. Vx
lets a function be generic over the topology that runs it, and over the memory of its data.

### One helper for host and device code

Write `on Topology::Any` after the parameters. Then the function runs on the topology of the code
that calls it. Each topology that calls it gets its own copy of the function.

`on Topology::Any` is short for a topology parameter that no argument binds. A topology
parameter is a generic parameter with the bound `Topology`, such as `<D : Topology>`. Inside the
function, `Topology::D` names it. So `fn f<D : Topology>(..) on Topology::D` means the same thing.
This program uses both forms:

```rust
{{#include ../../../tests/backend/pass/book_topology_any_helper.vx:program}}
```

It prints `gpu 5 13 host 0 -2`. With no GPU in the machine, the GPU region runs on the host, so
the program runs on any computer. Without `on Topology::Any`, the call `clamp(12, 0, 5)` in the
GPU region gets the error E6001.

`Topology::Any` is correct only after `on` in a function signature. A program cannot declare a
topology with the name `Any` (E6016).

### A function for any memory

A tensor's type records its memory. A parameter of type `&Tensor<i32, [?]>` names no memory, and
the host cannot give it a tensor in GPU memory (E3003). To accept a tensor in any memory, give the
function a memory parameter, such as `<M : Memory>`. Write the tensor type with `Memory::M` as
its memory:

```rust
{{#include ../../../tests/backend/pass/book_topology_memory_parameter.vx:program}}
```

It prints `host 6 gpu 30`. Each call binds `M` to the memory of its tensor argument. Two
arguments that both name `Memory::M` must be in the same memory. A call must also give `M` a
memory through a tensor argument (E3016).

The memory parameter does not change where the code runs. The host still cannot read a tensor in
GPU memory. If `main` calls `sum_three(&g)`, the compiler reports E6003 and asks for a `transfer`.

### Your own topology

A program can declare a topology of its own with a `Topology` block:

- `memory:` names the default memory of the topology. Every `Topology` block must have it.
  The memory can be a new name.
- `visible:` lists the memories that its code can read. It defaults to the `memory:` space.
- Each `transfer A -> B : cost` line adds a route for data, in one direction only.

`Reachable<A, B>` is true when data can go from topology `A` to topology `B` on the declared
routes. In an `if comptime` statement, the compiler tests it while it compiles. It compiles only
the branch that applies. A topology variable such as `D` can be an argument:

```rust
{{#include ../../../tests/backend/pass/book_topology_reachable_if_comptime.vx:program}}
```

It prints:

```text
host to acme 1
acme to host: no
gpu to acme: yes
```

The host can send data to `Acme`, but `Acme` has no route back. GPU data can go to `Acme`
through the host. If a topology has no route from the host, the compiler gives warning W1026.

### Asking for a route

A generic function that moves data between two topology parameters must say that a route exists.
Write `where Reachable<S, D>` after the return type. The compiler checks the clause at each call,
for the topologies of that call. A call with no route is a compile error. To give more than one
clause, separate them with commas.

```rust
{{#include ../../../tests/backend/pass/book_topology_reachable_where.vx:program}}
```

It prints `sum 14`. The argument binds `S` to the GPU, and the caller `main` binds `D` to the
host. If `x` is in `Acme` memory, the call fails with
`` unsatisfied `where Reachable<S, D>` in call to 'fetch_sum' ``.

Name only topology parameters in a `where` clause. The compiler does not check a clause that names
anything else yet ([Vx#3230](https://github.com/vx-lang/Vx/issues/3230)).

## What gets rejected

The point of putting placement in the type system is the errors you get for free.

**Dereferencing a device pointer from the host.** A `Pinned<T, Topology::NPU>` that escapes into a host
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
fn custom_matmul(a : Pinned<Tensor<f32, [4, 4]>, Topology::NPU[0]>,
                 b : Pinned<Tensor<f32, [4, 4]>, Topology::NPU[0]>)
    -> Verified<Tensor<f32, [4, 4], Memory::NPU_HBM>> {

    let mut result = unsafe { Tensor<f32, [4, 4], Memory::NPU_HBM>::uninit() };

    spawn on(Topology::NPU[0]) {
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = 0.0;
                for k in 0..4 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    return Verified(result);
}
```

`Pinned<T, Topology>` says the value is resident on a particular device. Note the third type
argument on `Tensor` — the memory space it lives in — and that `Verified(...)` wraps the result
*after* the region, not inside it.

## Choosing a machine at compile time

The same program compiles and runs against different hardware by swapping the machine file:

```bash
vxc --machine $HOME/.vx/current/fleet/h100-sxm.vx --host default program.vx
vxc --machine $HOME/.vx/current/fleet/m4-uma.vx --host default program.vx
```

`--host default` says which machine runs `main` ([Machine files](machine-files.md)).

You can ask what the compiler concluded, as JSON, rather than reading it out of diagnostics:

```bash
vxc --machine $HOME/.vx/current/fleet/h100-sxm.vx --host default program.vx --diagnostics-json out.json
```

That record carries every diagnostic with its structured fields — a capacity rejection includes the
space, the amount required, the amount available and the margin — plus the staging routes and
per-edge costs that an admitted program resolved to.

## Next

[Machine files](machine-files.md) covers how a real part gets described, and what the compiler
derives from that description.
