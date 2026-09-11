# Carrying facts across boundaries

A *correlation* is a fact about two or more values rather than about one: `j <= i`,
`tile * B + r == idx`, "these two tensors share a dimension", "this buffer lives on
that device", "the caller is still holding a tile while the callee runs".

Programs are full of them, and analyses are not built to hold them. A production
dataflow analysis records one fact per value, so a relation between values is not
weakened at a merge, it is destroyed. The mechanisms that recover one — peepholes,
GVN, ScalarEvolution, the affine/Presburger layer — are all scope-local. The single
mechanism that crosses a scope is inlining, and it works by deleting the boundary, so
it stops at the inliner's budget, again at an opaque framework op, and completely at a
separately compiled kernel launch, which cannot be inlined at all.

At that last seam the loss is not a matter of analysis effort. One kernel body can be
linked into a host program where the relation holds and into one where it does not. An
analysis reading only the kernel sees the same body in both and must return the same
verdict, and the only verdict sound for both is "do not optimize". The fact is not
underdetermined; it is absent.

So the question is not how to re-derive a relation on the far side of a boundary. It is
how to **carry** it. The seven programs below are the ways Vx does that today. Each one
is a program the compiler refuses, or compiles into code that carries the fact onward,
and each names the boundary it is about.

| | Boundary | The relation | Elsewhere | In Vx |
|---|---|---|---|---|
| 1 | a function return | this tensor lives in the GPU's memory | both buffers are `float *`; a host read faults at run time | the space is in the type, so the return type carries it — `E6003` |
| 2 | a call | the caller holds 3 MiB while the callee places its own | recovered only by inlining, so lost past its budget and at recursion | each function exports a summary; one whole-program fold composes them — `E6027` names the path |
| 3 | a generic call site | data can get from device A to device B | not expressible; the pair is checked, if at all, inside each instantiation | `where Reachable<A, B>` is stated once and discharged at every call site |
| 4 | a host→device launch | what the device reads is what the host wrote | a relaxed copy drops the release silently; the kernel looks identical | the consumer's `assert` becomes a seam obligation, discharged by z3 — `E6004` |
| 5 | a host→device launch | this key block is above the diagonal, so its work is dead | no carrier: `-O3` keeps the chain, and provably cannot do otherwise | the host's proof is re-materialized in the kernel as `llvm.intr.assume`, and `-O3` folds |
| 6 | source ↔ machine model | these bytes fit that memory | learned by renting the GPU and watching the allocation fail | the SKU loads as a peer module; one program text, a flag per SKU — `E6009` |
| 7 | a generic call site | these two tensors agree on a dimension | discovered by a tracer at trace time and dropped at lowering | a `const` name used twice is one binding, and the second argument has to agree with it |

## The two kinds

**Refusing** (1, 2, 3, 4, 6, 7). The relation is the premise of a correctness check.
Losing it does not make the program slower; it makes the compiler agree to something it
cannot support: a host read of device memory, a working set that does not fit, a
transfer with no path, a buffer that may be read stale, a tile that does not fit the
part it will be rented on, a pair of tensors computed against different extents. Every
one of these is refused on a laptop, before a machine is booked.

**Licensing** (5). The relation is the premise of an optimization. Nothing is wrong
with the program; there is work in it that is dead, and only the host knows so. The
certificate is what lets the device compiler act on that.

## 1. Residency across a call

The memory space is part of the type, so the producer's return type carries it across
the call and the caller's read is refused with both spaces named.

```rust
fn stage_to_device() -> Tensor<f32, [4, 4], Memory::GPU_HBM> {
  let host_data : Tensor<f32, [4, 4]> = Tensor<f32, [4, 4]>::new();
  return transfer(host_data, Memory::GPU_HBM);
}

fn main() -> i32 {
  let kv = stage_to_device();
  print(kv[0][0]);   // refused: the host cannot address GPU_HBM
  return 0;
}
```

```
Error[E6003] at 43:9: 'kv' lives in GPU_HBM but CPU sees only [CPU_DRAM, NPU_HBM];
insert an explicit transfer to CPU_DRAM (cost 50 on the declared path)
```

The repair is one line — `let home = transfer(kv, Memory::CPU_DRAM);` — and it is the
copy the C++ version also needed and did not get told about. The price in the message
comes from the declared machine.

## 2. A working set across a call

Neither function overflows the 4 MiB space on its own. Only the sum does, and the sum
exists only across the call.

```rust
// 3 MiB in W. Fits on its own.
fn stage<T>(_t: T) -> i32 {
  let y = Tensor<f32, [1024, 768]>::uninit();
  let _sy = transfer(y, Memory::W);
  return 1;
}

fn main() -> i32 {
  let x = Tensor<f32, [1024, 768]>::uninit();
  // `sx` is read after the call, so its 3 MiB is still resident while `stage`
  // runs. That liveness is what the summary records at the call site.
  let sx = transfer(x, Memory::W);
  let r = stage(7);
  let _back = transfer(sx, Memory::CPU_DRAM);
  return r;
}
```

```
Error[E6027] at 59:11: the working set along call path 'main -> stage$i32' in memory
space 'W' peaks at 6291456 bytes, over its 4194304 byte capacity: 'main' holds 3145728
bytes across its call, 'stage$i32' itself peaks at 3145728 bytes
```

Vx does not inline to get this. Each function exports a summary — its own peak per
memory space, and per call site the bytes still live at that site — and one
whole-program fold over the call graph composes them. Sequential calls compose by max;
a tile held across a call composes by `+`. The refusal names the path rather than the
function, because the overflow is a property of the path.

## 3. Reachability across a generic call

"Data can get from topology A to topology B" holds between two values rather than of
either one, which is exactly the kind of fact a non-relational analysis cannot record.
The function below is generic over three devices and never names one:

```rust
fn pipeline<A: Topology, B: Topology, C: Topology>(
  a: Pinned<i32, Topology::A>,
  b: Pinned<i32, Topology::B>,
  c: Pinned<i32, Topology::C>
) -> i32
where Reachable<A, B>, Reachable<B, C>
{
  return 0;
}
```

`Island` declares memory with no transfer edge into it, so nothing can reach it:

```
Error: unsatisfied `where Reachable<B, C>` in call to 'pipeline':
no transfer path from CPU to Custom("Island")
```

The relation is written once and discharged at every call site. The failing constraint
is named; the first hop, `GPU -> CPU`, holds and is not reported.

## 4. Freshness across the launch

`to_device_relaxed()` drops the release, and with it the visibility guarantee the
consumer depends on. The kernel body is identical either way.

```rust
fn stage(a: Tensor<i32, [4]>) -> i32 {
  let local_a = a.to_device_relaxed();
  spawn on(Topology::NPU[0]) {
    assert(local_a[0] == 42);   // the contract the seam must preserve
  };
  return 0;
}
```

```
Error[E6004] at 40:17: relaxed transfer of 'a' across the CPUDRAM -> NPUHBM seam
violates the boundary contract: the buffer carries no synchronizing release, so a
consumer may read it stale
```

The consumer's `assert` becomes an obligation on the seam, discharged by z3. Swap in
`to_device()` and the same program is admitted. This check needs z3 on `PATH` and is
requested with `--verify-seams`.

## 5. A certificate across the launch

Nothing is wrong with this program. There is work in it that is dead, and only the host
knows so:

```rust
fn launch(kblk_start: i32, qblk_end: i32, x: f32) -> f32 {
  // The host's proof. This is the certificate; everything else is transport.
  assert(kblk_start > qblk_end);
  let mut out = Tensor<f32, [64]>::uninit();
  spawn on(Topology::GPU) {
    // Mask-as-data: the causal condition is a value, not a loop constraint.
    // There is no iteration domain left to split.
    let masked = kblk_start > qblk_end;
    for i in 0..64 {
      let mut e: f32 = x;
      for _k in 0..256 {
        e = e * x + 1.5;     // dead whenever `masked` holds
      }
      if masked { out[i] = 0.0; } else { out[i] = e; }
    }
  };
  return out[0];
}
```

The kernel alone cannot know `masked` is always true, so `-O3` keeps the 256-trip FMA
chain and provably cannot do otherwise. With `--emit-seam-certs` the host's proof is
re-materialized inside the kernel as `llvm.intr.assume`, and the chain folds:

```
base: fmul=2 fadd=2
cert: fmul=0 fadd=0
```

Both versions print the same answer.

## 6. Capacity against a declared machine

One program text, a flag per SKU. The machine loads as a peer module, so the same
source is admitted or refused according to the part it is compiled for:

```rust
fn main() -> i32 {
  // 50 GiB of f16: over the 40 GB part, under the 80 GB one.
  let kv : Tensor<f16, [51200, 524288]> = Tensor<f16, [51200, 524288]>::uninit();
  let _staged = transfer(kv, Memory::HBM);
  return 0;
}
```

```bash
vxc --host default --machine fleet/a100-40.vx 06_capacity_against_a_declared_machine.vx
```

```
Error[E6009]: transferred tensor needs 53687091200 bytes but memory space 'HBM'
has capacity 42949672960 bytes
```

Against `fleet/a100-80.vx` the same text is admitted.

## What makes transport sound

A certificate is only as good as the proof behind it. Example 5 transports exactly the
facts the host has already established — the condition of an `assert` — and only into
a kernel that names every variable the fact mentions, so each operand resolves to a value
the kernel body can already see. Break the relation and the `assert` fires before the
kernel is ever reached, so the assumption the device compiled against is never live.

Nothing here re-derives a relation on the far side. The fact is proved once, where it is
known, and moved forward.

## 7. A shared dimension across two arguments

A `const` name used by two parameters is the relation "these two tensors agree on this
dimension" — `q` and `k` on a sequence length, an operand pair on a contraction. It is
the relation a framework tracer discovers at trace time and drops at lowering, and
neither argument's own type says anything about it.

```rust
fn scores<const S: i32>(q: Tensor<f32, [S]>, k: Tensor<f32, [S]>) -> i32 {
  return S;
}

fn main() -> i32 {
  let q = Tensor<f32, [4]>::uninit();
  let k = Tensor<f32, [7]>::uninit();   // a different sequence length
  print(scores(q, k));
  return 0;
}
```

```
Error: Failed to deduce types for generic function 'scores': Expected Tensor<f32, [S]>,
got Tensor<f32, [7]>; 'S' is bound to 4 by argument 1, but argument 2 has extent 7
```

Deduction binds `S` once, from the first argument, and every later argument is checked
against that binding rather than replacing it. So the refusal is about the call as a
whole — there is no `S` that satisfies both — and the message names the conflict rather
than one argument: which argument bound `S`, to what, and what this one asked for
instead.

## Running them

The seven programs live next to this page, with a script that runs each one and checks
its verdict:

```bash
./www/book/src/correlation/run.sh
```

A clean run is one where the compiler says "no" six times for six different reasons,
and folds the FMA chain once. Example 4 needs `z3` and is skipped without it; the `-O3`
comparison in example 5 needs `mlir-translate` and `opt` from the same LLVM as the
build.
