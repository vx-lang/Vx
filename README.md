<div align="center">
  <h1>Vx</h1>
  <p><b>One language, every core.</b></p>
  <p>A heterogeneous-systems programming language that puts placement and reachability in the type system.</p>
</div>

______________________________________________________________________

> **This is v0.0.1.** `vxc` is a cross compiler: the machine it targets is *declared*, not detected —
> `--host` for the CPU and `--machine fleet/<sku>.vx` for the accelerator — so A100 binaries can be
> built on an x86 EC2 box and shipped to the GPU machine. It supports the architectures LLVM targets.
> It has been tested most on a MacBook Air M4 (AArch64 + Apple ANE) and an NVIDIA A100 (x86_64 +
> CUDA); the placement and topology checks are tested on both. It is also an early research
> compiler: the syntax is mostly stable, the standard library is thin, there is no package manager, and
> some features described in `docs/` are yet to be implemented.

## What Vx is

Vx (pronounced "vee-ex") is a compiled, statically typed systems language for programs that span more
than one kind of processor: a CPU, a GPU, an NPU, or a second machine. Its one idea is that *where a
value lives* and *where code runs* belong in the types. A tensor's type carries its element type, its
shape, and its memory space. A `spawn on(Topology::NPU[0]) { ... }` block runs on a declared device.
Reading host memory from inside that block is a compile error, with a message naming the space the
value is in, the spaces the device can see, and the `transfer` that fixes it.

Certain machine descriptions are declared in a source file: memory spaces with capacities and bandwidths,
devices with the spaces they can address, and transfer edges with costs. The compiler checks a
program against that declaration before anything runs, so "this working set does not fit HBM" or
"this device cannot see that buffer" is a type error rather than a crash after the hardware has been
rented.

## Topology and data-placement simplifies heterogeneous programming

```rust
fn main() -> i32 {
  let mut a = Tensor<f32, [4, 4]>::uninit();
  let mut b = Tensor<f32, [4, 4]>::uninit();
  let mut c = Tensor<f32, [4, 4], Memory::NPU_HBM>::uninit();
  for i in 0..4 {
    for j in 0..4 {
      a[i][j] = 2.0;
      b[i][j] = 3.0;
    }
  }
  let a_npu = transfer(a, Memory::NPU_HBM);
  let b_npu = transfer(b, Memory::NPU_HBM);
  spawn on(Topology::NPU[0]) {
    c = a_npu @ b_npu;
  }
  let c_host = transfer(c, Memory::CPU_DRAM);
  print(c_host[0][0]);
  return 0;
}
```

```
$ vxc matmul.vx
24
```

If you forget to add the first `transfer` (`let a_npu = transfer(a, Memory::NPU_HBM);`), the compiler refuses the program with two errors; the first names
the fix:

```
Error[E6003] at 13:9: 'a' lives in CPU_DRAM but NPU[0] sees only [NPU_HBM];
insert an explicit transfer to NPU_HBM (cost 50 on the declared path)
```

What produced that `24`, on the two machines it was run on for this release:

**A Mac (Apple silicon).** The spawn region is dispatched through CoreML, which places an fp32 4×4
matmul on the CPU. The placement is declared, checked and enforced; the arithmetic ran on the host.

**Linux with an NVIDIA A100** (driver 580.126.16, CUDA 12.8). Change the placement to the device
that is actually there — `Memory::GPU_HBM` and `Topology::GPU[0]` — and, with `VX_DISPATCH_VERBOSE=1`,
the same program logs:

```
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 dispatch
[Vx CUDA] GEMM 4x4x4 f32 -> buffer
[Vx CUDA] device 0 free
[Vx CUDA] device 0 free
24
```

Each `transfer` into `GPU_HBM` is a `cudaMalloc` and a host-to-device copy, the spawn is a device
dispatch, the multiply runs as a cuBLAS GEMM on the A100, and the buffers are freed on the device.
The program as written above, with `NPU_HBM` and `NPU[0]`, also prints `24` on that box and its
multiply also runs on the A100 — the dispatcher recognises the region as a GEMM and cuBLAS stages the
operands itself — but its two `transfer`s do not touch device memory, because the CUDA backend owns
the `GPU` topology and not `NPU`. The declaration is the placement; the compiler holds you to what
you named, not to what is plugged in. No timings are quoted from either machine.

## What the compiler checks today

These are implemented, have diagnostic codes, and are held by tests:

| Check | Rules out |
| --- | --- |
| Address-space visibility (`E6003`) | A device reading a memory space it cannot address; a host reading device memory |
| Capacity admission (`E6009`, `E6010`) | One tensor larger than its space; a working set that fits tile by tile and not together, with granule rounding |
| Transfer reachability (`E6002`) | A move between spaces with no declared path |
| Transfer routing and cost | The cheapest legal route over the declared edges, charged at both ends of a containment hop |
| Borrow checking | Aliasing and lifetime errors, with region tracking on the default codegen path |
| Linear types | Use after move of a consumed buffer |
| Seam contracts | Reading a buffer whose asynchronous `transfer` has not been made visible; obligations are discharged by calling `z3` |
| Generic calls | The visibility obligation runs after substitution, so a generic device parameter is checked at its instantiation, not its declaration |

A differential suite of six paired programs exercises the visibility and capacity checks against
CUDA on an A100: the same mistake written both ways, with Vx refusing at compile time what CUDA
reports at synchronization, at `cudaMalloc`, or as a segmentation fault. A regression test asserts each pair still reaches its claimed verdict.

Two limits on the list. The calculus behind these checks is a design with tests, not a
soundness theorem. And `grad()` (autodiff through an Enzyme plugin, enabled by `ENZYME_LIB`) is
present but young: it currently accepts differentiation of a discrete-valued function (#503).

## The machine description of topology and memory

```rust
Memory HBM  { capacity: 80 GiB, bandwidth: 3.35 TB/s, managed: explicit, scope: device }
Memory L2   { within: Memory::HBM, capacity: 50 MiB, bandwidth: 12 TB/s, managed: cached }
Memory SMEM { within: Memory::L2, capacity: 228 KiB, bandwidth: 128 B/cyc,
              clock: 1.98 GHz, replicas: 132, granule: 1 KiB, scope: sm }

Topology Device {
    arch: nvptx64,
    memory: Memory::HBM,
    transfer Memory::CPU_DRAM -> Memory::HBM : 63 GB/s,
}
```

`fleet/` holds twelve such files — A100 (40 and 80 GB), H100, H200, B200, MI300X, an Apple M4, two
2-GPU nodes, an 8-GPU node, an x86 Xeon as a compute target, and an x86 host — at 8 to 24
declaration lines each. Each cites
its sources and marks unverified figures as such. Units are exact integer conversions (`GB` is 10⁹,
`GiB` is 2³⁰), so a number copied from a vendor sheet means what the sheet meant.

The declared model has been checked against hardware for one field on one SKU. The other figures
carry citations, not measurements. `utils/memalg/` has the instruments; see
[`docs/memory_algebra.md`](docs/memory_algebra.md).

## How it compiles

**Frontend.** Every symbol, type and monomorphized instantiation is a 256-bit content-hashed
identifier carried by value in flat arrays. Modules are parsed and type-checked in parallel; the
generic instantiations that cannot be named by content alone are minted locally and reconciled once
at a barrier. No lock primitive appears on the frontend path, and a CI lint keeps it that way. The
emitted MLIR is byte-identical whether the compile runs on one thread, on four, or with the thread
pool removed entirely; a test asserts this rather than assuming it.

**Two codegen paths.** The default lowers the type-checked program to a flat bytecode (a parallel frontend) and emits MLIR
from that. An older path emits MLIR directly from the AST and is kept as an oracle
(`--legacy-codegen`). Per function, the default path declines what it cannot yet lower and falls back
to the oracle; 12 programs in the test corpus currently take that fallback, and 4 do not build on
the default path at all. Both lists are asserted in a test so that a change in either direction is
noticed.

**Backends.**

| Target | What exists | What has run |
| --- | --- | --- |
| CPU, x86-64 and arm64 | MLIR → LLVM IR → native, JIT or object file | Everything in the test suite |
| NVIDIA GPU | MLIR → NVVM → PTX, shipped in the dispatch payload and loaded by the driver | Fused attention and a disaggregated prefill/decode split, on A100 and H100 |
| Apple | CoreML dispatch from a native plugin | One f16 512×512 matmul on the Neural Engine, confirmed through the compute-plan API; every fp32 kernel is placed on the CPU by CoreML |
| Remote | A wire protocol carrying memref descriptors and dispatch payloads to a worker on another machine | An arm64 laptop dispatching to an x86-64 worker over an SSH tunnel |

The JIT compiles at `-O0` by default; pass `-O` for `-O3`.

## Known limitations

Things a new user will hit, with the issue that tracks each:

- There is no `while` loop. `for` over a range and recursion are what exist (#506).
- `spawn on` is a statement. The design in [`docs/spawn_on.md`](docs/spawn_on.md) makes it an
  expression yielding a future; there is no future type and no `await` today.
- `Ref<T, Memory>` parses and type-checks but has no effect (#507).
- `Vec` has no destructor; `free()` is manual (#495).
- An installed toolchain cannot yet find its own runtime library or `mlir-translate` without the
  source tree on `PATH` (#496, #498). Run from a checkout for now.
- Item visibility (`pub`) is reserved in the identifier layout and absent from the language (#489).
- Two standard library modules, `iter` and `tensor`, do not type-check on their own (#487).
- Of the five directories under `packages/`, four are empty placeholders. `packages/README.md` says so.
- The Apple backend is tested only on macOS; CI runs on Linux, so 20 tests are gated `REQUIRES: macos`
  and do not run there.

The full list is the [issue tracker](https://github.com/vx-lang/Vx/issues).

## Building

You need Rust, LLVM/MLIR 22 with the MLIR C API, `z3` on `PATH` for the seam prover, and `cmake`.
On macOS, `brew install llvm z3 cmake`; on Debian-family Linux, LLVM 22 from apt.llvm.org and
`apt install z3 lld`. `setup.sh` locates LLVM and writes `config.local`; the CI workflow in
`.github/workflows/ci.yml` is the reference for a working Linux install.

```bash
./setup.sh            # once: finds LLVM, writes config.local
source config.local   # every shell, before any cargo command
cargo build --release
```

## Running

```bash
# Compile and run under the JIT (the default action)
./target/release/vxc program.vx

# Check and compile against a declared machine
./target/release/vxc --host default --machine fleet/h100-sxm.vx program.vx -o program

# The admission and cost decisions, as JSON
./target/release/vxc --host default --machine fleet/h100-sxm.vx program.vx --diagnostics-json out.json

# Look at the MLIR
./target/release/vxc program.vx --action emit-mlir
```

## Testing

```bash
source config.local
cargo test
```

About 535 unit tests and 42 integration suites run on every commit through a pre-commit hook. The
integration suites include the placement fixtures under `tests/frontend`, the codegen fixtures under
`tests/backend` (FileCheck-style `RUN:` lines), the differential pairs, and the determinism gate.
Setting `ENZYME_LIB` to a built Enzyme MLIR plugin enables the autodiff tests.

## Repository layout

```
src/                  the vxc compiler (Rust)
  lexer, parser/      source → AST
  syntax/             AST, types, topologies, declarations
  hir/                type checking, borrow checking, the memory algebra, flattening
  codegen/            MLIR emission for both paths
  dialect/            the vx MLIR dialect and its two lowering passes (C++)
  gid.rs              the 256-bit identifier
  pipeline.rs         the parallel frontend
  plugin/             vendor dispatch plugins
  jit.rs              JIT and object emission
runtime/              dispatch backends (host, CUDA, CoreML) and the remote worker (C++)
fleet/                twelve machine files, cited
stdlib/               21 modules: tensor, simd, vec, hash_map, io, fs, net, mmap, math, …
examples/llama.vx     a Llama 2 forward pass; compiles under the test suite
docs/                 design documents, some describing work not yet done
utils/                measurement harnesses for the cost model and the parallel frontend
vx-analyzer/          language server (diagnostics, hover, go-to-definition)
vscode-vx/            VS Code extension
tests/                the suites listed above
```

## Tools

| Tool | Purpose |
| --- | --- |
| `vx-format` | Source formatter; CI checks that `tests/` and `stdlib/` are formatted |
| `vx-opt` | `mlir-opt`-style driver for the vx dialect passes |
| `cargo vx-bench` | Benchmark runner |
| `vx-analyzer` | Language server. Set `VX_ANALYZER_LOG=<path>` for a debug log |
| `vscode-vx` | VS Code client; set `vx.analyzerPath` if the server is not on `PATH` |

## Where this is going

[ROADMAP.md](./ROADMAP.md) tracks features against the language's stated goals and marks each done or
not. [`docs/`](docs/) holds the design documents; several describe things that do not exist yet, and
they say so at the top when they do. The parallel frontend, the placement checker and the machine
model are each written up in more depth under `docs/`.

## License

Apache License 2.0 with LLVM Exceptions. See [LICENSE](LICENSE).
