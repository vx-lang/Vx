# Writing a flash attention kernel

This chapter writes a fused attention forward pass and runs it on an NVIDIA A100. The kernel is
tiled, keeps a running softmax, and never materializes the score matrix — the FlashAttention
shape.

The point of the chapter is not the kernel. It is *where the mistakes happen*. Nearly every error
below is reported by `vxc` on a laptop, before any hardware is rented: a tile that does not fit on
chip, a buffer the device cannot address, a shared-memory race that no barrier can order. Those
are the failures that normally cost an afternoon of `cuda-memcheck` on a machine billed by the
hour. Here they cost a rebuild.

One of them is not caught, and it gets a section of its own rather than a footnote, because an
exception to this claim is more useful to a reader than the claim is.

Only one section needs a GPU, and by the time it arrives the program is already known to compute
the right answer.

## The machine, declared

Vx does not detect the machine. It is told, and then it holds the program to what it was told.
An A100's memory hierarchy is three spaces and the moves between them:

```vx
Memory CPU_DRAM {}

Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}

Memory SMEM {
  within: Memory::GPU_HBM, capacity: 164 KiB, bandwidth: 128 B/cyc,
  granule: 1 KiB, managed: explicit, scope: sm
}

Topology Dev {
  arch: nvptx64,
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM, Memory::SMEM],
  transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 63 GB/s,
  transfer Memory::GPU_HBM -> Memory::SMEM
}

fn main() -> i32 {
  return 0;
}
```

`scope: sm` is what makes `SMEM` shared memory rather than another pool of global memory: the
space is private to a streaming multiprocessor, so a tensor placed there becomes `.shared`
storage in the emitted PTX. `capacity: 164 KiB` is the A100's real per-SM budget, and the
compiler spends it.

The `fleet/` directory ships these descriptions ready-made — `fleet/a100-40.vx` and
`fleet/a100-80.vx` — for use with `--machine`. Declaring the spaces inline, as above, keeps
everything in one file and is what the rest of this chapter does.

## Attention, written the obvious way

Attention is `softmax(QKᵀ · scale) · V`. Written directly, that expression needs the score matrix,
and the score matrix wants to be on chip where the softmax can reach it cheaply:

```vx
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
Memory SMEM {
  within: Memory::GPU_HBM, capacity: 164 KiB, bandwidth: 128 B/cyc,
  granule: 1 KiB, managed: explicit, scope: sm
}

fn main() -> i32 {
  let mut scores = Tensor<f32, [1024, 4096]>::uninit();
  for i in 0..1024 {
    for j in 0..4096 {
      scores[i][j] = 0.0;
    }
  }
  let on_chip = transfer(scores, Memory::SMEM);
  return 0;
}
```

The compiler refuses it, with the arithmetic:

```
Error[E6009]: transferred tensor needs 16777216 bytes but memory space 'SMEM' has capacity 167936 bytes
```

Sixteen mebibytes of scores against a hundred and sixty-four kibibytes of shared memory, for a
sequence of four thousand and ninety-six. This is the constraint the FlashAttention paper opens
with, and it arrives here as a diagnostic rather than as a citation. It is also the constraint
that decides the whole shape of the kernel: the score matrix cannot exist, so the softmax has to
be computed without ever seeing all of it at once.

## Tiling, and the running softmax

The way out is to walk the keys in tiles and carry three pieces of state per query — the running
maximum `m`, the running denominator `l`, and the running output row — rescaling the accumulated
output whenever a new tile raises the maximum. That rescale is what makes a streamed softmax
exact rather than approximate.

Here is the whole kernel, at a size small enough to check by hand:

```vx
import std::math;

Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}

fn main() -> i32 {
  let mut q_h = Tensor<f32, [2, 4]>::uninit();
  let mut k_h = Tensor<f32, [4, 4]>::uninit();
  let mut v_h = Tensor<f32, [4, 4]>::uninit();
  let mut o_h = Tensor<f32, [2, 4]>::uninit();
  for d in 0..4 {
    q_h[0][d] = 1.0;
    q_h[1][d] = 0.0;
    k_h[0][d] = 0.0;
    k_h[1][d] = 1.0;
    k_h[2][d] = 2.0;
    k_h[3][d] = 3.0;
    v_h[0][d] = 0.0;
    v_h[1][d] = 1.0;
    v_h[2][d] = 2.0;
    v_h[3][d] = 3.0;
  }
  for i in 0..2 {
    for d in 0..4 {
      o_h[i][d] = 0.0;
    }
  }

  let q = transfer(q_h, Memory::GPU_HBM);
  let k = transfer(k_h, Memory::GPU_HBM);
  let v = transfer(v_h, Memory::GPU_HBM);
  let mut o = transfer(o_h, Memory::GPU_HBM);
  let scale : f32 = 0.5;

  spawn on(Topology::GPU) {
    let mut ts = Tensor<f32, [1, 2]>::uninit();
    for i in 0..2 {
      let mut m : f32 = -1000000.0;
      let mut l : f32 = 0.0;
      for t in 0..2 {
        let mut tm : f32 = -1000000.0;
        for jj in 0..2 {
          let j = t * 2 + jj;
          let mut acc : f32 = 0.0;
          for d in 0..4 {
            acc += q[i][d] * k[j][d];
          }
          let sij : f32 = acc * scale;
          ts[0][jj] = sij;
          if sij > tm {
            tm = sij;
          }
        }
        let m_new : f32 = if tm > m {
          tm
        } else {
          m
        };
        let corr : f32 = (m - m_new).exp();
        l = l * corr;
        for d in 0..4 {
          o[i][d] = o[i][d] * corr;
        }
        for jj in 0..2 {
          let j = t * 2 + jj;
          let p : f32 = (ts[0][jj] - m_new).exp();
          l = l + p;
          for d in 0..4 {
            o[i][d] = o[i][d] + p * v[j][d];
          }
        }
        m = m_new;
      }
      for d in 0..4 {
        o[i][d] = o[i][d] / l;
      }
    }
  }

  let o_home = transfer(o, Memory::CPU_DRAM);
  print(o_home);
  return 0;
}
```

Three things in that program are placement rather than arithmetic. The four `transfer` calls move
the operands into `GPU_HBM`, and each one is checked against the space's capacity and charged a
cost derived from its declared bandwidth. The `spawn on(Topology::GPU)` block is the region that
will become a device kernel; reading `q_h` instead of `q` inside it is a compile error naming the
space the value is in and the transfer that fixes it. And the last `transfer` brings the output
back to `CPU_DRAM` so that `print` can read it — the section after next is about what happens
when that line is missing.

### The mistake this section is really about

Delete the `let mut o = transfer(o_h, ...)` line and write to `o_h` inside the region. In CUDA the
equivalent — handing a kernel a host pointer — compiles, launches, and dies at run time with an
illegal memory access, on the GPU, after the queue wait. Vx reports `E6003` before the file
finishes compiling, and names the space, the spaces the device can address, and the transfer to
insert.

## Running it before there is a GPU

The program above is placed on a GPU topology and runs on a laptop. When no device is present the
dispatcher falls back to the host, so the arithmetic is exercised without the hardware:

```console
$ source config.local
$ ./target/release/vxc flash.vx
[flat-codegen] emitted module via the flat path
[JIT] Executing native binary...
Unranked Memref base@ = 0x105187be0 rank = 2 offset = 0 sizes = [2, 4] strides = [4, 1] data =
[[2.84482,   2.84482,   2.84482,   2.84482],
 [1.5,   1.5,   1.5,   1.5]]
```

Both rows are checkable without running anything. The second query is all zeros, so every score is
zero, so the softmax is exactly uniform and the output is the mean of the rows of V — `(0+1+2+3)/4 = 1.5`, which is what printed. The first row is the softmax over scores `0, 2, 4, 6` weighting the
same four values, and it comes to `2.84482`.

> A closed form that the kernel cannot accidentally produce is worth more here than a reference
> implementation. A missing rescale, a tile staged at the wrong offset, or a denominator
> accumulated in the wrong order all move these numbers. Every attention benchmark in
> `scripts/campaigns/flash/` is built around one.

This is the section that earns the chapter's claim. The kernel is now known to be correct. What
remains for the GPU is whether it is *fast*, which is a different question and cannot corrupt the
answer.

## Running it on the A100

Nothing about the program changes. The compiler lowers the spawn region through NVVM to PTX —
text, not a cubin, so no `ptxas` is needed on the machine doing the compiling — and ships it in
the dispatch payload for the driver to load. The default chip is `sm_80`, which is the A100;
`VX_GPU_CHIP` overrides it.

```console
$ VX_DISPATCH_VERBOSE=1 ./vxc flash.vx --run
[flat-codegen] emitted module via the flat path
[[2.84482,   2.84482,   2.84482,   2.84482],
 [1.5,   1.5,   1.5,   1.5]]
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 dispatch
[Vx CUDA] vx_npu_kernel_0 ran on GPU 0 from its own image (28 params, 1x2 threads)
[Vx CUDA] device 0 fetch
[Vx CUDA] device 0 free
[Vx CUDA] device 0 free
[Vx CUDA] device 0 free
[Vx CUDA] device 0 free
```

Four stages for the four operands, one launch of a kernel the compiler emitted, a fetch for the
transfer home, four frees, and the same two rows that printed on the laptop. The launch is two
threads because this toy shape has two queries; at a real size the same kernel goes wide — the
8192-query version of it launches as 64 blocks of 128 threads, three blocks resident per SM.

> `spawn on` places a region; it does not promise the region reaches the device. A region the
> backend cannot lower is refused and run on the host, which costs performance and never
> correctness. Read the trace rather than assuming: `run_flash_bench.sh` greps for exactly this
> and prints a verdict, because a CPU number and a GPU number look identical in a table.

To run it at a real size on a rented machine, `scripts/provision/bootstrap_dev_pod.sh` ships the
tree, builds the compiler, and assembles the bench bundle. On a fresh 255-core pod the whole
toolchain install plus a release build took under four minutes:

```console
$ scripts/provision/bootstrap_dev_pod.sh -h root@<pod> -p <port> -i ~/.ssh/<key>
$ cd /root/bundle && ulimit -s 524288 && ./run_flash_bench.sh -q 8192 -d 64
```

### The mistake this chapter's own machine model allows

Everything else here fails on a laptop. This one does not, and it is worth the space because the
reason is the declaration rather than the compiler.

Drop the `let o_home = transfer(o, Memory::CPU_DRAM);` line and print `o` directly. On a machine
with no device the program is fine: the fallback never moved `o` anywhere, so `print` reads host
memory and the right numbers appear. On the A100 the kernel runs correctly and then the program
dies in the printer:

```console
$ VX_DISPATCH_VERBOSE=1 ./vxc flash.vx --run
[flat-codegen] emitted module via the flat path
Caught SIGSEGV: Segmentation Fault!
Backtrace [
    { fn: "vx_sigsegv_handler" },
    { fn: "_ZN4impl17MemRefDataPrinterIfE5printERSoPflllPKlS5_" },
    { fn: "printMemrefF32" },
    { fn: "main" },
]
[Vx CUDA] vx_npu_kernel_0 ran on GPU 0 from its own image (28 params, 1x2 threads)
Program was killed by signal 6
```

`o` is a device handle, and `print` dereferences it on the host.

The compiler has a rule for exactly this, and it did not fire. Add one attribute to the `GPU_HBM`
declaration at the top of the program — `managed: explicit` — and the same mistake is refused
before anything runs, by plain `vxc flash.vx` with no flags:

```
Error[E6003]: `print` reads its argument on CPU, which sees only [CPU_DRAM, NPU_HBM],
but the value lives in GPU_HBM; bring it home first with `transfer(.., Memory::CPU_DRAM)`
```

`managed:` is what says whether the host may read a space. `explicit` means it may not, and the
visibility rule applies. Left out, or written `cached`, the space is one the host is allowed to
read, and the checker is right not to complain. The declaration in this chapter never claimed the
host was locked out of `GPU_HBM`, so nothing was violated.

What is wrong is underneath: the runtime never consults `managed` at all. A space stages to the
device through `cudaMalloc` whichever way it was declared, so a space declared host-readable is
allocated host-unreadable, and the segfault lives in that gap. The declaration is checked against
the program and not against the backend.

> This is not an exotic corner. `fleet/a100-80.vx` declares the A100's HBM `managed: explicit`,
> which is correct. Every GPU program in this repository that actually *runs* — the benches in
> `scripts/campaigns/flash/`, the placed tests, and this chapter — declares it host-readable
> instead, because that is what lets the host fallback stand in for the device on a laptop. The
> model that runs and the model that is accurate are not the same model, and this class of bug
> lives in the difference.

So: `managed: explicit` buys the compile-time refusal and costs the laptop rehearsal — with it,
`vxc` will not compile this kernel on a machine with no CUDA backend, because falling back to the
host would dereference device memory, and it says so rather than doing it. That refusal is the
correct behaviour and it is the same rule, seen from the other side. Until the runtime honours
`managed`, transfer device-resident results home before reading them, and treat a clean laptop run
as evidence about the arithmetic rather than about the placement.

## Shared memory, and the two refusals

The kernel above reads K and V from global memory once per query. A block of queries can instead
stage each K/V tile into shared memory once and have all of its threads consume it from there.
That is the FlashAttention-2 work partition, and `scripts/campaigns/flash/flash_coop_bench.vx`
writes it in Vx: a block owns a tile of queries, its threads cooperatively fill `kt`/`vt` in
`SMEM`, a `barrier()` separates filling from consuming, and each thread walks one query.

Two classes of bug become compile errors at this point, and both are ones that shared memory is
notorious for.

**A tile that does not fit.** The budget at 128 queries, a 16-key tile and head dimension 64 is 41
KiB of the A100's 164. Widen the tile past the budget and the transfer is refused with the byte
count, exactly as in the second section. There is no launch, and no silently truncated tile.

**A race no barrier can order.** A tensor written from a thread loop at that thread's own row and
read at a *neighbour's* row inside the same loop has no ordering that a barrier could supply. The
two-level prover rejects it rather than emitting a kernel whose answer depends on the warp
schedule. Writes to a shared tensor are allowed only from a thread loop, at the loop's own row,
and a thread loop that wrote one must be followed by a barrier.

> The cooperative kernel is slower than the simpler one. Measured on an A100 at SQ=8192, K=2048,
> TQ=64, TILE=16, it took 6.42 ms against the flat kernel's 3.74 ms: three blocks per SM,
> shared-memory-limited at 40.5 KiB each, losing to the L2 broadcast that the simpler kernel rides
> for free. The structure is proven and the tuning is not done. A tutorial that ended with the
> tiled version winning would be a nicer story and a false one.

## Where this kernel sits

Attention on an A100 has a long ladder above it, and it is worth knowing which rung this chapter
reaches. Measured at SQ=8192, SK=2048, HD=64:

| | time | rate | |
| --- | --- | --- | --- |
| The kernel in this chapter, fp32 | 3.72 ms | 1.15 TF/s | measured |
| A tuned split-K fp32 kernel | 1.125 ms | 3.82 TF/s | cited |
| Unfused, two routed TF32 GEMMs | 0.486 ms | 8.84 TF/s | cited |
| Unfused, two routed f16 GEMMs | 0.374 ms | 11.5 TF/s | cited |
| `flash_attention_into`, routed to a vendor provider | 0.067 ms | ~64 TF/s | measured |
| PyTorch SDPA on the same machine | 0.063 ms | | cited |

The rows marked *measured* were taken on an A100-SXM4-80GB (driver 580.159.04, CUDA 12.8) with
CUDA event timing, three runs each, nothing else on the device. The chapter's kernel came in at
3.723, 3.725 and 3.743 ms; the routed path at 0.067, 0.068 and 0.075 ms.

The first row and the second are the same algorithm at different amounts of tuning, and the first
and the fifth are not the same workload at all: this chapter's kernel is fp32, and the routed path
is f16 on tensor cores. A fifty-fold gap sounds like a verdict on the language, and most of it is
those two facts.

> Time the device, not the program. `run_flash_bench.sh` derives a kernel cost from how total wall
> time grows with sequence length, which was the right instrument before the compiler could emit a
> device image. It no longer is: the host-side loops that fill K and V grow with the sequence too,
> and at `-O0` they dominate. On the run above it reported 46.9 GFLOP/s for a kernel that CUDA
> events put at 1154. `VX_TIME_KERNEL=1` is the number to quote.

The last rung is one call:

```vx
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s
}
Topology Dev {
  arch: nvptx64,
  memory: Memory::GPU_HBM,
  visible: [Memory::GPU_HBM],
  transfer Memory::CPU_DRAM -> Memory::GPU_HBM : 63 GB/s
}

fn main() -> i32 {
  let mut q_h = Tensor<f16, [128, 64]>::uninit();
  let mut k_h = Tensor<f16, [128, 64]>::uninit();
  let mut v_h = Tensor<f16, [128, 64]>::uninit();
  let mut o_h = Tensor<f16, [128, 64]>::uninit();

  let q = transfer(q_h, Memory::GPU_HBM);
  let k = transfer(k_h, Memory::GPU_HBM);
  let v = transfer(v_h, Memory::GPU_HBM);
  let mut o = transfer(o_h, Memory::GPU_HBM);

  spawn on(Topology::Dev) {
    flash_attention_into(&mut o, &q, &k, &v, 0.125);
  }
  return 0;
}
```

The compiler classifies that region as `kind=attention`, stamps the operand roles onto it, and the
runtime `dlopen`s whatever provider `VX_FLASH_LIB` names — a FlashAttention-2 shim or cuDNN's
fused SDPA, both behind one symbol (`vx_flash_fwd_f16_hd64`). The 0.067 ms is NVIDIA's code,
reached by one line of Vx. The call takes rank-2 `f16` tensors only.

> Warm the shape before anyone is watching. The first call at a given shape pays for the provider's
> setup, and the two providers differ by two orders of magnitude in what that costs: the cuDNN
> provider spent **511 ms** building its plan for the shape above before settling at 0.067, while
> the FlashAttention-2 shim's one-off `dlopen` and module load is about 4 ms. Both are per process,
> and cuDNN's is per shape. A benchmark that reports its first region is measuring the plan
> builder.

Building the cuDNN provider needs one correction to the recipe in its own header comment: link
`-lnvrtc` as well, or the library loads under `RTLD_LAZY` and the runtime's `RTLD_NOW` refuses it.

Without the library, the same binary still answers: the compiler always emits the fallback nest
alongside, and a provider that cannot be found costs performance rather than correctness. That is
the same contract the whole dispatch path keeps, and it is why the correctness section of this
chapter comes before the hardware section.

## What this chapter did not do

- Batch and head dimensions. The kernel here is a single head; `flash_attention_into` is rank-2.
- The backward pass. `tests/backend/pass/flash_attention_backward.vx` has one.
- Tuning the cooperative kernel past its first configuration.

The programs are all in `scripts/campaigns/flash/`, parameterized by shape, with the runner that
says where they ran.
