# Writing a flash attention kernel

This chapter writes a fused attention forward pass and runs it on an NVIDIA A100. The kernel is
tiled, keeps a running softmax, and never materializes the score matrix — the FlashAttention
shape.

The point of the chapter is not the kernel. It is *where the mistakes happen*. Every error in the
sections below is reported by `vxc` on a laptop, before any hardware is rented: a tile that does
not fit on chip, a buffer the device cannot address, a shared-memory race that no barrier can
order. Those are the failures that normally cost an afternoon of `cuda-memcheck` on a machine
billed by the hour. Here they cost a rebuild.

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

  print(o);
  return 0;
}
```

Two things in that program are placement rather than arithmetic. The four `transfer` calls move
the operands into `GPU_HBM`, and each one is checked against the space's capacity and charged a
cost derived from its declared bandwidth. The `spawn on(Topology::GPU)` block is the region that
will become a device kernel; reading `q_h` instead of `q` inside it is a compile error naming the
space the value is in and the transfer that fixes it.

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
$ VX_DISPATCH_VERBOSE=1 ./vxc flash.vx
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] device 0 stage
[Vx CUDA] vx_npu_kernel_0 ran on GPU 0 from its own image
[[2.84482,   2.84482,   2.84482,   2.84482],
 [1.5,   1.5,   1.5,   1.5]]
```

Four stages for the four operands, one launch of a kernel the compiler emitted, and the same two
rows that printed on the laptop.

> `spawn on` places a region; it does not promise the region reaches the device. A region the
> backend cannot lower is refused and run on the host, which costs performance and never
> correctness. Read the trace rather than assuming: `run_flash_bench.sh` greps for exactly this
> and prints a verdict, because a CPU number and a GPU number look identical in a table.

To run it at a real size on a rented machine, `scripts/provision/bootstrap_dev_pod.sh` ships the
tree, builds the compiler, and runs the sweep:

```console
$ scripts/provision/bootstrap_dev_pod.sh -h root@<pod> -p <port> -i ~/.ssh/<key>
$ scripts/campaigns/flash/run_flash_bench.sh -q 8192 -d 64 -k "512 1024 2048 4096"
```

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

| | time | rate |
| --- | --- | --- |
| The fused kernel above, fp32 | 1.125 ms | 3.82 TF/s |
| Unfused, two routed TF32 GEMMs | 0.486 ms | 8.84 TF/s |
| Unfused, two routed f16 GEMMs | 0.374 ms | 11.5 TF/s |
| `flash_attention_into`, routed to FlashAttention-2 | 0.066 ms | ~65 TF/s |
| PyTorch SDPA on the same machine | 0.063 ms | |

The first row and the last are not the same workload: the kernel in this chapter is fp32, and the
routed path is f16 on tensor cores. The gap is mostly that.

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
fused SDPA, both behind one symbol. The 0.066 ms is NVIDIA's code, reached by one line of Vx.
The call takes rank-2 `f16` tensors only.

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
