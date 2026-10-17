# Your first afternoon with Vx

This page is for a hackathon or a free afternoon: install Vx, run a program on a declared GPU,
see the compiler catch two mistakes, then pick a project. You do not need a GPU.

## Install

```bash
curl -fsSL https://vxlang.org/install.sh | sh
export PATH="$HOME/.vx/bin:$PATH"
vxc --version
```

The installer puts the compiler in `~/.vx/bin` and the machine files in `~/.vx/current/fleet`.
It does not change your shell profile: add the `export` line to `~/.zshrc` or `~/.bashrc` to
keep it. If something is missing, the installer prints the command to install it.

## A matmul on an H100

Save this as `matmul.vx`:

```rust
fn main() -> i32 {
  let a = Tensor<f32, [4, 4]>::fill(2.0);
  let b = Tensor<f32, [4, 4]>::fill(3.0);
  let mut c = Tensor<f32, [4, 4], Memory::HBM>::uninit();

  let a_gpu = transfer(a, Memory::HBM);
  let b_gpu = transfer(b, Memory::HBM);

  spawn on(Topology::Device) {
    c = a_gpu @ b_gpu;
  }

  let c_host = transfer(c, Memory::CPU_DRAM);
  print(c_host[0][0]);
  return 0;
}
```

Run it against the H100 machine file:

```bash
vxc --host default --machine ~/.vx/current/fleet/h100-sxm.vx matmul.vx
```

It prints `24`. The machine file says what the H100 has: 80 GiB of HBM that the host cannot read,
and a 63 GB/s link to it. The compiler checks the program against that. On a computer without the
GPU, the code in `spawn on` runs on the host.

Three things in the program are new:

- `Memory::HBM` in the tensor's type says where `c` lives.
- `transfer` moves a tensor from one memory to another.
- `spawn on` says which device runs a block of code.

## Mistake 1: forget the transfer back

Delete the `let c_host = ...` line and print `c` directly:

```rust
  print(c[0][0]);
```

The program no longer compiles:

```
Error[E6003] at 13:9: 'c' lives in HBM but CPU sees only [CPU_DRAM, NPU_HBM]; insert an explicit transfer to CPU_DRAM (cost 1 on the declared path)
```

In CUDA, the same mistake is an illegal memory access when the program runs. The error names the
transfer that fixes it.

## Mistake 2: a tensor that does not fit

Save this as `too_big.vx`:

```rust
fn main() -> i32 {
  // 200,000 x 200,000 f32 values: 160 GB.
  let weights = Tensor<f32, [200000, 200000], Memory::HBM>::uninit();
  let back = transfer(weights, Memory::CPU_DRAM);
  print(back[0][0]);
  return 0;
}
```

```bash
vxc --host default --machine ~/.vx/current/fleet/h100-sxm.vx too_big.vx
```

```
Error[E6009]: variable 'weights' needs 160000000000 bytes but memory space 'HBM' has capacity 85899345920 bytes
```

The capacity comes from the machine file, not from your computer.

## Pick a project

Each of these fits in an afternoon.

**Describe your own machine.** Write a machine file for your laptop, your workstation or a
cluster you use. Start from a file in `~/.vx/current/fleet` that is close to it, and read
[Machine files](machine-files.md). Then check `matmul.vx` against it.

**Port a kernel.** Write softmax, layer norm or attention in Vx, place its tensors, and see what
the checker says. [A tour of Vx](tour.md) and [Topologies and memory](heterogeneous.md) cover
the language.

**Break the checker.** Write a program that reads memory it should not, or uses more memory than
the machine has, and that the compiler accepts anyway. That is a real bug: file it at
[github.com/vx-lang/Vx/issues](https://github.com/vx-lang/Vx/issues). The
[Diagnostic index](error-index.md) lists every check.

## Good to know

- The built-in host is modelled on a chip whose host and NPU share memory, so the host may read
  `Memory::NPU_HBM`. Use a machine file such as `h100-sxm.vx` to work with GPU memory the host
  cannot read.
- Vx is young. If the compiler crashes or a message is unclear, that is a bug worth filing.
- Questions: ask on [Discord](https://discord.gg/et5GJCKj9p).
