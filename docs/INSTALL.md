# Installing Vx

Build instructions for the two supported development platforms. Every command below was executed on
the platform it documents.

| Platform | Status |
| --- | --- |
| **macOS on Apple Silicon** (M1–M4) | Primary development target. ANE/AMX dispatch available. |
| **Linux x86_64** (Ubuntu 24.04) | Supported. Used for the GPU build boxes. No ANE. |

Intel Macs are not supported: the ANE/AMX dispatch path and the `arm64` runtime assume Apple Silicon.

______________________________________________________________________

## What the build needs, and why

Four things matter on both platforms. The first is the one that goes wrong.

| Requirement | Why |
| --- | --- |
| **LLVM/MLIR 22** | Pinned by `mlir-sys = "220.0.2"` in `Cargo.toml` — `220.x` means LLVM 22. A different major version fails to link, with undefined MLIR C-API symbols. |
| **Rust** (stable) | The compiler is Rust. |
| **`z3` — the binary, not just the library** | Seam verification shells out to it (`Command::new("z3")` in `src/hir/seam.rs`) to discharge `relaxed` transfer obligations in QF_BV. Without it four suites fail in ways that read as compiler regressions rather than as a missing tool. |
| **libffi** | The dispatch runtime calls outlined kernels through their MLIR C-interface (`runtime/host_dispatch.cpp`). |

Plus `cmake`, `ninja`, `pkg-config`, `python3` and a C++ compiler.

______________________________________________________________________

## macOS (Apple Silicon)

```bash
# 1. Xcode command line tools
xcode-select --install

# 2. Homebrew, if you do not have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Dependencies
brew install llvm z3 cmake ninja pkg-config python3 rustup
rustup-init -y --default-toolchain stable
```

Confirm Homebrew's LLVM is the pinned major version:

```bash
$(brew --prefix llvm)/bin/llvm-config --version    # expect 22.x
```

If Homebrew has moved to a later major, install the pinned one with `brew install llvm@22` and run
`LLVM_VERSION=22 ./setup.sh`.

Then [configure and build](#configure-and-build).

______________________________________________________________________

## Linux x86_64 (EC2 or any Ubuntu 24.04 box)

### Instance sizing

| | Recommendation | Measured |
| --- | --- | --- |
| Instance | `c6i.8xlarge` or larger | built on 36 vCPU / 58 GB |
| Disk | **40 GB** gp3 | `target/` is 2.0 GB after a release build; `cargo test` adds a second profile |
| AMI | Ubuntu 24.04 LTS (noble) | 24.04.4 |

### Packages

`scripts/provision/setup_linux.sh` installs all of the following. Run it, or install them by hand if your
image is managed:

```bash
# Build tooling
build-essential ninja-build cmake pkg-config git curl ca-certificates gnupg

# Libraries
libffi-dev libz3-dev z3 zlib1g-dev libzstd-dev libedit-dev libxml2-dev

# LLVM/MLIR 22, from apt.llvm.org — Ubuntu's own repositories lag the pinned version
llvm-22 llvm-22-dev llvm-22-tools clang-22 lld-22
libmlir-22-dev mlir-22-tools libpolly-22-dev
```

`libmlir-22-dev` is the one people miss: it ships the MLIR C API that `melior` links against, and
it is not pulled in by `llvm-22-dev`.

Rust is installed separately by the script via `rustup` (stable, minimal profile).

```bash
./scripts/provision/setup_linux.sh
```

Override the LLVM major version if the pin ever moves:

```bash
LLVM_VERSION=23 ./scripts/provision/setup_linux.sh
```

### Getting the source onto a box you cannot clone to

If the machine must not have repository credentials, ship a source archive instead. `git archive`
carries tracked files only — no `.git`, no `target/`, ~2 MB:

```bash
# on your workstation
git archive --format=tar.gz -o /tmp/vx-src.tgz HEAD
scp -i <key.pem> /tmp/vx-src.tgz ubuntu@<host>:/tmp/

# on the box
mkdir -p ~/vx && tar xzf /tmp/vx-src.tgz -C ~/vx && cd ~/vx
```

______________________________________________________________________

## Configure and build

Identical on both platforms.

```bash
./setup.sh           # writes config.local: LLVM path, cargo/rustup homes, PATH
source config.local  # required before every cargo command, in every new shell

cargo build --release
```

That one command is enough. `vx_std_core` is declared `crate-type = ["staticlib", "cdylib"]`, and
a dependency edge alone only ever asks for an rlib, so the shared library the JIT loads at run
time used to need a second `cargo build --release -p vx_std_core`. The workspace now names the
package in `default-members`, which covers it. `cargo test` on its own still does not produce it —
test builds never emit a `cdylib` — so a checkout that has only been tested has to be built once.

`setup.sh` puts LLVM *first* on `PATH` so the unsuffixed `llvm-config`, `mlir-translate` and
`clang++` resolve to the pinned version rather than to Xcode's or to another LLVM on the box. It
warns if the version it found does not match the pin.

Reference build: **1m48s** on 36 vCPU, release profile, warm crate cache.

______________________________________________________________________

## Verify the install

`--run` JITs a program and propagates its exit code, so a non-zero exit here is the *program's*
return value, not a failure:

```bash
cat > /tmp/hello.vx <<'EOF'
fn main() -> i32 {
    let x: i32 = 21;
    return x * 2;
}
EOF

source config.local
./target/release/vxc --run /tmp/hello.vx
```

Expected tail:

```
[JIT] Program exited with code: 42
```

Then the suite:

```bash
cargo test
```

______________________________________________________________________

## Optional components

The build succeeds without any of these. Each unlocks one path.

### Autodiff (Enzyme)

Needed only for the autodiff tests and `grad`/`jvp`/`vjp` lowering.

```bash
./scripts/provision/install_enzyme.sh
export ENZYME_LIB="$(pwd)/.cargo/enzyme/LLVMEnzyme-22.dylib"   # .so on Linux
```

### Apple ANE primitives (macOS only)

`build.rs` compiles CoreML primitive models for ANE dispatch. It invokes **bare `python3`**, so
`coremltools` must be importable from *that* interpreter — a virtualenv only counts if it is the
active one:

```bash
python3 -m pip install coremltools
```

Without it the build prints and continues:

```
warning: Failed to compile matmul_4x4 with coremlc
```

Harmless. Kernels fall back to CPU execution through libffi; only ANE dispatch is unavailable.

### CUDA (Linux)

Detected automatically when `/usr/local/cuda` is present, and reported at build time:

```
warning: Built the CUDA dispatch backend against /usr/local/cuda; recognised matmuls run on
         the GPU, everything else on the CPU.
```

Set `VX_DISABLE_CUDA=1` to force it off. The toolkit alone is enough to build — a GPU is only
needed to run.

______________________________________________________________________

## Troubleshooting

**`llvm-config: command not found`, or `cargo: command not found`**
You did not `source config.local`, or you opened a new shell. Once per shell, before any cargo
command.

**`the Vx runtime library is missing: .../libvx_std_core.so`**
The tree has been tested but never built — a test build does not emit the shared library. Run
`cargo build --release`. See [Configure and build](#configure-and-build).

**Undefined MLIR C-API symbols at link time**
The LLVM on `PATH` is not version 22, or `libmlir-22-dev` is missing. Check with
`llvm-config --version` *after* sourcing `config.local`.

**`setup.sh`: "Could not automatically locate LLVM 22"**
macOS: `brew install llvm`. Linux: `./scripts/provision/setup_linux.sh`. If LLVM is installed somewhere
unusual, set `LLVM_VERSION` or write `config.local` by hand from `config.template`.

**Four seam/verification suites fail with missing diagnostics**
`z3` is not installed or not on `PATH`. `libz3-dev` alone is not enough — the *binary* is executed.

**Two `remote_client_test` cases fail on Linux**

`a_placed_tensor_can_be_read_home_from_a_worker` and `a_handoff_between_two_workers_moves_the_data`
both fail with *"the local run printed nothing"*. Everything else passes: 473 lib tests and 229 of
231 integration tests.

This is a real defect, not a platform gap. `src/jit.rs` concatenates the child process's stderr
onto its stdout when `--run` succeeds, so a runtime diagnostic lands in the middle of the program's
output. On a box with the CUDA toolkit but no GPU, the dispatch backend writes

```
[Vx CUDA] no CUDA device (no CUDA-capable device is detected); kernels will run on the host
```

to stderr, which then trails the program's real output with a newline. The tests take the last line
of stdout and get an empty string.

It shows up only on Linux because that is where the CUDA backend is built and finds no device. A
Linux box with no CUDA toolkit, or one with a working GPU, passes. Tracked as a bug against the JIT
output handling — the streams should stay separate.
