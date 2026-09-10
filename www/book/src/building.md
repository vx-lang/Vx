# Building from source

You need this if you are on a platform with no prebuilt toolchain — Linux on arm64, or an Intel Mac
— or if you intend to work on the compiler itself.

Everything below has been run on the platform it documents.

## What the build needs

| Requirement | Why |
| --- | --- |
| **LLVM/MLIR 22** | Pinned by `mlir-sys` in `Cargo.toml`. A different major version fails to link with undefined MLIR C-API symbols. |
| **Rust (stable)** | The compiler is written in Rust. |
| **z3** — the binary | Seam verification executes it. The library alone is not enough. |
| **libffi** | The dispatch runtime calls outlined kernels through their MLIR C interface. |

Plus `cmake`, `ninja`, `pkg-config`, `python3` and a C++ compiler.

## macOS (Apple Silicon)

```bash
xcode-select --install

brew install llvm z3 cmake ninja pkg-config python3 rustup
rustup-init -y --default-toolchain stable
```

Check that Homebrew's LLVM is the pinned major version:

```bash
$(brew --prefix llvm)/bin/llvm-config --version    # expect 22.x
```

If Homebrew has moved on, install the pinned one with `brew install llvm@22` and run
`LLVM_VERSION=22 ./setup.sh`.

## Linux x86_64 (Ubuntu 24.04)

`scripts/provision/setup_linux.sh` installs everything:

```bash
./scripts/provision/setup_linux.sh
```

It installs build tooling (`build-essential`, `ninja-build`, `cmake`, `pkg-config`), libraries
(`libffi-dev`, `libz3-dev`, `z3`, `zlib1g-dev`, `libzstd-dev`, `libedit-dev`, `libxml2-dev`), and
LLVM 22 from `apt.llvm.org` — Ubuntu's own repositories lag the pinned version.

**`libmlir-22-dev` is the package people miss.** It ships the MLIR C API that the Rust bindings link
against, and it is not pulled in by `llvm-22-dev`.

Rust is installed separately by the script, via rustup.

Sizing, if you are provisioning a box for this: 40 GB of disk (a release build leaves `target/` at
about 2 GB, and `cargo test` adds a second profile), and as many cores as you can get. A reference
release build takes about 1m48s on 36 vCPU with a warm crate cache.

## Configure and build

Identical on both platforms:

```bash
./setup.sh           # writes config.local -- LLVM path, cargo/rustup homes, PATH
source config.local  # required in every new shell, before any cargo command
cargo build --release
```

`setup.sh` puts LLVM *first* on `PATH`, so an unsuffixed `llvm-config`, `mlir-translate` or
`clang++` resolves to the pinned version rather than to Xcode's or to another LLVM on the box. It
warns if what it found does not match the pin.

If you get `llvm-config: command not found` or `cargo: command not found`, you opened a new shell
and did not `source config.local`. It is once per shell, every shell.

## Verify

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

Expect the last line to read `[JIT] Program exited with code: 42`. Then run the suite:

```bash
cargo test
```

## Optional components

The build succeeds without any of these; each unlocks one path.

**Autodiff (Enzyme)** — needed for the autodiff tests and for `grad`/`jvp`/`vjp` lowering:

```bash
./scripts/provision/install_enzyme.sh
export ENZYME_LIB="$(pwd)/.cargo/enzyme/LLVMEnzyme-22.dylib"   # .so on Linux
```

**Apple neural engine primitives (macOS only)** — `build.rs` compiles CoreML primitive models, and
invokes bare `python3`, so `coremltools` has to be importable from *that* interpreter:

```bash
python3 -m pip install coremltools
```

Without it the build prints `warning: Failed to compile matmul_4x4 with coremlc` and continues.
Kernels fall back to CPU execution; only accelerator dispatch is unavailable.

**CUDA (Linux)** — detected automatically when `/usr/local/cuda` is present. Set `VX_DISABLE_CUDA=1`
to force it off. The toolkit alone is enough to build; a GPU is only needed to run.

## Packaging a toolchain

To produce a redistributable tarball from a source build:

```bash
./scripts/release/package.sh v0.0.1
```

That stages the compiler, its runtime library, the standard library and the machine files under
`dist/`, rewrites the linked library paths so the binaries find their own dependencies, generates
the wrapper scripts, and writes a `.tar.gz` alongside its SHA-256.
