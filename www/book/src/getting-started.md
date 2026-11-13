# Install Vx

## Quick install

```bash
curl -fsSL https://vxlang.org/install.sh | sh
```

This downloads a prebuilt toolchain, verifies it against its published SHA-256, and unpacks it into
`~/.vx`. Nothing is written outside that directory and nothing needs root.

Then put it on your `PATH`:

```bash
export PATH="$HOME/.vx/bin:$PATH"
```

Add that line to `~/.zshrc` or `~/.bashrc` to make it permanent.

## Supported platforms

| Platform | Status |
| --- | --- |
| macOS on Apple Silicon (M1–M4) | Supported. Apple accelerator dispatch available. |
| Linux x86_64 (glibc 2.35+) | Supported. CUDA dispatch needs the NVIDIA driver and CUDA 12. |
| Intel macOS | Not supported — the runtime assumes Apple Silicon. |
| Linux arm64, Windows | No prebuilt toolchain. [Build from source](building.md). |

On a machine with the NVIDIA driver, the installer takes the CUDA build. That build needs the
CUDA 12 runtime and cuBLAS 12. When they are missing, the installer takes the build without CUDA
and says so. `VX_NO_CUDA=1` always takes the build without CUDA.

## Prerequisites

The installer checks for these and stops with the exact command to run if one is missing. It
does **not** install them for you: a script piped into a shell should not quietly run your package
manager.

### LLVM 22 (macOS, and Linux before v0.0.3)

Vx lowers through MLIR, and shells out to `mlir-translate`, `opt`, `llc` and `clang` from LLVM 22
when it compiles. From v0.0.3, the Linux toolchains include these, so there is nothing to install
on Linux. On macOS you install LLVM 22 yourself. The version matters — the MLIR C API changes
between major releases, so LLVM 21 or 23 will not work.

```bash
# macOS
brew install llvm@22

# Ubuntu / Debian, only for a release before v0.0.3
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22
sudo apt-get install -y libmlir-22-dev mlir-22-tools
```

`libmlir-22-dev` is the package people miss. It ships the MLIR C API and is *not* pulled in by
`llvm-22-dev`.

On macOS, Homebrew's LLVM is keg-only, so it is deliberately not on your `PATH`. You do not need to
change that — the installer records where it found LLVM, and the `vxc` wrapper points the compiler
at it directly.

### z3

The compiler runs `z3` to prove contracts: `requires`, `ensures`, `invariant` and `Verified<T>`.
`--verify-seams` uses it too. Without z3, a program with a contract does not compile.

The **binary** is executed, so installing `libz3-dev` alone is not enough.

```bash
brew install z3          # macOS
sudo apt-get install z3  # Ubuntu / Debian
```

## Verify the install

The installer compiles and runs a small program as its last step, so if it finished without
complaint you are already working. To check by hand:

```bash
vxc --version
```

This prints the version, and for a release build also the commit it was built from, such as
`vxc 0.0.3 (f1fb0333)`. To get the commit in a build of your own, set `VX_VERSION_WITH_COMMIT=1`
when you run `cargo build`. Keep that line with anything you report: two builds of the same
version can generate different code.

Then compile something. A program that returns a value other than 0 makes `vxc` exit with status
1 and print the value. Here that is the program's return value, not a failure:

```bash
cat > hello.vx <<'EOF'
fn main() -> i32 {
    let x : i32 = 21;
    return x * 2;
}
EOF

vxc --run hello.vx
```

The last two lines should read:

```
[JIT] Program exited with code: 42
Program exited with non-zero code: 42
```

## Installing a specific version

```bash
curl -fsSL https://vxlang.org/install.sh | VX_VERSION=v0.0.4 sh
```

`VX_VERSION` goes after the `|`, so that it is set for the `sh` that runs the installer.

Toolchains are unpacked side by side under `~/.vx/toolchains/`, and `~/.vx/current` is a symlink to
the active one, so switching versions is a matter of repointing that link.

## Uninstalling

```bash
rm -rf ~/.vx
```

Then remove the `PATH` line from your shell profile. The installer puts nothing anywhere else.

## Troubleshooting

**`vxc: command not found`**
`~/.vx/bin` is not on your `PATH`. See the export line above.

**`Failed to run mlir-translate: No such file or directory`**
The compiler cannot find the LLVM tools. This normally means LLVM was installed *after* Vx, so the
installer never recorded its location. Re-run the installer — it will find LLVM and rewrite
`~/.vx/current/etc/llvm-env.sh`.

**`the Vx runtime library is missing`**
The toolchain directory is incomplete, usually from an interrupted download. Re-run the installer.

**Undefined MLIR symbols, or a crash on startup**
The LLVM on your system is not version 22. Check with `llvm-config --version`, and note that on
macOS a bare `llvm-config` usually resolves to Xcode's copy rather than Homebrew's.

**macOS refuses to run the binary**
Gatekeeper quarantines downloads from the internet. Clear the attribute:

```bash
xattr -dr com.apple.quarantine ~/.vx
```

## Next

- [Your first program](first-program.md) — write something real.
- [A tour of Vx](tour.md) — the language in one sitting.
