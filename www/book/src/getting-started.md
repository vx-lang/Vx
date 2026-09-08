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
| Linux x86_64 (glibc 2.35+) | Supported. CUDA dispatch when a toolkit is present. |
| Intel macOS | Not supported — the runtime assumes Apple Silicon. |
| Linux arm64, Windows | No prebuilt toolchain. [Build from source](building.md). |

## Prerequisites

The installer checks for these and stops with the exact command to run if either is missing. It
does **not** install them for you: a script piped into a shell should not quietly run your package
manager.

### LLVM 22

Vx lowers through MLIR, and shells out to `mlir-translate`, `opt`, `llc` and `clang` from LLVM 22
when it compiles. The version matters — the MLIR C API changes between major releases, so LLVM 21
or 23 will not work.

```bash
# macOS
brew install llvm@22

# Ubuntu / Debian
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22
sudo apt-get install -y libmlir-22-dev mlir-22-tools
```

`libmlir-22-dev` is the package people miss. It ships the MLIR C API and is *not* pulled in by
`llvm-22-dev`.

On macOS, Homebrew's LLVM is keg-only, so it is deliberately not on your `PATH`. You do not need to
change that — the installer records where it found LLVM, and the `vxc` wrapper points the compiler
at it directly.

### z3 (optional)

Seam verification — proving that an asynchronous `transfer` has been made visible before the buffer
is read — is discharged by shelling out to `z3`. It is opt-in behind `--verify-seams`, so a
toolchain without z3 compiles and runs everything else normally; you only need it when you turn that
flag on.

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

Then compile something. Note that `--run` propagates *the program's own* exit code, so a non-zero
exit status here is your program's return value rather than a failure:

```bash
cat > hello.vx <<'EOF'
fn main() -> i32 {
    let x : i32 = 21;
    return x * 2;
}
EOF

vxc --run hello.vx
```

The last line should read:

```
[JIT] Program exited with code: 42
```

## Installing a specific version

```bash
VX_VERSION=v0.1.0 curl -fsSL https://vxlang.org/install.sh | sh
```

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
