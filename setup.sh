#!/usr/bin/env bash
set -e

PROJECT_DIR=$(pwd)

# The LLVM major version mlir-sys is pinned against -- `mlir-sys = "220.x"` means LLVM 22.
# Keep in sync with Cargo.toml and scripts/setup_linux.sh.
LLVM_VERSION="${LLVM_VERSION:-22}"

echo "Locating LLVM installation..."
# The VERSIONED name is tried before the unsuffixed one on purpose. apt.llvm.org installs
# `llvm-config-22` and no unsuffixed `llvm-config` at all, so on Linux every branch this script
# used to have missed: no brew, no `llvm-config` on PATH, no /opt/homebrew. setup_linux.sh would
# install LLVM 22 correctly and then tell you to run this script, which could not find it.
#
# Preferring the versioned name also picks the PINNED LLVM on a box carrying several. This machine
# had llvm-18 and llvm-22 side by side; an unsuffixed `llvm-config` pointing at 18 would have
# produced a config.local that links against the wrong major version, which surfaces much later as
# undefined MLIR C-API symbols.
if command -v brew >/dev/null 2>&1 && brew --prefix llvm >/dev/null 2>&1; then
    LLVM_PREFIX=$(brew --prefix llvm)
    LLVM_PATH="$LLVM_PREFIX/bin"
elif command -v "llvm-config-${LLVM_VERSION}" >/dev/null 2>&1; then
    # --bindir, not dirname: /usr/bin/llvm-config-22 is a shim, and the directory that holds the
    # UNSUFFIXED clang++/mlir-translate is /usr/lib/llvm-22/bin. config.template puts this first on
    # PATH precisely so those unsuffixed names resolve to the pinned version.
    LLVM_PATH="$("llvm-config-${LLVM_VERSION}" --bindir)"
elif [ -d "/usr/lib/llvm-${LLVM_VERSION}/bin" ]; then
    LLVM_PATH="/usr/lib/llvm-${LLVM_VERSION}/bin"
elif command -v llvm-config >/dev/null 2>&1; then
    LLVM_PATH=$(dirname $(command -v llvm-config))
elif [ -d "/opt/homebrew/opt/llvm/bin" ]; then
    LLVM_PATH="/opt/homebrew/opt/llvm/bin"
else
    echo "Could not automatically locate LLVM ${LLVM_VERSION}."
    echo "  macOS: brew install llvm"
    echo "  Linux: ./scripts/setup_linux.sh"
    echo "Or create config.local by hand from config.template."
    exit 1
fi

# Found a path, but not necessarily the right version. This used to go unchecked, and a mismatch
# only announced itself as undefined MLIR symbols at link time.
FOUND_VERSION="$("$LLVM_PATH/llvm-config" --version 2>/dev/null | cut -d. -f1)"
if [ -n "$FOUND_VERSION" ] && [ "$FOUND_VERSION" != "$LLVM_VERSION" ]; then
    echo "WARNING: found LLVM $FOUND_VERSION at $LLVM_PATH, but mlir-sys is pinned to $LLVM_VERSION." >&2
    echo "         The build will fail to link. Set LLVM_VERSION=$FOUND_VERSION only if you also" >&2
    echo "         changed the mlir-sys pin in Cargo.toml." >&2
fi

echo "LLVM found at: $LLVM_PATH"

# Homebrew's LLVM reports the extra system libraries it needs -- ask it with
# `llvm-config --system-libs` and it names things like -lzstd and -lxml2. Those libraries live in
# Homebrew's own lib directory, and macOS does not look there by default. Any crate that hands
# those flags to the linker then fails with "ld: library 'zstd' not found".
#
# Which libraries get named depends on how that particular LLVM was built, so this breaks on one
# machine and not the next. LLVM 22.1.4 from the `llvm` formula names none; 22.1.8 from `llvm@22`
# names three. A developer on the first version sees nothing wrong, and CI on the second cannot
# link at all.
#
# Putting the directory on LIBRARY_PATH is enough to fix it, and it costs nothing when the
# libraries were already findable. Linux packages install theirs where the linker already looks,
# so there is nothing to add there.
if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    BREW_LIB="$(brew --prefix)/lib"
    LINK_PATH_EXPORT="export LIBRARY_PATH=\"$BREW_LIB\${LIBRARY_PATH:+:\$LIBRARY_PATH}\""
    echo "Homebrew libraries at: $BREW_LIB"
else
    LINK_PATH_EXPORT="# Not a Homebrew machine, so nothing to add: the libraries LLVM asks for are already on the linker's default path."
fi

# Two separate questions, which this script used to answer as one.
#
# CARGO_HOME/RUSTUP_HOME say where cargo keeps its *state*: the registry cache,
# .cargo/config.toml with the repository's aliases, the extra subcommands in
# .cargo/bin, and the toolchains rustup manages. This checkout carries its own,
# deliberately, and that is what config.template names.
#
# PATH says where the `cargo` *binary* is, which is a different directory and
# frequently a different tree -- the in-repo .cargo/bin holds cargo-format and
# cargo-fuzz but no cargo, because rustup's shims live wherever rustup was
# installed. config.template never set it at all, so sourcing config.local
# exported CARGO_HOME and left `cargo` unresolvable. On a developer machine
# nobody noticed, because the shell rc had already put it on PATH. On a box
# freshly provisioned by scripts/setup_linux.sh -- which installs to rustup's
# default $HOME/.cargo -- `source config.local && cargo build` fails with
# "cargo: command not found".
if [ -d "$PROJECT_DIR/.cargo" ] && [ -d "$PROJECT_DIR/.rustup" ]; then
    CARGO_DIR="$PROJECT_DIR/.cargo"
    RUSTUP_DIR="$PROJECT_DIR/.rustup"
else
    CARGO_DIR="$HOME/.cargo"
    RUSTUP_DIR="$HOME/.rustup"
fi

# Whichever of the two trees actually holds the driver. Both go on PATH so the
# repository's own subcommands stay reachable when the driver comes from $HOME.
if [ -x "$PROJECT_DIR/.cargo/bin/cargo" ]; then
    CARGO_BIN="$PROJECT_DIR/.cargo/bin"
elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO_BIN="$HOME/.cargo/bin:$PROJECT_DIR/.cargo/bin"
else
    echo "Could not find a cargo binary in $PROJECT_DIR/.cargo/bin or $HOME/.cargo/bin." >&2
    echo "Install Rust -- scripts/setup_linux.sh does this -- and re-run." >&2
    exit 1
fi

echo "Cargo state at: $CARGO_DIR"
echo "Cargo binary on: $CARGO_BIN"

echo "Generating config.local from config.template..."
sed -e "s|{{PROJECT_DIR}}|$PROJECT_DIR|g" \
    -e "s|{{CARGO_DIR}}|$CARGO_DIR|g" \
    -e "s|{{RUSTUP_DIR}}|$RUSTUP_DIR|g" \
    -e "s|{{CARGO_BIN}}|$CARGO_BIN|g" \
    -e "s|{{LLVM_PATH}}|$LLVM_PATH|g" \
    -e "s|{{LINK_PATH_EXPORT}}|$LINK_PATH_EXPORT|g" \
    config.template > config.local

echo "config.local successfully generated! You can now source config.local to load the toolchain environment."
