#!/usr/bin/env bash
set -e

PROJECT_DIR=$(pwd)

echo "Locating LLVM installation..."
if command -v brew >/dev/null 2>&1 && brew --prefix llvm >/dev/null 2>&1; then
    LLVM_PREFIX=$(brew --prefix llvm)
    LLVM_PATH="$LLVM_PREFIX/bin"
elif command -v llvm-config >/dev/null 2>&1; then
    LLVM_PATH=$(dirname $(command -v llvm-config))
elif [ -d "/opt/homebrew/opt/llvm/bin" ]; then
    LLVM_PATH="/opt/homebrew/opt/llvm/bin"
else
    echo "Could not automatically locate LLVM. Please install LLVM or manually create config.local based on config.template."
    exit 1
fi

echo "LLVM found at: $LLVM_PATH"

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
    config.template > config.local

echo "config.local successfully generated! You can now source config.local to load the toolchain environment."
