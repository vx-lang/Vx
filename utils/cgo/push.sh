#!/usr/bin/env bash
#===- push.sh - Vx Compiler ----------------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Copy this working tree to a measurement instance, and pull results back.
# **Runs locally**, not on the instance.
#
#   utils/cgo/push.sh ubuntu@1.2.3.4                     # push the tree
#   utils/cgo/push.sh -i ~/.ssh/vx.pem ubuntu@1.2.3.4 --build
#   utils/cgo/push.sh ubuntu@1.2.3.4 --pull              # bring results home
#
# The instance never needs a git credential. The only key involved is the EC2
# keypair, used here for ssh/rsync in this direction -- so no personal SSH key,
# GitHub deploy key or token is ever copied to a machine that is going to be
# terminated. That is the reason this script exists rather than `git clone` on
# the instance (`ec2_setup.sh --clone` still works if a public HTTPS clone is
# what you want).
#
# What crosses: exactly `git ls-files`, minus `config.local`. That excludes
# `target/` (tens of gigabytes), `.rustup/` (a macOS-arch toolchain, useless
# there) and `.git/`, without needing a hand-written exclude list that drifts.
# Uncommitted edits *are* sent -- iterating on a remote box is the normal case --
# and the stamp file records whether the tree was dirty, so a result can always
# be traced to a state.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

DEST="${DEST:-/opt/vx}"
SSH_KEY=""
DO_BUILD=0
DO_PULL=0
HOST=""

while [ $# -gt 0 ]; do
    case "$1" in
        -i) SSH_KEY="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --pull) DO_PULL=1; shift ;;
        -h|--help) sed -n '10,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "unknown flag: $1" >&2; exit 2 ;;
        *) HOST="$1"; shift ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "usage: $0 [-i key.pem] [--dest DIR] [--build|--pull] user@host" >&2
    exit 2
fi

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

SSH_OPTS=()
[ -n "$SSH_KEY" ] && SSH_OPTS=(-i "$SSH_KEY")
SSH=(ssh "${SSH_OPTS[@]}" -o StrictHostKeyChecking=accept-new)

if [ "$DO_PULL" -eq 1 ]; then
    # Results are timestamped directories, so this merges rather than overwrites
    # and pulling twice is harmless.
    mkdir -p utils/cgo/results
    echo "== pulling results from $HOST:$DEST/utils/cgo/results =="
    "${SSH[@]}" "$HOST" "cd '$DEST/utils/cgo' && tar -czf - results 2>/dev/null" \
        | tar -xzf - -C utils/cgo
    ls -1 utils/cgo/results
    exit 0
fi

# `git ls-files` is the manifest: tracked files only, so every ignored artifact
# is excluded by construction rather than by a list someone has to maintain.
# `config.local` is dropped deliberately -- it hardcodes this machine's Homebrew
# LLVM and an in-repo macOS toolchain, and sourcing it on the instance would
# point the build at paths that do not exist. A Linux one is written below.
FILELIST="$(mktemp)"
trap 'rm -f "$FILELIST"' EXIT
git ls-files | grep -v '^config\.local$' > "$FILELIST"

COMMIT="$(git rev-parse HEAD)"
DIRTY="clean"
git diff --quiet && git diff --cached --quiet || DIRTY="DIRTY (uncommitted changes included)"

echo "== pushing $(wc -l < "$FILELIST" | tr -d ' ') files to $HOST:$DEST =="
echo "   commit $COMMIT ($DIRTY)"

"${SSH[@]}" "$HOST" "sudo mkdir -p '$DEST' && sudo chown \$(id -u):\$(id -g) '$DEST'"

# tar over ssh rather than rsync. The whole tracked tree is ~1.6 MB compressed,
# so incremental transfer buys nothing worth a portability question -- and there
# is a real one: macOS 15 ships openrsync as /usr/bin/rsync, which does not
# implement the same flag set, so `--files-from` is not something to assume on a
# machine you did not configure. `tar -T -` and `ssh` are everywhere.
#
# Sources are *replaced*, not merged: a file deleted locally would otherwise
# linger on the instance and keep compiling. Two things survive the wipe -- the
# build directory, because relinking against LLVM from scratch costs minutes,
# and the results, which are the entire point of the machine. The results sit at
# utils/cgo/results, i.e. *inside* a directory the wipe removes, so they are
# moved aside first rather than trusted to an exclusion that cannot express them.
"${SSH[@]}" "$HOST" "cd '$DEST' \
    && rm -rf .results-keep \
    && { [ -d utils/cgo/results ] && mv utils/cgo/results .results-keep || true; } \
    && find . -mindepth 1 -maxdepth 1 ! -name target ! -name .git ! -name .results-keep \
         -exec rm -rf {} + 2>/dev/null || true"

tar -czf - -T "$FILELIST" | "${SSH[@]}" "$HOST" "tar -xzf - -C '$DEST'"

"${SSH[@]}" "$HOST" "cd '$DEST' && if [ -d .results-keep ]; then \
    mkdir -p utils/cgo/results && cp -R .results-keep/. utils/cgo/results/ && rm -rf .results-keep; fi"

# The stamp travels with the tree so a results directory copied off the box is
# still attributable to a source state. A dirty tree is recorded as dirty rather
# than silently presented as its last commit.
"${SSH[@]}" "$HOST" "cat > '$DEST/PUSHED_FROM' <<EOF
commit: $COMMIT
state:  $DIRTY
pushed: \$(date -u +%Y-%m-%dT%H:%M:%SZ) (instance clock)
from:   $(hostname)
EOF"

# The instance's own `config.local`, so the `source config.local` habit works
# there too. Deliberately does not set CARGO_HOME/RUSTUP_HOME: rustup installs
# to \$HOME on the instance, and pointing those into the repo would mean rsync
# fighting with cargo over the same directory.
"${SSH[@]}" "$HOST" "cat > '$DEST/config.local' <<'EOF'
# Generated by utils/cgo/push.sh for this instance. Not the macOS config.local.
export PATH=\"/usr/lib/llvm-22/bin:\$PATH\"
export LLVM_CONFIG_PATH=llvm-config-22
export RUSTFLAGS=\"-C link-arg=-fuse-ld=lld\"
[ -f \"\$HOME/.cargo/env\" ] && . \"\$HOME/.cargo/env\"
EOF"

echo "pushed."

if [ "$DO_BUILD" -eq 1 ]; then
    echo "== remote build =="
    "${SSH[@]}" "$HOST" "cd '$DEST' && . ./config.local && cargo build --release --bin intern_bench"
fi

echo
echo "next:"
echo "  ssh ${SSH_KEY:+-i $SSH_KEY} $HOST"
echo "  cd $DEST && . ./config.local && utils/cgo/run_e1.sh"
echo "then, from here:"
echo "  utils/cgo/push.sh ${SSH_KEY:+-i $SSH_KEY} $HOST --pull"
