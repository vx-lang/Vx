#!/usr/bin/env bash
#===- tsan.sh - Vx Compiler ----------------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Run the parallel frontend under ThreadSanitizer.
#
#   utils/cgo/tsan.sh              # build + sweep a corpus at high thread counts
#   utils/cgo/tsan.sh --tests      # also run the library test suite instrumented
#   utils/cgo/tsan.sh --build-only
#
# The whole design claim is that the frontend is data-race-free *by construction*
# rather than by locking: workers share one frozen, read-only session and own
# everything they mutate. Determinism tests are evidence for that -- identical
# output across thread counts -- but they are not proof, because a race that
# happens to be benign on this scheduler produces identical output too. TSan
# checks the property directly.
#
# Linux only. Rust's ThreadSanitizer support does not cover macOS on ARM, which
# is why this lives next to the EC2 scripts rather than in the normal test path.
#
# ## Why the MLIR link is not a problem
#
# `intern_bench` links libMLIR, and libMLIR is not instrumented -- TSan cannot
# see races inside it. That would normally be a serious gap. It is not one here:
# `compile_pipeline_mlir` produces MLIR *text* in pure Rust and never calls into
# MLIR at all (that is what made #311 tractable), so no uninstrumented code runs
# on the measured path. The link exists; the code does not execute.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO"

TARGET=x86_64-unknown-linux-gnu
MODULES="${MODULES:-200}"
THREADS="${THREADS:-1,8,32,48}"
REPS="${REPS:-1}"
RUN_TESTS=0
BUILD_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --tests) RUN_TESTS=1; shift ;;
        --build-only) BUILD_ONLY=1; shift ;;
        --modules) MODULES="$2"; shift 2 ;;
        --threads) THREADS="$2"; shift 2 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

[ "$(uname -s)" = "Linux" ] || { echo "TSan for Rust is Linux-only here; run this on the measurement box." >&2; exit 1; }

# shellcheck disable=SC1091
[ -f ./config.local ] && . ./config.local

# `-Zbuild-std` is not optional: the standard library ships uninstrumented, and
# an uninstrumented std means TSan cannot see the synchronisation *inside* it --
# every `Arc` clone and every channel operation becomes an invisible edge in the
# happens-before graph, and the missing edges show up as false positives rather
# than as silence.
export RUSTFLAGS="-Zsanitizer=thread -C link-arg=-fuse-ld=lld"

# A suppression file, empty by default. LLVM's static initialisers run before
# main and are not instrumented; if they ever report, add them here rather than
# lowering the exit code, so a suppressed race stays visible in the file.
SUPP="$REPO/utils/cgo/tsan.supp"
[ -f "$SUPP" ] || printf '# TSan suppressions. One "kind:pattern" per line; empty is the goal.\n' > "$SUPP"

# `halt_on_error=0` so one report does not hide the rest -- the question is
# "which races exist", not "is there at least one". `history_size` is raised
# because the default keeps too few prior accesses to name the other side of a
# race in a deep rayon stack, and a report that cannot name both sides cannot be
# acted on.
export TSAN_OPTIONS="halt_on_error=0 history_size=7 second_deadlock_stack=1 suppressions=$SUPP"

echo "== building instrumented (this rebuilds std; it is slow) =="
cargo +nightly build -Zbuild-std --target "$TARGET" --release --bin intern_bench -j "$(( $(nproc) - 2 ))"
BIN="$REPO/target/$TARGET/release/intern_bench"
[ -x "$BIN" ] || { echo "instrumented binary missing" >&2; exit 1; }

# ASLR. TSan maps its shadow memory at fixed addresses, and a kernel with
# high-entropy randomisation can place the program somewhere that collides --
# it then prints "memory layout is incompatible" and its results cannot be
# trusted. Observed intermittently on this host: same binary, same command, most
# runs fine and the occasional one not. `setarch -R` removes the randomness
# rather than relying on winning the draw.
SETARCH=()
command -v setarch >/dev/null 2>&1 && SETARCH=(setarch -R)

# A clean report from a run that warned about its own shadow mapping is not a
# clean report, so that warning is fatal here rather than advisory.
assert_layout_ok() {
    if grep -q "memory layout is incompatible" "$1" 2>/dev/null; then
        echo "FATAL: TSan reported an incompatible memory layout in $1." >&2
        echo "       Its findings (including 'no races') mean nothing. Lower" >&2
        echo "       vm.mmap_rnd_bits to 28, or run under setarch -R." >&2
        exit 1
    fi
}

# The detector has to be shown to work before its silence is worth anything.
# A linked runtime is not a working one: instrumentation can be dropped, the
# shadow map can be wrong, a flag can be mis-set. So build a program whose only
# purpose is to race, with the *same* flags, and require a report. If this stops
# firing, every "0 races" below it is decoration.
echo "== positive control: a deliberate race must be detected =="
CTL="$(mktemp -d)"
trap 'rm -rf "$CTL"' EXIT
mkdir -p "$CTL/src"
printf '[package]\nname="tsanctl"\nversion="0.1.0"\nedition="2021"\n' > "$CTL/Cargo.toml"
cat > "$CTL/src/main.rs" <<'CTLEOF'
fn main() {
    let mut v = 0u64;
    let p = &mut v as *mut u64 as usize;
    let h: Vec<_> = (0..2)
        .map(|_| std::thread::spawn(move || {
            let q = p as *mut u64;
            for _ in 0..1000 { unsafe { *q += 1 }; }
        }))
        .collect();
    for t in h { t.join().unwrap(); }
    std::hint::black_box(v);
}
CTLEOF
CTL_LOG="$CTL/out.log"
( cd "$CTL" && cargo +nightly run -Zbuild-std --target "$TARGET" --release ) > "$CTL_LOG" 2>&1 || true
assert_layout_ok "$CTL_LOG"
if grep -q "WARNING: ThreadSanitizer: data race" "$CTL_LOG"; then
    echo "   detector works (the control raced and was caught)"
else
    echo "FATAL: the positive control did NOT report a race." >&2
    echo "       TSan is not detecting anything in this configuration, so a" >&2
    echo "       clean run below would be meaningless. Not proceeding." >&2
    tail -20 "$CTL_LOG" >&2
    exit 1
fi

[ "$BUILD_ONLY" -eq 1 ] && { echo "built: $BIN"; exit 0; }

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$REPO/utils/cgo/results/tsan-$STAMP"
mkdir -p "$OUT"

echo "== sweeping under TSan (modules=$MODULES threads=$THREADS) =="
# A layered corpus, so the cross-module work -- the symbol map, the registry
# freeze, the global env -- is exercised rather than skipped. Those are the
# phases where a race would actually live: the per-function work is isolated by
# construction, the shared state is what the threads all read.
VX_PIPELINE_QUIET=1 "${SETARCH[@]}" "$BIN" \
    --modules "$MODULES" --fns 16 --density 1 --files-per-layer 10 --deps 3 \
    --threads "$THREADS" --reps "$REPS" \
    > "$OUT/sweep.csv" 2> "$OUT/sweep.log" || true
assert_layout_ok "$OUT/sweep.log"

if [ "$RUN_TESTS" -eq 1 ]; then
    echo "== library tests under TSan =="
    "${SETARCH[@]}" cargo +nightly test -Zbuild-std --target "$TARGET" --release --lib \
        > "$OUT/tests.log" 2>&1 || true
    assert_layout_ok "$OUT/tests.log"
fi

echo
echo "== report =="
RACES=$(grep -c "WARNING: ThreadSanitizer: data race" "$OUT"/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
OTHER=$(grep -c "WARNING: ThreadSanitizer:" "$OUT"/*.log 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
echo "data races:            $RACES"
echo "all TSan warnings:     $OTHER"
if [ "$RACES" -gt 0 ]; then
    echo
    grep -A 25 "WARNING: ThreadSanitizer: data race" "$OUT"/*.log | head -80
fi
echo
echo "results: $OUT"
# A non-zero exit on a race, so this can gate CI later. A *clean* run exiting 0
# is the claim being made, so it has to be the exit code that says so.
[ "$RACES" -eq 0 ]
