#!/usr/bin/env bash
#===- run_e1.sh - Vx Compiler --------------------------------*- bash -*-===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# The CGO 2027 E1 sweep (#295) on a many-core Linux host.
#
#   ./run_e1.sh                       # full ladder + lock profile
#   ./run_e1.sh --max-threads 32      # cap the ladder
#   ./run_e1.sh --quick               # smoke test, ~2 minutes
#
# Everything lands in results/<timestamp>/: raw CSV, per-cell stderr, the lock
# contention profiles, and env.txt describing the machine it ran on.
#
#===----------------------------------------------------------------------===#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$REPO/target/release/intern_bench"
REPS="${REPS:-10}"
MAX_THREADS=0
QUICK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --max-threads) MAX_THREADS="$2"; shift 2 ;;
        --reps) REPS="$2"; shift 2 ;;
        --quick) QUICK=1; shift ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

[ -x "$BIN" ] || { echo "build first: cargo build --release --bin intern_bench" >&2; exit 1; }

# ---- CPU topology -----------------------------------------------------------
# One CPU per *physical* core, in core order. Two hyperthread siblings share an
# execution unit and an L1, so a ladder that lands two rayon workers on one core
# measures SMT, not scaling -- and the mutex-contention question this run exists
# to answer is exactly the kind that sibling placement distorts.
mapfile -t CORE_CPUS < <(lscpu -p=CPU,CORE | grep -v '^#' | sort -t, -k2,2n -k1,1n | awk -F, '!seen[$2]++ {print $1}')
PHYS=${#CORE_CPUS[@]}
echo "physical cores available: $PHYS"

LADDER=()
for t in 1 2 4 8 16 32 64 128; do
    [ "$t" -le "$PHYS" ] || continue
    [ "$MAX_THREADS" -eq 0 ] || [ "$t" -le "$MAX_THREADS" ] || continue
    LADDER+=("$t")
done
[ ${#LADDER[@]} -gt 0 ] || { echo "no usable thread counts" >&2; exit 1; }
THREADS=$(IFS=,; echo "${LADDER[*]}")
MAXT=${LADDER[-1]}
echo "thread ladder: $THREADS"

# ---- environment ------------------------------------------------------------
# VX_PIPELINE_QUIET is not optional. parse_phase println!s per module from inside
# the rayon parallel-for, and println! takes the global stdout mutex (#306) --
# without this the workers serialise on stdout inside the measured region, and
# `perf lock` reports contention on stdout rather than on the interner.
export VX_PIPELINE_QUIET=1
export TMPDIR="${TMPDIR:-/dev/shm/vxbench}"
mkdir -p "$TMPDIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$REPO/utils/cgo/results/$STAMP"
mkdir -p "$OUT"

{
    echo "date_utc: $STAMP"
    echo "host: $(hostname)"
    echo "kernel: $(uname -r)"
    echo "reps: $REPS"
    echo "ladder: $THREADS"
    echo "tmpdir: $TMPDIR"
    echo "commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "branch: $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo
    lscpu
    echo
    echo "-- frequency governor --"
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unavailable"
    echo "-- intel_pstate turbo (1 = disabled) --"
    cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || echo "unavailable"
} > "$OUT/env.txt"

# Frequency policy. On a virtualised instance these files do not exist, and the
# host is free to move the clock underneath the run. Warn rather than fail:
# a ladder measured under an unpinned clock is still worth having, provided the
# write-up says so instead of implying a controlled environment.
if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done
    echo "governor: performance"
else
    echo "WARNING: cannot set the frequency governor (not a bare-metal instance?)."
    echo "         Clock drift will appear as scaling noise. Recorded in env.txt."
fi
if [ -w /sys/devices/system/cpu/intel_pstate/no_turbo ]; then
    echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo
    echo "turbo: disabled"
else
    echo "WARNING: cannot disable turbo. Single-thread times will be flattered"
    echo "         relative to many-thread times, inflating apparent speedups."
fi

# ---- the sweep --------------------------------------------------------------
# Each cell pins to one CPU per physical core, taking the first MAXT of them, so
# every configuration in a run sees the same cores.
PIN=$(IFS=,; echo "${CORE_CPUS[*]:0:$MAXT}")
echo "pinned to CPUs: $PIN"

run_cell() {
    local name="$1"; shift
    echo
    echo "=== $name ==="
    taskset -c "$PIN" "$BIN" "$@" \
        --threads "$THREADS" --reps "$REPS" \
        > "$OUT/$name.csv" 2> "$OUT/$name.log"
    tail -n 14 "$OUT/$name.log"
}

if [ "$QUICK" -eq 1 ]; then
    run_cell quick --modules 16 --fns 16 --density 0,1
else
    # The plan's grid (#296): N x M, each at density 0 and 1. Density 0 is the
    # control -- same function count, same signature widths, same type-stream
    # length, no interning -- so any divergence between the modes at density 1
    # that is absent at density 0 is attributable to interning and not to the
    # corpus being bigger.
    run_cell grid --modules 8,64,512 --fns 16,128 --density 0,1

    # Maximum pressure this generator can produce. On 10 cores the two designs
    # finish within 1% here; if a crossover exists anywhere, this is the cell it
    # shows up in first.
    run_cell maxpressure --modules 64 --fns 16 --density 0,1 \
        --params-per-fn 16 --arity 2 --locals-per-module 32
fi

# ---- lock contention --------------------------------------------------------
# The load-bearing measurement. A wall-clock comparison on a modest core count
# cannot separate the designs (see src/bin/corpus/README.md); what separates them
# is whether contention on the interner mutex grows with thread count. `locked`
# is the only mode with a lock to contend on, so a profile that shows nothing is
# itself the result -- report it either way rather than only on a positive.
echo
echo "=== lock contention (locked mode, $MAXT threads) ==="
if command -v perf >/dev/null 2>&1; then
    for d in 0 1; do
        # `-b` is the BPF collector: no lockdep kernel required, and it attributes
        # to the acquiring callsite, which is what turns "locked got slower" into
        # "locked got slower waiting on *this*".
        # The bench's own CSV goes to /dev/null inside the wrapped shell, so the
        # profile file holds the profile and nothing else. Same corpus parameters
        # as the maxpressure cell above, so it is served from cache and corpus
        # generation does not appear in the profile.
        taskset -c "$PIN" perf lock contention -b -- \
            sh -c "'$BIN' --modules 64 --fns 16 --density $d \
                   --params-per-fn 16 --arity 2 --locals-per-module 32 \
                   --threads $MAXT --reps 3 >/dev/null 2>&1" \
            > "$OUT/perflock_density$d.txt" 2>&1 || \
            echo "perf lock failed for density=$d (see $OUT/perflock_density$d.txt)"
        echo "-- density=$d --"
        head -n 20 "$OUT/perflock_density$d.txt"
    done
    # Density 0 is the baseline for the profile too: whatever contention appears
    # there is *not* the interner, and has to be subtracted before reading the
    # density-1 profile as an interning result.
else
    echo "perf not installed; skipping. This is the half of E1 that decides the"
    echo "result, so a run without it is incomplete."
fi

echo
echo "results: $OUT"
