#!/usr/bin/env bash
#===- run_perf_matrix.sh - what disaggregation costs, measured -----------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Times the same program along three paths on one two-GPU box (#348):
#
#   single   both phases on GPU 0            -- the baseline
#   disagg   decode on GPU 1, direct CUDA    -- what disaggregation itself costs
#   fleet    decode on GPU 1, over loopback  -- what the wire protocol costs
#
# The three differ in path, not in work: same binary, same program text, same
# devices in runs 2 and 3. So `disagg - single` is the cost of splitting the
# model across two devices, and `fleet - disagg` is the cost of the remote
# dispatch stack, with the network held out of it. Every previous fleet run went
# between rented machines through an SSH tunnel, which is why none of them could
# be quoted as a latency figure. Loopback removes that confound.
#
# Timing is by slope, not by total. `vxc --run` compiles, JITs, links and loads
# 60MB of weights before the first token, and that fixed cost is large enough to
# swamp what is being measured. Running each path at several token counts and
# taking the slope of time against tokens cancels it exactly: the intercept is
# the startup, the slope is the per-token cost. Since `steps` is 1 and prefill's
# length is set by the prompt rather than by the token count, the slope is
# specifically the cost of one *decode* step -- the phase that moves.
#
# Which is why the counts start at 48. Prefill runs for as many steps as the
# prompt is long -- 37, for tests/backend/pass/prompt.txt -- and decode begins
# only after that. Below 37 the entire generation is prefill: decode's loop body
# never executes, and `single` and `disagg` are not two placements of one
# computation but one computation run twice. Sampled at 16 and 32 tokens the
# disaggregated row would have come out free, because nothing had been
# disaggregated. Above 37 every point contains decode and the relation is a
# line -- checked at 48 and 64 tokens, where device 1 served 11 and 27 decode
# steps against a predicted 48-37 and 64-37.
#
# Usage, from the bundle directory on the pod after ./setup_gpu_pod.sh:
#   ./run_perf_matrix.sh [-n reps] [-t "48 64 96 128"] [-o outdir]
#
#===----------------------------------------------------------------------===#

set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM="tests/backend/pass/llama2.vx"
TOKEN_COUNTS="48 64 96 128"
REPS=3
OUTDIR="$BUNDLE_DIR/perf-evidence"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--reps)    REPS="$2"; shift 2 ;;
    -t|--tokens)  TOKEN_COUNTS="$2"; shift 2 ;;
    -o|--out)     OUTDIR="$2"; shift 2 ;;
    -p|--program) PROGRAM="$2"; shift 2 ;;
    -h|--help)    sed -n '10,41p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$BUNDLE_DIR"
# shellcheck disable=SC1091
[ -f env.sh ] && source env.sh
mkdir -p "$OUTDIR"

gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
if [ "$gpus" -lt 2 ]; then
  echo "error: this measures two-device placement; the box has $gpus GPU(s)" >&2
  exit 1
fi

# Whether anything else is on these GPUs. A neighbouring tenant does not make the
# run fail, it makes it slow, and a timing harness that does not record this
# produces numbers nobody can interpret later.
{
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  echo
  nvidia-smi topo -m
} > "$OUTDIR/hardware.txt" 2>&1
# Prefill's length is a property of the prompt, so a token count below it
# measures a run with no decode in it at all.
PROMPT_STEPS=37
for tk in $TOKEN_COUNTS; do
  if [ "$tk" -le "$PROMPT_STEPS" ]; then
    echo "error: $tk tokens is at or below the $PROMPT_STEPS-step prefill, so that" >&2
    echo "       point has no decode in it and belongs to a different line." >&2
    exit 2
  fi
done

echo "==> GPU memory in use before starting:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/    /'

CSV="$OUTDIR/timings.csv"
echo "config,tokens,rep,seconds" > "$CSV"

# One timed run. Verbose tracing stays off: it is a write to stderr per dispatch
# and there are thousands, which is a cost of the measurement rather than of the
# thing measured.
time_one() {  # config tokens rep extra_env...
  local config="$1" tokens="$2" rep="$3"; shift 3
  local t0 t1 secs
  t0=$(date +%s.%N)
  env "$@" LLAMA_TOKENS_CONFIG="$tokens;1" \
    ./vxc "$PROGRAM" --run > "$OUTDIR/out-$config-$tokens-$rep.txt" 2>&1
  local status=$?
  t1=$(date +%s.%N)
  secs=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')
  if [ $status -ne 0 ]; then
    echo "    $config/$tokens rep $rep FAILED (status $status)" >&2
    return 1
  fi
  echo "$config,$tokens,$rep,$secs" >> "$CSV"
  printf '    %-7s %4s tokens  rep %d  %8ss\n' "$config" "$tokens" "$rep" "$secs"
}

# An untimed run before each config's first timed one.
#
# The first run of a configuration pays for a cold page cache on a 60MB
# checkpoint and a cold CUDA context, and it is not a smaller version of the
# steady-state cost -- it is a different cost that happens once. Left in, it
# landed on the smallest token count, which is the point with the most leverage
# over a fit: the single-device row came out at -14 ms/token. A negative
# per-token cost is not a slow measurement, it is an arithmetically impossible
# one, and it says the sample was contaminated rather than noisy.
warmup() {
  env "$@" LLAMA_TOKENS_CONFIG="48;1" ./vxc "$PROGRAM" --run > /dev/null 2>&1
}

# --- 1. single, 2. disagg: no workers involved ------------------------------
for config in single disagg; do
  d=0; [ "$config" = disagg ] && d=1
  echo "==> $config (VX_LLAMA_DISAGG=$d)"
  warmup "VX_LLAMA_DISAGG=$d"
  for tokens in $TOKEN_COUNTS; do
    for rep in $(seq 1 "$REPS"); do
      time_one "$config" "$tokens" "$rep" "VX_LLAMA_DISAGG=$d"
    done
  done
done

# --- 3. fleet: the same two devices, reached over TCP -----------------------
#
# Worker 2 serves topology 501, not 501-retargeted-to-500: the point of this row
# is to hold placement fixed against run 2 and vary only the path. Started under
# setsid so a dropped shell cannot take a worker with it mid-run.
echo "==> fleet (two workers on loopback, GPU 0 and GPU 1)"
cat > "$OUTDIR/loopback.manifest" <<EOF
# The names have to be the ones the program spawns on. llama2.vx says
# Topology::GPU[D], so these are GPU[0] and GPU[1] -- a name that is not here
# means local, so a manifest of names the program never mentions would run
# everything on this process and still print plausible numbers.
GPU[0]  127.0.0.1  19501
GPU[1]  127.0.0.1  19502
EOF

VX_DISPATCH_VERBOSE=1 setsid ./vx-worker --port 19501 --topology 500 --worker-id 1 \
  > "$OUTDIR/worker-1.log" 2>&1 < /dev/null &
W1=$!
VX_DISPATCH_VERBOSE=1 setsid ./vx-worker --port 19502 --topology 501 --worker-id 2 \
  > "$OUTDIR/worker-2.log" 2>&1 < /dev/null &
W2=$!
sleep 2

fleet_ok=1
for w in 19501 19502; do
  if ! (exec 3<>/dev/tcp/127.0.0.1/$w) 2>/dev/null; then
    echo "  worker on $w never came up; see $OUTDIR/worker-*.log" >&2
    fleet_ok=0
  fi
done

if [ "$fleet_ok" = 1 ]; then
  warmup "VX_LLAMA_DISAGG=1" "VX_FLEET_MANIFEST=$OUTDIR/loopback.manifest"
  for tokens in $TOKEN_COUNTS; do
    for rep in $(seq 1 "$REPS"); do
      time_one fleet "$tokens" "$rep" \
        "VX_LLAMA_DISAGG=1" "VX_FLEET_MANIFEST=$OUTDIR/loopback.manifest"
    done
  done

  # The witness. Identical tokens cannot tell a distributed run from a local
  # one, because a local run produces identical tokens too -- only the far side
  # knows whether it did any work.
  # `grep -c` exits 1 on zero matches, so `|| echo 0` appended a second line and
  # the count became "0\n0" -- which `[ "$n" -eq 0 ]` then rejected as a
  # non-integer instead of reporting the failure it was written to report. The
  # test is on the exit status, and the count is whatever grep printed.
  echo "  work served, by worker:"
  for w in 1 2; do
    n=$(grep -c '^\[Vx worker\] DISPATCH' "$OUTDIR/worker-$w.log" 2>/dev/null)
    [ -z "$n" ] && n=0
    printf '    worker %d: %s dispatches\n' "$w" "$n"
    if [ "$n" -eq 0 ]; then
      echo "    FAILURE: worker $w served nothing; the fleet row ran locally" >&2
      echo "             and its timings describe the single-machine path." >&2
    fi
  done
fi

kill "$W1" "$W2" 2>/dev/null
pkill -x vx-worker 2>/dev/null
wait "$W1" "$W2" 2>/dev/null

# --- the numbers ------------------------------------------------------------
#
# Least squares over the *fastest* run at each size, not over every rep.
#
# Using every rep sounds more principled and is not. These runs share a box with
# whatever else the pod is doing, so the distribution has a floor and a long
# right tail: interference can only add time. One 8.7s sample against a 3.9s
# neighbour at the same size is not the measurement being noisy in both
# directions, it is one run that got hit -- and averaging it in moves the
# estimate away from the quantity of interest. It moved it far enough to make
# the single-device slope negative in one run of this script and to make
# disaggregation look 2.5x *faster* than not disaggregating in another. The
# minimum is the run least perturbed by everything that is not the program.
echo
echo "=== Per-token cost (least-squares slope of seconds against tokens) ==="
awk -F, 'NR>1 {
           if (!(($1 SUBSEP $2) in best) || $4 < best[$1,$2]) best[$1,$2] = $4
         }
         END {
           for (k in best) {
             split(k, kv, SUBSEP)
             c = kv[1]; x = kv[2] + 0; y = best[k] + 0
             n[c]++; sx[c]+=x; sy[c]+=y; sxx[c]+=x*x; sxy[c]+=x*y
           }
           printf "  %-8s %12s %14s %10s\n", "config", "ms/token", "startup (s)", "points"
           split("single disagg fleet", order, " ")
           for (i=1; i<=3; i++) {
             c = order[i]
             if (!(c in n) || n[c] < 2) continue
             d = n[c]*sxx[c] - sx[c]*sx[c]
             if (d == 0) continue
             m = (n[c]*sxy[c] - sx[c]*sy[c]) / d
             b = (sy[c] - m*sx[c]) / n[c]
             printf "  %-8s %12.2f %14.2f %10d\n", c, m*1000, b, n[c]
             slope[c] = m
           }
           if ("single" in slope && "disagg" in slope)
             printf "\n  splitting across two devices: %+.2f ms/token (%.1fx)\n", \
               (slope["disagg"]-slope["single"])*1000, slope["disagg"]/slope["single"]
           if ("disagg" in slope && "fleet" in slope)
             printf "  the wire protocol on top:    %+.2f ms/token (%.1fx)\n", \
               (slope["fleet"]-slope["disagg"])*1000, slope["fleet"]/slope["disagg"]
         }' "$CSV"

echo
echo "=== Fastest run at each size (seconds) ==="
awk -F, 'NR>1 { if (!(($1 SUBSEP $2) in b) || $4 < b[$1,$2]) b[$1,$2]=$4; t[$2]=1; c[$1]=1 }
         END {
           printf "  %-8s", "tokens"
           split("single disagg fleet", order, " ")
           for (i=1;i<=3;i++) if (order[i] in c) printf "%10s", order[i]
           printf "\n"
           nt=0; for (k in t) tk[++nt]=k
           for (i=1;i<nt;i++) for (j=i+1;j<=nt;j++) if (tk[i]+0>tk[j]+0) { s=tk[i]; tk[i]=tk[j]; tk[j]=s }
           for (i=1;i<=nt;i++) {
             printf "  %-8s", tk[i]
             for (k=1;k<=3;k++) if (order[k] in c)
               printf "%10s", ((order[k] SUBSEP tk[i]) in b ? b[order[k],tk[i]] : "-")
             printf "\n"
           }
         }' "$CSV"

# Same text along all three paths, or the timings are of three different things.
echo
echo "=== Same tokens along all three paths? ==="
strip_noise() {
  grep -vE '^\[Vx |^Expanding macro call:|^\[JIT\]|^\[flat-codegen\]' "$1" 2>/dev/null
}
ref=""
for config in single disagg fleet; do
  f="$OUTDIR/out-$config-64-1.txt"
  [ -f "$f" ] || continue
  strip_noise "$f" > "$OUTDIR/tokens-$config.txt"
  if [ -z "$ref" ]; then
    ref="$OUTDIR/tokens-$config.txt"
    printf '  %-8s (reference)\n' "$config"
  elif diff -q "$ref" "$OUTDIR/tokens-$config.txt" >/dev/null; then
    printf '  %-8s IDENTICAL\n' "$config"
  else
    printf '  %-8s DIFFERS -- the timings above compare different computations:\n' "$config"
    diff "$ref" "$OUTDIR/tokens-$config.txt" | head -10
  fi
done

echo
echo "Evidence in $OUTDIR/ (timings.csv is the raw data)"
