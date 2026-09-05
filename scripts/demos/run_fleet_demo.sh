#!/usr/bin/env bash
#===- run_fleet_demo.sh - Llama across two machines ----------------------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Runs llama2.vx three times against the same binary and the same program text,
# changing only which machines the workers are on (#348).
#
#   1. no manifest          -- everything here
#   2. manifest, localhost  -- both workers on this machine, over TCP
#   3. manifest, two hosts  -- prefill on one machine, decode on another
#
# The three must produce the same tokens. That is the claim: the program does
# not know where it ran, and the file that decided is not the program.
#
# On the worker machines, first:
#   ./vx-worker --port 9001 --topology 500 --worker-id 1   # PrefillWorker
#   ./vx-worker --port 9001 --topology 500 --worker-id 2   # DecodeWorker
#
# Then here:
#   ./run_fleet_demo.sh --prefill 10.0.0.4:9001 --decode 10.0.0.5:9001
#
#===----------------------------------------------------------------------===#

set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROGRAM="tests/backend/pass/llama2.vx"
TOKENS=64
PREFILL=""
DECODE=""
OUTDIR="$BUNDLE_DIR/fleet-evidence"

while [ $# -gt 0 ]; do
  case "$1" in
    --prefill) PREFILL="$2"; shift 2 ;;
    --decode)  DECODE="$2"; shift 2 ;;
    -t|--tokens) TOKENS="$2"; shift 2 ;;
    -o|--out)  OUTDIR="$2"; shift 2 ;;
    -p|--program) PROGRAM="$2"; shift 2 ;;
    -h|--help) sed -n '10,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$BUNDLE_DIR"
# shellcheck disable=SC1091
[ -f env.sh ] && source env.sh
mkdir -p "$OUTDIR"
export LLAMA_TOKENS_CONFIG="$TOKENS;1"

strip_noise() {
  grep -vE '^\[Vx |^Expanding macro call:|^\[JIT\]|^\[flat-codegen\]' "$1"
}

# --- 1. one machine, no manifest ------------------------------------------
echo "==> Run 1: no manifest, everything here"
./vxc "$PROGRAM" --run > "$OUTDIR/raw-local.txt" 2> "$OUTDIR/trace-local.txt"
strip_noise "$OUTDIR/raw-local.txt" > "$OUTDIR/tokens-local.txt"

# --- 2. both workers on this machine, over TCP ----------------------------
#
# Worth doing even when two real machines are available: it separates "the
# distribution works" from "the network works", and if run 3 differs from run 2
# the difference is the network rather than the design.
if command -v ./vx-worker >/dev/null 2>&1 || [ -x ./vx-worker ]; then
  echo "==> Run 2: both workers here, over TCP"
  # setsid, so a worker outlives the shell that started it.
  #
  # Started with a plain `&` these die with their session, and on a rented pod
  # an interactive SSH is exactly the session that goes away -- one dropped
  # connection killed a worker mid-generation and failed the run, twice, for a
  # reason that had nothing to do with what was being tested.
  VX_DISPATCH_VERBOSE=1 setsid ./vx-worker --port 19501 --topology 500 --worker-id 1 \
    > "$OUTDIR/worker-1.log" 2>&1 < /dev/null &
  W1=$!
  VX_DISPATCH_VERBOSE=1 setsid ./vx-worker --port 19502 --topology 501 --worker-id 2 \
    > "$OUTDIR/worker-2.log" 2>&1 < /dev/null &
  W2=$!
  sleep 2
  for w in 19501 19502; do
    (exec 3<>/dev/tcp/127.0.0.1/$w) 2>/dev/null ||
      { echo "  worker on $w did not come up; see $OUTDIR/worker-*.log" >&2; }
  done

  # GPU[0] and GPU[1], not PrefillWorker/DecodeWorker: these names have to be
  # the ones the *program* dispatches to, and llama2.vx spawns on
  # `Topology::GPU[D]`. A name absent from a manifest means local, by design --
  # so a manifest of names the program never mentions is not an error, it is a
  # fully local run. It would produce identical tokens and report success while
  # proving nothing at all. The check at the end of run 2 is what makes that
  # failure loud instead of silent.
  cat > "$OUTDIR/local.manifest" <<EOF
# Both workers on this machine. Same program, same binary as run 1.
GPU[0]  127.0.0.1  19501
GPU[1]  127.0.0.1  19502
EOF

  # VX_LLAMA_DISAGG=1, or decode targets GPU[0] like prefill and the second
  # worker serves nothing. The first run of this script reported 2752
  # dispatches on worker 1 and none on worker 2, with the tokens identical --
  # which was true, and described a run that used one worker twice.
  #
  # Run 1 stays at DISAGG=0 deliberately: it is the single-device oracle, and
  # on a one-GPU machine naming GPU[1] locally would abort. That the two agree
  # is the claim -- same tokens, different placement.
  VX_LLAMA_DISAGG=1 VX_FLEET_MANIFEST="$OUTDIR/local.manifest" \
    ./vxc "$PROGRAM" --run > "$OUTDIR/raw-tcp.txt" 2> "$OUTDIR/trace-tcp.txt"
  strip_noise "$OUTDIR/raw-tcp.txt" > "$OUTDIR/tokens-tcp.txt"

  # Did any work actually reach a worker? Matching tokens cannot answer this:
  # a run that stayed local matches too, and matches perfectly. Only the far
  # side can say, so the worker's own log is the witness.
  for w in 1 2; do
    # `grep -c` exits 1 on zero matches, so `|| echo 0` appended a second line
    # and the count became "0\n0" -- which the integer test then rejected as
    # malformed rather than reporting the failure it was written to report.
    n=$(grep -c '^\[Vx worker\] DISPATCH' "$OUTDIR/worker-$w.log" 2>/dev/null)
    [ -z "$n" ] && n=0
    if [ "$n" -eq 0 ]; then
      echo "  FAILURE: worker $w served no dispatches. The run was local." >&2
      echo "           Check that the manifest names what the program spawns on" >&2
      echo "           -- an unlisted name is local, not an error." >&2
    else
      printf '  worker %d served %s dispatches\n' "$w" "$n"
    fi
  done

  # pkill -x, not -f: the pattern `vx-worker` appears in this script's own
  # command line, so -f matches the shell doing the killing.
  kill "$W1" "$W2" 2>/dev/null
  pkill -x vx-worker 2>/dev/null
  wait "$W1" "$W2" 2>/dev/null
fi

# --- 3. two machines -------------------------------------------------------
if [ -n "$PREFILL" ] && [ -n "$DECODE" ]; then
  echo "==> Run 3: prefill on $PREFILL, decode on $DECODE"
  cat > "$OUTDIR/fleet.manifest" <<EOF
# Two machines. The program is unchanged from runs 1 and 2; this file is the
# only thing that says where its workers are.
GPU[0]  ${PREFILL%%:*}  ${PREFILL##*:}
GPU[1]  ${DECODE%%:*}   ${DECODE##*:}
EOF
  cat "$OUTDIR/fleet.manifest"

  VX_LLAMA_DISAGG=1 VX_FLEET_MANIFEST="$OUTDIR/fleet.manifest" \
    ./vxc "$PROGRAM" --run > "$OUTDIR/raw-fleet.txt" 2> "$OUTDIR/trace-fleet.txt"
  strip_noise "$OUTDIR/raw-fleet.txt" > "$OUTDIR/tokens-fleet.txt"
fi

# --- the claim -------------------------------------------------------------
echo
echo "=== Same tokens, wherever it ran? ==="
base="$OUTDIR/tokens-local.txt"
for variant in tcp fleet; do
  f="$OUTDIR/tokens-$variant.txt"
  [ -f "$f" ] || continue
  if diff -q "$base" "$f" >/dev/null; then
    printf '  %-6s IDENTICAL to the single-machine run\n' "$variant"
  else
    printf '  %-6s DIFFERS -- this is a failure, not a curiosity:\n' "$variant"
    diff "$base" "$f" | head -20
  fi
done

echo
echo "=== What the tokens were ==="
head -c 400 "$base"; echo

echo
echo "Evidence in $OUTDIR/"
