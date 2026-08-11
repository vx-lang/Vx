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
  ./vx-worker --port 19501 --topology 500 --worker-id 1 > "$OUTDIR/worker-1.log" 2>&1 &
  W1=$!
  ./vx-worker --port 19502 --topology 500 --worker-id 2 > "$OUTDIR/worker-2.log" 2>&1 &
  W2=$!
  sleep 1

  cat > "$OUTDIR/local.manifest" <<EOF
# Both workers on this machine. Same program, same binary as run 1.
PrefillWorker  127.0.0.1  19501
DecodeWorker   127.0.0.1  19502
EOF

  VX_FLEET_MANIFEST="$OUTDIR/local.manifest" \
    ./vxc "$PROGRAM" --run > "$OUTDIR/raw-tcp.txt" 2> "$OUTDIR/trace-tcp.txt"
  strip_noise "$OUTDIR/raw-tcp.txt" > "$OUTDIR/tokens-tcp.txt"

  kill "$W1" "$W2" 2>/dev/null
  wait "$W1" "$W2" 2>/dev/null
fi

# --- 3. two machines -------------------------------------------------------
if [ -n "$PREFILL" ] && [ -n "$DECODE" ]; then
  echo "==> Run 3: prefill on $PREFILL, decode on $DECODE"
  cat > "$OUTDIR/fleet.manifest" <<EOF
# Two machines. The program is unchanged from runs 1 and 2; this file is the
# only thing that says where its workers are.
PrefillWorker  ${PREFILL%%:*}  ${PREFILL##*:}
DecodeWorker   ${DECODE%%:*}   ${DECODE##*:}
EOF
  cat "$OUTDIR/fleet.manifest"

  VX_FLEET_MANIFEST="$OUTDIR/fleet.manifest" \
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
