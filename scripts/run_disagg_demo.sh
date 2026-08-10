#!/usr/bin/env bash
#===- run_disagg_demo.sh - the disaggregated run, and its evidence --------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Runs one program twice on a two-GPU pod -- once with both phases on GPU 0,
# once with decode on GPU 1 -- and collects what makes the second run a claim
# rather than an assertion (#347).
#
# The claim has three parts and each has a file here:
#
#   1. The two runs produce the same tokens.  tokens-single.txt vs tokens-disagg.txt
#   2. The second one really used two GPUs.   trace-disagg.txt
#   3. The KV cache really crossed between    trace-disagg.txt, the `peer` line
#      them, and decode could not have
#      worked without it.                    tokens-nohandoff.txt differs
#
# Run from the bundle directory on the pod, after ./setup_gpu_pod.sh.
#
# Usage:
#   ./run_disagg_demo.sh [-t tokens] [-o outdir]
#
#===----------------------------------------------------------------------===#

set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKENS=64
OUTDIR="$BUNDLE_DIR/disagg-evidence"
PROGRAM="tests/backend/pass/llama2.vx"

while [ $# -gt 0 ]; do
  case "$1" in
    -t|--tokens) TOKENS="$2"; shift 2 ;;
    -o|--out)    OUTDIR="$2"; shift 2 ;;
    -p|--program) PROGRAM="$2"; shift 2 ;;
    -h|--help)   sed -n '10,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$BUNDLE_DIR"
# shellcheck disable=SC1091
[ -f env.sh ] && source env.sh

mkdir -p "$OUTDIR"
export LLAMA_TOKENS_CONFIG="$TOKENS;1"

# What the box actually is. Recorded first, so the interconnect a run crossed is
# archived with it rather than remembered afterwards -- fleet/node-2gpu-a100.vx
# declares the pessimistic PCIe figure precisely because a pod does not say.
{
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
  echo
  nvidia-smi topo -m
} > "$OUTDIR/hardware.txt" 2>&1

gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
echo "==> $gpus GPU(s) present"
if [ "$gpus" -lt 2 ]; then
  echo "error: the disaggregated run names GPU 1; this box has $gpus GPU(s)." >&2
  echo "       The plugin would abort with the device count, which is correct" >&2
  echo "       but a worse place to find out. Rent a two-GPU pod." >&2
  exit 1
fi

# --- 1. Both phases on GPU 0 -- the oracle ----------------------------------
#
# Same code as the disaggregated run: same handoff, same two RunStates, same two
# weight replicas. The only difference is that the destination index is 0.
# Verbose here too. The narration goes to stderr, so it cannot contaminate the
# tokens, and it makes this run evidence rather than just a reference: the trace
# has to show device 0 and *only* device 0. Without it, "run 2 used two GPUs"
# rests on run 2 alone, and a reader cannot tell a working disaggregation from a
# plugin that always reports two.
echo "==> Run 1: both phases on GPU 0"
VX_LLAMA_DISAGG=0 VX_DISPATCH_VERBOSE=1 ./vxc "$PROGRAM" --run \
  > "$OUTDIR/raw-single.txt" 2> "$OUTDIR/trace-single.txt"
single_status=$?

# --- 2. Prefill on GPU 0, decode on GPU 1 -----------------------------------
echo "==> Run 2: prefill on GPU 0, decode on GPU 1"
VX_LLAMA_DISAGG=1 VX_DISPATCH_VERBOSE=1 ./vxc "$PROGRAM" --run \
  > "$OUTDIR/raw-disagg.txt" 2> "$OUTDIR/trace-disagg.txt"
disagg_status=$?

strip_noise() {
  grep -v '^Expanding macro call:\|^\[JIT\]\|^\[flat-codegen\]' "$1"
}
strip_noise "$OUTDIR/raw-single.txt" > "$OUTDIR/tokens-single.txt"
strip_noise "$OUTDIR/raw-disagg.txt" > "$OUTDIR/tokens-disagg.txt"

echo
echo "=== Exit status ==="
echo "  single-device: $single_status"
echo "  disaggregated: $disagg_status"

echo
echo "=== 1. Same tokens? ==="
if diff -q "$OUTDIR/tokens-single.txt" "$OUTDIR/tokens-disagg.txt" >/dev/null; then
  echo "  IDENTICAL -- decode on GPU 1 produced the same text as decode on GPU 0"
else
  echo "  DIFFER -- this is a failure, not a curiosity:"
  diff "$OUTDIR/tokens-single.txt" "$OUTDIR/tokens-disagg.txt" | head -20
fi

echo
echo "=== 2. Did it use two GPUs? ==="
# Every dispatch prints the device it selected. Prefill's dispatches should all
# say 0 and decode's all say 1, with one switch between them. Two devices
# appearing at all is the minimum; the switch happening once, at the phase
# boundary, is the actual claim.
#
# `prev` starts at "none" rather than empty because awk reads an uninitialized
# variable as both "" and 0, so comparing it against a `device 0` line comes out
# equal and the run's first device is never reported. And dispatches are counted
# separately from NR, which would otherwise include the peer-transfer line and
# report the switch one dispatch late.
awk 'BEGIN{prev="none"; n=0}
     /^\[Vx CUDA\] device /{
       n++; d=$4;
       if (d != prev) { print "  device " d " from dispatch " n; prev=d }
     }' "$OUTDIR/trace-disagg.txt" | head -10
echo "  dispatches per device:"
for run in single disagg; do
  d0=$(grep -c '^\[Vx CUDA\] device 0' "$OUTDIR/trace-$run.txt")
  d1=$(grep -c '^\[Vx CUDA\] device 1' "$OUTDIR/trace-$run.txt")
  printf '    %-8s device 0: %-8s device 1: %s\n' "$run" "$d0" "$d1"
done
echo "  (the single-device run must show device 1: 0 -- otherwise the second"
echo "   run's two devices prove nothing about the placement)"

echo
echo "=== 3. Did the KV cache cross? ==="
grep '^\[Vx CUDA\] peer ' "$OUTDIR/trace-disagg.txt" | sed 's/^/  /' || echo "  NONE -- the handoff did not happen"
# What the program asked to move: n_layers x seq_len x kv_dim x 4 bytes, twice
# (keys and values). stories15M is 6 layers, kv_dim 288.
predicted=$(( 6 * TOKENS * 288 * 4 ))
echo "  predicted per cache: $predicted bytes (6 layers x $TOKENS positions x 288 x 4)"

echo
echo "Evidence in $OUTDIR/"
