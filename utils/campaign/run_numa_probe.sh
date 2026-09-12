#!/usr/bin/env bash
# Measure what a NUMA hop costs on this machine, and compare it against what
# fleet/xeon-e5-2666v3.vx predicts.
#
#   ./utils/campaign/run_numa_probe.sh            # both kernels, 2 GiB, 5 reps
#   MIB=4096 REPS=9 ./utils/campaign/run_numa_probe.sh
#
# Needs a machine with at least two NUMA nodes, `numactl`, and a C compiler.
# It will refuse rather than guess on anything else -- an Apple Silicon laptop
# has one unified memory and there is no hop here to measure.
#
# WHAT IS BEING COMPARED, AND WHAT IS NOT
#
# The model's absolute numbers are not the claim. A memcpy moves two bytes of
# traffic per byte copied, and the declared rates are the memory controllers'
# peak rather than an achievable copy rate, so the model over-predicts any one
# edge -- deliberately, and by at least 2x (see fleet/m4-uma.vx). What IS
# comparable is the RATIO of remote to local, because both sides of that
# division carry the same over-prediction and it cancels.
#
# So the number this script exists to produce is the last line: measured remote
# / local against predicted remote / local. The four cells above it are there so
# a reader can see the measurement was not one lucky pair.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MIB="${MIB:-2048}"
REPS="${REPS:-5}"

die() { echo "$*" >&2; exit 1; }

command -v numactl >/dev/null 2>&1 || die "no numactl: install numactl (Debian/Ubuntu: apt install numactl)"
command -v lscpu   >/dev/null 2>&1 || die "no lscpu: this script wants a Linux host"
command -v python3 >/dev/null 2>&1 || die "no python3: the script reads its own JSON and the compiler's with it"

NODES=$(numactl -H | awk '/^available:/ {print $2}')
[ -n "$NODES" ] || die "could not read the node count from numactl -H"
if [ "$NODES" -lt 2 ]; then
  die "this machine reports $NODES NUMA node(s); there is no hop to measure.
The experiment wants the two-socket build box (E5-2666 v3), not a laptop."
fi

echo "== machine =="
numactl -H | sed -n '1,/^node distances/p' | sed 's/^/  /'
echo

CC="${CC:-gcc}"
command -v "$CC" >/dev/null 2>&1 || die "no C compiler: set CC= or install gcc"

BIN="$(mktemp -d)/numa_probe"
trap 'rm -rf "$(dirname "$BIN")"' EXIT

OMP="-fopenmp"
if ! "$CC" -O3 -march=native $OMP "$HERE/numa_probe.c" -o "$BIN" 2>/dev/null; then
  echo "  (no OpenMP; measuring single-threaded, which reads latency more than bandwidth)"
  OMP=""
  "$CC" -O3 -march=native "$HERE/numa_probe.c" -o "$BIN" \
    || die "could not build numa_probe.c"
fi

# One cell of the matrix: threads on $1, memory on $2.
cell() {
  local cpun="$1" memn="$2" kern="$3"
  numactl --cpunodebind="$cpun" --membind="$memn" "$BIN" "$kern" "$MIB" "$REPS" 2>/dev/null
}

gbs() { python3 -c "import json,sys; print(json.load(sys.stdin)['best_gbs'])"; }

declare -A RESULT
for kern in read copy; do
  echo "== $kern, ${MIB} MiB, best of $REPS =="
  printf '  %-22s %10s\n' "cpu node / mem node" "GB/s"
  for c in 0 1; do
    for m in 0 1; do
      out="$(cell "$c" "$m" "$kern")"
      if [ -z "$out" ]; then
        printf '  %-22s %10s\n' "$c / $m" "FAILED"
        RESULT["$kern,$c,$m"]=""
        continue
      fi
      v="$(echo "$out" | gbs)"
      RESULT["$kern,$c,$m"]="$v"
      tag=""; [ "$c" = "$m" ] && tag="local" || tag="remote"
      printf '  %-22s %10s  %s\n' "$c / $m" "$v" "$tag"
    done
  done
  echo
done

# The measured ratio, averaged over both directions so a single asymmetric pair
# cannot carry the result on its own.
RATIO=$(python3 - "${RESULT[read,0,0]:-}" "${RESULT[read,1,1]:-}" \
                   "${RESULT[read,0,1]:-}" "${RESULT[read,1,0]:-}" <<'PY'
import sys
try:
    l0, l1, r01, r10 = (float(x) for x in sys.argv[1:5])
    local = (l0 + l1) / 2
    remote = (r01 + r10) / 2
    print(f"{local / remote:.3f}" if remote else "")
except Exception:
    print("")
PY
)

# What the model says, read out of the compiler rather than restated here, so
# this cannot drift from the fleet file.
FIX="$ROOT/tests/optimizations/pass/numa_peer_node_is_priced.vx"
VXC="${VXC:-}"
if [ -z "$VXC" ]; then
  for cand in "$ROOT/target/release/vxc" "$ROOT/target/debug/vxc" "$(command -v vxc || true)"; do
    [ -x "$cand" ] && { VXC="$cand"; break; }
  done
fi

echo "== the comparison =="
if [ -z "$VXC" ] || [ ! -f "$FIX" ]; then
  echo "  measured remote/local ratio: ${RATIO:-unavailable}"
  echo "  (no vxc or fixture found, so nothing to compare against; set VXC=)"
  exit 0
fi

PRED=$("$VXC" "$FIX" --host default --machine "$ROOT/fleet/xeon-e5-2666v3.vx" \
        --action emit-mlir -o /dev/null --diagnostics-json 2>/dev/null |
  python3 -c '
import json, sys
doc = ""
for line in sys.stdin:
    doc += line
    if line.rstrip() == "}":
        break
try:
    d = json.loads(doc)
    cost = {tuple(r["path"]): r["derived_cost"] for r in d["routes"]}
    local = cost[("CPU_DRAM", "HBM")]
    remote = cost[("HBM", "PEER_HBM")]
    print(f"{remote / local:.3f}")
except Exception:
    print("")
')

echo "  predicted remote/local : ${PRED:-unavailable}   (68 GB/s local vs 38 GB/s across the link)"
echo "  measured  remote/local : ${RATIO:-unavailable}"
if [ -n "$PRED" ] && [ -n "$RATIO" ]; then
  python3 - "$PRED" "$RATIO" <<'PY'
import sys
pred, meas = float(sys.argv[1]), float(sys.argv[2])
err = (meas - pred) / pred * 100
print(f"  the model is off by     : {err:+.1f}%")
print()
if abs(err) <= 25:
    print("  Within 25%. The declared QPI rate is the right order, and the")
    print("  UNVERIFIED note on it in fleet/xeon-e5-2666v3.vx can be replaced by")
    print("  this measurement.")
else:
    print("  Outside 25%. Before changing the fleet file, rule out the harness:")
    print("   - is the buffer larger than both sockets' LLC put together?")
    print("   - did OpenMP actually start threads on the bound node (threads>1)?")
    print("   - is anything else running on the box?")
    print("  A derived figure being wrong is a finding; a harness being wrong is not.")
PY
fi
