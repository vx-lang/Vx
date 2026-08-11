#!/usr/bin/env bash
#===- run_two_machine_demo.sh - one program, two machines ----------------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Runs a Vx program across two real machines of different architectures, and
# measures what placement is worth on a link that has latency (#348, #321).
#
# The loopback harness (scripts/measure_wire_traffic.sh) counts messages; it
# cannot tell you what one costs, because a message to 127.0.0.1 is free. Here
# the worker is somewhere else, and the round trip is ~37 ms -- so a dispatch
# that takes eight of them takes a third of a second whatever it computes.
#
# Two things are being demonstrated and they are separable:
#
#   communication  the tensors reach the other machine, the answer comes back,
#                  and placing a tensor there once is cheaper than sending it
#                  every dispatch. That is what this script measures, and it
#                  needs no GPU.
#   execution      how fast the arithmetic runs once the data is there. Not
#                  measured here. The worker is a CPU, and the point of these
#                  numbers is the wire.
#
# The host is arm64 (an Apple laptop) and the worker x86-64 (an EC2 box), which
# also makes this the test that the wire format, the descriptors and the
# dispatch payload survive a change of architecture. The answers are compared,
# not assumed.
#
# Nothing is exposed publicly: the worker binds its own loopback and an SSH
# tunnel forwards to it, so the manifest names 127.0.0.1 on both sides.
#
#   ./scripts/run_two_machine_demo.sh -H ubuntu@host -i ~/.ssh/key.pem
#
# Measured on a laptop against us-west-2, and these are the rows this script
# actually prints:
#
#                    messages   bytes      wall
#   matmul local            -        -     0.349 s
#   matmul unplaced        64   384 KiB    3.080 s
#   kv local                -        -     0.281 s
#   kv unplaced            64   35.9 KiB   2.518 s
#   kv resident            50    7.1 KiB   1.967 s
#
# A fully placed matmul -- operands *and* result on the worker -- is 14 messages
# and 48 KiB, a 4.6x reduction, but it is not run here: with the result resident
# there is no way to read it home, so there would be nothing to compare against
# the local answer. That gap is the next piece of work, and until it is closed a
# fully placed program cannot be checked, only counted.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

REMOTE=""
KEY=""
PORT=24000
OUTDIR="${TMPDIR:-/tmp}/vx-two-machine"

while [ $# -gt 0 ]; do
  case "$1" in
    -H|--host) REMOTE="$2"; shift 2 ;;
    -i|--key)  KEY="$2"; shift 2 ;;
    -p|--port) PORT="$2"; shift 2 ;;
    -o|--out)  OUTDIR="$2"; shift 2 ;;
    -h|--help) sed -n '9,50p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$REMOTE" ] || { echo "error: -H user@host is required" >&2; exit 2; }
SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20)
[ -n "$KEY" ] && SSH+=(-i "$KEY")

VXC="${VXC:-./target/release/vxc}"
[ -x "$VXC" ] || { echo "error: no $VXC (cargo build --release)" >&2; exit 1; }
mkdir -p "$OUTDIR"

cleanup() {
  [ -f "$OUTDIR/tunnel.pid" ] && kill "$(cat "$OUTDIR/tunnel.pid")" 2>/dev/null || true
  rm -f "$OUTDIR/tunnel.pid"
  # The worker and its copy of the runtime do not outlive the run. It is
  # someone else's machine.
  "${SSH[@]}" "$REMOTE" 'pkill -x vx-worker 2>/dev/null; rm -rf ~/vx-fleet' 2>/dev/null || true
}
trap cleanup EXIT

echo "==> shipping the runtime (an archive, not a checkout)"
tar czf "$OUTDIR/rt.tgz" runtime include
scp "${SSH[@]:1}" "$OUTDIR/rt.tgz" "$REMOTE:/tmp/vx_fleet_rt.tgz" >/dev/null

echo "==> building the worker there"
"${SSH[@]}" "$REMOTE" "
  set -e
  pkill -x vx-worker 2>/dev/null || true
  rm -rf ~/vx-fleet && mkdir -p ~/vx-fleet
  tar xzf /tmp/vx_fleet_rt.tgz -C ~/vx-fleet 2>/dev/null
  cd ~/vx-fleet
  echo \"    arch: \$(uname -m)\"
  g++ -std=c++17 -O2 -Wall runtime/vx_worker_main.cpp runtime/host_dispatch.cpp \
      -Iinclude -Iruntime -lffi -o vx-worker
  setsid ./vx-worker --port $PORT --topology 0 --worker-id 2 --verbose \
    > ~/vx-fleet/worker.log 2>&1 < /dev/null &
  sleep 1
  pgrep -x vx-worker >/dev/null || { echo 'the worker did not start'; exit 1; }
"

echo "==> tunnelling 127.0.0.1:$PORT to its loopback"
"${SSH[@]}" -o ExitOnForwardFailure=yes -N -L "$PORT:127.0.0.1:$PORT" "$REMOTE" \
  > "$OUTDIR/tunnel.log" 2>&1 &
echo $! > "$OUTDIR/tunnel.pid"
sleep 3
(exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null || {
  echo "error: the tunnel is not up; see $OUTDIR/tunnel.log" >&2; exit 1; }

echo "GPU[0]  127.0.0.1  $PORT" > "$OUTDIR/manifest"

# --- the programs -----------------------------------------------------------
# Two shapes. The first is a plain repeated matmul, where placement removes the
# operands from the wire entirely. The second is attention at decode: a KV cache
# that stays and a query that arrives, which is the movement a disaggregated
# run actually makes.
write_programs() {
  cat > "$OUTDIR/matmul_unplaced.vx" <<'EOF'
fn main() -> i32 {
  let mut a = Tensor<f32>([ 64, 64 ]);
  let mut b = Tensor<f32>([ 64, 64 ]);
  let mut c = Tensor<f32>([ 64, 64 ]);
  for i in 0..64 {
    for j in 0..64 {
      a[i][j] = ((i + j) as f32) * 0.01;
      b[i][j] = ((i - j) as f32) * 0.02;
    }
  }
  for step in 0..8 {
    spawn on(Topology::GPU) {
      matmul_into(&mut c, &a, &b);
    }
  }
  print(c[0][0]);
  return 0;
}
EOF

  cat > "$OUTDIR/kv_resident.vx" <<'EOF'
// Attention at decode: the KV cache lives on the worker, the query travels.
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}

fn main() -> i32 {
  let mut kt_h = Tensor<f32>([ 16, 64 ]);
  for d in 0..16 {
    for j in 0..64 {
      kt_h[d][j] = ((j - d) as f32) * 0.03;
    }
  }
  let kt = transfer(kt_h, Memory::GPU_HBM);

  let mut q = Tensor<f32>([ 1, 16 ]);
  let mut s = Tensor<f32>([ 1, 64 ]);
  let mut acc : f32 = 0.0;
  for step in 0..8 {
    for d in 0..16 {
      q[0][d] = ((step + d) as f32) * 0.05;
    }
    spawn on(Topology::GPU) {
      matmul_into(&mut s, &q, &kt);
    }
    let mut m : f32 = -1000000.0;
    for j in 0..64 {
      if s[0][j] > m {
        m = s[0][j];
      }
    }
    acc = acc + m;
  }
  print(acc);
  return 0;
}
EOF

  # The same attention with the cache left where it was, so the difference is
  # the placement and nothing else.
  sed -e 's|^  let kt = transfer(kt_h, Memory::GPU_HBM);|  // unplaced: the cache crosses every step|' \
      -e 's|matmul_into(&mut s, \&q, \&kt);|matmul_into(\&mut s, \&q, \&kt_h);|' \
      "$OUTDIR/kv_resident.vx" > "$OUTDIR/kv_unplaced.vx"
}
write_programs

run_one() { # label program remote?
  local label="$1" prog="$2" remote="$3" t0 t1 out
  t0=$(python3 -c 'import time; print(time.time())')
  if [ "$remote" = yes ]; then
    out=$(VX_FLEET_STRICT=1 VX_FLEET_MANIFEST="$OUTDIR/manifest" \
          "$VXC" "$prog" --run 2>/dev/null | tail -1)
  else
    out=$("$VXC" "$prog" --run 2>/dev/null | tail -1)
  fi
  t1=$(python3 -c 'import time; print(time.time())')
  printf '  %-18s %-14s %ss\n' "$label" "$out" \
    "$(python3 -c "print(f'{$t1-$t0:.3f}')")"
  echo "$out" > "$OUTDIR/answer-$label.txt"
  sleep 1
}

echo
echo "=== answers, and how long they took ==="
run_one matmul-local     "$OUTDIR/matmul_unplaced.vx" no
run_one matmul-unplaced  "$OUTDIR/matmul_unplaced.vx" yes
run_one kv-local         "$OUTDIR/kv_unplaced.vx"     no
run_one kv-unplaced      "$OUTDIR/kv_unplaced.vx"     yes
run_one kv-resident      "$OUTDIR/kv_resident.vx"     yes

echo
echo "=== did the other architecture agree? ==="
# The worker is x86-64 and the host arm64. Same program, same numbers, or the
# wire format does not survive the crossing and every timing above is noise.
for pair in "matmul-local matmul-unplaced" "kv-local kv-unplaced" "kv-local kv-resident"; do
  set -- $pair
  a=$(cat "$OUTDIR/answer-$1.txt"); b=$(cat "$OUTDIR/answer-$2.txt")
  if [ "$a" = "$b" ] && [ -n "$a" ]; then
    printf '  %-14s == %-14s  %s\n' "$1" "$2" "$a"
  else
    printf '  %-14s != %-14s  DISAGREE: %s vs %s\n' "$1" "$2" "$a" "$b"
  fi
done

echo
echo "=== what crossed the wire ==="
"${SSH[@]}" "$REMOTE" 'grep "served" ~/vx-fleet/worker.log' 2>/dev/null \
  | sed 's/^/  /' || echo "  (no worker log)"
echo
echo "  One line per run, in the order above. Counters reset per connection."
echo "Evidence in $OUTDIR/"
