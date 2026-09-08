#!/usr/bin/env bash
#===- run_two_machine_demo.sh - one program, two machines ----------------===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#                    messages   operands   wall
#   matmul local            -        -     0.30 s
#   matmul unplaced        64   384 KiB    2.9 - 7.6 s
#   matmul placed          15    48 KiB    0.83 - 0.93 s
#   kv local                -        -     0.29 s
#   kv unplaced            64   35.9 KiB   2.3 - 2.5 s
#   kv resident            50    7.1 KiB   1.88 - 1.99 s
#
# Ranges over three runs, not one draw. Every row but `matmul unplaced` repeats
# to within ten percent; that one moved between 2.9 and 7.6 seconds, and it is
# the only row whose cost is bytes rather than round trips -- 384 KiB against
# the next largest 36 KiB. Which is the shape of the argument: a program paying
# per round trip is paying something predictable, and a program re-sending an
# unchanged operand is at the mercy of the link.
#
# The placed row is operands *and* result on the worker, with the answer read
# home once at the end: 15 messages against 64, and 3.4x less wall clock. It
# could not be run at all until recently -- a `transfer` home was lowered to a
# copy from the source address, which on a fleet is a handle naming memory in
# the worker's process, so the program died on a signal instead of printing
# (#321). It is also the row that makes the others trustworthy: a resident
# result nothing reads is a message count with no answer attached, and this one
# is checked against the local run like every other.
#
# The bytes column is TRANSFER only -- the operands, which is what placement
# removes. It is not the total on the wire: the worker counts what it receives,
# so a FETCH appears as its 16-byte request and the 16 KiB of result it sends
# back does not appear at all. Read the message count first; it is the one that
# is a round trip each.
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
COPYFILE_DISABLE=1 tar czf "$OUTDIR/rt.tgz" runtime include
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
  let mut a = Tensor<f32, [64, 64]>::uninit();
  let mut b = Tensor<f32, [64, 64]>::uninit();
  let mut c = Tensor<f32, [64, 64]>::uninit();
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

  # The same matmul with everything placed: operands and result cross once, the
  # dispatches name handles, and the answer is read home at the end. Same
  # arithmetic as the program above, so the two answers have to agree -- which
  # is the point of running it, since a resident result that is never read is a
  # count with nothing checking it.
  cat > "$OUTDIR/matmul_placed.vx" <<'EOF'
Memory CPU_DRAM {}
Memory GPU_HBM {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}

fn main() -> i32 {
  let mut a_h = Tensor<f32, [64, 64]>::uninit();
  let mut b_h = Tensor<f32, [64, 64]>::uninit();
  let mut c_h = Tensor<f32, [64, 64]>::uninit();
  for i in 0..64 {
    for j in 0..64 {
      a_h[i][j] = ((i + j) as f32) * 0.01;
      b_h[i][j] = ((i - j) as f32) * 0.02;
    }
  }
  let a = transfer(a_h, Memory::GPU_HBM);
  let b = transfer(b_h, Memory::GPU_HBM);
  let mut c = transfer(c_h, Memory::GPU_HBM);
  for step in 0..8 {
    spawn on(Topology::GPU) {
      matmul_into(&mut c, &a, &b);
    }
  }
  let home = transfer(c, Memory::CPU_DRAM);
  print(home[0][0]);
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
  let mut kt_h = Tensor<f32, [16, 64]>::uninit();
  for d in 0..16 {
    for j in 0..64 {
      kt_h[d][j] = ((j - d) as f32) * 0.03;
    }
  }
  let kt = transfer(kt_h, Memory::GPU_HBM);

  let mut q = Tensor<f32, [1, 16]>::uninit();
  let mut s = Tensor<f32, [1, 64]>::uninit();
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
# The first compile of the session pays for a cold page cache and the dynamic
# loader, and the first row was collecting it: 5.407 s for the local matmul
# against 0.298 s for the same run a minute later, which read as the local case
# being the slow one. Throw one away first so every row below is warm.
"$VXC" "$OUTDIR/matmul_unplaced.vx" --run >/dev/null 2>&1 || true

echo "=== answers, and how long they took ==="
run_one matmul-local     "$OUTDIR/matmul_unplaced.vx" no
run_one matmul-unplaced  "$OUTDIR/matmul_unplaced.vx" yes
run_one matmul-placed    "$OUTDIR/matmul_placed.vx"   yes
run_one kv-local         "$OUTDIR/kv_unplaced.vx"     no
run_one kv-unplaced      "$OUTDIR/kv_unplaced.vx"     yes
run_one kv-resident      "$OUTDIR/kv_resident.vx"     yes

echo
echo "=== did the other architecture agree? ==="
# The worker is x86-64 and the host arm64. Same program, same numbers, or the
# wire format does not survive the crossing and every timing above is noise.
for pair in "matmul-local matmul-unplaced" "matmul-local matmul-placed" \
            "kv-local kv-unplaced" "kv-local kv-resident"; do
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
