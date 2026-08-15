#!/usr/bin/env bash
#===- run_gpu_fleet_demo.sh - one program, two GPUs, another machine -----===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Runs a Vx program whose data lives on two GPUs in a rented box, from a laptop
# that has none (#347, #348).
#
# The host compiles and drives; the pod holds the data and does the arithmetic.
# Two worker processes, one per GPU, each allocating on the device its own
# `--topology` names -- so which GPU a tensor lands on is decided by the
# manifest and the program, and nothing here sets CUDA_VISIBLE_DEVICES.
#
# Three things are being demonstrated, and they fail independently:
#
#   residency   operands cross once and stay. Measured as messages per
#               dispatch, which is what a round trip costs on a real link.
#   read-back   the answer comes home and equals the local one. Without this
#               the rest is a message count with nothing checking it.
#   handoff     data produced on GPU 0 moves to GPU 1 and is still correct
#               there. This is the disaggregated shape -- prefill on one
#               device, decode on another -- and the only one of the three
#               that needs the second GPU.
#
#   ./scripts/run_gpu_fleet_demo.sh -H ubuntu@pod -i ~/.ssh/key.pem
#
# `--backend cpu` builds the workers against host_dispatch.cpp instead, which
# needs no GPU and exercises every line of this script apart from nvcc's. That
# is how the two-GPU shape was developed: run it against any Linux box first,
# and a failure on the pod is then about CUDA rather than about the harness.
#
# Rehearsed that way on loopback, where it found four defects that a GPU would
# have found more expensively -- a handoff with no caller in generated code, a
# connection pool that deadlocked on two roles per machine, a staging buffer
# that moved zeros while the message counts stayed right, and a build that did
# not rebuild. See tests/integration_test/remote_client_test.rs.
#
# Nothing is exposed publicly: each worker binds its own loopback and an SSH
# tunnel forwards to it, so the manifest names 127.0.0.1 on both sides. The
# workers and the sources they were built from do not outlive the run.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

REMOTE=""
KEY=""
BACKEND="cuda"
PORT_A=24001
PORT_B=24002
OUTDIR="${TMPDIR:-/tmp}/vx-gpu-fleet"

while [ $# -gt 0 ]; do
  case "$1" in
    -H|--host)    REMOTE="$2"; shift 2 ;;
    -i|--key)     KEY="$2"; shift 2 ;;
    -b|--backend) BACKEND="$2"; shift 2 ;;
    -o|--out)     OUTDIR="$2"; shift 2 ;;
    -h|--help)    sed -n '9,46p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$REMOTE" ] || { echo "error: -H user@host is required" >&2; exit 2; }
case "$BACKEND" in cuda|cpu) ;; *) echo "error: --backend cuda|cpu" >&2; exit 2 ;; esac

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20)
[ -n "$KEY" ] && SSH+=(-i "$KEY")

VXC="${VXC:-./target/release/vxc}"
[ -x "$VXC" ] || { echo "error: no $VXC (cargo build --release)" >&2; exit 1; }
mkdir -p "$OUTDIR"

cleanup() {
  for p in "$OUTDIR"/tunnel-*.pid; do
    [ -f "$p" ] && kill "$(cat "$p")" 2>/dev/null || true
  done
  rm -f "$OUTDIR"/tunnel-*.pid
  # Someone else's machine: no workers left running, no sources left behind.
  "${SSH[@]}" "$REMOTE" 'pkill -x vx-worker 2>/dev/null; rm -rf ~/vx-fleet /tmp/vx_fleet_rt.tgz' 2>/dev/null || true
}
trap cleanup EXIT

echo "==> what the pod has"
"${SSH[@]}" "$REMOTE" '
  echo "    arch:    $(uname -m)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "    gpus:    $(nvidia-smi --query-gpu=name --format=csv,noheader | tr "\n" "," | sed "s/,$//")"
  else
    echo "    gpus:    none visible (nvidia-smi absent)"
  fi
  echo "    nvcc:    $(command -v nvcc || echo absent)"
' || true

echo "==> shipping the runtime (an archive, not a checkout)"
COPYFILE_DISABLE=1 tar czf "$OUTDIR/rt.tgz" runtime include
scp "${SSH[@]:1}" "$OUTDIR/rt.tgz" "$REMOTE:/tmp/vx_fleet_rt.tgz" >/dev/null

# The worker holds no vendor code. What makes it a GPU worker is the backend it
# is linked against, and nothing in the worker itself knows which that was.
if [ "$BACKEND" = cuda ]; then
  BUILD='
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    [ -d "$CUDA_HOME" ] || CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    g++ -std=c++17 -O2 -Wall runtime/vx_worker_main.cpp runtime/cuda_dispatch.cpp \
      -Iinclude -Iruntime -I"$CUDA_HOME/include" \
      -L"$CUDA_HOME/lib64" -Wl,-rpath,"$CUDA_HOME/lib64" \
      -lcudart -lcublas -lffi -o vx-worker'
else
  BUILD='
    g++ -std=c++17 -O2 -Wall runtime/vx_worker_main.cpp runtime/host_dispatch.cpp \
      -Iinclude -Iruntime -lffi -o vx-worker'
fi

echo "==> building the $BACKEND worker there"
"${SSH[@]}" "$REMOTE" "
  set -e
  pkill -x vx-worker 2>/dev/null || true
  rm -rf ~/vx-fleet && mkdir -p ~/vx-fleet
  tar xzf /tmp/vx_fleet_rt.tgz -C ~/vx-fleet 2>/dev/null
  cd ~/vx-fleet
  $BUILD
"

# Two workers, one per device. The worker allocates and dispatches on the
# topology *it* was given, so `--topology 501` is what puts a tensor on GPU 1 --
# the manifest only decides which worker a placement reaches.
echo "==> starting two workers (topology 500 -> GPU 0, 501 -> GPU 1)"
"${SSH[@]}" "$REMOTE" "
  set -e
  cd ~/vx-fleet
  setsid ./vx-worker --port $PORT_A --topology 500 --worker-id 1 --verbose \
    > worker-a.log 2>&1 < /dev/null &
  setsid ./vx-worker --port $PORT_B --topology 501 --worker-id 2 --verbose \
    > worker-b.log 2>&1 < /dev/null &
  sleep 1
  [ \"\$(pgrep -xc vx-worker)\" = 2 ] || { echo 'the workers did not both start'; cat worker-*.log; exit 1; }
"

echo "==> tunnelling both"
for P in "$PORT_A" "$PORT_B"; do
  "${SSH[@]}" -o ExitOnForwardFailure=yes -N -L "$P:127.0.0.1:$P" "$REMOTE" \
    > "$OUTDIR/tunnel-$P.log" 2>&1 &
  echo $! > "$OUTDIR/tunnel-$P.pid"
done
sleep 3
for P in "$PORT_A" "$PORT_B"; do
  (exec 3<>/dev/tcp/127.0.0.1/"$P") 2>/dev/null || {
    echo "error: no tunnel on $P; see $OUTDIR/tunnel-$P.log" >&2; exit 1; }
done

# A device is two names: a memory space that data lands in, and a topology that
# work is placed on. Both resolve to the same worker, which is why the pool has
# to key connections by machine rather than by name.
cat > "$OUTDIR/manifest" <<EOF
HBM_A  127.0.0.1  $PORT_A
DevA   127.0.0.1  $PORT_A
HBM_B  127.0.0.1  $PORT_B
DevB   127.0.0.1  $PORT_B
EOF

cat > "$OUTDIR/resident.vx" <<'EOF'
// Operands placed once, eight dispatches over them, the answer read home.
Memory CPU_DRAM {}
Memory HBM_A {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Topology DevA { memory: Memory::HBM_A, visible: [ Memory::HBM_A ] }

fn main() -> i32 {
  let mut a_h = Tensor<f32>([ 64, 64 ]);
  let mut b_h = Tensor<f32>([ 64, 64 ]);
  let mut c_h = Tensor<f32>([ 64, 64 ]);
  for i in 0..64 {
    for j in 0..64 {
      a_h[i][j] = ((i + j) as f32) * 0.01;
      b_h[i][j] = ((i - j) as f32) * 0.02;
    }
  }
  let a = transfer(a_h, Memory::HBM_A);
  let b = transfer(b_h, Memory::HBM_A);
  let mut c = transfer(c_h, Memory::HBM_A);
  for step in 0..8 {
    spawn on(Topology::DevA) {
      matmul_into(&mut c, &a, &b);
    }
  }
  let home = transfer(c, Memory::CPU_DRAM);
  print(home[0][0]);
  return 0;
}
EOF

cat > "$OUTDIR/handoff.vx" <<'EOF'
// Produced on the first device, handed to the second, read home from there.
Memory CPU_DRAM {}
Memory HBM_A {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Memory HBM_B {
  within: Memory::CPU_DRAM, capacity: 40 GiB, bandwidth: 3 TB/s, managed: cached
}
Topology DevA { memory: Memory::HBM_A, visible: [ Memory::HBM_A ] }
Topology DevB { memory: Memory::HBM_B, visible: [ Memory::HBM_B ] }

fn main() -> i32 {
  let mut a_h = Tensor<f32>([ 64, 64 ]);
  let mut b_h = Tensor<f32>([ 64, 64 ]);
  let mut c_h = Tensor<f32>([ 64, 64 ]);
  for i in 0..64 {
    for j in 0..64 {
      a_h[i][j] = ((i + j) as f32) * 0.01;
      b_h[i][j] = ((i - j) as f32) * 0.02;
    }
  }
  let a = transfer(a_h, Memory::HBM_A);
  let b = transfer(b_h, Memory::HBM_A);
  let mut c = transfer(c_h, Memory::HBM_A);
  for step in 0..8 {
    spawn on(Topology::DevA) {
      matmul_into(&mut c, &a, &b);
    }
  }
  // The handoff. On one machine with two GPUs this is a peer copy; across a
  // fleet it is a read from one worker and a write to the other.
  let c_b = transfer(c, Memory::HBM_B);
  let home = transfer(c_b, Memory::CPU_DRAM);
  print(home[0][0]);
  return 0;
}
EOF

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
  printf '  %-16s %-14s %ss\n' "$label" "$out" \
    "$(python3 -c "print(f'{$t1-$t0:.3f}')")"
  echo "$out" > "$OUTDIR/answer-$label.txt"
  sleep 1
}

# The first compile of the session pays for a cold page cache and the loader,
# and left in the table it reads as the local case being slow.
"$VXC" "$OUTDIR/resident.vx" --run >/dev/null 2>&1 || true

echo
echo "=== answers, and how long they took ==="
run_one resident-local  "$OUTDIR/resident.vx" no
run_one resident-pod    "$OUTDIR/resident.vx" yes
run_one handoff-local   "$OUTDIR/handoff.vx"  no
run_one handoff-pod     "$OUTDIR/handoff.vx"  yes

echo
echo "=== did the pod agree with this machine? ==="
for pair in "resident-local resident-pod" "handoff-local handoff-pod"; do
  set -- $pair
  a=$(cat "$OUTDIR/answer-$1.txt"); b=$(cat "$OUTDIR/answer-$2.txt")
  if [ "$a" = "$b" ] && [ -n "$a" ]; then
    printf '  %-16s == %-16s %s\n' "$1" "$2" "$a"
  else
    printf '  %-16s != %-16s DISAGREE: %s vs %s\n' "$1" "$2" "$a" "$b"
  fi
done

echo
echo "=== what crossed the wire ==="
for W in a b; do
  echo "  worker $W:"
  "${SSH[@]}" "$REMOTE" "grep 'served' ~/vx-fleet/worker-$W.log" 2>/dev/null \
    | sed 's/^/    /' || echo "    (nothing)"
done
echo
echo "  One line per connection. A handoff shows a FETCH on the first worker"
echo "  and a TRANSFER on the second: the bytes were read from one and written"
echo "  to the other."
echo "Evidence in $OUTDIR/"
