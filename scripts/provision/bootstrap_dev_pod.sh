#!/usr/bin/env bash
#===- bootstrap_dev_pod.sh - a rented box to a working bench, in one go --===#
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===#
#
# Takes a freshly rented x86 Linux GPU box from nothing to a compiler that
# builds, a bundle that runs, and a flash bench that reports where it ran.
#
# This is the *compiler development* loop -- the bounded exception in
# "Build & deployment discipline" in gpu_disaggregated_inference.md. A pod
# running a workload gets an artifact instead (scripts/make_gpu_bundle.sh).
# The difference that earns the exception: developing kernel emission means a
# compiler build per iteration, and shipping 2.2 MB of source to a 256-core pod
# beats shipping a 230 MB binary by roughly fifty times the wall clock.
#
# What it does NOT do, and must not start doing: clone, add a remote, or carry
# a credential. The tarball is built here from `git ls-files` and streamed over
# ssh. The pod can never reach the repository.
#
# Usage:
#   scripts/bootstrap_dev_pod.sh -h root@1.2.3.4 -p 10081 -i ~/.ssh/id_ed25519
#   scripts/bootstrap_dev_pod.sh -h ubuntu@ec2-... -i ~/.ssh/key.pem --skip-setup
#
#   -h  user@host              (required)
#   -p  ssh port               (default 22)
#   -i  identity file          (required)
#   -r  remote directory       (default /root/vx, or ~/vx for non-root)
#   -t  target dir on the pod  (default <remote>-target -- see below)
#   --skip-setup               toolchain is already installed; just ship + build
#   --no-bench                 stop after the build; do not assemble the bundle
#
# Every step below is a thing that cost time on 2026-08-19/20 across an EC2 x86
# box, an H100 pod and two A100 pods. The comments say which.
#
#===----------------------------------------------------------------------===#

set -euo pipefail

HOST="" PORT=22 KEY="" REMOTE="" TARGET="" DO_SETUP=1 DO_BENCH=1
while [ $# -gt 0 ]; do
  case "$1" in
    -h) HOST="$2"; shift 2 ;;
    -p) PORT="$2"; shift 2 ;;
    -i) KEY="$2"; shift 2 ;;
    -r) REMOTE="$2"; shift 2 ;;
    -t) TARGET="$2"; shift 2 ;;
    --skip-setup) DO_SETUP=0; shift ;;
    --no-bench) DO_BENCH=0; shift ;;
    --help) sed -n '24,34p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] && [ -n "$KEY" ] || { echo "error: -h and -i are required" >&2; exit 2; }

SSH_PROBE=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$HOST")
if [ -z "$REMOTE" ]; then
  # Resolved on the far side rather than assumed: every path below is embedded in
  # single-quoted remote commands, so a literal `$HOME` would arrive unexpanded and
  # `mkdir -p '$HOME/vx'` would make a directory named `$HOME`. Ask once, use the answer.
  REMOTE="$("${SSH_PROBE[@]}" 'echo "$HOME/vx"')" \
    || { echo "error: cannot reach $HOST" >&2; exit 1; }
fi
# The build tree, deliberately NOT under the source tree and NOT under
# /workspace. On RunPod /workspace is network storage: `cargo build` there took
# 10m57s and then failed with `rust-lld: Input/output error`, against 1m30s on
# the container's local disk (#369). Anywhere local is fine; this is local.
TARGET="${TARGET:-${REMOTE}-target}"

SSH=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p "$PORT" -i "$KEY" "$HOST")
say() { printf '\n==> %s\n' "$*"; }

# --- 0. what are we even talking to? ----------------------------------------
say "the machine"
"${SSH[@]}" 'echo "  host:   $(uname -sm), $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "  cores:  $(nproc)"
  echo "  cuda:   $(ls -d /usr/local/cuda-* 2>/dev/null | tr "\n" " ")"
  # nvidia-smi reports "could not communicate with the driver" on STDOUT and still
  # exits nonzero, so neither 2>/dev/null nor `||` alone suppresses it: piping
  # straight to sed prints the error as though it were a GPU, and then an empty
  # line as well. Capture, check, then decide.
  g=$(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>/dev/null) \
    && [ -n "$g" ] && echo "$g" | sed "s/^/  gpu:    /" || echo "  gpu:    none visible"'

# --- 1. the source, without a credential in sight ---------------------------
#
# `git ls-files` rather than `tar .`: the working tree carries `target/` (115 GB
# on the dev machine), `.rustup`, and `venv`, and a plain `tar czf` of it
# produced a 764 MB archive before this was noticed. Tracked files are 2.2 MB.
#
# Streamed through ssh rather than scp: scp to a RunPod box was refused outright
# on 2026-08-20 and was flaky on an earlier pod. `cat | ssh 'cat >'` has worked
# on every box tried.
say "shipping the source ($(git ls-files | wc -l | tr -d ' ') tracked files)"
git ls-files -z | tar czf - --null -T - \
  | "${SSH[@]}" "mkdir -p '$REMOTE' && cd '$REMOTE' && tar xzf - 2>/dev/null; echo '  unpacked'"

# --- 2. toolchain -----------------------------------------------------------
if [ "$DO_SETUP" = 1 ]; then
  say "toolchain (LLVM 22, rust, z3) -- several minutes on a cold box"
  # RunPod containers run as root with no `sudo` binary at all, which
  # setup_linux.sh calls unconditionally. A shim is honest here: the script's
  # `sudo` means "with privilege", and root already has it.
  "${SSH[@]}" "command -v sudo >/dev/null 2>&1 || {
       printf '#!/bin/sh\nexec \"\$@\"\n' > /usr/local/bin/sudo && chmod +x /usr/local/bin/sudo
       echo '  installed a sudo shim (root container, no sudo binary)'; }
     cd '$REMOTE' && bash scripts/provision/setup_linux.sh 2>&1 | tail -3"
fi

# --- 3. config.local, then the build -----------------------------------------
#
# One build covers both: the workspace names stdlib/rust_core in `default-members`,
# so libvx_std_core.so comes out alongside vxc. The `ls` is the check that it did --
# without that library the compiler looks built and fails on the first `--run`.
say "building vxc and libvx_std_core"
"${SSH[@]}" "cd '$REMOTE' && export PATH=\"\$HOME/.cargo/bin:\$PATH\" \
  && ./setup.sh >/dev/null 2>&1 && . ./config.local \
  && export CARGO_TARGET_DIR='$TARGET' CARGO_BUILD_JOBS=\$(( \$(nproc) - 2 )) \
  && time cargo build --release 2>&1 | grep -E '^error|Finished' \
  && ls -la '$TARGET/release/vxc' '$TARGET/release/libvx_std_core.so'"

[ "$DO_BENCH" = 1 ] || { say "done (build only)"; exit 0; }

# --- 4. the bundle ----------------------------------------------------------
#
# Three symlinks, each for a failure that reads as something else:
#
#   target/   the JIT links against `target/release/libvx_std_core.so` by a
#             path relative to the CWD, so a bundle without it dies in clang
#             with "no such file or directory" naming a path that does exist --
#             just not from there. CARGO_TARGET_DIR does not reach the JIT.
#   stdlib/   imports resolve against `stdlib/std` relative to the CWD unless
#             VX_STD_PATH is set (src/module_loader.rs). Without it every
#             program fails at "Could not resolve import 'std::math'", which
#             looks like a broken bundle rather than a missing symlink.
#   tests/    `import tests::modules::llama_rt` for the llama benchmarks.
say "assembling the bench bundle"
"${SSH[@]}" "set -e; cd '$REMOTE'
  ln -sfn '$TARGET' target
  B='$REMOTE/../bundle'; mkdir -p \"\$B\"
  cp scripts/campaigns/flash/run_flash_bench.sh \"\$B/\"
  cp scripts/campaigns/flash/flash_attention_bench.vx \"\$B/\"
  ln -sfn '$REMOTE/stdlib' \"\$B/stdlib\"
  ln -sfn '$REMOTE/tests'  \"\$B/tests\"
  ln -sfn '$TARGET'        \"\$B/target\"
  ln -sf '$TARGET/release/vxc' \"\$B/vxc\"
  ln -sf '$TARGET/release/libvx_std_core.so' \"\$B/libvx_std_core.so\"
  printf '. %s/config.local\nexport PATH=/usr/local/cuda/bin:\$PATH\n' '$REMOTE' > \"\$B/env.sh\"
  chmod +x \"\$B/run_flash_bench.sh\"
  echo \"  bundle at \$(cd \"\$B\" && pwd)\""

# --- 5. prove it ------------------------------------------------------------
#
# `ulimit -s`: above K=512 the MLIR memref printer recurses deep enough to
# overflow an 8 MB stack, and the handler needs stack to print a backtrace, so
# it dies with no diagnostic at all (Vx#376). The bench raises it too; this is
# here so the smoke test matches what the bench will do.
say "smoke test"
"${SSH[@]}" "cd '$REMOTE/../bundle' && . ./env.sh && ulimit -s 524288
  sed -e 's/__VX_SQ__/128/g' -e 's/__VX_SK__/512/g' -e 's/__VX_HD__/64/g' \
      -e 's/__VX_TILE__/64/g' -e 's/__VX_NT__/8/g' -e 's/__VX_BENCH_NOTE__/bootstrap smoke/' \
      flash_attention_bench.vx > /tmp/smoke.vx
  VX_DISPATCH_VERBOSE=1 ./vxc /tmp/smoke.vx --run 2>&1 \
    | grep -E 'Vx CUDA|SIGSEGV' | sed 's/^/  /'
  # 0.01*(512-1)/2 = 2.555, the closed form the bench checks at every size.
  got=\$(VX_DISPATCH_VERBOSE=0 ./vxc /tmp/smoke.vx --run 2>/dev/null | grep -oE '^\[\[[0-9.]+' | tr -d '[')
  echo \"  o[0][0] = \${got:-<nothing>} (expected 2.555)\""

say "ready.  ssh -p $PORT -i $KEY $HOST, then:"
echo "    cd $(dirname "$REMOTE")/bundle && ulimit -s 524288 && ./run_flash_bench.sh"
