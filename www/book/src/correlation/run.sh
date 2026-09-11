#!/usr/bin/env bash
# Run every example and print its verdict. Each one is meant to be refused, or
# to compile into IR that shows the transported fact -- so a clean run here is
# a run where the compiler says "no" six times for six different reasons.
#
#   ./www/book/src/correlation/run.sh
#
# Needs `vxc` (from `cargo build --release`, or on PATH). Example 04 needs z3
# and is skipped without it. The -O3 comparison in 05 needs `mlir-translate`
# and `opt` from the same LLVM as the build; it is skipped without them.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
VXC="${VXC:-}"
if [ -z "$VXC" ]; then
  for cand in "$ROOT/target/release/vxc" "$ROOT/target/debug/vxc" "$(command -v vxc || true)"; do
    if [ -x "$cand" ]; then VXC="$cand"; break; fi
  done
fi
if [ -z "$VXC" ]; then
  echo "no vxc found: build one with \`cargo build --release\` or set VXC=" >&2
  exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0

rule() { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }

# Report whether the expected marker showed up, and show the line it came from.
expect() {
  local marker="$1" out="$2"
  if grep -q -- "$marker" "$out"; then
    grep -m1 -- "$marker" "$out" | sed 's/^/  /'
    echo "  -> as expected"
  else
    echo "  -> MISSING: expected '$marker'" >&2
    fail=1
  fi
}

rule "01  residency across a call"
"$VXC" "$HERE/01_residency_across_a_call.vx" >"$TMP/01" 2>&1
expect "E6003" "$TMP/01"

rule "02  working set across a call"
"$VXC" "$HERE/02_working_set_across_a_call.vx" >"$TMP/02" 2>&1
expect "E6027" "$TMP/02"

rule "03  reachability across a generic call"
"$VXC" "$HERE/03_reachability_across_a_generic_call.vx" >"$TMP/03" 2>&1
expect "Reachable<B, C>" "$TMP/03"

rule "04  freshness across the launch"
if command -v z3 >/dev/null 2>&1; then
  "$VXC" "$HERE/04_freshness_across_the_launch.vx" --verify-seams >"$TMP/04" 2>&1
  expect "E6004" "$TMP/04"
  # The same obligation, discharged: the synchronizing transfer is admitted.
  sed 's/to_device_relaxed/to_device/' "$HERE/04_freshness_across_the_launch.vx" >"$TMP/04ok.vx"
  if "$VXC" "$TMP/04ok.vx" --verify-seams --action emit-mlir -o /dev/null >"$TMP/04ok" 2>&1; then
    echo "  .to_device() instead: admitted"
  else
    echo "  -> MISSING: .to_device() should be admitted" >&2
    fail=1
  fi
else
  echo "  -> skipped: needs z3 on PATH"
fi

rule "05  certificate across the launch"
"$VXC" "$HERE/05_certificate_across_the_launch.vx" --action emit-mlir \
       --emit-seam-certs --legacy-codegen >"$TMP/05" 2>&1
expect "llvm.intr.assume" "$TMP/05"
if command -v mlir-translate >/dev/null 2>&1 && command -v opt >/dev/null 2>&1; then
  # What the certificate buys: -O3 folds the guard and deletes the FMA chain.
  for v in base cert; do
    certs=""; [ "$v" = cert ] && certs="--emit-seam-certs"
    "$VXC" "$HERE/05_certificate_across_the_launch.vx" --action emit-llvm \
           --legacy-codegen $certs >"$TMP/$v.mlir" 2>/dev/null
    mlir-translate --mlir-to-llvmir "$TMP/$v.mlir" -o "$TMP/$v.ll" 2>/dev/null
    opt -O3 -S "$TMP/$v.ll" -o "$TMP/$v.o3.ll" 2>/dev/null
    echo "  $v: fmul=$(grep -c fmul "$TMP/$v.o3.ll") fadd=$(grep -c fadd "$TMP/$v.o3.ll")"
  done
  if [ "$(grep -c fmul "$TMP/cert.o3.ll")" -ge "$(grep -c fmul "$TMP/base.o3.ll")" ]; then
    echo "  -> MISSING: the certificate did not reduce the chain" >&2
    fail=1
  else
    echo "  -> the chain is gone only with the certificate"
  fi
else
  echo "  (skipping the -O3 comparison: needs mlir-translate and opt)"
fi

rule "06  capacity against a declared machine"
"$VXC" --host default --machine "$ROOT/fleet/a100-40.vx" \
       "$HERE/06_capacity_against_a_declared_machine.vx" \
       --action emit-mlir -o /dev/null >"$TMP/06" 2>&1
expect "E6009" "$TMP/06"
if "$VXC" --host default --machine "$ROOT/fleet/a100-80.vx" \
          "$HERE/06_capacity_against_a_declared_machine.vx" \
          --action emit-mlir -o /dev/null >"$TMP/06ok" 2>&1; then
  echo "  same text on a100-80: admitted"
else
  echo "  -> MISSING: a100-80 should be admitted" >&2
  fail=1
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "all examples behaved as documented"
else
  echo "some examples did not behave as documented" >&2
fi
exit "$fail"
