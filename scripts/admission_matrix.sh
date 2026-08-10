#!/usr/bin/env bash
#===- admission_matrix.sh - one config against every SKU -----------------===#
#
# Part of the Vx Project, under the BSD 3-Clause License.
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
#
#===----------------------------------------------------------------------===#
#
# Compiles one serving configuration against every machine model in fleet/ and
# prints the verdicts as a table, keeping the per-cell `--diagnostics-json`
# record that produced each one.
#
# This is the demo's compile-time half (#319). The claim is not that a program
# runs; it is that the compiler says which machines it will run on *before* one
# is rented, in bytes, and can be checked afterwards. One (config, SKU) pair is
# one compile is one row.
#
# The program is never edited to change SKU -- only the flag changes. That is
# the property on trial, so the harness cannot be allowed to help: it passes the
# same file every time.
#
# Usage:
#   scripts/admission_matrix.sh [-p program.vx] [-o outdir] [--host <file>]
#
# Defaults to fleet/admit.vx, whose configuration is a 70B-class model at f16.
# Edit the `admit<...>` call in that file to move to another configuration; that
# is a new matrix, not a new program.
#
#===----------------------------------------------------------------------===#

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROGRAM="fleet/admit.vx"
OUTDIR="admission-matrix"
HOST="default"

while [ $# -gt 0 ]; do
  case "$1" in
    -p|--program) PROGRAM="$2"; shift 2 ;;
    -o|--out)     OUTDIR="$2"; shift 2 ;;
    --host)       HOST="$2"; shift 2 ;;
    -h|--help)    sed -n '11,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"

VXC="target/release/vxc"
if [ ! -x "$VXC" ]; then
  echo "error: no compiler at $VXC -- run 'cargo build --release' first" >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# Machine models only. A host file (host-*.vx) describes the machine an
# accelerator hangs off and is passed with --host, not --machine; admit.vx is
# the program under test rather than a SKU.
SKUS=()
for f in fleet/*.vx; do
  base="$(basename "$f")"
  case "$base" in
    admit.vx|host-*) continue ;;
  esac
  SKUS+=("$f")
done

printf '%-26s %-10s %16s %16s %16s\n' SKU VERDICT REQUIRED AVAILABLE MARGIN
printf '%.0s-' {1..88}; echo

for sku in "${SKUS[@]}"; do
  cell="$OUTDIR/$(basename "${sku%.vx}").json"
  "$VXC" --host "$HOST" --machine "$sku" "$PROGRAM" \
         --action emit-mlir -o /dev/null --diagnostics-json "$cell" >/dev/null 2>&1

  python3 - "$cell" "$sku" <<'PY'
import json, os, sys
cell, sku = sys.argv[1], sys.argv[2]
name = os.path.basename(sku)

if not os.path.exists(cell):
    print(f"{name:<26} {'no record':<10}")
    sys.exit(0)

with open(cell) as fh:
    rec = json.load(fh)

# `resident_sets` rather than the capacity field on a diagnostic. Both directions
# carry it -- an admitted cell has no diagnostic to hang numbers off, so reading
# them from one made every acceptance a bare word and every refusal quantified.
# It is also the better number: the working-set total across every resident,
# where a diagnostic's figure may be one offending tile.
sets = rec.get("resident_sets") or []
row = max(sets, key=lambda r: r.get("total_bytes", 0)) if sets else None

def gib(n):
    return "-" if n is None else f"{n / (1 << 30):.1f} GiB"

if row:
    total = row.get("total_bytes")
    cap = row.get("capacity_bytes")
    margin = None if (total is None or cap is None) else cap - total
    print(f"{name:<26} {rec['verdict']:<10} {gib(total):>16} {gib(cap):>16} {gib(margin):>16}")
else:
    print(f"{name:<26} {rec['verdict']:<10} {'':>16} {'':>16} {'':>16}")
PY
done

echo
echo "Per-cell records in $OUTDIR/ -- each carries the verdict, the diagnostic"
echo "codes, and the byte-level capacity fields the row above summarises."
