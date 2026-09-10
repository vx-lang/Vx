#!/usr/bin/env python3
"""M5: does the graph pick the faster route?

When two routes connect the same pair of spaces, the graph picks one. This asks whether it picked
the faster one, and whether the answer depends on transfer size.

The predicted half of M5 needs no hardware, because both halves of the comparison are things the
compiler already computes: which route it chose, and what every route costs. So this runs on a
laptop and the machine time is spent only on confirming the *measured* column.

How a route's cost is obtained, and why it is not simply computed here:

  The compiler is the authority on cost. This script reimplements the per-hop rule -- ceil(bytes /
  bandwidth), summed over hops -- only so it can price the routes the compiler *did not* pick, and
  it is not trusted to do that until it has reproduced the compiler's own number for the route the
  compiler *did* pick, exactly, on every row. A mismatch aborts the run. Without that check this
  script would be quoting a model of a model, and the headline ratio would be an artifact of my
  arithmetic rather than a finding about the graph.

Usage:
    python3 utils/memalg/route_choice.py --machine fleet/node-2gpu-h100.vx \\
        --from HBM --to PEER_HBM
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile

# The frozen cell sizes, so an M5 row can be read next to an M1 row for the same transfer.
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Decimal, not binary. fleet/README.md records that reading `TB` as 2^40 cost 9% against every
# `spec:` line that meant 10^12, and the machine files are written in the vendors' units.
SCALE = {"B": 1.0, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}

EDGE_RE = re.compile(
    r"transfer\s+Memory::(\w+)\s*->\s*Memory::(\w+)\s*(?::\s*([0-9.]+)\s*([KMGT]?B)/s)?"
)


def parse_edges(path):
    """[(src, dst, bytes_per_second or None)] from a machine file's `transfer` declarations.

    A rate of None is a containment-derived edge: the file declares the hop exists and leaves its
    cost to the roofline between the spaces at its ends. Those are not priced here -- see
    `price_path`.
    """
    edges = []
    with open(path) as f:
        for line in f:
            line = line.split("//", 1)[0]  # a commented-out edge is not an edge
            m = EDGE_RE.search(line)
            if not m:
                continue
            src, dst, val, unit = m.groups()
            edges.append((src, dst, float(val) * SCALE[unit] if val else None))
    return edges


def simple_paths(edges, src, dst, limit=6):
    """Every simple path src->dst, shortest first. Small graphs, so an exhaustive walk is fine."""
    adj = {}
    for a, b, _ in edges:
        adj.setdefault(a, []).append(b)
    out = []

    def walk(node, path, seen):
        if len(path) > limit:
            return
        if node == dst and len(path) > 1:
            out.append(list(path))
            return
        for nxt in adj.get(node, []):
            if nxt in seen:
                continue
            seen.add(nxt)
            path.append(nxt)
            walk(nxt, path, seen)
            path.pop()
            seen.discard(nxt)

    walk(src, [src], {src})
    return sorted(out, key=len)


def price_path(edges, path, nbytes):
    """(picoseconds, None) for a path every hop of which declares a rate, else (None, reason).

    Per-hop `ceil`, then sum -- matching the compiler, which rounds each hop to a whole picosecond
    before adding. Summing first and rounding once differs by up to one quantum per hop, which is
    exactly the size of the discrepancy that would otherwise be read as a size effect at 4 KiB.
    """
    rate = {(a, b): r for a, b, r in edges}
    total = 0
    for a, b in zip(path, path[1:]):
        r = rate.get((a, b))
        if r is None:
            return None, f"hop {a}->{b} is containment-derived, not a declared link rate"
        total += math.ceil(nbytes / r * 1e12)
    return total, None


def picked_route(vxc, machine, src, dst, dim, workdir):
    """((path, picoseconds), None) — what the compiler chose and what it charged for it."""
    prog = os.path.join(workdir, f"probe_{dim}.vx")
    with open(prog, "w") as f:
        f.write(
            f"fn main() -> i32 {{\n"
            f"  let tile : Tensor<f32, [{dim}, {dim}]> = Tensor<f32>([{dim}, {dim}]);\n"
            f"  let a = transfer(tile, Memory::{src});\n"
            f"  let b = transfer(a, Memory::{dst});\n"
            f"  let _sink = b;\n"
            f"  return 0;\n"
            f"}}\n"
        )
    js = os.path.join(workdir, f"probe_{dim}.json")
    proc = subprocess.run(
        [vxc, "--machine", machine, "--host", "default", prog,
         "--diagnostics-json", js, "--emit-mlir", "-o", os.devnull],
        capture_output=True, text=True,
    )
    if not os.path.exists(js):
        return None, (proc.stderr or proc.stdout).strip()[:300]
    rec = json.load(open(js))
    # The staging hop into `src` is a separate transfer in the program and not part of the route
    # under test. Dropped by its source space rather than by position, because a route that
    # legitimately begins at the host would otherwise be silently truncated.
    routes = [r for r in rec.get("routes", []) if r["path"][0] != "CPU_DRAM"]
    if not routes:
        return None, f"verdict={rec.get('verdict')}, no non-staging route"
    if any(r.get("derived_cost") is None for r in routes):
        return None, "the compiler did not price every hop of the chosen route"
    path = [routes[0]["path"][0]] + [r["path"][-1] for r in routes]
    return (path, sum(r["derived_cost"] for r in routes)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vxc", default="target/release/vxc")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    args = ap.parse_args()

    edges = parse_edges(args.machine)
    cands = simple_paths(edges, args.src, args.dst)
    if not cands:
        sys.exit(f"no route {args.src} -> {args.dst} in {args.machine}")

    print(f"machine   : {args.machine}")
    print(f"transfer  : {args.src} -> {args.dst}")
    print(f"routes    : {len(cands)} simple path(s) in the declared graph")
    for p in cands:
        print(f"            {' -> '.join(p)}  ({len(p) - 1} hop(s))")
    if len(cands) == 1:
        print(
            "\nOnly one route exists, so there is no choice to score. M5 needs a pair whose\n"
            "endpoints are connected two ways -- see utils/memalg/m5_partial_nvlink.vx."
        )

    workdir = tempfile.mkdtemp(prefix="m5-")
    rows = []
    for dim in DIMS:
        nbytes = 4 * dim * dim
        got, err = picked_route(args.vxc, args.machine, args.src, args.dst, dim, workdir)
        if err:
            print(f"  {nbytes}B: {err}", file=sys.stderr)
            continue
        path, cost = got

        # The check that licenses every other number in this table. If this script cannot
        # reproduce the compiler's charge for the route the compiler chose, its prices for the
        # routes it did not choose mean nothing, and the honest move is to stop rather than to
        # publish a ratio between one real number and one invented one.
        mine, why = price_path(edges, path, nbytes)
        if mine is None:
            sys.exit(
                f"cannot price the chosen route {' -> '.join(path)} at {nbytes}B: {why}.\n"
                "This is a limit of this script, not a defect in the route: a containment hop is\n"
                "priced from the roofline between the spaces at its ends, which the compiler\n"
                "composes and this script does not reimplement. So M5 currently scores routes\n"
                "made of declared link rates -- the between-device edges, which is where two\n"
                "routes actually compete. Scoring an on-die alternative needs the compiler to\n"
                "report the routes it rejected, not a second copy of the roofline here."
            )
        if mine != cost:
            sys.exit(
                f"NOT AUTHORITATIVE: at {nbytes}B the compiler charges {cost} ps for "
                f"{' -> '.join(path)} and this script computes {mine}. The per-hop rule here has "
                "drifted from the compiler's; every alternative-route price below would be wrong "
                "in the same unknown way."
            )

        priced = []
        for p in cands:
            ps, _ = price_path(edges, p, nbytes)
            if ps is not None:
                priced.append((ps, p))
        if not priced:
            continue
        best_ps, best_p = min(priced)
        rows.append(
            {
                "bytes": nbytes,
                "picked": " -> ".join(path),
                "picked_ps": cost,
                "best": " -> ".join(best_p),
                "best_ps": best_ps,
                "ratio": cost / best_ps if best_ps else float("nan"),
            }
        )

    if not rows:
        sys.exit("no size produced a comparable route")

    print()
    hdr = (
        f"{'bytes':>12}  {'route the graph picked':<26}"
        f"{'picked ps':>16}{'best ps':>16}{'slowdown':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['bytes']:>12}  {r['picked']:<26}{r['picked_ps']:>16}{r['best_ps']:>16}"
            f"{r['ratio']:>9.4f}x"
        )

    picks = {r["picked"] for r in rows}
    bests = {r["best"] for r in rows}
    ratios = [r["ratio"] for r in rows]
    print()
    print(f"route chosen        : {'INVARIANT' if len(picks) == 1 else 'SIZE-DEPENDENT'} "
          f"({len(picks)} distinct choice(s) across {len(rows)} sizes)")
    print(f"cheapest route      : {'INVARIANT' if len(bests) == 1 else 'SIZE-DEPENDENT'} "
          f"({len(bests)} distinct)")
    print(f"slowdown            : {min(ratios):.4f}x .. {max(ratios):.4f}x")

    # The structural result, and the reason no amount of machine time will produce a crossover
    # from the *predicted* column. Every edge costs bytes/bandwidth, a line through the origin, so
    # the ratio between any two routes is a constant in bytes. A crossover needs a fixed per-hop
    # term the model does not have, so it is not merely unobserved here -- it is unrepresentable.
    if max(ratios) - min(ratios) < 1e-3:
        print(
            "\nThe slowdown is constant in size to within the per-hop rounding quantum. That is\n"
            "structural, not a coincidence: every edge is priced bytes/bandwidth, a line through\n"
            "the origin, so the ratio between two routes cannot depend on bytes. The model\n"
            "carries no fixed per-transfer term, and so cannot express a size crossover at all --\n"
            "which is the effect M5 set out to locate."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
