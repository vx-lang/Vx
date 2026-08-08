#!/usr/bin/env python3
"""Harvest every M-series prediction, one JSON per cell (vx-review#14).

The pre-registration mechanism: predicted first, measured second, provably. After the tag these
files can only change with a dated note saying why, so a measurement that disagrees with them is a
*result* rather than an invitation to re-tune the model.

Each cell is one (machine, target space, transfer size). For each, the compiler is asked what it
predicts and the answer is written verbatim from `--diagnostics-json` -- not summarised, not
rounded, and not hand-transcribed, because a hand-copied prediction is exactly the step where a
number quietly becomes the number that fits.

Cells that the compiler *rejects* are recorded too, with the diagnostic. A capacity rejection is a
prediction as much as a cost is: it says this placement cannot happen on this SKU, and if the
hardware runs it anyway the model is wrong in a way no error percentage would show.
"""

import argparse
import json
import os
import subprocess
import sys

# Log-spaced f32 tile shapes: bytes = 4*d*d, so these land on powers of two from 4 KiB to 1 GiB.
# The small end is where a bandwidth-only model is pre-registered to under-predict cost (a latency
# floor it does not model); the large end is past every cache on every SKU in the fleet.
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# The spaces a program can be asked to place a tile in. `SMEM` exists on every fleet SKU but is
# small, so most sizes will be rejected there -- which is the point of recording rejections.
TARGETS = ["HBM", "L2", "SMEM"]

MACHINES = [
    "fleet/a100-40.vx",
    "fleet/a100-80.vx",
    "fleet/b200.vx",
    "fleet/h100-sxm.vx",
    "fleet/h200.vx",
    "fleet/m4-uma.vx",
    "fleet/mi300x.vx",
    "fleet/node-8gpu.vx",
]


def probe_source(dim, target):
    """A program that stages one tile down to `target`, so every hop on the way is predicted."""
    chain = {"HBM": ["HBM"], "L2": ["HBM", "L2"], "SMEM": ["HBM", "L2", "SMEM"]}[target]
    lines = [
        "fn main() -> i32 {",
        f"  let tile : Tensor<f32, [{dim}, {dim}]> = Tensor<f32>([{dim}, {dim}]);",
    ]
    prev = "tile"
    for i, space in enumerate(chain):
        name = f"t{i}"
        lines.append(f"  let {name} = transfer({prev}, Memory::{space});")
        prev = name
    lines.append(f"  let _sink = {prev};")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def harvest(vxc, machine, dim, target, workdir):
    src = os.path.join(workdir, f"probe_{dim}_{target}.vx")
    with open(src, "w") as f:
        f.write(probe_source(dim, target))
    js = os.path.join(workdir, f"probe_{dim}_{target}.json")
    proc = subprocess.run(
        [vxc, "--machine", machine, src, "--diagnostics-json", js, "--emit-mlir", "-o", os.devnull],
        capture_output=True,
        text=True,
    )
    if not os.path.exists(js):
        return {"harvest_error": (proc.stderr or proc.stdout).strip()[:400]}
    with open(js) as f:
        rec = json.load(f)
    # The record names the probe by its absolute path, which is wherever this ran. Left in, the
    # frozen artifact is not byte-reproducible (regenerating elsewhere diffs on every file) and it
    # commits a local filesystem path into the paper repo. Replaced with the logical cell name,
    # which is the only part that identifies anything.
    if "file" in rec:
        rec["file"] = f"probe_{dim}_{target}.vx"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vxc", default="target/release/vxc")
    ap.add_argument("--out", required=True, help="directory to write one JSON per cell into")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="regenerate into a second directory and require the two to be byte-identical",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    workdir = os.path.join(args.out, ".probes")
    os.makedirs(workdir, exist_ok=True)

    vx_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    vx_dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()

    index = []
    n_cost, n_reject, n_nocost = 0, 0, 0

    for machine in MACHINES:
        sku = os.path.basename(machine).replace(".vx", "")
        for target in TARGETS:
            for dim in DIMS:
                rec = harvest(args.vxc, machine, dim, target, workdir)
                cell = f"{sku}__{target}__{4 * dim * dim}B"
                routes = rec.get("routes", [])
                verdict = rec.get("verdict", "harvest_error")
                priced = [r for r in routes if r.get("derived_cost") is not None]
                if verdict == "rejected":
                    n_reject += 1
                elif priced:
                    n_cost += 1
                else:
                    n_nocost += 1

                out = {
                    "cell": cell,
                    "machine": machine,
                    "target_space": target,
                    "tile": {"elem": "f32", "dims": [dim, dim], "bytes": 4 * dim * dim},
                    "vx_commit": vx_commit,
                    "vx_dirty": bool(vx_dirty),
                    "prediction": rec,
                }
                path = os.path.join(args.out, f"{cell}.json")
                with open(path, "w") as f:
                    json.dump(out, f, indent=2, sort_keys=True)
                    f.write("\n")
                index.append(
                    {
                        "cell": cell,
                        "machine": machine,
                        "target": target,
                        "bytes": 4 * dim * dim,
                        "verdict": verdict,
                        "priced_hops": len(priced),
                        "total_hops": len(routes),
                    }
                )

    with open(os.path.join(args.out, "INDEX.json"), "w") as f:
        json.dump(
            {"vx_commit": vx_commit, "vx_dirty": bool(vx_dirty), "cells": index},
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(f"cells written : {len(index)}")
    print(f"  with a cost : {n_cost}")
    print(f"  rejected    : {n_reject}  (capacity — a prediction too)")
    print(f"  no cost     : {n_nocost}")
    print(f"vx_commit     : {vx_commit}{' [DIRTY]' if vx_dirty else ''}")
    if vx_dirty:
        print(
            "\nWARNING: the tree is dirty, so these predictions do not correspond to any commit.\n"
            "Commit first — a frozen prediction whose compiler cannot be reconstructed is not\n"
            "pre-registered, it is just a file.",
            file=sys.stderr,
        )
        return 1

    if args.verify:
        # A separate process, because that is the only way to catch the class of bug this check
        # exists for: Rust randomises `HashMap`'s hasher per process, so two harvests *within* one
        # process share a seed and would agree even when the compiler is nondeterministic. This is
        # not hypothetical -- `resident_sets` was emitted in hash order until this check found it,
        # and the same compiler on the same input produced different JSON on consecutive runs.
        import filecmp
        import shutil
        import tempfile

        print("\n== verify: regenerating in a second process ==")
        tmp = tempfile.mkdtemp(prefix="memalg-verify-")
        try:
            rc = subprocess.run(
                [sys.executable, __file__, "--vxc", args.vxc, "--out", tmp],
                capture_output=True,
                text=True,
            )
            if rc.returncode != 0:
                print(f"verify run failed:\n{rc.stderr}", file=sys.stderr)
                return 1
            shutil.rmtree(os.path.join(tmp, ".probes"), ignore_errors=True)
            a = {f for f in os.listdir(args.out) if f.endswith(".json")}
            b = {f for f in os.listdir(tmp) if f.endswith(".json")}
            if a != b:
                print(f"cell sets differ: {a ^ b}", file=sys.stderr)
                return 1
            match, mismatch, errors = filecmp.cmpfiles(args.out, tmp, sorted(a), shallow=False)
            if mismatch or errors:
                print(
                    f"NOT REPRODUCIBLE: {len(mismatch)} of {len(a)} cells differ across runs\n"
                    f"  first few: {mismatch[:5]}\n"
                    "The compiler is nondeterministic; a frozen prediction that cannot be\n"
                    "regenerated is not evidence of anything.",
                    file=sys.stderr,
                )
                return 1
            print(f"   reproducible: {len(match)} cells byte-identical across processes")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
