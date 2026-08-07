#!/usr/bin/env python3
"""Predict -> measure -> compare, end to end (vx-review#13).

The Tier-0 dry run. Its output is an error table for one memory link on the machine we already
own, and its purpose is to find bugs in OUR harness -- a wrong formula, a mislabeled column, a
unit mismatch -- before any GPU time is paid for. The MLSys admission matrix was dry-run the same
way and it caught an 8x arithmetic error before any money was spent.

Three steps, each of which can fail loudly:

  predict  generate a Vx program that transfers a tile of each swept size, compile it against the
           machine file, and harvest `--diagnostics-json` -> predicted picoseconds per hop.
  measure  run the C benchmark -> achieved copy time per size.
  compare  join on byte count, report signed relative error.

The join is on `bytes`, which is why the extractor had to emit it (vx-review#12): without a byte
count in the prediction record there is nothing to join on and the two halves cannot be related at
all.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

# The tile shapes swept. Square f32 tiles, so bytes = 4*d*d; chosen so the byte counts line up
# with the measurement's powers of two.
DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]


def tile_bytes(d):
    return 4 * d * d


def predict(vxc, machine, workdir):
    """Compile one probe program per size and harvest its predicted hop costs.

    One program per size rather than one program with every size: the resident-set check would
    reject a program holding every tile at once, and a rejected compile emits no routes.
    """
    out = {}
    for d in DIMS:
        src = os.path.join(workdir, f"probe_{d}.vx")
        with open(src, "w") as f:
            f.write(
                "fn main() -> i32 {\n"
                f"  let tile : Tensor<f32, [{d}, {d}]> = Tensor<f32>([{d}, {d}]);\n"
                "  let _h = transfer(tile, Memory::HBM);\n"
                "  return 0;\n"
                "}\n"
            )
        js = os.path.join(workdir, f"probe_{d}.json")
        proc = subprocess.run(
            [vxc, "--machine", machine, src, "--diagnostics-json", js, "--emit-mlir"],
            capture_output=True,
            text=True,
        )
        if not os.path.exists(js):
            print(f"  !! no diagnostics for {d}x{d}: {proc.stderr.strip()[:200]}", file=sys.stderr)
            continue
        rec = json.load(open(js))
        for r in rec.get("routes", []):
            if r["path"] == ["CPU_DRAM", "HBM"] and r.get("derived_cost") is not None:
                if r.get("derived_unit") != "ps":
                    # A cycle-denominated cost cannot be compared against a wall-clock measurement
                    # without a declared clock. Refusing beats silently treating cycles as ps.
                    print(
                        f"  !! {d}x{d}: predicted unit is {r['derived_unit']}, not ps -- "
                        "cannot compare against wall-clock without a clock figure",
                        file=sys.stderr,
                    )
                    continue
                out[r["bytes"]] = r["derived_cost"]
    return out


def measure(binary):
    proc = subprocess.run([binary], capture_output=True, text=True, check=True)
    rows = {}
    for row in csv.DictReader(proc.stdout.splitlines()):
        rows[int(row["bytes"])] = {
            "ns": float(row["median_ns"]),
            "q1": float(row["q1_ns"]),
            "q3": float(row["q3_ns"]),
            "copy_gbps": float(row["copy_GBps"]),
            "traffic_gbps": float(row["traffic_GBps"]),
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vxc", default="target/release/vxc")
    ap.add_argument("--machine", default="fleet/m4-uma.vx")
    ap.add_argument("--bench", default="utils/memalg/measure_link")
    ap.add_argument("--out", default=None, help="write the error table here as CSV")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as workdir:
        print("== predict ==", file=sys.stderr)
        pred = predict(args.vxc, args.machine, workdir)
        print(f"   {len(pred)} predictions harvested", file=sys.stderr)

        print("== measure ==", file=sys.stderr)
        meas = measure(args.bench)
        print(f"   {len(meas)} sizes measured", file=sys.stderr)

    common = sorted(set(pred) & set(meas))
    if not common:
        print(
            "FATAL: no byte count appears in both halves, so nothing can be compared.\n"
            f"  predicted sizes: {sorted(pred)}\n"
            f"  measured sizes:  {sorted(meas)}",
            file=sys.stderr,
        )
        return 1

    rows = []
    print()
    print(f"{'bytes':>12} {'pred_ns':>10} {'meas_ns':>10} {'err_%':>9} "
          f"{'copy_GB/s':>10} {'traffic_GB/s':>13}")
    print("-" * 70)
    for b in common:
        pred_ns = pred[b] / 1000.0  # ps -> ns
        m = meas[b]
        err = (pred_ns - m["ns"]) / m["ns"] * 100.0
        rows.append(
            {
                "bytes": b,
                "predicted_ns": round(pred_ns, 1),
                "measured_ns": round(m["ns"], 1),
                "rel_err_pct": round(err, 1),
                "measured_copy_GBps": round(m["copy_gbps"], 2),
                "measured_traffic_GBps": round(m["traffic_gbps"], 2),
            }
        )
        print(f"{b:>12} {pred_ns:>10.1f} {m['ns']:>10.1f} {err:>+9.1f} "
              f"{m['copy_gbps']:>10.2f} {m['traffic_gbps']:>13.2f}")

    # The headline the dry run exists to produce. A bandwidth-only model is pre-registered to
    # under-predict cost at small sizes (a latency floor it does not model), so a large negative
    # error at the small end is the EXPECTED result, not a harness bug.
    print()
    small = [r for r in rows if r["bytes"] <= 64 * 1024]
    large = [r for r in rows if r["bytes"] >= 16 * 1024 * 1024]
    if small:
        avg = sum(r["rel_err_pct"] for r in small) / len(small)
        print(f"small sizes (<=64 KiB): mean error {avg:+.1f}%")
    if large:
        avg = sum(r["rel_err_pct"] for r in large) / len(large)
        print(f"large sizes (>=16 MiB): mean error {avg:+.1f}%")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nerror table -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
