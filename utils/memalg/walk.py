#!/usr/bin/env python3
"""M6: one full walk down the hierarchy, predicted per hop.

M1 prices one seam at a time against a log-spaced sweep of square tiles. This prices a whole walk
-- CPU_DRAM -> HBM -> L2 -> SMEM -- for the tiles a real attention kernel actually moves, and
splits the predicted total per hop so that when the measured total disagrees we can say WHICH edge
was wrong rather than "the walk was off by 40%".

The predicted column needs no hardware and is what this script produces. The measured column needs
a pod; `--measured` joins one in when it exists.

**Tile shapes are not invented.** They are the FlashAttention-2 configurations (Dao 2023, sec. 3.1
and Table 1): query block Br and key block Bc in {64, 128}, head dimension d in {64, 128}. The
working set of one such block is Q[Br,d] + K[Bc,d] + V[Bc,d], which is what a placement into SMEM
has to hold at once, and it is the reason this walk is worth pricing rather than a single tile: at
d=128 the three tiles together are 192 KiB against SMEM's declared 228 KiB, so the walk sits near
the admission boundary where the model's arithmetic actually matters.

Everything is f32 because the frozen predictions and every fleet `bandwidth:` figure are in bytes
and the algebra has no dtype. Real FA2 runs in bf16, so a production working set is half of what
is priced here -- the ratios between hops are unaffected, the absolute costs are 2x pessimistic,
and this is stated rather than silently corrected because halving them would be a number chosen
after the fact.

**Where the walk stops.** TMEM is not declared on any fleet SKU -- fleet/b200.vx says so and gives
the reason: nothing places into it yet, and a declared-but-unused space would appear in the
hierarchy without any placement exercising it. So the honest end of the walk is SMEM. M6 allows
exactly this ("or state the boundary and stop"), and TMEM only accepts data through tensor-core
instructions anyway, so measuring it means measuring a matmul rather than a transfer.

Usage:
    python3 utils/memalg/walk.py --machine fleet/h100-sxm.vx
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# (name, Br, Bc, d) -- FlashAttention-2 block configurations.
WALKS = [
    ("fa2-64x64-d64", 64, 64, 64),
    ("fa2-128x64-d64", 128, 64, 64),
    ("fa2-128x128-d64", 128, 128, 64),
    ("fa2-64x64-d128", 64, 64, 128),
    ("fa2-128x64-d128", 128, 64, 128),
    ("fa2-128x128-d128", 128, 128, 128),
]

# The staging chain. Each name is a Memory space every discrete fleet SKU declares.
CHAIN = ["HBM", "L2", "SMEM"]


def walk_source(br, bc, d):
    """Q, K and V staged together down the chain -- the working set, not one tile at a time.

    Priced as one walk on purpose. Capacity is checked against the *resident set* of a space, so
    three tiles arriving separately and three tiles arriving together are different questions, and
    the one a kernel asks is the second.
    """
    lines = ["fn main() -> i32 {"]
    for nm, rows in (("q", br), ("k", bc), ("v", bc)):
        lines.append(f"  let {nm} : Tensor<f32, [{rows}, {d}]> = Tensor<f32>([{rows}, {d}]);")
    prev = {"q": "q", "k": "k", "v": "v"}
    for hop, space in enumerate(CHAIN):
        for nm in ("q", "k", "v"):
            cur = f"{nm}{hop}"
            lines.append(f"  let {cur} = transfer({prev[nm]}, Memory::{space});")
            prev[nm] = cur
    lines.append(f"  let _a = {prev['q']}; let _b = {prev['k']}; let _c = {prev['v']};")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines) + "\n"


def load_measured(path):
    """The instrument's walk rows, keyed (hop, bytes), plus the measured SM clock.

    The `walk/L2->SMEM` rows are in CYCLES (the instrument's native unit for that seam) while the
    prediction is picoseconds. The conversion uses `device/SM_clock` FROM THE SAME CSV -- a
    measured number travelling with the measurement, never an assumed one, which is what protocol
    decision 5 requires. No clock row, no conversion: the hop reports unscored.
    """
    import csv

    rows = {}
    clock_hz = None
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["seam"] == "device/SM_clock" and row["median"]:
                clock_hz = float(row["median"])
            if not row["seam"].startswith("walk/") or not row["median"]:
                continue
            hop = row["seam"][len("walk/") :]
            rows[(hop, int(row["bytes"]))] = {
                "median": float(row["median"]),
                "unit": row["unit"],
            }
    return {"rows": rows, "clock_hz": clock_hz}


def predict(vxc, machine, br, bc, d, workdir):
    src = os.path.join(workdir, f"walk_{br}_{bc}_{d}.vx")
    with open(src, "w") as f:
        f.write(walk_source(br, bc, d))
    js = os.path.join(workdir, f"walk_{br}_{bc}_{d}.json")
    proc = subprocess.run(
        [vxc, "--machine", machine, "--host", "default", src,
         "--diagnostics-json", js, "--emit-mlir", "-o", os.devnull],
        capture_output=True, text=True,
    )
    if not os.path.exists(js):
        return None, (proc.stderr or proc.stdout).strip()[:300]
    return json.load(open(js)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vxc", default="target/release/vxc")
    ap.add_argument("--machine", default="fleet/h100-sxm.vx")
    ap.add_argument("--measured", help="measure_device CSV; fills in the measured column")
    args = ap.parse_args()

    measured = load_measured(args.measured) if args.measured else None

    workdir = tempfile.mkdtemp(prefix="m6-")
    print(f"machine : {args.machine}")
    print(f"walk    : CPU_DRAM -> {' -> '.join(CHAIN)}   (TMEM undeclared; see the header)")
    print(f"tiles   : Q[Br,d] + K[Bc,d] + V[Bc,d], f32\n")

    hdr = (
        f"{'walk':<20}{'set B':>9}{'CPU->HBM':>13}{'HBM->L2':>13}"
        f"{'L2->SMEM':>13}{'total ps':>14}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))

    rejected = []
    shares = []
    for name, br, bc, d in WALKS:
        rec, err = predict(args.vxc, args.machine, br, bc, d, workdir)
        if err:
            print(f"{name:<20} harvest error: {err}", file=sys.stderr)
            continue
        verdict = rec.get("verdict")
        setb = (br * d + bc * d + bc * d) * 4

        # Per-hop totals, keyed by the hop rather than by position: a route list that grows a hop
        # would otherwise shift every column silently.
        per_hop = {}
        for r in rec.get("routes", []):
            if r.get("derived_cost") is None:
                continue
            per_hop[tuple(r["path"])] = per_hop.get(tuple(r["path"]), 0) + r["derived_cost"]

        cols = []
        for a, b in [("CPU_DRAM", "HBM"), ("HBM", "L2"), ("L2", "SMEM")]:
            cols.append(per_hop.get((a, b)))
        total = sum(c for c in cols if c is not None)
        cells = "".join(f"{c:>13,}" if c is not None else f"{'-':>13}" for c in cols)
        print(f"{name:<20}{setb:>9,}{cells}{total:>14,}  {verdict}")

        # The measured column, when a CSV was supplied: per hop, joined on exact bytes, with the
        # signed error the campaign always reports. The instrument measures the WORKING SET per
        # hop (all three tiles arrive together), so the join is on `setb`.
        if measured is not None:
            hop_map = [
                ("CPU_DRAM->HBM", "CPU_DRAM->HBM", "ps"),
                ("HBM->L2", "HBM->L2_fill", "ps"),
                ("L2->SMEM", "L2->SMEM", "cyc"),
            ]
            parts = []
            for (pred_hop, meas_hop, unit), pred_ps in zip(hop_map, cols):
                m = measured["rows"].get((meas_hop, setb))
                if pred_ps is None or m is None:
                    parts.append(f"{meas_hop}: unmeasured")
                    continue
                if unit == "cyc":
                    if not measured["clock_hz"]:
                        parts.append(f"{meas_hop}: no device/SM_clock row, unscored")
                        continue
                    meas_ps = m["median"] / measured["clock_hz"] * 1e12
                else:
                    meas_ps = m["median"]
                err = (pred_ps - meas_ps) / meas_ps * 100.0
                parts.append(f"{meas_hop}: {err:+.1f}%")
            print(f"{'':<20}{'measured:':>9} " + "   ".join(parts))
        if verdict == "rejected":
            rejected.append((name, setb, rec))
        if total and all(c is not None for c in cols):
            shares.append([c / total for c in cols])

    # The per-hop split is the point of the walk, so the shares are computed from the run rather
    # than asserted. A prior belief about which hop dominates is exactly what a walk exists to
    # check, and on this hierarchy the obvious guess -- "the host link dominates a discrete part"
    # -- turns out to be wrong.
    if shares:
        avg = [sum(s[i] for s in shares) / len(shares) for i in range(3)]
        names = ["CPU_DRAM->HBM", "HBM->L2", "L2->SMEM"]
        print("\nshare of the predicted walk, averaged over the configurations above:")
        for nm, a in zip(names, avg):
            print(f"  {nm:<16}{a * 100:>6.1f}%")
        quiet = [nm for nm, a in zip(names, avg) if a < 0.05]
        if quiet:
            print(
                f"\nRead the split, not the total. {' and '.join(quiet)} is under 5% of the walk, "
                "so the\nmodel could be wrong about it by a factor of two and the last "
                "column would\nbarely move. A walk-level residual is not evidence about "
                "an edge that small."
            )

    # The implied rate per hop, back-computed from what the model charged. This is where the walk
    # earns its keep: the two dominant hops turn out to run at almost the same rate, which is not
    # something either seam says on its own.
    #
    # It is a scope mismatch, not a coincidence. `CPU_DRAM->HBM` is a whole-device figure -- the
    # host link, all of it. `L2->SMEM` is a per-SM figure, because SMEM is `scope: sm` and the
    # containment rule divides L2's device-wide bandwidth by `replicas:`. So the walk adds a number
    # describing one block's slice to a number describing the entire part, and calls the result the
    # cost of a walk. A real attention kernel runs hundreds of blocks at once, and its on-die hop is
    # therefore ~`replicas:` times cheaper relative to the host hop than this total implies.
    #
    # Which of the two the walk *should* report depends on a question the algebra does not ask:
    # whether a placement describes one block's working set or the device's. Capacity is checked
    # per block (see the SMEM boundary note below); cost is summed as though it were per device.
    if shares:
        rep = WALKS[-1]
        setb = (rep[1] * rep[3] + 2 * rep[2] * rep[3]) * 4
        rec, err = predict(args.vxc, args.machine, rep[1], rep[2], rep[3], workdir)
        if not err:
            per_hop = {}
            for r in rec.get("routes", []):
                if r.get("derived_cost") is not None:
                    k = tuple(r["path"])
                    per_hop[k] = per_hop.get(k, 0) + r["derived_cost"]
            print(f"\nimplied rate per hop ({rep[0]}, {setb:,} B moved):")
            for a, b in [("CPU_DRAM", "HBM"), ("HBM", "L2"), ("L2", "SMEM")]:
                ps = per_hop.get((a, b))
                if ps:
                    print(f"  {a + '->' + b:<16}{setb / ps * 1e12 / 1e9:>9.1f} GB/s")
            print(
                "\nThe host hop and L2->SMEM come out at nearly the same rate, and that is\n"
                "a scope mismatch rather than a fact about the hardware: CPU_DRAM->HBM is\n"
                "the whole link, while L2->SMEM is divided by `replicas:` because SMEM is\n"
                "`scope: sm`. The walk adds one block's slice to the entire part's.\n"
                "Capacity is checked per block; cost is summed as though it were per device."
            )

    for name, setb, rec in rejected:
        sets = {s["space"]: s for s in rec.get("resident_sets", [])}
        sm = sets.get("SMEM")
        if sm:
            print(
                f"\n{name}: working set {setb:,} B rejected -- SMEM holds {sm['total_bytes']:,} B "
                f"against {sm['capacity_bytes']:,} B ({sm['utilization']:.2f}x)"
            )

    # A prediction M4's instrument settles, recorded here because this is the walk that lands on
    # it. `capacity:` is checked against a space's resident set, and on a discrete SKU that set
    # lives in ONE BLOCK. The fleet files declare SMEM at the per-SM figure (228 KiB on H100,
    # 164 KiB on A100), but a block cannot opt into all of it -- the driver reserves one granule,
    # so the per-block cap is 227 KiB and 163 KiB. The model therefore admits a working set the
    # hardware refuses, over a window exactly one granule wide:
    #
    #     [227,256] f32 = 232,448 B  model admits, hardware admits
    #     [228,256] f32 = 233,472 B  model admits, hardware REFUSES   <- the window
    #     [229,256] f32 = 234,496 B  model refuses, hardware refuses
    #
    # Verified against fleet/h100-sxm.vx on 2026-08-12 for the model's side. The hardware's side is
    # `device/SMEM_per_block_optin` from measure_device, which is why that row emits
    # both SMEM figures rather than the one the machine files quote. Until it runs this is a
    # prediction from a recalled CUDA constant, not a measurement, and is written down now so that
    # it is on the record before the number that settles it is read.
    print(
        "\nPre-registered, settled by device/SMEM_per_block_optin: the fleet files\n"
        "declare SMEM at the per-SM figure, but `capacity:` is checked against a resident\n"
        "set that lives in one block, and a block cannot opt into the last granule. If that\n"
        "row comes back "
        "below the declared capacity, every SKU admits a one-granule window the hardware refuses."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
