#!/usr/bin/env python3
"""M1: join measured seams against the FROZEN predictions and emit the error table (vx-review#15).

Reads predictions only from the frozen directory, never by re-running the compiler. That is the
point of the freeze: if this script could regenerate them it could also regenerate them *after*
seeing the data, and the pre-registration would be decorative.

Scoring follows PREDICTIONS.md exactly:
  * signed relative error `(predicted - measured) / measured`; negative = model under-predicts cost
  * cycle-denominated cells are compared in cycles, never converted to wall-clock
  * pinned host transfers are the headline; pageable is reported as the positive control
"""

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict


def load_predictions(pred_dir, sku):
    """{(seam, bytes): (cost, unit, source)} from the frozen cells for one SKU."""
    out = {}
    pat = os.path.join(pred_dir, f"{sku}__*.json")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no frozen predictions matching {pat}")
    for f in files:
        d = json.load(open(f))
        for r in d["prediction"].get("routes", []):
            if r.get("derived_cost") is None:
                continue
            key = ("->".join(r["path"]), r["bytes"])
            val = (r["derived_cost"], r["derived_unit"], r.get("cost_source"))
            if key in out and out[key] != val:
                sys.exit(f"frozen predictions disagree for {key}: {out[key]} vs {val}")
            out[key] = val
    return out


def load_measurements(path):
    """({(seam, bytes, note): median}, [fact rows]) — note distinguishes pinned from pageable.

    `device/*` rows are not seams. They are the machine file's own declared numbers read back off
    the hardware (vx-review#18) and they join against no prediction, so they are split out here
    rather than left to fall into the "no frozen prediction" list: there are ~15 of them, that
    list prints at most 8, and they would push out the one thing it exists to show -- a real seam
    that was measured and never predicted.
    """
    out = {}
    facts = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get("median"):
                continue  # a skipped cell (capacity), carried through as absent
            if row["seam"].startswith("device/"):
                facts.append(row)
                continue
            # walk/* rows are the M6 measured column (utils/memalg/walk.py --measured joins them);
            # they have no frozen per-seam cell and would otherwise crowd the unmatched list.
            if row["seam"].startswith("walk/"):
                continue
            out[(row["seam"], int(row["bytes"]), row.get("note", ""))] = {
                "median": float(row["median"]),
                "q1": float(row["q1"]) if row.get("q1") else None,
                "q3": float(row["q3"]) if row.get("q3") else None,
                "unit": row["unit"],
                "rate": float(row["derived_rate_GBps"]) if row.get("derived_rate_GBps") else None,
            }
    return out, facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="the FROZEN predictions directory")
    ap.add_argument("--measured", required=True, help="measure_device CSV")
    ap.add_argument("--sku", default="h100-sxm")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pred = load_predictions(args.predictions, args.sku)
    meas, facts = load_measurements(args.measured)

    rows = []
    unmatched_pred, unmatched_meas = [], []

    for (seam, size, note), m in sorted(meas.items()):
        key = (seam, size)
        if key not in pred:
            unmatched_meas.append((seam, size, note))
            continue
        cost, unit, source = pred[key]
        if unit != m["unit"]:
            # Protocol decision 5. A cycles-vs-ps mismatch is not something to paper over with a
            # clock figure invented here.
            print(
                f"  !! {seam} {size}B: predicted in {unit}, measured in {m['unit']} — skipped",
                file=sys.stderr,
            )
            continue
        err = (cost - m["median"]) / m["median"] * 100.0
        rows.append(
            {
                "seam": seam,
                "bytes": size,
                "variant": note,
                "unit": unit,
                "cost_source": source,
                "predicted": cost,
                "measured": round(m["median"], 1),
                "rel_err_pct": round(err, 1),
                "measured_GBps": m["rate"],
            }
        )

    matched = {(r["seam"], r["bytes"]) for r in rows}
    unmatched_pred = [k for k in pred if k not in matched]

    if not rows:
        sys.exit("no cell matched between the frozen predictions and the measurements")

    hdr = (
        f"{'seam':<16}{'bytes':>12}  {'variant':<14}{'pred':>14}{'meas':>14}"
        f"{'err_%':>9}{'GB/s':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        gb = f"{r['measured_GBps']:.1f}" if r["measured_GBps"] else "-"
        print(
            f"{r['seam']:<16}{r['bytes']:>12}  {r['variant']:<14}"
            f"{r['predicted']:>14}{r['measured']:>14.0f}{r['rel_err_pct']:>+9.1f}{gb:>10}"
        )

    # Per-seam summary. The shape of the residual is the result, not any single cell.
    print()
    by_seam = defaultdict(list)
    for r in rows:
        # Variants counted in the per-seam summary. A variant missing from this list silently
        # drops its whole seam from the summary -- renaming the L2 note to l2_read_l1_bypassed did
        # exactly that on the first H100 run, and the seam just stopped being mentioned. Anything
        # measured that is not an explicitly excluded control belongs here.
        if r["variant"] not in ("pageable",):
            by_seam[r["seam"]].append(r["rel_err_pct"])
    for seam, errs in sorted(by_seam.items()):
        lo, hi = min(errs), max(errs)
        mean = sum(errs) / len(errs)
        spread = hi - lo
        shape = "FLAT (constant factor)" if spread < 10 else "SIZE-DEPENDENT"
        print(f"{seam:<16} n={len(errs):<3} mean {mean:+7.1f}%   range {lo:+.1f}..{hi:+.1f}%   {shape}")

    # The positive control from the instrument: pinned and pageable must differ, or the harness is
    # not resolving what it claims to.
    pin = {r["bytes"]: r["measured"] for r in rows if r["variant"] == "pinned"}
    page = {r["bytes"]: r["measured"] for r in rows if r["variant"] == "pageable"}
    both = sorted(set(pin) & set(page))
    print()
    ratios = [page[b] / pin[b] for b in both if pin[b] > 0]
    if ratios:
        worst = max(ratios)
        if worst < 1.05:
            print(
                f"!! CONTROL FAILED: pageable is at most {worst:.2f}x pinned. These paths "
                "differ on real hardware; if the instrument cannot tell them apart it is not "
                "resolving transfer time. Do not cite this run."
            )
        else:
            print(f"control OK: pageable/pinned up to {worst:.2f}x — the instrument resolves the path")
    else:
        # Silence would read as success. It is not: a missing control is an unrun control, and the
        # usual cause is that every pinned allocation was refused (locked-memory limit in a
        # container), which also removes the headline measurement.
        print(
            f"!! CONTROL NOT EVALUATED: {len(pin)} pinned and {len(page)} pageable cells, "
            f"{len(both)} at a common size. The control did not run, so this run carries no "
            "evidence that the instrument resolves transfer time. Check measured.log for SKIP "
            "lines before citing anything."
        )

    # M4 (vx-review#18): the declared numbers, read off the hardware. Nothing is scored here --
    # the frozen cells carry costs, not capacities, so there is no predicted side to subtract.
    # This is the table someone reads while turning a `spec:` line into a `measured:` one, and it
    # is printed rather than summarised because the transcription is the fragile step.
    if facts:
        print(f"\ndeclared numbers, read off the hardware ({len(facts)}):")
        for r in facts:
            name = r["seam"].split("/", 1)[1]
            size = f" @ {r['bytes']}B" if r["bytes"] and int(r["bytes"]) else ""
            print(f"  {name + size:<34}{float(r['median']):>20,.0f} {r['unit']:<6} {r['note']}")

    if unmatched_pred:
        print(f"\n{len(unmatched_pred)} frozen cell(s) had no measurement:")
        for seam, size in sorted(unmatched_pred)[:8]:
            print(f"  {seam} {size}B")
    if unmatched_meas:
        print(f"\n{len(unmatched_meas)} measurement(s) had no frozen prediction:")
        for seam, size, note in unmatched_meas[:8]:
            print(f"  {seam} {size}B ({note})")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nerror table -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
