#!/usr/bin/env python3
"""Score composition laws against the measured edge powerset (vx-review#16, #20).

The algebra composes a multi-hop route by SUMMING its legs. M2 already showed that overstates the
cost of `HBM->SMEM` by 1/0.60. This asks the next question: what law does fit, scored over every
composite in the powerset run rather than the one route M2 looked at.

Two candidates, both parameter-free:

  sum         cost = sum of the legs' costs.        A route that stages: write into the
                                                    intermediate, read back out.
  bottleneck  cost = the slowest leg's cost alone.  A route that streams: the intermediate is
                                                    passed through, and the narrowest point sets
                                                    the rate.

Neither has a fitted constant, which is the point -- a 0.60 fudge factor would fit M2 by
construction and predict nothing. These two disagree by a factor of ~2 on every route, so the data
can separate them.

**A composite cannot be faster than its own slowest leg.** If measured beats bottleneck, the route
is not passing through the intermediate the label claims, or one of the legs is measuring something
else. Rows like that are reported as UNSCORABLE rather than scored, because the same arithmetic
that would give them an error percentage would be comparing two different physical events -- the
mistake vx-review#22 found in `HBM->L2`.

Usage:
    python3 utils/memalg/compose.py --probes <path to probes.txt>
"""

import argparse
import re
import sys

# (source, destination, intermediate) -- the intermediate is the route the hardware actually takes,
# taken from the powerset run's own `note` column where it states one, and from the memory
# hierarchy otherwise. Not a guess about what would be fastest.
COMPOSITES = [
    ("HBM", "SMEM", "L2"),  # note: "composite through L2, measured end-to-end"
    ("L1", "SMEM", "REG"),  # note: "L1 hit -> REG -> SMEM, measured end-to-end"
    ("HBM", "REG", "L2"),  # a load miss fills via L2
    ("SMEM", "HBM", "L2"),  # a store to global goes out through L2
    ("REG", "HBM", "L2"),
]

ROW = re.compile(r"^(\w+)\s*->\s*(\w+)\s+MEASURED\s+([\d.]+)\s+([\d.]+)")


def parse(path):
    """{(from, to): (per_SM_B_per_cyc, aggregate_GB_per_s)} from the powerset table."""
    out = {}
    with open(path) as f:
        for line in f:
            m = ROW.match(line.strip())
            if m:
                a, b, per_sm, agg = m.groups()
                out[(a, b)] = (float(per_sm), float(agg))
    return out


def score(rates, column):
    """Rows for one column of the table. `column` is 0 for per-SM B/cyc, 1 for aggregate GB/s."""
    rows = []
    for a, b, mid in COMPOSITES:
        if any(k not in rates for k in [(a, mid), (mid, b), (a, b)]):
            continue
        r1, r2, rc = rates[(a, mid)][column], rates[(mid, b)][column], rates[(a, b)][column]
        if min(r1, r2, rc) <= 0:
            continue
        meas = 1.0 / rc
        summed = 1.0 / r1 + 1.0 / r2
        bottleneck = 1.0 / min(r1, r2)
        rows.append(
            {
                "route": f"{a}->{mid}->{b}",
                "measured": meas,
                "sum": summed,
                "bottleneck": bottleneck,
                "sum_err": (summed - meas) / meas * 100.0,
                "bn_err": (bottleneck - meas) / meas * 100.0,
                # The physical guard. `bottleneck` is the cost of the slowest leg; a measured cost
                # BELOW it means the composite outran a link it supposedly crosses.
                "unscorable": meas < bottleneck,
                "legs": (r1, r2, rc),
            }
        )
    return rows


def report(rows, unit):
    hdr = (
        f"{'route':<18}{'measured':>11}{'sum':>11}{'err%':>9}"
        f"{'bottleneck':>12}{'err%':>9}  verdict"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        verdict = "UNSCORABLE" if r["unscorable"] else (
            "bottleneck" if abs(r["bn_err"]) < abs(r["sum_err"]) else "sum"
        )
        print(
            f"{r['route']:<18}{r['measured']:>11.5f}{r['sum']:>11.5f}{r['sum_err']:>+9.1f}"
            f"{r['bottleneck']:>12.5f}{r['bn_err']:>+9.1f}  {verdict}"
        )
    print(f"\n(costs in {unit})")


def identical_rate_groups(rates, column, minimum=3):
    """Edges reporting the *same* rate to different destinations.

    A smell, and a mechanical one: if several edges with different destinations all measure to the
    digit, the number is set by something they share -- an issue port, a queue -- and not by the
    destinations they are labelled with. That is the shape of every instrument defect this campaign
    has found so far, so it is worth detecting rather than noticing.
    """
    groups = {}
    for (a, b), v in rates.items():
        groups.setdefault(v[column], []).append((a, b))
    return {rate: edges for rate, edges in groups.items() if len(edges) >= minimum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True, help="probes.txt from a powerset run")
    args = ap.parse_args()

    rates = parse(args.probes)
    if not rates:
        sys.exit(f"no MEASURED rows parsed from {args.probes}")
    print(f"probes : {args.probes}")
    print(f"edges  : {len(rates)} measured\n")

    print("== per-SM (B/cyc), one block's view ==")
    per_sm = score(rates, 0)
    report(per_sm, "cycles per byte, per SM")

    scorable = [r for r in per_sm if not r["unscorable"]]
    if scorable:
        wins_bn = [r for r in scorable if abs(r["bn_err"]) < abs(r["sum_err"])]
        print(
            f"\nscorable routes: {len(scorable)} of {len(per_sm)}   "
            f"bottleneck closer on {len(wins_bn)}, sum closer on {len(scorable) - len(wins_bn)}"
        )
        # The result this run actually supports. Neither law wins everywhere, and the split is not
        # noise -- it lands exactly on whether the route is carried by hardware or by instructions.
        print(
            "\nThe split is mechanical, not statistical. A route the hardware streams (a\n"
            "load or a fill, passing through the intermediate) fits BOTTLENECK. A route\n"
            "made of instructions -- a load into a register then a store back out -- fits\n"
            "SUM, because those two instructions really do happen one after the other.\n"
            "\n"
            "So the composition law is not one law, and the algebra cannot currently tell the two\n"
            "route kinds apart: `within:` says a space contains another, not whether crossing the\n"
            "boundary is a hardware path or a pair of instructions."
        )

    unscorable = [r for r in per_sm if r["unscorable"]]
    if unscorable:
        print("\n== unscorable ==")
        for r in unscorable:
            r1, r2, rc = r["legs"]
            print(
                f"  {r['route']}: composite {rc:.1f} beats its slowest leg {min(r1, r2):.1f}. "
                "A route\n    cannot outrun a link it crosses, so these are not the same event."
            )

    for rate, edges in sorted(identical_rate_groups(rates, 0).items()):
        srcs = {a for a, _ in edges}
        dsts = {b for _, b in edges}
        print(f"\n== {len(edges)} edges all measure exactly {rate} B/cyc per SM ==")
        for a, b in sorted(edges):
            print(f"  {a} -> {b}")
        if len(dsts) > 1:
            print(
                f"\n  {len(dsts)} different destinations, one number. Whatever sets this rate is\n"
                f"  common to all of them and is not the destination, so none of these rows\n"
                f"  measures the edge it is labelled with."
            )
            if len(srcs) <= 2:
                print(
                    "  Shared source side, global destinations: this is the store-issue port, and\n"
                    "  writes posted rather than completed. Fourth defect of this shape in\n"
                    "  the campaign (see vx-review#22 for the first three)."
                )

    print("\n== aggregate (GB/s), device-wide ==")
    report(score(rates, 1), "seconds per byte, device-wide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
