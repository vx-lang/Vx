#!/usr/bin/env python3
"""Per-seam solver cost as a curve in the number of seams.

The 52-115 us/seam figure this replaces was a point measurement taken inside the
solver, timing one push/check-sat/pop round trip on one obligation. That is not
what a compilation pays -- it excludes building the obligation and it cannot show
whether the cost stays flat as seams accumulate.

Two numbers are reported per point, and they answer different questions:

  us/seam (wall)   whole-compile delta between `--verify-seams` and an ordinary
                   compile of the same program, divided by seams. This is what a
                   user pays: obligation construction, solver round trip, verdict
                   handling. `--verify-seams` is off by default, so the baseline
                   is a plain compile and the delta is the entire opt-in cost.
  us/seam (solver) the compiler's own `[seam]` accounting -- marginal solving time
                   only, with the one-time process spawn reported separately.

Neither column answers "is it flat" on its own. A fixed per-compilation cost divided
by N falls like 1/N whatever the marginal cost is doing, so a per-seam figure decays
even when nothing scales. The script fits `delta = fixed + marginal * seams` and reports
both terms: a marginal that holds across the sweep is the flat result.

A seam is one *hop*, not one transfer: `stage_multi_hop` rewrites a k-hop route
into k single-hop transfers and each discharges its own obligation. This generator
varies transfers and reads the resulting hop count out of the `[seam]` line rather
than assuming the two are equal -- for the route used here they are not.

Two guards, because a scaling curve of nothing would look like a very good result:

  * the obligation count must be non-zero and must grow with N. An earlier draft of
    this generator used a relaxed transfer into a `managed: cached` space, which
    `run_seam_hop` short-circuits before the solver -- it compiled clean under
    `--verify-seams` and measured no obligations at all.
  * no obligation may go undischarged. With no solver the counter still reports N
    obligations and the solving time collapses toward zero, which is precisely the
    misleadingly fast number this measurement must not produce.

Usage:
    python3 scripts/seam_scaling.py                     # default sweep to 1000
    python3 scripts/seam_scaling.py --max 100 --reps 3  # quick
"""

import argparse
import json
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

# [seam] 6 obligation(s): solver init 0.456 ms (once) + 9.530 ms solving total = 1.5883 ms/seam
SEAM_LINE = re.compile(
    r"\[seam\] (\d+) obligation\(s\): solver init ([\d.]+) ms \(once\) \+ "
    r"([\d.]+) ms solving total"
)


def seam_program(n: int) -> str:
    """N transfers into an `explicit` space: a real seam each, every verdict an accept.

    `managed: explicit` is what makes these obligations reach the solver at all --
    hardware coherence would discharge them for free. Accepts (rather than the
    rejects the E6004 tests use) keep this a measurement of proving, not reporting.
    """
    body = "\n".join(
        f"  let _t{i} : Tensor<f32, []> = {1.0 + i};\n"
        f"  let _s{i} = transfer(_t{i}, Memory::Local_SRAM);"
        for i in range(n)
    )
    return (
        "Memory Local_SRAM {\n  managed: explicit\n}\n\n"
        f"fn main() -> i32 {{\n{body}\n  return 0;\n}}\n"
    )


def control_program(n: int) -> str:
    """The same shape with the transfers removed: N statements, zero seams.

    This is what makes the delta attributable. Any per-statement work `--verify-seams`
    happens to switch on would show up here too.
    """
    body = "\n".join(
        f"  let _t{i} : Tensor<f32, []> = {1.0 + i};\n"
        f"  let _s{i} : Tensor<f32, []> = _t{i};"
        for i in range(n)
    )
    return (
        "Memory Local_SRAM {\n  managed: explicit\n}\n\n"
        f"fn main() -> i32 {{\n{body}\n  return 0;\n}}\n"
    )


def compile_once(vxc: str, path: pathlib.Path, verify: bool):
    """One compile; returns (wall seconds, stderr). Stops at MLIR: the seam check runs
    in the type checker, and going on to JIT would add codegen noise to both arms."""
    cmd = [vxc, str(path), "--action", "emit-mlir"]
    if verify:
        cmd.append("--verify-seams")
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if r.returncode != 0:
        sys.exit(f"compile failed (verify={verify}) for {path}:\n{r.stderr[-2000:]}")
    return dt, r.stderr


def seam_stats(stderr: str):
    """(obligations, init_ms, solve_ms) from the compiler's own accounting, or None."""
    m = SEAM_LINE.search(stderr)
    if not m:
        return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def measure(vxc: str, src: str, path: pathlib.Path, reps: int):
    path.write_text(src)
    # One warm compile per arm: the first run of a new source pays page-cache costs
    # that have nothing to do with seams.
    compile_once(vxc, path, False)
    _, warm_err = compile_once(vxc, path, True)

    if "could not be discharged" in warm_err:
        sys.exit(
            f"{path}: an obligation went undischarged -- the solving time below would be "
            "the time to give up, not to prove. Check that z3 is on PATH."
        )

    base = statistics.median(compile_once(vxc, path, False)[0] for _ in range(reps))
    vers, stats = [], []
    for _ in range(reps):
        dt, err = compile_once(vxc, path, True)
        vers.append(dt)
        s = seam_stats(err)
        if s:
            stats.append(s)
    ver = statistics.median(vers)
    obligations = stats[0][0] if stats else 0
    init_ms = statistics.median(s[1] for s in stats) if stats else 0.0
    solve_ms = statistics.median(s[2] for s in stats) if stats else 0.0
    return base, ver, obligations, init_ms, solve_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vxc", default="./target/debug/vxc")
    ap.add_argument("--max", type=int, default=1000, help="largest transfer count")
    ap.add_argument("--reps", type=int, default=5, help="compiles per arm per point")
    ap.add_argument("--json", type=pathlib.Path, help="write raw measurements here")
    args = ap.parse_args()

    z3 = shutil.which("z3")
    if z3 is None:
        sys.exit("no z3 on PATH -- a solver-cost measurement needs the solver")
    ver = subprocess.run([z3, "--version"], capture_output=True, text=True).stdout.strip()

    points = [n for n in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000) if n <= args.max]
    print(f"solver: {ver}")
    print(f"vxc:    {args.vxc}")
    print(f"reps:   {args.reps} per arm per point (median reported)\n")
    print(
        f"{'xfers':>6} {'seams':>6} {'base ms':>9} {'verify ms':>9} {'delta ms':>9} "
        f"{'us/seam':>8} {'solver us':>10} {'init ms':>8} {'ctrl ms':>8}"
    )
    print("-" * 82)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        for n in points:
            base, verify, obl, init_ms, solve_ms = measure(
                args.vxc, seam_program(n), d / f"seam_{n}.vx", args.reps
            )
            if obl == 0:
                sys.exit(
                    f"N={n}: the compiler reports zero obligations -- this program has no "
                    "seams and the delta below would measure nothing."
                )
            cbase, cverify, cobl, _, _ = measure(
                args.vxc, control_program(n), d / f"ctrl_{n}.vx", args.reps
            )
            if cobl != 0:
                sys.exit(f"N={n}: the control raised {cobl} obligations; it is not a control.")

            delta = verify - base
            rows.append(
                {
                    "transfers": n,
                    "seams": obl,
                    "baseline_ms": base * 1e3,
                    "verify_ms": verify * 1e3,
                    "delta_ms": delta * 1e3,
                    "us_per_seam_wall": delta / obl * 1e6,
                    "us_per_seam_solver": solve_ms / obl * 1e3,
                    "solver_init_ms": init_ms,
                    "control_delta_ms": (cverify - cbase) * 1e3,
                }
            )
            r = rows[-1]
            print(
                f"{n:>6} {obl:>6} {r['baseline_ms']:>9.1f} {r['verify_ms']:>9.1f} "
                f"{r['delta_ms']:>9.1f} {r['us_per_seam_wall']:>8.1f} "
                f"{r['us_per_seam_solver']:>10.1f} {init_ms:>8.3f} "
                f"{r['control_delta_ms']:>8.1f}"
            )

    # Flat or not is the whole question, and a per-seam column cannot answer it: a fixed
    # cost divided by N falls like 1/N no matter what the marginal cost does, so per-seam
    # figures decay even when nothing is scaling. Fitting `delta = fixed + marginal * seams`
    # separates the two. A marginal that is constant across the sweep is the flat result;
    # a curve that bends shows up as a fit that does not hold.
    def fit(xs, ys):
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        return a, b, (1 - ss_res / ss_tot) if ss_tot else float("nan")

    seams = [r["seams"] for r in rows]
    if len(seams) >= 3:
        wall = [r["delta_ms"] for r in rows]
        solve = [r["us_per_seam_solver"] * r["seams"] / 1e3 for r in rows]
        aw, bw, r2w = fit(seams, wall)
        as_, bs, r2s = fit(seams, solve)
        print(
            f"\nwhole compile:  delta_ms = {aw:.3f} + {bw*1e3:.1f} us/seam x seams"
            f"   R^2={r2w:.5f}"
        )
        print(
            f"solver only:    total_ms = {as_:.3f} + {bs*1e3:.1f} us/seam x seams"
            f"   R^2={r2s:.5f}"
        )
        print("\nresiduals (measured - predicted, whole compile):")
        for x, y in zip(seams, wall):
            print(f"  {x:>5} seams  {y:>8.1f} ms  {y - (aw + bw * x):>+7.2f} ms")
        init = statistics.median(r["solver_init_ms"] for r in rows)
        print(
            f"\nmarginal per seam: {bw*1e3:.1f} us, of which {bs*1e3:.1f} us is the solver"
        )
        print(
            f"fixed per compile: {aw:.2f} ms = {init:.2f} spawn+preamble + "
            f"{as_:.2f} first-check warmup + {aw - as_ - init:.2f} elsewhere"
        )
        if r2w > 0.99:
            print(
                "\nflat: one marginal cost fits every point, so the per-seam figure does not "
                "drift with N. The decay in the us/seam column above is the fixed cost being "
                "amortized, not the marginal cost falling."
            )
        else:
            print(
                f"\nNOT flat: a constant marginal cost does not fit (R^2={r2w:.4f}) -- "
                "per-seam cost depends on how many seams there are."
            )

    worst = max(abs(r["control_delta_ms"]) for r in rows)
    print(f"\ncontrol: largest |delta| on a program with no seams: {worst:.1f} ms")

    if args.json:
        args.json.write_text(json.dumps({"z3": ver, "rows": rows}, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
