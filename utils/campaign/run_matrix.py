#!/usr/bin/env python3
"""Generate the (config x SKU) admission matrix.

    utils/campaign/run_matrix.py --out raw/            # predict every cell
    utils/campaign/run_matrix.py --out raw/ --execute  # + launch on hardware

Two phases, deliberately separable:

**Predict** (no hardware). For each cell, rewrite the reference program's
configuration constants, compile it against the SKU's machine model, and record
Vx's verdict as a JSON cell. This is the paper's *prediction* half and it runs
anywhere -- which is the point: the whole matrix can be dry-run locally before
any rented time is burning, so a toolchain bug is found for free rather than at
an hourly rate.

**Execute** (needs the SKU). For each admitted cell, map the verdict to engine
arguments and launch, capturing the outcome verbatim. Not implemented here: see
`classify_outcome` for why the classifier is deliberately left to a pilot run.

The reference program is never edited. Only the constants in its single `admit<
...>` call change per cell, which is exactly the claim under test -- one program
text, N SKUs, N configs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ADMIT = REPO / "fleet" / "admit.vx"
FLEET = REPO / "fleet"

SKUS = ["a100-40", "a100-80", "h100-sxm", "h200", "b200", "mi300x"]


@dataclass(frozen=True)
class Config:
    """One serving configuration: the const-generic arguments of `admit`."""
    name: str
    layers: int
    heads: int
    kv_heads: int
    hdim: int
    ctx: int
    batch: int
    tp: int

    @property
    def cell_id(self) -> str:
        return self.name


# The sweep. Chosen to put cells on both sides of every SKU's ceiling rather
# than to be exhaustive: a matrix where every cell agrees is not evidence, it is
# a tautology. The 70B family at increasing context is the spine (KV growth is
# what makes long-context serving a capacity problem); the TP ladder tests the
# shard arithmetic; 8B and 405B anchor the ends.
#
# KV-head counts are the checkpoints' `num_key_value_heads` (GQA), not the query
# head count -- see the note in fleet/admit.vx. Llama-3.1 uses 8 across all sizes.
CONFIGS = [
    #        name               L    H  KVH  HD    CTX  B  TP
    Config("8b-ctx4k",         32,  32,  8, 128,   4096, 1, 1),
    Config("8b-ctx32k",        32,  32,  8, 128,  32768, 1, 1),
    Config("8b-ctx128k",       32,  32,  8, 128, 131072, 1, 1),
    Config("70b-ctx4k",        80,  64,  8, 128,   4096, 1, 1),
    Config("70b-ctx8k",        80,  64,  8, 128,   8192, 1, 1),
    Config("70b-ctx32k",       80,  64,  8, 128,  32768, 1, 1),
    Config("70b-ctx128k",      80,  64,  8, 128, 131072, 1, 1),
    Config("70b-ctx4k-tp2",    80,  64,  8, 128,   4096, 1, 2),
    Config("70b-ctx4k-tp4",    80,  64,  8, 128,   4096, 1, 4),
    Config("70b-ctx4k-tp8",    80,  64,  8, 128,   4096, 1, 8),
    Config("70b-ctx32k-tp4",   80,  64,  8, 128,  32768, 1, 4),
    Config("70b-ctx32k-tp8",   80,  64,  8, 128,  32768, 1, 8),
    Config("70b-ctx4k-batch8", 80,  64,  8, 128,   4096, 8, 1),
    Config("405b-ctx4k-tp8",  126, 128,  8, 128,   4096, 1, 8),
    Config("405b-ctx32k-tp8", 126, 128,  8, 128,  32768, 1, 8),
]

# `return admit<L, H, KVH, HD, CTX, B, TP>();` -- the one line a cell rewrites.
ADMIT_CALL = re.compile(r"admit<" + r"\s*\d+\s*,"*6 + r"\s*\d+\s*>")


def program_for(cfg: Config, template: str) -> str:
    """The reference program with this cell's constants substituted."""
    call = (f"admit<{cfg.layers}, {cfg.heads}, {cfg.kv_heads}, {cfg.hdim}, "
            f"{cfg.ctx}, {cfg.batch}, {cfg.tp}>")
    out, n = ADMIT_CALL.subn(call, template)
    if n != 1:
        sys.exit(
            f"run_matrix: expected exactly one `admit<...>` call in {ADMIT}, found {n}. "
            "The harness rewrites that call per cell; if the program's shape changed, "
            "update ADMIT_CALL rather than letting cells silently share a configuration."
        )
    return out


def predict(cfg: Config, sku: str, vxc: Path, workdir: Path) -> dict:
    """Compile one cell and return its record, or a record of why it failed to compile."""
    src = workdir / f"{cfg.cell_id}.vx"
    src.write_text(program_for(cfg, ADMIT.read_text()))
    cell_json = workdir / f"{cfg.cell_id}__{sku}.json"

    proc = subprocess.run(
        [
            str(vxc), "--machine", str(FLEET / f"{sku}.vx"), str(src),
            "--action", "emit-mlir", "-o", "/dev/null",
            "--diagnostics-json", str(cell_json),
        ],
        capture_output=True, text=True,
    )
    if not cell_json.exists():
        # No record written: the compiler failed before the verdict stage. This is a
        # harness/toolchain fault, not an admission result, and must not be silently
        # folded in as a rejection -- that would fabricate agreement with a rejecting
        # engine and inflate precision.
        return {
            "cell": cfg.cell_id, "sku": sku, "verdict": "TOOLCHAIN_ERROR",
            "stderr": proc.stderr[-2000:], "returncode": proc.returncode,
        }
    rec = json.loads(cell_json.read_text())
    resident = next((r for r in rec.get("resident_sets", []) if r["space"] == "HBM"), None)
    return {
        "cell": cfg.cell_id,
        "sku": sku,
        "config": asdict(cfg),
        "verdict": rec["verdict"],
        "error_codes": [d["code"] for d in rec["diagnostics"] if d.get("code", "").startswith("E")],
        "resident_bytes": resident["total_bytes"] if resident else None,
        "capacity_bytes": resident["capacity_bytes"] if resident else None,
        "utilization": resident["utilization"] if resident else None,
        "record": str(cell_json),
    }


def classify_outcome(_stdout: str, _stderr: str) -> str:
    """Map an engine launch's output to the ground-truth taxonomy.

    DELIBERATELY UNIMPLEMENTED. The taxonomy is fixed --
    BOOT_OK_SERVE_OK / ENGINE_REFUSES / OOM_AT_LOAD / OOM_AT_PREFILL / OTHER_FAIL
    -- but the *patterns* that map real output onto it are not, and writing them
    from memory is how a campaign silently mislabels its own ground truth. In
    particular ENGINE_REFUSES is the paper's most valuable category (it is vLLM
    performing runtime admission, the very check Vx claims to do statically), and
    a pattern that misses it would score as OTHER_FAIL and quietly discard the
    best evidence in the dataset.

    Build these from a pilot run's captured logs, not from recollection of what
    vLLM prints. Until then `--execute` records output verbatim and classifies
    everything as UNCLASSIFIED, which is honest and reversible: the raw logs can
    be re-classified once the patterns are known.
    """
    return "UNCLASSIFIED"


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the (config x SKU) admission matrix.")
    p.add_argument("--out", type=Path, required=True, help="output directory for cell records")
    p.add_argument("--vxc", type=Path, default=REPO / "target" / "debug" / "vxc")
    p.add_argument("--skus", nargs="*", default=SKUS)
    p.add_argument("--configs", nargs="*", default=None, help="config names (default: all)")
    p.add_argument("--execute", action="store_true",
                   help="launch admitted cells on hardware (requires the SKU; not implemented)")
    args = p.parse_args()

    if args.execute:
        sys.exit(
            "run_matrix: --execute is not implemented. It requires the rented SKU and a "
            "classifier built from a pilot run's real logs; see classify_outcome(). "
            "Predict-only runs anywhere and is what this script is for today."
        )
    if not args.vxc.exists():
        sys.exit(f"run_matrix: vxc not found at {args.vxc} (build it, or pass --vxc)")

    configs = CONFIGS
    if args.configs:
        by_name = {c.name: c for c in CONFIGS}
        missing = [n for n in args.configs if n not in by_name]
        if missing:
            sys.exit(f"run_matrix: unknown config(s): {', '.join(missing)}")
        configs = [by_name[n] for n in args.configs]

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "cells"
    work.mkdir(exist_ok=True)

    results = []
    for cfg in configs:
        for sku in args.skus:
            r = predict(cfg, sku, args.vxc, work)
            results.append(r)
            mark = {"admitted": "+", "rejected": "-"}.get(r["verdict"], "!")
            print(f"{mark} {cfg.name:<18} {sku:<10} {r['verdict']}", file=sys.stderr)

    manifest = args.out / "predicted_matrix.json"
    manifest.write_text(json.dumps(results, indent=2))

    admitted = sum(1 for r in results if r["verdict"] == "admitted")
    rejected = sum(1 for r in results if r["verdict"] == "rejected")
    errors = [r for r in results if r["verdict"] == "TOOLCHAIN_ERROR"]
    print(
        f"\n{len(results)} cells: {admitted} admitted, {rejected} rejected, "
        f"{len(errors)} toolchain errors -> {manifest}",
        file=sys.stderr,
    )
    if errors:
        print(
            "TOOLCHAIN_ERROR cells did not produce a verdict and are NOT rejections. "
            "Resolve before the campaign; counting them as rejections would fabricate "
            "agreement with a rejecting engine.",
            file=sys.stderr,
        )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
