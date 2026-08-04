#!/usr/bin/env python3
"""Map a Vx admission verdict onto vLLM engine arguments.

    vxc --machine fleet/h100-sxm.vx fleet/admit.vx \
        --action emit-mlir -o /dev/null --diagnostics-json cell.json
    utils/vllm/map_admission.py --cell cell.json \
        --layers 80 --heads 64 --hdim 128 --ctx 4096 --batch 1 --tp 1

The toolchain's contract ends at emitting verified facts in a stable format
(`--diagnostics-json`: the verdict, per-space resident totals, capacities,
margins). Mapping those facts onto one engine's CLI surface is downstream
integration, which is why this lives in `utils/` rather than in `vxc` or in
`fleet/` -- vLLM's flag namespace churns release to release, and coupling the
toolchain to it would cut against the claim that Vx is an engine-agnostic
admission layer. Compare `llvm/utils/opt-viewer`: the compiler emits remark
YAML, the consumer ships in-tree, outside the toolchain contract.

The equivalent mapper for another engine is the same afternoon of work, and
that is the point.

Scope: this utility is *stateless and general* -- one cell in, one launch
command out. Campaign policy (pinning epsilon once, making --verify-model
mandatory, the boot-outcome taxonomy, one raw log per cell) belongs to the
harness in #289, not here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The vLLM release these flag names and dtype spellings were checked against.
# vLLM's engine arguments change between releases: re-check on upgrade and
# update this string, so a campaign artifact records what it was built for.
#
# Verified 2026-08-03 against vLLM documentation:
#   --kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}
#   (fp8 is an alias for fp8_e4m3; ROCm supports fp8 = fp8_e4m3)
PINNED_VLLM = "0.9.x (flag spellings verified 2026-08-03; re-check on upgrade)"

# Vx element type -> vLLM --kv-cache-dtype spelling.
KV_DTYPE = {
    "f16": "auto",
    "bf16": "auto",
    "f8e4m3": "fp8_e4m3",
    "f8e5m2": "fp8_e5m2",
}

# Vx element type -> vLLM --dtype spelling (weights/activations).
WEIGHT_DTYPE = {
    "f16": "float16",
    "bf16": "bfloat16",
}

# (layers, heads, head_dim) -> HuggingFace checkpoint id.
#
# Deliberately a lookup rather than a heuristic: a geometry triple does not
# uniquely determine a checkpoint (fine-tunes share it), so guessing would
# silently mislabel a data point. An unknown geometry is an error the caller
# resolves by adding an entry, not something this script infers.
CHECKPOINTS = {
    (80, 64, 128): "meta-llama/Llama-3.1-70B-Instruct",
    (32, 32, 128): "meta-llama/Llama-3.1-8B-Instruct",
    (126, 128, 128): "meta-llama/Llama-3.1-405B-Instruct",
}


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"map_admission: {msg}", file=sys.stderr)
    sys.exit(2)


def load_cell(path: Path) -> dict:
    try:
        cell = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        die(f"cannot read cell record {path}: {e}")
    schema = cell.get("schema", "")
    if not schema.startswith("vx-diagnostics-v"):
        die(f"{path} is not a Vx diagnostics record (schema={schema!r})")
    return cell


def device_resident(cell: dict, space: str) -> dict:
    """The resident set for `space`, or exit with a diagnosis."""
    sets = cell.get("resident_sets") or []
    for r in sets:
        if r["space"] == space:
            return r
    if not sets:
        die(
            "cell record carries no resident_sets. Either the program placed "
            "nothing in a capacity-bearing space, or it was produced by a vxc "
            "predating the resident_sets field. Re-emit with a current vxc."
        )
    have = ", ".join(sorted(r["space"] for r in sets))
    die(f"no resident set for space {space!r}; record has: {have}")


def verify_model(config_path: Path, layers: int, heads: int, hdim: int) -> None:
    """Assert the cell's const-generic geometry matches a real checkpoint.

    The admission verdict is arithmetic over (layers, heads, head_dim). If those
    do not match the checkpoint actually served, the verdict describes a model
    that was never run and the ground truth is contaminated -- a harness bug, not
    a data point. Checked *before* launch for exactly that reason.
    """
    try:
        cfg = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        die(f"--verify-model: cannot read {config_path}: {e}")

    actual_layers = cfg.get("num_hidden_layers")
    actual_heads = cfg.get("num_attention_heads")
    hidden = cfg.get("hidden_size")
    # head_dim is usually implicit; prefer an explicit field when present.
    actual_hdim = cfg.get("head_dim")
    if actual_hdim is None and hidden and actual_heads:
        actual_hdim = hidden // actual_heads

    mismatches = []
    for name, want, got in (
        ("layers", layers, actual_layers),
        ("heads", heads, actual_heads),
        ("head_dim", hdim, actual_hdim),
    ):
        if got is None:
            mismatches.append(f"{name}: cell={want}, checkpoint=<absent from config.json>")
        elif int(got) != int(want):
            mismatches.append(f"{name}: cell={want}, checkpoint={got}")

    if mismatches:
        die(
            "--verify-model: cell geometry does not match the checkpoint:\n  "
            + "\n  ".join(mismatches)
            + "\nThe admission verdict describes a different model than would be "
            "served. Fix the cell constants or the checkpoint; do not record this "
            "as a data point."
        )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Map a Vx admission verdict (--diagnostics-json) to vLLM engine arguments.",
        epilog=f"vLLM surface pinned at: {PINNED_VLLM}",
    )
    p.add_argument("--cell", type=Path, required=True,
                   help="cell.json from `vxc --diagnostics-json`")
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--heads", type=int, required=True)
    p.add_argument("--hdim", type=int, required=True)
    p.add_argument("--ctx", type=int, required=True, help="max context; becomes --max-model-len")
    p.add_argument("--batch", type=int, required=True, help="becomes --max-num-seqs")
    p.add_argument("--tp", type=int, required=True, help="becomes --tensor-parallel-size")
    p.add_argument("--weights-dtype", default="f16", choices=sorted(WEIGHT_DTYPE),
                   help="Vx element type of the weights (default: f16)")
    p.add_argument("--kv-dtype", default="f16", choices=sorted(KV_DTYPE),
                   help="Vx element type of the KV cache (default: f16)")
    p.add_argument("--space", default="HBM",
                   help="device memory space in the machine model (default: HBM)")
    p.add_argument(
        "--epsilon", type=float, default=0.05,
        help=(
            "headroom added to the measured utilization, for runtime overhead Vx "
            "does not model (CUDA context, fragmentation, non-KV allocations). "
            "Default 0.05. PIN THIS ONCE per campaign and record it: tuning it "
            "per cell would launder runtime tuning back into what is claimed to "
            "be a static admission decision."
        ),
    )
    p.add_argument("--max-utilization", type=float, default=0.95,
                   help="ceiling on --gpu-memory-utilization (default: 0.95)")
    p.add_argument("--verify-model", type=Path, metavar="CONFIG_JSON",
                   help="assert the checkpoint's config.json matches the cell geometry")
    p.add_argument("--model", help="override the checkpoint id instead of the lookup table")
    p.add_argument("--format", choices=("args", "json", "shell"), default="shell",
                   help="output form (default: shell, a `vllm serve` command)")
    args = p.parse_args()

    cell = load_cell(args.cell)

    # A rejected cell has no engine config to emit. Emitting one anyway would
    # invite launching it "to see", which is how a rejected cell becomes an
    # unlabelled data point.
    if cell.get("verdict") != "admitted":
        print(
            f"map_admission: cell verdict is {cell.get('verdict')!r}, not 'admitted'; "
            "no engine configuration is implied. The expected observation for a "
            "rejected cell is a boot failure, which the harness records without "
            "needing launch args from here.",
            file=sys.stderr,
        )
        return 1

    if args.verify_model:
        verify_model(args.verify_model, args.layers, args.heads, args.hdim)

    model = args.model
    if not model:
        key = (args.layers, args.heads, args.hdim)
        model = CHECKPOINTS.get(key)
        if not model:
            die(
                f"no checkpoint known for geometry {key}. Add it to CHECKPOINTS "
                "or pass --model; this is not inferred, since a geometry triple "
                "does not uniquely identify a checkpoint."
            )

    resident = device_resident(cell, args.space)
    util = resident["utilization"] + args.epsilon
    if util > args.max_utilization:
        print(
            f"map_admission: required utilization {util:.4f} exceeds the "
            f"{args.max_utilization} ceiling; clamping. This cell sits close "
            "enough to the device ceiling that a boot failure is a plausible "
            "outcome even though Vx admitted it -- Vx models the resident set, "
            "not vLLM's runtime overhead. Record the outcome either way.",
            file=sys.stderr,
        )
    util = min(args.max_utilization, util)

    conf = {
        "model": model,
        "tensor_parallel_size": args.tp,
        "max_model_len": args.ctx,
        "max_num_seqs": args.batch,
        "dtype": WEIGHT_DTYPE[args.weights_dtype],
        "kv_cache_dtype": KV_DTYPE[args.kv_dtype],
        "gpu_memory_utilization": round(util, 4),
    }
    provenance = {
        "cell": str(args.cell),
        "machine": cell.get("machine"),
        "program": cell.get("file"),
        "resident_bytes": resident["total_bytes"],
        "capacity_bytes": resident["capacity_bytes"],
        "measured_utilization": resident["utilization"],
        "epsilon": args.epsilon,
        "vllm_pinned": PINNED_VLLM,
    }

    if args.format == "json":
        print(json.dumps({"engine": conf, "provenance": provenance}, indent=2))
    else:
        flags = [
            f"--tensor-parallel-size {conf['tensor_parallel_size']}",
            f"--max-model-len {conf['max_model_len']}",
            f"--max-num-seqs {conf['max_num_seqs']}",
            f"--dtype {conf['dtype']}",
            f"--kv-cache-dtype {conf['kv_cache_dtype']}",
            f"--gpu-memory-utilization {conf['gpu_memory_utilization']}",
        ]
        if args.format == "args":
            print(" ".join(flags))
        else:
            print(f"vllm serve {conf['model']} \\\n  " + " \\\n  ".join(flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
