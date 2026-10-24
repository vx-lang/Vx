#!/usr/bin/env python3
"""Generate the (config x SKU) admission matrix.

    utils/campaign/run_matrix.py --out raw/            # predict every cell
    utils/campaign/run_matrix.py --out raw/ --execute --configs 70b-ctx4k-tp2 \
        --skus h100-sxm --epsilon 0.05                 # + launch one cell here

Two phases, deliberately separable:

**Predict** (no hardware). For each cell, rewrite the reference program's
configuration constants, compile it against the SKU's machine model, and record
Vx's verdict as a JSON cell. This is the paper's *prediction* half and it runs
anywhere -- which is the point: the whole matrix can be dry-run locally before
any rented time is burning, so a toolchain bug is found for free rather than at
an hourly rate.

**Execute** (needs the SKU). For one admitted cell, on the machine the script runs on:
get the `vllm serve` command from utils/vllm/map_admission.py, launch it, send the
cell's requests at full length, and stop it. The engine's stdout and stderr go to
one log per cell, verbatim and in order. `classify_outcome` maps the log to an
outcome, with the patterns of the pilot run's real output. The record names the
vLLM version and the GPU. Nothing here depends on where the machine is, so it runs
the same in a container.

The reference program is never edited. Only the constants in its single `admit<
...>` call change per cell, which is exactly the claim under test -- one program
text, N SKUs, N configs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAP_ADMISSION = REPO / "utils" / "vllm" / "map_admission.py"
ADMIT = REPO / "fleet" / "admit.vx"
FLEET = REPO / "fleet"
# The program stages every resident through host memory, so a host must be
# declared (E6014). A named file rather than `--host default`: the frozen
# artifact should record *which* host, not "the machine of the day". Verdicts
# are host-independent by construction -- a host declares no capacity -- so
# this choice cannot move a cell; E6014 exists to make that an explicit input
# rather than an ambient default.
HOST = FLEET / "host-x86-e5-2666v3.vx"

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
            str(vxc), "--machine", str(FLEET / f"{sku}.vx"), "--host", str(HOST), str(src),
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
    codes = [d["code"] for d in rec["diagnostics"] if d.get("code", "").startswith("E")]
    # The JSON's verdict is "rejected iff error_count > 0", so a *configuration*
    # error -- E6014 (no host named), E6012 (a machine file passed as --host) --
    # also reads "rejected". A cell is a rejection only if every error it
    # carries is capacity arithmetic; anything else is the fabricated-agreement
    # trap described above, including mixed cases: a cell lost to manual review
    # is cheaper than a contaminated one.
    if rec["verdict"] == "rejected" and any(c not in ("E6009", "E6010") for c in codes):
        return {
            "cell": cfg.cell_id, "sku": sku, "verdict": "TOOLCHAIN_ERROR",
            "error_codes": codes, "stderr": proc.stderr[-2000:],
            "note": "non-capacity diagnostics present -- config error, not a capacity verdict",
        }
    resident = next((r for r in rec.get("resident_sets", []) if r["space"] == "HBM"), None)
    return {
        "cell": cfg.cell_id,
        "sku": sku,
        "config": asdict(cfg),
        "verdict": rec["verdict"],
        "error_codes": codes,
        "resident_bytes": resident["total_bytes"] if resident else None,
        "capacity_bytes": resident["capacity_bytes"] if resident else None,
        "utilization": resident["utilization"] if resident else None,
        "record": str(cell_json),
    }


# Verbatim substrings of vLLM 0.31.0's output, from the pilot runs on H100, A100-80GB, H200 and
# B200 (#289, 2026-10-05). Each outcome below names the substring that decides it.
#
# Printed once startup has passed: "GPU KV cache size: N tokens".
KV_CACHE_SIZE = "GPU KV cache size:"
# vLLM refuses a max_model_len that its KV cache cannot hold, and says how long it could be.
ESTIMATED_MAX_LEN = "estimated maximum model length is"
# Weights loaded; the out-of-memory error of a load comes after this, in the profiling run.
MODEL_LOADING_TOOK = "Model loading took"
TORCH_OOM = "torch.OutOfMemoryError: CUDA out of memory"
# Out of memory on the first full-length prompt, after startup passed.
PREFILL_FAILURES = ("cudaErrorMemoryAllocation", "EngineDeadError")
# Two crashes of the image, not of memory: no CUDA compiler, and no C compiler.
SETUP_FAULTS = (
    "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist",
    "InductorError: RuntimeError: Failed to find C compiler",
)


@dataclass(frozen=True)
class Outcome:
    """What one engine launch did."""
    label: str
    # The substring that decided the label, or None when none did.
    pattern: str | None = None
    # True when the image lacks a tool. Such a launch says nothing about memory, so it is
    # OTHER_FAIL and must be rerun in a fixed image rather than scored.
    setup_fault: bool = False


def classify_outcome(log: str, served: bool | None = None) -> Outcome:
    """Map one launch's output to BOOT_OK_SERVE_OK, ENGINE_REFUSES, OOM_AT_LOAD,
    OOM_AT_PREFILL or OTHER_FAIL, from the pilot's verbatim patterns.

    `log` is the engine's stdout and stderr in the order they were written. `served` is whether
    the request sent after startup came back: None when no request was sent.

    A log that no pattern covers is UNCLASSIFIED, not OTHER_FAIL. The raw log is kept, so it can
    be classified once its pattern is known.
    """
    for fault in SETUP_FAULTS:
        if fault in log:
            return Outcome("OTHER_FAIL", fault, setup_fault=True)
    if ESTIMATED_MAX_LEN in log:
        return Outcome("ENGINE_REFUSES", ESTIMATED_MAX_LEN)
    started = log.find(KV_CACHE_SIZE)
    oom = log.find(TORCH_OOM)
    if oom != -1 and (started == -1 or oom < started):
        return Outcome("OOM_AT_LOAD", TORCH_OOM)
    if started != -1:
        after = log[started:]
        for failure in PREFILL_FAILURES:
            if failure in after:
                return Outcome("OOM_AT_PREFILL", failure)
        if served:
            return Outcome("BOOT_OK_SERVE_OK", KV_CACHE_SIZE)
        if served is False:
            return Outcome("OTHER_FAIL")
    return Outcome("UNCLASSIFIED")


# One request's generated tokens. Its prompt takes the rest of the context, so each request is
# full length, as the first full-length prompt is where the pilot saw prefill run out of memory.
GENERATED_TOKENS = 16
# The token id that fills a prompt. Any id below the vocabulary size would do.
PROMPT_TOKEN = 1000


def map_admission(cfg: Config, record: Path, epsilon: float, model: str | None,
                  form: str, model_config: Path | None = None) -> str:
    """The output of utils/vllm/map_admission.py for an admitted cell, in `form`."""
    command = [
        sys.executable, str(MAP_ADMISSION), "--cell", str(record),
        "--layers", str(cfg.layers), "--heads", str(cfg.heads), "--kv-heads", str(cfg.kv_heads),
        "--hdim", str(cfg.hdim), "--ctx", str(cfg.ctx), "--batch", str(cfg.batch),
        "--tp", str(cfg.tp), "--epsilon", str(epsilon), "--format", form,
    ]
    if model:
        command += ["--model", model]
    if model_config:
        command += ["--verify-model", str(model_config)]
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"run_matrix: map_admission.py refused the cell:\n{proc.stderr}")
    return proc.stdout


def model_config_for(model: str, given: Path | None) -> Path:
    """The checkpoint's config.json: the file given, or the one huggingface_hub downloads."""
    if given:
        return given
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("run_matrix: pass --model-config, or install huggingface_hub to download it")
    return Path(hf_hub_download(model, "config.json"))


def engine_command(cfg: Config, record: Path, args: argparse.Namespace) -> tuple[list[str], dict]:
    """The `vllm serve` command that map_admission.py gives for an admitted cell, after it
    has checked the cell's geometry against the checkpoint's config.json, and the JSON form
    of the same mapping."""
    model = json.loads(map_admission(cfg, record, args.epsilon, args.model, "json"))
    model = model["engine"]["model"]
    config = model_config_for(model, args.model_config)
    mapped = json.loads(map_admission(cfg, record, args.epsilon, model, "json", config))
    flags = shlex.split(map_admission(cfg, record, args.epsilon, model, "args", config))
    command = [*shlex.split(args.vllm), "serve", model, *flags,
               "--host", "127.0.0.1", "--port", str(args.port)]
    return command, mapped


def ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def complete(port: int, model: str, ctx: int, timeout: float) -> bool:
    """One full-length request. True when it came back with generated text."""
    body = json.dumps({
        "model": model,
        "prompt": [PROMPT_TOKEN] * (ctx - GENERATED_TOKENS),
        "max_tokens": GENERATED_TOKENS,
        "temperature": 0,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:
            choices = json.loads(r.read()).get("choices") or []
            return r.status == 200 and bool(choices) and "text" in choices[0]
    except (urllib.error.URLError, OSError, ValueError):
        return False


def run_engine(command: list[str], log: Path, port: int, model: str, ctx: int, batch: int,
               startup_timeout: float, request_timeout: float) -> dict:
    """Launch the engine with its output in `log`, send `batch` full-length requests once it
    is up, and stop it. `served` is None when it never came up."""
    started = time.monotonic()
    with open(log, "wb") as out:
        proc = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        served = None
        try:
            while proc.poll() is None and time.monotonic() - started < startup_timeout:
                if ready(port):
                    with concurrent.futures.ThreadPoolExecutor(batch) as pool:
                        done = list(pool.map(lambda _: complete(port, model, ctx, request_timeout),
                                             range(batch)))
                    served = all(done)
                    break
                time.sleep(2)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    return {"served": served, "returncode": proc.returncode,
            "seconds": round(time.monotonic() - started, 1)}


def first_line(command: list[str]) -> str | None:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = proc.stdout.strip().splitlines()
    return lines[0].strip() if proc.returncode == 0 and lines else None


def gpu_names() -> list[str]:
    """What nvidia-smi, or else rocm-smi, says each GPU of this machine is called."""
    for command in (["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    ["rocm-smi", "--showproductname", "--csv"]):
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return []


def execute(cfg: Config, sku: str, record: dict, args: argparse.Namespace) -> dict:
    """Launch one predicted cell here and record what the engine did."""
    out = {"cell": cfg.cell_id, "sku": sku, "verdict": record["verdict"],
           "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if record["verdict"] != "admitted":
        # map_admission.py gives no engine arguments for a cell that Vx does not admit.
        out["outcome"] = "NOT_LAUNCHED"
        out["note"] = "the cell is not admitted, so map_admission.py gives no launch command"
        return out
    command, mapped = engine_command(cfg, Path(record["record"]), args)
    model = mapped["engine"]["model"]
    log = args.out / "logs" / f"{cfg.cell_id}__{sku}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    run = run_engine(command, log, args.port, model, cfg.ctx, cfg.batch,
                     args.startup_timeout, args.request_timeout)
    outcome = classify_outcome(log.read_text(errors="replace"), run["served"])
    out.update({
        "command": command,
        "engine": mapped["engine"],
        "provenance": mapped["provenance"],
        "epsilon": args.epsilon,
        "log": str(log),
        "vllm_version": first_line([*shlex.split(args.vllm), "--version"]),
        "gpus": gpu_names(),
        **run,
        "outcome": outcome.label,
        "pattern": outcome.pattern,
        "setup_fault": outcome.setup_fault,
    })
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Generate the (config x SKU) admission matrix.")
    p.add_argument("--out", type=Path, required=True, help="output directory for cell records")
    p.add_argument("--vxc", type=Path, default=REPO / "target" / "debug" / "vxc")
    p.add_argument("--skus", nargs="*", default=SKUS)
    p.add_argument("--configs", nargs="*", default=None, help="config names (default: all)")
    p.add_argument("--execute", action="store_true",
                   help="launch one cell (one --configs, one --skus) on this machine")
    p.add_argument("--epsilon", type=float,
                   help="headroom for map_admission.py; required with --execute, and the "
                        "same for every cell of a campaign")
    p.add_argument("--vllm", default="vllm", help="the command that runs vLLM (default: vllm)")
    p.add_argument("--model", help="the checkpoint, in place of map_admission.py's table")
    p.add_argument("--model-config", type=Path,
                   help="the checkpoint's config.json (default: download it with huggingface_hub)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-timeout", type=float, default=3600.0, help="seconds")
    p.add_argument("--request-timeout", type=float, default=1800.0, help="seconds")
    args = p.parse_args()

    if args.execute:
        if args.epsilon is None:
            sys.exit("run_matrix: --execute needs --epsilon, the campaign's one value")
        if not (args.configs and len(args.configs) == 1 and len(args.skus) == 1):
            sys.exit("run_matrix: --execute runs one cell: give one --configs and one --skus")
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

    if args.execute:
        cfg, record = configs[0], results[0]
        outcome = execute(cfg, args.skus[0], record, args)
        path = args.out / f"{cfg.cell_id}__{args.skus[0]}.outcome.json"
        path.write_text(json.dumps(outcome, indent=2))
        print(f"{outcome['outcome']} -> {path}", file=sys.stderr)
        return 0 if record["verdict"] != "TOOLCHAIN_ERROR" else 1

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
