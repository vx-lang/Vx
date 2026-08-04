# `utils/vllm/` — mapping a Vx admission verdict onto vLLM

`map_admission.py` turns one Vx admission record into a `vllm serve` command.

```console
$ vxc --machine fleet/h100-sxm.vx fleet/admit.vx \
      --action emit-mlir -o /dev/null --diagnostics-json cell.json
$ utils/vllm/map_admission.py --cell cell.json \
      --layers 80 --heads 64 --hdim 128 --ctx 4096 --batch 1 --tp 1
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --dtype float16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.7274
```

## Why this is in `utils/`, not in the compiler

The toolchain's contract ends at emitting verified **facts** in a stable format: `--diagnostics-json`
gives the verdict, per-space resident totals, capacities, and margins. Mapping those onto one
engine's CLI is downstream integration. vLLM's flag namespace churns between releases, and wiring
`vxc` to it would undercut the claim that Vx is an engine-agnostic admission layer.

Precedent: `llvm/utils/opt-viewer` — the compiler emits remark YAML; the consumer ships in-tree,
outside the toolchain contract. The equivalent mapper for TensorRT-LLM or SGLang is the same
afternoon of work against the same JSON.

## The mapping

| From | To |
|---|---|
| `--tp` | `--tensor-parallel-size` |
| `--ctx` | `--max-model-len` (static max-context admission is the V1 rule) |
| `--batch` | `--max-num-seqs` |
| `--weights-dtype` | `--dtype` (`f16`→`float16`, `bf16`→`bfloat16`) |
| `--kv-dtype` | `--kv-cache-dtype` (`f8e4m3`→`fp8_e4m3`, `f8e5m2`→`fp8_e5m2`, else `auto`) |
| resident/capacity from `cell.json` | `--gpu-memory-utilization min(0.95, util + ε)` |
| `(layers, heads, hdim)` | HF checkpoint id, via the `CHECKPOINTS` table |

Note the utility never parses a `.vx` file. The compiler already resolved the SKU's capacity into
the record, so both operands of the utilization arrive together and cannot disagree.

## ε, and why it must not be tuned per cell

Vx models the **resident set** — weights, KV cache, activations. It does not model vLLM's runtime
overhead: CUDA context, allocator fragmentation, non-KV buffers. `ε` (default `0.05`) is the
headroom covering that gap.

**Pin it once per campaign and record it.** Tuning ε per cell would launder runtime tuning back
into a decision claimed to be static, and would make the precision/recall table meaningless — every
mismatch could be tuned away instead of reported. A cell that needs a larger ε than the campaign
value is a **finding**: it marks where the admission model's abstraction is too coarse, which is
exactly the calibration data the paper wants.

## Guard rails

- **A rejected cell produces no launch command** (exit 1). Emitting one would invite launching it
  "just to see", which is how a rejected cell becomes an unlabelled data point.
- **`--verify-model config.json`** asserts the cell's `(layers, heads, head_dim)` match the
  checkpoint actually being served. A mismatch means the verdict describes a model that was never
  run — a harness bug, not a data point. The campaign harness should make this mandatory.
- **Unknown geometry is an error**, not a guess: a geometry triple does not uniquely identify a
  checkpoint (fine-tunes share it), so inferring one would silently mislabel a result.
- **Utilization above the ceiling warns before clamping** — such a cell sits close enough to the
  device ceiling that a boot failure is plausible despite Vx admitting it. Record the outcome
  either way.

## Version pinning

`PINNED_VLLM` in the script records the vLLM release its flag spellings were checked against
(`--kv-cache-dtype {auto,fp8,fp8_e5m2,fp8_e4m3}`, verified 2026-08-03). Re-check and update on
upgrade: a campaign artifact should say what surface it was built for.

## Scope

Stateless and general — one cell in, one command out. Campaign *policy* (pinning ε, mandatory
`--verify-model`, the boot-outcome taxonomy, one raw log per cell, precision/recall from logs
alone) belongs to the harness in #289.

Out of scope for V1, and named in the paper's scope paragraph: pipeline parallelism, multi-node
serving, quantized weights beyond dtype selection, speculative decoding, chunked prefill tuning.
