# Dry-run baseline — predicted admission matrix

Generated 2026-08-03 by `utils/campaign/run_matrix.py`, before any hardware was rented.

Two things are recorded here. The **predicted matrix** is regenerable at any time (`run_matrix.py --out raw/`) and is kept as a regression baseline: if a compiler or fleet change moves a cell, it
should be because someone meant it to. The **GQA correction** below is *not* regenerable — its
"before" column required the buggy program — which is why it is written down rather than left to be
re-derived.

## Predicted matrix, 2026-10-24

Generated at main e9b13550, with the routes back to the host of #2184. Without them the H200 and
MI300X columns are 30 toolchain errors. This table is the baseline from this date. The 2026-08-03
table below stays as the record of its own date.

15 configurations × 6 SKUs = 90 cells. **65 admitted, 25 rejected, 0 toolchain errors.**

The inputs changed since 2026-08-03:

- `admit.vx` takes ten parameters (#1627). The weights are the whole checkpoint: the embedding
  table, the output layer, and per layer the attention projections and a gated MLP of width FFN.
  The 70B resident set at 4k context grew from 121.3 to 132.7 GiB.
- FFN and VOCAB come from each checkpoint's config.json: 14336, 28672 and 53248, and 128256.
- STEP, the tokens of one step, is the smaller of 16384 and BATCH × CTX, as `admit.vx` says.
- The B200's capacity is the measured 183,359 MiB, or 179.06 GiB (#1596), not 192 GiB.

| config | a100-40 | a100-80 | h100-sxm | h200 | b200 | mi300x | resident GiB |
|---|---|---|---|---|---|---|---|
| `8b-ctx4k` | admit | admit | admit | admit | admit | admit | 15.5 |
| `8b-ctx32k` | admit | admit | admit | admit | admit | admit | 19.1 |
| `8b-ctx128k` | admit | admit | admit | admit | admit | admit | 31.1 |
| `70b-ctx4k` | reject | reject | reject | admit | admit | admit | 132.7 |
| `70b-ctx8k` | reject | reject | reject | admit | admit | admit | 134.0 |
| `70b-ctx32k` | reject | reject | reject | reject | admit | admit | 141.7 |
| `70b-ctx128k` | reject | reject | reject | reject | admit | admit | 171.7 |
| `70b-ctx4k-tp2` | reject | admit | admit | admit | admit | admit | 66.4 |
| `70b-ctx4k-tp4` | admit | admit | admit | admit | admit | admit | 33.2 |
| `70b-ctx4k-tp8` | admit | admit | admit | admit | admit | admit | 16.6 |
| `70b-ctx32k-tp4` | admit | admit | admit | admit | admit | admit | 35.6 |
| `70b-ctx32k-tp8` | admit | admit | admit | admit | admit | admit | 17.9 |
| `70b-ctx4k-batch8` | reject | reject | reject | reject | admit | admit | 141.7 |
| `405b-ctx4k-tp8` | reject | reject | reject | admit | admit | admit | 94.9 |
| `405b-ctx32k-tp8` | reject | reject | reject | admit | admit | admit | 97.0 |

SKU capacities for reading the table: a100-40 = 40 GiB, a100-80 / h100-sxm = 80 GiB, h200 = 141 GiB,
b200 = 179.06 GiB, mi300x = 192 GiB.

### Cells whose verdict differs from 2026-08-03

**2 of 90.** Both are on the H200, and both flipped from admit to reject:

| config | SKU | 2026-08-03 | 2026-10-24 | resident GiB |
|---|---|---|---|---|
| `70b-ctx32k` | h200 | admit | reject | 130.5 → 141.7 |
| `70b-ctx4k-batch8` | h200 | admit | reject | 130.5 → 141.7 |

The whole-checkpoint weights put both 0.7 GiB over the H200's 141 GiB, so they are marginal cells,
inside the ~10% band that the reading notes below treat as unresolved. No B200 cell flipped with
its smaller capacity: `70b-ctx128k` needs 171.7 of its 179.06 GiB. If the H200's 141 is decimal
GB, as `fleet/README.md` says may be so, its capacity is 131.3 GiB, and `70b-ctx4k` and
`70b-ctx8k` reject on it too.

## Predicted matrix (current, post-#303)

15 configurations × 6 SKUs = 90 cells. **67 admitted, 23 rejected, 0 toolchain errors.**

Resident GiB is per GPU after TP division, and is SKU-independent (it depends only on the
configuration), so one column serves the whole row.

| config | a100-40 | a100-80 | h100-sxm | h200 | b200 | mi300x | resident GiB |
|---|---|---|---|---|---|---|---|
| `8b-ctx4k` | admit | admit | admit | admit | admit | admit | 12.5 |
| `8b-ctx32k` | admit | admit | admit | admit | admit | admit | 16.2 |
| `8b-ctx128k` | admit | admit | admit | admit | admit | admit | 29.0 |
| `70b-ctx4k` | reject | reject | reject | admit | admit | admit | 121.3 |
| `70b-ctx8k` | reject | reject | reject | admit | admit | admit | 122.6 |
| `70b-ctx32k` | reject | reject | reject | admit | admit | admit | 130.5 |
| `70b-ctx128k` | reject | reject | reject | reject | admit | admit | 162.0 |
| `70b-ctx4k-tp2` | reject | admit | admit | admit | admit | admit | 60.7 |
| `70b-ctx4k-tp4` | admit | admit | admit | admit | admit | admit | 30.4 |
| `70b-ctx4k-tp8` | admit | admit | admit | admit | admit | admit | 15.2 |
| `70b-ctx32k-tp4` | admit | admit | admit | admit | admit | admit | 33.0 |
| `70b-ctx32k-tp8` | admit | admit | admit | admit | admit | admit | 16.8 |
| `70b-ctx4k-batch8` | reject | reject | reject | admit | admit | admit | 130.5 |
| `405b-ctx4k-tp8` | reject | reject | reject | admit | admit | admit | 94.9 |
| `405b-ctx32k-tp8` | reject | reject | reject | admit | admit | admit | 97.5 |

SKU capacities for reading the table: a100-40 = 40 GiB, a100-80 / h100-sxm = 80 GiB, h200 = 141 GiB,
b200 = 192 GiB, mi300x = 192 GiB.

### Reading this before the campaign runs

- **`70b-ctx128k` (162.0 GiB) is the sharpest boundary cell**: it clears the 141 GiB h200 by
  rejecting and the 192 GiB b200/mi300x by admitting, so it is the row most likely to disagree with
  the engine and the most informative if it does.
- **The TP ladder is the cleanest internal consistency check.** `70b-ctx4k` at TP 1/2/4/8 gives
  121.3 / 60.7 / 30.4 / 15.2 GiB — halving each step, as sharded weights should. A campaign result
  that breaks that progression indicates a harness fault, not a hardware finding.
- **The 8B row admits everywhere.** It is a control, not evidence: if any 8B cell fails to boot, the
  fault is in the harness or the checkpoint, not in the admission model.

### These are device-fit verdicts, not deployment-fit verdicts

`capacity × utilization` is the real admission bound and the model expresses only the first factor
(see `fleet/README.md`). At vLLM's default `gpu_memory_utilization` of 0.9, a 192 GiB B200 offers
~173 GiB, so `70b-ctx128k` at 162.0 GiB sits inside the modelled ceiling but close to the practical
one. **Cells within ~10% of a SKU's capacity should be treated as unresolved until the campaign
measures them**, not as predicted admissions.

## The GQA correction (#303) — not reproducible, hence recorded

The reference program originally sized the KV cache from the *query* head count, i.e. it assumed
multi-head attention. Llama-3.1 uses grouped-query attention (8 KV heads against 64 query heads at
70B), so the KV cache was overestimated by up to 8×. Reproducing the "before" column would require
reverting `fleet/admit.vx`.

### Resident set per configuration, MHA assumption → GQA

| config | before (GiB) | after (GiB) | overestimate |
|---|---|---|---|
| `8b-ctx4k` | 14.0 | 12.5 | 1.12× |
| `8b-ctx32k` | 28.2 | 16.2 | 1.74× |
| `8b-ctx128k` | 77.0 | 29.0 | 2.66× |
| `70b-ctx4k` | 130.1 | 121.3 | 1.07× |
| `70b-ctx8k` | 140.1 | 122.6 | 1.14× |
| `70b-ctx32k` | 200.5 | 130.5 | 1.54× |
| `70b-ctx128k` | 442.0 | 162.0 | **2.73×** |
| `70b-ctx4k-tp2` | 65.1 | 60.7 | 1.07× |
| `70b-ctx4k-tp4` | 32.6 | 30.4 | 1.07× |
| `70b-ctx4k-tp8` | 16.3 | 15.2 | 1.07× |
| `70b-ctx32k-tp4` | 50.5 | 33.0 | 1.53× |
| `70b-ctx32k-tp8` | 25.5 | 16.8 | 1.52× |
| `70b-ctx4k-batch8` | 200.5 | 130.5 | 1.54× |
| `405b-ctx4k-tp8` | 98.6 | 94.9 | 1.04× |
| `405b-ctx32k-tp8` | 127.0 | 97.5 | 1.30× |

### Cells whose verdict flipped

**10 of 90 (11%), every one a false reject.** Matrix totals moved from 57 admitted / 33 rejected to
67 / 23.

| config | SKUs that flipped reject → admit |
|---|---|
| `70b-ctx32k` | h200, b200, mi300x |
| `70b-ctx4k-batch8` | h200, b200, mi300x |
| `70b-ctx128k` | b200, mi300x |
| `8b-ctx128k` | a100-40 |
| `70b-ctx32k-tp4` | a100-40 |

### Why this table is worth keeping

The error scaled with context length — 1.07× at 4k, 2.73× at 128k — because it inflated only the KV
term, and KV is what grows with context. So it was smallest on the cells that would have looked
fine and largest on the long-context cells that are the paper's actual subject.

Had the campaign run first, those ten cells would have appeared as **precision failures**: Vx
rejects, vLLM serves the configuration without difficulty. The natural reading of ten such cells is
"the admission model is conservative" — a plausible, publishable-sounding conclusion — when the real
cause was that the model was wrong about attention geometry. A wrong result that resembles a
legitimate finding is more expensive than a crash, and roughly a third of the interesting cells
would have carried it, at rental rates.

That is the argument for dry-running the matrix before booking hardware, and this table is the
evidence for it.
