# scripts/

Grouped by lifecycle, because that is what predicts breakage: provisioning breaks
when a vendor changes, campaign runners when the language moves, workloads when
semantics move. Mixing them in one directory meant a change in any of those
weathered the whole set, and nothing walked it to notice.

| Directory | What lives here | Breaks when |
|---|---|---|
| `campaigns/` | Multi-run measurements that produce numbers | The language moves under a template |
| `provision/` | Pod and box setup | A vendor, driver or distro changes |
| `tools/` | Analysis one-offs — kernel extraction, device probes, generators | The thing being analysed changes shape |
| `demos/` | Runnable demonstrations, not measurements | Either of the above |
| `pre-commit.sh` | Stays at the root: the installed git hook is a copy of it | — |

## Campaigns keep their templates

Each campaign holds the runner **and the template it substitutes**, in one
directory:

```
campaigns/flash/run_flash_bench.sh  +  flash_*_bench.vx
campaigns/gemm/run_gemm_bench.sh    +  gpu_gemm_bench.vx
campaigns/llama/run_perf_matrix.sh
```

A template is an input to exactly one script. They used to sit in a shared
`scripts/templates/`, which expressed no such coupling — and all nine had rotted
against a language that moved, with nothing to say so. Seven compile today; the
three that do not (`flash_coop`, `flash_fa2`, `flash_splitk`) stage a row from
GPU_HBM into an SMEM tile, which Vx#352 has not materialised, and are left
failing rather than papered over.

## Running one

Every campaign runner works from a checkout **and** from a pod bundle. It finds
its template beside itself and the compiler at the nearest ancestor holding
`stdlib/` (or `$VXC`). They previously took their own directory as the bundle
root, so from a checkout they could not run at all — which is most of why the
templates rotted unseen.

```
./scripts/campaigns/flash/run_flash_bench.sh -k 512 -n 2 -o /tmp/flash
```

A runner exits non-zero when it measured nothing. That is worth stating because
it did not: `run_gemm_bench.sh` could have every run fail, print a grid of `?`,
write a header-only CSV, and exit 0.

## Where numbers come from

Per-workload measurements come from `benchmarks/` through `cargo vx-bench`, which
records the commit and the machine alongside each figure — see
`benchmarks/manifest.txt`. The campaigns here are the multi-run sweeps that need
hardware; CI gates only the hardware-free half.

## Paths that moved

Earlier campaign notes cite the old locations. This table resolves them:

| Was | Is |
|---|---|
| `scripts/run_flash_bench.sh` | `scripts/campaigns/flash/run_flash_bench.sh` |
| `scripts/run_gemm_bench.sh` | `scripts/campaigns/gemm/run_gemm_bench.sh` |
| `scripts/run_perf_matrix.sh` | `scripts/campaigns/llama/run_perf_matrix.sh` |
| `scripts/templates/*.vx` | beside their runner in `scripts/campaigns/*/` |
| `scripts/seam_scaling.py`, `admission_matrix.sh`, `measure_wire_traffic.sh` | `scripts/campaigns/` |
| `scripts/setup_*.sh`, `bootstrap_dev_pod.sh`, `make_gpu_bundle.sh`, `install_enzyme.sh` | `scripts/provision/` |
| `scripts/run_*_demo.sh` | `scripts/demos/` |
| everything else | `scripts/tools/` |
