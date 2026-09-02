# `utils/differential/` — the same mistake, in Vx and in CUDA

`tests/frontend/fail/` proves Vx *refuses* a class of programs. On its own that shows the type
system has rules. This directory supplies the other half: the same mistake written in CUDA, compiled
by `nvcc` without complaint, and then failing at run time.

The pairing is the claim. Each case records four facts for both toolchains:

| | |
| ---------- | ---------------------------------------------------------- |
| `compiles` | did the toolchain accept it |
| `runs` | did the process exit 0 |
| `correct` | did it produce the right answer |
| `evidence` | the failure text, **verbatim** — a paraphrase is not evidence |

## Not cherry-picked

`05_smem_static_over_ceiling` is here because `nvcc` **catches** it. A suite where every case favours
Vx is a suite someone chose the cases for. The interesting claim is about which mistakes each
toolchain catches and where in the pipeline, so a case CUDA catches at compile time belongs in the
table as much as one it misses.

`06_smem_under_fixed_over_declared` is the same ceiling from the other side, and it contains no
mistake at all: a 64 KiB tile that the A100 has room for, that the declared model admits, and that
`nvcc` will not accept as a static `__shared__` array. Run 05 and 06 together or neither. Alone, 05
flatters CUDA and 06 flatters Vx; together they say the accurate thing, which is that a fixed ceiling
is wrong in both directions and a declared one is not.

A pair therefore says which outcome it expects from each side — `EXPECT_VX` (`refused` by default,
`accepted` for 06) and `EXPECT_CUDA` (`runtime-failure`, `compile-error`, or `runtime-success`).
A Vx acceptance is reported as `ACCEPTED` in capitals when the pair expected a refusal, because for
those pairs it means the program the case is built around is no longer being rejected.

## Running it

```
./run.sh                 # every pair
./run.sh 01 03           # named pairs only
./run.sh --cuda-only     # the GPU half alone, no vxc needed
./run.sh --list          # what is here, no compiling
```

Writes `results/<utc-timestamp>/` with one directory per pair, the raw stdout/stderr of every
command, and `results.json` collecting the four facts per toolchain per pair.

### It takes two machines, and that is fine

The Vx half is a compile-time refusal. It says the same thing on any machine, so run it wherever the
compiler is and keep the output.

The CUDA half needs a real GPU, because the claim is about run-time behaviour and has to be observed
rather than argued. A rented pod is adequate — pod *timing* is indicative only (see
`docs/discussions/walkthrough_gpu_campaign_m5_2026_08_10.md`), and nothing here is timed. A fault is
a fault.

**On the GPU box, use `--cuda-only`.** `vxc` is built for whichever host built it, so a macOS arm64
binary will not run on a Linux pod, and carrying a compiler over to re-derive a machine-independent
refusal buys nothing. `--cuda-only` skips the Vx half and records it as `not-run-here` rather than
inventing a verdict for it.

Without `--cuda-only`, a missing `vxc` is a hard error before anything runs. That guard is there for
a specific reason: a missing binary exits 127, 127 is non-zero, and a runner that reads "non-zero
means refused" would record a clean refusal for all five pairs without ever starting a compiler. The
same reasoning is why a non-zero exit with no diagnostic on stdout or stderr is reported as
`NO-DIAGNOSTIC` instead of `refused` — a crash and a refusal are not the same result.

## Capacity cases and the machine model

The capacity pairs are sized against a **cited** fleet model rather than a number chosen to match
whatever card is rented. `fleet/a100-40.vx` declares 40 GiB with its provenance recorded in
`fleet/README.md`. If the two boundaries coincide because the number was typed to make them
coincide, the pairing establishes nothing; they have to agree because the model is accurate.

Run the capacity pairs on the card their `machine:` names, or not at all. Each pair's `pair.env`
records which card it expects, and `run.sh` refuses to run a pair whose card is not the one present.

## Adding a pair

One directory under `pairs/`, containing:

- `pair.env` — `TITLE`, `VX_SOURCE`, `E_CODE`, `EXPECT_CUDA` (`runtime-failure` or `compile-error`),
  and `EXPECT_GPU` when the case needs a particular card.
- `cuda.cu` — the same computation, written the way someone would write it.
- `notes.md` — what the mistake is, and what each toolchain does about it.

`VX_SOURCE` points at the existing test rather than copying it. The Vx side of this suite is the
corpus that already exists; duplicating a program here would let the two drift.
