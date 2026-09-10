# CGO 2027 E1 — running the sweep on a many-core host

Scaffolding for #295/#296. Eval-branch only.

```bash
# 1. provision the box (on the instance, as root). No repo URL: deps only.
sudo ./ec2_setup.sh

# 2. send the code (from your machine, not the instance)
utils/cgo/push.sh -i ~/.ssh/vx.pem ubuntu@<ip> --build

# 3. run it (on the instance)
cd /opt/vx && sudo ./utils/cgo/run_e1.sh

# 4. bring the results home (from your machine)
utils/cgo/push.sh -i ~/.ssh/vx.pem ubuntu@<ip> --pull
```

`run_e1.sh` needs root for the frequency governor, turbo, and `perf lock -b`. Results land in
`utils/cgo/results/<timestamp>/`: raw CSV, per-cell stderr, lock profiles, and `env.txt` recording
the machine, kernel, commit, governor, and turbo state.

## Getting the code there without giving the box a credential

`push.sh` copies the working tree over the EC2 keypair you already use for `ssh`. **No personal SSH
key, deploy key or token is ever copied to the instance** — which matters because that instance is
rented, shared-tenant hardware that gets terminated, and a key that has been on it should be treated
as disclosed. `ec2_setup.sh --clone <https-url>` still exists for a public HTTPS clone, but that is
the only keyless form of it; do not reach for `git clone git@github.com:` there.

What crosses is exactly `git ls-files`, minus `config.local`. That means:

- `target/` (tens of gigabytes) and `.rustup/` (a macOS-arch toolchain, useless on x86 Linux) are
  excluded because they are ignored, not because of a hand-written exclude list that drifts;
- **uncommitted edits are included**, since iterating against a remote box is the normal case, and
  `PUSHED_FROM` on the instance records the commit *and* whether the tree was dirty, so a result is
  always traceable to a state;
- `config.local` is dropped deliberately — it hardcodes this machine's Homebrew LLVM and an in-repo
  toolchain. `push.sh` writes a Linux one on the instance so `source config.local` works there too.

Pushing **replaces** the sources rather than merging, so a file deleted locally stops compiling
remotely. `target/` and `utils/cgo/results/` survive.

Transfer is `tar` over `ssh`, not `rsync`: the tree is ~1.6 MB compressed, so incremental sync buys
nothing worth a portability question — and there is a real one, since macOS 15 ships openrsync as
`/usr/bin/rsync` with a different flag set.

## Instance choice

Take a **bare-metal** instance — `c7i.metal-24xl` (96 physical cores) or larger. Not for the core
count alone: on a virtualised instance `scaling_governor` and `intel_pstate/no_turbo` are not
writable, so the clock moves underneath the run and appears as scaling noise. `run_e1.sh` warns and
continues rather than failing, and records the state in `env.txt` — a ladder measured under an
unpinned clock is still worth having, provided the write-up says so.

The script pins to **one CPU per physical core** (`lscpu -p=CPU,CORE`, first per core). Hyperthread
siblings share an execution unit and an L1; two rayon workers on one core measure SMT rather than
scaling, and mutex-contention questions are exactly the kind that sibling placement distorts.

## What this run is actually testing

Read `src/bin/corpus/README.md` first. The short version:

On 10 cores, **the interning strategy makes no measurable difference at any pressure the corpus
generator can produce** — including a cell where interning more than doubles single-thread time over
~11k distinct keys, where the two designs still finish within 1% of each other.

That is not a measurement failure. `locked`'s critical section is a hash lookup and an occasional
push — roughly 100 ns, taken about every 17 µs per thread at 8 threads. Two threads almost never
want it at once. Contention grows superlinearly in thread count, so whether `locked` breaks down is
a question about core count, and 10 cores cannot answer it.

**So this run is testing a prediction, not confirming an observation.** No measurement anywhere has
yet shown deferred interning winning. If the 64-core ladder also shows nothing, that is the result,
and the framing has to change rather than the experiment being re-run until it cooperates.

Note also that every measurement taken before `d300b6ee` is withdrawn: the frontend was building a
transfer-cost graph per function, and removing that cut wall clock by 2.35x. Anything quoted from a
run older than that describes a compiler that no longer exists.

`--intern-mode=content` (#307) removes the barrier entirely rather than making it cheap: identity
comes from a digest of the argument GIDs, so there is no arena, no deferred bit and nothing to
reconcile. `intern_bench` sweeps both modes automatically.

A third mode — a mutex-based interning baseline — lives on the `parallel-frontend-eval` branch,
which is this branch plus exactly one commit. It is what gives the comparison an *alternative
design* to attribute against rather than only "faster than our own serial self", and it is the one
piece that cannot live here: it is built out of exactly the primitives CI forbids in `src/`.

## Why `perf lock` is the load-bearing half

Wall clock cannot separate the designs at these core counts, so the question becomes direct: does
contention on the interner mutex grow with thread count? `perf lock contention -b` attributes waits
to acquiring callsites, which turns "locked got slower" into "locked got slower waiting on *this*".

Two things make that profile trustworthy:

- **`VX_PIPELINE_QUIET=1` is mandatory** and `run_e1.sh` sets it. `parse_phase` `println!`s per module
  from inside the rayon parallel-for, and `println!` takes the global stdout mutex (#306). Without
  the gate, workers serialise on stdout inside the measured region and `perf lock` reports contention
  on stdout instead of on the interner.
- **Density 0 is the control for the profile, not just for the timings.** It has the same function
  count, signature widths, and type-stream length as density 1, and no interning. Whatever contention
  appears at density 0 is not the interner and must be subtracted before reading the density-1
  profile as an interning result.

`locked` is the only mode with a lock to contend on, so a profile showing nothing is itself a
finding. Report it either way.

## Corpus location

`TMPDIR` defaults to `/dev/shm/vxbench`. The corpus is regenerated per cell and read once per rep; on
instance storage that is filesystem-cache behaviour inside the timed region, on tmpfs it is not.

## Known gap

The synthetic corpus is not the 400-module corpus used for the paper, and the two do not agree — the
synthetic one does not reproduce that corpus's reported 4.57×@8. Both should be run before anything
is written up; if they disagree on this hardware too, the difference between them is itself the
finding, and the paper cannot quote one number as "the" speedup.
