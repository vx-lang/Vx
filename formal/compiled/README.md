# Checking the compiled library with GenMC

The tests in `formal/memory_model` check a model of the library. The tests here check the LLVM
IR that vxc generates for the `Mutex`, the `Semaphore`, the channel and `Arc`. So they also find
a bug in the lowering of atomics. Tracking issue: #2170.

[GenMC](https://github.com/MPI-SWS/genmc) is a model checker for LLVM IR. It runs a small program
in every execution that RC11 allows. It reports a failed assertion, a data race, an access to
freed memory, and a thread that waits forever.

| File | What it is |
| --- | --- |
| `mutex_three_tasks.vx` | Three tasks lock a `Mutex` and add 1 to its value. |
| `semaphore_three_tasks.vx` | Three tasks take the one permit of a `Semaphore` and add 1 to a plain value. |
| `channel_keeps_the_order_of_one_sender.vx` | One task sends 1 and 2, and the values arrive in order. |
| `arc_two_holders.vx` | Two tasks read through an `Arc` and drop it. The last drop writes to the value. |
| `harness.c` | The runtime functions that the library calls, written for GenMC. |
| `run_genmc.py` | Compiles each client, links it with the harness, and runs GenMC. |
| `genmc.patch` | Two small additions to GenMC that the IR of the channel needs. |

## What is real and what is a stub

Real: the IR that vxc generates for each client and for the library: `lock`, the drop of the
guard, `acquire`, the drop of the `Permit`, `send`, `recv`, `clone` and the drop of `Arc`, with
the atomic orders that vxc chose.

Stubs, in `harness.c`:

- A `spawn` is a `pthread_create`, and the end of a `tasks` block joins each thread.
- `wait_while`, `wake_all` and `wake_one` work as the Rust runtime does: a lock and a list of the
  threads that sleep. A wait sleeps until a wake takes it off the list. `wake_all` takes each
  thread, and `wake_one` takes the thread that waits the longest on the word. With
  `--check-liveness`, GenMC reports a thread that sleeps forever, so a lost wake is an error.
- No block is cancelled, and the clock does not move. So the paths for a cancel and a deadline
  are not checked. An `acquire` that gives up must wake the next waiter. The model check in
  `tests/integration_test/library_model_check.rs` and the test
  `a_semaphore_acquire_that_gives_up_wakes_the_next_waiter.vx` check that.
- `malloc`, `free` and `llvm.memset` go to the versions of GenMC.

## Changes to the IR before GenMC reads it

GenMC cannot read IR that loads or stores a whole struct, so the script optimizes the IR first:

- A `-O0` compile is optimized with the LLVM passes of `-O1`.
- A `-O3` compile is optimized as vxc does it at `-O3`, without the SLP vectorizer. The SLP
  vectorizer joins two plain 32-bit loads into one load, and GenMC cannot check accesses of mixed
  sizes. An LLVM pass must keep the order of each atomic access.

## Results

| Client | Executions at each level | Time |
| --- | --- | --- |
| `mutex_three_tasks.vx` | 33678 | 21 s |
| `semaphore_three_tasks.vx` | 14160, and 490 blocked | 15 s |
| `channel_keeps_the_order_of_one_sender.vx` | 15537 | 10 s |
| `arc_two_holders.vx` | 4 | < 1 s |

GenMC stops an execution as blocked when a loop runs again and nothing changed. In the semaphore,
this is probably a task that wakes, finds no free permit and sleeps again.

A channel with two senders did not finish in 20 minutes, with room for 1 value or for 2.

Each of these changes to a copy of the library made the check fail:

- `lock` reads `serving` with `Relaxed`, with `-fweak-relaxed-atomics`: a data race on the value.
  Without the flag the check passes, because a `Relaxed` read is an `Acquire` read.
- The drop of the guard adds to `serving` with `Relaxed`: a lost wake.
- The drop of the guard wakes only when two threads sleep: a lost wake.
- `acquire` takes a permit with `Relaxed`, with `-fweak-relaxed-atomics`: a data race on the value.
  Without the flag the check passes.
- The drop of a `Permit` gives the permit back with `Relaxed`: a data race on the value.
- The drop of a `Permit` wakes only when two threads sleep: a lost wake.
- The drop of a `Permit` wakes no thread: a lost wake.
- The drop of a `Permit` wakes one thread before it gives the permit back: a lost wake.
- `acquire` does not add itself to `sleepers` before it sleeps: a lost wake.
- The drop of an `Arc` takes 1 from the count with `Relaxed`: an access to freed memory.
- `send` marks its slot full with a `Relaxed` store: a data race on the slot.

## Running it

Build GenMC with the LLVM 22 of the machine (about 10 minutes with 6 jobs):

```bash
git clone https://github.com/MPI-SWS/genmc.git && cd genmc   # tested at fab5bba
git apply /path/to/Vx/formal/compiled/genmc.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/lib/llvm-22/lib/cmake \
  -DCMAKE_INSTALL_PREFIX=$HOME/genmc
cmake --build build -j 6 && cmake --install build
```

Then, from the root of the repository:

```bash
source config.local      # puts mlir-translate on PATH
python3 formal/compiled/run_genmc.py --genmc $HOME/genmc/bin/genmc
```

Each line gives the number of executions that GenMC explored. On an error, the script prints
the execution that GenMC found. `--vxc-flag=-fweak-relaxed-atomics` keeps `Relaxed` reads relaxed
(docs/lang/semantics.md, section 2.5). `--keep DIR` keeps the IR files.

CI runs these checks in `.github/workflows/genmc.yml` when the library, the lowering of atomics
or this directory changes. It keeps the build of GenMC in its cache.
