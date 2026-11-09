# Models of the runtime protocols

TLA+ models of the protocols in `stdlib/rust_core/src/ffi` and `runtime`. The TLC model checker checks each
model at a small size, and CI runs it (`.github/workflows/formal.yml`).

- `WaitWake.tla`: a task or a thread waits on a word, and a waker or a cancel ends the wait
  (`wait.rs`, and the park of a fiber in `sched.rs`). It also models the deadlock watcher. It
  checks that the watcher reports a deadlock only when there is one, and that it reports each
  deadlock that stays. Other models turn on a task under a callback, waits with a time limit, the
  main thread that runs the tasks with a seed, and waits for a socket or a helper call (`io.rs`). The watcher makes a model about ten times
  larger, so only the models with one fiber and one thread turn it on.
- `Rendezvous.tla`: the ranks of a job join for the collectives (`runtime/vx_collective.h`). Rank
  0 writes the NCCL id file, and the other ranks wait for it. It checks that no rank reads half an
  id, that an old id file stops rank 0, and that each rank joins or stops with an error. It
  assumes that NCCL stops with an error when its ranks can never meet.
- `Agent.tla`: hosts send work to a remote worker (`runtime/vx_remote_client.h` and
  `runtime/vx_worker_main.cpp`). It checks that each host ends, that a worker holds only the
  buffers of the host that it serves, and that a FETCH finds its buffer. A worker process that
  stops gives each host EOF. The model does not cover a lost machine or link, which gives no EOF.

`run_tlc.py` runs each model. Then it puts each mistake of the list of the model into it, and
checks that TLC reports the claim that the mistake breaks.

```sh
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar
TLA2TOOLS=$PWD/tla2tools.jar python3 formal/runtime/run_tlc.py
```

The snapshot of #2360, which finds a deadlock across machines, has no code yet. Its model comes
with that code.
