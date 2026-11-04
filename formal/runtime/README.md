# Models of the runtime protocols

TLA+ models of the protocols in `stdlib/rust_core/src/ffi`. The TLC model checker checks each
model at a small size, and CI runs it (`.github/workflows/formal.yml`).

- `WaitWake.tla`: a task or a thread waits on a word, and a waker or a cancel ends the wait
  (`wait.rs`, and the park of a fiber in `sched.rs`). It also models the deadlock watcher. It
  checks that the watcher reports a deadlock only when there is one, and that it reports each
  deadlock that stays. Other models turn on a task under a callback, waits with a time limit, and
  the main thread that runs the tasks with a seed. The watcher makes a model about ten times
  larger, so only the models with one fiber and one thread turn it on.

`run_tlc.py` runs each model. Then it puts each mistake of its `BUGS` list into the model, and
checks that TLC reports the claim that the mistake breaks.

```sh
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar
TLA2TOOLS=$PWD/tla2tools.jar python3 formal/runtime/run_tlc.py
```

The protocols of the fleet (the agent protocol, the collectives and the snapshot of #2360) have
no code yet. Their models come with that code.
