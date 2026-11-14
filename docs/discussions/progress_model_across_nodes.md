# A progress model across nodes that pass messages

Status: agreed on 2026-10-30. Issue: #2360. Tracking: #2170. Plan:
`formal_proofs_for_tasks_channels_and_errors.md`.

This note records the discussion that led to the design in #2360. The issue has the design. This
note has the questions, the options and the reasons for each decision.

## Where the formal work stood

By 2026-10-30, the PRs for every step of the plan in #2170 had merged:

- The footprint rule covers a loop body with code around its tasks block (#2241), tasks of
  different loop passes that share places (#2242), and a block whose code holds other blocks
  (#2243, #2285, #2286).
- The typed layer has four parts (#2244). Part 1 has locals, structs, cells and atomics. Part 2
  adds references. Part 3 adds channels and locals in a mutex. Part 4 adds raw pointers and
  `unsafe impl`, and T3 (scopes) has its own PR. A well-typed
  program follows the footprint rule, so it has no data race. The claims of `unsafe` code are a
  hypothesis of the theorems, not an axiom.

Four items stayed under "Later" in #2170: a progress model, release and acquire in the library,
GPU atomics and fences in `across` loops (#2139), and a check of the compiled library against
RC11. We chose the progress model first, because it is the largest guarantee that the proofs
lack. Waiting tasks are also open work for the concurrency goal of Vx.

## The gap

Each proof is about a finite prefix of an execution. So the model can say "no data race", but
not "finishes" or "is stuck". The rules that need more are W6, W7, F9 and M3:

- W6: a scheduler on one thread stops only when each task is done or waits.
- W7: a task finishes if each of its waits can continue.
- F9: the runtime reports a deadlock only when no task can ever continue.
- M3: `select` is round robin, and a branch that can continue is chosen within `k` calls.

## Three facts from the code

1. **A wait is a spin.** In `Thread.lean`, `wait x` reads `x` and stays at the `wait` while `x`
   is 0. A join and a channel wait on a flag this way. A lock reads `serving` in a loop until
   it is its ticket. So no thread ever blocks in the model. "Waits" has to be a defined state
   of a thread.
1. **A wait that can continue stays so until its thread moves.** A `done` flag stays 1. A
   channel flag for element `k` belongs to the one task with the number `k`. Only the holder of
   the lock adds 1 to `serving`. So weak fairness is enough, and strong fairness is not needed.
1. **The runtime has no scheduler yet.** `stdlib/rust_core/src/ffi/tasks.rs` starts one OS
   thread for each task. So for the first runtime, "the OS is fair" can only be a hypothesis.

## The first proposal: one memory

The first proposal used one memory with no weak behavior. `drf_sc` justifies it: a program with
no race has only executions in which the accesses happen one at a time, and every well-typed
program has no race.

The user agreed with three of the four decisions: a run is `Nat → Config`, "waits" is a defined
state, and weak fairness is enough. The user did not agree with one memory. One memory is fine
for one machine. A Vx program runs on a fleet, with many memories. The user asked for Lamport's
view of the problem.

## Lamport's view

Lamport's model of a distributed system has processes that share no memory and pass messages.
Vx already has several memories:

- memory spaces on one machine (host, GPU HBM, SMEM), with an explicit `transfer` between them.
  The host cannot see device memory (#1826).
- remote workers (`vx_agent` and the remote routing in the runtime).
- collectives (`std::collective`, all-reduce and all-gather).

Five ideas from Lamport's work fit:

1. **Happens-before (1978):** the order inside a process, and a send before its delivery. It is
   the `hb` of the model, with one more kind of edge.
1. **The WF1 rule of TLA.** Assume: `P` stays true until `Q` holds, an action `A` can happen
   while `P` holds, and `A` makes `Q` true. Then weak fairness of `A` gives "`P` leads to `Q`".
   Fact 2 above gives the first two conditions for each wait. A message in flight stays
   deliverable, so the same rule gives the delivery of each message.
1. **A deadlock across nodes is a stable property.** A node whose tasks all wait can still get a
   message that is in flight to it. No one node can see a deadlock. The Chandy and Lamport
   snapshot (1985) detects a stable property soundly.
1. **The first error needs one owner (F3).** Nodes share no slot for a CAS. So the error slot
   lives on the node of the `tasks` block. A failure on another node is a message to it.
1. **TLA+ next to Lean.** Lean proves the theorems of the language. The TLC model checker can
   check the protocols of the runtime at small sizes, as herd7 checks the litmus tests today.

## Two questions for the user

**What is a node?** The proposal was: each memory that a program reaches directly is a node.

The user agreed that a memory as a node is fine for proofs. The user then proposed a better
unit: a strongly connected group of memories. Inside a GPU, all kinds of concurrency happen at
once. External messages go only through `transfer`, with a pointer to a memory. So a node is a
group of memories that use each other directly, and the fleet proofs treat it as opaque. A node
has its own model inside, and the fleet proofs use only its interface: the messages that it sends
and takes, and its promise of local progress. A CPU node is one memory, through `drf_sc`. A GPU
node can use the scoped model of #2139 later.

**What happens when a remote node or a link dies?** With no reliable network, no program can
tell a dead node from a slow one (Fischer, Lynch and Paterson). The options were:

- (a) a wait on a remote node can hang, and W7 assumes that links deliver and nodes do not crash.
- (b) the runtime turns a timeout into an error in the task, and W7 becomes "finishes, or fails
  with an error".

The user agreed with (b). It fits the error design (F1 to F9).

## Decisions

1. **Nodes and messages.** A node is a strongly connected group of memories, opaque to the
   fleet proofs. A message carries a pointer and the right to use its memory. Each place has one
   owner node at each time, so no race between nodes can happen.
1. **A run is `Nat → Config`**, not a coinductive type.
1. **"Waits" is a defined state.** The lowerings stay as they are. Each construct gets one lemma:
   its wait, once it can continue, stays so until its thread moves.
1. **Weak fairness** of each local step and of each delivery.
1. **A timeout of a wait across nodes is an error in the task.**

## The plan

#2360 lists seven parts, one PR each:

1. The machine: nodes, messages, runs and weak fairness, with a CPU node made from `Thread.lean`.
1. The WF1 rule.
1. F9 and W6, with the snapshot for a deadlock across nodes.
1. W7: a task finishes, or fails with the timeout error.
1. A FIFO scheduler is fair.
1. M3, the bound of `select`.
1. TLA+ models of the protocols of the runtime, checked with TLC.
