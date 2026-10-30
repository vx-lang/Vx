What I found in the model
- Waits are spins. A wait x reads x again and again while it is 0, and the lock reads serving until it gets its ticket. So no thread ever blocks in the model, and "waits" must be a defined state of a thread.
- A wait that can continue stays so until its thread moves. A done flag stays 1, and each channel slot and each lock ticket belongs to one task. So weak fairness is enough: "a thread that can step from some point on steps later".
- One memory is enough for programs with no race, by drf_sc. Every well-typed program has no race.

The plan, one PR for each part
1. An interleaving machine: one memory, with one thread moving at each step. A proof shows that its runs give the same executions as the model, for programs with no race.
2. F9 and W6:
   - When every unfinished task waits, that stays true forever, so a deadlock report is sound.
   - A scheduler on one thread stops only when each task is done or waits.
3. W7: in a fair run, a task finishes if each of its waits can continue at some time and each of its loops ends. A spin loop on the program's own atomic can still hang, as rule T7 allows.
4. A FIFO ready queue is fair, so W7 holds for a second runtime with its own scheduler. For the current runtime, which gives each task its own OS thread, "the OS is fair" is an assumption.
5. M3: the round-robin bound for select.

Decisions for you to confirm (I recommend each one)
1. Use one memory, not weak memory.
2. Make a run a function Nat → Config.
3. Make "waits" a defined state, and keep today's lowerings.
4. Use weak fairness only.

Out of scope:
- The monitor thread for a foreign call that blocks (W8).
- The OS scheduler.
- Listing the cycle of waits in a deadlock report. The model proves only that a report is sound.

If the four decisions are fine, I'll start part 1, the interleaving machine.

1. you say use one memory. this is fine for a single machine. what about multiple memories? we need to pull in a Leslie Lamport here.
     2,3,4: agreed

Your point is right: one memory covers one machine only. I'll check what Vx already says about several memories (devices, remote machines, collectives), so the revised design matches the language.

  Ran 3 shell commands

Yes, and Vx needs this more than most languages, because it already has several memories:
- Memory spaces on one machine: host, GPU HBM and SMEM, with an explicit transfer between them. The host cannot see device memory (#1826).
- Remote workers: vx_agent and the remote routing in the runtime.
- Collectives: std::collective all-reduce and all-gather.

So I'd replace decision 1 with Lamport's model. Each memory is a process, and messages are the only link between processes. One memory then stays only as the rule inside each process.

Revised decision 1: nodes and messages
- The configuration has one memory for each node, the threads of each node, and the messages in flight.
- The steps:
  - A local step changes only its own node's memory.
  - A send puts a message in flight, and a delivery removes it at the other node.
  - transfer, a kernel launch, a remote call and each round of a collective become messages.
- Inside a node, drf_sc still applies, so each node can use one memory. This needs one new rule: no place belongs to two nodes. The memory-space types (Pinned(T, A)) already give that.
- Across nodes, happens-before is Lamport's 1978 relation: program order plus send-before-receive. This is the same hb that the model has now, with one more kind of edge.
- One machine is the case with one node. Part 1 builds the machine with nodes from the start, because adding nodes later would mean redoing the proofs.

What else to take from Lamport
1. Prove progress with the WF1 rule of TLA.
   - Here is the rule: if P stays true until Q holds, and action A stays enabled while P holds, and A makes Q true, then weak fairness of A gives "P leads to Q".
   - Our "a wait that can continue stays so" lemma is the first two conditions. So W7 becomes repeated uses of one general lemma.
   - The rule also covers messages: a message in flight stays deliverable, so weak fairness of delivery gives "each message is delivered".
2. F9 changes across nodes.
   - A node whose tasks all wait may still get a message that is in flight to it. So "every task waits" is not a deadlock. A deadlock also needs no message in flight to a waiting task.
   - No single node can see this.
   - Deadlock never stops once it is true, so the Chandy–Lamport snapshot (1985) detects it soundly. That is the theorem for the runtime's report.
3. F3, the first error, needs one owner. Nodes share no slot that a CAS could decide on. So the error slot must live on the node that runs the tasks block. A failure in a task on another node becomes a message to that node, and the CAS stays local. Then the current proof still applies.
4. Use TLA+ next to Lean, as we use herd7 today.
   - Lean proves the theorems about the language.
   - The TLC model checker checks the actual runtime protocols at small sizes: the agent protocol, all-reduce and the snapshot. These are the parts that are easy to get wrong and hard to prove in full.

Two questions that only you can answer
1. Is a node a machine, or each memory that a program reaches directly? I'd make each memory a node, so a GPU is a node, with transfer as messages. Then one model covers devices and the fleet.
2. What happens when a remote node or a link dies? In an asynchronous network, no program can tell a dead node from a slow one (Fischer–Lynch–Paterson). The choice decides what W7 promises in a fleet:
   - (a) a wait on a remote node can hang, and W7 assumes that links deliver and nodes do not crash.
   - (b) the runtime turns a timeout into an error in the task, and W7 becomes "finishes or fails". I'd pick (b), because it fits the error design (F1 to F9). It needs a design decision in the waiting-tasks note, and probably a new issue.

When you answer these two questions, I'll update #2360 with the revised decision 1, the new parts (WF1, the snapshot, the TLA+ checks) and your answers. Then I'll start part 1.

 1. memory as node is fine for proofs. it is a bit tricy as we might benefit more by reasoning each node as SCC. because within a GPU all kinds of concurrency is happening but 'external messages' are done via transfer with pointer to a memory so all ood.
  2. timeout can be an error, no problem. agree with you.

I'll read "SCC" as a strongly connected component: a group of memories that use each other directly, such as one GPU's registers, SMEM and HBM. The group is one node, and inside it any kind of concurrency can happen. Only a transfer that carries a pointer crosses a node's edge. That fits Lamport's later work on composition: each node has its own internal model, and the fleet proofs use only what the node promises at its edge.


