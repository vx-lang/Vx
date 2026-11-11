------------------------------- MODULE Agent -------------------------------
(***************************************************************************)
(* Hosts that send work to remote workers, in runtime/vx_remote_client.h   *)
(* (the host) and runtime/vx_worker_main.cpp (the worker).                 *)
(*                                                                         *)
(* A host opens one connection to each worker. Two names in the program    *)
(* can name the same worker, and they share its connection. On each        *)
(* connection the host sends a request and waits for its reply. For each   *)
(* worker, the host sends TRANSFER through one name, DISPATCH through the  *)
(* other name, then FETCH and FREE. The result of DISPATCH stays on the    *)
(* worker. A host keeps each connection open until it ends.                *)
(*                                                                         *)
(* Host h1 uses worker w1 and then w2. Host h2 uses w2 and then w1. A      *)
(* worker serves the requests of its connections in turn. Each connection  *)
(* has its own handles. When a host closes a connection, the worker frees  *)
(* each buffer of that connection.                                         *)
(*                                                                         *)
(* `Bug` names one mistake to put into the protocol. run_tlc.py checks     *)
(* that TLC finds each one. "none" is the protocol of the runtime.         *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS
    StopEarly,   \* TRUE: a host can stop between two requests, with no FREE
    WorkerDies,  \* TRUE: a worker can stop, or its machine or link can be lost
    Bug

Hosts == {"h1", "h2"}
Workers == {"w1", "w2"}
Order == [h \in Hosts |-> IF h = "h1" THEN <<"w1", "w2">> ELSE <<"w2", "w1">>]
\* The names of a worker in a host program: a memory and a device.
Names == {"mem", "dev"}

Ops == IF Bug = "free_before_the_fetch" THEN <<"transfer", "dispatch", "free", "fetch">>
                                        ELSE <<"transfer", "dispatch", "fetch", "free">>
NameOf(op) == IF op = "transfer" THEN "mem" ELSE "dev"
Steps == Len(Order["h1"]) * Len(Ops)
OpAt(i) == Ops[((i - 1) % Len(Ops)) + 1]
WorkerAt(h, i) == Order[h][((i - 1) \div Len(Ops)) + 1]

\* The pool of the host finds a connection by the address of the worker, not by the name.
ConnOf(h, w, n) == IF Bug = "one_connection_per_name" THEN <<h, w, n>> ELSE <<h, w, "any">>
Conns == {ConnOf(h, w, n) : h \in Hosts, w \in Workers, n \in Names}
HostOf(c) == c[1]
WorkerOf(c) == c[2]
\* No connection: a host that waits for no worker, or a buffer with no owner.
NoConn == <<"none", "none", "none">>
NoMsg == [op |-> "none", handle |-> 0, owner |-> NoConn]
\* The step of a host that ended, and of a host that stopped with an error.
DONE == 100
ERROR == 101

VARIABLES
    hpc,       \* the next step of each host, or DONE or ERROR
    waitsOn,   \* the connection whose reply each host waits for, or NoConn
    staged,    \* the handle of the buffer that each host staged last
    conn,      \* each connection: "none", "open", "closed" or "gone"
    req,       \* the request on each connection
    rep,       \* the reply on each connection
    holder,    \* the connection that each worker serves until it closes. Only for the bug.
    live,      \* the buffers on the workers, as <<connection, handle>>
    next,      \* the next handle of each connection. A handle is never used again.
    alive,     \* the workers whose process runs
    bad        \* a FETCH got no buffer or a buffer of another connection, or a DISPATCH was refused

vars == <<hpc, waitsOn, staged, conn, req, rep, holder, live, next, alive, bad>>

Init ==
    /\ hpc = [h \in Hosts |-> 1]
    /\ waitsOn = [h \in Hosts |-> NoConn]
    /\ staged = [h \in Hosts |-> 0]
    /\ conn = [c \in Conns |-> "none"]
    /\ req = [c \in Conns |-> NoMsg]
    /\ rep = [c \in Conns |-> NoMsg]
    /\ holder = [w \in Workers |-> NoConn]
    /\ live = {}
    /\ next = [c \in Conns |-> 1]
    /\ alive = Workers
    /\ bad = {}

-----------------------------------------------------------------------------
(* A host.                                                                 *)

Working(h) == hpc[h] \in 1..Steps /\ waitsOn[h] = NoConn
Cur(h) == ConnOf(h, WorkerAt(h, hpc[h]), NameOf(OpAt(hpc[h])))

\* connect() returns when the queue of the worker holds the connection.
Connect(h) ==
    /\ Working(h)
    /\ conn[Cur(h)] = "none"
    /\ IF WorkerOf(Cur(h)) \in alive
       THEN conn' = [conn EXCEPT ![Cur(h)] = "open"] /\ UNCHANGED hpc
       ELSE hpc' = [hpc EXCEPT ![h] = ERROR] /\ UNCHANGED conn
    /\ UNCHANGED <<waitsOn, staged, req, rep, holder, live, next, alive, bad>>

Send(h) ==
    /\ Working(h)
    /\ conn[Cur(h)] = "open"
    /\ req' = [req EXCEPT ![Cur(h)] = [op |-> OpAt(hpc[h]), handle |-> staged[h], owner |-> NoConn]]
    /\ waitsOn' = [waitsOn EXCEPT ![h] = Cur(h)]
    /\ UNCHANGED <<hpc, staged, conn, rep, holder, live, next, alive, bad>>

Receive(h) ==
    /\ waitsOn[h] # NoConn
    /\ rep[waitsOn[h]] # NoMsg
    /\ LET c == waitsOn[h]
           r == rep[c] IN
       /\ staged' = IF r.op = "transfer" THEN [staged EXCEPT ![h] = r.handle] ELSE staged
       /\ bad' = bad \cup (IF r.op = "fetch" /\ r.owner # c THEN {"fetch"} ELSE {})
                     \cup (IF r.op = "dispatch" /\ r.handle = 0 THEN {"dispatch"} ELSE {})
       /\ rep' = [rep EXCEPT ![c] = NoMsg]
    /\ waitsOn' = [waitsOn EXCEPT ![h] = NoConn]
    /\ hpc' = [hpc EXCEPT ![h] = hpc[h] + 1]
    /\ UNCHANGED <<conn, req, holder, live, next, alive>>

\* The worker stopped. The host gets EOF, or the link time limit for a lost machine or
\* link, and stops with an error.
GetEof(h) ==
    /\ waitsOn[h] # NoConn
    /\ WorkerOf(waitsOn[h]) \notin alive
    /\ Bug # "a_host_waits_after_eof"
    /\ waitsOn' = [waitsOn EXCEPT ![h] = NoConn]
    /\ hpc' = [hpc EXCEPT ![h] = ERROR]
    /\ UNCHANGED <<staged, conn, req, rep, holder, live, next, alive, bad>>

Close(h) ==
    /\ conn' = [c \in Conns |-> IF HostOf(c) = h /\ conn[c] = "open" THEN "closed" ELSE conn[c]]
    /\ hpc' = [hpc EXCEPT ![h] = DONE]
    /\ UNCHANGED <<waitsOn, staged, req, rep, holder, live, next, alive, bad>>

Finish(h) == hpc[h] = Steps + 1 /\ Close(h)

Quit(h) == StopEarly /\ Working(h) /\ Close(h)

-----------------------------------------------------------------------------
(* A worker.                                                               *)

\* The old worker served one connection until the host closed it.
MayServe(c) ==
    \/ Bug # "one_host_at_a_time"
    \/ holder[WorkerOf(c)] = c
    \/ holder[WorkerOf(c)] = NoConn

Hold(c) == IF Bug = "one_host_at_a_time" THEN [holder EXCEPT ![WorkerOf(c)] = c] ELSE holder

\* The buffers that a handle names on connection c.
Named(c, x) == IF Bug = "one_table_for_every_host"
               THEN {b \in live : b[2] = x /\ WorkerOf(b[1]) = WorkerOf(c)}
               ELSE {b \in live : b = <<c, x>>}

Serve(c) ==
    /\ WorkerOf(c) \in alive
    /\ conn[c] \in {"open", "closed"}
    /\ req[c] # NoMsg
    /\ MayServe(c)
    /\ holder' = Hold(c)
    /\ LET m == req[c] IN
       CASE m.op = "transfer" ->
              /\ live' = live \cup {<<c, next[c]>>}
              /\ next' = [next EXCEPT ![c] = next[c] + 1]
              /\ rep' = [rep EXCEPT ![c] = [op |-> "transfer", handle |-> next[c], owner |-> c]]
         [] m.op = "dispatch" ->
              \* The kernel makes a result buffer. A handle that names no buffer is refused.
              IF Named(c, m.handle) # {}
              THEN /\ live' = live \cup {<<c, next[c]>>}
                   /\ next' = [next EXCEPT ![c] = next[c] + 1]
                   /\ rep' = [rep EXCEPT ![c] = [op |-> "dispatch", handle |-> next[c], owner |-> c]]
              ELSE /\ rep' = [rep EXCEPT ![c] = [op |-> "dispatch", handle |-> 0, owner |-> NoConn]]
                   /\ UNCHANGED <<live, next>>
         [] m.op = "fetch" ->
              /\ IF Named(c, m.handle) = {}
                 THEN rep' = [rep EXCEPT ![c] = [op |-> "fetch", handle |-> 0, owner |-> NoConn]]
                 ELSE \E b \in Named(c, m.handle) :
                        rep' = [rep EXCEPT ![c] = [op |-> "fetch", handle |-> b[2], owner |-> b[1]]]
              /\ UNCHANGED <<live, next>>
         [] m.op = "free" ->
              /\ live' = live \ Named(c, m.handle)
              /\ rep' = [rep EXCEPT ![c] = [op |-> "free", handle |-> 0, owner |-> NoConn]]
              /\ UNCHANGED next
    /\ req' = [req EXCEPT ![c] = NoMsg]
    /\ UNCHANGED <<hpc, waitsOn, staged, conn, alive, bad>>

\* The host closed the connection. The worker frees each buffer of it.
Eof(c) ==
    /\ WorkerOf(c) \in alive
    /\ conn[c] = "closed"
    /\ req[c] = NoMsg
    /\ MayServe(c)
    /\ conn' = [conn EXCEPT ![c] = "gone"]
    /\ live' = IF Bug = "no_free_after_a_host_leaves" THEN live ELSE {b \in live : b[1] # c}
    /\ holder' = IF holder[WorkerOf(c)] = c THEN [holder EXCEPT ![WorkerOf(c)] = NoConn] ELSE holder
    /\ UNCHANGED <<hpc, waitsOn, staged, req, rep, next, alive, bad>>

Die(w) ==
    /\ WorkerDies
    /\ w \in alive
    /\ alive' = alive \ {w}
    /\ live' = {b \in live : WorkerOf(b[1]) # w}
    /\ UNCHANGED <<hpc, waitsOn, staged, conn, req, rep, holder, next, bad>>

-----------------------------------------------------------------------------

\* A host that waits forever is a state with no step. EachHostEnds reports it.
Next ==
    \/ \E h \in Hosts : Connect(h) \/ Send(h) \/ Receive(h) \/ GetEof(h) \/ Finish(h) \/ Quit(h)
    \/ \E c \in Conns : Serve(c) \/ Eof(c)
    \/ \E w \in Workers : Die(w)
    \/ UNCHANGED vars

Spec ==
    /\ Init /\ [][Next]_vars
    /\ \A h \in Hosts : WF_vars(Connect(h)) /\ WF_vars(Send(h)) /\ WF_vars(Receive(h))
                        /\ WF_vars(GetEof(h)) /\ WF_vars(Finish(h))
    /\ \A c \in Conns : WF_vars(Serve(c)) /\ WF_vars(Eof(c))

-----------------------------------------------------------------------------
(* The claims.                                                             *)

\* Each host ends, or stops with an error when a worker stops.
EachHostEnds == <>(\A h \in Hosts : hpc[h] \in {DONE, ERROR})

\* No buffer stays on a worker after its connection closed and the worker saw EOF.
NoBufferOutlivesItsHost == \A b \in live : conn[b[1]] # "gone"

\* A FETCH gets a buffer of its own connection, and a DISPATCH finds its buffer.
EachRequestFindsItsBuffer == bad = {}

=============================================================================
