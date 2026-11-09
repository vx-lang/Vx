------------------------------- MODULE Agent -------------------------------
(***************************************************************************)
(* Hosts that send work to one remote worker, in                          *)
(* runtime/vx_remote_client.h (the host) and runtime/vx_worker_main.cpp    *)
(* (the worker).                                                           *)
(*                                                                         *)
(* A host opens one connection to each worker. Two names in the program    *)
(* can name the same worker, and they share its connection. On each        *)
(* connection the host sends a request and waits for its reply. For each   *)
(* name, the host sends TRANSFER (a buffer to the worker), DISPATCH (a     *)
(* kernel on it, which makes a result buffer), FETCH (the buffer back) and *)
(* FREE. The result buffer stays on the worker.                            *)
(*                                                                         *)
(* The worker serves one connection at a time, until the host closes it.   *)
(* Then it frees each buffer that the host left.                           *)
(*                                                                         *)
(* `Bug` names one mistake to put into the protocol. run_tlc.py checks     *)
(* that TLC finds each one. "none" is the protocol of the runtime.         *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    Hosts,
    StopEarly,   \* TRUE: a host can stop between two requests, with no FREE
    WorkerDies,  \* TRUE: the worker process can stop. Each host then gets EOF.
    Bug

\* Two names in each host program. Both name the one worker.
Names == <<"gpu", "hbm">>

Ops == IF Bug = "free_before_the_fetch" THEN <<"transfer", "dispatch", "free", "fetch">>
                                        ELSE <<"transfer", "dispatch", "fetch", "free">>
Steps == Len(Names) * Len(Ops)
OpAt(i) == Ops[((i - 1) % Len(Ops)) + 1]
NameAt(i) == Names[((i - 1) \div Len(Ops)) + 1]

\* The connection that a host uses for a name. The pool of the host finds a
\* connection by the address of the worker, not by the name.
ConnOf(h, n) == IF Bug = "one_connection_per_name" THEN <<h, n>> ELSE <<h, "worker">>
Conns == {ConnOf(h, Names[i]) : h \in Hosts, i \in 1..Len(Names)}
HostConns(h) == {c \in Conns : c[1] = h}
NoMsg == [op |-> "none", handle |-> 0]
\* No connection.
NoConn == <<"none", "none">>
\* The step of a host that ended, and of a host that stopped with an error.
DONE == 100
ERROR == 101

VARIABLES
    hpc,       \* the next step of each host, or DONE or ERROR
    waiting,   \* the hosts that wait for a reply
    staged,    \* the handle of the buffer that each host staged last
    conn,      \* each connection: "none", "open" or "closed"
    req,       \* the request on each connection
    rep,       \* the reply on each connection
    served,    \* the connections that the worker finished
    serving,   \* the connection that the worker serves, or NoConn
    live,      \* the buffers on the worker: [handle |-> the connection that made it]
    next,      \* the next handle. The worker never uses a handle again.
    alive,     \* the worker process runs
    empty      \* a FETCH got no bytes

vars == <<hpc, waiting, staged, conn, req, rep, served, serving, live, next, alive, empty>>

Init ==
    /\ hpc = [h \in Hosts |-> 1]
    /\ waiting = {}
    /\ staged = [h \in Hosts |-> 0]
    /\ conn = [c \in Conns |-> "none"]
    /\ req = [c \in Conns |-> NoMsg]
    /\ rep = [c \in Conns |-> NoMsg]
    /\ served = {}
    /\ serving = NoConn
    /\ live = <<>>
    /\ next = 1
    /\ alive = TRUE
    /\ empty = FALSE

-----------------------------------------------------------------------------
(* A host.                                                                 *)

Working(h) == hpc[h] \in 1..Steps /\ h \notin waiting
Cur(h) == ConnOf(h, NameAt(hpc[h]))

\* connect() returns when the worker's queue holds the connection, before the
\* worker accepts it.
Connect(h) ==
    /\ Working(h)
    /\ conn[Cur(h)] = "none"
    /\ IF alive THEN conn' = [conn EXCEPT ![Cur(h)] = "open"] /\ UNCHANGED hpc
                ELSE hpc' = [hpc EXCEPT ![h] = ERROR] /\ UNCHANGED conn
    /\ UNCHANGED <<waiting, staged, req, rep, served, serving, live, next, alive, empty>>

Send(h) ==
    /\ Working(h)
    /\ conn[Cur(h)] = "open"
    /\ req' = [req EXCEPT ![Cur(h)] = [op |-> OpAt(hpc[h]), handle |-> staged[h]]]
    /\ waiting' = waiting \cup {h}
    /\ UNCHANGED <<hpc, staged, conn, rep, served, serving, live, next, alive, empty>>

Receive(h) ==
    /\ h \in waiting
    /\ rep[Cur(h)] # NoMsg
    /\ LET r == rep[Cur(h)] IN
       /\ staged' = IF r.op = "transfer" THEN [staged EXCEPT ![h] = r.handle] ELSE staged
       /\ empty' = (empty \/ (r.op = "fetch" /\ r.handle = 0))
    /\ rep' = [rep EXCEPT ![Cur(h)] = NoMsg]
    /\ waiting' = waiting \ {h}
    /\ hpc' = [hpc EXCEPT ![h] = hpc[h] + 1]
    /\ UNCHANGED <<conn, req, served, serving, live, next, alive>>

\* The worker process stopped, so the host reads EOF and stops with an error.
GetEof(h) ==
    /\ h \in waiting
    /\ ~alive
    /\ Bug # "a_host_waits_after_eof"
    /\ waiting' = waiting \ {h}
    /\ hpc' = [hpc EXCEPT ![h] = ERROR]
    /\ UNCHANGED <<staged, conn, req, rep, served, serving, live, next, alive, empty>>

Close(h) ==
    /\ conn' = [c \in Conns |-> IF c \in HostConns(h) /\ conn[c] = "open" THEN "closed"
                                ELSE conn[c]]
    /\ hpc' = [hpc EXCEPT ![h] = DONE]
    /\ UNCHANGED <<waiting, staged, req, rep, served, serving, live, next, alive, empty>>

Finish(h) == hpc[h] = Steps + 1 /\ Close(h)

Quit(h) == StopEarly /\ Working(h) /\ Close(h)

-----------------------------------------------------------------------------
(* The worker.                                                             *)

Accept ==
    /\ alive
    /\ serving = NoConn
    /\ \E c \in Conns :
        /\ conn[c] # "none"
        /\ c \notin served
        /\ serving' = c
    /\ UNCHANGED <<hpc, waiting, staged, conn, req, rep, served, live, next, alive, empty>>

Live(x) == x \in DOMAIN live
Without(x) == [y \in DOMAIN live \ {x} |-> live[y]]
With(x, c) == [y \in DOMAIN live \cup {x} |-> IF y = x THEN c ELSE live[y]]

Serve ==
    /\ alive
    /\ serving # NoConn
    /\ req[serving] # NoMsg
    /\ LET c == serving
           m == req[c] IN
       CASE m.op = "transfer" ->
              /\ live' = With(next, c)
              /\ next' = next + 1
              /\ rep' = [rep EXCEPT ![c] = [op |-> "transfer", handle |-> next]]
         [] m.op = "dispatch" ->
              \* The kernel makes a result buffer. A handle that names no buffer is refused.
              /\ live' = IF Live(m.handle) THEN With(next, c) ELSE live
              /\ next' = next + 1
              /\ rep' = [rep EXCEPT ![c] = [op |-> "dispatch", handle |-> next]]
         [] m.op = "fetch" ->
              /\ rep' = [rep EXCEPT ![c] = [op |-> "fetch",
                                            handle |-> IF Live(m.handle) THEN m.handle ELSE 0]]
              /\ UNCHANGED <<live, next>>
         [] m.op = "free" ->
              /\ live' = IF Live(m.handle) THEN Without(m.handle) ELSE live
              /\ rep' = [rep EXCEPT ![c] = [op |-> "free", handle |-> 0]]
              /\ UNCHANGED next
    /\ req' = [req EXCEPT ![serving] = NoMsg]
    /\ UNCHANGED <<hpc, waiting, staged, conn, served, serving, alive, empty>>

\* The host closed the connection. The worker frees each buffer that the host left.
Eof ==
    /\ alive
    /\ serving # NoConn
    /\ conn[serving] = "closed"
    /\ req[serving] = NoMsg
    /\ live' = IF Bug = "no_free_after_a_host_leaves" THEN live ELSE <<>>
    /\ served' = served \cup {serving}
    /\ serving' = NoConn
    /\ UNCHANGED <<hpc, waiting, staged, conn, req, rep, next, alive, empty>>

Die ==
    /\ WorkerDies
    /\ alive
    /\ alive' = FALSE
    /\ live' = <<>>
    /\ serving' = NoConn
    /\ UNCHANGED <<hpc, waiting, staged, conn, req, rep, served, next, empty>>

-----------------------------------------------------------------------------

\* A host that waits forever is a state with no step. EachHostEnds reports it.
Next ==
    \/ \E h \in Hosts : Connect(h) \/ Send(h) \/ Receive(h) \/ GetEof(h) \/ Finish(h) \/ Quit(h)
    \/ Accept \/ Serve \/ Eof \/ Die
    \/ UNCHANGED vars

Spec ==
    /\ Init /\ [][Next]_vars
    /\ \A h \in Hosts : WF_vars(Connect(h)) /\ WF_vars(Send(h)) /\ WF_vars(Receive(h))
                        /\ WF_vars(GetEof(h)) /\ WF_vars(Finish(h))
    /\ WF_vars(Accept) /\ WF_vars(Serve) /\ WF_vars(Eof)

-----------------------------------------------------------------------------
(* The claims.                                                             *)

\* Each host ends, or stops with an error when the worker stops.
EachHostEnds == <>(\A h \in Hosts : hpc[h] \in {DONE, ERROR})

\* Each buffer on the worker belongs to the connection that it serves. So a host
\* never gets the memory of another, and no buffer outlives its host.
EachBufferBelongsToTheServedHost == \A x \in DOMAIN live : live[x] = serving

\* A FETCH always names a buffer that is there.
EachFetchGetsItsBytes == ~empty

=============================================================================
