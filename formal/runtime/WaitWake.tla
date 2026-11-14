---------------------------- MODULE WaitWake ----------------------------
(***************************************************************************)
(* The wait and wake protocol of the task runtime, and its deadlock        *)
(* watcher, in stdlib/rust_core/src/ffi/wait.rs and sched.rs.              *)
(*                                                                         *)
(* A waiter waits until its word is 1, or until a cancel. A fiber (a task  *)
(* of the shared runtime) parks, and a thread sleeps on the condition      *)
(* variable of the bucket. A waker sets a word to 1 and then wakes the     *)
(* bucket. A cancel sets its mark and then wakes the bucket. All words     *)
(* share one bucket. A wake of a word wakes each thread of the bucket, but *)
(* only the fibers of that word. A cancel wakes every fiber.               *)
(*                                                                         *)
(* Each model can turn on one more part of the runtime:                    *)
(* - `OnC`: waiters on word "c", which no waker sets. They never end, and  *)
(*   the watcher must report the deadlock.                                 *)
(* - `OnWorker`: tasks under a callback from foreign code. They cannot     *)
(*   park, so they sleep on the thread of their worker, as a thread does.  *)
(*   The scheduler counts them as blocked, not the watcher.                *)
(* - `Timed`: waits with a time limit. The timer wakes a fiber at its      *)
(*   deadline. A thread sleeps with a time limit.                          *)
(* - `OnIo`: waiters on a socket or a helper call (io.rs). The io thread *)
(*   sets word "s" while the entry of the waiter is on its list, and then  *)
(*   wakes the bucket. A waiter that stops takes its entry off the list.   *)
(* - `Seed`: no workers. The main thread "t1" runs the ready tasks itself  *)
(*   while it waits for them to end, and a push wakes each sleeping thread.*)
(*                                                                         *)
(* `Bug` names one mistake to put into the protocol. run_tlc.py checks     *)
(* that TLC finds each one. "none" is the protocol of the runtime.         *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Fibers,         \* the tasks that park
    Threads,        \* the threads that sleep, and the tasks in `OnWorker`
    OnB,            \* the waiters that wait on word "b"
    OnC,            \* the waiters that wait on word "c". The others wait on word "a".
    OnWorker,       \* the tasks in `Threads` that wait under a callback
    OnIo,           \* the waiters that wait on word "s", which the io thread sets
    Timed,          \* the waiters whose wait has a time limit
    Seed,           \* TRUE: the main thread runs the tasks, as with VX_TASK_SEED
    SpuriousLimit,  \* the most spurious wakes of a condition variable
    Watch,          \* TRUE: the deadlock watcher runs. Without it, the model is smaller.
    SetWords,       \* TRUE: a waker sets word "a" and word "b" to 1
    Cancel,         \* TRUE: a cancel comes
    Bug

Waiters == Fibers \cup Threads
Words == {"a", "b", "c", "s"}
WordOf(w) == IF w \in OnIo THEN "s" ELSE IF w \in OnC THEN "c" ELSE IF w \in OnB THEN "b" ELSE "a"
Wakers == (IF SetWords THEN {"a", "b"} ELSE {}) \cup (IF Cancel THEN {"cancel"} ELSE {})
\* The timer thread of wait.rs. It is not a thread of the program.
Timer == IF Timed \cap Fibers # {} THEN {"timer"} ELSE {}
\* The poller thread or a helper thread of io.rs. It is not a thread of the program.
Io == IF OnIo # {} THEN {"io"} ELSE {}
\* The thread that runs the tasks with a seed. It waits for the end of each task.
Main == "t1"
PlainThreads == Threads \ OnWorker

\* The states of a fiber, as in sched.rs.
RUNNING == "running"
PARKING == "parking"
PARKED == "parked"
WOKEN == "woken"

\* The elements of a set, as a sequence.
InOrder(S) == CHOOSE s \in [1..Cardinality(S) -> S] : \A i, j \in DOMAIN s : i # j => s[i] # s[j]

VARIABLES
    word,       \* the value of each word
    cancelled,  \* the cancel mark
    lock,       \* who holds the lock of the bucket, or "none"
    fibers,     \* the list of parked fibers of the bucket
    sleepers,   \* the list of sleeping threads of the bucket
    notes,      \* the waiters that have a note for the deadlock report
    wakes,      \* the count of wakes of the bucket
    asleep,     \* WAITING of the watcher: the sleeping threads that are not workers
    blocked,    \* the tasks under a callback that the scheduler counts as blocked
    state,      \* the state of each fiber
    ready,      \* the ready queue of the scheduler
    seen,       \* the count of wakes that each thread saw before its sleep
    notified,   \* a notify of the condition variable came to the thread
    taken,      \* the fibers that a waker took from the list and did not wake yet
    pc,
    bad,        \* a check of the runtime failed: an `expect` or an `unreachable!`
    expired,    \* the waiters whose deadline came
    armed,      \* the fibers that the timer will wake at their deadline
    firing,     \* the fiber whose deadline the timer handles now
    listed,     \* the waiters whose entry is on the list of the io thread
    host,       \* with a seed, the task that the main thread runs, or "none"
    spurious,   \* the count of spurious wakes
    progressed, \* the PROGRESS count of the watcher went up since the watcher read it
    wpc,        \* the step of the watcher
    wsame,      \* the watcher saw no progress since its last look
    wparked,    \* what the watcher read from `all_parked`
    wstuck,     \* the last look of the watcher saw a stuck program
    reported    \* the watcher reported a deadlock, and the program stopped

watched == <<wpc, wsame, wparked, wstuck, reported>>
vars == <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked, state,
          ready, seen, notified, taken, pc, bad, expired, armed, firing, listed, host,
          spurious, progressed, watched>>

Init ==
    /\ word = [x \in Words |-> 0]
    /\ cancelled = FALSE
    /\ lock = "none"
    /\ fibers = {}
    /\ sleepers = {}
    /\ notes = {}
    /\ wakes = 0
    /\ asleep = 0
    /\ blocked = 0
    \* With a seed, each task starts on the queue, and the main thread runs it.
    /\ state = [f \in Fibers |-> IF Seed THEN WOKEN ELSE RUNNING]
    /\ ready = IF Seed THEN InOrder(Fibers) ELSE <<>>
    /\ seen = [t \in Threads |-> 0]
    /\ notified = [t \in Threads |-> FALSE]
    /\ taken = [k \in Wakers \cup Timer \cup Io |-> {}]
    /\ pc = [w \in Waiters |-> IF Seed /\ w \in Fibers THEN "parked" ELSE "check"] @@
           [k \in Wakers |-> IF Bug = "wake_before_set" THEN "wake" ELSE "set"] @@
           [k \in Timer \cup Io |-> "set"]
    /\ bad = "none"
    /\ expired = {}
    /\ armed = {}
    /\ firing = "none"
    /\ listed = OnIo
    /\ host = "none"
    /\ spurious = 0
    /\ progressed = FALSE
    /\ wpc = "progress"
    /\ wsame = FALSE
    /\ wparked = FALSE
    /\ wstuck = FALSE
    /\ reported = FALSE

-----------------------------------------------------------------------------
(* What a waiter waits for.                                                *)

\* With a seed, the main thread waits for the end of each task.
Joins(w) == Seed /\ w = Main
WordSet(w) == IF Joins(w) THEN \A f \in Fibers : pc[f] = "done" ELSE word[WordOf(w)] = 1
\* A join does not stop at a cancel.
CancelEnds(w) == cancelled /\ ~Joins(w)
CanStop(w) == WordSet(w) \/ CancelEnds(w) \/ w \in expired

\* `wait` reads the word and the cancel mark again, under the lock.
MustWait(w) ==
    /\ Bug = "no_word_check_under_the_lock" \/ ~WordSet(w)
    /\ Bug = "no_cancel_check_under_the_lock" \/ ~CancelEnds(w)
    /\ w \notin expired

\* The parts of a wake that run under the lock of the bucket: each thread comes off the list
\* and off its count, and the wake is counted.
Asleep(s) == IF Bug = "the_old_callback_wait" THEN s ELSE s \ OnWorker
Blocked(s) == IF Bug = "the_old_callback_wait" THEN {} ELSE s \cap OnWorker
WakeTheSleepers ==
    /\ sleepers' = {}
    /\ asleep' = IF Bug = "a_woken_thread_counts_as_waiting" THEN asleep
                 ELSE asleep - Cardinality(Asleep(sleepers))
    /\ blocked' = IF Bug = "a_woken_callback_task_stays_blocked" THEN blocked
                  ELSE blocked - Cardinality(Blocked(sleepers))
    /\ wakes' = wakes + 1

\* `notify_all`: it reaches each thread that sleeps now.
NotifyAll == notified' = [t \in Threads |-> notified[t] \/ pc[t] = "asleep"]

-----------------------------------------------------------------------------
(* A waiter. The caller of the wait reads its word in a loop.             *)

\* The loop of the caller: stop when it can. The end of a task wakes the word of its join, and
\* the end of a thread is counted. Both are progress for the watcher.
Check(w) ==
    /\ pc[w] = "check"
    /\ IF CanStop(w)
         THEN /\ pc' = [pc EXCEPT ![w] =
                          IF w \in listed /\ Bug # "the_waiter_returns_before_it_unlists"
                          THEN "unlist" ELSE "done"]
              /\ progressed' = Watch
         ELSE /\ pc' = [pc EXCEPT ![w] = "lock"]
              /\ UNCHANGED progressed
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, watched>>

\* `Poller::remove` takes the entry off the list, under the lock of the list.
Unlist(w) ==
    /\ pc[w] = "unlist"
    /\ listed' = listed \ {w}
    /\ pc' = [pc EXCEPT ![w] = "done"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, progressed, watched>>

Lock(w) ==
    /\ pc[w] = "lock"
    /\ lock = "none"
    /\ lock' = w
    /\ pc' = [pc EXCEPT ![w] = "test"]
    /\ UNCHANGED <<listed, word, cancelled, fibers, sleepers, notes, wakes, asleep, blocked, state,
                   ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, progressed, watched>>

Test(w) ==
    /\ pc[w] = "test"
    /\ ~MustWait(w)
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![w] = "check"]
    /\ UNCHANGED <<listed, word, cancelled, fibers, sleepers, notes, wakes, asleep, blocked, state,
                   ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, progressed, watched>>

\* A fiber marks itself `PARKING`, goes on the list, and lets the lock go. With a time limit,
\* it asks the timer for a wake at its deadline.
Enlist(f) ==
    /\ pc[f] = "test"
    /\ MustWait(f)
    /\ state' = IF Bug = "parking_after_the_unlock" THEN state
                ELSE [state EXCEPT ![f] = PARKING]
    /\ fibers' = fibers \cup {f}
    /\ notes' = IF Bug = "a_fiber_with_no_note" THEN notes ELSE notes \cup {f}
    /\ armed' = IF f \in Timed THEN armed \cup {f} ELSE armed
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![f] = IF Bug = "parking_after_the_unlock" THEN "mark" ELSE "switch"]
    /\ UNCHANGED <<listed, word, cancelled, sleepers, wakes, asleep, blocked, ready, seen, notified,
                   taken, bad, expired, firing, host, spurious, progressed, watched>>

MarkLate(f) ==
    /\ pc[f] = "mark"
    /\ state' = [state EXCEPT ![f] = PARKING]
    /\ pc' = [pc EXCEPT ![f] = "switch"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, progressed, watched>>

\* The worker, after the switch away from the fiber. A wake before the switch left the fiber
\* `WOKEN`, and the worker puts it on the queue.
Switch(f) ==
    /\ pc[f] = "switch"
    /\ IF state[f] = PARKING \/ Bug = "the_worker_does_not_look_for_woken"
         THEN /\ state' = [state EXCEPT ![f] = PARKED]
              /\ UNCHANGED ready
         ELSE /\ ready' = Append(ready, f)
              /\ UNCHANGED state
    /\ pc' = [pc EXCEPT ![f] = "parked"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   seen, notified, taken, bad, expired, armed, firing, host, spurious,
                   progressed, watched>>

\* A worker takes the fiber at the head of the queue and runs it. With a seed, there is no
\* worker.
Run(f) ==
    /\ ~Seed
    /\ ready # <<>>
    /\ Head(ready) = f
    /\ ready' = Tail(ready)
    /\ state' = [state EXCEPT ![f] = RUNNING]
    /\ pc' = [pc EXCEPT ![f] = "check"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   seen, notified, taken, bad, expired, armed, firing, host, spurious,
                   progressed, watched>>

\* With a seed, the main thread sees a ready task under the lock, lets the lock go, and runs
\* the task until it parks or ends.
RunHere(t) ==
    /\ Seed /\ t = Main
    /\ pc[t] = "test"
    /\ MustWait(t)
    /\ ready # <<>>
    /\ lock' = "none"
    /\ host' = Head(ready)
    /\ ready' = Tail(ready)
    /\ state' = [state EXCEPT ![Head(ready)] = RUNNING]
    /\ pc' = [pc EXCEPT ![t] = "host", ![Head(ready)] = "check"]
    /\ UNCHANGED <<listed, word, cancelled, fibers, sleepers, notes, wakes, asleep, blocked, seen,
                   notified, taken, bad, expired, armed, firing, spurious, progressed,
                   watched>>

HostDone(t) ==
    /\ pc[t] = "host"
    /\ pc[host] \in {"parked", "done"}
    /\ host' = "none"
    /\ pc' = [pc EXCEPT ![t] = "check"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing,
                   spurious, progressed, watched>>

\* A thread goes on the list and sleeps. The wait of the condition variable lets the lock go
\* in the same step. A thread of the program counts in WAITING, and a task under a callback
\* counts as blocked in its scheduler. A sleep with a time limit is on no list and no count.
Sleep(t) ==
    /\ pc[t] = "test"
    /\ MustWait(t)
    /\ ~(Seed /\ t = Main /\ ready # <<>>)
    /\ seen' = [seen EXCEPT ![t] = wakes]
    /\ IF t \in Timed
         THEN UNCHANGED <<listed, sleepers, notes, asleep, blocked>>
         ELSE /\ sleepers' = sleepers \cup {t}
              /\ notes' = notes \cup {t}
              /\ asleep' = asleep + Cardinality(Asleep({t}))
              /\ blocked' = blocked + Cardinality(Blocked({t}))
    /\ notified' = [notified EXCEPT ![t] = FALSE]
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![t] = "asleep"]
    /\ UNCHANGED <<listed, word, cancelled, fibers, wakes, state, ready, taken, bad, expired, armed,
                   firing, host, spurious, progressed, watched>>

\* The condition variable wakes the thread after a notify.
WakeUp(t) ==
    /\ pc[t] = "asleep"
    /\ notified[t]
    /\ pc' = [pc EXCEPT ![t] = "relock"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, progressed, watched>>

\* The time limit of a sleep ends it.
Timeout(t) ==
    /\ pc[t] = "asleep"
    /\ t \in Timed
    /\ expired' = expired \cup {t}
    /\ pc' = [pc EXCEPT ![t] = "relock"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, armed, firing, host, spurious,
                   progressed, watched>>

\* A condition variable can also wake a thread with no notify.
Spurious(t) ==
    /\ ~reported
    /\ pc[t] = "asleep"
    /\ spurious < SpuriousLimit
    /\ spurious' = spurious + 1
    /\ pc' = [pc EXCEPT ![t] = "relock"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing, host,
                   progressed, watched>>

\* The thread holds the lock again. With no wake of the bucket since its sleep, it takes
\* itself off the list and off its count, and it then lets the lock go.
Relock(t) ==
    /\ pc[t] = "relock"
    /\ lock = "none"
    /\ IF t \in Timed
         THEN UNCHANGED <<listed, sleepers, notes, asleep, blocked, bad, progressed>>
       ELSE IF wakes = seen[t]
         THEN /\ sleepers' = sleepers \ {t}
              /\ notes' = notes \ {t}
              /\ asleep' = asleep - Cardinality(Asleep({t}))
              /\ blocked' = blocked - Cardinality(Blocked({t}))
              /\ progressed' = Watch
              /\ bad' = IF t \in notes /\ t \in sleepers THEN bad
                        ELSE "a waiter has a note in its bucket"
         ELSE /\ asleep' = asleep - Cardinality(Asleep({t}) \cap
                    (IF Bug \in {"a_woken_thread_counts_again",
                                 "a_woken_thread_counts_as_waiting"} THEN {t} ELSE {}))
              /\ blocked' = blocked - Cardinality(Blocked({t}) \cap
                    (IF Bug = "a_woken_callback_task_stays_blocked" THEN {t} ELSE {}))
              /\ UNCHANGED <<listed, sleepers, notes, bad, progressed>>
    /\ pc' = [pc EXCEPT ![t] = "check"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, wakes, state, ready, seen, notified, taken,
                   expired, armed, firing, host, spurious, watched>>

-----------------------------------------------------------------------------
(* A waker of a word, or the cancel. It changes the word or the mark, and  *)
(* then wakes the bucket. The timer runs the same wake at a deadline.      *)

SetWord(k) ==
    /\ pc[k] = "set"
    /\ k \in Wakers
    /\ IF k = "cancel"
         THEN /\ cancelled' = TRUE
              /\ UNCHANGED word
         ELSE /\ word' = [word EXCEPT ![k] = 1]
              /\ UNCHANGED cancelled
    /\ pc' = [pc EXCEPT ![k] = IF Bug = "wake_before_set" THEN "done" ELSE "wake"]
    /\ UNCHANGED <<listed, lock, fibers, sleepers, notes, wakes, asleep, blocked, state, ready, seen,
                   notified, taken, bad, expired, armed, firing, host, spurious, progressed,
                   watched>>

\* The deadline of a fiber comes, and the timer wakes the word of the fiber.
Fire(k) ==
    /\ pc[k] = "set"
    /\ k \in Timer
    /\ \E f \in armed :
         /\ armed' = armed \ {f}
         /\ expired' = expired \cup {f}
         /\ firing' = f
    /\ pc' = [pc EXCEPT ![k] = "wake"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, host, spurious, progressed,
                   watched>>

\* The socket of a waiter is ready, or its helper call ended. Under the lock of its list, the io
\* thread takes the entry off the list and sets the word. A word of a wait that returned is gone.
IoTake(k) ==
    /\ pc[k] = "set"
    /\ k \in Io
    /\ \E w \in listed :
         /\ listed' = listed \ {w}
         /\ word' = [word EXCEPT !["s"] = 1]
         /\ bad' = IF pc[w] = "done" THEN "the io thread set the word of a wait that returned"
                   ELSE bad
    /\ pc' = [pc EXCEPT ![k] = "wake"]
    /\ UNCHANGED <<cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked, state,
                   ready, seen, notified, taken, expired, armed, firing, host, spurious,
                   progressed, watched>>

\* The fibers that the wake of `k` takes: the fibers of the word, or every fiber for a cancel.
FibersOf(k) ==
    CASE k = "cancel" -> fibers
      [] k = "timer" -> {f \in fibers : WordOf(f) = WordOf(firing)}
      [] k = "io" -> {f \in fibers : WordOf(f) = "s"}
      [] OTHER -> {f \in fibers : WordOf(f) = k}

\* `wake` takes the threads and the fibers of `k` off the lists, under the lock of the
\* bucket, with their notes. It counts the wake, and it is progress for the watcher.
Wake(k) ==
    /\ pc[k] = "wake"
    /\ lock = "none"
    /\ taken' = [taken EXCEPT ![k] = FibersOf(k)]
    /\ fibers' = fibers \ FibersOf(k)
    /\ notes' = IF Bug = "a_wake_clears_every_note" THEN {}
                ELSE notes \ (FibersOf(k) \cup sleepers)
    /\ WakeTheSleepers
    /\ progressed' = Watch
    /\ pc' = [pc EXCEPT ![k] = "notify"]
    /\ UNCHANGED <<listed, word, cancelled, lock, state, ready, seen, notified, bad, expired, armed,
                   firing, host, spurious, watched>>

\* `notify_all`, after the lock is let go.
Notify(k) ==
    /\ pc[k] = "notify"
    /\ NotifyAll
    /\ pc' = [pc EXCEPT ![k] = "wake_fibers"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, taken, bad, expired, armed, firing, host, spurious,
                   progressed, watched>>

\* `sched::wake` for one fiber that the waker took. With a seed, a push also wakes each
\* sleeping thread, because the main thread may sleep on any word.
WakeFiber(k) ==
    /\ pc[k] = "wake_fibers"
    /\ \E f \in taken[k] :
         /\ taken' = [taken EXCEPT ![k] = @ \ {f}]
         /\ state' = [state EXCEPT ![f] = WOKEN]
         /\ CASE state[f] = PARKED ->
                   /\ ready' = Append(ready, f)
                   /\ pc' = [pc EXCEPT ![k] =
                               IF Seed /\ Bug # "a_push_does_not_wake_the_main_thread"
                               THEN "push_wake" ELSE "wake_fibers"]
                   /\ UNCHANGED bad
              [] state[f] = PARKING ->
                   /\ ready' = IF Bug = "a_wake_of_a_parking_fiber_queues_it"
                               THEN Append(ready, f) ELSE ready
                   /\ UNCHANGED <<listed, bad, pc>>
              [] OTHER ->
                   /\ UNCHANGED <<listed, ready, pc>>
                   /\ bad' = "a wake of a fiber in state " \o state[f]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   seen, notified, expired, armed, firing, host, spurious, progressed,
                   watched>>

\* `wake_every_sleeper` after a push with a seed: it takes no fiber.
PushWake(k) ==
    /\ pc[k] = "push_wake"
    /\ lock = "none"
    /\ notes' = notes \ sleepers
    /\ WakeTheSleepers
    /\ NotifyAll
    /\ progressed' = Watch
    /\ pc' = [pc EXCEPT ![k] = "wake_fibers"]
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, state, ready, seen, taken, bad, expired,
                   armed, firing, host, spurious, watched>>

\* A waker is a thread of the program, and its end is progress. The timer waits for the next
\* deadline, and the io thread for the next ready socket.
WakerDone(k) ==
    /\ pc[k] = "wake_fibers"
    /\ taken[k] = {}
    /\ pc' = [pc EXCEPT ![k] = IF Bug = "wake_before_set" \/ k \in Timer \cup Io THEN "set"
                              ELSE "done"]
    /\ progressed' = Watch
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, bad, expired, armed, firing, host,
                   spurious, watched>>

-----------------------------------------------------------------------------
(* The deadlock watcher. At each tick it reads PROGRESS, then `stuck()`.   *)
(* It reports when two looks in a row saw a stuck program and the same     *)
(* PROGRESS. Then the program stops.                                       *)

\* The fibers that `running` counts: a fiber runs from `pop` to the end of `run`. A task under
\* a callback runs on its worker while it sleeps.
RunningCount ==
    Cardinality({f \in Fibers : pc[f] \notin {"parked", "done"}}) +
    Cardinality({t \in OnWorker : pc[t] # "done"})

AllParked == ready = <<>> /\ RunningCount = blocked

\* THREADS: the threads of the program that did not end. The wakers are threads too, and a
\* task under a callback is not.
ThreadCount ==
    Cardinality({t \in PlainThreads : pc[t] # "done"}) +
    Cardinality({k \in Wakers : pc[k] # "done"})

\* `timer::pending`: a deadline is still to come, or the timer took one and did not end its wake.
TimerPending ==
    \/ armed # {}
    \/ Bug # "the_timer_looks_idle_before_its_wake" /\ \E k \in Timer : pc[k] # "set"

\* `io::pending`: an entry is on the list, or the io thread took one and did not end its wake.
IoPending ==
    \/ listed # {}
    \/ Bug # "the_io_thread_looks_idle_before_its_wake" /\ \E k \in Io : pc[k] # "set"

\* The program runs while a thread or a task of it did not end.
ProgramRuns == \E x \in Waiters \cup Wakers : pc[x] # "done"

LookAtProgress ==
    /\ Watch
    /\ wpc = "progress"
    /\ ProgramRuns
    /\ wsame' = ~progressed
    /\ progressed' = FALSE
    /\ wpc' = "parked"
    /\ UNCHANGED <<listed, wparked, wstuck, reported>>

LookAtParked ==
    /\ wpc = "parked"
    /\ wparked' = AllParked
    /\ wpc' = "counts"
    /\ UNCHANGED <<listed, wsame, wstuck, reported, progressed>>

LookAtCounts ==
    /\ wpc = "counts"
    /\ LET stuck == /\ wparked
                    /\ asleep = ThreadCount
                    /\ Bug = "the_watcher_does_not_look_at_the_timer" \/ ~TimerPending
                    /\ Bug = "the_watcher_does_not_look_at_io" \/ ~IoPending
       IN /\ reported' = (stuck /\ (Bug = "the_watcher_reports_after_one_look"
                                      \/ (wstuck /\ wsame)))
          /\ wstuck' = stuck
    /\ wpc' = "progress"
    /\ UNCHANGED <<listed, wsame, wparked, progressed>>

WatcherStep ==
    /\ ~reported
    /\ LookAtProgress \/ LookAtParked \/ LookAtCounts
    /\ UNCHANGED <<listed, word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, blocked,
                   state, ready, seen, notified, taken, pc, bad, expired, armed, firing,
                   host, spurious>>

-----------------------------------------------------------------------------

WaiterStep(w) ==
    /\ ~reported
    /\ \/ Check(w) \/ Unlist(w) \/ Lock(w) \/ Test(w)
       \/ (w \in Fibers /\ (Enlist(w) \/ MarkLate(w) \/ Switch(w) \/ Run(w)))
       \/ (w \in Threads /\ (Sleep(w) \/ WakeUp(w) \/ Timeout(w) \/ Relock(w)
                             \/ RunHere(w) \/ HostDone(w)))

WakerStep(k) ==
    /\ ~reported
    /\ \/ SetWord(k) \/ Fire(k) \/ IoTake(k) \/ Wake(k) \/ Notify(k) \/ WakeFiber(k) \/ PushWake(k)
       \/ WakerDone(k)

Next ==
    \/ \E w \in Waiters : WaiterStep(w)
    \/ \E t \in Threads : Spurious(t)
    \/ \E k \in Wakers \cup Timer \cup Io : WakerStep(k)
    \/ WatcherStep

\* Each step but a spurious wake happens if it stays possible. The spurious wakes are few, so a
\* thread cannot keep the lock of the bucket from another one forever.
Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A w \in Waiters : WF_vars(WaiterStep(w))
    /\ \A k \in Wakers \cup Timer \cup Io : WF_vars(WakerStep(k))
    /\ WF_vars(WatcherStep)

-----------------------------------------------------------------------------
(* The claims.                                                             *)

\* No `expect` or `unreachable!` of the runtime fails.
NoFailedCheck == bad = "none"

\* The watcher counts each sleeping thread once, and the scheduler each task under a callback.
TheWatcherCountsEachSleeper ==
    /\ asleep = Cardinality(Asleep(sleepers))
    /\ blocked = Cardinality(Blocked(sleepers))

\* Each waiter on a list has a note for the deadlock report.
EachWaiterHasANote == fibers \cup sleepers \subseteq notes

\* A fiber is on the ready queue at most once, and only while it does not run.
ReadyOnce ==
    \A i, j \in 1..Len(ready) :
        /\ (i # j => ready[i] # ready[j])
        /\ state[ready[i]] # RUNNING

\* The program cannot go on: each waker ended, no deadline or socket is to come, and each waiter
\* that did not end waits for a word that stays 0, or for a task that cannot end.
TrulyStuck ==
    /\ \A k \in Wakers : pc[k] = "done"
    /\ \E w \in Waiters : pc[w] # "done"
    /\ \A w \in Waiters : pc[w] = "done" \/ ~(CanStop(w) \/ w \in Timed \cup listed)

\* The watcher reports a deadlock only when there is one.
NoFalseDeadlock == reported => TrulyStuck

\* Each wait ends, but a wait on word "c". A waiter whose word was set, or whose block was
\* cancelled, runs again.
EachWaitEnds == <>(\A w \in Waiters \ OnC : pc[w] = "done")

\* A deadlock that stays is reported, when the watcher runs.
DeadlockIsReported == Watch => (<>[]TrulyStuck => <>reported)

=============================================================================
