---------------------------- MODULE WaitWake ----------------------------
(***************************************************************************)
(* The wait and wake protocol of the task runtime, in                      *)
(* stdlib/rust_core/src/ffi/wait.rs and sched.rs.                          *)
(*                                                                         *)
(* A waiter waits until its word is 1, or until a cancel. A fiber (a task  *)
(* of the shared runtime) parks, and a thread sleeps on the condition      *)
(* variable of the bucket. A waker sets a word to 1 and then wakes the     *)
(* bucket. A cancel sets its mark and then wakes the bucket. All words     *)
(* share one bucket. A wake of a word wakes each thread of the bucket, but *)
(* only the fibers of that word. A cancel wakes every fiber.               *)
(*                                                                         *)
(* `Bug` names one mistake to put into the protocol. run_tlc.py checks     *)
(* that TLC finds each one. "none" is the protocol of the runtime.         *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Fibers,     \* the tasks that park
    Threads,    \* the threads that sleep
    OnB,        \* the waiters that wait on word "b". The others wait on word "a".
    SetWords,   \* TRUE: a waker sets each word to 1
    Cancel,     \* TRUE: a cancel comes
    Bug

Waiters == Fibers \cup Threads
Words == {"a", "b"}
WordOf(w) == IF w \in OnB THEN "b" ELSE "a"
Wakers == (IF SetWords THEN Words ELSE {}) \cup (IF Cancel THEN {"cancel"} ELSE {})

\* The states of a fiber, as in sched.rs.
RUNNING == "running"
PARKING == "parking"
PARKED == "parked"
WOKEN == "woken"

VARIABLES
    word,       \* the value of each word
    cancelled,  \* the cancel mark
    lock,       \* who holds the lock of the bucket, or "none"
    fibers,     \* the list of parked fibers of the bucket
    sleepers,   \* the list of sleeping threads of the bucket
    notes,      \* the waiters that have a note for the deadlock report
    wakes,      \* the count of wakes of the bucket
    asleep,     \* the count of sleeping threads that the deadlock watcher keeps
    state,      \* the state of each fiber
    ready,      \* the ready queue of the scheduler
    seen,       \* the count of wakes that each thread saw before its sleep
    notified,   \* a notify of the condition variable came to the thread
    taken,      \* the fibers that a waker took from the list and did not wake yet
    pc,
    bad         \* a check of the runtime failed: an `expect` or an `unreachable!`

vars == <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, state,
          ready, seen, notified, taken, pc, bad>>

Init ==
    /\ word = [x \in Words |-> 0]
    /\ cancelled = FALSE
    /\ lock = "none"
    /\ fibers = {}
    /\ sleepers = {}
    /\ notes = {}
    /\ wakes = 0
    /\ asleep = 0
    /\ state = [f \in Fibers |-> RUNNING]
    /\ ready = <<>>
    /\ seen = [t \in Threads |-> 0]
    /\ notified = [t \in Threads |-> FALSE]
    /\ taken = [k \in Wakers |-> {}]
    /\ pc = [w \in Waiters |-> "check"] @@
           [k \in Wakers |-> IF Bug = "wake_before_set" THEN "wake" ELSE "set"]
    /\ bad = "none"

-----------------------------------------------------------------------------
(* A waiter. The caller of the wait reads its word in a loop.             *)

\* The loop of the caller: stop when the word is 1 or the block was cancelled.
Check(w) ==
    /\ pc[w] = "check"
    /\ pc' = [pc EXCEPT ![w] = IF word[WordOf(w)] = 1 \/ cancelled THEN "done" ELSE "lock"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep,
                   state, ready, seen, notified, taken, bad>>

Lock(w) ==
    /\ pc[w] = "lock"
    /\ lock = "none"
    /\ lock' = w
    /\ pc' = [pc EXCEPT ![w] = "test"]
    /\ UNCHANGED <<word, cancelled, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, notified, taken, bad>>

\* `wait` reads the word and the cancel mark again, under the lock.
MustWait(w) ==
    /\ Bug = "no_word_check_under_the_lock" \/ word[WordOf(w)] = 0
    /\ Bug = "no_cancel_check_under_the_lock" \/ ~cancelled

Test(w) ==
    /\ pc[w] = "test"
    /\ ~MustWait(w)
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![w] = "check"]
    /\ UNCHANGED <<word, cancelled, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, notified, taken, bad>>

\* A fiber marks itself `PARKING`, goes on the list, and lets the lock go.
Enlist(f) ==
    /\ pc[f] = "test"
    /\ MustWait(f)
    /\ state' = IF Bug = "parking_after_the_unlock" THEN state
                ELSE [state EXCEPT ![f] = PARKING]
    /\ fibers' = fibers \cup {f}
    /\ notes' = IF Bug = "a_fiber_with_no_note" THEN notes ELSE notes \cup {f}
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![f] = IF Bug = "parking_after_the_unlock" THEN "mark" ELSE "switch"]
    /\ UNCHANGED <<word, cancelled, sleepers, wakes, asleep, ready, seen, notified,
                   taken, bad>>

MarkLate(f) ==
    /\ pc[f] = "mark"
    /\ state' = [state EXCEPT ![f] = PARKING]
    /\ pc' = [pc EXCEPT ![f] = "switch"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep,
                   ready, seen, notified, taken, bad>>

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
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, seen,
                   notified, taken, bad>>

\* A worker takes the fiber at the head of the queue and runs it.
Run(f) ==
    /\ ready # <<>>
    /\ Head(ready) = f
    /\ ready' = Tail(ready)
    /\ state' = [state EXCEPT ![f] = RUNNING]
    /\ pc' = [pc EXCEPT ![f] = "check"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, seen,
                   notified, taken, bad>>

\* A thread goes on the list and sleeps. The wait of the condition variable lets the lock go
\* in the same step.
Sleep(t) ==
    /\ pc[t] = "test"
    /\ MustWait(t)
    /\ seen' = [seen EXCEPT ![t] = wakes]
    /\ sleepers' = sleepers \cup {t}
    /\ notes' = notes \cup {t}
    /\ asleep' = asleep + 1
    /\ notified' = [notified EXCEPT ![t] = FALSE]
    /\ lock' = "none"
    /\ pc' = [pc EXCEPT ![t] = "asleep"]
    /\ UNCHANGED <<word, cancelled, fibers, wakes, state, ready, taken, bad>>

\* The condition variable wakes the thread after a notify.
WakeUp(t) ==
    /\ pc[t] = "asleep"
    /\ notified[t]
    /\ pc' = [pc EXCEPT ![t] = "relock"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, notified, taken, bad>>

\* A condition variable can also wake a thread with no notify.
Spurious(t) ==
    /\ pc[t] = "asleep"
    /\ pc' = [pc EXCEPT ![t] = "relock"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, notified, taken, bad>>

\* The thread holds the lock again. With no wake of the bucket since its sleep, it takes
\* itself off the list, and it then lets the lock go.
Relock(t) ==
    /\ pc[t] = "relock"
    /\ lock = "none"
    /\ IF wakes = seen[t]
         THEN /\ sleepers' = sleepers \ {t}
              /\ notes' = notes \ {t}
              /\ asleep' = asleep - 1
              /\ bad' = IF t \in notes /\ t \in sleepers THEN bad
                        ELSE "a waiter has a note in its bucket"
         ELSE /\ asleep' = IF Bug = "a_woken_thread_counts_again" THEN asleep - 1 ELSE asleep
              /\ UNCHANGED <<sleepers, notes, bad>>
    /\ pc' = [pc EXCEPT ![t] = "check"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, wakes, state, ready, seen, notified,
                   taken>>

-----------------------------------------------------------------------------
(* A waker of a word, or the cancel. It changes the word or the mark, and  *)
(* then wakes the bucket.                                                  *)

SetWord(k) ==
    /\ pc[k] = "set"
    /\ IF k = "cancel"
         THEN /\ cancelled' = TRUE
              /\ UNCHANGED word
         ELSE /\ word' = [word EXCEPT ![k] = 1]
              /\ UNCHANGED cancelled
    /\ pc' = [pc EXCEPT ![k] = IF Bug = "wake_before_set" THEN "done" ELSE "wake"]
    /\ UNCHANGED <<lock, fibers, sleepers, notes, wakes, asleep, state, ready, seen,
                   notified, taken, bad>>

\* The fibers that the wake of `k` takes: the fibers of the word, or every fiber for a cancel.
FibersOf(k) == IF k = "cancel" THEN fibers ELSE {f \in fibers : WordOf(f) = k}

\* `wake` takes the threads and the fibers of `k` off the lists, under the lock of the
\* bucket, with their notes. It counts the wake.
Wake(k) ==
    /\ pc[k] = "wake"
    /\ lock = "none"
    /\ taken' = [taken EXCEPT ![k] = FibersOf(k)]
    /\ fibers' = fibers \ FibersOf(k)
    /\ notes' = IF Bug = "a_wake_clears_every_note" THEN {}
              ELSE notes \ (FibersOf(k) \cup sleepers)
    /\ asleep' = asleep - Cardinality(sleepers)
    /\ sleepers' = {}
    /\ wakes' = wakes + 1
    /\ pc' = [pc EXCEPT ![k] = "notify"]
    /\ UNCHANGED <<word, cancelled, lock, state, ready, seen, notified, bad>>

\* `notify_all`, after the lock is let go. It reaches each thread that sleeps now.
Notify(k) ==
    /\ pc[k] = "notify"
    /\ notified' = [t \in Threads |-> notified[t] \/ pc[t] = "asleep"]
    /\ pc' = [pc EXCEPT ![k] = "wake_fibers"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, taken, bad>>

\* `sched::wake` for one fiber that the waker took.
WakeFiber(k) ==
    /\ pc[k] = "wake_fibers"
    /\ \E f \in taken[k] :
         /\ taken' = [taken EXCEPT ![k] = @ \ {f}]
         /\ state' = [state EXCEPT ![f] = WOKEN]
         /\ CASE state[f] = PARKED ->
                   /\ ready' = Append(ready, f)
                   /\ UNCHANGED bad
              [] state[f] = PARKING ->
                   /\ ready' = IF Bug = "a_wake_of_a_parking_fiber_queues_it"
                               THEN Append(ready, f) ELSE ready
                   /\ UNCHANGED bad
              [] OTHER ->
                   /\ UNCHANGED ready
                   /\ bad' = "a wake of a fiber in state " \o state[f]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, seen,
                   notified, pc>>

WakerDone(k) ==
    /\ pc[k] = "wake_fibers"
    /\ taken[k] = {}
    /\ pc' = [pc EXCEPT ![k] = IF Bug = "wake_before_set" THEN "set" ELSE "done"]
    /\ UNCHANGED <<word, cancelled, lock, fibers, sleepers, notes, wakes, asleep, state,
                   ready, seen, notified, taken, bad>>

-----------------------------------------------------------------------------

WaiterStep(w) ==
    \/ Check(w) \/ Lock(w) \/ Test(w)
    \/ (w \in Fibers /\ (Enlist(w) \/ MarkLate(w) \/ Switch(w) \/ Run(w)))
    \/ (w \in Threads /\ (Sleep(w) \/ WakeUp(w) \/ Relock(w)))

WakerStep(k) ==
    SetWord(k) \/ Wake(k) \/ Notify(k) \/ WakeFiber(k) \/ WakerDone(k)

Next ==
    \/ \E w \in Waiters : WaiterStep(w)
    \/ \E t \in Threads : Spurious(t)
    \/ \E k \in Wakers : WakerStep(k)

\* Each step but a spurious wake happens if it stays possible.
Spec ==
    /\ Init
    /\ [][Next]_vars
    /\ \A w \in Waiters : WF_vars(WaiterStep(w))
    /\ \A k \in Wakers : WF_vars(WakerStep(k))

-----------------------------------------------------------------------------
(* The claims.                                                             *)

\* No `expect` or `unreachable!` of the runtime fails.
NoFailedCheck == bad = "none"

\* The deadlock watcher counts each sleeping thread once.
TheWatcherCountsEachSleeper == asleep = Cardinality(sleepers)

\* Each waiter on a list has a note for the deadlock report.
EachWaiterHasANote == fibers \cup sleepers \subseteq notes

\* A fiber is on the ready queue at most once, and only while it does not run.
ReadyOnce ==
    \A i, j \in 1..Len(ready) :
        /\ (i # j => ready[i] # ready[j])
        /\ state[ready[i]] # RUNNING

\* Each wait ends: a waiter whose word was set, or whose block was cancelled, runs again.
EachWaitEnds == <>(\A w \in Waiters : pc[w] = "done")

=============================================================================
