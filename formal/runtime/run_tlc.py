#!/usr/bin/env python3
"""Runs the TLC model checker on the models of the runtime protocols.

Each check of each module must pass. Then each mistake in the list of the module goes into
the model, one at a time, and TLC must report the claim that the mistake breaks. A claim that
no mistake breaks may check nothing.

TLC is in tla2tools.jar. Give its path in TLA2TOOLS, or put it next to this file.
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The checks of WaitWake, each a change to the constants of WaitWake.cfg. The cancel and the second
# waker wake the bucket again, and that can hide a lost wake. So some checks have only one.
NO_CANCEL = {"Cancel": "FALSE"}
ONLY_A_CANCEL = {"SetWords": "FALSE"}
# Task f2 waits on a word that no waker sets.
A_DEADLOCK = {"OnB": "{}", "OnC": '{"f2"}', "Cancel": "FALSE"}
# Thread t1 is a task under a callback.
A_CALLBACK = {"OnWorker": '{"t1"}'}
# The task under a callback waits on a word that no waker sets.
A_DEADLOCK_IN_A_CALLBACK = {"OnWorker": '{"t1"}', "OnC": '{"t1"}', "Cancel": "FALSE"}
# Task f2 waits with a time limit on a word that no waker sets. The timer ends its wait.
A_TIMED_TASK = {"OnB": "{}", "OnC": '{"f2"}', "Timed": '{"f2"}', "Cancel": "FALSE"}
# Thread t1 does the same.
A_TIMED_THREAD = {"OnC": '{"t1"}', "Timed": '{"t1"}', "Cancel": "FALSE"}
A_SEED = {"Seed": "TRUE"}
# Task f2 waits for a socket, and only a cancel ends the wait of thread t1. The cancel can come
# before the socket is ready.
A_SOCKET = {"Fibers": '{"f2"}', "OnB": "{}", "OnIo": '{"f2"}', "SetWords": "FALSE"}
# Thread t1 waits for a helper call, and task f1 for word "a".
A_THREAD_ON_IO = {"Fibers": '{"f1"}', "OnB": "{}", "OnIo": '{"t1"}', "Cancel": "FALSE"}
# The deadlock watcher makes a model about ten times larger. So it runs only in models with
# one fiber and one thread, which both wait on word "a".
WATCH = {"Watch": "TRUE", "Fibers": '{"f1"}', "OnB": "{}"}
WATCH_A_CALLBACK = {**WATCH, **A_CALLBACK}
# Fiber f2 waits on word "c", which no waker sets.
WATCH_A_DEADLOCK = {**WATCH, "Fibers": '{"f2"}', "OnC": '{"f2"}', "Cancel": "FALSE"}
WATCH_A_DEADLOCK_IN_A_CALLBACK = {**WATCH, **A_DEADLOCK_IN_A_CALLBACK}
WATCH_A_TIMED_TASK = {**WATCH_A_DEADLOCK, "Timed": '{"f2"}'}
# Task f2 waits for a socket. Thread t1 waits on word "c", which no waker sets, so the watcher
# must report a deadlock after the wait of f2 ended, and not before.
WATCH_A_SOCKET = {**WATCH, "Fibers": '{"f2"}', "OnIo": '{"f2"}', "OnC": '{"t1"}',
                  "SetWords": "FALSE", "Cancel": "FALSE"}
MODELS = [
    ("as in WaitWake.cfg", {}),
    ("with no cancel", NO_CANCEL),
    ("with only a cancel", ONLY_A_CANCEL),
    ("with a deadlock", A_DEADLOCK),
    ("with a task under a callback", A_CALLBACK),
    ("with a deadlock in a callback", A_DEADLOCK_IN_A_CALLBACK),
    ("with a task that waits with a time limit", A_TIMED_TASK),
    ("with a thread that waits with a time limit", A_TIMED_THREAD),
    ("with a seed", A_SEED),
    ("with a task that waits for a socket", A_SOCKET),
    ("with a thread that waits for a helper call", A_THREAD_ON_IO),
    ("with the watcher", WATCH),
    ("with the watcher and a task under a callback", WATCH_A_CALLBACK),
    ("with the watcher and a deadlock", WATCH_A_DEADLOCK),
    ("with the watcher and a deadlock in a callback", WATCH_A_DEADLOCK_IN_A_CALLBACK),
    ("with the watcher and a task that waits with a time limit", WATCH_A_TIMED_TASK),
    ("with the watcher and a task that waits for a socket", WATCH_A_SOCKET),
]

# (a mistake, the constants of the check, the claim that TLC must report). The check of a
# mistake looks only at its claim, so no other claim can report the mistake first.
BUGS = [
    ("no_word_check_under_the_lock", NO_CANCEL, "EachWaitEnds"),
    ("no_cancel_check_under_the_lock", ONLY_A_CANCEL, "EachWaitEnds"),
    ("wake_before_set", NO_CANCEL, "EachWaitEnds"),
    ("the_worker_does_not_look_for_woken", {}, "EachWaitEnds"),
    ("parking_after_the_unlock", {}, "NoFailedCheck"),
    ("a_wake_of_a_parking_fiber_queues_it", {}, "ReadyOnce"),
    ("a_woken_thread_counts_again", {}, "TheWatcherCountsEachSleeper"),
    ("a_fiber_with_no_note", {}, "EachWaiterHasANote"),
    ("a_wake_clears_every_note", NO_CANCEL, "EachWaiterHasANote"),
    ("a_woken_thread_counts_as_waiting", WATCH, "NoFalseDeadlock"),
    ("a_woken_callback_task_stays_blocked", WATCH_A_CALLBACK, "NoFalseDeadlock"),
    ("the_old_callback_wait", WATCH_A_DEADLOCK_IN_A_CALLBACK, "DeadlockIsReported"),
    ("the_watcher_does_not_look_at_the_timer", WATCH_A_TIMED_TASK, "NoFalseDeadlock"),
    ("the_timer_looks_idle_before_its_wake", WATCH_A_TIMED_TASK, "NoFalseDeadlock"),
    ("the_watcher_reports_after_one_look", WATCH, "NoFalseDeadlock"),
    ("the_watcher_does_not_look_at_io", WATCH_A_SOCKET, "NoFalseDeadlock"),
    ("the_io_thread_looks_idle_before_its_wake", WATCH_A_SOCKET, "NoFalseDeadlock"),
    ("the_waiter_returns_before_it_unlists", A_SOCKET, "NoFailedCheck"),
    ("a_push_does_not_wake_the_main_thread", {**A_SEED, **NO_CANCEL}, "EachWaitEnds"),
]

# The checks of Rendezvous. With an old id file, rank 0 stops at its check. With faults, rank 0
# can fail before it writes the id file, and a rank can stop waiting for it.
AN_OLD_FILE = {"OldFile": "TRUE"}
FAULTS = {"Faults": "TRUE"}
RENDEZVOUS_MODELS = [
    ("as in Rendezvous.cfg", {}),
    ("with an old id file", AN_OLD_FILE),
    ("with faults", FAULTS),
]
RENDEZVOUS_BUGS = [
    ("write_the_id_in_place", {}, "NoHalfId"),
    ("no_check_for_an_old_file", AN_OLD_FILE, "AnOldIdStopsRankZero"),
    ("a_rank_waits_with_no_time_limit", FAULTS, "EachRankEnds"),
    ("rank_0_removes_the_file_before_its_init", {}, "EveryRankJoins"),
]

# The checks of Agent. A host can stop with no FREE, and the worker process can stop.
STOP_EARLY = {"StopEarly": "TRUE"}
WORKER_DIES = {"WorkerDies": "TRUE"}
AGENT_MODELS = [
    ("as in Agent.cfg", {}),
    ("with a host that stops early", STOP_EARLY),
    ("with a worker that stops", WORKER_DIES),
    ("with both", {**STOP_EARLY, **WORKER_DIES}),
]
AGENT_BUGS = [
    ("one_host_at_a_time", {}, "EachHostEnds"),
    ("one_connection_per_name", {}, "EachRequestFindsItsBuffer"),
    ("one_table_for_every_host", {}, "EachRequestFindsItsBuffer"),
    ("no_free_after_a_host_leaves", {}, "NoBufferOutlivesItsHost"),
    ("free_before_the_fetch", {}, "EachRequestFindsItsBuffer"),
    ("a_host_waits_after_eof", WORKER_DIES, "EachHostEnds"),
]

# Each module: its checks, its mistakes, and its temporal claims.
MODULES = {
    "WaitWake": (MODELS, BUGS, {"EachWaitEnds", "DeadlockIsReported"}),
    "Rendezvous": (RENDEZVOUS_MODELS, RENDEZVOUS_BUGS, {"EachRankEnds", "EveryRankJoins"}),
    "Agent": (AGENT_MODELS, AGENT_BUGS, {"EachHostEnds"}),
}


def config(module, constants, temporal, claim=None):
    """The text of the .cfg of `module` with `constants`, and with only `claim` if it is given."""
    text = open(os.path.join(HERE, f"{module}.cfg")).read()
    for name, value in constants.items():
        text, n = re.subn(rf"^(\s*{name} = ).*$", rf"\g<1>{value}", text, flags=re.M)
        assert n == 1, name
    if claim:
        kind = "PROPERTIES" if claim in temporal else "INVARIANTS"
        text, n = re.subn(r"^INVARIANTS\n(\s+\w+\n)+PROPERTIES\n(\s+\w+\n)+",
                          f"{kind}\n    {claim}\n", text, flags=re.M)
        assert n == 1, f"the claims of {module}.cfg"
    return text


def tlc(jar, module, cfg_text, meta):
    # TLC names its folder by the time, so two runs in one second need different parents.
    meta = tempfile.mkdtemp(dir=meta)
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", dir=meta, delete=False) as f:
        f.write(cfg_text)
    workers = max(1, (os.cpu_count() or 4) - 2)
    run = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-cp", jar, "tlc2.TLC", "-workers", str(workers),
         "-metadir", meta, "-config", f.name, module],
        cwd=HERE, capture_output=True, text=True)
    return run.stdout + run.stderr


def violated(out):
    """The claim that TLC reports as broken, or None when it found no error."""
    if "Model checking completed. No error has been found." in out:
        return None
    m = re.search(r"Invariant (\w+) is violated|Temporal property (\w+) was violated", out)
    if m:
        return m.group(1) or m.group(2)
    # TLC 1.7 does not name the temporal property. The caller knows which ones it checked.
    if "Temporal properties were violated" in out:
        return "a temporal property"
    return "an error of TLC:\n" + out


def main():
    jar = os.environ.get("TLA2TOOLS", os.path.join(HERE, "tla2tools.jar"))
    if not os.path.exists(jar):
        sys.exit(f"tla2tools.jar is not at {jar}. Set TLA2TOOLS.")
    failed = 0
    with tempfile.TemporaryDirectory() as meta:
        for module, (models, bugs, temporal) in MODULES.items():
            for name, constants in models:
                got = violated(tlc(jar, module, config(module, constants, temporal), meta))
                ok = got is None
                failed += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {module} {name}: " +
                      ("no error" if ok else f"TLC reported {got}"))
            for bug, constants, claim in bugs:
                cfg = config(module, {**constants, "Bug": f'"{bug}"'}, temporal, claim)
                got = violated(tlc(jar, module, cfg, meta))
                # The check has one claim, so a temporal property that TLC does not name is it.
                if got == "a temporal property" and claim in temporal:
                    got = claim
                ok = got == claim
                failed += not ok
                print(f"{'ok  ' if ok else 'FAIL'} {module} with {bug}: TLC reported {got}" +
                      ("" if ok else f", not {claim}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
