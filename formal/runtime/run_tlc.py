#!/usr/bin/env python3
"""Runs the TLC model checker on the models of the runtime protocols.

Each model must pass. Then each mistake in BUGS goes into the model, one at a time, and TLC
must report the claim that the mistake breaks. A claim that no mistake breaks may check
nothing.

TLC is in tla2tools.jar. Give its path in TLA2TOOLS, or put it next to this file.
"""

import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# The model checks, each a change to the constants of WaitWake.cfg. The cancel and the second
# waker wake the bucket again, and that can hide a lost wake. So some checks have only one.
NO_CANCEL = {"Cancel": "FALSE"}
ONLY_A_CANCEL = {"SetWords": "FALSE"}
MODELS = [{}, NO_CANCEL, ONLY_A_CANCEL]

# (a mistake, the constants of the check, the claim that TLC must report).
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
]


def config(constants):
    text = open(os.path.join(HERE, "WaitWake.cfg")).read()
    for name, value in constants.items():
        text, n = re.subn(rf"^(\s*{name} = ).*$", rf"\g<1>{value}", text, flags=re.M)
        assert n == 1, name
    return text


def tlc(jar, cfg_text, meta):
    with tempfile.NamedTemporaryFile("w", suffix=".cfg", dir=meta, delete=False) as f:
        f.write(cfg_text)
    workers = max(1, (os.cpu_count() or 4) - 2)
    run = subprocess.run(
        ["java", "-XX:+UseParallelGC", "-cp", jar, "tlc2.TLC", "-workers", str(workers),
         "-metadir", meta, "-config", f.name, "WaitWake"],
        cwd=HERE, capture_output=True, text=True)
    return run.stdout + run.stderr


def violated(out):
    """The claim that TLC reports as broken, or None when it found no error."""
    if "Model checking completed. No error has been found." in out:
        return None
    m = re.search(r"Invariant (\w+) is violated|Temporal property (\w+) was violated", out)
    if m:
        return m.group(1) or m.group(2)
    # TLC 1.7 does not name the temporal property, and the model has only one.
    if "Temporal properties were violated" in out:
        return "EachWaitEnds"
    return "an error of TLC:\n" + out


def main():
    jar = os.environ.get("TLA2TOOLS", os.path.join(HERE, "tla2tools.jar"))
    if not os.path.exists(jar):
        sys.exit(f"tla2tools.jar is not at {jar}. Set TLA2TOOLS.")
    failed = 0
    with tempfile.TemporaryDirectory() as meta:
        for constants in MODELS:
            got = violated(tlc(jar, config(constants), meta))
            ok = got is None
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} WaitWake {constants or 'as in WaitWake.cfg'}: " +
                  ("no error" if ok else f"TLC reported {got}"))
        for bug, constants, claim in BUGS:
            got = violated(tlc(jar, config({**constants, "Bug": f'"{bug}"'}), meta))
            ok = got == claim
            failed += not ok
            print(f"{'ok  ' if ok else 'FAIL'} WaitWake with {bug}: TLC reported {got}" +
                  ("" if ok else f", not {claim}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
