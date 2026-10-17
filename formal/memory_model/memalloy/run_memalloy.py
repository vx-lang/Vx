#!/usr/bin/env python3
"""Check the CPU part of the Vx memory model with memalloy.

memalloy looks for a small execution that one model allows and another forbids. When it finds
none up to a number of events, the two agree up to that size. Each check below says how many
executions it expects: 0 means "they agree", 1 means "memalloy must find a counterexample".

Usage: run_memalloy.py [--quick]. --quick uses fewer events.
Set MEMALLOY to the memalloy directory (default ~/memalloy), built with "make install".
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (what it checks, expected executions, events, events quick, comparator arguments)
CHECKS = [
    ("Vx allows nothing that RC11 forbids", 0, 6, 4,
     ["-arch", "C", "-omitdeadness", "-satisfies", "vx_cpu.cat", "-violates", "c11_lahav.cat"]),
    ("RC11 allows nothing that Vx forbids", 0, 6, 4,
     ["-arch", "C", "-omitdeadness", "-satisfies", "c11_lahav.cat", "-violates", "vx_cpu.cat"]),
    ("Vx and RC11 agree on which executions have a data race", 0, 6, 4,
     ["-arch", "C", "-omitdeadness", "-satisfies", "vx_cpu_without_races.cat",
      "-violates", "rc11_without_races.cat"]),
    ("RC11 and Vx agree on which executions have a data race", 0, 6, 4,
     ["-arch", "C", "-omitdeadness", "-satisfies", "rc11_without_races.cat",
      "-violates", "vx_cpu_without_races.cat"]),
    ("The AArch64 mapping gives no result that Vx forbids", 0, 5, 4,
     ["-arch", "C", "-arch2", "ARM8", "-fencerels", "-violates", "vx_cpu.cat",
      "-satisfies", "aarch64.cat", "-mapping", "vx_to_aarch64.als"]),
    ("Without the dependency, the AArch64 mapping gives a result that Vx forbids", 1, 4, 4,
     ["-arch", "C", "-arch2", "ARM8", "-fencerels", "-violates", "vx_cpu.cat",
      "-satisfies", "aarch64.cat", "-mapping", "vx_to_aarch64_without_dependency.als"]),
]


def run(memalloy, work, check, quick):
    name, want, events, events_quick, args = check
    n = str(events_quick if quick else events)
    rel = os.path.basename(work)
    args = [f"{rel}/{a}" if a.endswith((".cat", ".als")) else a for a in args]
    args += ["-events", n]
    if "-mapping" in args:
        args += ["-events2", n]
    # memalloy names its results directory by the second, so two runs in one second collide.
    time.sleep(1)
    out = subprocess.run([os.path.join(memalloy, "comparator"), "-batch", "-desc", "vx"] + args,
                         cwd=memalloy, capture_output=True, text=True, check=False,
                         env=dict(os.environ, JAVA_HEAP_SIZE="3g"))
    found = re.search(r"Alloy found (\d+) solutions", out.stdout)
    if not found:
        return f"{name}: memalloy gave no result:\n{out.stdout}{out.stderr}"
    got = int(found.group(1))
    if got != want:
        return (f"{name} ({n} events): expected {want} executions, found {got}. See "
                f"{os.path.join(memalloy, 'results', '_latest', 'litmus')}.")
    print(f"ok    {name} ({n} events)")
    return None


def main():
    quick = "--quick" in sys.argv[1:]
    memalloy = os.environ.get("MEMALLOY", os.path.expanduser("~/memalloy"))
    if not os.access(os.path.join(memalloy, "comparator"), os.X_OK):
        sys.exit(f"memalloy not found in {memalloy}. Set MEMALLOY.")
    # memalloy finds a model only inside its own directory, so copy the models there.
    work = tempfile.mkdtemp(prefix="vx_models_", dir=memalloy)
    try:
        for d in (os.path.join(memalloy, "models"), HERE):
            for f in os.listdir(d):
                if f.endswith((".cat", ".als")):
                    shutil.copy(os.path.join(d, f), work)
        errors = [e for e in (run(memalloy, work, c, quick) for c in CHECKS) if e]
    finally:
        shutil.rmtree(work)
    for e in errors:
        print(f"FAIL  {e}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
