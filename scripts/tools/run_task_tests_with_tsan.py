#!/usr/bin/env python3
"""Runs each test program in tests/backend/pass that starts tasks with `vxc --sanitize=thread`.

The script fails when ThreadSanitizer reports a data race in a program. A program can also fail
on purpose, so its exit code does not matter. vxc must be on PATH, or in VXC.

    python3 scripts/tools/run_task_tests_with_tsan.py
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TESTS = os.path.join(ROOT, "tests", "backend", "pass")
# This program has a data race on purpose.
SKIP = {"thread_sanitizer_reports_a_data_race.vx"}
STARTS_TASKS = re.compile(r"^\s*(tasks|spawn)\b", re.M)
# A test for a GPU or for another machine.
OTHER_MACHINE = re.compile(r"// REQUIRES: (gpu|cuda|macos|aarch64)")


def programs():
    for name in sorted(os.listdir(TESTS)):
        if not name.endswith(".vx") or name in SKIP:
            continue
        text = open(os.path.join(TESTS, name)).read()
        if STARTS_TASKS.search(text) and not OTHER_MACHINE.search(text):
            yield name


def run(name):
    vxc = os.environ.get("VXC", "vxc")
    try:
        out = subprocess.run([vxc, "--sanitize=thread", os.path.join(TESTS, name)],
                             capture_output=True, text=True, timeout=300)
        text = out.stdout + out.stderr
    except subprocess.TimeoutExpired:
        return name, "the program ran longer than 300 seconds"
    if "WARNING: ThreadSanitizer" in text:
        return name, text
    return name, None


def main():
    names = list(programs())
    workers = max(1, (os.cpu_count() or 4) - 2)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run, names))
    failed = [(n, why) for n, why in results if why]
    for n, why in failed:
        print(f"FAIL {n}:\n{why}\n")
    print(f"{len(names) - len(failed)} of {len(names)} programs that start tasks had no data race.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
