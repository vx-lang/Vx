#!/usr/bin/env python3
"""Run the litmus tests of the Vx memory model with herd7, and check each result.

Each test says what it expects in a line "Expect: <never|sometimes|always>, <no data race|a
data race>." This script runs the test on vx.cat and compares.

A test without a data race, whose accesses are plain or sequentially consistent, must also give
the same final states as sequential.cat: such a program cannot see weak memory ordering
(DRF-SC). The script checks this too.

Usage: run_litmus.py [test.litmus ...]. With no arguments it runs every test in tests/.
herd7 comes from herdtools7 (opam install herdtools7). Set HERD7 to use another binary.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = re.compile(r"Expect:\s*(never|sometimes|always),\s*(no data race|a data race)\.")
WEAK_TAG = re.compile(r"\[(rlx|acq|rel|acq_rel)\b")


def herd(herd7, model, test):
    out = subprocess.run(
        [herd7, "-I", HERE, "-bell", os.path.join(HERE, "vx.bell"),
         "-model", os.path.join(HERE, model), test],
        capture_output=True, text=True, check=False).stdout
    obs = re.search(r"^Observation \S+ (Never|Sometimes|Always)", out, re.M)
    if not obs:
        sys.exit(f"{test}: herd7 gave no result:\n{out}")
    states = re.search(r"^States \d+\n(.*?)^(Ok|No)$", out, re.M | re.S)
    return {
        "observation": obs.group(1).lower(),
        "race": "Flag data-race" in out,
        "states": sorted(s for s in states.group(1).split("\n") if s) if states else [],
    }


def check(herd7, test):
    text = open(test).read()
    want = EXPECT.search(text)
    if not want:
        return [f"no 'Expect:' line"]
    got = herd(herd7, "vx.cat", test)
    errors = []
    if got["observation"] != want.group(1):
        errors.append(f"expected {want.group(1)}, got {got['observation']}")
    want_race = want.group(2) == "a data race"
    if got["race"] != want_race:
        errors.append("expected a data race, got none" if want_race
                      else "expected no data race, got one")
    if not want_race and not WEAK_TAG.search(text):
        seq = herd(herd7, "sequential.cat", test)
        if seq["states"] != got["states"]:
            errors.append(f"DRF-SC: final states {got['states']} differ from the sequential "
                          f"states {seq['states']}")
    return errors


def main():
    herd7 = os.environ.get("HERD7") or shutil.which("herd7")
    if not herd7:
        sys.exit("herd7 not found. Install herdtools7 (opam install herdtools7), or set HERD7.")
    tests = sys.argv[1:] or sorted(
        os.path.join(HERE, "tests", f)
        for f in os.listdir(os.path.join(HERE, "tests")) if f.endswith(".litmus"))
    failed = 0
    for test in tests:
        errors = check(herd7, test)
        name = os.path.basename(test)
        if errors:
            failed += 1
            print(f"FAIL {name}: " + "; ".join(errors))
        else:
            print(f"ok   {name}")
    print(f"{len(tests) - failed} of {len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
