#!/bin/bash
# Runs the RUN lines of the tests of the Qwen app, as the fixture runner of Vx runs them, from the
# root of the Vx repository. vxc, FileCheck, and a python3 that has NumPy must be on PATH.
#
#   apps/qwen/tests/run.sh              # the Python tests of the scripts, and every test here
#   apps/qwen/tests/run.sh FILE...      # only these tests
#
# The tests live here and not in tests/, so that they move with the app to its own repository.
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -u
cd "$(dirname "$0")/../../.." || exit 2
failed=0
if [ $# -eq 0 ]; then
  set -- apps/qwen/tests/*.vx
  if output=$(python3 -m unittest discover -s apps/qwen/scripts -p 'test_*.py' 2>&1); then
    echo "PASS the Python tests of apps/qwen/scripts"
  else
    echo "FAIL the Python tests of apps/qwen/scripts"
    echo "$output" | tail -20 | sed 's/^/    /'
    failed=1
  fi
fi
export VX_NO_AST_FALLBACK=1
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for test in "$@"; do
  n=0
  while IFS= read -r line; do
    n=$((n + 1))
    cmd=${line//%s/$test}
    cmd=${cmd//%S/$(dirname "$test")}
    cmd=${cmd//%t/$tmp/$(basename "$test" .vx)}
    if bash -o pipefail -c "$cmd" > "$tmp/output" 2>&1; then
      echo "PASS $test, RUN line $n"
    else
      echo "FAIL $test, RUN line $n: $line"
      tail -20 "$tmp/output" | sed 's/^/    /'
      failed=1
    fi
  done < <(sed -n 's|^// RUN: ||p' "$test")
done
exit $failed
