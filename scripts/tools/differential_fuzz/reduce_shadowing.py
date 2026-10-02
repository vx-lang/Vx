#!/usr/bin/env python3
"""Shrink a fuzz program while `vxc [flags]` still prints a given message.

usage: reduce.py SEED VXC "MESSAGE" [--legacy-codegen]

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import copy, os, random, subprocess, sys, tempfile
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
SCRATCH = os.path.join(tempfile.mkdtemp(), "reduced.vx")
import shadowing as gen

seed, vxc, msg = int(sys.argv[1]), sys.argv[2], sys.argv[3]
flags = sys.argv[4:]

rng = random.Random(seed)
g = gen.Gen(rng)
prologue = [("let", n, True, str(rng.randint(0, 9))) for n in gen.NAMES]
rest = g.block(set(gen.NAMES) | {"t"}, set(gen.NAMES), 0)

def source(stmts):
    stmts = prologue + stmts
    ret = " + ".join(f"{n} * {10 ** i}" for i, n in enumerate(gen.NAMES))
    ret += " + (t[0] + t[1] * 3 + t[2] * 5 + t[3] * 7) * 1000"
    return "\n".join(["fn bump(p : &mut i32, v : i32) -> i32 {", "  *p = *p + v;", "  return 0;", "}", "",
                      "fn f() -> i32 {", "  let mut t = Tensor<i32>([4]);", "  for i in 0..4 {", "    t[i] = 0;", "  }"]
                     + gen.render(stmts, "vx", 1) +
                     [f"  return {ret};", "}", "", "fn main() -> i32 {", "  print(f());", "  return 0;", "}", ""])

def fails(stmts):
    open(SCRATCH, "w").write(source(stmts))
    out = subprocess.run([vxc, *flags, SCRATCH],
                         capture_output=True, text=True, timeout=120)
    return msg in out.stdout + out.stderr

def lists(stmts):
    """Every statement list in the tree, outermost first."""
    yield stmts
    for s in stmts:
        if s[0] == "if":
            yield from lists(s[2])
            if s[3] is not None:
                yield from lists(s[3])
        elif s[0] in ("for", "loop"):
            yield from lists(s[3])
        elif s[0] == "unsafe":
            yield from lists(s[1])
        elif s[0] == "letif":
            yield from lists(s[3]); yield from lists(s[5])
        elif s[0] == "match":
            for arm in s[2]:
                yield from lists(arm)

body = rest
assert fails(body), "the original does not show the message"
changed = True
while changed:
    changed = False
    for lst in list(lists(body)):
        i = 0
        while i < len(lst):
            removed = lst.pop(i)
            if fails(body):
                changed = True
            else:
                lst.insert(i, removed)
                i += 1
print(source(body))
