#!/usr/bin/env python3
"""Shrink a gen_tensor program while `vxc [flags]` still prints MESSAGE when run.

usage: reduce_t.py SEED VXC "MESSAGE" [flags]

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import os, random, subprocess, sys, tempfile
sys.path.insert(0, sys.argv[0].rsplit("/", 1)[0])
SCRATCH = os.path.join(tempfile.mkdtemp(), "reduced.vx")
import tensors as gt

seed, vxc, msg = int(sys.argv[1]), sys.argv[2], sys.argv[3]
flags = sys.argv[4:]
here = sys.argv[0].rsplit("/", 1)[0]
rng = random.Random(seed)
g = gt.Gen(rng)
body = g.body([], ["k"], 0)
last = rng.choice([s[1] for s in body if s[0] == "new"] or ["NONE"])
if last == "NONE":
    body = [("new", "v0")] + body
    last = "v0"

def source(stmts):
    return "\n".join(["fn g(n : i32, k : i32) -> Tensor<i32, [?]> {"] + gt.render(stmts, "vx", 1) +
                     [f"  return {last};", "}", "", "fn main() -> i32 {", "  for k in 0..4 {",
                      "    let r = g(3, k);", "    for j in 0..3 {", "      print(r[j]);", "      print!(\" \");",
                      "    }", "  }", "  return 0;", "}", ""])

def fails(stmts):
    path = SCRATCH
    open(path, "w").write(source(stmts))
    import os
    for _ in range(int(os.environ.get("REPEAT", "1"))):
        out = subprocess.run([vxc, *flags, path], capture_output=True, text=True, timeout=120)
        if msg in out.stdout + out.stderr:
            return True
    return False

def lists(stmts):
    yield stmts
    for s in stmts:
        if s[0] == "if":
            yield from lists(s[2])
            if s[3] is not None:
                yield from lists(s[3])
        elif s[0] == "for":
            yield from lists(s[3])

def declared(stmts):
    for s in stmts:
        if s[0] == "new":
            yield s[1]

assert fails(body), "the original does not show the message"
changed = True
while changed:
    changed = False
    for lst in list(lists(body)):
        i = 0
        while i < len(lst):
            removed = lst.pop(i)
            # keep the returned tensor's declaration at the top level
            if removed[0] == "new" and removed[1] == last and lst is body:
                lst.insert(i, removed); i += 1; continue
            try:
                ok = fails(body)
            except Exception:
                ok = False
            if ok:
                changed = True
            else:
                lst.insert(i, removed)
                i += 1
print(source(body))
