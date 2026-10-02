"""Functions that build tensors in nested blocks and return one, often early.

`g(n, k)` makes tensors of length `n` in nested `if`s and `for` loops, adds to them, and
returns one, possibly from inside a block. `main` calls it for four values of `k` and prints
every element. The Rust twin uses `Vec<i32>`. Aimed at the freeing of heap buffers: an early
return from nested blocks is where #993 and #1014 were.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program

class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.count = 0

    def fresh(self):
        self.count += 1
        return f"v{self.count}"

    def val(self, ints):
        r = self.rng.random()
        if r < 0.3 or not ints:
            return str(self.rng.randint(0, 9))
        a = self.rng.choice(ints)
        if r < 0.7:
            return f"({a} + {self.rng.choice(ints + ['1', '2'])})"
        return a

    def body(self, tensors, ints, depth):
        out = []
        for _ in range(self.rng.randint(1, 4)):
            k = self.rng.random()
            if k < 0.25 or not tensors:
                t = self.fresh()
                out.append(("new", t))
                tensors = tensors + [t]
            elif k < 0.5:
                out.append(("fill", self.rng.choice(tensors), self.val(ints)))
            elif k < 0.65 and depth < 3:
                out.append(("if", self.val(ints), self.body(tensors, ints, depth + 1),
                            self.body(tensors, ints, depth + 1) if self.rng.random() < 0.5 else None))
            elif k < 0.8 and depth < 3:
                iv = f"i{depth}"
                out.append(("for", iv, self.rng.randint(0, 3), self.body(tensors, ints + [iv], depth + 1)))
            elif k < 0.9 and depth > 0:
                out.append(("ret", self.val(ints), self.rng.choice(tensors)))
            else:
                a, b = self.rng.choice(tensors), self.rng.choice(tensors)
                out.append(("add", a, b))
        return out

def render(stmts, lang, ind):
    pad = "  " * ind
    L = []
    for s in stmts:
        if s[0] == "new":
            if lang == "vx":
                L += [f"{pad}let mut {s[1]} = Tensor<i32>([n]);", f"{pad}for j in 0..n {{", f"{pad}  {s[1]}[j] = 0;", f"{pad}}}"]
            else:
                L += [f"{pad}let mut {s[1]}: Vec<i32> = vec![0; n as usize];"]
        elif s[0] == "fill":
            idx = "j" if lang == "vx" else "j as usize"
            L += [f"{pad}for j in 0..n {{", f"{pad}  {s[1]}[{idx}] = {s[1]}[{idx}] + j + {s[2]};", f"{pad}}}"]
        elif s[0] == "add":
            idx = "j" if lang == "vx" else "j as usize"
            L += [f"{pad}for j in 0..n {{", f"{pad}  {s[1]}[{idx}] = {s[1]}[{idx}] + {s[2]}[{idx}];", f"{pad}}}"]
        elif s[0] == "if":
            L.append(f"{pad}if ({s[1]}) % 2 == 0 {{")
            L += render(s[2], lang, ind + 1)
            if s[3] is not None:
                L.append(f"{pad}}} else {{")
                L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
        elif s[0] == "for":
            L.append(f"{pad}for {s[1]} in 0..{s[2]} {{")
            L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
        elif s[0] == "ret":
            L.append(f"{pad}if ({s[1]}) % 3 == 0 {{")
            L.append(f"{pad}  return {s[2]};")
            L.append(f"{pad}}}")
    return L


def generate(seed):
    rng = random.Random(seed)
    g = Gen(rng)
    tree = g.body([], ["k"], 0)
    last = rng.choice([s[1] for s in tree if s[0] == "new"] or ["NONE"])
    if last == "NONE":
        tree.insert(0, ("new", "v0"))
        last = "v0"
    returned = next(s for s in tree if s[0] == "new" and s[1] == last)

    def source(stmts, lang):
        if lang == "vx":
            return "\n".join(
                ["fn g(n : i32, k : i32) -> Tensor<i32, [?]> {"] + render(stmts, "vx", 1)
                + [f"  return {last};", "}", "", "fn main() -> i32 {", "  for k in 0..4 {",
                   "    let r = g(3, k);", "    for j in 0..3 {", "      print(r[j]);",
                   "      print!(\" \");", "    }", "  }", "  return 0;", "}", ""])
        return "\n".join(
            ["#![allow(unused, unused_mut, unused_variables, unreachable_code)]",
             "fn g(n: i32, k: i32) -> Vec<i32> {"] + render(stmts, "rs", 1)
            + [f"  return {last};", "}", "fn main() {", "  let mut s = String::new();",
               "  for k in 0..4 { let r = g(3, k); for j in 0..3 { s += &format!(\"{} \", r[j]); } }",
               "  print!(\"{}\", s);", "}", ""])

    return Program(tree, source, keep=lambda stmt, block: stmt is returned)
