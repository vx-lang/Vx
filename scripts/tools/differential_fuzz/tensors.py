#!/usr/bin/env python3
"""Random functions that build and return tensors, with early returns from nested blocks.

`g(n, k)` makes one or more tensors of length `n`, fills and changes them in loops, and returns
one of them, possibly early from inside an `if` or a loop. `main` calls it for a few `k`, after
making and dropping other tensors, and prints every element. The Rust twin uses `Vec<i32>`.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
import random, sys

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

def program(seed):
    rng = random.Random(seed)
    g = Gen(rng)
    body = g.body([], ["k"], 0)
    last = rng.choice([s[1] for s in body if s[0] == "new"] or ["NONE"])
    if last == "NONE":
        body = [("new", "v0")] + body
        last = "v0"
    vx = ["fn g(n : i32, k : i32) -> Tensor<i32, [?]> {"] + render(body, "vx", 1) + [f"  return {last};", "}", "",
          "fn main() -> i32 {",
          "  for k in 0..4 {",
          "    let r = g(3, k);",
          "    for j in 0..3 {",
          "      print(r[j]);",
          "      print!(\" \");",
          "    }",
          "  }",
          "  return 0;",
          "}", ""]
    rs = ["#![allow(unused, unused_mut, unused_variables, unreachable_code)]",
          "fn g(n: i32, k: i32) -> Vec<i32> {"] + render(body, "rs", 1) + [f"  return {last};", "}",
          "fn main() {", "  let mut s = String::new();",
          "  for k in 0..4 { let r = g(3, k); for j in 0..3 { s += &format!(\"{} \", r[j]); } }",
          "  print!(\"{}\", s);", "}", ""]
    return "\n".join(vx), "\n".join(rs)

if __name__ == "__main__":
    seed = int(sys.argv[1]); out = sys.argv[2]
    vx, rs = program(seed)
    open(f"{out}/p{seed}.vx", "w").write(vx)
    open(f"{out}/p{seed}.rs", "w").write(rs)
