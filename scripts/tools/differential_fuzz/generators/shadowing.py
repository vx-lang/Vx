"""Functions whose blocks reuse a few names.

One function `f() -> i32` built from `let`, assignment, `if`/`else`, value `if`, `for`, `while`,
`loop`, `match` and `unsafe` blocks over the names `a`, `b`, `x`, plus a 4-element `i32` tensor
`t`, a helper taking `&mut i32`, and `break`/`continue`/`return` behind conditions. `main`
prints `f()`. The Rust twin indexes `t` through a small wrapper type, so one expression string
is valid in both languages. Values stay small, and both sides wrap on overflow.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program

NAMES = ["a", "b", "x"]

class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.loops = 0

    def expr(self, scope, depth=0):
        r = self.rng.random()
        if not scope or r < 0.25:
            return str(self.rng.randint(0, 9))
        if r < 0.35 and "t" in scope:
            return f"t[{self.index(scope, depth + 1)}]"
        v = self.rng.choice(sorted(scope - {"t"}))
        if depth < 2 and r < 0.6:
            op = self.rng.choice(["+", "-"])
            return f"({v} {op} {self.expr(scope, depth + 1)})"
        return v

    def index(self, scope, depth):
        inner = self.expr(scope - {"t"}, 2) if depth < 3 else str(self.rng.randint(0, 3))
        return f"(({inner}) % 4 + 4) % 4"

    def block(self, scope, mutable, depth):
        """A list of statements; `scope` and `mutable` are what is visible on entry."""
        scope = set(scope)
        mutable = set(mutable)
        out = []
        for _ in range(self.rng.randint(1, 4)):
            k = self.rng.random()
            if k < 0.35:
                name = self.rng.choice(NAMES)
                is_mut = self.rng.random() < 0.5
                e = self.expr(scope)
                out.append(("let", name, is_mut, e))
                scope.add(name)
                if is_mut:
                    mutable.add(name)
                else:
                    mutable.discard(name)
            elif k < 0.55 and mutable:
                name = self.rng.choice(sorted(mutable))
                out.append(("assign", name, self.expr(scope)))
            elif k < 0.7 and depth < 3:
                out.append(("if", self.expr(scope), self.block(scope, mutable, depth + 1),
                            self.block(scope, mutable, depth + 1) if self.rng.random() < 0.5 else None))
            elif k < 0.85 and depth < 3:
                iv = self.rng.choice(NAMES + ["i"])
                inner_scope = scope | {iv}
                inner_mut = mutable - {iv}
                self.loops += 1
                out.append(("for", iv, self.rng.randint(0, 3), self.block(inner_scope, inner_mut, depth + 1)))
                self.loops -= 1
            elif depth < 3 and k < 0.9:
                out.append(("unsafe", self.block(scope, mutable, depth + 1)))
            elif depth < 3 and k < 0.94:
                # `let <name> = if c { ..; e } else { ..; e };`
                name = self.rng.choice(NAMES)
                t = self.block(scope, mutable, depth + 1)
                e = self.block(scope, mutable, depth + 1)
                out.append(("letif", name, self.expr(scope), t, self.tail(t, scope), e, self.tail(e, scope)))
                scope.add(name)
                mutable.discard(name)
            elif depth < 3 and k < 0.97:
                arms = [self.block(scope, mutable, depth + 1) for _ in range(3)]
                out.append(("match", self.expr(scope), arms))
            elif depth < 3 and k < 0.98:
                counter = f"k{depth}"
                self.loops += 1
                out.append(("loop", counter, self.rng.randint(0, 3), self.block(scope, mutable, depth + 1)))
                self.loops -= 1
            elif depth < 3:
                counter = f"w{depth}"
                self.loops += 1
                out.append(("while", counter, self.rng.randint(0, 3), self.block(scope, mutable, depth + 1)))
                self.loops -= 1
            if self.rng.random() < 0.12 and "t" in scope:
                out.append(("store", self.index(scope, depth), self.expr(scope)))
            r2 = self.rng.random()
            if r2 < 0.04 and self.loops > 0 and depth > 0:
                out.append(("guarded", self.expr(scope), self.rng.choice(["break", "continue"])))
            elif r2 < 0.06 and depth > 0:
                out.append(("guarded", self.expr(scope), "return " + self.expr(scope)))
            if self.rng.random() < 0.1 and mutable - {"t"}:
                target = self.rng.choice(sorted(mutable - {"t"}))
                out.append(("bump", target, self.expr(scope - {target})))
        return out

    def tail(self, stmts, scope):
        # The value of a branch: names declared in the branch are visible there.
        inner = set(scope)
        for s in stmts:
            if s[0] in ("let", "letif"):
                inner.add(s[1])
        return self.expr(inner)

def render(stmts, lang, ind):
    pad = "  " * ind
    lines = []
    for s in stmts:
        if s[0] == "let":
            _, name, is_mut, e = s
            m = "mut " if is_mut else ""
            ty = " : i32" if lang == "vx" else ": i32"
            lines.append(f"{pad}let {m}{name}{ty} = {e};")
        elif s[0] == "assign":
            lines.append(f"{pad}{s[1]} = {s[2]};")
        elif s[0] == "if":
            _, cond, then, els = s
            c = f"({cond}) % 2 == 0" if lang == "rs" else f"({cond}) % 2 == 0"
            lines.append(f"{pad}if {c} {{")
            lines += render(then, lang, ind + 1)
            if els is not None:
                lines.append(f"{pad}}} else {{")
                lines += render(els, lang, ind + 1)
            lines.append(f"{pad}}}")
        elif s[0] == "for":
            _, iv, hi, body = s
            lines.append(f"{pad}for {iv} in 0..{hi} {{")
            lines += render(body, lang, ind + 1)
            lines.append(f"{pad}}}")
        elif s[0] == "letif":
            _, name, cond, t, te, e, ee = s
            ty = " : i32" if lang == "vx" else ": i32"
            lines.append(f"{pad}let {name}{ty} = if ({cond}) % 2 == 0 {{")
            lines += render(t, lang, ind + 1)
            lines.append(f"{pad}  {te}")
            lines.append(f"{pad}}} else {{")
            lines += render(e, lang, ind + 1)
            lines.append(f"{pad}  {ee}")
            lines.append(f"{pad}}};")
        elif s[0] == "match":
            _, subj, arms = s
            lines.append(f"{pad}match (({subj}) % 3 + 3) % 3 {{")
            for label, arm in zip(["0", "1", "_"], arms):
                lines.append(f"{pad}  {label} => {{")
                lines += render(arm, lang, ind + 2)
                lines.append(f"{pad}  }}")
            lines.append(f"{pad}}}")
        elif s[0] == "loop":
            _, counter, hi, body = s
            lines.append(f"{pad}let mut {counter} = 0;")
            lines.append(f"{pad}loop {{")
            lines.append(f"{pad}  if {counter} >= {hi} {{")
            lines.append(f"{pad}    break;")
            lines.append(f"{pad}  }}")
            lines.append(f"{pad}  {counter} = {counter} + 1;")
            lines += render(body, lang, ind + 1)
            lines.append(f"{pad}}}")
        elif s[0] == "while":
            _, counter, hi, body = s
            lines.append(f"{pad}let mut {counter} = 0;")
            lines.append(f"{pad}while {counter} < {hi} {{")
            lines.append(f"{pad}  {counter} = {counter} + 1;")
            lines += render(body, lang, ind + 1)
            lines.append(f"{pad}}}")
        elif s[0] == "guarded":
            lines.append(f"{pad}if ({s[1]}) % 3 == 0 {{")
            lines.append(f"{pad}  {s[2]};")
            lines.append(f"{pad}}}")
        elif s[0] == "store":
            lines.append(f"{pad}t[{s[1]}] = {s[2]};")
        elif s[0] == "bump":
            lines.append(f"{pad}bump(&mut {s[1]}, {s[2]});")
        elif s[0] == "unsafe":
            lines.append(f"{pad}unsafe {{")
            lines += render(s[1], lang, ind + 1)
            lines.append(f"{pad}}}")
    return lines


def generate(seed):
    rng = random.Random(seed)
    g = Gen(rng)
    # The outer names are all mutable, so an assignment to one anywhere type-checks.
    prologue = [("let", n, True, str(rng.randint(0, 9))) for n in NAMES]
    tree = prologue + g.block(set(NAMES) | {"t"}, set(NAMES), 0)
    ret = " + ".join(f"{n} * {10 ** i}" for i, n in enumerate(NAMES))
    ret += " + (t[0] + t[1] * 3 + t[2] * 5 + t[3] * 7) * 1000"

    def source(stmts, lang):
        if lang == "vx":
            return "\n".join(
                ["fn bump(p : &mut i32, v : i32) -> i32 {", "  *p = *p + v;", "  return 0;", "}", "",
                 "fn f() -> i32 {", "  let mut t = Tensor<i32>([4]);", "  for i in 0..4 {",
                 "    t[i] = 0;", "  }"]
                + render(stmts, "vx", 1)
                + [f"  return {ret};", "}", "", "fn main() -> i32 {", "  print(f());",
                   "  return 0;", "}", ""])
        return "\n".join(
            ["#![allow(unused, unused_mut, unused_assignments, unused_unsafe, unused_variables)]",
             "use std::ops::{Index, IndexMut};",
             "struct T([i32; 4]);",
             "impl Index<i32> for T { type Output = i32; fn index(&self, i: i32) -> &i32 { &self.0[i as usize] } }",
             "impl IndexMut<i32> for T { fn index_mut(&mut self, i: i32) -> &mut i32 { &mut self.0[i as usize] } }",
             "fn bump(p: &mut i32, v: i32) -> i32 { *p = p.wrapping_add(v); 0 }",
             "fn f() -> i32 {", "  let mut t = T([0; 4]);"]
            + render(stmts, "rs", 1)
            + [f"  {ret}", "}", "fn main() { println!(\"{}\", f()); }", ""])

    return Program(tree, source, keep=lambda stmt, block: block is tree and stmt in prologue)
