"""Tensors made, moved, borrowed and given new values, in nested `if`s and `for` loops.

`main` keeps a running total `s`. It makes tensors of 5 numbers with a `?` length and of 4
numbers with a fixed length, and then:

- passes a tensor by value to a function, which then owns it, or by `&`;
- passes one through a function that hands it back, or adds two together;
- reads and writes its elements, and gives a variable a new tensor;
- makes a tensor nobody names: `look(&make(5, 3))`, `take4(make4(2))`.

A tensor moved inside an `if` is moved on one path only, so whoever frees it has to know at
run time whether it moved. A tensor is moved inside a loop only when the loop's own body made
it. At the end `main` adds up every tensor still alive and prints `s`.

The Rust twin uses `Vec<i32>` for both kinds. With `--heap`, every program should free all it
allocates.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program


class Var:
    def __init__(self, name, kind, loops):
        self.name = name
        self.kind = kind  # "d": `?` length 5, "s": fixed length 4
        self.loops = loops  # how many loops enclose its `let`
        self.alive = True


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.count = 0

    def fresh(self):
        self.count += 1
        return f"t{self.count}"

    def num(self):
        return self.rng.randint(0, 9)

    def body(self, scope, loops, depth):
        """Statements for one block. `scope` lists the variables visible here; the block's own
        `let`s are added to a copy, so they are gone after it."""
        scope = list(scope)
        own = []
        out = []
        for _ in range(self.rng.randint(1, 5)):
            alive = [v for v in scope if v.alive]
            # A variable made outside the innermost loop would be moved again on its next
            # round, which both languages refuse.
            movable = [v for v in alive if v.loops == loops]
            k = self.rng.random()
            if k < 0.2 or not alive:
                v = Var(self.fresh(), self.rng.choice("ds"), loops)
                out.append(("make", v.name, v.kind, self.num()))
                scope.append(v)
                own.append(v)
            elif k < 0.32 and movable:
                v = self.rng.choice(movable)
                v.alive = False
                out.append(("take", v.name, v.kind))
            elif k < 0.44:
                v = self.rng.choice(alive)
                out.append(("look", v.name, v.kind))
            elif k < 0.52:
                v = self.rng.choice(alive)
                out.append(("read", v.name, self.rng.randrange(4)))
            elif k < 0.58:
                v = self.rng.choice(alive)
                out.append(("write", v.name, self.rng.randrange(4), self.num()))
            elif k < 0.65 and movable:
                v = self.rng.choice(movable)
                v.alive = False
                w = Var(self.fresh(), v.kind, loops)
                out.append(("pass", w.name, v.name, v.kind))
                scope.append(w)
            elif k < 0.7 and len([v for v in movable if v.kind == "s"]) >= 2:
                a, b = self.rng.sample([v for v in movable if v.kind == "s"], 2)
                a.alive = b.alive = False
                w = Var(self.fresh(), "s", loops)
                out.append(("add", w.name, a.name, b.name))
                scope.append(w)
            elif k < 0.74:
                # A new value for a variable this block made; one moved before gets a value
                # again. One made outside would hold a value on some paths only.
                if own:
                    v = self.rng.choice(own)
                    v.alive = True
                    out.append(("refill", v.name, v.kind, self.num()))
            elif k < 0.8:
                out.append(("temp", self.rng.choice("ds"), self.num()))
            elif k < 0.9 and depth < 3:
                then_b = self.body(scope, loops, depth + 1)[0]
                else_b = None
                if self.rng.random() < 0.5:
                    else_b = self.body(scope, loops, depth + 1)[0]
                out.append(("if", self.num(), then_b, else_b))
            elif depth < 3:
                out.append(("for", f"i{depth}", self.rng.randint(0, 3),
                            self.body(scope, loops + 1, depth + 1)[0]))
        return out, scope


def render(stmts, lang, ind):
    pad = "  " * ind
    vx = lang == "vx"
    L = []
    for s in stmts:
        kind = s[0]
        if kind == "make":
            _, v, k, x = s
            made = f"make(5, {x})" if k == "d" else f"make4({x})"
            L.append(f"{pad}let mut {v} = {made};")
        elif kind == "take":
            _, v, k = s
            L.append(f"{pad}s = s + {'take(' + v + ')' if k == 'd' else 'take4(' + v + ')'};")
        elif kind == "look":
            _, v, k = s
            L.append(f"{pad}s = s + {'look' if k == 'd' else 'look4'}(&{v});")
        elif kind == "read":
            L.append(f"{pad}s = s + {s[1]}[{s[2]}];")
        elif kind == "write":
            L.append(f"{pad}{s[1]}[{s[2]}] = {s[3]};")
        elif kind == "pass":
            _, w, v, k = s
            L.append(f"{pad}let mut {w} = {'pass' if k == 'd' else 'pass4'}({v});")
        elif kind == "add":
            _, w, a, b = s
            L.append(f"{pad}let mut {w} = {a + ' + ' + b if vx else 'add(' + a + ', ' + b + ')'};")
        elif kind == "refill":
            _, v, k, x = s
            L.append(f"{pad}{v} = {'make(5, ' + str(x) + ')' if k == 'd' else 'make4(' + str(x) + ')'};")
        elif kind == "temp":
            _, k, x = s
            if k == "d":
                L.append(f"{pad}s = s + look(&make(5, {x}));")
            else:
                L.append(f"{pad}s = s + take4(make4({x}));")
        elif kind == "if":
            L.append(f"{pad}if (s + {s[1]}) % 2 == 0 {{")
            L += render(s[2], lang, ind + 1)
            if s[3] is not None:
                L.append(f"{pad}}} else {{")
                L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
        elif kind == "for":
            L.append(f"{pad}for {s[1]} in 0..{s[2]} {{")
            L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
    return L


VX_HEAD = """fn make(n : i32, x : i32) -> Tensor<i32, [?]> {
  let mut t = Tensor<i32>([n]);
  for i in 0..n {
    t[i] = x + i;
  }
  return t;
}

fn make4(x : i32) -> Tensor<i32, [4]> {
  let mut t = Tensor<i32, [4]>::new();
  for i in 0..4 {
    t[i] = x + i;
  }
  return t;
}

fn take(t : Tensor<i32, [?]>) -> i32 {
  return t[0] + t[1] + t[2] + t[3] + t[4];
}

fn take4(t : Tensor<i32, [4]>) -> i32 {
  return t[0] + t[1] + t[2] + t[3];
}

fn look(t : &Tensor<i32, [?]>) -> i32 {
  return t[0] + t[4];
}

fn look4(t : &Tensor<i32, [4]>) -> i32 {
  return t[0] + t[3];
}

fn pass(t : Tensor<i32, [?]>) -> Tensor<i32, [?]> {
  return t;
}

fn pass4(t : Tensor<i32, [4]>) -> Tensor<i32, [4]> {
  return t;
}

fn main() -> i32 {
  let mut s = 0;"""

RS_HEAD = """#![allow(unused, unused_mut, unused_variables, unused_assignments)]
fn make(n: i32, x: i32) -> Vec<i32> { (0..n).map(|i| x + i).collect() }
fn make4(x: i32) -> Vec<i32> { make(4, x) }
fn take(t: Vec<i32>) -> i32 { t[0] + t[1] + t[2] + t[3] + t[4] }
fn take4(t: Vec<i32>) -> i32 { t[0] + t[1] + t[2] + t[3] }
fn look(t: &Vec<i32>) -> i32 { t[0] + t[4] }
fn look4(t: &Vec<i32>) -> i32 { t[0] + t[3] }
fn pass(t: Vec<i32>) -> Vec<i32> { t }
fn pass4(t: Vec<i32>) -> Vec<i32> { t }
fn add(a: Vec<i32>, b: Vec<i32>) -> Vec<i32> { a.iter().zip(&b).map(|(x, y)| x + y).collect() }
fn main() {
  let mut s = 0i32;"""


def generate(seed):
    rng = random.Random(seed)
    tree, scope = Gen(rng).body([], 0, 0)
    finish = [("look", v.name, v.kind) for v in scope if v.alive]

    def source(stmts, lang):
        lines = render(stmts + finish, lang, 1)
        if lang == "vx":
            return "\n".join([VX_HEAD] + lines + ['  print!("{}", s);', "  return 0;", "}", ""])
        return "\n".join([RS_HEAD] + lines + ['  print!("{}", s);', "}", ""])

    return Program(tree, source)
