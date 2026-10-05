"""Structs that implement `Drop` and print when they are dropped, made, moved and given new values.

`Noisy` holds an id and a tensor, and its `drop` prints the id. `Pair` holds two of them and
has no `drop` of its own, so dropping it drops its fields in the order they are declared.
`Boxed` holds one and prints its own number before its field is dropped. `main` makes these,
passes them to functions by value (which drop them) and by `&`, moves them inside `if`s and
loops, gives variables new values, and opens nested blocks. Every `drop` prints, so the output
is the order of the drops, which has to be Rust's: a block drops its variables in reverse
order, a moved value is not dropped, and a variable given a new value drops the old one.

The Rust twin is the same program, with `Vec<i32>` for the tensor.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program


class Var:
    def __init__(self, name, kind, loops):
        self.name = name
        self.kind = kind  # "n": Noisy, "p": Pair, "b": Boxed
        self.loops = loops
        self.alive = True


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.count = 0
        self.ids = 0

    def fresh(self):
        self.count += 1
        return f"v{self.count}"

    def new_id(self):
        self.ids += 1
        return self.ids

    def body(self, scope, loops, depth):
        scope = list(scope)
        own = []
        out = []
        for _ in range(self.rng.randint(1, 5)):
            alive = [v for v in scope if v.alive]
            movable = [v for v in alive if v.loops == loops]
            k = self.rng.random()
            if k < 0.25 or not alive:
                v = Var(self.fresh(), self.rng.choice("nnpb"), loops)
                out.append(("make", v.name, v.kind, self.new_id(), self.new_id()))
                scope.append(v)
                own.append(v)
            elif k < 0.37 and movable:
                v = self.rng.choice(movable)
                v.alive = False
                out.append(("take", v.name, v.kind))
            elif k < 0.5:
                v = self.rng.choice(alive)
                out.append(("look", v.name, v.kind))
            elif k < 0.58 and [v for v in movable if v.kind == "n"]:
                # Two Noisy moved into a Pair.
                ns = [v for v in movable if v.kind == "n"]
                if len(ns) >= 2:
                    a, b = self.rng.sample(ns, 2)
                    a.alive = b.alive = False
                    p = Var(self.fresh(), "p", loops)
                    out.append(("pair", p.name, a.name, b.name))
                    scope.append(p)
                    own.append(p)
            elif k < 0.66:
                if own:
                    v = self.rng.choice(own)
                    v.alive = True
                    out.append(("refill", v.name, v.kind, self.new_id(), self.new_id()))
            elif k < 0.8 and depth < 3:
                then_b = self.body(scope, loops, depth + 1)
                else_b = None
                if self.rng.random() < 0.5:
                    else_b = self.body(scope, loops, depth + 1)
                out.append(("if", self.rng.randint(0, 9), then_b, else_b))
            elif depth < 3:
                out.append(("for", f"i{depth}", self.rng.randint(0, 3),
                            self.body(scope, loops + 1, depth + 1)))
        return out


def make_expr(kind, a, b, lang):
    vx = lang == "vx"
    if kind == "n":
        return f"noisy({a})"
    if kind == "p":
        return f"Pair {{ a: noisy({a}), b: noisy({b}) }}"
    return f"Boxed {{ n: {b + 1000}, inner: noisy({a}) }}"


def render(stmts, lang, ind):
    pad = "  " * ind
    vx = lang == "vx"
    L = []
    for s in stmts:
        kind = s[0]
        if kind == "make":
            _, v, k, a, b = s
            L.append(f"{pad}let mut {v} = {make_expr(k, a, b, lang)};")
        elif kind == "take":
            _, v, k = s
            L.append(f"{pad}take_{k}({v});")
        elif kind == "look":
            _, v, k = s
            L.append(f"{pad}look_{k}(&{v});")
        elif kind == "pair":
            _, p, a, b = s
            L.append(f"{pad}let mut {p} = Pair {{ a: {a}, b: {b} }};")
        elif kind == "refill":
            _, v, k, a, b = s
            L.append(f"{pad}{v} = {make_expr(k, a, b, lang)};")
        elif kind == "if":
            L.append(f"{pad}if (k + {s[1]}) % 2 == 0 {{")
            L += render(s[2], lang, ind + 1)
            if s[3] is not None:
                L.append(f"{pad}}} else {{")
                L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
        elif kind == "for":
            L.append(f"{pad}for {s[1]} in 0..{s[2]} {{")
            L.append(f"{pad}  k = k + 1;")
            L += render(s[3], lang, ind + 1)
            L.append(f"{pad}}}")
    return L


VX_HEAD = """import core::ops;

struct Noisy {
  id : i32,
  t : Tensor<i32, [?]>,
}

impl Drop for Noisy {
  fn drop(self : &mut Noisy) -> void {
    print(self.id);
    print!(" ");
  }
}

struct Pair {
  a : Noisy,
  b : Noisy,
}

struct Boxed {
  n : i32,
  inner : Noisy,
}

impl Drop for Boxed {
  fn drop(self : &mut Boxed) -> void {
    print(self.n);
    print!(" ");
  }
}

fn noisy(id : i32) -> Noisy {
  return Noisy { id: id, t: Tensor<i32>([4]) };
}

fn take_n(x : Noisy) -> void {
  print!("t ");
}

fn take_p(x : Pair) -> void {
  print!("t ");
}

fn take_b(x : Boxed) -> void {
  print!("t ");
}

fn look_n(x : &Noisy) -> void {
  print!("l ");
}

fn look_p(x : &Pair) -> void {
  print!("l ");
}

fn look_b(x : &Boxed) -> void {
  print!("l ");
}

fn main() -> i32 {
  let mut k = 0;"""

RS_HEAD = """#![allow(unused, unused_mut, unused_variables, unused_assignments)]
struct Noisy { id: i32, t: Vec<i32> }
impl Drop for Noisy { fn drop(&mut self) { print!("{} ", self.id); } }
struct Pair { a: Noisy, b: Noisy }
struct Boxed { n: i32, inner: Noisy }
impl Drop for Boxed { fn drop(&mut self) { print!("{} ", self.n); } }
fn noisy(id: i32) -> Noisy { Noisy { id, t: vec![0; 4] } }
fn take_n(x: Noisy) { print!("t "); }
fn take_p(x: Pair) { print!("t "); }
fn take_b(x: Boxed) { print!("t "); }
fn look_n(x: &Noisy) { print!("l "); }
fn look_p(x: &Pair) { print!("l "); }
fn look_b(x: &Boxed) { print!("l "); }
fn main() {
  let mut k = 0i32;"""


def generate(seed):
    rng = random.Random(seed)
    tree = Gen(rng).body([], 0, 0)

    def source(stmts, lang):
        lines = render(stmts, lang, 1)
        if lang == "vx":
            return "\n".join([VX_HEAD] + lines + ['  print!("end");', "  return 0;", "}", ""])
        return "\n".join([RS_HEAD] + lines + ['  print!("end");', "}", ""])

    return Program(tree, source)
