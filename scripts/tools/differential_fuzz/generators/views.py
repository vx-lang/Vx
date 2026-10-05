"""Rows and reshapes of two tensors, used while the borrow rules allow it.

`main` makes two 3x4 tensors `q` and `p` and a running total `s`. It takes views of them: a
row `q[i]`, shared or `mut`, a reshape, a row chosen by an `if`, and a row taken through a
reference. It writes and reads through each view, passes rows to functions by `&` and `&mut`,
and clones rows, inside nested `if`s and `for` loops. At the end it prints `s`, every element,
and the total of `q` passed by value.

It also copies rows into rows (`p[i] = q[k]`) and reads a row through a closure.

Each view is used only in the statements right after it is made. While a `mut` view is used,
its tensor is not touched in any other way; while a shared view is used, its tensor is only
read. So every program keeps to the borrow rules of both Vx and Rust: a program Vx refuses is
a borrow checker bug, and a different output is a code generation bug. The Rust twin uses
`[[i32; 4]; 3]` arrays, `&q[i]`, `&mut q[i]` and `q.as_flattened()`.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program

ROWS, COLS = 3, 4
TENSORS = ["q", "p"]


class Gen:
    def __init__(self, rng):
        self.rng = rng
        self.count = 0

    def fresh(self, prefix):
        self.count += 1
        return f"{prefix}{self.count}"

    def val(self, loops):
        """A small number: a constant, or a loop variable plus one."""
        if loops and self.rng.random() < 0.4:
            v = self.rng.choice(loops)
            return ("var", v, self.rng.randint(0, 3))
        return ("num", self.rng.randint(0, 9))

    def index(self, loops, size):
        """An index below `size`: a constant, or a loop variable (every loop runs fewer than
        3 times, so it is below both sizes)."""
        if loops and self.rng.random() < 0.4:
            return ("var", self.rng.choice(loops), 0)
        return ("num", self.rng.randrange(size))

    def view_uses(self, view, tensor, loops, writes):
        """What happens while a view is used: reads through it, writes through it when
        `writes`, and reads of the other tensor, or of its own when it is shared."""
        out = []
        for _ in range(self.rng.randint(1, 3)):
            r = self.rng.random()
            if writes and r < 0.4:
                out.append(("vwrite", view, self.index(loops, COLS), self.val(loops)))
            elif r < 0.8:
                out.append(("vread", view, self.index(loops, COLS)))
            else:
                other = [t for t in TENSORS if t != tensor or not writes]
                out.append(("read", self.rng.choice(other), self.index(loops, ROWS),
                            self.index(loops, COLS)))
        return out

    def body(self, loops, depth):
        out = []
        for _ in range(self.rng.randint(1, 4)):
            k = self.rng.random()
            t = self.rng.choice(TENSORS)
            row = self.index(loops, ROWS)
            if k < 0.12:
                out.append(("write", t, row, self.index(loops, COLS), self.val(loops)))
            elif k < 0.2:
                out.append(("read", t, row, self.index(loops, COLS)))
            elif k < 0.35:
                mut = self.rng.random() < 0.5
                v = self.fresh("r")
                out.append(("row", v, t, row, mut, self.view_uses(v, t, loops, mut)))
            elif k < 0.42:
                v = self.fresh("f")
                uses = [("vread", v, ("num", self.rng.randrange(ROWS * COLS)))
                        for _ in range(self.rng.randint(1, 2))]
                out.append(("flat", v, t, uses))
            elif k < 0.49:
                v = self.fresh("c")
                out.append(("pick", v, t, self.val(loops), row, self.index(loops, ROWS),
                            self.view_uses(v, t, loops, False)))
            elif k < 0.56:
                v = self.fresh("w")
                out.append(("viaref", self.fresh("rq"), v, t, row,
                            self.view_uses(v, t, loops, False)))
            elif k < 0.63:
                out.append(("byref", t, row))
            elif k < 0.7:
                out.append(("bymut", t, row, self.val(loops)))
            elif k < 0.74:
                out.append(("rowcopy", t, row, self.rng.choice(TENSORS), self.index(loops, ROWS)))
            elif k < 0.77:
                v = self.fresh("k")
                uses = self.view_uses(v, "", loops, True)
                # A clone is a tensor of its own: its tensor may change while it is used.
                uses.insert(self.rng.randint(0, len(uses)),
                            ("write", t, row, self.index(loops, COLS), self.val(loops)))
                out.append(("clone", v, t, row, uses))
            elif k < 0.82:
                v = self.fresh("h")
                out.append(("closure", v, self.fresh("g"), t, row, self.index(loops, COLS)))
            elif k < 0.87 and depth < 3:
                out.append(("if", self.val(loops), self.body(loops, depth + 1),
                            self.body(loops, depth + 1) if self.rng.random() < 0.5 else None))
            elif depth < 3:
                lv = f"i{depth}"
                out.append(("for", lv, self.rng.randint(0, 3), self.body(loops + [lv], depth + 1)))
        return out


def num(e, lang):
    """A value: a constant, or a loop variable plus a constant. A Rust loop variable is a
    `usize`, so it is cast where it becomes a number."""
    if e[0] == "num":
        return str(e[1])
    var = e[1] if lang == "vx" else f"({e[1]} as i32)"
    return f"({var} + {e[2]})" if e[2] else var


def idx(e):
    """An index: a constant, or a loop variable, which is already a `usize` in Rust."""
    return str(e[1])


def render(stmts, lang, ind):
    pad = "  " * ind
    vx = lang == "vx"
    L = []
    for s in stmts:
        kind = s[0]
        if kind == "write":
            _, t, i, j, v = s
            L.append(f"{pad}{t}[{idx(i)}][{idx(j)}] = {num(v, lang)};")
        elif kind == "read":
            _, t, i, j = s
            L.append(f"{pad}s = s + {t}[{idx(i)}][{idx(j)}];")
        elif kind == "vread":
            L.append(f"{pad}s = s + {s[1]}[{idx(s[2])}];")
        elif kind == "vwrite":
            _, v, j, x = s
            L.append(f"{pad}{v}[{idx(j)}] = {v}[{idx(j)}] + {num(x, lang)};")
        elif kind == "row":
            _, v, t, i, mut, uses = s
            if vx:
                L.append(f"{pad}let {'mut ' if mut else ''}{v} = {t}[{idx(i)}];")
            else:
                L.append(f"{pad}let {v} = &{'mut ' if mut else ''}{t}[{idx(i)}];")
            L += render(uses, lang, ind)
        elif kind == "flat":
            _, v, t, uses = s
            made = f"{t}.reshape([{ROWS * COLS}])" if vx else f"{t}.as_flattened()"
            L.append(f"{pad}let {v} = {made};")
            L += render(uses, lang, ind)
        elif kind == "pick":
            _, v, t, c, i1, i2, uses = s
            amp = "" if vx else "&"
            L.append(f"{pad}let {v} = if ({num(c, lang)}) % 2 == 0 {{ {amp}{t}[{idx(i1)}] }} "
                     f"else {{ {amp}{t}[{idx(i2)}] }};")
            L += render(uses, lang, ind)
        elif kind == "viaref":
            _, rq, v, t, i, uses = s
            L.append(f"{pad}let {rq} = &{t};")
            L.append(f"{pad}let {v} = {'' if vx else '&'}{rq}[{idx(i)}];")
            L += render(uses, lang, ind)
        elif kind == "rowcopy":
            _, dst, i, src, k = s
            L.append(f"{pad}{dst}[{idx(i)}] = {src}[{idx(k)}];")
        elif kind == "closure":
            _, v, f, t, i, j = s
            L.append(f"{pad}let {v} = {'' if vx else '&'}{t}[{idx(i)}];")
            L.append(f"{pad}let {f} = || {v}[{idx(j)}];")
            L.append(f"{pad}s = s + {f}();")
        elif kind == "byref":
            L.append(f"{pad}s = s + rsum(&{s[1]}[{idx(s[2])}]);")
        elif kind == "bymut":
            L.append(f"{pad}rbump(&mut {s[1]}[{idx(s[2])}], {num(s[3], lang)});")
        elif kind == "clone":
            _, v, t, i, uses = s
            made = f"{t}[{idx(i)}].clone()" if vx else f"{t}[{idx(i)}]"
            L.append(f"{pad}let mut {v} = {made};")
            L += render(uses, lang, ind)
        elif kind == "if":
            L.append(f"{pad}if ({num(s[1], lang)}) % 2 == 0 {{")
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


VX_HEAD = """fn rsum(v : &Tensor<i32, [4]>) -> i32 {
  return v[0] + v[1] + v[2] + v[3];
}

fn rbump(v : &mut Tensor<i32, [4]>, x : i32) -> void {
  for j in 0..4 {
    v[j] = v[j] + x;
  }
}

fn total(t : Tensor<i32, [3, 4]>) -> i32 {
  let mut s = 0;
  for i in 0..3 {
    for j in 0..4 {
      s = s + t[i][j];
    }
  }
  return s;
}

fn main() -> i32 {
  let mut q = Tensor<i32, [3, 4]>::new();
  let mut p = Tensor<i32, [3, 4]>::new();
  let mut s = 0;"""

VX_TAIL = """  print!("{} ", s);
  for i in 0..3 {
    for j in 0..4 {
      print!("{} ", q[i][j]);
    }
  }
  for i in 0..3 {
    for j in 0..4 {
      print!("{} ", p[i][j]);
    }
  }
  print!("{}", total(q));
  return 0;
}
"""

RS_HEAD = """#![allow(unused, unused_mut, unused_variables, unused_assignments)]
fn rsum(v: &[i32; 4]) -> i32 { v[0] + v[1] + v[2] + v[3] }
fn rbump(v: &mut [i32; 4], x: i32) { for j in 0..4 { v[j] = v[j] + x; } }
fn total(t: [[i32; 4]; 3]) -> i32 {
  let mut s = 0;
  for i in 0..3 { for j in 0..4 { s = s + t[i][j]; } }
  s
}
fn main() {
  let mut q = [[0i32; 4]; 3];
  let mut p = [[0i32; 4]; 3];
  let mut s = 0i32;"""

RS_TAIL = """  print!("{} ", s);
  for i in 0..3 { for j in 0..4 { print!("{} ", q[i][j]); } }
  for i in 0..3 { for j in 0..4 { print!("{} ", p[i][j]); } }
  print!("{}", total(q));
}
"""


def generate(seed):
    rng = random.Random(seed)
    tree = Gen(rng).body([], 0)

    def source(stmts, lang):
        head, tail = (VX_HEAD, VX_TAIL) if lang == "vx" else (RS_HEAD, RS_TAIL)
        return "\n".join([head] + render(stmts, lang, 1) + [tail])

    return Program(tree, source)
