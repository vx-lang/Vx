"""Tasks that add to atomics and a `Mutex`, send on channels, spawn tasks, join them and fail.

Each task is a function. It adds to an `AtomicI64`, to an atomic in an `Arc`, and to a total
in a `Mutex`, and sends numbers on one to three channels. It loops, branches, waits in a
`within` with a short time limit, and spawns tasks in a `tasks` block of its own. The code of
the main `tasks` block reads the channels, with `select` when there are two or more, and then
joins some tasks and adds up the values they give. A second block has a task that fails and
tasks that wait until the block cancels them.

Each total is a sum, so it does not depend on the order of the tasks. The generator works the
totals out itself, and the "Rust twin" is a program that prints them. The configurations are
one OS thread per task, the shared runtime, and the shared runtime with the program's seed as
`VX_TASK_SEED`. Each one must print the same totals.

`--heap` does not work for these programs yet. The task runtime keeps threads and stacks until
the program exits, and their number depends on how many tasks ran at the same time. The
output still finds two leaks: the count of owners of the `Arc` must be 1 at the end, and a
`Sender` that is never dropped keeps the receive loop waiting until the time limit.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random

from fuzzlib import Program

# `{seed}` is the seed of the program, so each program runs in an order of its own.
CONFIGS = [
    ("threads", [], {"VX_TASK_RUNTIME": "threads"}),
    ("shared", [], {"VX_TASK_RUNTIME": "shared"}),
    ("seed", [], {"VX_TASK_SEED": "{seed}"}),
]

ORDERS = ["Relaxed", "Release", "Acquire", "AcqRel", "SeqCst"]

# What `wait_for_nothing` gives for a wait that a cancel or a time limit ends.
CANCELLED = -2


class Gen:
    def __init__(self, rng, channels):
        self.rng = rng
        self.channels = channels

    def task(self, depth):
        return ("task", self.rng.random() < 0.6, self.rng.randint(0, 9), self.body(depth, 0))

    def body(self, depth, nest):
        out = []
        for _ in range(self.rng.randint(1, 5)):
            k = self.rng.random()
            if k < 0.25:
                out.append(("atomic", self.rng.choice(["atom", "arc"]),
                            self.rng.choice(["add", "sub"]), self.rng.randint(1, 9),
                            self.rng.choice(ORDERS)))
            elif k < 0.4:
                out.append(("lock", self.rng.randint(1, 9)))
            elif k < 0.6:
                out.append(("send", self.rng.randrange(self.channels), self.rng.randint(0, 99)))
            elif k < 0.7 and nest < 2:
                out.append(("for", self.rng.randint(0, 3), self.body(depth, nest + 1)))
            elif k < 0.8 and nest < 2:
                else_b = self.body(depth, nest + 1) if self.rng.random() < 0.5 else None
                out.append(("if", self.rng.randint(0, 9), self.body(depth, nest + 1), else_b))
            elif k < 0.85:
                out.append(("within", self.rng.randint(1, 20)))
            elif depth < 2:
                out.append(("tasks", [self.task(depth + 1)
                                      for _ in range(self.rng.randint(1, 3))]))
        return out


class Totals:
    """What the program must print, found by running the statement tree in Python."""

    def __init__(self):
        self.atom = 0
        self.arc = 0
        self.mutex = 0
        self.count = 0
        self.sum = 0

    def task(self, task):
        _, _, c, body = task
        state = {"k": 0, "got": 0}
        self.body(body, state)
        return state["got"] + c

    def body(self, stmts, state):
        for s in stmts:
            kind = s[0]
            if kind == "atomic":
                _, target, op, n, _ = s
                delta = n if op == "add" else -n
                if target == "atom":
                    self.atom += delta
                else:
                    self.arc += delta
            elif kind == "lock":
                self.mutex += s[1]
            elif kind == "send":
                self.count += 1
                self.sum += s[2]
            elif kind == "for":
                for _ in range(s[1]):
                    state["k"] += 1
                    self.body(s[2], state)
            elif kind == "if":
                if (state["k"] + s[1]) % 2 == 0:
                    self.body(s[2], state)
                elif s[3] is not None:
                    self.body(s[3], state)
            elif kind == "within":
                state["got"] += CANCELLED
            elif kind == "tasks":
                for child in s[1]:
                    value = self.task(child)
                    if child[1]:
                        state["got"] += value


VX_HEAD = """import core::result;
import core::sync::atomic;
import core::time;
import std::sync::arc;
import std::sync::channel;
import std::sync::mutex;
import std::sync::wait;
import std::vec;

fn put(tx : &Sender<i64>, v : i64) -> void {
  match tx.send(v) {
    Result<void, SendError<i64>>::Ok() => {}
    Result<void, SendError<i64>>::Err(_e) => {
      println!("a send failed");
    }
  }
}

fn add(m : &Mutex<i64>, v : i64) -> void {
  let mut guard = m.lock();
  let value = guard.get();
  *value = *value + v;
}

// -1 when the channel closed, -2 when the wait was cancelled or ran out of time.
fn wait_for_nothing(rx : Receiver<i64>) -> i64 {
  let mut out : i64 = 0;
  match rx.recv() {
    Result<i64, RecvError>::Ok(v) => {
      out = v;
    }
    Result<i64, RecvError>::Err(e) => {
      match e {
        RecvError::Closed() => {
          out = -1;
        }
        RecvError::Cancelled() => {
          out = -2;
        }
        RecvError::TimedOut() => {
          out = -3;
        }
      }
    }
  }
  return out;
}

fn fails(code : i64) -> Result<i64, i64> {
  return Result<i64, i64>::Err(code);
}
"""


class Render:
    def __init__(self, channels, capacities):
        self.channels = channels
        self.capacities = capacities
        self.functions = []
        self.count = 0
        self.waits = 0

    def senders(self):
        return [f"t{c}" for c in range(self.channels)]

    def spawn(self, task, pad, in_main):
        """The lines that clone what a task takes and spawn it. Returns (lines, handle)."""
        self.count += 1
        n = self.count
        self.function(task, n)
        L = []
        args = []
        for c in range(self.channels):
            L.append(f"{pad}let s{n}_{c} = {'tx' if in_main else 't'}{c}.clone();")
            args.append(f"s{n}_{c}")
        L.append(f"{pad}let a{n} = {'shared' if in_main else 'a'}.clone();")
        args += ["&atom", "&total"] if in_main else ["atom", "total"]
        args.append(f"a{n}")
        call = f"task_{n}({', '.join(args)})"
        if task[1]:
            L += [f"{pad}let h{n} = spawn {{", f"{pad}  {call}", f"{pad}}};"]
            return L, f"h{n}"
        L += [f"{pad}spawn {{", f"{pad}  {call};", f"{pad}}};"]
        return L, None

    def function(self, task, n):
        _, _, c, body = task
        params = [f"t{i} : Sender<i64>" for i in range(self.channels)]
        params += ["atom : &AtomicI64", "total : &Mutex<i64>", "a : Arc<AtomicI64>"]
        L = [f"fn task_{n}({', '.join(params)}) -> i64 {{",
             "  let mut k : i64 = 0;",
             "  let mut got : i64 = 0;"]
        L += self.body(body, 1)
        L += [f"  return got + {c};", "}", ""]
        self.functions.append("\n".join(L))

    def body(self, stmts, ind):
        pad = "  " * ind
        L = []
        for s in stmts:
            kind = s[0]
            if kind == "atomic":
                _, target, op, n, order = s
                on = "atom" if target == "atom" else "a.get()"
                L.append(f"{pad}{on}.fetch_{op}({n}, AtomicOrdering::{order});")
            elif kind == "lock":
                L.append(f"{pad}add(total, {s[1]});")
            elif kind == "send":
                L.append(f"{pad}put(&t{s[1]}, {s[2]});")
            elif kind == "for":
                L.append(f"{pad}for _i in 0..{s[1]} {{")
                L.append(f"{pad}  k = k + 1;")
                L += self.body(s[2], ind + 1)
                L.append(f"{pad}}}")
            elif kind == "if":
                L.append(f"{pad}if (k + {s[1]}) % 2 == 0 {{")
                L += self.body(s[2], ind + 1)
                if s[3] is not None:
                    L.append(f"{pad}}} else {{")
                    L += self.body(s[3], ind + 1)
                L.append(f"{pad}}}")
            elif kind == "within":
                self.waits += 1
                w = self.waits
                L += [f"{pad}let (_never{w}, rx{w}) = channel<i64>(1);",
                      f"{pad}got = got + within(Duration::from_millis({s[1]}), "
                      f"|| -> i64 {{ wait_for_nothing(rx{w}) }});"]
            elif kind == "tasks" and s[1]:
                L.append(f"{pad}tasks {{")
                handles = []
                for child in s[1]:
                    lines, handle = self.spawn(child, pad + "  ", False)
                    L += lines
                    if handle:
                        handles.append(handle)
                L += [f"{pad}  got = got + {h}.join();" for h in handles]
                L.append(f"{pad}}};")
        return L

    def main(self, tree):
        L = ["fn main() -> i32 {",
             "  let atom = AtomicI64::new(0);",
             "  let shared = Arc<AtomicI64>::new(AtomicI64::new(0));",
             "  let total = Mutex<i64>::new(0);",
             "  let waits = AtomicI64::new(0);"]
        for c in range(self.channels):
            L.append(f"  let (tx{c}, rx{c}) = channel<i64>({self.capacities[c]});")
        L += ["  let mut count : i64 = 0;",
              "  let mut sum : i64 = 0;",
              "  let mut joined : i64 = 0;",
              "  tasks {"]
        handles = []
        for s in tree:
            if s[0] == "task":
                lines, handle = self.spawn(s, "    ", True)
                L += lines
                if handle:
                    handles.append(handle)
        L += [f"    drop(tx{c});" for c in range(self.channels)]
        L += self.receive()
        L += [f"    joined = joined + {h}.join();" for h in handles]
        L.append("  };")
        fail = [s for s in tree if s[0] == "fail"]
        for s in fail:
            L += self.fail(s)
        L.append('  println!("atomic {} arc {} owners {} mutex {} count {} sum {} joined {}", '
                 "atom.load(AtomicOrdering::SeqCst), shared.get().load(AtomicOrdering::SeqCst), "
                 "shared.strong_count(), total.into_inner(), count, sum, joined);")
        L += ["  return 0;", "}", ""]
        return L

    def receive(self):
        """The loop of the main block that reads each channel until each one closes."""
        if self.channels == 1:
            return ["    let mut open = true;",
                    "    while open {",
                    "      match rx0.recv() {",
                    "        Result<i64, RecvError>::Ok(v) => {",
                    "          count = count + 1;",
                    "          sum = sum + v;",
                    "        }",
                    "        Result<i64, RecvError>::Err(_e) => {",
                    "          open = false;",
                    "        }",
                    "      }",
                    "    }"]
        L = ["    let mut rs = Vec<Receiver<i64>>::new();"]
        L += [f"    rs.push(rx{c});" for c in range(self.channels)]
        return L + ["    let mut sel = Selector<i64>::new(rs);",
                    "    let mut open = true;",
                    "    while open {",
                    "      match sel.select() {",
                    "        Result<Selected<i64>, RecvError>::Ok(s) => {",
                    "          count = count + 1;",
                    "          sum = sum + s.value;",
                    "        }",
                    "        Result<Selected<i64>, RecvError>::Err(_e) => {",
                    "          open = false;",
                    "        }",
                    "      }",
                    "    }"]

    def fail(self, s):
        """A block where one task fails and the others wait until the block cancels them."""
        _, code, waiters, position = s
        L = [f"  let (keep{w}, wait{w}) = channel<i64>(1);" for w in range(waiters)]
        L.append("  let failed = tasks {")
        for w in range(waiters + 1):
            if w == position:
                L += ["    let f = spawn {", f"      fails({code})", "    };"]
            if w < waiters:
                L += ["    spawn {",
                      f"      waits.fetch_add(wait_for_nothing(wait{w}), AtomicOrdering::Relaxed);",
                      "    };"]
        L += ["    let v = f.join();", "    Result<i64, i64>::Ok(v)", "  };"]
        L += [f"  drop(keep{w});" for w in range(waiters)]
        L += ["  match failed {",
              "    Result<i64, i64>::Ok(v) => {",
              '      println!("the block gave Ok {}", v);',
              "    }",
              "    Result<i64, i64>::Err(e) => {",
              '      println!("the block gave Err {}, waits {}", e, '
              "waits.load(AtomicOrdering::SeqCst));",
              "    }",
              "  }"]
        return L


def expected(tree):
    totals = Totals()
    joined = 0
    for s in tree:
        if s[0] == "task":
            value = totals.task(s)
            if s[1]:
                joined += value
    lines = [f"the block gave Err {s[1]}, waits {CANCELLED * s[2]}"
             for s in tree if s[0] == "fail"]
    lines.append(f"atomic {totals.atom} arc {totals.arc} owners 1 mutex {totals.mutex} "
                 f"count {totals.count} sum {totals.sum} joined {joined}")
    return "\n".join(lines)


def generate(seed):
    rng = random.Random(seed)
    channels = rng.randint(1, 3)
    capacities = [rng.randint(1, 3) for _ in range(channels)]
    gen = Gen(rng, channels)
    tree = [gen.task(0) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.4:
        waiters = rng.randint(0, 3)
        tree.append(("fail", rng.randint(1, 99), waiters, rng.randint(0, waiters)))

    def source(stmts, lang):
        if lang == "rs":
            text = expected(stmts).replace("\n", "\\n")
            return f'fn main() {{\n    print!("{text}");\n}}\n'
        r = Render(channels, capacities)
        main = r.main(stmts)
        return "\n".join([VX_HEAD] + r.functions + main)

    return Program(tree, source)
