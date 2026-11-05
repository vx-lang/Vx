#!/usr/bin/env python3
"""Check that the Lean model and vx.cat agree on the litmus tests.

For each test without branches, this script lists every candidate execution: each read takes
its value from one write to its place, and the writes to each place come in some order. Lean
decides which executions the model allows and which have a data race. It uses the definitions
of lean/VxMemoryModel/Scopes.lean, and for a test without scopes and barriers also those of
lean/VxMemoryModel/Rc11.lean. The final states, the observation and the race flag must be the
same as those of herd7 on vx.cat.

Usage: run_lean.py [test.litmus ...]. Needs lake (Lean 4) and herd7.
"""

import itertools
import os
import re
import shutil
import subprocess
import sys
import tempfile

import run_litmus

HERE = os.path.dirname(os.path.abspath(__file__))
LEAN = os.path.join(HERE, "lean")
ORDERS = {"plain": ".plain", "rlx": ".rlx", "acq": ".acq", "rel": ".rel",
          "acq_rel": ".acqRel", "sc": ".sc", "barrier": ".plain"}
LEVELS = ["subgroup", "block", "device", "system"]
INSTR = re.compile(r"^(r|w|f|rmw)\[(\w+)(?:,(\w+))?\]\s*(.*)$")
RMW_ADD = re.compile(r"^(\w+)\s+\(add\s+(\w+)\s+(-?\d+)\)\s+(\w+)$")


def parse(text):
    """The initial values, the code of each thread, and the filter and exists conditions."""
    init = dict((k, int(v)) for k, v in
                re.findall(r"(\w+)\s*=\s*(-?\d+)", re.search(r"\{(.*?)\}", text, re.S).group(1)))
    lines = [line for line in text.split("\n") if line.strip().endswith(";") and "|" in line]
    threads = [[] for _ in lines[0].split("|")]
    for line in lines[1:]:
        for t, cell in enumerate(line.rstrip().rstrip(";").split("|")):
            cell = cell.strip()
            if cell:
                kind, order, scope, args = INSTR.match(cell).groups()
                threads[t].append((kind, order, scope or "system", args.split()))
    filt = re.search(r"^filter\s*\((.*)\)\s*$", text, re.M)
    exists = re.search(r"^exists\s*\((.*)\)\s*$", text, re.M).group(1)
    tree = re.search(r"^scopes:\s*(.*)$", text, re.M).group(1)
    return init, threads, filt.group(1) if filt else None, exists, groups(tree, len(threads))


def groups(tree, n):
    """The subgroup, block and device of each thread, from the scope tree. As in herd7, two
    threads share a group of a scope when a node of that scope or a narrower one holds both.
    A thread outside every such node gets a group of its own."""
    tokens = re.findall(r"\(|\)|\w+", tree)
    result = [[None] * 3 for _ in range(n)]
    count = [0]

    def node(i, outer):
        # tokens[i] is "(", tokens[i + 1] the scope of the node.
        count[0] += 1
        ident, level = count[0], LEVELS.index(tokens[i + 1])
        # For each scope from this one up, the outermost node of that scope or narrower.
        mine = [o if o is not None else (ident if k >= level else None)
                for k, o in enumerate(outer)]
        i += 2
        while tokens[i] != ")":
            if tokens[i] == "(":
                i = node(i, mine)
            else:
                result[int(tokens[i][1:])] = mine[:]
                i += 1
        return i + 1

    node(0, [None] * 3)
    return [tuple(g if g is not None else 10000 + 3 * t + k for k, g in enumerate(gs))
            for t, gs in enumerate(result)]


def variables(cond):
    return re.findall(r"(\d+:\w+|\w+)\s*=", cond)


def holds(cond, state):
    disjuncts = cond.split("\\/")
    return any(all(state[v.strip()] == int(n) for v, n in
                   (atom.split("=") for atom in d.split("/\\"))) for d in disjuncts)


def candidates(init, threads):
    """Every candidate execution: its events, and its rf and co edges."""
    # An event is (kind, order, place, value, thread, scope). The value of a write is a number,
    # a register, or ("add", index of the read, number) for a read-modify-write.
    events = [("w", "plain", loc, init[loc], None, "system") for loc in sorted(init)]
    first = len(events)
    rmw = []
    for t, code in enumerate(threads):
        for kind, order, scope, args in code:
            if kind == "f":
                events.append(("f", order, None, None, t, scope))
            elif kind == "rmw":
                reg, reg2, k, loc = RMW_ADD.match(" ".join(args)).groups()
                assert reg == reg2, "a read-modify-write must add to the value it reads"
                events.append(("r", order, loc, reg, t, scope))
                events.append(("w", order, loc, ("add", len(events) - 1, int(k)), t, scope))
                rmw.append((len(events) - 2, len(events) - 1))
            else:
                loc, val = args if kind == "w" else args[::-1]
                events.append((kind, order, loc, val, t, scope))
    reads = [i for i, e in enumerate(events) if e[0] == "r"]
    writes = {loc: [i for i, e in enumerate(events) if e[0] == "w" and e[2] == loc] for loc in init}
    rf_choices = [writes[events[r][2]] for r in reads]
    co_choices = [[(w[0],) + p for p in itertools.permutations(w[1:])] for w in writes.values()]
    for sources in itertools.product(*rf_choices):
        for orders in itertools.product(*co_choices):
            rf = list(zip(sources, reads))
            co = [(o[i], o[j]) for o in orders for i in range(len(o)) for j in range(i + 1, len(o))]
            yield events, first, rf, co, rmw, orders


def final_state(events, threads, rf, orders, names):
    """The values of the registers and places in `names`, or None if a value depends on itself."""
    source = {r: w for w, r in rf}
    reg_of = {}
    for i, e in enumerate(events):
        if e[0] == "r":
            reg_of[(e[4], e[3])] = i

    def value(i, seen):
        if i in seen:
            return None
        e = events[i]
        if e[0] == "r":
            return value(source[i], seen | {i})
        if isinstance(e[3], tuple):
            v = value(e[3][1], seen | {i})
            return None if v is None else v + e[3][2]
        if isinstance(e[3], int) or str(e[3]).lstrip("-").isdigit():
            return int(e[3])
        return value(reg_of[(e[4], e[3])], seen | {i})

    state = {}
    for name in names:
        if ":" in name:
            t, reg = name.split(":")
            state[name] = value(reg_of[(int(t), reg)], set())
        else:
            last = next(o[-1] for o in orders if events[o[0]][2] == name)
            state[name] = value(last, set())
    return state


def lean_data(events, first, rf, co, rmw, groups):
    """The execution in Lean. With `groups`, an SData for Scopes.lean, else a Data."""
    n = len(events)
    kinds = ", ".join({"r": ".read", "w": ".write", "f": ".fence"}[e[0]] for e in events)
    orders = ", ".join(ORDERS[e[1]] for e in events)
    # A fence has no place. Give each fence a place of its own, so that it shares none.
    locs = sorted({e[2] for e in events if e[2] is not None})
    loc = ", ".join(str(locs.index(e[2]) if e[2] is not None else len(locs) + i)
                    for i, e in enumerate(events))
    thread = ", ".join(str(1000 + i if e[4] is None else e[4]) for i, e in enumerate(events))
    po = [(i, j) for i in range(first) for j in range(first, n)]
    po += [(i, j) for i in range(first, n) for j in range(i + 1, n)
           if events[i][4] == events[j][4]]
    pairs = lambda ps: "[" + ", ".join(f"({a}, {b})" for a, b in ps) + "]"
    common = (f"[{kinds}] [{orders}] [{loc}] [{thread}] "
              f"{pairs(po)} {pairs(rf)} {pairs(co)} {pairs(rmw)}")
    if groups is None:
        return f"(Data.ofLists {n} {common}).check"
    scopes = ", ".join("." + e[5] for e in events)
    groups = ", ".join(f"({a}, {b}, {c})" for a, b, c in groups)
    barrier = ", ".join("true" if e[1] == "barrier" else "false" for e in events)
    return (f"(SData.ofLists {n} {common} [{scopes}] [{groups}] [{barrier}]).check")


def lean_check(datas):
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as f:
        f.write("import VxMemoryModel.ScopedCheck\nopen Vx\n")
        for d in datas:
            f.write(f"#eval {d}\n")
        path = f.name
    try:
        out = subprocess.run(["lake", "-d", LEAN, "env", "lean", path], capture_output=True,
                             text=True, check=False)
    finally:
        os.unlink(path)
    results = re.findall(r"^(none|some \((true|false), (true|false)\))$", out.stdout, re.M)
    if len(results) != len(datas):
        sys.exit(f"lean gave {len(results)} results for {len(datas)} executions:\n"
                 f"{out.stdout}{out.stderr}")
    return [None if r[0] == "none" else (r[1] == "true", r[2] == "true") for r in results]


def scoped_test(text):
    """A test with a scope tag or a barrier needs the definitions of Scopes.lean."""
    return re.search(r"\[\w+,\w+\]", text) is not None


def check(herd7, test):
    """A test with a scope tag or a barrier needs Scopes.lean. A test for CPUs is checked with
    both Rc11.lean and Scopes.lean, which must agree on it."""
    text = open(test).read()
    init, threads, filt, exists, groups = parse(text)
    herd = run_litmus.herd(herd7, "vx.cat", test)
    errors = []
    for model in (["Scopes"] if scoped_test(text) else ["Rc11", "Scopes"]):
        found = compare(herd, init, threads, filt, exists, groups if model == "Scopes" else None)
        errors += [f"{model}.lean: {e}" for e in found]
    return errors


def compare(herd, init, threads, filt, exists, groups):
    names = sorted(set(variables(exists) + (variables(filt) if filt else [])))
    shown = sorted(set(variables(exists)))
    cands = list(candidates(init, threads))
    results = lean_check([lean_data(e, f, rf, co, rmw, groups) for e, f, rf, co, rmw, _ in cands])
    states, race = set(), False
    for (events, _, rf, _, _, orders), result in zip(cands, results):
        if result is None:
            return ["a closure in Lean did not converge"]
        allowed, racy = result
        state = final_state(events, threads, rf, orders, names)
        if None in state.values():
            if allowed:
                return ["Lean allows an execution where a value depends on itself"]
            continue
        if allowed and (filt is None or holds(filt, state)):
            states.add(tuple(f"{v}={state[v]}" for v in shown))
            race = race or racy
    seen = [holds(exists, {k: int(v) for k, v in (s.split("=") for s in st)}) for st in states]
    observation = "never" if not any(seen) else "always" if all(seen) else "sometimes"
    herd_states = sorted(tuple(sorted(x.strip() for x in s.split(";") if x.strip()))
                         for s in herd["states"])
    errors = []
    if sorted(tuple(sorted(s)) for s in states) != herd_states:
        errors.append(f"final states: Lean {sorted(states)}, herd7 {herd_states}")
    if observation != herd["observation"]:
        errors.append(f"observation: Lean {observation}, herd7 {herd['observation']}")
    if race != herd["race"]:
        errors.append(f"data race: Lean {race}, herd7 {herd['race']}")
    return errors


def main():
    herd7 = os.environ.get("HERD7") or shutil.which("herd7")
    if not herd7 or not shutil.which("lake"):
        sys.exit("herd7 and lake must be on PATH.")
    subprocess.run(["lake", "-d", LEAN, "build", "-q", "VxMemoryModel.ScopedCheck"], check=True)
    tests = sys.argv[1:] or sorted(
        os.path.join(HERE, "tests", f) for f in os.listdir(os.path.join(HERE, "tests"))
        if f.endswith(".litmus"))
    # The script lists the events of each thread once, so it cannot follow a branch.
    tests = [t for t in tests if not re.search(r"\bb\[", open(t).read())]
    failed = 0
    for test in tests:
        errors = check(herd7, test)
        name = os.path.basename(test)
        failed += bool(errors)
        print(f"FAIL {name}: " + "; ".join(errors) if errors else f"ok   {name}")
    print(f"{len(tests) - failed} of {len(tests)} tests agree")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
