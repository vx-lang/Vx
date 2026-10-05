"""The parts of the differential fuzzer every generator shares.

A *generator* is a module in `generators/`, whose docstring describes it, with a function
`generate(seed) -> Program`. A `Program` holds a statement tree and knows how to print it as Vx
and, when there is one, as a Rust twin that must print exactly the same thing.

A *configuration* is a name and the `vxc` flags it adds, such as `ast=--legacy-codegen`, after
any environment variables it sets, such as `drops=VX_DROPS=scope`. A program is run once per
configuration; its outputs must all agree with each other and with Rust.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import importlib
import os
import re
import shlex
import subprocess
import tempfile

DEFAULT_CONFIGS = [("flat", [], {}), ("ast", ["--legacy-codegen"], {})]

# Counts the heap blocks a compiled program allocates and frees: see heap_count.c.
HEAP_COUNTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heap_count.c")
EMPTY_PROGRAM = "fn main() -> i32 {\n  print(1);\n  return 0;\n}\n"


class Program:
    """A generated program: a statement tree, and how to print it.

    Statements are tuples whose first element names the kind (`("let", "x", True, "3")`). A
    list inside a statement whose items are statements is a block; a list of blocks (the arms
    of a `match`) works too. That is all the reducer needs to know: it deletes statements
    from blocks. `keep(stmt, block)` names statements it must not delete.
    """

    def __init__(self, tree, render, keep=None):
        self.tree = tree
        self._render = render
        self._keep = keep or (lambda stmt, block: False)

    def source(self, lang):
        """The program as `lang` ("vx" or "rs"), or None when there is no Rust twin."""
        return self._render(self.tree, lang)

    def keep(self, stmt, block):
        return self._keep(stmt, block)

    def blocks(self):
        """Every list of statements in the tree, outermost first."""
        def walk(block):
            yield block
            for stmt in block:
                yield from inner(stmt)

        def inner(value):
            if isinstance(value, tuple):
                for part in value:
                    yield from inner(part)
            elif isinstance(value, list) and value and all(isinstance(s, tuple) for s in value):
                yield from walk(value)
            elif isinstance(value, list):
                for part in value:
                    yield from inner(part)

        yield from walk(self.tree)


def load_generator(name):
    return importlib.import_module(f"generators.{name}")


def generator_names():
    here = os.path.join(os.path.dirname(__file__), "generators")
    return sorted(f[:-3] for f in os.listdir(here) if f.endswith(".py") and f != "__init__.py")


def parse_config(text):
    """`name=flags` as given on the command line, e.g. `o0=-O0` or `ast=--legacy-codegen`.
    Words before the flags that look like `VAR=value` set the environment, as `env` does:
    `drops=VX_DROPS=scope --legacy-codegen`."""
    name, _, rest = text.partition("=")
    words = shlex.split(rest)
    env = {}
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
        var, _, value = words.pop(0).partition("=")
        env[var] = value
    return name, words, env


def build_heap_counter(workdir):
    """Compile the heap counter into `workdir`. Returns the library's path."""
    library = os.path.join(workdir, "heap_count.so")
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", library, HEAP_COUNTER], check=True)
    return library


# Printed by `print` of a tensor (`base@ = 0x5b78d8e168c0`), different on every run.
ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def clean(output, ignore=()):
    """What a program printed, without the compiler's progress lines or its warnings (with
    their `help:` lines), and with addresses and anything matching an `ignore` pattern
    replaced by `?`."""
    keep = [line for line in output.splitlines()
            if not line.startswith(("[", "Warning", "  help:"))]
    text = ADDRESS.sub("0x?", "\n".join(keep).strip())
    for pattern in ignore:
        text = re.sub(pattern, "?", text)
    return text


def run_vx(vxc, flags, path, timeout, ignore=(), env=None, heap=None):
    """What the program prints. With `heap`, the counter library, also the heap blocks the
    compiled program left allocated when it exited: `(output, blocks)`, with `None` blocks when
    it did not exit normally."""
    env = dict(os.environ, **(env or {}))
    counts = None
    if heap:
        counts = path + ".heap"
        if os.path.exists(counts):
            os.remove(counts)
        env.update(LD_PRELOAD=heap, VX_HEAP_COUNT_FILE=counts)
    try:
        out = subprocess.run([vxc, *flags, path], capture_output=True, text=True,
                             timeout=timeout, env=env)
        text = clean(out.stdout + out.stderr, ignore)
    except subprocess.TimeoutExpired:
        text = "<timeout>"
    if not heap:
        return text
    left = None
    if os.path.exists(counts):
        with open(counts) as f:
            allocs, frees = map(int, f.read().split()[-2:])
        left = allocs - frees
    return text, left


def empty_program_blocks(vxc, configs, heap, timeout):
    """The heap blocks an empty program leaves under each configuration: what the runtime
    itself keeps, such as the buffer behind standard output."""
    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "empty.vx")
        with open(path, "w") as f:
            f.write(EMPTY_PROGRAM)
        return {name: run_vx(vxc, flags, path, timeout, env=env, heap=heap)[1]
                for name, flags, env in configs}


def run_rust(source, workdir, timeout):
    """Compile and run the Rust twin. Returns its output, or None if rustc refuses it."""
    rs = os.path.join(workdir, "twin.rs")
    binary = os.path.join(workdir, "twin")
    with open(rs, "w") as f:
        f.write(source)
    built = subprocess.run(["rustc", "-O", "-o", binary, rs], capture_output=True, text=True)
    if built.returncode != 0:
        return None
    try:
        out = subprocess.run([binary], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "<timeout>"
    return (out.stdout + out.stderr).strip()


def run_all(vx_source, rs_source, vxc, configs, timeout, workdir=None, ignore=(), heap=None):
    """Run one program every way. Returns {name: output}, with "rust" when there is a twin.

    With `heap`, the counter library and the blocks an empty program leaves under each
    configuration, a run that leaves a different number gets an entry `name:heap` saying
    how many blocks it leaked."""
    workdir = workdir or tempfile.mkdtemp()
    path = os.path.join(workdir, "program.vx")
    with open(path, "w") as f:
        f.write(vx_source)
    outputs = {}
    if rs_source is not None:
        rust = run_rust(rs_source, workdir, timeout)
        outputs["rust"] = "<rustc refused the twin>" if rust is None else rust
    for name, flags, env in configs:
        if not heap:
            outputs[name] = run_vx(vxc, flags, path, timeout, ignore, env)
            continue
        library, empty = heap
        outputs[name], left = run_vx(vxc, flags, path, timeout, ignore, env, library)
        if left is not None and empty.get(name) is not None and left != empty[name]:
            extra = left - empty[name]
            outputs[f"{name}:heap"] = (f"leaks {extra} blocks" if extra > 0
                                       else f"frees {-extra} blocks more than it allocates")
    return outputs


def disagreement(outputs):
    """The names whose output differs from the reference: Rust when there is a twin, else the
    first configuration. An empty tuple means every run agreed."""
    if outputs.get("rust") == "<rustc refused the twin>":
        return ("rust-refused",)
    leaks = tuple(n for n in outputs if n.endswith(":heap"))
    names = [n for n in outputs if n not in leaks]
    reference = outputs["rust"] if "rust" in outputs else outputs[names[0]]
    return tuple(n for n in names if n != "rust" and outputs[n] != reference) + leaks


def symptom(outputs, names):
    """A short label for grouping failures: who disagrees, and how. Output made only of numbers
    is a wrong answer; anything else is labelled by its first line with the digits hidden, so
    the same error at different places groups together."""
    first = outputs[names[0]].splitlines()[0] if names and outputs.get(names[0]) else ""
    if all(c.isdigit() or c in " -.,e+" for c in first):
        how = "wrong output" if first else "no output"
    else:
        how = "".join("#" if c.isdigit() else c for c in first)[:80]
    return f"{'+'.join(names)}: {how}"


def reduce(program, still_fails):
    """Delete statements from `program` for as long as `still_fails(program)` stays true.

    Each pass tries every statement of every block, outermost blocks first, and keeps a
    deletion when the program still fails. Passes repeat until one deletes nothing.
    """
    changed = True
    while changed:
        changed = False
        for block in list(program.blocks()):
            i = 0
            while i < len(block):
                stmt = block.pop(i)
                if program.keep(stmt, block):
                    block.insert(i, stmt)
                    i += 1
                    continue
                try:
                    fails = still_fails(program)
                except Exception:
                    fails = False
                if fails:
                    changed = True
                else:
                    block.insert(i, stmt)
                    i += 1
    return program
