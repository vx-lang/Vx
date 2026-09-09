#!/usr/bin/env python3
"""Generate the standard-library reference from the standard library's own sources.

There was no API reference at all: 21 modules and roughly 200 functions, and the only way to find
out what `std::vec` offered was to read it. Generating the reference from the signatures keeps it
complete and stops it drifting, which a hand-written one would do within a release.

    python3 scripts/tools/gen_stdlib_reference.py            # write the reference
    python3 scripts/tools/gen_stdlib_reference.py --check    # fail if it is out of date

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import re
import subprocess
import sys
from pathlib import Path

STD_DIR = Path("stdlib/std")
OUTPUT = Path("www/book/src/stdlib-reference.md")

# What each module is for. The sources carry implementation notes rather than a summary, and a
# reference that only lists signatures makes the reader guess which module to open.
MODULE_BLURB = {
    "alloc": "Raw allocation and deallocation.",
    "box": "`Box<T>`, a single-owner heap allocation. Required for recursive types.",
    "closure": "The closure types the compiler lowers `|x| ...` into.",
    "fs": "Files and directories.",
    "googletest": "Assertions for tests written in Vx.",
    "hash_map": "`HashMap<K, V>`.",
    "hash_set": "`HashSet<T>`.",
    "io": "Standard input, output and error.",
    "iter": "The `Iterator` trait and its adaptors, which `for` loops and `.map` build on.",
    "libc": "Direct bindings to the C library.",
    "llama": "Helpers used by the Llama 2 example.",
    "math": "Mathematical functions and constants.",
    "mmap": "Memory-mapped files.",
    "net": "TCP and UDP sockets.",
    "option": "`Option<T>`, for a value that may be absent.",
    "result": "`Result<T, E>`, for an operation that may fail.",
    "simd": "SIMD vector types and operations.",
    "string": "`String` and text manipulation.",
    "tensor": "Operations on `Tensor`, including shape queries and elementwise maths.",
    "time": "Clocks and durations.",
    "vec": "`Vec<T>`, a growable array.",
}

# These blocks list signatures, not programs: they have no bodies and cannot compile. The
# documentation-example checker is told so explicitly rather than left to guess.
SIGNATURE_SKIP = "<!-- vx-doctest: skip -- signature listing, not a program -->\n"

HEADER = """# Standard library reference

Every public type and function in the 21 `std` modules, taken from their signatures.

Import a module with its path, then use the names it declares:

```rust
import std::vec;

fn main() -> i32 {
    let mut v = Vec<i32>::new();
    v.push(10);
    v.push(32);
    return v.get(0) + v.get(1);
}
```

The toolchain also ships a `graph` library outside `std`, imported as `graph::traversal` and
friends.

> This page is generated from `stdlib/std/*.vx` by `scripts/tools/gen_stdlib_reference.py`.
> Signatures are exactly what the source declares.

"""


def split_externs(text):
    """Return (vx_source, extern_source) with the licence banner and comments removed.

    The two are listed separately rather than merged, because an `extern` block means different
    things in different modules. In `vec` it is the Rust core backing the collection — plumbing a
    caller never names. In `io` and `libc` the extern block *is* the module: there is nothing else
    in the file. Dropping externs outright lost six of the twenty-one modules entirely, including
    `std::io`.
    """
    text = re.sub(r"//.*", "", text)

    kept, externs, i = [], [], 0
    while True:
        m = re.search(r'\bextern\b\s*(?:"[^"]*")?\s*\{', text[i:])
        if not m:
            kept.append(text[i:])
            break
        start = i + m.start()
        kept.append(text[i:start])
        depth, j = 0, i + m.end() - 1
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        externs.append(text[start : j + 1])
        i = j + 1
    return "".join(kept), "\n".join(externs)


def normalise(sig):
    """Collapse the formatter's line breaks so a signature reads as one line.

    vx-format wraps declarations at column width and does not always leave a space around `->` or
    inside the parameter list, so the raw text is joined and then re-spaced rather than printed as
    found.
    """
    sig = re.sub(r"\s+", " ", sig)
    sig = sig.replace("( ", "(").replace(" )", ")")
    sig = re.sub(r"\s*->\s*", " -> ", sig)
    sig = re.sub(r"\s*,\s*", ", ", sig)
    sig = re.sub(r"\s*:\s*", " : ", sig)
    return sig.strip()


FN_RE = (
    r"\b(unsafe\s+)?fn\s+([A-Za-z_]\w*)\s*(<[^(]*?>)?\s*\(([^)]*)\)\s*(->\s*[^{;]+)?"
)


def signatures(text):
    for m in re.finditer(FN_RE, text):
        yield m.start(), normalise(
            f"{'unsafe ' if m.group(1) else ''}fn {m.group(2)}{m.group(3) or ''}"
            f"({m.group(4)}) {m.group(5) or '-> void'}"
        )


def parse_module(path):
    text, extern_text = split_externs(path.read_text())

    types = []
    for m in re.finditer(r"\b(struct|enum|trait)\s+([A-Za-z_]\w*)\s*(<[^{]*?>)?\s*\{", text):
        types.append(normalise(f"{m.group(1)} {m.group(2)}{m.group(3) or ''}"))

    # Associate each function with the impl block it sits in, so `Vec<T>::push` is not listed as a
    # free function. Impl headers may wrap across lines after formatting.
    impls = [
        (m.start(), normalise(m.group(1)))
        for m in re.finditer(r"\bimpl\b\s*(?:<[^>]*>)?\s*([^{]+?)\{", text)
    ]

    funcs = []
    for pos, sig in signatures(text):
        owner = ""
        for impl_pos, name in impls:
            if impl_pos < pos:
                owner = name
            else:
                break
        funcs.append((owner, sig))

    externs = [sig for _, sig in signatures(extern_text)]

    return types, funcs, externs


def render(modules):
    out = [HEADER, "## Contents\n"]
    for name, _types, _funcs, _externs in modules:
        out.append(f"- [`std::{name}`](#std{name}) — {MODULE_BLURB.get(name, '')}")
    out.append("")

    total_fns = 0
    for name, types, funcs, externs in modules:
        out.append(f"## `std::{name}`\n")
        blurb = MODULE_BLURB.get(name)
        if blurb:
            out.append(f"{blurb}\n")

        if types:
            out.append("**Types**\n")
            for t in types:
                out.append(f"- `{t}`")
            out.append("")

        by_owner = {}
        for owner, sig in funcs:
            by_owner.setdefault(owner, []).append(sig)

        for owner, sigs in by_owner.items():
            total_fns += len(sigs)
            out.append(f"**{'Functions' if not owner else f'`{owner}` methods'}**\n")
            out.append(SIGNATURE_SKIP)
            out.append("```rust")
            out.extend(sigs)
            out.append("```")
            out.append("")

        if externs:
            total_fns += len(externs)
            label = (
                "**Functions** *(bound directly to C)*"
                if not funcs
                else "**C bindings** *(the native functions this module is built on)*"
            )
            out.append(f"{label}\n")
            out.append(SIGNATURE_SKIP)
            out.append("```rust")
            out.extend(externs)
            out.append("```")
            out.append("")

    out.append("______________________________________________________________________\n")
    out.append(f"{total_fns} functions across {len(modules)} modules.\n")
    return "\n".join(out)


def main():
    if not STD_DIR.is_dir():
        print(f"error: {STD_DIR} not found; run from the repository root", file=sys.stderr)
        return 1

    modules = []
    for path in sorted(STD_DIR.glob("*.vx")):
        types, funcs, externs = parse_module(path)
        if types or funcs or externs:
            modules.append((path.stem, types, funcs, externs))

    if not modules:
        print("error: no modules parsed", file=sys.stderr)
        return 1

    rendered = render(modules)
    try:
        rendered = subprocess.run(
            ["mdformat", "-"], input=rendered, capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if "--check" in sys.argv:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != rendered:
            print(
                f"error: {OUTPUT} is out of date.\n"
                f"  Run: python3 {Path(__file__).as_posix()}",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT} is up to date ({len(modules)} modules)")
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered)
    print(f"wrote {OUTPUT} ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
