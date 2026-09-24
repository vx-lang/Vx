#!/usr/bin/env python3
"""Generate the standard-library reference from the standard library's own sources.

There was no API reference at all: 21 modules and roughly 200 functions, and the only way to find
out what `std::vec` offered was to read it. Generating the reference from the signatures keeps it
complete and stops it drifting, which a hand-written one would do within a release.

The `///` lines above each item are carried through, so the specification of a function lives
next to the function and cannot drift from it. That is Rust's arrangement, and it is why a
comment stripper used to sit at the top of this file: the reference listed signatures and threw
the prose away.

Macro-stamped impls are resolved rather than printed raw. `core::convert` stamps `From` once per
source-and-target pair, and the page used to read ``From<$from> for $to`` -- the macro's own
placeholders, which mean nothing to a reader. The parameters are now shown as `T`, `U`, `V` and
the concrete instantiations are listed beneath.

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

# Each shipped library, in the order a reader should meet them: `core` is the layer
# `std` is written on top of.
LIBRARIES = [("core", Path("stdlib/core")), ("std", Path("stdlib/std"))]
OUTPUT = Path("www/book/src/stdlib-reference.md")

# What each module is for. The sources carry implementation notes rather than a summary, and a
# reference that only lists signatures makes the reader guess which module to open.
MODULE_BLURB = {
    "alloc": "Raw allocation and deallocation.",
    "box": "`Box<T>`, a single-owner heap allocation. Required for recursive types.",
    "clone": "`Clone`, an explicit duplicate of a value.",
    "cmp": "Ordering and equality: `PartialEq`, `Ord`, `PartialOrd` and `Ordering`.",
    "convert": "`From`, the conversions that cannot fail and lose nothing.",
    "default": "`Default`, the value a type starts from.",
    "fs": "Files and directories.",
    "googletest": "Assertions for tests written in Vx.",
    "hash_map": "`HashMap<K, V>`.",
    "hash_set": "`HashSet<T>`.",
    "io": "Standard input, output and error.",
    "iter": "`Range` and the iterator adaptors, `map`, `filter`, `take` and `skip`.",
    "iter::traits": "The `Iterator` trait: one required `next`, and the methods written over it.",
    "libc": "Direct bindings to the C library.",
    "llama": "Helpers used by the Llama 2 example.",
    "marker": "The traits that say something about a type without giving it a method.",
    "mem": "Moving values around without looking at what they are.",
    "mmap": "Memory-mapped files.",
    "net": "TCP and UDP sockets.",
    "num": "The integer and float methods, stamped over every width.",
    "ops": "The callable types a closure literal lowers into.",
    "option": "`Option<T>`, for a value that may be absent.",
    "ptr": "Raw pointers: making one, and reading or writing through it.",
    "rand": "Seeded pseudo-random numbers, one stream per `Rng`.",
    "result": "`Result<T, E>`, for an operation that may fail.",
    "simd": "SIMD vector types and operations.",
    "string": "`String` and text manipulation.",
    "tensor": "Operations on `Tensor`, including shape queries and elementwise maths.",
    "time": "Clocks and durations.",
    "tuple": "The structs tuple syntax stands for, `Tuple2` to `Tuple6`; imported by any module that writes a tuple.",
    "vec": "`Vec<T>`, a growable array.",
}

HEADER = """# Standard library reference

Every public type and function in the shipped library modules, taken from their signatures, with
the documentation each one carries in the source.

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

> This page is generated from `stdlib/core/*.vx` and `stdlib/std/*.vx` by
> `scripts/tools/gen_stdlib_reference.py`. Signatures are exactly what the source declares, and
> the prose under each is its `///` comment. An item with no description has none in the source.

"""

# The letters a macro's parameters are shown as, in the order they are declared. Only the ones
# that reach a signature are ever printed -- `core::num` passes a width and two limits that do
# not -- but every parameter needs a letter so the substitution is total.
PARAM_LETTERS = ["T", "U", "V", "W", "X", "Y", "Z"]


def letters_for(params):
    """A letter per parameter, running past the table if a macro ever takes more."""
    out = []
    for i in range(len(params)):
        out.append(PARAM_LETTERS[i] if i < len(PARAM_LETTERS) else f"P{i}")
    return out


def strip_comments(text):
    """Remove ordinary comments and keep `///` ones.

    The banner, the implementation notes and the `// RUN:` lines all go; the doc comments are
    what this file exists to carry through.

    The lookbehind is what makes that work. Without it the pattern matches the *second and
    third* slashes of `///`, strips from there, and leaves a bare `/` where the description
    was -- so every doc comment survived the scan and arrived empty.
    """
    return re.sub(r"(?<!/)//(?!/).*", "", text)


def split_externs(text):
    """Return (vx_source, extern_source) with ordinary comments removed.

    The two are listed separately rather than merged, because an `extern` block means different
    things in different modules. In `vec` it is the Rust core backing the collection -- plumbing a
    caller never names. In `io` and `libc` the extern block *is* the module: there is nothing else
    in the file. Dropping externs outright lost six of the twenty-one modules entirely, including
    `std::io`.
    """
    text = strip_comments(text)

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


def doc_above(text, pos):
    """The `///` block immediately above the item starting at `pos`, as one string.

    Blank lines between the comment and the item are allowed, so that a doc comment separated
    from its function by the formatter still belongs to it. Anything else in between ends the
    block, which is what stops a module header being read as the first function's description.
    """
    lines = text[:pos].split("\n")
    # `lines[-1]` is the partial line the item starts on.
    i = len(lines) - 2
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    out = []
    while i >= 0 and lines[i].strip().startswith("///"):
        out.append(lines[i].strip()[3:].strip())
        i -= 1
    if not out:
        return None
    out.reverse()
    # A doc comment's blank line is written as a bare `///`, which strips to "".
    text = "\n".join(out).strip()
    return text or None


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


def balanced_block(text, open_pos):
    """The text between the brace at `open_pos` and its match, and the index just past it."""
    depth, j = 0, open_pos
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_pos + 1 : j], j + 1
        j += 1
    return text[open_pos + 1 :], len(text)


def take_macros(text):
    """Pull `macro_rules` definitions and their invocations out of a module.

    Returns (text_without_them, [(params, body, [arg_tuples])]). The definition and the
    invocations are removed from the text so the ordinary parse below does not see a body full
    of `$t` and report it as a function on a type called `$t`.
    """
    macros = []
    for m in list(re.finditer(r"\bmacro_rules\s+([A-Za-z_]\w*)\s*\{", text)):
        name = m.group(1)
        whole, _ = balanced_block(text, m.end() - 1)
        rule = re.search(r"\(([^)]*)\)\s*=>\s*\{", whole)
        if not rule:
            continue
        params = [p.strip() for p in re.findall(r"(\$\w+)\s*:", rule.group(1))]
        body, _ = balanced_block(whole, rule.end() - 1)
        args = []
        for inv in re.finditer(rf"\b{name}!\(([^)]*)\)", text):
            args.append([a.strip() for a in inv.group(1).split(",")])
        if args:
            macros.append((params, body, args))
        # Remove the definition and every invocation from the text the ordinary parse sees.
        text = text.replace(text[m.start() : m.end() - 1 + len(whole) + 1], "")
        text = re.sub(rf"\b{name}!\([^)]*\)\s*;", "", text)
    return text, macros


def parse_impl_groups(text):
    """Group each function with the impl block it sits in, and carry its doc comment.

    A free function has an empty owner. Impl headers may wrap across lines after formatting.
    """
    # The impl's whole extent, not just where it starts. Keyed on the opening position alone,
    # a free function written *after* an impl block was filed as one of its methods --
    # `core::convert`'s `identity` was listed under `From<T> for Option<T>`.
    impls = []
    for m in re.finditer(r"\bimpl\b\s*(?:<[^>]*>)?\s*([^{]+?)\{", text):
        brace = text.index("{", m.start())
        _, end = balanced_block(text, brace)
        impls.append((m.start(), end, normalise(m.group(1))))

    # A trait's own methods are requirements and defaults, not free functions, and were
    # labelled as free functions because only impl blocks had an owner.
    for m in re.finditer(r"\btrait\s+([A-Za-z_]\w*)\s*(<[^{]*?>)?\s*\{", text):
        brace = text.index("{", m.start())
        _, end = balanced_block(text, brace)
        impls.append((m.start(), end, normalise(f"trait {m.group(1)}{m.group(2) or ''}")))

    out = []
    for pos, sig in signatures(text):
        owner = ""
        for start, end, name in impls:
            if start < pos < end:
                owner = name
                break
        out.append((owner, sig, doc_above(text, pos)))
    return out


def parse_module(path):
    raw, extern_text = split_externs(path.read_text())
    body, macros = take_macros(raw)

    types = []
    for m in re.finditer(
        r"\b(struct|enum|trait)\s+([A-Za-z_]\w*)\s*(<[^{]*?>)?\s*\{", body
    ):
        types.append(
            (normalise(f"{m.group(1)} {m.group(2)}{m.group(3) or ''}"), doc_above(body, m.start()))
        )

    funcs = parse_impl_groups(body)

    # A stamped impl is rendered once, with its parameters lettered and the concrete
    # instantiations listed, rather than once per invocation.
    stamped = []
    for params, mbody, args in macros:
        subst = dict(zip(params, letters_for(params)))
        shown = mbody
        for p, letter in subst.items():
            shown = re.sub(re.escape(p) + r"\b", letter, shown)
        groups = parse_impl_groups(shown)
        if not groups:
            continue
        # Only the parameters that survive into a signature are worth listing: `core::num`
        # passes a width and a limit that never appear in one.
        used = [i for i, p in enumerate(params) if any(subst[p] in s for _, s, _ in groups)]
        if not used:
            used = [0]
        tuples = []
        for a in args:
            picked = [a[i] for i in used if i < len(a)]
            if picked:
                tuples.append(" → ".join(picked) if len(picked) > 1 else picked[0])
        stamped.append(([subst[params[i]] for i in used], groups, tuples))

    externs = [(sig, doc_above(extern_text, pos)) for pos, sig in signatures(extern_text)]
    return types, funcs, stamped, externs


def render_items(out, items):
    """One bullet per item: signature, then its description if it has one."""
    for sig, doc in items:
        if doc:
            summary, _, rest = doc.partition("\n")
            out.append(f"- `{sig}`<br>")
            out.append(f"  {summary}")
            for line in rest.split("\n"):
                if line.strip():
                    out.append(f"  {line.strip()}")
        else:
            out.append(f"- `{sig}`")
    out.append("")


def render(modules):
    out = [HEADER, "## Contents\n"]
    for lib, name, *_ in modules:
        anchor = lib + name.replace("::", "")
        out.append(f"- [`{lib}::{name}`](#{anchor}) — {MODULE_BLURB.get(name, '')}")
    out.append("")

    total_fns = 0
    for lib, name, types, funcs, stamped, externs in modules:
        out.append(f"## `{lib}::{name}`\n")
        blurb = MODULE_BLURB.get(name)
        if blurb:
            out.append(f"{blurb}\n")

        if types:
            out.append("**Types**\n")
            render_items(out, types)

        by_owner = {}
        for owner, sig, doc in funcs:
            by_owner.setdefault(owner, []).append((sig, doc))

        for owner, items in by_owner.items():
            total_fns += len(items)
            out.append(f"**{'Functions' if not owner else f'`{owner}` methods'}**\n")
            render_items(out, items)

        for letters, groups, tuples in stamped:
            by_owner = {}
            for owner, sig, doc in groups:
                by_owner.setdefault(owner, []).append((sig, doc))
            for owner, items in by_owner.items():
                total_fns += len(items) * max(len(tuples), 1)
                label = f"`{owner}` methods" if owner else "Functions"
                out.append(f"**{label}**, stamped for {len(tuples)} instantiations\n")
                render_items(out, items)
                joined = ", ".join(f"`{t}`" for t in tuples)
                pair = "".join(letters) if len(letters) == 1 else f"({', '.join(letters)})"
                out.append(f"{pair} = {joined}\n")

        if externs:
            total_fns += len(externs)
            label = (
                "**Functions** *(bound directly to C)*"
                if not funcs and not stamped
                else "**C bindings** *(the native functions this module is built on)*"
            )
            out.append(f"{label}\n")
            render_items(out, externs)

    out.append("______________________________________________________________________\n")
    out.append(f"{total_fns} functions across {len(modules)} modules.\n")
    return "\n".join(out)


def main():
    modules = []
    for lib, directory in LIBRARIES:
        if not directory.is_dir():
            print(
                f"error: {directory} not found; run from the repository root",
                file=sys.stderr,
            )
            return 1
        # Recursive, so a nested module such as `core::iter::traits` is listed rather than
        # silently left out; its name is its path under the library.
        for path in sorted(directory.rglob("*.vx")):
            name = "::".join(path.relative_to(directory).with_suffix("").parts)
            types, funcs, stamped, externs = parse_module(path)
            if types or funcs or stamped or externs:
                modules.append((lib, name, types, funcs, stamped, externs))

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
