#!/usr/bin/env python3
"""Compile every Vx example in the documentation.

Documentation drifts silently because nothing executes it. An audit of this repository found the
README's own headline example did not parse, a tutorial teaching a topology syntax that was never
implemented, `grad` documented with the wrong arity, and a grammar production admitting two types
the parser rejects. Every one of those had been correct when written.

This makes that class of error a build failure.

    python3 scripts/tools/check_doc_examples.py                 # check everything
    python3 scripts/tools/check_doc_examples.py --list          # show what would be checked
    python3 scripts/tools/check_doc_examples.py README.md       # check specific files

A block is checked when it is fenced ```vx or ```rust, contains a top-level declaration, and is
not skipped by one of the rules below. To exempt a block deliberately, put

    <!-- vx-doctest: skip -->

on the line before its fence, ideally with a reason.

Most blocks are only parsed. A block marked

    <!-- vx-doctest: run -->

is compiled and run instead, and what it prints must equal the ```text block that follows it
(one short line, such as "It prints:", may come between them).

The hand-written pages under `www/` are checked as well, through their `<pre><code>` blocks. The
landing page carries the most-read Vx code on the site and was the only code on it that nothing
compiled -- everything in the book is covered because the book is markdown. The same skip comment
works there, on the line before the `<pre>`.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = re.compile(
    r"(?:<!--\s*vx-doctest:\s*(skip|run).*?-->\s*\n)?```(vx|rust)\n(.*?)```", re.S
)
# The expected output, after at most one short line such as "It prints:".
OUTPUT = re.compile(r"\s*(?:[^`\n]+\n\s*)?```text\n(.*?)```", re.S)

# The same idea for the hand-written pages: a <pre><code> block, optionally preceded by the skip
# comment. Their code carries syntax highlighting as <span> tags, which are stripped before the
# body is compiled.
HTML_BLOCK = re.compile(
    r"(?:(<!--\s*vx-doctest:\s*skip.*?-->)\s*\n?\s*)?<pre><code>(.*?)</code></pre>", re.S
)
TAG = re.compile(r"<[^>]+>")

# A block has to declare something to be a compilable unit. Bare statements are illustrative
# fragments and are left alone.
DECLARES = re.compile(r"^\s*(fn |unsafe fn |safe fn |struct |enum |trait |impl |extern |macro_rules!|Memory |Topology )", re.M)

# Constructs that exist in Rust and not in Vx. Architecture documents embed compiler internals in
# ```rust fences, and those are Rust, not examples of the language.
RUST_ONLY = re.compile(
    r"^\s*use\s+\w|#\[|#!\[|\bpub\s+(fn|struct|enum|mod|const|trait)\b|\bcrate::|"
    r"&self\b|&mut self\b|\bimpl\s+\w+\s+for\s+\w+\s*\{[^}]*fn\s+\w+\s*\(\s*&|"
    # Types and idioms from the standard Rust library that Vx has no equivalent of. The
    # architecture documents embed real compiler internals, and these are what give them away.
    r"\bArc<|\bRc<|\bWeak<|\bRefCell<|\bMutex<|\bRwLock<|\bBox<dyn\b|\bOption<&|"
    r"&\[|\.par_iter\(|\.iter\(\)\.|\bSelf::|\bmatches!\(|\bVec::<|::<",
    re.M,
)

# An elision is not a program.
ELIDED = re.compile(r"(^|\s)\.\.\.(\s|$)|/\*\s*\.\.\.\s*\*/|\$\{|<<")


def blocks(path):
    text = path.read_text()

    if path.suffix == ".html":
        for m in HTML_BLOCK.finditer(text):
            skip_marker, body = m.group(1), m.group(2)
            # Highlighting spans first, then entities -- in that order, or an escaped &lt;span&gt;
            # shown as literal text in a page about markup would be unescaped into a real tag and
            # then stripped.
            body = html.unescape(TAG.sub("", body))
            line = text[: m.start()].count("\n") + 1
            yield line, "vx", body, bool(skip_marker), None
        return

    for m in FENCE.finditer(text):
        marker, lang, body = m.group(1), m.group(2), m.group(3)
        line = text[: m.start()].count("\n") + 1
        expected = None
        if marker == "run":
            out = OUTPUT.match(text, m.end())
            if not out:
                raise SystemExit(f"{path}:{line}: a `run` block needs a ```text block after it")
            expected = out.group(1)
        yield line, lang, body, marker == "skip", expected


def classify(body, lang, skipped):
    if skipped:
        return "explicitly skipped"
    if not DECLARES.search(body):
        return "fragment (no declaration)"
    if ELIDED.search(body):
        return "contains an elision"
    if RUST_ONLY.search(body):
        return "Rust, not Vx"
    return None


def run_and_compare(vxc, source, expected):
    """Compile and run one program; an error message, or None when it printed `expected`."""
    ran = subprocess.run([vxc, source], capture_output=True, text=True)
    # The compiler's own progress lines are printed beside the program's output.
    printed = "".join(
        l for l in ran.stdout.splitlines(keepends=True) if not l.startswith(("[JIT]", "[flat-codegen]"))
    )
    if ran.returncode != 0:
        lines = (ran.stdout + ran.stderr).strip().splitlines()
        return next((l for l in lines if re.search(r"\berror\b", l, re.I)), lines[-1] if lines else "failed")
    # The runner ends with an empty line of its own.
    if printed.rstrip("\n") != expected.rstrip("\n"):
        return f"printed {printed!r}, the text block after it says {expected!r}"
    return None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    list_only = "--list" in sys.argv

    vxc = os.environ.get("VXC", "target/release/vxc")
    if not list_only and not Path(vxc).exists():
        print(f"error: {vxc} not found. Build it first, or set VXC.", file=sys.stderr)
        return 1

    if args:
        files = [Path(a) for a in args]
    else:
        out = subprocess.run(
            ["git", "ls-files", "*.md", "www/*.html", "www/blog/*.html"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        files = [Path(p) for p in out.split() if p]

    checked = skipped = 0
    failures = []

    for path in sorted(files):
        if not path.exists():
            continue
        for line, lang, body, marked, expected in blocks(path):
            reason = classify(body, lang, marked)
            if reason:
                skipped += 1
                if list_only:
                    print(f"  skip  {path}:{line}  ({reason})")
                continue

            checked += 1
            if list_only:
                print(f"  CHECK {path}:{line}")
                continue

            with tempfile.NamedTemporaryFile("w", suffix=".vx", delete=False) as fh:
                fh.write(body)
                tmp = fh.name
            try:
                if expected is not None:
                    err = run_and_compare(vxc, tmp, expected)
                    if err:
                        failures.append((path, line, err.replace(tmp, f"{path}:{line}")))
                    continue
                proc = subprocess.run(
                    [vxc, "--parse-only", tmp], capture_output=True, text=True
                )
                combined = proc.stdout + proc.stderr
                if proc.returncode != 0 or re.search(r"\berror\b", combined, re.I):
                    first = next(
                        (
                            l.strip()
                            for l in combined.splitlines()
                            if re.search(r"\berror\b", l, re.I)
                        ),
                        combined.strip().splitlines()[0] if combined.strip() else "unknown",
                    )
                    first = first.replace(tmp, f"{path}:{line}")
                    failures.append((path, line, first))
            finally:
                os.unlink(tmp)

    if list_only:
        print(f"\n{checked} blocks would be checked, {skipped} skipped")
        return 0

    for path, line, err in failures:
        print(f"{path}:{line}: {err}")

    print(f"\n{checked} documentation examples compiled, {skipped} skipped, {len(failures)} failed")

    if failures:
        print(
            "\nAn example in the documentation no longer compiles. Either fix the example, or if\n"
            "the block is deliberately not a complete program, mark it:\n"
            "    <!-- vx-doctest: skip -->  (with a reason)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
