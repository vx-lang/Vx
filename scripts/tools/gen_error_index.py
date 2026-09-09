#!/usr/bin/env python3
"""Generate the diagnostic index from the compiler's own diagnostic table.

`src/diagnostic.rs` is the single place every code is declared, each with a doc comment saying
what it means and a section header grouping it. That makes it the only honest source for a
reference: a hand-written index drifts the moment someone adds a code, and 78 of the 113 codes
were undocumented anywhere before this existed.

    python3 scripts/tools/gen_error_index.py            # write the index
    python3 scripts/tools/gen_error_index.py --check    # fail if it is out of date

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import re
import subprocess
import sys
from pathlib import Path

SOURCE = Path("src/diagnostic.rs")
OUTPUT = Path("www/book/src/error-index.md")

# The section headers in diagnostic.rs are terse. These say what the stage actually does, so the
# index reads as a guide rather than as a dump of the enum.
SECTION_BLURB = {
    "Warnings": "Reported without stopping the compile. A warning means the program is accepted "
    "but something in it is probably not what was intended.",
    "Parser Errors": "Raised while turning source text into an AST. The program is not "
    "syntactically valid Vx.",
    "Name Resolution Errors": "Raised when a name cannot be resolved to a declaration, or "
    "resolves to something of the wrong kind.",
    "Type Errors": "Raised by the type checker. Vx performs no implicit numeric conversion, so "
    "many of these are mismatches that a language with coercion would have silently accepted.",
    "Borrow/Ownership Errors": "Raised by the borrow checker and the linear-type rules. These "
    "rule out use-after-move, aliasing violations and lifetimes that outlive what they point at.",
    "Safety Errors": "Raised where an operation needs an `unsafe` context and does not have one.",
    "Topology/Hardware Errors": "Raised by the placement and capacity rules — the checks that "
    "make Vx different from a single-address-space language. A value in the wrong memory space, "
    "a region on the wrong device, a working set that does not fit, or a transfer with no "
    "declared route.",
    "Tensor/Math Errors": "Raised on tensor shapes and numeric operations, including shape "
    "mismatches that are decided at compile time.",
    "Contract/Verification Errors": "Raised when a `requires`, `ensures` or `invariant` clause "
    "cannot be discharged, or when a seam obligation is left unproven.",
}

HEADER = """# Diagnostic index

Every diagnostic the Vx compiler can emit, with the code it reports and what it means.

Codes are grouped by the compilation stage that raises them, and the group is readable off the
number: `E1xxx` is the parser, `E3xxx` the type checker, `E6xxx` the placement and capacity rules,
and so on. A `W` prefix is a warning rather than an error.

> This file is generated from `src/diagnostic.rs` by `scripts/tools/gen_error_index.py`. Edit the
> doc comments on the codes there, not this file.

"""


def parse(source_text):
    """Yield (section, code, description) in declaration order."""
    section = None
    pending = []

    for line in source_text.splitlines():
        stripped = line.strip()

        header = re.match(r"//\s*---\s*(.+?)\s*\((?:[EW]\dxxx)\)\s*---", stripped)
        if header:
            section = header.group(1)
            pending = []
            continue

        doc = re.match(r"///\s?(.*)", stripped)
        if doc:
            pending.append(doc.group(1).strip())
            continue

        code = re.match(r"^([EW][0-9]{4}),$", stripped)
        if code:
            yield section, code.group(1), " ".join(p for p in pending if p).strip()
            pending = []
            continue

        # Anything else (attributes, blank lines, the enum braces) resets the doc buffer only if
        # it is a real statement; blank lines inside a doc block are kept.
        if stripped and not stripped.startswith("#"):
            pending = []


def render(entries):
    out = [HEADER]
    by_section = {}
    for section, code, desc in entries:
        by_section.setdefault(section or "Uncategorised", []).append((code, desc))

    out.append("## Contents\n")
    for section in by_section:
        anchor = section.lower().replace(" ", "-").replace("/", "")
        codes = [c for c, _ in by_section[section]]
        out.append(f"- [{section}](#{anchor}) — `{codes[0]}`–`{codes[-1]}` ({len(codes)} codes)")
    out.append("")

    for section, items in by_section.items():
        out.append(f"## {section}\n")
        blurb = SECTION_BLURB.get(section)
        if blurb:
            out.append(f"{blurb}\n")
        out.append("| Code | Meaning |")
        out.append("| --- | --- |")
        for code, desc in items:
            out.append(f"| `{code}` | {desc or '—'} |")
        out.append("")

    total = sum(len(v) for v in by_section.values())
    out.append("______________________________________________________________________\n")
    out.append(f"{total} diagnostics.\n")
    return "\n".join(out)


def main():
    if not SOURCE.exists():
        print(f"error: {SOURCE} not found; run from the repository root", file=sys.stderr)
        return 1

    entries = list(parse(SOURCE.read_text()))
    if not entries:
        print("error: no diagnostic codes parsed -- has the table's shape changed?", file=sys.stderr)
        return 1

    rendered = render(entries)

    # Matches the shape mdformat produces, so the generated file needs no follow-up pass.

    try:
        rendered = subprocess.run(
            ["mdformat", "-"], input=rendered, capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # mdformat is optional locally; CI has it.

    if "--check" in sys.argv:
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != rendered:
            print(
                f"error: {OUTPUT} is out of date.\n"
                f"  Run: python3 {Path(__file__).as_posix()}",
                file=sys.stderr,
            )
            return 1
        print(f"{OUTPUT} is up to date ({len(entries)} diagnostics)")
        return 0

    OUTPUT.write_text(rendered)
    print(f"wrote {OUTPUT} ({len(entries)} diagnostics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
