#!/usr/bin/env python3
# ===- unicode_tables.py - Vx Compiler ---------------------------*- Python -*-===//
#
# Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===//
# Writes the tables of stdlib/std/unicode.vx from Python's `unicodedata`: the runs of code points
# in general category L (letters) and in general category N (numbers). It replaces the text
# between the two marker lines of the file. The Unicode version is the one of the Python that
# runs it.
#
#   python3 scripts/tools/unicode_tables.py stdlib/std/unicode.vx
# ===----------------------------------------------------------------------===//
import sys
import unicodedata

BEGIN = "// BEGIN TABLES: written by scripts/tools/unicode_tables.py"
END = "// END TABLES"


def runs(category_letter):
    """The first and last code point of each run of code points whose general category
    starts with `category_letter`."""
    found = []
    start = None
    for code in range(0x110000):
        inside = unicodedata.category(chr(code)).startswith(category_letter)
        if inside and start is None:
            start = code
        if not inside and start is not None:
            found.append((start, code - 1))
            start = None
    if start is not None:
        found.append((start, 0x10FFFF))
    return found


def table(name, numbers):
    lines = [f"const {name} : Tensor<u32, [{len(numbers)}]> = ["]
    for i in range(0, len(numbers), 8):
        row = ", ".join(f"{n}u32" for n in numbers[i : i + 8])
        lines.append(f"  {row},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append("];")
    return "\n".join(lines)


def tables():
    out = [BEGIN, f"// Unicode {unicodedata.unidata_version}."]
    for name, letter, what in (("LETTER", "L", "letters"), ("NUMBER", "N", "numbers")):
        found = runs(letter)
        out.append("")
        out.append(f"/// The number of runs of {what}.")
        out.append(f"const UNICODE_{name}_RUNS : i32 = {len(found)};")
        out.append("")
        out.append(f"/// The first code point of each run of {what} (general category {letter}).")
        out.append(table(f"UNICODE_{name}_FIRST", [a for a, _ in found]))
        out.append("")
        out.append(f"/// The last code point of each run of {what}.")
        out.append(table(f"UNICODE_{name}_LAST", [b for _, b in found]))
    out.append(END)
    return "\n".join(out)


def main():
    path = sys.argv[1]
    with open(path) as f:
        text = f.read()
    begin = text.index(BEGIN)
    end = text.index(END) + len(END)
    with open(path, "w") as f:
        f.write(text[:begin] + tables() + text[end:])


if __name__ == "__main__":
    main()
