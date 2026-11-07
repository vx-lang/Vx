#!/usr/bin/env python3
"""Write a compiled Vx program's per-kernel dispatch payloads, in dispatch order.

`vxc prog.vx --emit-llvm` prints each placed region's payload as one escaped
string global. `vx_spirv_extract.py` decodes one; this decodes all of them and
prints the order the program dispatches them in, which is the order the regions
appear in the source. The conformance runner pairs that order with its case
list, so a kernel is never matched to the wrong case by guessing its number.

Usage:
    extract_payloads.py <ir.ll> <out-dir>

Writes `<out-dir>/<kernel>.payload` for each kernel and prints one line per
kernel: its dispatch position and its name.
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vx_spirv_extract as extract  # noqa: E402


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    with open(argv[1], "r") as f:
        ir = f.read()
    out = argv[2]
    os.makedirs(out, exist_ok=True)

    order = []
    for name in re.findall(r"addressof @([A-Za-z0-9_]+)_str", ir):
        if name not in order:
            order.append(name)
    if not order:
        raise SystemExit("no dispatch payload in %s" % argv[1])

    for name in order:
        payload = extract.payload_of(ir, name)
        with open(os.path.join(out, name + ".payload"), "wb") as f:
            f.write(payload)
    for i, name in enumerate(order):
        print("%d %s" % (i, name))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
