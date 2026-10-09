#!/usr/bin/env python3
"""Take the device image out of a compiled Vx program's dispatch payload.

`vxc prog.vx --emit-llvm` prints the payload as one escaped string constant.
The payload is the kernel name, NUL-separated `key=value` entries, an empty
entry, then a little-endian 64-bit length followed by that many bytes of
image. This walks it the way `tests/integration_test/device_image_test.rs`
does and writes the section to a file.

Usage:
    vx_spirv_extract.py <ir.ll> <kernel_name> <out.spv>

Prints the payload's text entries and the image's size, so the caller can
check that the payload really says `format=spirv` and see what it loaded.
"""

import re
import struct
import sys


def decode_escaped(text):
    """Turn LLVM's escaped-string bytes back into bytes.

    The literal prints a non-printable byte as `\\XX` (hex) and a backslash
    as `\\\\`; a quote inside is `\\22`. Anything else is itself.
    """
    raw = text.encode("utf-8", "surrogateescape")
    out = bytearray()
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == 0x5C and i + 1 < len(raw):
            if raw[i + 1] == 0x5C:
                out.append(0x5C)
                i += 2
                continue
            # Two hex digits, no fewer: a trailing single digit is not an escape,
            # and the Rust twin in device_image_test.rs reads it the same way.
            digits = raw[i + 1 : i + 3]
            if len(digits) == 2:
                try:
                    out.append(int(digits.decode("ascii"), 16))
                    i += 3
                    continue
                except (ValueError, UnicodeDecodeError):
                    pass
        out.append(c)
        i += 1
    return bytes(out)


def payload_of(ir, kernel):
    """The payload bytes for `kernel`, out of `@<kernel>_str("...")`."""
    marker = '@%s_str("' % kernel
    start = ir.find(marker)
    if start < 0:
        raise SystemExit("no dispatch payload global for %s" % kernel)
    rest = ir[start + len(marker) :]
    end = rest.find('"')
    if end < 0:
        raise SystemExit("the payload global is not terminated")
    return decode_escaped(rest[:end])


def section_of(payload):
    """The payload's section, walked the way the runtime walks it."""
    pos = payload.index(b"\0") + 1  # past the kernel name
    while True:
        length = payload.index(b"\0", pos) - pos
        if length == 0:
            break  # the empty entry: the section starts after it
        pos += length + 1
    start = pos + 1
    if len(payload) < start + 8:
        raise SystemExit("the payload is too short to hold the section's length")
    (length,) = struct.unpack_from("<Q", payload, start)
    if length != len(payload) - start - 8:
        raise SystemExit(
            "the section's length does not account for the rest of the payload"
        )
    return payload[start + 8 :]


def entries_of(payload):
    """The payload's text part as a list of strings."""
    end = payload.index(b"\0")
    pos = end + 1
    out = [payload[:end].decode("utf-8", "replace")]
    while True:
        length = payload.index(b"\0", pos) - pos
        if length == 0:
            return out
        out.append(payload[pos : pos + length].decode("utf-8", "replace"))
        pos += length + 1


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    ir_path, kernel, out_path = argv[1], argv[2], argv[3]
    with open(ir_path, "r") as f:
        ir = f.read()
    payload = payload_of(ir, kernel)
    for entry in entries_of(payload)[1:]:
        print("  entry: %s" % entry)
    section = section_of(payload)
    with open(out_path, "wb") as f:
        f.write(section)
    print("  image: %d bytes -> %s" % (len(section), out_path))
    if section[:4] != bytes((0x03, 0x02, 0x23, 0x07)):
        raise SystemExit("the section does not start with SPIR-V's magic number")


if __name__ == "__main__":
    main(sys.argv)
