#!/usr/bin/env python3
"""Fail if any blog post h2/h3 is missing a proper id and permalink.

Every section heading in www/blog carries a stable id and an <a href="#id">
wrapper so deep links and in-page permalinks work. New posts forget that
unless something checks. Run from the repo root.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(
    r"<h([23])(\s[^>]*)?>(.*?)</h\1>",
    re.DOTALL,
)
ID_ATTR = re.compile(r'\sid="([^"]+)"')
LINK = re.compile(r'^<a\s+href="#([^"]+)">(.*)</a>$', re.DOTALL)


def main() -> int:
    blog = Path("www/blog")
    if not blog.is_dir():
        print("www/blog not found; run from the repository root", file=sys.stderr)
        return 2

    errors: list[str] = []
    count = 0
    for path in sorted(blog.glob("*.html")):
        if path.name == "index.html":
            continue
        text = path.read_text(encoding="utf-8")
        seen: dict[str, int] = {}
        for m in HEADING.finditer(text):
            attrs = m.group(2) or ""
            content = m.group(3)
            line = text.count("\n", 0, m.start()) + 1
            id_m = ID_ATTR.search(attrs)
            if not id_m:
                errors.append(f"{path}:{line}: missing id= attribute: <h{m.group(1)}{attrs}>")
                continue
            hid = id_m.group(1)
            if hid in seen:
                errors.append(
                    f"{path}:{line}: duplicate id={hid!r} (also line {seen[hid]})"
                )
            else:
                seen[hid] = line
            link_m = LINK.match(content.strip())
            if not link_m:
                errors.append(
                    f"{path}:{line}: heading content must be "
                    f'<a href="#{hid}">…</a>, got: {content.strip()[:80]!r}'
                )
                continue
            if link_m.group(1) != hid:
                errors.append(
                    f"{path}:{line}: href=#{link_m.group(1)!r} does not match id={hid!r}"
                )
                continue
            count += 1

    if errors:
        print("Blog heading check failed:")
        print("\n".join(errors))
        return 1

    print(f"All {count} blog h2/h3 headings have an id and matching permalink.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
