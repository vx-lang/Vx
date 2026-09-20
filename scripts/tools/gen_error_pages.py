#!/usr/bin/env python3
"""Generate one page per diagnostic code for vxlang.org/errors/.

Pasting an error code into a search box is how a systems programmer finds a language, so
every code the compiler can emit gets its own URL: `vxlang.org/errors/E6003`.

Nothing here is written by hand. The code and its description come from `src/diagnostic.rs`,
the same table `gen_error_index.py` reads. The example program and the message come from a
real fixture under `tests/`, chosen because it asserts that code -- so a page cannot claim a
diagnostic the test suite does not already hold the compiler to.

Codes with no such fixture get a page with the description alone, saying so, rather than an
invented example. `--check` prints which codes those are, which doubles as a list of the
diagnostics no test currently asserts.

    python3 scripts/tools/gen_error_pages.py              # write www/errors/
    python3 scripts/tools/gen_error_pages.py --check      # report coverage, write nothing

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import html
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_error_index import SOURCE, parse  # noqa: E402

TESTS = Path("tests")
OUTPUT = Path("www/errors")
REPO = "https://github.com/vx-lang/Vx"

# Where a reader should go next, per diagnostic group. The book chapter that explains the rule
# the code enforces, not merely one that mentions it.
SECTION_CHAPTER = {
    "Warnings": ("A tour of Vx", "/docs/tour.html"),
    "Parser Errors": ("A tour of Vx", "/docs/tour.html"),
    "Name Resolution Errors": ("A tour of Vx", "/docs/tour.html"),
    "Type Errors": ("A tour of Vx", "/docs/tour.html"),
    "Borrow/Ownership Errors": ("Ownership and borrowing", "/docs/ownership.html"),
    "Safety Errors": ("Unsafe and FFI", "/docs/unsafe-and-ffi.html"),
    "Topology/Hardware Errors": ("Topologies and memory", "/docs/heterogeneous.html"),
    "Tensor/Math Errors": ("Standard library", "/docs/stdlib-reference.html"),
    "Contract/Verification Errors": ("Contracts and verification", "/docs/contracts.html"),
}

KEYWORDS = {
    "fn", "let", "mut", "return", "if", "else", "for", "in", "while", "loop", "break",
    "continue", "struct", "enum", "union", "impl", "trait", "type", "const", "static",
    "unsafe", "extern", "comptime", "spawn", "on", "as", "match", "use", "mod", "pub",
    "requires", "ensures", "invariant", "assert", "true", "false",
    "self", "where", "dyn", "move", "ref",
}

PRIMITIVES = {
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f16", "f32", "f64", "bool",
    "char", "str", "usize", "isize", "void",
}

# One pass over a line, so a match inside a comment or a string cannot be re-highlighted.
TOKEN = re.compile(
    r"""(?P<comment>//[^\n]*)
      | (?P<string>"(?:[^"\\]|\\.)*")
      | (?P<number>\b\d+(?:\.\d+)?\b)
      | (?P<call>\b[a-z_][A-Za-z0-9_]*(?=\())
      | (?P<word>\b[A-Za-z_][A-Za-z0-9_]*\b)
    """,
    re.VERBOSE,
)


def prose(text):
    """A doc comment as HTML. They are written in markdown-ish Rust style: `x` and ` -- `."""
    out = html.escape(text or "")
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out.replace(" -- ", " — ")


def plain(text):
    """The same, for an attribute value, where no tag may appear."""
    return html.escape((text or "").replace("`", "").replace(" -- ", " — "), quote=True)


def highlight(source):
    """Wrap Vx source in the span classes the site stylesheet already defines."""
    out = []
    for line in source.split("\n"):
        pos = 0
        buf = []
        for m in TOKEN.finditer(line):
            buf.append(html.escape(line[pos : m.start()]))
            text = html.escape(m.group(0))
            kind = m.lastgroup
            if kind == "comment":
                buf.append(f'<span class="c">{text}</span>')
            elif kind == "string":
                buf.append(f'<span class="n">{text}</span>')
            elif kind == "number":
                buf.append(f'<span class="n">{text}</span>')
            elif m.group(0) in KEYWORDS:
                # Checked before `call`, so `spawn on(...)` keeps `on` a keyword.
                buf.append(f'<span class="k">{text}</span>')
            elif kind == "call":
                buf.append(f'<span class="f">{text}</span>')
            elif m.group(0) in PRIMITIVES or m.group(0)[:1].isupper():
                buf.append(f'<span class="t">{text}</span>')
            else:
                buf.append(text)
            pos = m.end()
        buf.append(html.escape(line[pos:]))
        out.append("".join(buf))
    return "\n".join(out)


def scaffolding(line):
    """True for a line that belongs to the test harness rather than to the program."""
    s = line.strip()
    return (
        s.startswith("//===")
        or s.startswith("//-*-")
        or re.match(r"^//\s*(RUN|REQUIRES|UNSUPPORTED|XFAIL|DEFINE|REDEFINE):", s)
        or re.match(r"^//\s*CHECK(-[A-Z]+)?:", s)
    )


def program_of(path):
    """The fixture with its test harness removed, so a reader sees only the program."""
    kept = [ln for ln in path.read_text(errors="ignore").split("\n") if not scaffolding(ln)]
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    # Collapse the runs of blank lines that removing the CHECK lines leaves behind.
    out = []
    for ln in kept:
        if not ln.strip() and out and not out[-1].strip():
            continue
        out.append(ln)
    return "\n".join(out)


def assertions_of(path, code):
    """The message fragments the fixture holds the compiler to, in order.

    Returns a list of lines, where a line is the CHECK that started it joined with any
    CHECK-SAME that FileCheck requires to land on that same line.
    """
    lines = []
    for raw in path.read_text(errors="ignore").split("\n"):
        m = re.match(r"^\s*//\s*CHECK(-(?P<kind>[A-Z]+))?:\s*(?P<text>.*?)\s*$", raw)
        if not m:
            continue
        kind, text = m.group("kind"), m.group("text")
        if kind == "NOT" or not text:
            continue
        if kind == "SAME" and lines:
            lines[-1] += f" … {text}"
        else:
            lines.append(text)
    # Only the run of assertions that concerns this code. A fixture may assert several.
    start = next((i for i, ln in enumerate(lines) if code in ln), None)
    if start is None:
        return []
    end = start + 1
    while end < len(lines) and not re.match(r"^[EW][0-9]{4}\b", lines[end]):
        end += 1
    return lines[start:end]


def fixture_for(code, index):
    """The clearest fixture asserting `code`: the shortest one that names it in a CHECK."""
    candidates = [p for p in index.get(code, []) if assertions_of(p, code)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (len(p.read_text(errors="ignore")), str(p)))


def build_index():
    """Map each code to every fixture that mentions it."""
    index = {}
    for path in sorted(TESTS.rglob("*.vx")):
        text = path.read_text(errors="ignore")
        for code in set(re.findall(r"\b[EW][0-9]{4}\b", text)):
            index.setdefault(code, []).append(path)
    return index


SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">

<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="article">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{image}">
<meta property="og:image:alt" content="{image_alt}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="{image}">

<link rel="stylesheet" href="/style.css">
</head>
<body>

<header class="nav">
  <a class="brand" href="/">
    <span class="mark">Vx</span>
  </a>
  <nav>
    <a href="/docs/getting-started.html">Install</a>
    <a href="/docs/tour.html">Language tour</a>
    <a href="/docs/">Docs</a>
    <a href="/blog/">Blog</a>
    <a href="https://github.com/vx-lang/Vx">GitHub</a>
  </nav>
</header>

<main>

{body}
</main>

<footer>
  <p>Vx is Apache 2.0 with the LLVM exception. <a href="https://github.com/vx-lang/Vx">Source</a>.</p>
</footer>

</body>
</html>
"""


def render_code_page(section, code, desc, fixture, neighbours):
    kind = "Warning" if code.startswith("W") else "Error"
    summary = desc or f"{kind} {code}, raised by the Vx compiler."
    # "Error · Topology/Hardware Errors" stutters; the group name already ends in its kind.
    group = re.sub(r"\s*(Errors|Warnings)$", "", section).strip()
    meta = f"{kind} &middot; {html.escape(group)}" if group else kind
    parts = [
        '<article class="post">',
        "",
        f"<h1>{code}</h1>",
        f'<p class="post-meta">{meta}</p>',
        "",
        f'<blockquote class="tldr"><strong>{code}</strong> — {prose(summary)}</blockquote>',
    ]

    if fixture:
        asserted = assertions_of(fixture, code)
        parts += [
            "",
            "<h2>What the compiler reports</h2>",
            '<pre class="diag"><code>' + html.escape("\n".join(asserted)) + "</code></pre>",
            '<p class="small">The fragments the test suite holds the compiler to. An ellipsis '
            "marks text the test does not constrain; the emitted message also carries a source "
            "location.</p>",
            "",
            "<h2>A program that triggers it</h2>",
            "<pre><code>" + highlight(program_of(fixture)) + "</code></pre>",
            f'<p class="small">From <a href="{REPO}/blob/main/{fixture.as_posix()}">'
            f"<code>{fixture.as_posix()}</code></a>, which asserts this diagnostic on every "
            "commit.</p>",
        ]
    else:
        parts += [
            "",
            "<h2>No example yet</h2>",
            "<p>No fixture in the test suite asserts this code, so there is no program here "
            "that is known to trigger it. That is a gap in our coverage rather than a "
            "statement about the code.</p>",
            f'<p class="small">Contributing one is a good first change — add a fixture under '
            f'<a href="{REPO}/tree/main/tests">'
            "<code>tests/</code></a> with a <code>CHECK</code> line naming "
            f"<code>{code}</code>, and this page picks it up.</p>",
        ]

    chapter = SECTION_CHAPTER.get(section)
    parts += ["", "<h2>Related</h2>", '<div class="links">']
    if chapter:
        parts.append(
            f'<a href="{chapter[1]}"><strong>{html.escape(chapter[0])}</strong>'
            f"<span>The chapter covering the rule this code enforces.</span></a>"
        )
    parts.append(
        '<a href="/errors/"><strong>All diagnostics</strong>'
        "<span>Every code the compiler can emit, grouped by stage.</span></a>"
    )
    for other in neighbours:
        parts.append(
            f'<a href="/errors/{other[0]}/"><strong>{other[0]}</strong>'
            f"<span>{prose(other[1] or 'Neighbouring code in this group.')}</span></a>"
        )
    parts += ["</div>", "", "</article>"]

    return SHELL.format(
        title=f"{code} — Vx",
        description=plain(summary)[:300],
        url=f"https://vxlang.org/errors/{code}/",
        image=f"https://vxlang.org/cards/errors/{code}.png",
        image_alt=plain(f"{code}: {summary}")[:200],
        body="\n".join(parts),
    )


def render_index(entries, index):
    by_section = {}
    for section, code, desc in entries:
        by_section.setdefault(section or "Uncategorised", []).append((code, desc))

    covered = sum(1 for _, c, _ in entries if fixture_for(c, index))
    parts = [
        '<article class="post">',
        "",
        "<h1>Diagnostics</h1>",
        f'<p class="post-meta">{len(entries)} codes &middot; {covered} with a worked example</p>',
        "",
        "<p class=\"lede\">Every diagnostic <code>vxc</code> can emit, each on its own page with "
        "the message and, where the test suite has one, a program that triggers it.</p>",
        "",
        '<div class="links">',
    ]

    for section, items in by_section.items():
        anchor = section.lower().replace(" ", "-").replace("/", "")
        codes = [c for c, _ in items]
        parts.append(
            f'<a href="#{anchor}"><strong>{html.escape(section)}</strong>'
            f"<span><code>{codes[0]}</code>–<code>{codes[-1]}</code>, "
            f"{len(codes)} codes</span></a>"
        )
    parts.append("</div>")

    for section, items in by_section.items():
        anchor = section.lower().replace(" ", "-").replace("/", "")
        parts += [
            "",
            f'<h2 id="{anchor}">{html.escape(section)}</h2>',
            "<table>",
            "<tr><th>Code</th><th>Meaning</th></tr>",
        ]
        for code, desc in items:
            mark = "" if fixture_for(code, index) else ' <span class="small">(no example)</span>'
            parts.append(
                f'<tr><td><a href="/errors/{code}/"><code>{code}</code></a></td>'
                f"<td>{prose(desc or '—')}{mark}</td></tr>"
            )
        parts.append("</table>")

    parts += ["", "</article>"]
    return SHELL.format(
        title="Diagnostics — Vx",
        description=(
            f"Every one of the {len(entries)} diagnostics the Vx compiler can emit, "
            "each with the message and a program that triggers it."
        ),
        url="https://vxlang.org/errors/",
        image="https://vxlang.org/cards/errors.png",
        image_alt=f"The Vx diagnostic index: {len(entries)} codes the compiler can emit.",
        body="\n".join(parts),
    )


def main():
    if not SOURCE.exists():
        print(f"error: {SOURCE} not found; run from the repository root", file=sys.stderr)
        return 1

    entries = list(parse(SOURCE.read_text()))
    if not entries:
        print("error: no diagnostic codes parsed -- has the table's shape changed?", file=sys.stderr)
        return 1

    index = build_index()
    fixtures = {code: fixture_for(code, index) for _, code, _ in entries}
    covered = sum(1 for f in fixtures.values() if f)

    if "--check" in sys.argv:
        print(f"{len(entries)} diagnostics, {covered} with a fixture asserting them")
        for section, code, _ in entries:
            if not fixtures[code]:
                print(f"  no example: {code}  ({section})")
        return 0

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    in_section = {}
    for section, code, desc in entries:
        in_section.setdefault(section, []).append((code, desc))

    for section, code, desc in entries:
        siblings = in_section[section]
        at = next(i for i, (c, _) in enumerate(siblings) if c == code)
        neighbours = [siblings[i] for i in (at - 1, at + 1) if 0 <= i < len(siblings) and i != at]
        page = OUTPUT / code
        page.mkdir()
        (page / "index.html").write_text(
            render_code_page(section, code, desc, fixtures[code], neighbours)
        )

    (OUTPUT / "index.html").write_text(render_index(entries, index))
    print(f"wrote {OUTPUT}/ — {len(entries)} pages, {covered} with a worked example")
    return 0


if __name__ == "__main__":
    sys.exit(main())
