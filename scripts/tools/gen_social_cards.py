#!/usr/bin/env python3
"""Render the social preview image for every page of vxlang.org.

A link posted to Twitter or LinkedIn shows the image the page names in `og:image`. Without one
a `summary_large_image` card falls back to bare text, which is what every Vx link did before
this existed.

The cards are screenshots of a small HTML page rendered headless, so they use the site's own
colours and type rather than a second set that drifts. For a diagnostic the card *is* the
message the compiler prints, which is the thing worth looking at.

    python3 scripts/tools/gen_social_cards.py            # write www/cards/
    python3 scripts/tools/gen_social_cards.py --only default,errors

Needs a Chrome or Chromium binary. Set CHROME to name one explicitly.

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import html
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_error_index import SOURCE, parse  # noqa: E402
from gen_error_pages import assertions_of, build_index, fixture_for  # noqa: E402

OUTPUT = Path("www/cards")
BLOG = Path("www/blog")
WIDTH, HEIGHT = 1200, 630

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  :root {{
    --bg: #0c0d10; --bg-code: #16181d; --border: #24272e;
    --fg: #e6e8ec; --fg-muted: #9aa1ad; --fg-faint: #6b7280;
    --accent: #7dd3fc; --accent-2: #c084fc;
    --mono: ui-monospace, "SF Mono", "DejaVu Sans Mono", Menlo, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
            "DejaVu Sans", Helvetica, Arial, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ width: {width}px; height: {height}px; }}
  body {{
    background: var(--bg); color: var(--fg); font-family: var(--sans);
    padding: 64px 72px; display: flex; flex-direction: column;
    justify-content: space-between; position: relative; overflow: hidden;
  }}
  .rule {{
    position: absolute; top: 0; left: 0; right: 0; height: 6px;
    background: linear-gradient(100deg, var(--accent), var(--accent-2));
  }}
  .top {{ display: flex; align-items: center; gap: 18px; }}
  .mark {{
    font-family: var(--mono); font-weight: 700; font-size: 30px;
    letter-spacing: -0.03em; border: 1px solid var(--border);
    border-radius: 9px; padding: 5px 14px;
  }}
  .kicker {{ font-size: 21px; color: var(--fg-faint); }}
  h1 {{
    font-size: {title_size}px; line-height: 1.06; letter-spacing: -0.035em;
    font-weight: 700; max-width: 21ch;
  }}
  h1.code {{ font-family: var(--mono); letter-spacing: -0.02em; }}
  .grad {{
    background: linear-gradient(100deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .msg {{
    margin-top: 26px; font-family: var(--mono); font-size: 22px; line-height: 1.55;
    color: var(--fg-muted); background: var(--bg-code); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 22px; max-width: 1010px;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical;
    overflow: hidden;
  }}
  .lede {{
    margin-top: 24px; font-size: 25px; line-height: 1.5; color: var(--fg-muted);
    max-width: 52ch; display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
  }}
  .foot {{
    font-size: 21px; color: var(--fg-faint); display: flex;
    justify-content: space-between; gap: 32px;
  }}
  .foot span:first-child {{ overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
  .foot span:last-child {{ white-space: nowrap; }}
</style>
</head>
<body>
  <div class="rule"></div>
  <div class="top">
    <span class="mark">Vx</span>
    <span class="kicker">{kicker}</span>
  </div>
  <div>
    <h1 class="{title_class}">{title}</h1>
    {extra}
  </div>
  <div class="foot"><span>{foot_left}</span><span>{foot_right}</span></div>
</body>
</html>
"""


def find_chrome():
    for candidate in [c for c in [__import__("os").environ.get("CHROME")] if c] + CHROME_CANDIDATES:
        found = candidate if Path(candidate).exists() else shutil.which(candidate)
        if found:
            return found
    return None


def clip(text, limit):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def card_html(kicker, title, *, title_class="", msg=None, lede=None, foot_left="", foot_right="vxlang.org", title_size=72):
    extra = ""
    if msg:
        extra = f'<div class="msg">{html.escape(msg)}</div>'
    elif lede:
        extra = f'<div class="lede">{html.escape(lede)}</div>'
    return TEMPLATE.format(
        width=WIDTH,
        height=HEIGHT,
        kicker=html.escape(kicker),
        title=title,  # caller escapes; it may carry a <span class="grad">
        title_class=title_class,
        title_size=title_size,
        extra=extra,
        foot_left=html.escape(foot_left),
        foot_right=html.escape(foot_right),
    )


def meta_of(path):
    """(title, description) from a hand-written page's head."""
    text = path.read_text(errors="ignore")
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    desc = re.search(r'<meta name="description" content="(.*?)">', text, re.S)
    strip = lambda m: html.unescape(" ".join(m.group(1).split())) if m else ""  # noqa: E731
    return strip(title), strip(desc)


def targets():
    """Yield (output path relative to OUTPUT, html) for every card."""
    yield (
        "default.png",
        card_html(
            "Systems language for heterogeneous compute",
            'One Language,<br><span class="grad">Every Core</span>',
            lede="Placement and reachability in the type system. A host thread dereferencing a "
            "device pointer is a compile error.",
            foot_left="CPU · GPU · NPU · accelerators",
        ),
    )

    for post in sorted(BLOG.glob("*.html")):
        title, desc = meta_of(post)
        title = title.split(" — ")[0].strip()
        if post.name == "index.html":
            yield (
                "blog.png",
                card_html("Blog", html.escape("Notes on building Vx"), lede=desc,
                          foot_left="vxlang.org/blog", foot_right="Aditya Kumar"),
            )
        else:
            yield (
                f"blog-{post.stem}.png",
                card_html("Blog", html.escape(title), lede=desc,
                          foot_left="vxlang.org/blog", foot_right="Aditya Kumar",
                          title_size=60),
            )

    entries = list(parse(SOURCE.read_text()))
    index = build_index()
    yield (
        "errors.png",
        card_html(
            "Reference",
            html.escape("Diagnostics"),
            lede=f"Every one of the {len(entries)} diagnostics vxc can emit, each with the "
            "message and a program that triggers it.",
            foot_left="vxlang.org/errors",
        ),
    )

    for section, code, desc in entries:
        kind = "Warning" if code.startswith("W") else "Error"
        group = re.sub(r"\s*(Errors|Warnings)$", "", section).strip()
        fixture = fixture_for(code, index)
        message = " ".join(assertions_of(fixture, code)) if fixture else ""
        # The code's own prefix is already the headline; drop it from the message body.
        message = re.sub(rf"^{code}\s*…?\s*", "", message).strip()
        yield (
            f"errors/{code}.png",
            card_html(
                f"{kind} · {group}" if group else kind,
                f'<span class="grad">{code}</span>',
                title_class="code",
                title_size=96,
                msg=clip(message, 200) if message else None,
                lede=None if message else clip(desc, 150),
                foot_left=clip(desc, 78) if message else "",
                foot_right="vxlang.org/errors",
            ),
        )


def main():
    chrome = find_chrome()
    if not chrome:
        print(
            "error: no Chrome or Chromium found. Set CHROME to the binary.\n"
            f"  Looked for: {', '.join(CHROME_CANDIDATES)}",
            file=sys.stderr,
        )
        return 1

    only = None
    for arg in sys.argv[1:]:
        if arg.startswith("--only"):
            only = set(arg.split("=", 1)[1].split(",")) if "=" in arg else None

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "errors").mkdir(parents=True)

    written = 0
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "card.html"
        for name, markup in targets():
            if only and Path(name).stem not in only:
                continue
            page.write_text(markup)
            out = OUTPUT / name
            result = subprocess.run(
                [
                    chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--window-size={WIDTH},{HEIGHT}", f"--screenshot={out}", str(page),
                ],
                capture_output=True,
            )
            if not out.exists():
                print(
                    f"error: {chrome} wrote no image for {name}\n"
                    f"{result.stderr.decode(errors='ignore')[:400]}",
                    file=sys.stderr,
                )
                return 1
            written += 1

    print(f"wrote {OUTPUT}/ — {written} cards at {WIDTH}x{HEIGHT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
