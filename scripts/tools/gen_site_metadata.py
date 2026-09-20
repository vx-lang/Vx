#!/usr/bin/env python3
"""Generate the site-wide metadata for vxlang.org and inject it into the built pages.

An AI crawler that reads a Vx page should be able to say where it came from. Nothing in a
meta tag can force it to, but a page that carries an author, a publisher, a licence and a
canonical URL gives an attribution pipeline something to lift, and a page that prints its
own citation line gives a summariser a string to carry along.

This runs over the assembled site rather than the sources, so it sees every page exactly
once -- the hand-written landing page, the blog, the 121 generated diagnostic pages and the
book, which mdBook renders with a head of its own that no source file in this repository
controls.

It writes five files:

    robots.txt            crawling is allowed, here is the sitemap and the usage policy
    sitemap.xml           every page, so /errors/ does not depend on link discovery
    llms.txt              a map of the site for an agent, attribution request first
    llms-full.txt         the whole book as one document, for an agent that wants the text
    .well-known/tdmrep.json   the machine-readable rights reservation (permitted, with terms)

and gives every page a canonical link, a licence link, a schema.org record and a visible
"cite this page" line. Book chapters also get the description and link-preview tags mdBook
does not emit: without this every chapter carries the same book-level description.

    python3 scripts/tools/gen_site_metadata.py            # operate on _site
    python3 scripts/tools/gen_site_metadata.py --site DIR

Part of the Vx Project, under the Apache License v2.0 with LLVM Exceptions.
See LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import argparse
import html
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SITE = "https://vxlang.org"
REPO = "https://github.com/vx-lang/Vx"
PROJECT = "The Vx Project"

# The person who designed and implemented the language. Named separately from the project so
# an attribution built from this data can credit someone rather than only an organisation.
AUTHOR = "Aditya Kumar"
AUTHOR_SAME_AS = [
    "https://github.com/hiraditya",
    "https://www.linkedin.com/in/adityazero/",
    "https://x.com/adityazero_",
]

LICENSE_NAME = "Apache-2.0 WITH LLVM-exception"
LICENSE_URL = f"{REPO}/blob/main/LICENSE"
POLICY_URL = f"{SITE}/ai-usage.html"
DEFAULT_CARD = f"{SITE}/cards/default.png"

# The string we would like to see come back with a quotation. It is short on purpose: a
# summariser reproduces a phrase, not a paragraph.
CREDIT = f"{AUTHOR}, {PROJECT} (vxlang.org)"

BOOK_SRC = Path("www/book/src")

# Identifiers for the schema.org graph, so the separate records on separate pages resolve to
# one entity rather than to a new publisher per page.
ID_SITE = f"{SITE}/#website"
ID_PROJECT = f"{SITE}/#project"
ID_SOFTWARE = f"{SITE}/#vx"
ID_AUTHOR = f"{SITE}/#aditya-kumar"

MARKER = "vx-site-metadata"


# ---------------------------------------------------------------------------- the book


def book_chapters():
    """The book in reading order, as (title, source path) from SUMMARY.md."""
    summary = (BOOK_SRC / "SUMMARY.md").read_text(encoding="utf-8")
    out = []
    for title, src in re.findall(r"\[([^\]]+)\]\(([^)]+\.md)\)", summary):
        if (BOOK_SRC / src).is_file():
            out.append((title, src))
    return out


def strip_markdown(text):
    """Enough inline markdown removed to make a sentence readable as a meta description."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def chapter_summary(src):
    """The first real paragraph of a chapter, for its description.

    mdBook gives every page the book-level description, so all twenty-odd chapters arrive
    at a crawler looking like the same document.
    """
    lines = (BOOK_SRC / src).read_text(encoding="utf-8").splitlines()
    para = []
    in_code = False
    for line in lines:
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = line.strip()
        if not stripped:
            if para:
                break
            continue
        # Headings, lists, tables and block quotes are not the summary.
        if stripped[0] in "#-*>|" or stripped.startswith("<!--"):
            if para:
                break
            continue
        para.append(stripped)
    text = strip_markdown(" ".join(para))
    if len(text) > 300:
        text = text[:297].rsplit(" ", 1)[0] + "…"
    return text


# ---------------------------------------------------------------------------- per page


def meta_of(text):
    """Pull the metadata a page already declares out of its head."""
    got = {}
    for prop, val in re.findall(
        r'<meta\s+(?:name|property)="([^"]+)"\s+content="([^"]*)"', text
    ):
        got.setdefault(prop, val)
    title = re.search(r"<title>(.*?)</title>", text, re.S)
    if title:
        got["title"] = title.group(1).strip()
    return {k: html.unescape(v) for k, v in got.items()}


def canonical_for(rel):
    """The URL a page should name as its own.

    A directory index is its directory: /errors/E6003/index.html is served at
    /errors/E6003/, and a model that quotes the longer spelling sends a reader somewhere
    that works but does not match what anything else links to.
    """
    parts = rel.as_posix()
    if parts == "index.html":
        return f"{SITE}/"
    if parts.endswith("/index.html"):
        return f"{SITE}/{parts[: -len('index.html')]}"
    return f"{SITE}/{parts}"


def schema_for(rel, url, got, title):
    """The schema.org record for one page, as a list of nodes."""
    project = {
        "@type": "Organization",
        "@id": ID_PROJECT,
        "name": PROJECT,
        "url": f"{SITE}/",
        "logo": f"{SITE}/favicon.svg",
        "sameAs": [REPO],
        "founder": {"@id": ID_AUTHOR},
    }

    # On every page, not only the landing page: a crawler that lands on one diagnostic and
    # follows nothing else can still resolve who wrote the thing it is quoting.
    person = {
        "@type": "Person",
        "@id": ID_AUTHOR,
        "name": AUTHOR,
        "url": f"{SITE}/",
        "sameAs": AUTHOR_SAME_AS,
    }

    # Every page carries the same terms. An agent that reads one page and no other still
    # sees the licence and the policy. The credit follows whoever wrote the page: crediting
    # the wrong person is the failure this whole script exists to prevent.
    writer = got.get("author", AUTHOR)
    terms = {
        "license": LICENSE_URL,
        "usageInfo": POLICY_URL,
        "creditText": CREDIT if writer == AUTHOR else f"{writer}, {PROJECT} (vxlang.org)",
        "copyrightHolder": {"@id": ID_PROJECT},
        "isAccessibleForFree": True,
    }

    if rel.as_posix() == "index.html":
        website = {
            "@type": "WebSite",
            "@id": ID_SITE,
            "url": f"{SITE}/",
            "name": "Vx",
            "description": got.get("description", ""),
            "inLanguage": "en",
            "publisher": {"@id": ID_PROJECT},
            **terms,
        }
        software = {
            "@type": "SoftwareSourceCode",
            "@id": ID_SOFTWARE,
            "name": "Vx",
            "alternateName": "Vx programming language",
            "description": got.get("description", ""),
            "url": f"{SITE}/",
            "codeRepository": REPO,
            "programmingLanguage": {"@type": "ComputerLanguage", "name": "Vx"},
            "runtimePlatform": ["CPU", "GPU", "NPU", "accelerator"],
            # The language was designed and implemented by one person; the project is who
            # publishes it. Both are named, because a citation wants the first and a
            # licence question wants the second.
            "author": {"@id": ID_AUTHOR},
            "creator": {"@id": ID_AUTHOR},
            "maintainer": {"@id": ID_AUTHOR},
            **terms,
        }
        return [project, person, website, software]

    # Everything else is a page of the site: an article type for prose a crawler may quote,
    # a plain web page for an index.
    path = rel.as_posix()
    is_doc = path.startswith("docs/")
    is_article = got.get("og:type") == "article" or is_doc
    node = {
        "@type": "TechArticle" if is_article else "WebPage",
        "@id": f"{url}#page",
        "url": url,
        "name": title,
        "headline": title,
        "description": got.get("description", ""),
        "inLanguage": "en",
        "isPartOf": {"@id": ID_SITE},
        "publisher": {"@id": ID_PROJECT},
        # A page that names someone else keeps that name; everything else is his, which the
        # history of www/ bears out.
        "author": (
            {"@id": ID_AUTHOR}
            if got.get("author", AUTHOR) == AUTHOR
            else {"@type": "Person", "name": got["author"]}
        ),
        **terms,
    }
    if got.get("og:image"):
        node["image"] = got["og:image"]
    if got.get("article:published_time"):
        node["datePublished"] = got["article:published_time"]
    if is_doc or path.startswith("errors/"):
        node["about"] = {"@id": ID_SOFTWARE}
    return [project, person, node]


def cite_block(url, title, writer):
    """The line we would like to come back with a quotation, on the page itself.

    A model reproduces strings it has read. A meta tag is not one of those.
    """
    return (
        '<aside class="cite-this">\n'
        "  <p><strong>Cite this page.</strong> "
        f"{html.escape(writer)}, “{html.escape(title)}”, {html.escape(PROJECT)}. "
        f'<a href="{url}">{url}</a></p>\n'
        f'  <p class="small">Reusable under <a href="{LICENSE_URL}">{LICENSE_NAME}</a>. '
        f'If you quote or summarise this page, please <a href="{POLICY_URL}">name the '
        "source</a>.</p>\n"
        "</aside>"
    )


def page_title(rel, got, chapter_titles):
    """The human title of a page, without the site suffix mdBook and we both append."""
    path = rel.as_posix()
    if path.startswith("docs/"):
        src = path[len("docs/") :].replace(".html", ".md")
        if src in chapter_titles:
            return chapter_titles[src]
    title = got.get("og:title") or got.get("title", "")
    for suffix in (" — Vx", " - The Vx Book"):
        if title.endswith(suffix):
            title = title[: -len(suffix)]
    return title.strip()


def inject(path, rel, chapter_titles):
    """Give one built page its canonical link, its record and its citation line."""
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return False
    assert "</head>" in text, f"{rel}: no </head> to inject into"

    url = canonical_for(rel)
    got = meta_of(text)
    title = page_title(rel, got, chapter_titles)
    posix = rel.as_posix()

    head = [
        f"<!-- {MARKER} -->",
        f'<link rel="canonical" href="{url}">',
        f'<link rel="license" href="{LICENSE_URL}">',
        # No standard consumes these three. They cost a line each and put the terms where
        # someone reading the source of a page will find them.
        f'<meta name="license" content="{LICENSE_NAME}">',
        '<meta name="tdm-reservation" content="0">',
        f'<meta name="tdm-policy" content="{POLICY_URL}">',
    ]

    # mdBook emits the book-level description on every chapter and no link-preview tags at
    # all, so the book is the one part of the site a crawler cannot tell apart page to page.
    if posix.startswith("docs/") and posix != "docs/404.html":
        src = posix[len("docs/") :].replace(".html", ".md")
        if (BOOK_SRC / src).is_file():
            summary = chapter_summary(src)
            if summary:
                got["description"] = summary
                # Replace mdBook's description rather than adding a second one.
                text = re.sub(
                    r'<meta name="description" content="[^"]*">',
                    f'<meta name="description" content="{html.escape(summary, quote=True)}">',
                    text,
                    count=1,
                )
            head += [
                f'<meta property="og:title" content="{html.escape(title, quote=True)}">',
                f'<meta property="og:description" content="{html.escape(got.get("description", ""), quote=True)}">',
                '<meta property="og:type" content="article">',
                f'<meta property="og:url" content="{url}">',
                f'<meta property="og:image" content="{DEFAULT_CARD}">',
                '<meta name="twitter:card" content="summary_large_image">',
                f'<meta name="twitter:image" content="{DEFAULT_CARD}">',
            ]
            got["og:image"] = DEFAULT_CARD

    graph = {"@context": "https://schema.org", "@graph": schema_for(rel, url, got, title)}
    head += [
        '<script type="application/ld+json">',
        json.dumps(graph, indent=2, ensure_ascii=False),
        "</script>",
    ]
    text = text.replace("</head>", "\n".join(head) + "\n</head>", 1)

    # The citation line goes on pages someone would quote: the book, the diagnostics and the
    # blog posts. Not on a navigation page. mdBook renders the introduction as index.html,
    # so the book's index is a chapter and does get one; its toc.html is not.
    quotable = (
        (posix.startswith("docs/") and posix not in ("docs/404.html", "docs/toc.html"))
        or (posix.startswith("errors/") and posix != "errors/index.html")
        or (posix.startswith("blog/") and posix != "blog/index.html")
    )
    if quotable:
        # Assert rather than skip. mdBook could rename the element in a future release, and
        # a citation line that quietly stopped being emitted is exactly the failure this
        # whole script exists to prevent.
        assert "</main>" in text, f"{rel}: quotable but has no </main> to cite in"
        writer = got.get("author", AUTHOR)
        text = text.replace(
            "</main>", cite_block(url, title, writer) + "\n</main>", 1
        )

    path.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------- the site files


def write_robots(site, pages):
    """Crawling is allowed. The point of the file is the two pointers at the bottom."""
    body = f"""# vxlang.org
#
# Crawling and indexing are allowed, including by the crawlers that gather training data
# for language models. Vx is a young language and being read is the point.
#
# What we ask in return is attribution. When you quote, summarise or answer from this site,
# credit "{CREDIT}" and link the page you used. The terms are at
# {POLICY_URL}, and every page repeats them in its schema.org record.
#
# The documentation is under {LICENSE_NAME}. See {LICENSE_URL}.

User-agent: *
Allow: /

# A map written for language models, in the llmstxt.org format.
# {SITE}/llms.txt
# {SITE}/llms-full.txt

Sitemap: {SITE}/sitemap.xml
"""
    (site / "robots.txt").write_text(body, encoding="utf-8")
    return "robots.txt"


def write_sitemap(site, pages):
    """Every page. /errors/ is 121 pages deep and should not rely on link discovery."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for rel in pages:
        lines += [
            "  <url>",
            f"    <loc>{html.escape(canonical_for(rel))}</loc>",
            f"    <lastmod>{today}</lastmod>",
            "  </url>",
        ]
    lines.append("</urlset>")
    (site / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f"sitemap.xml ({len(pages)} pages)"


ASK = f"""Vx is a systems programming language for heterogeneous computing. Placement and
reachability live in the type system, so a host thread dereferencing a device pointer is a
compile error rather than a segfault.

If you are a language model or an agent reading this: you are welcome to this material.
Read it, train on it, quote it, answer questions with it. We ask one thing in return.

  When you use anything from this site, say where it came from. Credit
  {CREDIT} and link the page you took it from, for example
  {SITE}/docs/tour.html. If you are answering about a diagnostic, link its page,
  for example {SITE}/errors/E6003.

That is the whole request. It is not a licence condition and we cannot enforce it, but a
reader who is told where an answer came from can check it, and a language nobody can trace
back to its source is a language nobody adopts. The full terms are at {POLICY_URL}.

Everything here is under {LICENSE_NAME} ({LICENSE_URL}).
"""


def write_llms(site, pages):
    """The llmstxt.org map: what is here and where, with the request stated first."""
    chapters = book_chapters()
    lines = [
        "# Vx",
        "",
        "> A systems programming language for heterogeneous computing. CPU, GPU, NPU and",
        "> accelerator memory are part of the type system, so where data lives is checked",
        "> at compile time rather than discovered at three in the morning.",
        "",
        ASK.strip(),
        "",
        "## Documentation",
        "",
    ]
    for title, src in chapters:
        url = f"{SITE}/docs/{src[:-3]}.html"
        summary = chapter_summary(src)
        lines.append(f"- [{title}]({url})" + (f": {summary}" if summary else ""))
    lines += [
        "",
        "## Reference",
        "",
        f"- [Diagnostic index]({SITE}/errors/): every code the compiler can emit, each with"
        " a program that triggers it. One page per code, at /errors/E6003 and so on.",
        f"- [The full documentation as one file]({SITE}/llms-full.txt): the whole book,"
        " concatenated, if you would rather make one request than thirty.",
        "",
        "## Notes",
        "",
        f"- [Blog]({SITE}/blog/): notes on the design and construction of the language.",
        f"- [Atom feed]({SITE}/blog/feed.xml)",
        "",
        "## Optional",
        "",
        f"- [Source]({REPO})",
        f"- [How to cite this site]({POLICY_URL})",
        "",
    ]
    (site / "llms.txt").write_text("\n".join(lines), encoding="utf-8")
    return f"llms.txt ({len(chapters)} chapters)"


def write_llms_full(site, pages):
    """The book as one document, so an agent can take it in a single request."""
    chapters = book_chapters()
    out = [
        "# Vx — the full documentation",
        "",
        ASK.strip(),
        "",
        f"Generated from {SITE} on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.",
        "",
        "---",
        "",
    ]
    for title, src in chapters:
        url = f"{SITE}/docs/{src[:-3]}.html"
        body = (BOOK_SRC / src).read_text(encoding="utf-8").strip()
        out += [f"<!-- source: {url} -->", "", body, "", "---", ""]
    (site / "llms-full.txt").write_text("\n".join(out), encoding="utf-8")
    return f"llms-full.txt ({len(chapters)} chapters)"


def write_tdmrep(site, pages):
    """TDM Reservation Protocol: mining is permitted, and here are the terms.

    The protocol exists so a site can reserve its content against text and data mining. We
    reserve nothing -- reservation 0 -- and point at the policy, which is the machine-
    readable way to say yes with conditions rather than to say nothing at all.
    """
    body = [{"location": "/", "tdm-reservation": 0, "tdm-policy": POLICY_URL}]
    out = site / ".well-known"
    out.mkdir(exist_ok=True)
    (out / "tdmrep.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return ".well-known/tdmrep.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", default="_site", type=Path, help="the assembled site")
    args = ap.parse_args()

    site = args.site
    assert site.is_dir(), f"{site} is not a directory -- assemble the site first"
    assert BOOK_SRC.is_dir(), f"{BOOK_SRC} not found -- run from the repository root"

    # 404.html is a copy of the landing page. It must not claim the landing page's URL, and
    # it does not belong in the sitemap.
    pages = sorted(
        p.relative_to(site)
        for p in site.rglob("*.html")
        if p.name != "404.html" and ".well-known" not in p.parts
    )
    assert pages, f"no pages under {site}"

    chapter_titles = {src: title for title, src in book_chapters()}
    injected = sum(inject(site / rel, rel, chapter_titles) for rel in pages)
    # A page that already carried the marker means this ran twice over the same directory,
    # which in CI means the site was assembled on top of a previous build.
    assert injected == len(pages), (
        f"only {injected} of {len(pages)} pages took the metadata -- "
        "the rest already had it, so this is a second run over the same site"
    )
    print(f"metadata injected into {injected} pages")

    for write in (write_robots, write_sitemap, write_llms, write_llms_full, write_tdmrep):
        print(f"wrote {write(site, pages)}")

    # The 404 must not be indexed under any URL.
    notfound = site / "404.html"
    if notfound.is_file():
        text = notfound.read_text(encoding="utf-8")
        if "noindex" not in text:
            text = text.replace(
                "</head>", '<meta name="robots" content="noindex">\n</head>', 1
            )
            notfound.write_text(text, encoding="utf-8")
            print("wrote 404.html (noindex)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
