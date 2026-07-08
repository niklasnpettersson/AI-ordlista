#!/usr/bin/env python3
"""Generate a standalone HTML glossary from docs/AI_GLOSSARY.md."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ENTRY_PATTERN = re.compile(r"^\*\*(.+?)\*\* — (.+)$")
SECTION_PATTERN = re.compile(r"^## (.+)$")
SKIP_SECTIONS = {"Innehållsförteckning"}


def slugify(title: str) -> str:
    slug = title.lower()
    slug = slug.replace(" & ", "-")
    slug = slug.replace(" ", "-")
    slug = slug.replace(":", "")
    slug = slug.replace("ä", "a").replace("å", "a").replace("ö", "o")
    slug = slug.replace("é", "e")
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def parse_glossary(markdown_text: str) -> tuple[str, list[tuple[str, str, list[tuple[str, str]]]]]:
    lines = markdown_text.splitlines()
    intro_lines: list[str] = []
    sections: list[tuple[str, str, list[tuple[str, str]]]] = []
    current_title: str | None = None
    current_slug: str | None = None
    current_entries: list[tuple[str, str]] = []
    in_intro = True

    for line in lines:
        section_match = SECTION_PATTERN.match(line)
        if section_match:
            title = section_match.group(1).strip()
            if title in SKIP_SECTIONS:
                in_intro = False
                continue
            if current_title is not None:
                sections.append((current_title, current_slug or "", current_entries))
            current_title = title
            current_slug = slugify(title)
            current_entries = []
            in_intro = False
            continue

        entry_match = ENTRY_PATTERN.match(line)
        if entry_match and current_title is not None:
            current_entries.append((entry_match.group(1).strip(), entry_match.group(2).strip()))
            continue

        if in_intro and line.strip() and not line.startswith("#"):
            intro_lines.append(line.strip())
        elif line.startswith("Total terms:"):
            break

    if current_title is not None:
        sections.append((current_title, current_slug or "", current_entries))

    intro = " ".join(intro_lines)
    return intro, sections


def render_html(intro: str, sections: list[tuple[str, str, list[tuple[str, str]]]]) -> str:
    total_entries = sum(len(entries) for _, _, entries in sections)
    nav_items = "\n".join(
        f'          <a href="#{slug}" data-section="{html.escape(title)}">{html.escape(title)}</a>'
        for title, slug, _ in sections
    )
    section_blocks: list[str] = []
    for title, slug, entries in sections:
        entry_cards = "\n".join(
            f"""          <article class="entry" data-term="{html.escape(term.lower())}" data-text="{html.escape((term + ' ' + explanation).lower())}">
            <h3>{html.escape(term)}</h3>
            <p>{html.escape(explanation)}</p>
          </article>"""
            for term, explanation in entries
        )
        section_blocks.append(
            f"""      <section id="{slug}" class="section">
        <h2>{html.escape(title)}</h2>
        <p class="section-meta">{len(entries)} termer</p>
        <div class="entries">
{entry_cards}
        </div>
      </section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="sv">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI-ordlista (Svenska)</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f8fafc;
      --panel: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0b1220;
        --panel: #111827;
        --text: #e5e7eb;
        --muted: #94a3b8;
        --border: #1f2937;
        --accent: #60a5fa;
        --accent-soft: #1e3a8a;
        --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 1.25rem;
      border-right: 1px solid var(--border);
      background: var(--panel);
    }}
    .sidebar h1 {{
      font-size: 1.15rem;
      margin: 0 0 0.5rem;
    }}
    .sidebar p {{
      margin: 0 0 1rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .search {{
      width: 100%;
      padding: 0.7rem 0.85rem;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      background: var(--bg);
      color: var(--text);
      margin-bottom: 1rem;
    }}
    .search:focus {{
      outline: 2px solid var(--accent-soft);
      border-color: var(--accent);
    }}
    .nav {{
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }}
    .nav a {{
      color: var(--text);
      text-decoration: none;
      padding: 0.45rem 0.6rem;
      border-radius: 0.55rem;
      font-size: 0.92rem;
    }}
    .nav a:hover, .nav a.active {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    .stats {{
      margin-top: 1rem;
      font-size: 0.85rem;
      color: var(--muted);
    }}
    main {{
      padding: 2rem;
      max-width: 980px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 1.5rem;
      box-shadow: var(--shadow);
      margin-bottom: 1.5rem;
    }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .section {{
      margin-bottom: 2.5rem;
      scroll-margin-top: 1rem;
    }}
    .section h2 {{
      margin: 0 0 0.25rem;
      font-size: 1.5rem;
    }}
    .section-meta {{
      margin: 0 0 1rem;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .entries {{
      display: grid;
      gap: 0.85rem;
    }}
    .entry {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 0.9rem;
      padding: 1rem 1.1rem;
      box-shadow: var(--shadow);
    }}
    .entry.hidden {{ display: none; }}
    .entry h3 {{
      margin: 0 0 0.35rem;
      font-size: 1rem;
      color: var(--accent);
    }}
    .entry p {{
      margin: 0;
    }}
    .empty-state {{
      display: none;
      padding: 1rem;
      border: 1px dashed var(--border);
      border-radius: 0.75rem;
      color: var(--muted);
      background: var(--panel);
    }}
    .empty-state.visible {{ display: block; }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{
        position: relative;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--border);
      }}
      main {{ padding: 1rem; }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>AI-ordlista</h1>
      <p>Svenska förklaringar av AI/ML-begrepp.</p>
      <input id="search" class="search" type="search" placeholder="Sök term eller text…" aria-label="Sök i ordlistan">
      <nav class="nav" id="nav">
{nav_items}
      </nav>
      <div class="stats">{total_entries} termer · {len(sections)} sektioner</div>
    </aside>
    <main>
      <div class="hero">
        <p>{html.escape(intro)}</p>
      </div>
      <div id="empty" class="empty-state">Inga träffar. Prova ett annat sökord.</div>
{"".join(section_blocks)}
    </main>
  </div>
  <script>
    const searchInput = document.getElementById("search");
    const entries = Array.from(document.querySelectorAll(".entry"));
    const sections = Array.from(document.querySelectorAll(".section"));
    const emptyState = document.getElementById("empty");
    const navLinks = Array.from(document.querySelectorAll("#nav a"));

    function applyFilter() {{
      const query = searchInput.value.trim().toLowerCase();
      let visibleCount = 0;

      entries.forEach((entry) => {{
        const haystack = entry.dataset.text || "";
        const visible = !query || haystack.includes(query);
        entry.classList.toggle("hidden", !visible);
        if (visible) visibleCount += 1;
      }});

      sections.forEach((section) => {{
        const visibleEntries = section.querySelectorAll(".entry:not(.hidden)");
        section.style.display = visibleEntries.length ? "" : "none";
      }});

      emptyState.classList.toggle("visible", visibleCount === 0);
    }}

    searchInput.addEventListener("input", applyFilter);

    const observer = new IntersectionObserver((items) => {{
      items.forEach((item) => {{
        if (!item.isIntersecting) return;
        const id = item.target.id;
        navLinks.forEach((link) => {{
          link.classList.toggle("active", link.getAttribute("href") === "#" + id);
        }});
      }});
    }}, {{ rootMargin: "-30% 0px -60% 0px", threshold: 0 }});

    sections.forEach((section) => observer.observe(section));
  </script>
</body>
</html>
"""


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    source = root / "docs" / "AI_GLOSSARY.md"
    target = root / "docs" / "AI_GLOSSARY.html"
    index_target = root / "docs" / "index.html"

    if not source.exists():
        print(f"Missing source file: {source}", file=sys.stderr)
        return 1

    markdown_text = source.read_text(encoding="utf-8")
    intro, sections = parse_glossary(markdown_text)
    html_output = render_html(intro, sections)
    target.write_text(html_output, encoding="utf-8")
    index_target.write_text(html_output, encoding="utf-8")
    print(f"Wrote {target} and {index_target} ({len(sections)} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
