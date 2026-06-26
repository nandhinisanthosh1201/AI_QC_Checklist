"""
tablet_extractor_schedules.py
==============================
Extracts ONLY these "BY OTHERS" schedule tables from architectural
drawing PDFs:
  • PLUMBING SCHEDULE (BY OTHERS)
  • EQUIPMENT SCHEDULE (BY OTHERS)

The MATERIALS / HARDWARE schedule table (the shape-coded octagon/
rectangle table) is intentionally NOT extracted by this script. If you
need that table too, use the original tablet_extractor.py.

NO OCR IS USED ANYWHERE IN THIS PIPELINE. Everything is read directly
from the PDF's internal vector paths and embedded text objects via
PyMuPDF (page.get_drawings() / page.get_text("words")). This only works
on PDFs that contain real vector lines and selectable text (i.e. PDFs
exported/printed from CAD software), not on scanned raster images. If a
PDF page has no embedded text or vector lines, extraction simply returns
None for that table — there is no image-rasterization/OCR fallback.

APPROACH — simple grid schedule tables (Plumbing, Equipment, …):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Vector-only approach, no shape detection needed:                │
  │    1. Find the table's TITLE text by regex (e.g. "PLUMBING       │
  │       SCHEDULE") using contiguous same-line word grouping         │
  │       (breaks on large X gaps so unrelated same-height text       │
  │       elsewhere on the sheet can't get merged in)                 │
  │    2. The H-line directly below the title gives the table's       │
  │       x-extent; subsequent full-width H-lines give row bounds     │
  │    3. V-lines below the header row give column boundaries         │
  │    4. Words are clustered into rows/columns by x/y position       │
  │    5. First data row becomes the header; every other row is       │
  │       zipped 1:1 against those header names — no code/shape       │
  │       parsing, since this table type is a plain grid with no      │
  │       octagon/rectangle markers                                   │
  └─────────────────────────────────────────────────────────────────┘

Output JSON per table:
  {
    "title":   "PLUMBING SCHEDULE (BY OTHERS)",
    "headers": ["ARCH. TAG", "MAKE", "MODEL", "FINISH", "COMMENTS"],
    "rows": [
      {"ARCH. TAG": "L-2", "MAKE": "AMERICAN STANDARD", "MODEL": "9482",
       "FINISH": "WHITE", "COMMENTS": "SINK"},
      ...
    ],
    "table_bbox": [x0, y0, x1, y1]
  }

Usage
-----
    python tablet_extractor_schedules.py <pdf_file> [--page <page_num>] [--all]

Install
-------
    pip install PyMuPDF
"""

import os, re, json
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

# One regex per table type we know how to find; add more here as new
# schedule tables are encountered (e.g. ELECTRICAL SCHEDULE).
SIMPLE_SCHEDULE_TITLES = {
    "plumbing":  re.compile(r'\bPLUMBING\s+SCHEDULE\b', re.I),
    "equipment": re.compile(r'\bEQUIPMENT\s+SCHEDULE\b', re.I),
}


def _merge_vals(vals, tol=3):
    """Merge nearly-identical numeric values (e.g. row Y positions that
    differ by sub-pixel rounding) into a single representative value."""
    merged = []
    for v in sorted(vals):
        if merged and abs(v - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + v) / 2
        else:
            merged.append(v)
    return merged


def _build_contiguous_line_tokens(words: list, y_tol: float = 3.0,
                                  max_x_gap: float = 25.0) -> list[dict]:
    """
    Group raw PDF text words into logical lines, breaking whenever:
      (a) the Y-position differs by more than y_tol from the current band, OR
      (b) the gap to the next word (left-to-right) exceeds max_x_gap.

    Condition (b) is essential for free-floating title-text search across
    a whole page: it prevents unrelated text that merely shares a Y-band
    — e.g. a numbered note marker far to the left of a table title sitting
    at the same height — from being spliced into one "line".
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w[1], w[0]))

    bands: list[list] = []
    cur_band: list = []
    cur_y = None
    for w in sorted_words:
        y0 = w[1]
        if cur_y is None or abs(y0 - cur_y) > y_tol:
            if cur_band:
                bands.append(cur_band)
            cur_band = [w]
            cur_y = y0
        else:
            cur_band.append(w)
    if cur_band:
        bands.append(cur_band)

    result = []
    for band in bands:
        band.sort(key=lambda w: w[0])
        cur_line = [band[0]]
        for w in band[1:]:
            prev = cur_line[-1]
            gap = w[0] - prev[2]
            if gap > max_x_gap:
                result.append(cur_line)
                cur_line = [w]
            else:
                cur_line.append(w)
        result.append(cur_line)

    out = []
    for line in result:
        all_x0 = [w[0] for w in line]; all_y0 = [w[1] for w in line]
        all_x1 = [w[2] for w in line]; all_y1 = [w[3] for w in line]
        texts  = [w[4] for w in line]
        out.append({
            "text": " ".join(texts),
            "x0": min(all_x0), "x1": max(all_x1),
            "y0": min(all_y0), "y1": max(all_y1),
            "word_boxes": line,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLE GRID SCHEDULE TABLES — Plumbing Schedule, Equipment Schedule
# ═══════════════════════════════════════════════════════════════════════════

def _find_title_match(page, title_re, words=None):
    """
    Locate a schedule table's title text (e.g. "PLUMBING SCHEDULE") on the
    page and return its combined bounding box, or None if not found.

    Returns dict: {"x0", "x1", "y0", "y1", "text"} in PDF coordinates.

    Uses _build_contiguous_line_tokens so that unrelated text sharing the
    title's Y-band but sitting far away horizontally (e.g. a numbered
    note marker, a title-block label) is never merged into the match.
    """
    if words is None:
        words = page.get_text("words")
    lines = _build_contiguous_line_tokens(
        [(w[0], w[1], w[2], w[3], w[4]) for w in words],
        y_tol=3.0, max_x_gap=25.0,
    )
    for ln in lines:
        m = title_re.search(ln["text"].upper())
        if m:
            return {"x0": ln["x0"], "x1": ln["x1"],
                    "y0": ln["y0"], "y1": ln["y1"], "text": ln["text"]}
    return None


def _table_bounds_from_title(page, title_box, paths=None):
    """
    Given a title's bounding box, find the enclosing table's full geometry:
      - the title-row / header-row separator line directly below the title
      - the table's x-extent (TX0, TX1) taken from that separator line
      - every subsequent full-width horizontal line (row separators)
      - the table's bottom border

    Returns None if no separator line is found below the title (i.e. this
    title match isn't actually sitting inside a bordered table).
    """
    if paths is None:
        paths = page.get_drawings()

    all_h = []
    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.height < 2 and r.width > 20:
            all_h.append((r.x0, r.x1, r.y0))

    # The title-row/header-row separator must start at/near the title's
    # left edge or extend across it, and sit just below the title text.
    title_cx = (title_box["x0"] + title_box["x1"]) / 2
    candidates = [
        (x0, x1, y) for x0, x1, y in all_h
        if y >= title_box["y1"] - 2
        and x0 - 5 <= title_cx <= x1 + 5
    ]
    if not candidates:
        return None
    # Nearest such line below the title is the title/header separator.
    x0, x1, y_sep = min(candidates, key=lambda t: t[2])
    TX0, TX1 = x0, x1

    # Collect every full-width line from the separator downward — these
    # are the row boundaries (header/body separators + bottom border).
    table_ys = sorted(set(
        y for hx0, hx1, y in all_h
        if hx0 <= TX0 + 10 and hx1 >= TX1 - 10 and y >= title_box["y0"] - 2
    ))
    if not table_ys:
        return None

    # Merge contiguous lines into the table block; stop at the first large
    # gap, which marks the table's actual bottom edge vs. unrelated lines
    # further down the page.
    border_ys = [table_ys[0]]
    for y in table_ys[1:]:
        if y - border_ys[-1] <= 45:
            border_ys.append(y)
        else:
            break

    return {
        "TX0": TX0, "TX1": TX1,
        "title_y0": title_box["y0"], "title_y1": title_box["y1"],
        "header_sep_y": y_sep,
        "border_ys": border_ys,
        "y_top": title_box["y0"] - 3,
        "y_bot": border_ys[-1] + 3,
    }


def _table_columns_below(paths, TX0, TX1, y_from, y_to):
    """
    Column boundaries from vertical lines, restricted to the y-range BELOW
    the title row (y_from = header separator y). This matters because the
    title row spans the full table width with no internal dividers.
    """
    raw_vx = []
    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.width < 2 and r.height > 5:
            if TX0 - 2 <= r.x0 <= TX1 + 2 and r.y1 > y_from - 2 and r.y0 < y_to + 2:
                raw_vx.append(round(r.x0, 1))

    raw_vx = sorted(set(raw_vx))
    col_xs = [raw_vx[0]] if raw_vx else [TX0]
    for x in raw_vx[1:]:
        if x - col_xs[-1] > 5:
            col_xs.append(x)
    if not col_xs or col_xs[-1] < TX1 - 5:
        col_xs.append(TX1)
    if col_xs[0] > TX0 + 5:
        col_xs.insert(0, TX0)
    return col_xs


def extract_simple_schedule_table(page, title_re, paths=None, words=None):
    """
    Extract a plain N-column "BY OTHERS" style schedule table (Plumbing
    Schedule, Equipment Schedule, etc.) identified by its title text.

    Parameters
    ----------
    page     : fitz.Page
    title_re : compiled regex matching the table's title line, e.g.
               re.compile(r'\\bPLUMBING\\s+SCHEDULE\\b', re.I)
    paths, words : optional pre-fetched page.get_drawings() / get_text("words")
               to avoid re-fetching when extracting multiple tables from one page.

    Returns
    -------
    dict or None:
      {
        "title":        "PLUMBING SCHEDULE (BY OTHERS)",
        "headers":      ["ARCH. TAG", "MAKE", "MODEL", "FINISH", "COMMENTS"],
        "rows":         [ {"ARCH. TAG": "L-2", "MAKE": "AMERICAN STANDARD", ...}, ... ],
        "table_bbox":   [TX0, y_top, TX1, y_bot],
      }
    None is returned if the title isn't found or no table geometry can be
    resolved beneath it (e.g. the title text appears in a sentence rather
    than as an actual table header).
    """
    if paths is None:
        paths = page.get_drawings()
    if words is None:
        words = page.get_text("words")

    title_box = _find_title_match(page, title_re, words=words)
    if not title_box:
        return None

    geo = _table_bounds_from_title(page, title_box, paths=paths)
    if geo is None:
        return None

    TX0, TX1   = geo["TX0"], geo["TX1"]
    border_ys  = geo["border_ys"]
    y_top, y_bot = geo["y_top"], geo["y_bot"]

    # ── Column boundaries (below the title band only) ─────────────────
    col_xs = _table_columns_below(
        paths, TX0, TX1,
        y_from=geo["header_sep_y"], y_to=border_ys[-1],
    )

    # ── Grab words strictly inside the table bbox ──────────────────────
    table_words = [
        (x0, y0, x1, y1, text)
        for x0, y0, x1, y1, text, *_ in words
        if TX0 - 5 <= x0 <= TX1 + 10 and y_top <= y0 <= y_bot
    ]
    if not table_words:
        return None

    # ── Cluster into text rows by Y ──────────────────────────────────
    raw_ys      = [w[1] for w in table_words]
    row_line_ys = _merge_vals(raw_ys, tol=3)

    lines_by_y = defaultdict(list)
    for x0, y0, x1, y1, text in table_words:
        ry = min(row_line_ys, key=lambda ly: abs(ly - y0))
        lines_by_y[ry].append((x0, text))

    n_cols = len(col_xs) - 1

    def _assign_cells(words_sorted):
        col_words = defaultdict(list)
        for wx, wt in words_sorted:
            col_idx = n_cols - 1
            for j in range(n_cols):
                if col_xs[j] - 5 <= wx < col_xs[j + 1] + 5:
                    col_idx = j
                    break
            col_words[col_idx].append(wt)
        return [" ".join(col_words.get(j, [])) for j in range(n_cols)]

    structured_rows = []
    for ry in sorted(lines_by_y.keys()):
        words_sorted = sorted(lines_by_y[ry], key=lambda w: w[0])
        cells = _assign_cells(words_sorted)
        structured_rows.append({"y": ry, "cells": cells})

    # ── Identify the header row (first row below the title separator
    #    whose y matches the header band) and use it for column names ──
    headers = None
    data_rows = []
    for row in structured_rows:
        if row["y"] < geo["header_sep_y"] - 5:
            # Inside the title band — skip (already consumed by title match)
            continue
        if headers is None:
            headers = [c.strip() for c in row["cells"]]
            continue
        data_rows.append(row)

    if not headers:
        return None

    # ── Build records: one dict per row, keyed by header name ─────────
    rows_out = []
    for row in data_rows:
        cells = [c.strip() for c in row["cells"]]
        if not any(cells):
            continue
        record = {headers[j]: (cells[j] if j < len(cells) else "")
                  for j in range(len(headers))}
        record["raw_cells"] = cells
        rows_out.append(record)

    return {
        "title":      title_box["text"].strip(),
        "headers":    headers,
        "rows":       rows_out,
        "table_bbox": [TX0, y_top, TX1, y_bot],
    }


def extract_all_simple_schedules_for_page(page) -> dict:
    """
    Run extract_simple_schedule_table() for every known schedule type
    (see SIMPLE_SCHEDULE_TITLES — currently "plumbing" and "equipment")
    on an already-open fitz.Page and return whatever was found.

    Returns
    -------
    {
      "plumbing":  {<extract_simple_schedule_table result>} or None,
      "equipment": {<extract_simple_schedule_table result>} or None,
    }
    """
    paths = page.get_drawings()
    words = page.get_text("words")

    results = {}
    for name, title_re in SIMPLE_SCHEDULE_TITLES.items():
        results[name] = extract_simple_schedule_table(
            page, title_re, paths=paths, words=words
        )
    return results


def extract_all_simple_schedules(pdf_path: str, page_num: int = 0) -> dict:
    """
    Convenience wrapper that opens the PDF, processes a single page, and
    closes it. For multi-page batch runs, prefer opening the document once
    and calling extract_all_simple_schedules_for_page(page) per page
    instead — see main()'s --all loop, which does this to avoid reopening
    an 89-page PDF from disk on every page.

    Returns
    -------
    {
      "plumbing":  {<extract_simple_schedule_table result>} or None,
      "equipment": {<extract_simple_schedule_table result>} or None,
    }
    """
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    results = extract_all_simple_schedules_for_page(page)
    doc.close()
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def extract(input_path: str, page_num: int = 0) -> dict:
    """
    PDF-only extraction — no OCR, no image input. Reads vector paths and
    embedded text directly via PyMuPDF.

    Returns
    -------
    {
      "simple_schedules": {
        "plumbing":  {...} or None,
        "equipment": {...} or None,
      }
    }
    or None if input_path isn't a PDF.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext != ".pdf":
        print(f"  [ERROR] '{input_path}' is not a PDF. This tool only "
              "supports PDF input (no OCR/image fallback).")
        return None

    print(f"  Scanning page {page_num + 1} for schedule tables "
          f"(Plumbing / Equipment) …")
    simple_schedules = extract_all_simple_schedules(input_path, page_num)
    found_names = [n for n, v in simple_schedules.items() if v]
    if found_names:
        print(f"  ✓ Found schedule table(s): {', '.join(found_names)}")
    else:
        print("  ✗ No schedule tables found on this page\n")

    return {"simple_schedules": simple_schedules}


def main():
    import argparse
    import fitz
    parser = argparse.ArgumentParser(
        description="Extract Plumbing Schedule / Equipment Schedule tables "
                     "from architectural drawing PDFs (no OCR).")
    parser.add_argument("input_file", help="Path to input PDF")
    parser.add_argument("--page", "-p", type=int, default=1,
                        help="Page number (1-based) to process from PDF")
    parser.add_argument("--all", action="store_true",
                        help="Process all pages in the PDF")
    args = parser.parse_args()

    input_path = args.input_file
    output_dir = "table extraction output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  Schedule Table Extractor  (Plumbing / Equipment)")
    print(f"  Input : {os.path.basename(input_path)}")
    print("=" * 65 + "\n")

    if not input_path.lower().endswith(".pdf"):
        print(f"  [ERROR] '{input_path}' is not a PDF. This tool only "
              "supports PDF input (no OCR/image fallback).")
        return

    try:
        doc = fitz.open(input_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    if args.all:
        pages_to_process = list(range(doc.page_count))
    else:
        pages_to_process = [max(0, min(args.page - 1, doc.page_count - 1))]

    total_pages = doc.page_count

    # Per-page presence summary, built as we go — this is the answer to
    # "does page N have a Plumbing/Equipment Schedule table or not".
    # page_summary[page_1_based] = {"plumbing": bool, "equipment": bool}
    page_summary  = {}
    master_tables = {}   # page_1_based -> found_schedules dict (only pages with ≥1 table)

    for page_num in pages_to_process:
        page_1based = page_num + 1
        if args.all:
            print(f"--- Page {page_1based}/{doc.page_count} ---", end="  ")

        page = doc[page_num]
        simple_schedules = extract_all_simple_schedules_for_page(page)
        found_schedules = {k: v for k, v in simple_schedules.items() if v}

        page_summary[page_1based] = {
            "plumbing":  bool(simple_schedules.get("plumbing")),
            "equipment": bool(simple_schedules.get("equipment")),
        }

        if args.all:
            tags = []
            if page_summary[page_1based]["plumbing"]:
                tags.append("PLUMBING")
            if page_summary[page_1based]["equipment"]:
                tags.append("EQUIPMENT")
            print(", ".join(tags) if tags else "—")

        if not found_schedules:
            if not args.all:
                print(f"  [Page {page_1based}] No Plumbing or Equipment "
                      "Schedule table found.\n")
            continue

        master_tables[page_1based] = found_schedules

        # Per-page JSON (kept for single-page runs and for anyone who
        # wants one file per page instead of the consolidated master file)
        json_out = os.path.join(output_dir, f"{page_1based}_schedules.json")
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({
                "input_file":       input_path,
                "page":             page_1based,
                "simple_schedules": found_schedules,
            }, f, indent=4, ensure_ascii=False)

        if not args.all:
            # ── Console report (single-page mode) ─────────────────────
            print("\n" + "=" * 65)
            print(f"  EXTRACTED SCHEDULE TABLES  (Page {page_1based})")
            print("=" * 65)
            for name, sched in found_schedules.items():
                print(f"\n  ▶  {sched['title']}")
                col_w = [max(len(h), 10) for h in sched["headers"]]
                hdr_line = "  " + "  ".join(
                    h.ljust(w) for h, w in zip(sched["headers"], col_w))
                print(hdr_line)
                print("  " + "-" * (len(hdr_line) - 2))
                for row in sched["rows"]:
                    cells = [row.get(h, "") for h in sched["headers"]]
                    print("  " + "  ".join(
                        c.ljust(w) for c, w in zip(cells, col_w)))
                print(f"  ({len(sched['rows'])} rows)")
            print(f"\n  JSON saved → {json_out}\n")

    doc.close()

    if not args.all:
        return

    # ── Multi-page summary (the actual answer to "which pages have       ─
    # Plumbing/Equipment tables") ──────────────────────────────────────
    plumbing_pages  = [p for p, v in page_summary.items() if v["plumbing"]]
    equipment_pages = [p for p, v in page_summary.items() if v["equipment"]]
    both_pages      = [p for p, v in page_summary.items()
                       if v["plumbing"] and v["equipment"]]
    plumbing_only   = [p for p in plumbing_pages if p not in equipment_pages]
    equipment_only  = [p for p in equipment_pages if p not in plumbing_pages]
    neither_pages   = [p for p, v in page_summary.items()
                       if not v["plumbing"] and not v["equipment"]]

    print("\n" + "=" * 65)
    print(f"  SUMMARY — {total_pages} pages scanned")
    print("=" * 65)
    print(f"  Pages with PLUMBING Schedule       : {len(plumbing_pages)}  {plumbing_pages}")
    print(f"  Pages with EQUIPMENT Schedule      : {len(equipment_pages)}  {equipment_pages}")
    print(f"  Pages with BOTH tables             : {len(both_pages)}  {both_pages}")
    print(f"  Pages with PLUMBING only           : {len(plumbing_only)}  {plumbing_only}")
    print(f"  Pages with EQUIPMENT only          : {len(equipment_only)}  {equipment_only}")
    print(f"  Pages with NEITHER (drawing only)  : {len(neither_pages)}  {neither_pages}")

    # Master JSON: one file with the per-page presence map (every page,
    # true/false) PLUS the full extracted table data for pages that have
    # at least one table. This is the single file to hand off if you need
    # "tell me for every page whether the tables are present or not".
    master_out = os.path.join(output_dir, "ALL_PAGES_summary.json")
    with open(master_out, "w", encoding="utf-8") as f:
        json.dump({
            "input_file":   input_path,
            "total_pages":  total_pages,
            "page_presence": page_summary,      # {page_num: {"plumbing": bool, "equipment": bool}} for EVERY page
            "tables_by_page": master_tables,    # {page_num: {schedule_name: {title, headers, rows, table_bbox}}} only for pages with ≥1 table found
        }, f, indent=4, ensure_ascii=False)

    print(f"\n  Master summary JSON → {master_out}")
    print(f"  Per-page table JSON → {output_dir}/<page>_schedules.json "
          f"(only for pages with a table)\n")


if __name__ == "__main__":
    main()