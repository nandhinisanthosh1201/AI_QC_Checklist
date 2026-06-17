

"""
tablet_extractor_paddle.py
===========================
Extracts the materials/hardware schedule table from architectural drawings,
then verifies which extracted codes are actually referenced in the drawing area.

APPROACH:
  ┌─────────────────────────────────────────────────────────────────┐
  │  PDF VECTOR EXTRACTION (Fast, perfect accuracy)                 │
  │    1. Extract all drawing paths (H+V lines) from the PDF       │
  │    2. Find the H-line span repeated 4+ times → table x-extent  │
  │    3. Get V-lines inside that x-range → column boundaries      │
  │    4. Extract all text words in the table region               │
  │    5. Cluster words by Y-position → one text row per line      │
  │    6. Assign words to (row, col) by x-position                 │
  │    7. Detect per-row code cell SHAPE from vector path items:   │
  │         • 8-segment 'l' polygon → "octagon"  (MATERIALS rows)  │
  │         • single 're' command   → "rectangle" (HARDWARE rows)  │
  │    8. Detect section headers (MATERIALS / HARDWARE)            │
  │       and build structured JSON output                         │
  └─────────────────────────────────────────────────────────────────┘

DRAWING PRESENCE VERIFICATION (runs after table extraction):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Condition: Only runs when a table was found on the page.      │
  │  If no table exists, the check is skipped entirely.            │
  │                                                                 │
  │  Multi-strategy matching (all run in parallel per word/token): │
  │    Strategy 1 — Single-word exact: CODE_RE match on each word  │
  │    Strategy 2 — Adjacent-word join: merge pairs/triples of     │
  │                 nearby words (handles "PL" + "-E" splits)      │
  │    Strategy 3 — Full-line scan: reconstruct whole text lines   │
  │                 from words sharing the same Y band and search  │
  │                 for each code anywhere in the joined string     │
  │    Strategy 4 — Strip-and-compare: remove all punctuation,     │
  │                 spaces, quotes from both sides and compare      │
  │                                                                 │
  │  Zone logic (refined):                                         │
  │    • TABLE ZONE    : the bounding box of the extracted         │
  │                      schedule table (codes here don't count)   │
  │    • DRAWING ZONE  : everything else on the page               │
  │      (NO blanket right-side exclusion — title blocks are       │
  │       excluded only if they sit inside the detected table bbox) │
  │                                                                 │
  │  For each match → classify enclosing vector shape              │
  │  (octagon / rectangle / none) and record x, y, shape.         │
  └─────────────────────────────────────────────────────────────────┘

Output JSON per row:
  {
    "code":               "PL-E",
    "shape":              "octagon",
    "qty":                "",
    "description":        "(PL-3) WILSONART ...",
    "present_in_drawing": true,
    "drawing_occurrences": [
      {"x": 200, "y": 190, "shape": "octagon"},
      {"x": 339, "y": 520, "shape": "octagon"}
    ]
  }

Shape values
------------
  "octagon"   – 8-sided cut-corner polygon  (MATERIALS section codes)
  "rectangle" – plain axis-aligned rectangle (HARDWARE section codes)
  "circle"    – circle  (detected by OCR path; not yet seen in vector PDFs)
  "none"      – no recognisable enclosing shape found

Usage
-----
    python tablet_extractor_paddle.py <image_or_pdf> [--page <page_num>] [--all]

Install
-------
    pip install PyMuPDF paddlepaddle paddleocr opencv-python pillow numpy
"""

import sys, os, io, re, json
import cv2
import numpy as np
from PIL import Image
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═══════════════════════════════════════════════════════════════════════════

ANCHOR_RE    = re.compile(r'\b(MATERIALS|HARDWARE|QTY\.?)\b', re.I)
# CODE_RE uses \b before but NOT after, so codes ending in digits/dots
# (e.g. "H-1.2") are captured correctly even when '.' is not a \w char.
CODE_RE      = re.compile(r'\b([A-Z]{1,3}-(?:[A-Z]{1,2}|[0-9][A-Z0-9.]*))')
# Boundary-aware pattern for loose scanning — only used internally in
# _match_code_in_token with an additional exact-length guard.
CODE_RE_LOOSE = re.compile(r'(?<![A-Z0-9-])([A-Z]{1,3}-(?:[A-Z]{1,2}|[0-9][A-Z0-9.]*))(?![A-Z0-9-])')
PREFIX_FIXES = {"PI": "PL", "P1": "PL", "G1": "GL", "6L": "GL"}


def _merge_vals(vals, tol=3):
    merged = []
    for v in sorted(vals):
        if merged and abs(v - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + v) / 2
        else:
            merged.append(v)
    return merged


def _clean_code(text):
    if not text:
        return ""
    t = text.upper().strip()
    t = re.sub(r'\s*-\s*', '-', t)
    t = re.sub(r'(\d),(\d)', r'\1.\2', t)
    # Only apply prefix fixes for known OCR mis-reads, not all 2-char prefixes.
    t = re.sub(r'\b(PI|P1|G1|6L)-',
               lambda m: PREFIX_FIXES.get(m.group(1), m.group(1)) + "-", t)
    t = re.sub(r'[.,;:!?]+$', '', t)
    m = CODE_RE.search(t)
    return m.group(1) if m else t.strip()


def _strip_punct(s: str) -> str:
    """Remove all non-alphanumeric, non-hyphen characters and uppercase."""
    return re.sub(r'[^A-Z0-9\-]', '', s.upper())


def _code_in_text(code: str, text: str) -> bool:
    """
    Return True only if `code` appears as a whole token in `text`.
    'Whole token' means surrounded by non-alphanumeric/non-hyphen characters
    (or string boundaries).  This prevents 'H-1' from matching inside 'H-10'
    and 'PL' from matching inside 'MAPLE'.
    """
    # Escape the code for use in a regex (hyphens and dots are special)
    escaped = re.escape(code)
    # Boundaries: not preceded/followed by alphanumeric chars or hyphens
    pattern = r'(?<![A-Z0-9\-])' + escaped + r'(?![A-Z0-9\-\.])'
    return bool(re.search(pattern, text, re.IGNORECASE))


# ═══════════════════════════════════════════════════════════════════════════
# PATH A — PDF VECTOR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

# ── Vector shape classifier ────────────────────────────────────────────────

def _classify_path_shape(path) -> str:
    """
    Classify the enclosing shape drawn around a code label using raw PDF
    vector path data returned by page.get_drawings().

    OCTAGON   – exactly 8 'l' line segments forming a convex cut-corner
                polygon. Used for MATERIALS codes (MM-A, PL-C, PL-E, WD-A).
    RECTANGLE – a single 're' rectangle command.
                Used for HARDWARE codes (H-1.x, H-2.x …).
    CIRCLE    – four 'c' bezier curves (standard PDF circle encoding).
    NONE      – anything else.
    """
    items = path.get('items', [])
    if not items:
        return "none"
    types = [it[0] for it in items]

    if len(items) == 1 and types[0] == 're':
        return "rectangle"
        
    if 4 <= len(items) <= 6:
        # Require exactly one move ('m') and the rest lines/curves,
        # meaning it's a single continuous path (no intersecting crosses)
        if types.count('m') == 1 and all(t in ('m', 'l', 'c') for t in types):
            # Also check if it's generally rectangular via bounding box aspect
            r = path.get('rect')
            if r and r.height > 0:
                aspect = r.width / r.height
                if 0.1 <= aspect <= 10.0:  # Loose aspect for long labels
                    return "rectangle"

    if len(items) == 8 and all(t == 'l' for t in types):
        r = path.get('rect')
        if r and r.height > 0:
            aspect = r.width / r.height
            # Octagons in these drawings are typically wider than tall but can
            # approach 1:1 for small labels, so accept 0.8–6.0 range.
            if 0.8 <= aspect <= 6.0:
                return "octagon"

    if len(items) == 4 and all(t == 'c' for t in types):
        return "circle"

    if len(items) == 1 and types[0] == 'c':
        return "circle"

    return "none"


def _build_code_shape_map(paths, code_col_x0: float, code_col_x1: float,
                          table_y0: float, table_y1: float) -> dict:
    """
    Map every drawing path inside the code column to its shape, keyed by
    y-centre.  Full-row-width border rectangles are excluded via a width cap.
    """
    shape_map: dict[float, str] = {}
    col_width   = code_col_x1 - code_col_x0
    max_shape_w = col_width * 3        # ~3× column width  ≈ 90 pts

    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.x0 < code_col_x0 - 5 or r.x0 > code_col_x1 + 5:
            continue
        if r.y0 < table_y0 - 2 or r.y1 > table_y1 + 2:
            continue
        if r.height < 3 or r.width < 3:
            continue
        if r.width > max_shape_w:
            continue

        shape = _classify_path_shape(p)
        if shape == "none":
            continue

        y_center = round((r.y0 + r.y1) / 2, 1)
        existing = shape_map.get(y_center, "none")
        if existing == "none" or (existing == "rectangle" and shape == "octagon"):
            shape_map[y_center] = shape

    return shape_map


def _lookup_shape(shape_map: dict, row_y: float, tol: float = 6.0) -> str:
    best_shape = "none"
    best_dist  = tol + 1
    for sy, shape in shape_map.items():
        dist = abs(sy - row_y)
        if dist < best_dist:
            best_dist  = dist
            best_shape = shape
    return best_shape


# ── Drawing-presence verification ─────────────────────────────────────────

def _find_enclosing_shape_at(paths, cx: float, cy: float) -> str:
    """
    Given a point (cx, cy) in PDF coordinates, find the smallest path whose
    bounding rectangle contains that point and return its classified shape.
    'Smallest' is measured by area to prefer the tight code-label shape over
    any large enclosing panel or border rectangle.
    """
    best_shape = "none"
    best_area  = float('inf')

    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.height < 3 or r.width < 3:
            continue
        # Code markers are small tags. Prevent large architectural features
        if r.width > 150 or r.height > 150:
            continue
        # Point must be inside path bbox
        if not (r.x0 - 2 <= cx <= r.x1 + 2 and r.y0 - 2 <= cy <= r.y1 + 2):
            continue
        shape = _classify_path_shape(p)
        if shape == "none":
            continue
        area = r.width * r.height
        if area < best_area:
            best_area  = area
            best_shape = shape

    return best_shape


def _build_line_tokens(words: list, y_tol: float = 4.0) -> list[dict]:
    """
    Group raw PDF text words that share the same Y-band into logical text
    lines, then join each group into a single string.

    Returns a list of:
      {"text": "PL-E some description", "cx": <mean_x>, "cy": <mean_y>,
       "x0": min_x, "x1": max_x, "y0": min_y, "y1": max_y,
       "word_boxes": [(x0,y0,x1,y1,text), ...]}
    """
    if not words:
        return []

    # Sort by y0 then x0
    sorted_words = sorted(words, key=lambda w: (w[1], w[0]))

    lines: list[list] = []
    cur_line: list = []
    cur_y: float = None

    for w in sorted_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        if cur_y is None or abs(y0 - cur_y) > y_tol:
            if cur_line:
                lines.append(cur_line)
            cur_line = [w]
            cur_y    = y0
        else:
            cur_line.append(w)

    if cur_line:
        lines.append(cur_line)

    result = []
    for line in lines:
        # Sort left→right within the line
        line.sort(key=lambda w: w[0])
        all_x0 = [w[0] for w in line]
        all_y0 = [w[1] for w in line]
        all_x1 = [w[2] for w in line]
        all_y1 = [w[3] for w in line]
        texts  = [w[4] for w in line]
        joined = " ".join(texts)
        result.append({
            "text":      joined,
            "cx":        (min(all_x0) + max(all_x1)) / 2,
            "cy":        (min(all_y0) + max(all_y1)) / 2,
            "x0":        min(all_x0),
            "x1":        max(all_x1),
            "y0":        min(all_y0),
            "y1":        max(all_y1),
            "word_boxes": line,
        })
    return result


def _build_adjacent_tokens(words: list, max_x_gap: float = 30.0,
                           y_tol: float = 4.0) -> list[dict]:
    """
    Build additional candidate tokens by joining adjacent word pairs and
    triples that are on the same Y-band and horizontally close.
    This catches codes like "PL" + "-E" that PDF extraction splits.

    Returns same dict format as _build_line_tokens but for 2- and 3-word
    combinations only.
    """
    if not words:
        return []

    # Group words by Y band — use integer bucket key for reliable grouping
    y_groups: dict[int, list] = defaultdict(list)
    y_tol_int = max(1, int(y_tol))
    for w in words:
        y_key = int(round(w[1] / y_tol_int)) * y_tol_int
        y_groups[y_key].append(w)

    results = []
    for y_key, group in y_groups.items():
        group.sort(key=lambda w: w[0])
        n = len(group)
        for i in range(n):
            # Pairs
            if i + 1 < n:
                w1, w2 = group[i], group[i + 1]
                gap = w2[0] - w1[2]   # x0 of next - x1 of current
                if gap <= max_x_gap:
                    joined = w1[4] + w2[4]   # no space — handles "PL" + "-E"
                    joined_sp = w1[4] + " " + w2[4]
                    cx = (w1[0] + w2[2]) / 2
                    cy = (w1[1] + w2[3]) / 2
                    for txt in (joined, joined_sp):
                        results.append({
                            "text": txt, "cx": cx, "cy": cy,
                            "x0": w1[0], "x1": w2[2],
                            "y0": min(w1[1], w2[1]), "y1": max(w1[3], w2[3]),
                            "word_boxes": [w1, w2],
                        })
            # Triples
            if i + 2 < n:
                w1, w2, w3 = group[i], group[i + 1], group[i + 2]
                gap12 = w2[0] - w1[2]
                gap23 = w3[0] - w2[2]
                if gap12 <= max_x_gap and gap23 <= max_x_gap:
                    joined = w1[4] + w2[4] + w3[4]
                    cx     = (w1[0] + w3[2]) / 2
                    cy     = (w1[1] + w3[3]) / 2
                    results.append({
                        "text": joined, "cx": cx, "cy": cy,
                        "x0": w1[0], "x1": w3[2],
                        "y0": min(w1[1], w3[1]), "y1": max(w1[3], w3[3]),
                        "word_boxes": [w1, w2, w3],
                    })
    return results


def _code_in_text(code: str, text: str) -> bool:
    """
    Returns True if `code` appears in `text` as a distinct "word".
    Uses negative lookbehinds/lookaheads so 'H-1' doesn't match inside 'H-10'.
    """
    escaped = re.escape(code)
    # Require no word-character (A-Z0-9) immediately before or after
    pattern = r'(?<![A-Z0-9])' + escaped + r'(?![A-Z0-9])'
    return bool(re.search(pattern, text))


def _match_code_in_token(token_text: str, code_set: set) -> list[str]:
    """
    Return all codes from code_set that appear in token_text.

    Uses multiple sub-strategies with strict boundary guards to avoid
    false positives (e.g. 'H-1' matching inside 'H-10'):

      1. CODE_RE_LOOSE boundary-aware scan — finds any AB-CD pattern that is
         NOT immediately surrounded by other alphanumeric/hyphen chars.
      2. Whole-token boundary check — uses _code_in_text() which applies a
         regex with negative look-ahead/behind for word-char boundaries.
      3. Stripped comparison — strips punctuation from both sides and requires
         an EXACT match (not just substring) for the stripped forms.  This
         handles OCR noise like 'PL—E' → 'PLE' matching 'PL-E' → 'PLE',
         but only when the stripped token IS the stripped code, preventing
         'PLE' from matching inside 'MAPLE'.
    """
    matches = set()
    upper_token = token_text.upper()
    stripped_token = _strip_punct(upper_token)

    # Strategy 1: boundary-aware regex scan on the token text
    for m in CODE_RE_LOOSE.finditer(upper_token):
        candidate = m.group(1)
        if candidate in code_set:
            matches.add(candidate)

    # Strategies 2 & 3: per-code checks
    for code in code_set:
        # Strategy 2: whole-token boundary check (prevents 'H-1' matching 'H-10')
        if _code_in_text(code, upper_token):
            matches.add(code)
            continue
        # Strategy 3: stripped exact-token or stripped substring with boundary
        # Only match if the stripped code equals the stripped token, or the
        # stripped token contains the stripped code as a whole "word" (no
        # adjacent alphanumeric chars).
        stripped_code = _strip_punct(code)
        if stripped_code and len(stripped_code) >= 3:  # ignore very short codes
            # Exact match of stripped forms (e.g. 'PLE' == 'PLE')
            if stripped_token == stripped_code:
                matches.add(code)
            else:
                # Boundary-aware stripped match
                sc_escaped = re.escape(stripped_code)
                sc_pattern = r'(?<![A-Z0-9])' + sc_escaped + r'(?![A-Z0-9])'
                if re.search(sc_pattern, stripped_token):
                    matches.add(code)

    return list(matches)


def verify_codes_in_drawing(page, code_shape_map: dict,
                             table_x0: float, table_x1: float,
                             table_y0: float, table_y1: float) -> dict:
    """
    Shape-first verification of extracted codes against the drawing area.

    Strategy
    --------
    For codes with a known shape (octagon / rectangle / circle):
      • Walk every vector path on the page (outside the table zone).
      • Classify each path's shape using _classify_path_shape.
      • Collect all text words whose centre falls inside the path's bbox.
      • If the reconstructed text contains an extracted code AND the path's
        shape matches that code's expected shape → mark as present.

    For codes with shape='none' (plain text tables without shape markers):
      • Fall back to multi-strategy text scanning (single-word, adjacent-
        word pairs, full-line tokens) just as before.

    Parameters
    ----------
    page            : fitz.Page object
    code_shape_map  : {code_str: expected_shape_str}  e.g. {'PL-C': 'octagon'}
    table_x0/x1/y0/y1 : bounding box of the extracted table on this page

    Returns
    -------
    dict keyed by code:
      {
        "PL-C": {
          "present_in_drawing": True,
          "drawing_occurrences": [{"x": 287, "y": 553, "shape": "octagon"}, ...]
        }, ...
      }
    """
    if not code_shape_map:
        return {}, []

    TOL = 2.0
    paths     = page.get_drawings()
    raw_words = page.get_text("words")   # (x0, y0, x1, y1, text, ...)

    # ── Pre-filter: exclude table zone words ─────────────────────────────
    drawing_words = []
    for w in raw_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        word_in_table = (
            x0 <= table_x1 + TOL and x1 >= table_x0 - TOL and
            y0 <= table_y1 + TOL and y1 >= table_y0 - TOL
        )
        if not word_in_table:
            drawing_words.append((x0, y0, x1, y1, text))

    results: dict[str, dict] = {
        code: {"present_in_drawing": False, "drawing_occurrences": []}
        for code in code_shape_map if code
    }
    extra_codes: list[dict] = []
    seen: set[tuple] = set()

    # Partition codes by whether they have a shape marker or not
    shaped_codes   = {c: s for c, s in code_shape_map.items()
                      if s in ("octagon", "rectangle", "circle")}
    shapeless_codes = {c for c, s in code_shape_map.items()
                       if s not in ("octagon", "rectangle", "circle")}

    # ════════════════════════════════════════════════════════════════════
    # SHAPE-FIRST SCAN  (primary strategy for shaped codes)
    # Walk every path; for each recognised shape outside the table, read
    # the text inside and check against shaped_codes.
    # ════════════════════════════════════════════════════════════════════
    for p in paths:
        r = p.get("rect")
        if r is None or r.height < 3 or r.width < 3:
            continue
            
        # Code markers are small tags. Prevent large architectural features
        # (like countertops or wall panels) drawn as rectangles from being 
        # mistaken for code markers.
        if r.width > 150 or r.height > 150:
            continue

        # Skip paths that are fully inside the table zone
        if (r.x0 >= table_x0 - TOL and r.x1 <= table_x1 + TOL and
                r.y0 >= table_y0 - TOL and r.y1 <= table_y1 + TOL):
            continue

        # Skip paths that partially overlap the table zone
        # (avoids counting code-label shapes that straddle the table edge)
        overlaps_table = not (
            r.x1 <= table_x0 - TOL or r.x0 >= table_x1 + TOL or
            r.y1 <= table_y0 - TOL or r.y0 >= table_y1 + TOL
        )
        if overlaps_table:
            continue

        shape = _classify_path_shape(p)
        if shape == "none":
            continue

        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2

        # Collect words whose centre falls inside this shape's bbox
        WORD_TOL = 5.0
        words_inside = [
            (wx0, wtext)
            for wx0, wy0, wx1, wy1, wtext in drawing_words
            if (r.x0 - WORD_TOL <= (wx0 + wx1) / 2 <= r.x1 + WORD_TOL and
                r.y0 - WORD_TOL <= (wy0 + wy1) / 2 <= r.y1 + WORD_TOL)
        ]
        if not words_inside:
            continue

        words_inside.sort(key=lambda w: w[0])
        text_inside = " ".join(w[1] for w in words_inside)

        # Match against expected shaped codes
        upper_text = text_inside.upper()
        upper_text_no_spaces = upper_text.replace(" ", "")
        
        matched_any = False
        for expected_code, expected_shape in shaped_codes.items():
            if expected_shape != shape:
                continue   # shape type mismatch — skip

            # Match either the exact word-bounded code, or the raw space-stripped string.
            if _code_in_text(expected_code, upper_text) or expected_code in upper_text_no_spaces:
                matched_any = True
                dedup_key = (expected_code, int(round(cx / 8)), int(round(cy / 8)))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                hit = {"x": round(cx), "y": round(cy), "shape": shape}
                results[expected_code]["drawing_occurrences"].append(hit)
                results[expected_code]["present_in_drawing"] = True

                print(f"    [VERIFY] Found '{expected_code}' at ({round(cx)},{round(cy)})  "
                      f"shape={shape}  text={text_inside[:30]!r}")

        # If it didn't match any scheduled code, check if it's an unscheduled code!
        if not matched_any:
            possible_codes = set()
            for m in CODE_RE_LOOSE.finditer(upper_text):
                possible_codes.add(m.group(1))
                
            # Only use space-removed text if no valid format was found in natural text
            # This prevents adjacent unrelated text (like '6.11A') from merging to form 'PL-C611A'
            if not possible_codes:
                for m in CODE_RE_LOOSE.finditer(upper_text_no_spaces):
                    possible_codes.add(m.group(1))
            
            for pc in possible_codes:
                # To prevent OCR noise (like H-1.2.), strip punctuation first
                clean_pc = _strip_punct(pc)
                if len(clean_pc) >= 3 and clean_pc not in code_shape_map:
                    dedup_key = ("__EXTRA__", clean_pc, int(round(cx / 8)), int(round(cy / 8)))
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        extra_codes.append({
                            "code": clean_pc,
                            "x": round(cx),
                            "y": round(cy),
                            "shape": shape
                        })
                        print(f"    [VERIFY] WARNING: Found unscheduled code '{clean_pc}' at ({round(cx)},{round(cy)}) shape={shape}")

    # ════════════════════════════════════════════════════════════════════
    # TEXT-ONLY FALLBACK  (for shapeless codes OR codes we missed the shape for)
    # ════════════════════════════════════════════════════════════════════
    # Gather ALL codes that haven't been found yet, plus any shapeless codes
    missing_codes = {c.upper() for c, d in results.items() if not d["present_in_drawing"]}
    for c in shapeless_codes:
        missing_codes.add(c.upper())
        
    if missing_codes:
        sl_code_set = missing_codes

        single_tokens = [
            {"text": w[4], "cx": (w[0]+w[2])/2, "cy": (w[1]+w[3])/2,
             "x0": w[0], "x1": w[2], "y0": w[1], "y1": w[3],
             "word_boxes": [w]}
            for w in drawing_words
        ]
        adjacent_tokens = _build_adjacent_tokens(drawing_words,
                                                  max_x_gap=25.0, y_tol=5.0)
        line_tokens     = _build_line_tokens(drawing_words, y_tol=5.0)

        for tok in single_tokens + adjacent_tokens + line_tokens:
            matched = _match_code_in_token(tok["text"], sl_code_set)
            if not matched:
                continue

            for code in matched:
                if code not in results:
                    continue

                # Resolve word-level position for multi-word tokens
                word_boxes = tok.get("word_boxes", [])
                cx, cy = tok["cx"], tok["cy"]
                if len(word_boxes) > 1:
                    best_wx, best_wy = None, None
                    for wb in word_boxes:
                        if _code_in_text(code, wb[4].upper()) or code[:3] in wb[4].upper():
                            best_wx = (wb[0] + wb[2]) / 2
                            best_wy = (wb[1] + wb[3]) / 2
                            break
                    if best_wx is None and len(word_boxes) >= 2:
                        for i in range(len(word_boxes) - 1):
                            joined = word_boxes[i][4] + word_boxes[i+1][4]
                            if _code_in_text(code, joined.upper()):
                                best_wx = (word_boxes[i][0] + word_boxes[i+1][2]) / 2
                                best_wy = (word_boxes[i][1] + word_boxes[i+1][3]) / 2
                                break
                    if best_wx is not None:
                        cx, cy = best_wx, best_wy

                dedup_key = (code, int(round(cx/8)), int(round(cy/8)))
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                enc_shape = _find_enclosing_shape_at(paths, cx, cy)
                hit = {"x": round(cx), "y": round(cy), "shape": enc_shape}
                results[code]["drawing_occurrences"].append(hit)
                results[code]["present_in_drawing"] = True

                print(f"    [VERIFY] Found '{code}' at ({round(cx)},{round(cy)})  "
                      f"shape={enc_shape}  via token={tok['text'][:30]!r}")

    return results, extra_codes


# ── Debug visualiser ───────────────────────────────────────────────────────

def save_pdf_vector_debug(page, TX0, TX1, BORDER_YS, col_xs,
                          table_words, Y_BOT, shape_map,
                          verification_results, output_path, extra_codes=None):
    """
    Debug image showing table region, row/column boundaries, word boxes,
    table code shapes, and drawing-area occurrences of each code.

    Colour key
    ----------
    Red          – table bounding box
    Green        – row boundaries
    Blue         – column boundaries
    Cyan         – word bounding boxes
    Magenta      – octagon annotation in table
    Yellow       – rectangle annotation in table
    Orange       – circle annotation in table
    Lime green   – code found in drawing area  (filled dot + label)
    Red dot      – code NOT found in drawing area (dot in table margin)
    Orange dot   – unscheduled code found in drawing area
    """
    import fitz

    SCALE = 3
    pix   = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
    img   = np.frombuffer(pix.samples, dtype=np.uint8)

    if pix.alpha:
        img = img.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = img.reshape(pix.height, pix.width, pix.n).copy()
        if pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    sx, sy = SCALE, SCALE
    y_top  = BORDER_YS[0] if BORDER_YS else 0

    # 1. Annotate Missed Codes IN THE TABLE (Red Box & Arrow from Left)
    missing_text_y = int(y_top * sy) + 40
    for code, info in verification_results.items():
        if not info.get("present_in_drawing"):
            found_box = False
            # Find this code in the table_words to highlight it
            for wx0, wy0, wx1, wy1, wtext in table_words:
                if code in wtext.upper() or _strip_punct(code) in _strip_punct(wtext.upper()):
                    # Draw a red bounding box over the word in the table
                    box_x0 = int(wx0 * sx) - 2
                    box_y0 = int(wy0 * sy) - 2
                    box_x1 = int(wx1 * sx) + 2
                    box_y1 = int(wy1 * sy) + 2
                    cv2.rectangle(img, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 255), 3)

                    # Text on the far left
                    text_str = f"MISSING: {code}"
                    text_x = max(10, int(TX0 * sx) - 400)
                    (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Draw a solid white background box so text is always readable
                    cv2.rectangle(img, (text_x - 5, missing_text_y - th - 5), (text_x + tw + 5, missing_text_y + 5), (255, 255, 255), -1)
                    
                    cv2.putText(img, text_str, (text_x, missing_text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Draw elbow arrow from text to box
                    (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    start_pt = (text_x + tw + 10, missing_text_y - th // 2)
                    end_pt = (box_x0, (box_y0 + box_y1) // 2)
                    
                    mid_x = start_pt[0] + min(60, max(10, (end_pt[0] - start_pt[0]) // 3))
                    cv2.line(img, start_pt, (mid_x, start_pt[1]), (0, 0, 255), 2)
                    cv2.arrowedLine(img, (mid_x, start_pt[1]), end_pt, (0, 0, 255), 2, tipLength=0.03)

                    missing_text_y += 45
                    found_box = True
                    break
            
            if not found_box:
                text_str = f"MISSING: {code}"
                text_x = max(10, int(TX0 * sx) - 400)
                cv2.putText(img, text_str, (text_x, missing_text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                missing_text_y += 45

    # 2. Annotate Unscheduled Extra Codes in DRAWING (Orange Highlight)
    if extra_codes:
        for occ in extra_codes:
            px = int(occ["x"] * sx)
            py = int(occ["y"] * sy)
            # Draw a hollow orange box marker at the location so it doesn't hide the text
            cv2.rectangle(img, (px - 20, py - 15), (px + 20, py + 15), (0, 140, 255), 3)
            
            # Text on the far left (stacking exactly aligned with missing codes)
            text_str = f"UNSCHEDULED: {occ['code']} ({occ['shape']})"
            text_x = max(10, int(TX0 * sx) - 400)
            
            # Draw elbow arrow from text to the dot in the drawing
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw a solid white background box so text is always readable
            cv2.rectangle(img, (text_x - 5, missing_text_y - th - 5), (text_x + tw + 5, missing_text_y + 5), (255, 255, 255), -1)
            
            cv2.putText(img, text_str, (text_x, missing_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            
            if px < text_x:
                # Dot is to the left of the text (arrow shoots left)
                start_pt = (text_x - 10, missing_text_y - th // 2)
                end_pt = (px + 14, py)
                mid_x = start_pt[0] - min(60, max(10, (start_pt[0] - end_pt[0]) // 3))
            else:
                # Dot is to the right of the text (arrow shoots right)
                start_pt = (text_x + tw + 10, missing_text_y - th // 2)
                end_pt = (px - 14, py)
                mid_x = start_pt[0] + min(60, max(10, (end_pt[0] - start_pt[0]) // 3))
            
            cv2.line(img, start_pt, (mid_x, start_pt[1]), (0, 140, 255), 2)
            cv2.arrowedLine(img, (mid_x, start_pt[1]), end_pt, (0, 140, 255), 2, tipLength=0.03)

            missing_text_y += 45

    cv2.imwrite(output_path, img)
    print(f"  [PDF DEBUG] Saved → {output_path}")


# ── Main vector extractor ──────────────────────────────────────────────────

def extract_pdf_vector(pdf_path: str, page_num: int = 0,
                       debug_path: str = None) -> dict | None:
    """
    Try to extract the schedule table from PDF vector data.
    Returns structured dict or None if the PDF has no useful vector lines.
    """
    try:
        import fitz
    except ImportError:
        return None

    doc   = fitz.open(pdf_path)
    page  = doc[page_num]
    paths = page.get_drawings()

    # ── A1. Collect horizontal line segments ──────────────────────────────
    all_h = []
    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.height < 2 and r.width > 20:
            all_h.append((round(r.x0, 0), round(r.x1, 0), round(r.y0, 2)))

    if not all_h:
        return None

    # ── A2. Identify schedule table span ─────────────────────────────────
    span_groups = defaultdict(list)
    for x0, x1, y in all_h:
        rx0 = 10 * round(x0 / 10)
        rx1 = 10 * round(x1 / 10)
        span_groups[(rx0, rx1)].append((x0, x1, y))

    words = page.get_text("words")

    TABLE_SPAN = None
    best_score = float('-inf')

    for (rx0, rx1), lines in span_groups.items():
        if len(lines) >= 1 and (rx1 - rx0) > 50:
            x0  = sum(l[0] for l in lines) / len(lines)
            x1  = sum(l[1] for l in lines) / len(lines)
            ys  = sorted(l[2] for l in lines)
            w   = x1 - x0
            y_top = ys[0] - 60
            y_bot = ys[0] + 60

            score = 0
            for wx0, wy0, wx1, wy1, text, *_ in words:
                if x0 - 15 <= wx0 <= x1 + 15 and y_top <= wy0 <= y_bot:
                    t = text.upper()
                    if "MATERIALS" in t or "HARDWARE" in t or "DESCRIPTION" in t:
                        score += 5
                    elif "CODE" in t or "SHAPE" in t or "QTY" in t:
                        score += 1

            total_score = (score * 10000) - w
            if score > 0 and total_score > best_score:
                best_score = total_score
                TABLE_SPAN = (x0, x1, ys)

    if TABLE_SPAN is None:
        return None

    TX0, TX1 = TABLE_SPAN[0], TABLE_SPAN[1]

    table_ys = []
    for x0, x1, y in all_h:
        if x0 <= TX0 + 10 and x1 >= TX1 - 10:
            table_ys.append(y)
    table_ys = sorted(set(table_ys))

    BORDER_YS = [table_ys[0]] if table_ys else []
    for y in table_ys[1:]:
        if y - BORDER_YS[-1] <= 45:
            BORDER_YS.append(y)
        else:
            break

    # A valid schedule table must have at least 2 horizontal lines (e.g., top and bottom bounds)
    if len(BORDER_YS) < 2:
        return None

    print(f"  [PDF] Table span x={TX0:.0f}→{TX1:.0f}  "
          f"border_ys={[round(y, 1) for y in BORDER_YS]}")

    # ── A3. Column boundaries from V-lines ────────────────────────────────
    all_v_x    = []
    max_table_y = BORDER_YS[-1] if BORDER_YS else 0
    v_groups   = defaultdict(list)

    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.width < 2 and r.height > 5:
            vx = round(r.x0, 0)
            if TX0 - 2 <= vx <= TX1 + 2:
                all_v_x.append(vx)
                if TX0 + 5 < vx < TX1 - 5:
                    v_groups[10 * round(vx / 10)].append((r.y0, r.y1))

    for vx, v_lines in v_groups.items():
        v_lines.sort()
        merged = []
        for y0, y1 in v_lines:
            if not merged:
                merged.append([y0, y1])
            else:
                last_y0, last_y1 = merged[-1]
                if y0 <= last_y1 + 10:
                    merged[-1][1] = max(last_y1, y1)
                else:
                    merged.append([y0, y1])
        for my0, my1 in merged:
            if BORDER_YS and my0 <= BORDER_YS[-1] + 30:
                if my1 > max_table_y:
                    max_table_y = my1

    raw_vx = sorted(set(all_v_x))
    col_xs = [raw_vx[0]] if raw_vx else [TX0]
    for x in raw_vx[1:]:
        if x - col_xs[-1] > 5:
            col_xs.append(x)
    if col_xs[-1] < TX1 - 5:
        col_xs.append(TX1)

    print(f"  [PDF] Column x-positions: {col_xs}")

    # ── A3b. Build code-column shape map ──────────────────────────────────
    CODE_COL_X0 = col_xs[0] if len(col_xs) >= 1 else TX0
    CODE_COL_X1 = TX0 + 50

    Y_TOP_TABLE = BORDER_YS[0] - 3 if BORDER_YS else 0
    Y_BOT_TABLE = max((BORDER_YS[-1] + 25) if BORDER_YS else 0,
                      max_table_y + 10)

    shape_map = _build_code_shape_map(
        paths,
        code_col_x0=CODE_COL_X0,
        code_col_x1=CODE_COL_X1,
        table_y0=Y_TOP_TABLE,
        table_y1=Y_BOT_TABLE,
    )

    print(f"  [PDF] Code-column shapes: "
          f"{sorted((v, k) for k, v in shape_map.items())}")

    # ── A4. Grab text words in table region ───────────────────────────────
    words = page.get_text("words")
    table_words = [
        (x0, y0, x1, y1, text)
        for x0, y0, x1, y1, text, *_ in words
        if TX0 - 5 <= x0 <= TX1 + 10 and Y_TOP_TABLE <= y0 <= Y_BOT_TABLE
    ]

    if not table_words:
        return None

    # ── A5. Cluster word Y-positions → text rows ──────────────────────────
    raw_ys      = [w[1] for w in table_words]
    row_line_ys = _merge_vals(raw_ys, tol=3)

    # ── A6. Assign words to (row_y, column) ──────────────────────────────
    lines_by_y = defaultdict(list)
    for x0, y0, x1, y1, text in table_words:
        ry = min(row_line_ys, key=lambda ly: abs(ly - y0))
        lines_by_y[ry].append((x0, text))

    structured_rows = []
    for ry in sorted(lines_by_y.keys()):
        words_sorted = sorted(lines_by_y[ry], key=lambda w: w[0])
        col_words    = defaultdict(list)
        for wx, wt in words_sorted:
            col_idx = len(col_xs) - 2
            for j in range(len(col_xs) - 1):
                if col_xs[j] - 5 <= wx < col_xs[j + 1] + 5:
                    col_idx = j
                    break
            col_words[col_idx].append(wt)
        cells = [" ".join(col_words.get(j, [])) for j in range(len(col_xs) - 1)]
        shape = _lookup_shape(shape_map, ry, tol=8.0)
        structured_rows.append({"y": ry, "cells": cells, "shape": shape})

    # ── A7. Column role assignment ────────────────────────────────────────
    def _row_to_record(cells):
        n = len(cells)
        if n == 0:   return "", "", ""
        if n == 1:   return cells[0], "", ""
        if n == 2:
            # cells[0] may be "H-1.2 SALICE" — split code from leftover
            raw0 = cells[0].strip()
            m0   = CODE_RE.search(raw0.upper())
            if m0:
                leftover0 = (raw0[:m0.start()] + " " + raw0[m0.end():]).strip()
                desc = (leftover0 + " " + cells[1]).strip() if leftover0 else cells[1].strip()
                return raw0[m0.start():m0.end()], "", desc
            return cells[0], "", cells[1]

        cells = list(cells)   # make mutable copy
        code  = cells[0].strip()
        desc_prefix = ""    # text from cells[0] beyond the code token

        # ── Fix A: code may land in col 1-3 due to narrow phantom columns ──
        # (octagon shape edges create extra hairline vertical columns)
        if not code:
            for j in range(1, min(4, n)):
                cand = cells[j].strip()
                if cand and CODE_RE.search(cand.upper()):
                    code     = cand
                    cells[j] = ""
                    break
        else:
            # ── Fix B: cells[0] may contain "H-1.2 SALICE" — split it ──────
            # Extract just the code part; keep the rest as a desc prefix.
            m0 = CODE_RE.search(code.upper())
            if m0:
                before   = code[:m0.start()].strip()
                after    = code[m0.end():].strip()
                desc_prefix = (before + " " + after).strip()
                code     = code[m0.start():m0.end()]

        qty  = cells[1].strip()
        desc = " ".join(c for c in cells[2:] if c)

        # Merge desc_prefix into desc
        if desc_prefix:
            desc = (desc_prefix + " " + desc).strip()

        # If qty isn't a real quantity token, fold it into description
        if qty and not re.match(r'^[XxNnAaDd0-9/]+$', qty):
            desc = qty + (" " + desc if desc else "")
            qty  = ""
        return code, qty, desc

    # ── A8. Build sections ────────────────────────────────────────────────
    sections = []
    cur_hdr  = None
    cur_rows = []

    for row in structured_rows:
        cells    = row["cells"]
        combined = " ".join(cells)
        shape    = row["shape"]

        if ANCHOR_RE.search(combined):
            if cur_hdr is not None and cur_rows:
                sections.append({"section_header": cur_hdr, "rows": cur_rows})
            cur_hdr  = combined.strip()
            cur_rows = []
            continue

        code, qty, desc = _row_to_record(cells)
        code = _clean_code(code)
        # Skip rows with no content at all
        if not any([code, qty, desc]):
            continue
        # Skip header / label rows whose code field does not look like a real
        # code (e.g. "TAG", "CODE", "SHAPE" from table header rows).
        # A real code must match CODE_RE (e.g. "H-1.2", "PL-C", "WD-A").
        if code and not CODE_RE.search(code):
            continue

        cur_rows.append({
            "code":        code,
            "shape":       shape,
            "qty":         qty.strip(),
            "description": desc.strip(),
            "raw_cells":   cells,
        })

    if cur_rows:
        sections.append({
            "section_header": cur_hdr or "GENERAL",
            "rows":           cur_rows,
        })

    if not sections:
        return None

    # ── A9. Drawing-presence verification ────────────────────────────────
    # Build a {code: shape} map so the verifier can use shape-first matching.
    code_shape_map = {
        r["code"]: r["shape"]
        for sec in sections
        for r in sec["rows"]
        if r["code"]
    }
    all_codes = list(code_shape_map.keys())
    print(f"  [PDF] Verifying {len(all_codes)} codes in drawing area …")

    verification, extra_codes = verify_codes_in_drawing(
        page,
        code_shape_map=code_shape_map,
        table_x0=TX0,
        table_x1=TX1,
        table_y0=Y_TOP_TABLE,
        table_y1=Y_BOT_TABLE,
    )

    # Merge verification results back into rows
    for sec in sections:
        for row in sec["rows"]:
            code = row["code"]
            info = verification.get(code, {})
            row["present_in_drawing"]  = info.get("present_in_drawing", False)
            row["drawing_occurrences"] = info.get("drawing_occurrences", [])

    found_count   = sum(1 for c, v in verification.items() if v["present_in_drawing"])
    missing_count = len(all_codes) - found_count
    print(f"  [PDF] Drawing verification: "
          f"{found_count} found, {missing_count} not found in drawing\n")

    if debug_path:
        save_pdf_vector_debug(
            page, TX0, TX1, BORDER_YS, col_xs,
            table_words, Y_BOT_TABLE, shape_map, verification, debug_path,
            extra_codes=extra_codes
        )

    return {"sections": sections, "method": "pdf_vector",
            "table_bbox": [TX0, Y_TOP_TABLE, TX1, Y_BOT_TABLE],
            "unscheduled_codes_in_drawing": extra_codes}



# ═══════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def extract(input_path: str, debug_path: str = None,
            page_num: int = 0) -> dict:
    ext    = os.path.splitext(input_path)[1].lower()
    result = None

    if ext == ".pdf":
        print(f"  Attempting PDF vector extraction on page {page_num + 1} …")
        result = extract_pdf_vector(input_path, page_num, debug_path)
        if result:
            print(f"  ✓ PDF vector extraction succeeded "
                  f"({sum(len(s['rows']) for s in result['sections'])} rows)")
        else:
            print("  ✗ PDF vector extraction failed\n")

    # Keyword gate
    if result and result.get("sections"):
        has_keywords = False
        for sec in result["sections"]:
            text = (sec["section_header"] + " "
                    + " ".join(r.get("description", "") + r.get("code", "")
                               for r in sec["rows"])).upper()
            if "MATERIALS" in text or "HARDWARE" in text:
                has_keywords = True
                break
        if not has_keywords:
            print("  [INFO] Table lacks MATERIALS/HARDWARE keywords. Discarding.\n")
            result["sections"] = []

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tablet Extractor")
    parser.add_argument("input_file", help="Path to input PDF or image")
    parser.add_argument("--page", "-p", type=int, default=1,
                        help="Page number (1-based) to process from PDF")
    parser.add_argument("--all", action="store_true",
                        help="Process all pages in the PDF")
    args = parser.parse_args()

    input_path = args.input_file
    output_dir = "table extraction output"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("  Tablet Extractor  (PDF-native Vector Extraction)")
    print(f"  Input : {os.path.basename(input_path)}")
    print("=" * 65 + "\n")

    pages_to_process = []
    if args.all and input_path.lower().endswith(".pdf"):
        import fitz
        try:
            doc = fitz.open(input_path)
            pages_to_process = list(range(doc.page_count))
            doc.close()
        except Exception as e:
            print(f"Error reading PDF for page count: {e}")
            return
    else:
        pages_to_process = [max(0, args.page - 1)]

    for page_num in pages_to_process:
        if args.all:
            print(f"--- Processing Page {page_num + 1} ---")

        debug_out        = os.path.join(output_dir,
                                        f"{page_num + 1}_ocr_debug.png")
        vector_debug_out = os.path.join(output_dir,
                                        f"{page_num + 1}_vector_debug.png")
        json_out         = os.path.join(output_dir,
                                        f"{page_num + 1}_results.json")

        result = extract(
            input_path,
            debug_path=(vector_debug_out
                        if input_path.lower().endswith(".pdf")
                        else debug_out),
            page_num=page_num,
        )

        if not result or not result.get("sections"):
            print(f"  [Page {page_num + 1}] No table found — "
                  "drawing presence check skipped.\n")
            continue

        output = {
            "input_file":     input_path,
            "method":         result.get("method", "unknown"),
            "total_sections": len(result["sections"]),
            "total_rows":     sum(len(s["rows"]) for s in result["sections"]),
            "sections":       result["sections"],
        }

        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        # ── Console report ────────────────────────────────────────────────
        print("=" * 65)
        print(f"  EXTRACTED TABLE  (Page {page_num + 1})")
        print("=" * 65)

        for sec in result["sections"]:
            print(f"\n  ▶  {sec['section_header']}")
            hdr = (f"  {'CODE':<10}  {'SHAPE':<12}  {'QTY':<5}  "
                   f"{'IN DRAWING':<12}  {'# LOCS':<7}  DESCRIPTION")
            print(hdr)
            print("  " + "-" * 80)
            for r in sec["rows"]:
                in_dwg = "✓ YES" if r.get("present_in_drawing") else "✗ NO"
                n_locs = len(r.get("drawing_occurrences", []))
                print(f"  {r['code']:<10}  {r.get('shape','n/a'):<12}  "
                      f"{r['qty']:<5}  {in_dwg:<12}  {n_locs:<7}  "
                      f"{r['description'][:45]}")
                # Print individual occurrence locations
                for occ in r.get("drawing_occurrences", []):
                    print(f"             └─ drawing @ "
                          f"x={occ['x']}, y={occ['y']}  "
                          f"shape={occ['shape']}")

        # Summary counts
        all_rows = [r for sec in result["sections"] for r in sec["rows"]]
        found    = sum(1 for r in all_rows if r.get("present_in_drawing"))
        missing  = len(all_rows) - found

        print(f"\n  Method         : {output['method']}")
        print(f"  Total sections : {output['total_sections']}")
        print(f"  Total rows     : {output['total_rows']}")
        print(f"  In drawing     : {found}  |  Not in drawing: {missing}")
        print(f"  JSON saved     → {json_out}\n")


if __name__ == "__main__":
    main()