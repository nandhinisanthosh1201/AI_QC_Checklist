"""
tablet_extractor_paddle.py
==========================
Extracts the materials/hardware schedule table from architectural drawings,
then verifies which extracted codes are actually referenced in the drawing
area -- validated collectively per drawing GROUP, per DVCS.docx §11.

(Despite the filename, no OCR is used: everything is read from PDF vector
data via PyMuPDF.)

GROUPING (current behavior)
---------------------------
Groups are built strictly from PDF page order:
  * Page 1 always starts group 1.
  * A later page starts a NEW group if it carries a main PLAN or KEY PLAN
    view title (see `_is_group_start_view_title`). PLAN SECTION, REFLECTED
    CEILING PLAN, RCP, ROOF PLAN and CEILING PLAN titles do not start a group.
  * Every other page joins the current group.
Schedule-table similarity and cross-reference bubbles are deliberately NOT
used for grouping, so an identical table reprinted in two adjacent groups
does not merge them. Bubbles and table signatures are still computed per
sheet for diagnostics and the validation cache.

This replaces two earlier approaches that no longer exist in the code:
  * bubble-graph grouping (former fixes #2, #6, #6b), and
  * merging sheets with identical tables (former fix #9,
    `_merge_sheets_sharing_identical_tables` -- removed).
`_filter_template_locator_callouts` still runs but only cleans each
sheet's `cross_refs`; it has no effect on grouping.

FIXES STILL IN EFFECT
---------------------
  1. Group-level validation is the only path main() runs (§11). The old
     single-sheet validator remains in section L for reuse only.
  3. PASS/FAIL fails on schedule_omissions as well as orphaned codes (§14).
  4. Schedule-section headers are limited to MATERIALS / HARDWARE (§7.2);
     a lone "QTY." token no longer starts a section.
  5. Codes must match ALLOWED_CODE_RE (§7.3): H-<numeric> hardware codes
     and the MM/PL/WD/WV/PT/GL/SS/LAM material/finish prefixes.
  7. MOD number is read position-based (JOB #:/MOD # label block paired with
     the adjacent value block), not by same-block regex.
  8. View-title anchors come from span-level `find_scale_anchors` (accepts
     scale with or without the word SCALE, rejects dimension strings via
     `_is_real_scale`, falls back to a standalone ARCH REF span).

NOTE: earlier revisions claimed the groups match DVCS §6 GROUP_A/B/C and the
FAIL results match §13/§15 on the 26022 Docusign 15th Floor submittal. That
was verified against the OLD grouping logic and has NOT been re-checked
against page-order grouping.

FILE LAYOUT
-----------
  A. Module setup                  H. Table extraction
  B. Constants & regex patterns    I. Validation cache
  C. PDF geometry utilities        J. Drawing-area verification (group-level)
  D. Code-text utilities           K. Debug rendering (legacy path only)
  E. Title / sheet identification  L. Legacy per-sheet path (not called by main)
  E2. Detector-box utilities       M. CLI entry point
      (reference-only, unused)
  F. Grouping
  G. Table signatures
  N. Per-view section-reference validation (additive)
"""
# ===========================================================================
# A. MODULE SETUP
# ===========================================================================

import sys, os, io, re, json

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

try:
    import fitz  # PyMuPDF -- optional at import time, required for PDF mode
except ImportError:
    fitz = None


# ===========================================================================
# B. CONSTANTS & REGEX PATTERNS
# ===========================================================================

# Fix: per requirements §7.2, "Materials" and "Hardware" are the ONLY valid
# schedule-tag names (§7.1 folds Finish into the Materials/octagon bucket).
# A standalone "QTY." column-header token was previously included here too,
# which meant a header row containing only "QTY" (no MATERIALS/HARDWARE/
# FINISH alongside it) could spuriously kick off a brand-new MATERIALS
# section and silently drop whatever rows follow into the wrong bucket.
# QTY. is a real column label but is not a schedule-tag name in its own
# right, so it no longer participates in section-anchor detection.
ANCHOR_RE    = re.compile(r'\b(MATERIALS?|HARDWARE|HDWE|FINISH)\b', re.I)
CODE_RE      = re.compile(r'\b([A-Z]{1,3}-(?:[A-Z]{1,2}|[0-9][A-Z0-9.]*))')
CODE_RE_LOOSE = re.compile(r'(?<![A-Z0-9-])([A-Z]{1,3}-(?:[A-Z]{1,2}|[0-9][A-Z0-9.]*))(?![A-Z0-9-])')
PREFIX_FIXES = {"PI": "PL", "P1": "PL", "G1": "GL", "6L": "GL"}

# Fix: §7.3 states "no other code formats shall be accepted unless they
# conform to the defined project standard" and gives PL-E, H-1.2, H-1.3.1
# as the approved patterns -- but CODE_RE alone would happily accept an
# arbitrary "XY-Z" token as a schedule code with no allow-list check
# anywhere. This mirrors the material/finish prefixes already trusted for
# shape-enforcement (MM/PL/WD/WV/PT/GL/SS/LAM) plus the H-<numeric> hardware
# pattern, and is used to reject anything outside the approved format.
ALLOWED_CODE_RE = re.compile(
    r'^(?:H-\d+(?:\.\d+)*|(?:MM|PL|WD|WV|PT|GL|SS|LAM)-[A-Z0-9]+)$', re.I
)

SCALE_RE = re.compile(r"\bSCALE\s+(?P<scale>\d+\s*:\s*\d+(?:\s*\([^)]*\))?|NTS)", re.I)
ARCH_RE = re.compile(r"\bARCH\s*REF\s*:\s*(?P<ref>.+)", re.I)
VIEW_ID_RE = re.compile(r"^(?:\d+|[A-Z]{1,3}\d*)\s+")
STOP_PREFIXES = ("ARCH REF", "SCALE", "MATL", "MATERIAL", "FINISH", "HDWE", "HARDWARE", "NOTES", "ISSUE", "JOB", "CRESTMARK")

KEY_PLAN_CALLOUT_IDS = {"KP"}


# ===========================================================================
# C. PDF GEOMETRY UTILITIES
# ===========================================================================

def _transform_bbox(x0, y0, x1, y1, matrix):
    p1 = fitz.Point(x0, y0) * matrix
    p2 = fitz.Point(x1, y1) * matrix
    nx0, nx1 = (p1.x, p2.x) if p1.x <= p2.x else (p2.x, p1.x)
    ny0, ny1 = (p1.y, p2.y) if p1.y <= p2.y else (p2.y, p1.y)
    return nx0, ny0, nx1, ny1


def get_display_words(page):
    raw = page.get_text("words")
    if page.rotation == 0:
        return raw
    m = page.rotation_matrix
    out = []
    for w in raw:
        x0, y0, x1, y1 = _transform_bbox(w[0], w[1], w[2], w[3], m)
        out.append((x0, y0, x1, y1) + tuple(w[4:]))
    return out


def get_display_drawings(page):
    raw = page.get_drawings()
    if page.rotation == 0:
        return raw
    m = page.rotation_matrix
    out = []
    for p in raw:
        r = p.get("rect")
        if r is not None:
            x0, y0, x1, y1 = _transform_bbox(r.x0, r.y0, r.x1, r.y1, m)
            p = dict(p)
            p["rect"] = fitz.Rect(x0, y0, x1, y1)
        out.append(p)
    return out


def _merge_vals(vals, tol=3):
    merged = []
    for v in sorted(vals):
        if merged and abs(v - merged[-1]) <= tol:
            merged[-1] = (merged[-1] + v) / 2
        else:
            merged.append(v)
    return merged


def _classify_path_shape(path) -> str:
    items = path.get('items', [])
    if not items:
        return "none"
    types = [it[0] for it in items]

    if len(items) == 1 and types[0] == 're':
        return "rectangle"

    if 4 <= len(items) <= 6:
        if types.count('m') == 1 and all(t in ('m', 'l', 'c') for t in types):
            r = path.get('rect')
            if r and r.height > 0:
                aspect = r.width / r.height
                if 0.1 <= aspect <= 10.0:
                    return "rectangle"

    if len(items) == 8 and all(t == 'l' for t in types):
        r = path.get('rect')
        if r and r.height > 0:
            aspect = r.width / r.height
            if 0.8 <= aspect <= 6.0:
                return "octagon"

    if len(items) == 4 and all(t == 'c' for t in types):
        return "circle"

    if len(items) == 1 and types[0] == 'c':
        return "circle"

    return "none"


def _build_code_shape_map(paths, code_col_x0: float, code_col_x1: float,
                          table_y0: float, table_y1: float) -> dict:
    shape_map = {}
    col_width   = code_col_x1 - code_col_x0
    max_shape_w = col_width * 3

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


def _lookup_shape(shape_map, row_y, tol=6.0):
    best_shape = "none"
    best_dist  = tol + 1
    for sy, shape in shape_map.items():
        dist = abs(sy - row_y)
        if dist < best_dist:
            best_dist  = dist
            best_shape = shape
    return best_shape


def _find_enclosing_shape_at(paths, cx, cy):
    best_shape = "none"
    best_area  = float('inf')

    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.height < 3 or r.width < 3:
            continue
        if r.width > 150 or r.height > 150:
            continue
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


# ===========================================================================
# D. CODE-TEXT UTILITIES
# ===========================================================================

def _clean_code(text):
    if not text:
        return ""
    t = text.upper().strip()
    t = re.sub(r'\s*-\s*', '-', t)
    t = re.sub(r'(\d),(\d)', r'\1.\2', t)
    t = re.sub(r'\b(PI|P1|G1|6L)-',
               lambda m: PREFIX_FIXES.get(m.group(1), m.group(1)) + "-", t)
    t = re.sub(r'[.,;:!?]+$', '', t)
    t_raw     = _restore_dots(t)
    t_compact = _restore_dots(re.sub(r'\s+', '', t))
    m_raw  = CODE_RE.search(t_raw)
    m_comp = CODE_RE.search(t_compact)
    if m_comp and (not m_raw or m_comp.group(1).count('.') > m_raw.group(1).count('.')):
        return m_comp.group(1)
    return m_raw.group(1) if m_raw else t.strip()


def _strip_punct(s: str) -> str:
    return re.sub(r'[^A-Z0-9\-]', '', s.upper())


def _code_in_text(code: str, text: str) -> bool:
    escaped = re.escape(code)
    pattern = r'(?<![A-Z0-9\-])' + escaped + r'(?![A-Z0-9\-\.])'
    return bool(re.search(pattern, text, re.IGNORECASE))

def _restore_dots(text: str) -> str:
    t = text.upper()
    prev = None
    while prev != t:
        prev = t
        t = re.sub(
            r'\b([A-Z]{1,3}-(?:[0-9][A-Z0-9.]*)?)([0-9])([0-9]+)(?![A-Z0-9.])',
            lambda m: f"{m.group(1)}{m.group(2)}.{m.group(3)}",
            t
        )
    return t


def _build_line_tokens(words, y_tol=4.0):
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w[1], w[0]))

    lines = []
    cur_line = []
    cur_y = None

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


def _build_adjacent_tokens(words, max_x_gap=30.0, y_tol=4.0):
    if not words:
        return []

    y_groups = defaultdict(list)
    y_tol_int = max(1, int(y_tol))
    for w in words:
        y_key = int(round(w[1] / y_tol_int)) * y_tol_int
        y_groups[y_key].append(w)

    results = []
    for y_key, group in y_groups.items():
        group.sort(key=lambda w: w[0])
        n = len(group)
        for i in range(n):
            if i + 1 < n:
                w1, w2 = group[i], group[i + 1]
                gap = w2[0] - w1[2]
                if gap <= max_x_gap:
                    joined = w1[4] + w2[4]
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


def _match_code_in_token(token_text, code_set):
    matches = set()
    upper_token = _restore_dots(token_text.upper())
    stripped_token = _strip_punct(upper_token)

    for m in CODE_RE_LOOSE.finditer(upper_token):
        candidate = m.group(1)
        if candidate in code_set:
            matches.add(candidate)

    for code in code_set:
        if _code_in_text(code, upper_token):
            matches.add(code)
            continue
        stripped_code = _strip_punct(code)
        if stripped_code and len(stripped_code) >= 3:
            if stripped_token == stripped_code:
                matches.add(code)
            else:
                sc_escaped = re.escape(stripped_code)
                sc_pattern = r'(?<![A-Z0-9])' + sc_escaped + r'(?![A-Z0-9])'
                if re.search(sc_pattern, stripped_token):
                    matches.add(code)

    return list(matches)


# ===========================================================================
# E. TITLE / SHEET IDENTIFICATION
# ===========================================================================

def _norm_str(s):
    return re.sub(r"\s+", " ", s).strip()

def clean_title(s):
    s = _norm_str(s)
    s = VIEW_ID_RE.sub("", s, count=1)
    s = re.sub(r"^[|:;,-]+\s*", "", s)
    return s.strip()

def extract_sheet_title_block(page, page_num):
    w, h = page.rect.width, page.rect.height
    blocks = page.get_text("blocks")

    br_candidates = []
    mod_str = None
    scope_lines = []

    # Locate the "JOB #:\nMOD #:" label block, then pair it with whichever
    # nearby block holds the two values ("26022\n2") to its right. This is
    # deliberately position-based rather than a same-block regex, since the
    # label and its value live in separate blocks in this title block layout.
    label_block = None
    for b in blocks:
        text_clean = _norm_str(b[4])
        if re.search(r'\bJOB\s*#\s*:.*\bMOD\s*#\s*:', text_clean, re.I) or \
           re.fullmatch(r'JOB\s*#\s*:\s*MOD\s*#\s*:', text_clean, re.I):
            label_block = b
            break

    if label_block is not None:
        lx0, ly0, lx1, ly1 = label_block[:4]
        best = None
        for b in blocks:
            bx0, by0, bx1, by1 = b[:4]
            if b is label_block:
                continue
            # value block sits immediately to the right, same row (boxes can
            # abut/overlap by a couple points, so anchor on left edges)
            if lx0 < bx0 < lx1 + 80 and abs(by0 - ly0) < 20:
                lines = [l.strip() for l in b[4].split('\n') if l.strip()]
                if len(lines) >= 2 and re.fullmatch(r'[A-Z0-9.\-]+', lines[1], re.I):
                    best = lines[1]
                    break
        if best:
            mod_str = f"MOD #{best.upper()}"

    for b in blocks:
        bx0, by0, bx1, by1, text = b[:5]
        text_clean = _norm_str(text)
        if not text_clean:
            continue

        if mod_str is None:
            m_mod = re.search(r'\bMOD\s*#?\s*[:\-]?\s*(\d[A-Z0-9.\-]*)\b', text_clean, re.I)
            if m_mod:
                mod_str = f"MOD #{m_mod.group(1).upper()}"

        if bx0 >= w * 0.7 and by0 >= h * 0.7:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for l in lines:
                if re.fullmatch(r'(?:[A-Z]{1,3}\d+(?:\.\d+)?|\d+\.\d+|[A-Z]\d+)', l, re.I):
                    if not any(k in l.upper() for k in ["REV", "MOD", "JOB", "ZIP", "DATE"]):
                        dist = (w - bx1)**2 + (h - by1)**2
                        br_candidates.append((dist, l.upper()))

        if by0 >= h * 0.6:
            for l in text.split('\n'):
                l_str = l.strip()
                if any(k in l_str.upper() for k in ["DESK", "PANELS", "CABINETS", "LOBBY", "BAR", "KITCHENETTE", "ROOM", "RECEPTION", "BREAKROOM"]):
                    if not any(k in l_str.upper() for k in ["CRESTMARK", "FOUNDATION", "DATABASE", "DWG", "PH (", "FAX"]):
                        scope_lines.append(l_str)

    br_candidates.sort(key=lambda x: x[0])
    sheet_id = br_candidates[0][1] if br_candidates else f"Page_{page_num}"
    scope = " — ".join(dict.fromkeys(scope_lines)) if scope_lines else f"Scope {sheet_id}"

    return {
        "sheet_id": sheet_id,
        "mod_number": mod_str or "MOD #1",
        "scope": scope,
        "page_num": page_num
    }


# ---------------------------------------------------------------------
# Span-level scale/ARCH-REF anchor detection.
#
# Replaces the previous approach of regex-matching whole PyMuPDF text
# BLOCKS for the literal word "SCALE". That approach missed two real
# cases on these submittals:
#   - a view whose scale is printed as a bare ratio/fraction with no
#     "SCALE" keyword at all (e.g. a lone "3/4"=1'-0"" or "1:16" sitting
#     under a title), and
#   - a view identified only by an "ARCH REF: n/An.n" callout with no
#     scale annotation nearby (e.g. KEY PLAN insets).
# `_is_real_scale` also guards against the reverse failure mode: a plain
# dimension string ("24'-6\"") that happens to start with digits must
# NOT be picked up as a scale value.
# ---------------------------------------------------------------------

_SCALE_VALUE_RE = re.compile(
    r'(\b\d+(\s+\d+/\d+|\s*/\s*\d+)?\s*["\']?\s*=\s*\d+[\'"]?|\b1\s*:\s*\d+'
    r'|\b(nts|n\.t\.s\.?|no\s*scale|as\s*noted)\b|\bscale\b)',
    re.IGNORECASE,
)


def _is_real_scale(text: str) -> bool:
    t = text.strip()
    if not t or len(t) >= 50:
        return False
    if re.match(r'^\d+[\'"]?\s*-\s*\d+', t):  # plain dimension string
        return False
    return bool(_SCALE_VALUE_RE.search(t))


def find_scale_anchors(page) -> list[dict]:
    """Scale/ARCH REF spans -> one anchor per drawing, in PDF points."""
    rot = page.rotation_matrix if page.rotation != 0 else None
    spans = []
    for b in page.get_text("dict").get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s["text"].strip()
                if not t:
                    continue
                r = fitz.Rect(s["bbox"])
                if rot:
                    r = r * rot
                spans.append({"text": t, "bbox": [r.x0, r.y0, r.x1, r.y1]})

    raw = []
    for s in spans:
        if _is_real_scale(s["text"]):
            raw.append(s)
        elif re.match(r'^arch\s*ref\b', s["text"], re.IGNORECASE):
            x0, y0 = s["bbox"][0], s["bbox"][1]
            if not any(
                _is_real_scale(o["text"])
                and abs(o["bbox"][0] - x0) < 50
                and 0 < (y0 - o["bbox"][3]) < 25
                for o in spans if o is not s
            ):
                raw.append(s)

    # dedupe: collapse spans within 40pt x / 14pt y of each other
    anchors = []
    for s in sorted(raw, key=lambda s: (s["bbox"][1], s["bbox"][0])):
        cx = (s["bbox"][0] + s["bbox"][2]) / 2.0
        cy = (s["bbox"][1] + s["bbox"][3]) / 2.0
        if any(abs(cx - a["cx"]) < 40 and abs(cy - a["cy"]) < 14 for a in anchors):
            continue
        anchors.append({"cx": cx, "cy": cy, "text": s["text"], "bbox": s["bbox"]})
    return anchors


def detect_view_titles_from_page(page, page_num):
    blocks = []
    for b in page.get_text("blocks"):
        t = _norm_str(b[4])
        if t:
            blocks.append({"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": t})

    # Fix: anchors now come from find_scale_anchors (span-level, see note
    # above) instead of "does this whole block contain the word SCALE".
    anchors = find_scale_anchors(page)

    results = []
    for anchor in anchors:
        acx, acy, atext = anchor["cx"], anchor["cy"], anchor["text"]

        # Find the text block containing this anchor span so the existing
        # prefix/remainder split and neighbor heuristic (unchanged below)
        # still have a block of surrounding text to work with.
        #
        # Fix: adjacent title-block cells can have bboxes that overlap the
        # same point (e.g. a "1  ARCH REF: 1/A2.1" label block whose bbox
        # spans the same y-range as a neighboring "TITLE ... SCALE ..."
        # block it sits beside) -- picking the FIRST bbox-containing block
        # in list order can grab the wrong one and silently drop a real
        # title (confirmed on a sheet with a title reading "PLAN - MDF
        # SILLS AT EXISTING STOREFRONT" that has this exact overlap).
        # Among all bbox-containing candidates, prefer the one whose own
        # joined text actually contains the anchor's text; only fall back
        # to bbox containment alone (smallest/tightest match) if none do.
        atext_norm = _norm_str(atext).upper()
        candidates_b = [
            ob for ob in blocks
            if ob["x0"] - 2 <= acx <= ob["x1"] + 2 and ob["y0"] - 2 <= acy <= ob["y1"] + 2
        ]
        b = next(
            (ob for ob in candidates_b if atext_norm in _norm_str(ob["text"]).upper()),
            None,
        )
        if b is None and candidates_b:
            b = min(candidates_b, key=lambda ob: (ob["x1"] - ob["x0"]) * (ob["y1"] - ob["y0"]))
        if b is None:
            # PyMuPDF didn't merge this span into a multi-line block (common
            # for an isolated ARCH REF/scale line) -- treat the span itself
            # as a single-line block so the rest of the logic is unchanged.
            b = {"x0": anchor["bbox"][0], "y0": anchor["bbox"][1],
                 "x1": anchor["bbox"][2], "y1": anchor["bbox"][3], "text": atext}

        # Re-locate the anchor text inside the block's own joined text so
        # the prefix/remainder split below still applies even though the
        # anchor was found at span granularity. SCALE_RE is tried first so
        # the emitted "scale" field keeps its previous format when the
        # block does contain a "SCALE ..." phrase; otherwise fall back to
        # locating the raw anchor text (or, failing that, treat the whole
        # block as the anchor span).
        m_scale_in_block = SCALE_RE.search(b["text"])
        if m_scale_in_block:
            span_start, span_end = m_scale_in_block.start(), m_scale_in_block.end()
            scale = m_scale_in_block.group("scale")
        else:
            idx = b["text"].upper().find(atext.upper())
            if idx != -1:
                span_start, span_end = idx, idx + len(atext)
            else:
                span_start, span_end = 0, len(b["text"])
            scale = atext

        remainder = _norm_str(b["text"][span_end:].strip(" -:|\t"))

        view_id = None
        title = None
        if remainder:
            if re.fullmatch(r"[A-Z0-9]{1,3}", remainder, re.I):
                view_id = remainder
            else:
                title = clean_title(remainder)
                if title and (len(title) <= 2 or title.upper().startswith(STOP_PREFIXES)):
                    title = None

        prefix = b["text"][:span_start].strip()
        m_id = re.match(r"^(\d+|[A-Z]{1,3})\s+", prefix)
        if m_id and not view_id:
            view_id = m_id.group(1)
            prefix = prefix[m_id.end():].strip()
        if prefix and not title:
            cand = clean_title(prefix)
            if cand and len(cand) > 2 and not cand.upper().startswith(STOP_PREFIXES):
                title = cand

        if not title:
            candidates = []
            for ob in blocks:
                if ob is b:
                    continue
                ot = ob["text"]
                if re.search(r"\b(SCALE|ARCH\s*REF)\b", ot, re.I):
                    continue
                if ot.upper().startswith(STOP_PREFIXES) or "CRESTMARK" in ot.upper() or ".DWG" in ot.upper():
                    continue

                overlap = max(0.0, min(b["x1"], ob["x1"]) - max(b["x0"], ob["x0"]))
                x_dist = max(0.0, ob["x0"] - b["x1"], b["x0"] - ob["x1"])
                dy = abs(ob["y0"] - b["y0"])
                dy_above = b["y0"] - ob["y1"]
                dy_below = ob["y0"] - b["y1"]
                is_kw = bool(re.search(r"\b(PLAN|ELEVATION|SECTION|DETAIL|REFLECTED|KEY PLAN)\b", ot, re.I))

                if dy <= 20 and (overlap > 0 or x_dist < 100):
                    score = (100 if is_kw else 50) - dy - x_dist * 0.2
                    candidates.append((score, ot, ob))
                elif -5 <= dy_above <= 40 and (overlap > 0 or x_dist < 80):
                    score = (120 if is_kw else 40) - abs(dy_above)
                    candidates.append((score, ot, ob))
                elif -5 <= dy_below <= 40 and (overlap > 0 or x_dist < 80):
                    score = (90 if is_kw else 30) - abs(dy_below)
                    candidates.append((score, ot, ob))

            if candidates:
                candidates.sort(key=lambda x: -x[0])
                best_title = clean_title(candidates[0][1])
                if best_title and len(best_title) > 2:
                    title = best_title

        arch_ref = None
        m_arch = ARCH_RE.search(b["text"])
        if m_arch:
            arch_ref = _norm_str(m_arch.group("ref"))
        else:
            arch_cands = []
            for ob in blocks:
                if "ARCH REF" in ob["text"].upper():
                    m_a = ARCH_RE.search(ob["text"])
                    if m_a:
                        dy = ob["y0"] - b["y1"]
                        overlap = max(0.0, min(b["x1"], ob["x1"]) - max(b["x0"], ob["x0"]))
                        x_dist = max(0.0, ob["x0"] - b["x1"], b["x0"] - ob["x1"])
                        if -10 <= dy <= 60 and (overlap > 0 or x_dist < 80):
                            arch_cands.append((dy, _norm_str(m_a.group("ref"))))
            if arch_cands:
                arch_cands.sort(key=lambda x: x[0])
                arch_ref = arch_cands[0][1]

        if title:
            results.append({
                "page": page_num,
                "view_id": view_id,
                "view_title": title,
                "arch_ref": arch_ref,
                "scale": scale,
                "bbox": [round(b["x0"], 1), round(b["y0"], 1), round(b["x1"], 1), round(b["y1"], 1)]
            })

    unique = []
    seen = set()
    for r in results:
        key = (r["view_title"].upper(), r["scale"], r["arch_ref"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique

def _detect_key_plan_bboxes(page):
    boxes = []
    for b in page.get_text("blocks"):
        t = _norm_str(b[4])
        if re.search(r'\bKEY\s*PLAN\b', t, re.I):
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            boxes.append((x0 - 40, y0 - 20, x1 + 250, y1 + 250))
    return boxes


def extract_sheet_bubbles(page, page_num, all_sheet_ids, own_sheet_id=None):
    key_plan_boxes = _detect_key_plan_bboxes(page)

    def _in_key_plan_zone(rect):
        if rect is None:
            return False
        rx0, ry0, rx1, ry1 = rect
        for kx0, ky0, kx1, ky1 in key_plan_boxes:
            if not (rx1 < kx0 or rx0 > kx1 or ry1 < ky0 or ry0 > ky1):
                return True
        return False

    text = page.get_text()
    refs = []
    seen = set()

    for m in re.finditer(r'\b([A-Z0-9]{1,3})\s*(?:/|->|→)\s*([A-Z0-9.]+)\b', text):
        callout_id, target = m.group(1).upper(), m.group(2).upper()
        if callout_id in KEY_PLAN_CALLOUT_IDS:
            continue
        matched_sh = None
        for sid in all_sheet_ids:
            if target == sid or target.startswith(sid) or sid.startswith(target):
                matched_sh = sid
                break
        if matched_sh:
            k = (callout_id, matched_sh)
            if k not in seen:
                seen.add(k)
                refs.append({"callout": callout_id, "target_sheet": matched_sh, "raw": m.group(0)})

    blocks = page.get_text("blocks")
    lines = []
    for b in blocks:
        for l in b[4].split('\n'):
            l_str = l.strip().upper()
            if l_str:
                lines.append((l_str, (b[0], b[1], b[2], b[3])))

    for i in range(len(lines) - 1):
        l1, box1 = lines[i]
        l2, box2 = lines[i + 1]
        if re.fullmatch(r'[A-Z0-9]{1,3}', l1) and l2 in all_sheet_ids:
            if l1 in KEY_PLAN_CALLOUT_IDS:
                continue
            if _in_key_plan_zone(box1) or _in_key_plan_zone(box2):
                continue
            k = (l1, l2)
            if k not in seen:
                seen.add(k)
                refs.append({"callout": l1, "target_sheet": l2, "raw": f"{l1}/{l2}"})

    # NOTE (fix #6b): per-page locator filtering used to happen right here
    # -- checking whether own_sheet_id appeared among a callout's targets
    # on THIS page, and dropping that callout's refs from THIS page only.
    # That's no longer done here. The same static locator icon repeats
    # unchanged on multiple sheets, and on any sheet that isn't one of its
    # own two targets, the "own_sheet_id in targets" test never fires --
    # so a per-page-only filter lets that sheet's copy of the icon through
    # unfiltered and re-bridges exactly the groups the filter exists to
    # keep apart (see fix #6b in the module docstring and
    # `_filter_template_locator_callouts` in section F). Detection is now
    # done once, globally, after every sheet's cross_refs are collected,
    # so a single decision applies consistently everywhere the icon shows
    # up -- not just on the page(s) that happen to self-reference.
    return refs


# ===========================================================================
# E2. DETECTOR-BOX UTILITIES (reference-only, not wired into this pipeline)
#
# Carried over unmodified from the same reference source as
# find_scale_anchors above. These five functions operate on bounding boxes
# coming out of a *tiled ML detector pass* (dedup duplicate detections
# across overlapping tiles, filter by size/aspect, then score a detected
# drawing region against the text-derived scale/ARCH-REF anchors above).
# This script has no such detector stage -- detect_view_titles_from_page
# only ever needs the anchors themselves -- so none of these five are
# called anywhere in this file's pipeline. They're kept here, exactly as
# supplied, in case a sibling script in this pipeline (the one that runs
# the tiled detector) wants to import them and cross-check its boxes
# against the same anchors this script already extracts.
# ===========================================================================

def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tiled_nms(dets, iou_thresh):
    """Union-merge true cross-tile duplicates, then greedy NMS."""
    if not dets:
        return []
    merged = list(dets)
    changed = True
    while changed:
        changed = False
        out, used = [], [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            box, score, label = list(a["box"]), a["score"], a["label"]
            for j, b in enumerate(merged):
                if i == j or used[j]:
                    continue
                if box_iou(box, b["box"]) >= iou_thresh:
                    box[0] = min(box[0], b["box"][0])
                    box[1] = min(box[1], b["box"][1])
                    box[2] = max(box[2], b["box"][2])
                    box[3] = max(box[3], b["box"][3])
                    score = max(score, b["score"])
                    used[j] = True
                    changed = True
            out.append({"label": label, "score": score, "box": box})
        merged = out
    dets2 = sorted(merged, key=lambda d: d["score"], reverse=True)
    kept = []
    while dets2:
        best = dets2.pop(0)
        kept.append(best)
        dets2 = [d for d in dets2 if box_iou(best["box"], d["box"]) < iou_thresh]
    return kept


def quality_filter(dets, img_w, img_h, cfg, log: list[str]):
    page_area = float(img_w * img_h)
    keep = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        if w < cfg["min_w"] or h < cfg["min_h"]:
            log.append(f"drop small {w:.0f}x{h:.0f}")
            continue
        aspect = max(w / h, h / w)
        frac = (w * h) / page_area
        if aspect > cfg["max_aspect"]:
            log.append(f"drop aspect {aspect:.1f}")
            continue
        if frac < cfg["min_area_frac"]:
            log.append(f"drop tiny {frac*100:.2f}%")
            continue
        if frac > cfg["max_area_frac"]:
            log.append(f"drop whole-sheet {frac*100:.1f}%")
            continue
        keep.append(d)
    return keep


def generate_tiles(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles, y = [], 0
    while y < height:
        x = 0
        while x < width:
            x1, y1 = x, y
            x2, y2 = min(x1 + tile_size, width), min(y1 + tile_size, height)
            if x2 - x1 < tile_size and width >= tile_size:
                x1, x2 = width - tile_size, width
            if y2 - y1 < tile_size and height >= tile_size:
                y1, y2 = height - tile_size, height
            t = (x1, y1, x2, y2)
            if t not in tiles:
                tiles.append(t)
            if x2 >= width:
                break
            x += stride
        if y2 >= height:
            break
        y += stride
    return tiles


def score_against_anchors(clusters_pdf, anchors, up_tol=40.0, side_tol=30.0):
    """A cluster claims an anchor if the anchor sits inside its x-range and at/below
    its body (drawings sit above their title block)."""
    claims = []
    for c in clusters_pdf:
        x0, y0, x1, y1 = c
        got = [
            i for i, a in enumerate(anchors)
            if (x0 - side_tol) <= a["cx"] <= (x1 + side_tol) and y0 <= a["cy"] <= (y1 + up_tol)
        ]
        claims.append(got)
    hit = {i for g in claims for i in g}
    return dict(
        anchors=len(anchors),
        hit=len(hit),
        recall=round(len(hit) / len(anchors), 3) if anchors else None,
        clusters=len(clusters_pdf),
        clean=sum(1 for g in claims if len(g) == 1),
        merged=sum(1 for g in claims if len(g) > 1),
        spurious=sum(1 for g in claims if len(g) == 0),
    ), claims


# ===========================================================================
# F. ELEVATION GROUPING
# ===========================================================================

def _filter_template_locator_callouts(sheets):
    """
    Fix #6b (post-audit follow-up to fix #6, see module docstring).

    A callout label that fires both a self-reference and an external
    reference on the SAME sheet (fix #6's signal) is a static key-plan /
    locator icon reusing a bare numeral -- not a genuine detail/section
    pull bubble. The problem: this icon is baked into the title block and
    repeats, unchanged, on every sheet in the set. Fix #6's check
    ("own_sheet_id in targets") only ever looked at the CURRENT page's own
    copy of the callout, so it only fired on the page(s) where that
    sheet's own id happened to be one of the icon's two targets. On any
    OTHER sheet carrying the same unchanged icon -- concretely, MW1.2 and
    MW1.4 sit "between" the icon's own two targets (MW1.1 and MW1.3) and
    are never themselves a target -- the self-reference check silently
    never fires, so that sheet's copy of the icon passes straight through
    as if it were a real cross-reference bubble, quietly re-merging
    GROUP_A (MW1.1+MW1.2) into GROUP_B (MW1.3+MW1.4) one hop later than
    fix #6 blocked it.

    Fix: detect these icons document-wide, after every sheet's cross_refs
    have already been collected, instead of per-page. A callout id is
    treated as a template locator -- and is stripped from EVERY sheet's
    cross_refs, not just the sheet(s) it was first spotted on -- when the
    exact same target set shows up for it on two or more different
    sheets AND at least one of those occurrences is self-referential. A
    genuine detail/section pull bubble doesn't behave this way: it points
    at whatever is actually being pulled on that specific sheet, so its
    target set varies sheet to sheet rather than repeating identically.
    """
    by_callout = defaultdict(list)  # callout_id -> [(sheet_id, frozenset(targets)), ...]

    for s in sheets:
        sid = s["sheet_id"]
        per_sheet_targets = defaultdict(set)
        for ref in s["cross_refs"]:
            per_sheet_targets[ref["callout"]].add(ref["target_sheet"])
        for cid, targets in per_sheet_targets.items():
            by_callout[cid].append((sid, frozenset(targets)))

    template_locator_ids = set()
    for cid, occurrences in by_callout.items():
        target_sets = [t for _, t in occurrences]
        has_self_hit = any(sid in targets for sid, targets in occurrences)
        repeats_identically = (
            len(occurrences) > 1
            and len(set(target_sets)) == 1
            and len(target_sets[0]) > 1
        )
        if has_self_hit and repeats_identically:
            template_locator_ids.add(cid)

    if not template_locator_ids:
        return

    for s in sheets:
        if any(r["callout"] in template_locator_ids for r in s["cross_refs"]):
            s["cross_refs"] = [
                r for r in s["cross_refs"] if r["callout"] not in template_locator_ids
            ]
def _is_group_start_view_title(view_title):
    """
    Return True when the view title looks like the MAIN drawing that starts
    a new consecutive drawing set.

    Important:
    - TABLE CONTENT IS NOT USED to decide groups.
    - CROSS-REFERENCE BUBBLES ARE NOT USED to decide groups.
    - A group is a consecutive page range.

    We treat KEY PLAN and normal PLAN titles as possible group starts.
    PLAN SECTION / REFLECTED CEILING PLAN are deliberately excluded because
    they are commonly continuation views inside an already-started set.
    """
    if not view_title:
        return False

    t = re.sub(r"\s+", " ", str(view_title).strip().upper())
    if not t:
        return False

    # Explicit key-plan marker.
    if re.search(r"\bKEY\s+PLAN\b", t):
        return True

    # Do not start a new group from secondary plan-type views.
    excluded = (
        "PLAN SECTION",
        "REFLECTED CEILING PLAN",
        "RCP",
        "ROOF PLAN",
        "CEILING PLAN",
    )
    if any(x in t for x in excluded):
        return False

    # Main plan / plan-of-room titles.
    return bool(re.search(r"\bPLAN\b", t))


def _page_starts_new_drawing_group(sheet):
    """Check whether this sheet contains a main PLAN/KEY PLAN title."""
    for view in sheet.get("view_titles", []):
        if _is_group_start_view_title(view.get("view_title", "")):
            return True
    return False


def _assign_consecutive_drawing_groups(sheets):
    """
    Build groups strictly from PDF page order.

    A new group starts when a new main PLAN/KEY PLAN drawing title is found.
    Once started, every following consecutive sheet stays in that group until
    another main PLAN/KEY PLAN starts the next group.

    This intentionally does NOT use:
      * identical table/schedule signatures
      * table similarity
      * cross-reference bubbles
      * graph/BFS connected components

    Therefore the same schedule table can appear in Group A and Group B
    without merging those groups.
    """
    ordered = sorted(sheets, key=lambda s: s["page_num"])
    if not ordered:
        return []

    groups = []
    current = []
    group_number = 0

    for index, sheet in enumerate(ordered):
        is_start = _page_starts_new_drawing_group(sheet)

        # The first PDF page always starts the first group.
        if index == 0:
            group_number = 1
            current = [sheet]
            continue

        # A main PLAN/KEY PLAN on a later page starts a NEW consecutive group.
        if is_start:
            groups.append((group_number, current))
            group_number += 1
            current = [sheet]
        else:
            current.append(sheet)

    if current:
        groups.append((group_number, current))

    return groups


def build_elevation_groups(doc):
    """
    Build drawing groups from consecutive PDF page flow.

    Grouping rule:
        NEW MAIN PLAN / KEY PLAN -> NEW GROUP
        Otherwise               -> SAME GROUP

    Tables are still extracted elsewhere for schedule/material/hardware
    validation, but table similarity is NOT a grouping criterion.
    Cross-reference bubbles are also retained on each sheet for downstream
    validation/diagnostics, but they do NOT merge groups.
    """
    sheets = []
    for pno in range(len(doc)):
        tb = extract_sheet_title_block(doc[pno], pno + 1)
        sheets.append(tb)

    all_sheet_ids = {s["sheet_id"] for s in sheets if s["sheet_id"]}

    # sheet_id should normally be unique. Keep the existing lookup behavior,
    # but grouping itself is page-order based and does not depend on it.
    sheet_by_id = {s["sheet_id"]: s for s in sheets if s.get("sheet_id")}

    for s in sheets:
        pno = s["page_num"]
        page = doc[pno - 1]
        s["view_titles"] = detect_view_titles_from_page(page, pno)
        s["cross_refs"] = extract_sheet_bubbles(
            page,
            pno,
            all_sheet_ids,
            own_sheet_id=s["sheet_id"],
        )

    # Keep the existing locator filtering because cross_refs are still useful
    # as sheet-level information. They are simply no longer allowed to merge
    # otherwise separate drawing groups.
    _filter_template_locator_callouts(sheets)

    # Keep table signatures available for downstream validation/cache logic.
    # IMPORTANT: these signatures are NOT used below to form groups.
    for s in sorted(sheets, key=lambda x: x["page_num"]):
        page = doc[s["page_num"] - 1]
        table_data = extract_table_data_from_page(page)
        if table_data:
            sections = table_data[0]
            s["table_tag_signature"] = get_table_tag_signature(sections)
        else:
            s["table_tag_signature"] = ()

    # ---------------------------------------------------------------
    # THE GROUPING DECISION
    # ---------------------------------------------------------------
    # This is intentionally a simple sequential page-flow algorithm.
    # No table edges, no cross-reference edges, no BFS graph.
    grouped_ranges = _assign_consecutive_drawing_groups(sheets)

    groups = []
    for group_number, comp_sheets in grouped_ranges:
        if not comp_sheets:
            continue

        first_sheet = comp_sheets[0]
        group_id = f"GROUP_{first_sheet['sheet_id']}"
        scope = first_sheet.get("scope", "")
        mod_num = first_sheet.get("mod_number", "")
        all_views = [v for sh in comp_sheets for v in sh.get("view_titles", [])]

        groups.append({
            "group_id": group_id,
            "group_number": group_number,
            "mod_number": mod_num,
            "scope": scope,
            "sheet_ids": [sh["sheet_id"] for sh in comp_sheets],
            "pages": [sh["page_num"] for sh in comp_sheets],
            "sheets": comp_sheets,
            "total_views": len(all_views),
            "view_titles": [v["view_title"] for v in all_views],
        })

    return groups

def extract_drawing_titles_and_groups(pdf_path):
    if not fitz:
        raise ImportError("PyMuPDF (fitz) is required for elevation grouping.")
    doc = fitz.open(pdf_path)
    groups = build_elevation_groups(doc)
    doc.close()
    return groups


# ===========================================================================
# G. TABLE SIGNATURES (for the validation cache)
# ===========================================================================

def normalize_tag(tag):
    if not tag:
        return ""
    return " ".join(str(tag).strip().split())


def get_table_tag_signature(sections):
    tags = set()
    for sec in sections:
        for r in sec.get("rows", []):
            code = r.get("code")
            if code:
                norm = normalize_tag(code)
                if norm:
                    tags.add(norm)
    return tuple(sorted(tags))


def get_table_content_signature(sections):
    content = []
    for sec in sections:
        hdr = normalize_tag(sec.get("section_header", "")).upper()
        sec_rows = []
        for r in sec.get("rows", []):
            code = normalize_tag(r.get("code", ""))
            shape = normalize_tag(r.get("shape", "")).lower()
            qty = normalize_tag(r.get("qty", ""))
            desc = normalize_tag(r.get("description", ""))
            sec_rows.append((code, shape, qty, desc))
        content.append((hdr, tuple(sec_rows)))
    return tuple(content)


def get_drawing_view_signature(page, table_bbox, page_num=1):
    if page is None:
        return {"tokens": set(), "path_count": 0, "title_sig": "", "view_titles": (), "sheet_id": "", "scope": ""}

    TOL = 2.0
    table_x0, table_y0, table_x1, table_y1 = (table_bbox if table_bbox else (0, 0, 0, 0))
    raw_words = get_display_words(page)

    drawing_tokens = set()
    title_words = []
    for w in raw_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        in_table = (
            table_bbox is not None and
            x0 <= table_x1 + TOL and x1 >= table_x0 - TOL and
            y0 <= table_y1 + TOL and y1 >= table_y0 - TOL
        )
        if not in_table:
            cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(cleaned) >= 2:
                drawing_tokens.add(cleaned)
            up = text.upper()
            if any(k in up for k in ("PLAN", "ELEVATION", "SECTION", "DETAIL", "SHEET", "FLOOR", "LEVEL")):
                title_words.append(up)

    paths = get_display_drawings(page)
    path_count = 0
    for p in paths:
        r = p.get("rect")
        if r is None:
            continue
        in_table = (
            table_bbox is not None and
            r.x0 >= table_x0 - TOL and r.x1 <= table_x1 + TOL and
            r.y0 >= table_y0 - TOL and r.y1 <= table_y1 + TOL
        )
        if not in_table:
            path_count += 1

    v_titles = []
    tb_info = {}
    try:
        v_titles = [v["view_title"] for v in detect_view_titles_from_page(page, page_num)]
        tb_info = extract_sheet_title_block(page, page_num)
    except Exception:
        pass

    return {
        "tokens": drawing_tokens,
        "path_count": path_count,
        "title_sig": " ".join(title_words[:6]),
        "view_titles": tuple(sorted(v_titles)),
        "sheet_id": tb_info.get("sheet_id", ""),
        "scope": tb_info.get("scope", ""),
        "mod_number": tb_info.get("mod_number", ""),
    }


def is_drawing_view_materially_different(view1, view2, similarity_threshold=0.35):
    if not view1 or not view2:
        return False

    vt1 = set(view1.get("view_titles", ()))
    vt2 = set(view2.get("view_titles", ()))
    if vt1 and vt2 and vt1 != vt2:
        return True

    sc1 = view1.get("scope", "")
    sc2 = view2.get("scope", "")
    if sc1 and sc2 and sc1 != sc2 and not (sc1.startswith("Scope ") or sc2.startswith("Scope ")):
        return True

    t1 = view1.get("tokens", set())
    t2 = view2.get("tokens", set())

    if not t1 and not t2:
        return False
    if not t1 or not t2:
        return True

    tit1 = view1.get("title_sig", "")
    tit2 = view2.get("title_sig", "")
    if tit1 and tit2 and tit1 != tit2:
        nums1 = set(re.findall(r'\d+', tit1))
        nums2 = set(re.findall(r'\d+', tit2))
        if nums1 and nums2 and nums1 != nums2:
            return True

    intersection = len(t1 & t2)
    union = len(t1 | t2)
    similarity = intersection / union if union > 0 else 1.0

    if similarity < similarity_threshold:
        return True

    return False


# ===========================================================================
# H. TABLE EXTRACTION
# ===========================================================================

def extract_table_data_from_page(page):
    paths = get_display_drawings(page)

    all_h = []
    for p in paths:
        r = p.get('rect')
        if r is None:
            continue
        if r.width > 20 and r.height < 2:
            all_h.append((round(r.x0, 0), round(r.x1, 0), round(r.y0, 2)))
        elif r.width > 20 and 2 <= r.height < 40:
            all_h.append((round(r.x0, 0), round(r.x1, 0), round(r.y0, 2)))
            all_h.append((round(r.x0, 0), round(r.x1, 0), round(r.y1, 2)))

    if not all_h:
        return None

    span_groups = defaultdict(list)
    for x0, x1, y in all_h:
        rx0 = 10 * round(x0 / 10)
        rx1 = 10 * round(x1 / 10)
        span_groups[(rx0, rx1)].append((x0, x1, y))

    words = get_display_words(page)

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
                    if ("MATERIAL" in t or "HARDWARE" in t or "DESCRIPTION" in t
                            or "SCHEDULE" in t):
                        score += 5
                    elif ("CODE" in t or "SHAPE" in t or "QTY" in t
                          or "MATL" in t or "HDWE" in t or "FINISH" in t):
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
    for i, y in enumerate(table_ys[1:], 1):
        gap = y - BORDER_YS[-1]
        if gap <= 45:
            BORDER_YS.append(y)
        elif gap <= 250:
            subsequent = [table_ys[j] for j in range(i, len(table_ys))]
            if len(subsequent) >= 2 and (subsequent[1] - subsequent[0] <= 45):
                has_schedule_kw = False
                for wx0, wy0, wx1, wy1, wtext, *_ in words:
                    if TX0 - 15 <= wx0 <= TX1 + 15 and y - 30 <= wy0 <= y + 60:
                        wt = wtext.upper()
                        if any(k in wt for k in ("HARDWARE", "HDWE", "MATERIAL", "FINISH", "SCHEDULE")):
                            has_schedule_kw = True
                            break
                if has_schedule_kw:
                    BORDER_YS.append(y)
                    continue
            break
        else:
            break

    if len(BORDER_YS) < 2:
        return None

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
    col_xs = [TX0]
    for x in raw_vx:
        if x - col_xs[-1] > 5:
            col_xs.append(x)
    if col_xs[-1] < TX1 - 5:
        col_xs.append(TX1)

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

    words = get_display_words(page)
    table_words = [
        (x0, y0, x1, y1, text)
        for x0, y0, x1, y1, text, *_ in words
        if TX0 - 5 <= x0 <= TX1 + 10 and Y_TOP_TABLE <= y0 <= Y_BOT_TABLE
    ]

    if not table_words:
        return None

    raw_ys      = [w[1] for w in table_words]
    row_line_ys = _merge_vals(raw_ys, tol=3)

    lines_by_y = defaultdict(list)
    for x0, y0, x1, y1, text in table_words:
        ry = min(row_line_ys, key=lambda ly: abs(ly - y0))
        lines_by_y[ry].append((x0, text))

    structured_rows = []
    for ry in sorted(lines_by_y.keys()):
        words_sorted = sorted(lines_by_y[ry], key=lambda w: w[0])
        col_words    = defaultdict(list)
        for wx, wt in words_sorted:
            col_idx = 0 if wx < col_xs[0] else len(col_xs) - 2
            for j in range(len(col_xs) - 1):
                if col_xs[j] - 5 <= wx < col_xs[j + 1] + 5:
                    col_idx = j
                    break
            col_words[col_idx].append(wt)
        cells = [" ".join(col_words.get(j, [])) for j in range(len(col_xs) - 1)]
        shape = _lookup_shape(shape_map, ry, tol=8.0)
        structured_rows.append({"y": ry, "cells": cells, "shape": shape})

    def _row_to_record(cells):
        n = len(cells)
        if n == 0:   return "", "", ""
        if n == 1:   return cells[0], "", ""
        if n == 2:
            raw0 = cells[0].strip()
            m0   = CODE_RE.search(raw0.upper())
            if m0:
                leftover0 = (raw0[:m0.start()] + " " + raw0[m0.end():]).strip()
                desc = (leftover0 + " " + cells[1]).strip() if leftover0 else cells[1].strip()
                return raw0[m0.start():m0.end()], "", desc
            return cells[0], "", cells[1]

        cells = list(cells)
        code  = cells[0].strip()
        desc_prefix = ""

        if not code:
            for j in range(1, min(4, n)):
                cand = cells[j].strip()
                if cand and CODE_RE.search(cand.upper()):
                    code     = cand
                    cells[j] = ""
                    break
        else:
            m0 = CODE_RE.search(code.upper())
            if m0:
                before   = code[:m0.start()].strip()
                after    = code[m0.end():].strip()
                desc_prefix = (before + " " + after).strip()
                code     = code[m0.start():m0.end()]

        qty  = cells[1].strip()
        desc = " ".join(c for c in cells[2:] if c)

        if desc_prefix:
            desc = (desc_prefix + " " + desc).strip()

        if qty and not re.match(r'^[XxNnAaDd0-9/]+$', qty):
            desc = qty + (" " + desc if desc else "")
            qty  = ""
        return code, qty, desc

    sections = []
    cur_hdr  = None
    cur_rows = []

    for row in structured_rows:
        cells    = row["cells"]
        combined = " ".join(cells)
        shape    = row["shape"]

        non_empty = [c.strip() for c in cells if c.strip()]
        leading_cell_is_code = bool(
            non_empty and CODE_RE.search(non_empty[0].upper())
        )

        header_zone = " ".join(non_empty[:4]) if non_empty else ""
        if ANCHOR_RE.search(header_zone) and not leading_cell_is_code:
            if cur_hdr is not None and cur_rows:
                sections.append({"section_header": cur_hdr, "rows": cur_rows})

            upper_comb = combined.upper()
            if "HARDWARE" in upper_comb or "HDWE" in upper_comb:
                cur_hdr = "HARDWARE"
            else:
                cur_hdr = "MATERIALS"

            cur_rows = []
            continue

        code, qty, desc = _row_to_record(cells)
        code = _clean_code(code)
        if not any([code, qty, desc]):
            continue
        if code and not CODE_RE.search(code):
            continue
        if code and not ALLOWED_CODE_RE.match(code):
            # Doesn't conform to the approved project code-format standard
            # (§7.3) -- drop the row rather than extracting an unrecognized
            # code format as if it were valid.
            continue
        if not code and not qty.strip():
            continue

        enforced_shape = shape
        if cur_hdr == "MATERIALS":
            enforced_shape = "octagon"
        elif cur_hdr == "HARDWARE":
            enforced_shape = "rectangle"

        if re.match(r'^H(?:DWE)?-?\d', code, re.I):
            enforced_shape = "rectangle"
        elif re.match(r'^(?:MM|PL|WD|WV|PT|GL|SS|LAM)-[A-Z]', code, re.I):
            if enforced_shape in ("none", "rectangle") and cur_hdr != "HARDWARE":
                enforced_shape = "octagon"

        cur_rows.append({
            "code":        code,
            "shape":       enforced_shape,
            "qty":         qty.strip(),
            "description": desc.strip(),
            "raw_cells":   cells,
        })

    if cur_rows:
        sections.append({
            "section_header": cur_hdr or "MATERIALS",
            "rows":           cur_rows,
        })

    if not sections:
        return None

    return (sections, TX0, TX1, BORDER_YS, col_xs, shape_map, table_words, Y_TOP_TABLE, Y_BOT_TABLE)


# ===========================================================================
# I. VALIDATION CACHE
# ===========================================================================

class TableValidationCache:
    def __init__(self):
        self.cached_tables = {}
        self.doc_scheduled_codes = {}
        self.current_pdf = None
        self._pre_indexed = False

    def reset_if_new_pdf(self, pdf_path):
        abs_path = os.path.abspath(pdf_path) if pdf_path else None
        if self.current_pdf != abs_path:
            self.clear()
            self.current_pdf = abs_path
            self.pre_index_doc_schedules(pdf_path)

    def clear(self):
        self.cached_tables.clear()
        self.doc_scheduled_codes.clear()
        self.current_pdf = None
        self._pre_indexed = False

    def pre_index_doc_schedules(self, pdf_path):
        if self._pre_indexed or not pdf_path or not pdf_path.lower().endswith(".pdf"):
            return
        self._pre_indexed = True
        try:
            import fitz
            doc = fitz.open(pdf_path)
            for p_idx in range(doc.page_count):
                page = doc[p_idx]
                t_data = extract_table_data_from_page(page)
                if t_data:
                    sections = t_data[0]
                    for sec in sections:
                        for r in sec.get("rows", []):
                            code = normalize_tag(r.get("code", ""))
                            if code and code not in self.doc_scheduled_codes:
                                self.doc_scheduled_codes[code] = {
                                    "sheet_page": p_idx + 1,
                                    "shape": r.get("shape", ""),
                                    "description": r.get("description", ""),
                                }
            doc.close()
            if self.doc_scheduled_codes:
                print(f"  [PDF] Pre-indexed {len(self.doc_scheduled_codes)} scheduled code(s) "
                      f"across entire PDF to eliminate cross-sheet false positives.")
        except Exception as e:
            pass

    def lookup(self, tag_signature, content_signature, view_signature):
        if not tag_signature:
            return None, "EMPTY_TAGS"

        if tag_signature not in self.cached_tables:
            return None, "NEW_TAG_SIGNATURE"

        candidates = self.cached_tables[tag_signature]
        for entry in candidates:
            if entry["content_signature"] != content_signature:
                continue

            if is_drawing_view_materially_different(view_signature, entry["view_signature"]):
                return None, "DRAWING_VIEW_DIFFERENT"

            return entry, "REUSED_FROM_PREVIOUS_TABLE"

        return None, "CONTENT_CHANGED"

    def store(self, tag_signature, content_signature, view_signature, sheet_page,
              verification, extra_codes, cross_sheet_codes=None):
        if not tag_signature:
            return
        if tag_signature not in self.cached_tables:
            self.cached_tables[tag_signature] = []
        self.cached_tables[tag_signature].append({
            "sheet_page": sheet_page,
            "tag_signature": tag_signature,
            "content_signature": content_signature,
            "view_signature": view_signature,
            "verification": verification,
            "extra_codes": extra_codes,
            "cross_sheet_codes": cross_sheet_codes or [],
        })


GLOBAL_VALIDATION_CACHE = TableValidationCache()


def reset_validation_cache():
    GLOBAL_VALIDATION_CACHE.clear()


# ===========================================================================
# J. DRAWING-AREA VERIFICATION (group-level -- the path main() actually runs)
# ===========================================================================

def verify_codes_in_drawing(page, code_shape_map, table_x0, table_x1, table_y0, table_y1, cache=None):
    if not code_shape_map:
        return {}, []

    TOL = 2.0
    paths     = get_display_drawings(page)
    raw_words = get_display_words(page)

    drawing_words = []
    for w in raw_words:
        x0, y0, x1, y1, text = w[0], w[1], w[2], w[3], w[4]
        word_in_table = (
            x0 <= table_x1 + TOL and x1 >= table_x0 - TOL and
            y0 <= table_y1 + TOL and y1 >= table_y0 - TOL
        )
        if not word_in_table:
            drawing_words.append((x0, y0, x1, y1, text))

    results = {
        code: {"present_in_drawing": False, "drawing_occurrences": []}
        for code in code_shape_map if code
    }
    extra_codes = []
    seen = set()

    shaped_codes   = {c: s for c, s in code_shape_map.items()
                      if s in ("octagon", "rectangle", "circle")}
    shapeless_codes = {c for c, s in code_shape_map.items()
                       if s not in ("octagon", "rectangle", "circle")}

    for p in paths:
        r = p.get("rect")
        if r is None or r.height < 3 or r.width < 3:
            continue

        if r.width > 150 or r.height > 150:
            continue

        if (r.x0 >= table_x0 - TOL and r.x1 <= table_x1 + TOL and
                r.y0 >= table_y0 - TOL and r.y1 <= table_y1 + TOL):
            continue

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

        upper_text = _restore_dots(text_inside.upper())
        upper_text_no_spaces = _restore_dots(upper_text.replace(" ", ""))

        matched_any = False
        for expected_code, expected_shape in shaped_codes.items():
            if expected_shape != shape:
                continue

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

        if not matched_any:
            possible_codes = set()
            upper_text_restored = _restore_dots(upper_text)
            for m in CODE_RE_LOOSE.finditer(upper_text_restored):
                possible_codes.add(m.group(1))

            if not possible_codes:
                for m in CODE_RE_LOOSE.finditer(_restore_dots(upper_text_no_spaces)):
                    possible_codes.add(m.group(1))

            doc_scheduled = cache.doc_scheduled_codes if cache else {}

            for pc in possible_codes:
                clean_pc = _strip_punct(pc)
                if len(clean_pc) < 3:
                    continue

                pc_norm = normalize_tag(pc)

                if pc_norm in doc_scheduled or pc in code_shape_map:
                    scheduled_on = doc_scheduled.get(pc_norm, {}).get("sheet_page", "?")
                    dedup_key = ("__CROSS_SHEET__", pc, int(round(cx / 8)), int(round(cy / 8)))
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        print(f"    [VERIFY] INFO: Found '{pc}' at ({round(cx)},{round(cy)}) "
                              f"— scheduled on Sheet {scheduled_on}, NOT flagged as unscheduled.")
                    continue

                dedup_key = ("__EXTRA__", pc, int(round(cx / 8)), int(round(cy / 8)))
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    extra_codes.append({
                        "code": pc,
                        "x": round(cx),
                        "y": round(cy),
                        "shape": shape
                    })
                    print(f"    [VERIFY] WARNING: Found unscheduled code '{pc}' at ({round(cx)},{round(cy)}) shape={shape}")

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


def validate_elevation_group(doc, group, cache=None):
    if cache is None:
        cache = GLOBAL_VALIDATION_CACHE

    per_sheet = {}
    group_code_info = {}

    for page_num in group["pages"]:
        page = doc[page_num - 1]
        table_data = extract_table_data_from_page(page)

        if table_data is None:
            per_sheet[page_num] = {"table_bbox": None, "page": page}
            continue

        sections, TX0, TX1, BORDER_YS, col_xs, shape_map, table_words, Y_TOP, Y_BOT = table_data
        per_sheet[page_num] = {"table_bbox": (TX0, Y_TOP, TX1, Y_BOT), "page": page}

        for sec in sections:
            for r in sec["rows"]:
                code = r["code"]
                if not code:
                    continue
                entry = group_code_info.setdefault(code, {
                    "shape": r["shape"], "table_sheets": set(), "description": r["description"],
                })
                entry["table_sheets"].add(page_num)
                if entry["shape"] in ("none", "") and r["shape"] not in ("none", ""):
                    entry["shape"] = r["shape"]
                if not entry["description"] and r["description"]:
                    entry["description"] = r["description"]

    group_code_shape_map = {code: info["shape"] for code, info in group_code_info.items()}

    code_present_any = {code: False for code in group_code_shape_map}
    code_occurrences = defaultdict(list)
    sheet_extra_codes = {}

    for page_num, info in per_sheet.items():
        bbox = info["table_bbox"]
        tx0, ty0, tx1, ty1 = bbox if bbox else (0.0, 0.0, 0.0, 0.0)

        verification, extra_codes = verify_codes_in_drawing(
            info["page"], group_code_shape_map,
            table_x0=tx0, table_x1=tx1, table_y0=ty0, table_y1=ty1,
            cache=cache,
        )
        sheet_extra_codes[page_num] = extra_codes

        for code, v in verification.items():
            if v.get("present_in_drawing"):
                code_present_any[code] = True
                for occ in v["drawing_occurrences"]:
                    code_occurrences[code].append(dict(occ, sheet_page=page_num))

    schedule_omissions = []
    for code, occs in code_occurrences.items():
        table_sheets = group_code_info.get(code, {}).get("table_sheets", set())
        leadered_sheets = {o["sheet_page"] for o in occs}
        for sheet_page in sorted(leadered_sheets - table_sheets):
            schedule_omissions.append({
                "code": code,
                "leadered_on_sheet": sheet_page,
                "scheduled_on_sheets": sorted(table_sheets),
            })

    orphaned_codes = sorted(c for c, present in code_present_any.items() if not present)
    # Fix: a schedule omission (a code leadered on a sheet's drawing but
    # missing from that sheet's own local table -- §12) is reported as a
    # defect in its own right in the spec, not merely informational. §14's
    # GROUP_B example fails for two independent reasons (an orphaned code
    # AND a schedule omission); nothing in the spec suggests a group with
    # zero orphaned codes but a live schedule omission should read as PASS.
    status = "PASS" if not orphaned_codes and not schedule_omissions else "FAIL"

    return {
        "group_id":   group["group_id"],
        "mod_number": group["mod_number"],
        "scope":      group["scope"],
        "sheet_ids":  group["sheet_ids"],
        "pages":      group["pages"],
        "status":     status,
        "codes": {
            code: {
                "shape":              info["shape"],
                "table_sheets":       sorted(info["table_sheets"]),
                "present_in_drawing": code_present_any.get(code, False),
                "occurrences":        code_occurrences.get(code, []),
            }
            for code, info in group_code_info.items()
        },
        "orphaned_codes":      orphaned_codes,
        "schedule_omissions":  schedule_omissions,
        "unscheduled_codes_in_drawing": sheet_extra_codes,
    }


def validate_all_elevation_groups(pdf_path, cache=None):
    if not fitz:
        raise ImportError("PyMuPDF (fitz) is required for elevation-group validation.")
    if cache is None:
        cache = GLOBAL_VALIDATION_CACHE

    doc = fitz.open(pdf_path)
    cache.reset_if_new_pdf(pdf_path)
    groups = build_elevation_groups(doc)
    results = [validate_elevation_group(doc, g, cache=cache) for g in groups]
    doc.close()
    return results


# ===========================================================================
# K. DEBUG RENDERING
# (only reachable from the legacy per-sheet path in section L below --
#  validate_all_elevation_groups(), the path main() actually runs, never
#  calls this)
# ===========================================================================

def save_pdf_vector_debug(page, TX0, TX1, BORDER_YS, col_xs,
                          table_words, Y_BOT, shape_map,
                          verification_results, output_path, extra_codes=None,
                          validation_status="VALIDATED", reused_from_sheet=None):
    """
    Debug image showing table region, row/column boundaries, word boxes,
    table code shapes, and drawing-area occurrences of each code.
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

    # Banner for Reused Validation
    if validation_status == "REUSED_FROM_PREVIOUS_TABLE":
        banner_str = f"STATUS: REUSED_FROM_PREVIOUS_TABLE (Sheet {reused_from_sheet})"
        (bw, bh), _ = cv2.getTextSize(banner_str, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (20, 20), (20 + bw + 20, 20 + bh + 20), (255, 255, 255), -1)
        cv2.rectangle(img, (20, 20), (20 + bw + 20, 20 + bh + 20), (0, 140, 255), 3)
        cv2.putText(img, banner_str, (30, 30 + bh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 220), 2)

    # 1. Annotate Missed Codes IN THE TABLE (Red Box & Arrow from Left)
    missing_text_y = int(y_top * sy) + 40
    for code, info in verification_results.items():
        if not info.get("present_in_drawing"):
            found_box = False
            for wx0, wy0, wx1, wy1, wtext in table_words:
                if code in wtext.upper() or _strip_punct(code) in _strip_punct(wtext.upper()):
                    box_x0 = int(wx0 * sx) - 2
                    box_y0 = int(wy0 * sy) - 2
                    box_x1 = int(wx1 * sx) + 2
                    box_y1 = int(wy1 * sy) + 2
                    cv2.rectangle(img, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 255), 3)

                    text_str = f"MISSING: {code}"
                    text_x = max(10, int(TX0 * sx) - 400)
                    (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    cv2.rectangle(img, (text_x - 5, missing_text_y - th - 5), (text_x + tw + 5, missing_text_y + 5), (255, 255, 255), -1)
                    cv2.putText(img, text_str, (text_x, missing_text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
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
            cv2.rectangle(img, (px - 20, py - 15), (px + 20, py + 15), (0, 140, 255), 3)
            
            text_str = f"UNSCHEDULED: {occ['code']} ({occ['shape']})"
            text_x = max(10, int(TX0 * sx) - 400)
            (tw, th), _ = cv2.getTextSize(text_str, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            cv2.rectangle(img, (text_x - 5, missing_text_y - th - 5), (text_x + tw + 5, missing_text_y + 5), (255, 255, 255), -1)
            cv2.putText(img, text_str, (text_x, missing_text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
            
            if px < text_x:
                start_pt = (text_x - 10, missing_text_y - th // 2)
                end_pt = (px + 14, py)
                mid_x = start_pt[0] - min(60, max(10, (start_pt[0] - end_pt[0]) // 3))
            else:
                start_pt = (text_x + tw + 10, missing_text_y - th // 2)
                end_pt = (px - 14, py)
                mid_x = start_pt[0] + min(60, max(10, (end_pt[0] - start_pt[0]) // 3))
            
            cv2.line(img, start_pt, (mid_x, start_pt[1]), (0, 140, 255), 2)
            cv2.arrowedLine(img, (mid_x, start_pt[1]), end_pt, (0, 140, 255), 2, tipLength=0.03)
            missing_text_y += 45

    cv2.imwrite(output_path, img)
    print(f"  [PDF DEBUG] Saved -> {output_path}")


# ===========================================================================
# L. LEGACY PER-SHEET PATH
# NOT called by main() / the CLI below (see section M) -- main() runs the
# group-level path in section J instead. Kept here, unmodified, in case
# another script in this pipeline (e.g. run_all_submittals.py, app.py)
# still imports and calls extract() or extract_pdf_vector() directly.
# ===========================================================================

def extract_pdf_vector(pdf_path: str, page_num: int = 0,
                       debug_path: str = None,
                       cache: TableValidationCache = None) -> dict | None:
    if not fitz:
        return None

    doc = fitz.open(pdf_path)
    if page_num < 0 or page_num >= doc.page_count:
        doc.close()
        return None

    page = doc[page_num]
    if page.rotation != 0:
        print(f"  [PDF] Page has rotation={page.rotation}° — "
              f"applying rotation correction to all extracted geometry.")

    t_data = extract_table_data_from_page(page)
    if not t_data:
        doc.close()
        return None

    sections, TX0, TX1, BORDER_YS, col_xs, shape_map, table_words, Y_TOP_TABLE, Y_BOT_TABLE = t_data

    code_shape_map = {
        r["code"]: r["shape"]
        for sec in sections
        for r in sec["rows"]
        if r["code"]
    }
    all_codes = list(code_shape_map.keys())

    tag_sig = get_table_tag_signature(sections)
    content_sig = get_table_content_signature(sections)
    view_sig = get_drawing_view_signature(page, table_bbox=(TX0, Y_TOP_TABLE, TX1, Y_BOT_TABLE))

    if cache is None:
        cache = GLOBAL_VALIDATION_CACHE
    cache.reset_if_new_pdf(pdf_path)

    cached_entry, reason = cache.lookup(tag_sig, content_sig, view_sig)

    if cached_entry is not None:
        validation_status = "REUSED_FROM_PREVIOUS_TABLE"
        reused_from_sheet = cached_entry["sheet_page"]
        verification = cached_entry["verification"]
        extra_codes = cached_entry.get("extra_codes", [])

        print(f"  [PDF] ─────────────────────────────────────────────────────────────")
        print(f"  [PDF] Tag signature MATCHES previously validated table on Sheet {reused_from_sheet}!")
        print(f"  [PDF] Status: REUSED_FROM_PREVIOUS_TABLE")
        print(f"  [PDF] Skipping redundant drawing validation ({len(all_codes)} codes reused)")
        print(f"  [PDF] ─────────────────────────────────────────────────────────────\n")
    else:
        if reason == "DRAWING_VIEW_DIFFERENT":
            print(f"  [PDF] [NOTE] Tag signature matches previous sheet, but associated drawing/view")
            print(f"        is materially different. Performing fresh validation.")
        elif reason == "CONTENT_CHANGED":
            print(f"  [PDF] [NOTE] Table content changed (tags added/removed/modified).")
            print(f"        Treating as new table and validating separately.")

        validation_status = "VALIDATED"
        reused_from_sheet = None

        print(f"  [PDF] Verifying {len(all_codes)} codes in drawing area …")
        verification, extra_codes = verify_codes_in_drawing(
            page,
            code_shape_map=code_shape_map,
            table_x0=TX0,
            table_x1=TX1,
            table_y0=Y_TOP_TABLE,
            table_y1=Y_BOT_TABLE,
            cache=cache,
        )

        cache.store(
            tag_signature=tag_sig,
            content_signature=content_sig,
            view_signature=view_sig,
            sheet_page=page_num + 1,
            verification=verification,
            extra_codes=extra_codes,
        )

    for sec in sections:
        for row in sec["rows"]:
            code = row["code"]
            info = verification.get(code, {})
            row["present_in_drawing"]  = info.get("present_in_drawing", False)
            row["drawing_occurrences"] = info.get("drawing_occurrences", [])
            row["validation_status"]   = validation_status
            if reused_from_sheet is not None:
                row["reused_from_sheet"] = reused_from_sheet

    found_count   = sum(1 for c, v in verification.items() if v.get("present_in_drawing"))
    missing_count = len(all_codes) - found_count
    print(f"  [PDF] Drawing verification ({validation_status}): "
          f"{found_count} found, {missing_count} not found in drawing\n")

    if debug_path:
        save_pdf_vector_debug(
            page, TX0, TX1, BORDER_YS, col_xs,
            table_words, Y_BOT_TABLE, shape_map, verification, debug_path,
            extra_codes=extra_codes,
            validation_status=validation_status,
            reused_from_sheet=reused_from_sheet,
        )

    doc.close()

    return {
        "sections": sections,
        "method": "pdf_vector",
        "table_bbox": [TX0, Y_TOP_TABLE, TX1, Y_BOT_TABLE],
        "unscheduled_codes_in_drawing": extra_codes,
        "sheet_page": page_num + 1,
        "validation_status": validation_status,
        "reused_from_sheet": reused_from_sheet,
        "tag_signature": list(tag_sig),
    }


def extract(input_path: str, debug_path: str = None,
            page_num: int = 0,
            cache: TableValidationCache = None) -> dict:
    ext    = os.path.splitext(input_path)[1].lower()
    result = None

    if ext == ".pdf":
        print(f"  Attempting PDF vector extraction on page {page_num + 1} …")
        result = extract_pdf_vector(input_path, page_num, debug_path, cache=cache)
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
            if "MATERIAL" in text or "HARDWARE" in text or "FINISH" in text:
                has_keywords = True
                break
        if not has_keywords:
            print("  [INFO] Table lacks MATERIAL/HARDWARE/FINISH keywords. Discarding.\n")
            result["sections"] = []

    return result


# ===========================================================================
# N. PER-VIEW SECTION-REFERENCE VALIDATION  (ADDITIVE)
#
# New feature (per Robin's spec): group-level validation in section J checks
# a group's schedule codes against the UNION of the group's drawing areas.
# This section adds a SEPARATE, finer-grained check that does NOT replace
# that logic -- it runs alongside it.
#
# Scope, as specified:
#   * Only PLAN / PLAN SECTION [...] views are in scope (matched on the
#     view_title text detect_view_titles_from_page already extracts).
#   * For each such view, look for a section/detail reference bubble
#     printed near it in the form "N/SHEETID" (e.g. "3/MW1.1" -- N is a
#     drawing/detail number, SHEETID is the sheet that drawing lives on).
#     The target can be the SAME sheet the PLAN view is on, or ANY OTHER
#     sheet in the PDF.
#   * Resolve the bubble to its target drawing, extract the schedule codes
#     visible IN that target drawing's own area (not the whole page, and
#     not the group union), and check each against:
#       (a) the schedule table on the TARGET drawing's own sheet, and
#       (b) the schedule table on the SOURCE sheet (the sheet holding the
#           PLAN / PLAN SECTION view + the bubble).
#
# Per-view drawing-area approximation
# ------------------------------------
# Nothing in this pipeline stores a bounding box for "the drawing itself"
# (as opposed to just its title/scale text) -- table extraction is the only
# code path that has a precise region, and that's specific to schedule
# tables. Views on these sheets are laid out as stacked horizontal bands
# (see MW1.1: "PLAN", "PLAN SECTION BASE", "ELEVATION SOUTH", each with its
# own title + scale line, one under another). So a view's own drawing area
# is approximated here as the page band from its own title down to the next
# detected view's title on that page (or the page bottom, if it's the last
# view) -- see `_view_band`. This is a proxy, not a hard vector boundary;
# it's good enough for locating both the "N/SHEETID" bubble text near a
# view and the codes drawn inside that view's band.
# ===========================================================================

SECTION_BUBBLE_RE = re.compile(r'\b(\d{1,2})\s*/\s*([A-Z]{1,3}\d+(?:\.\d+)?)\b')
PLAN_VIEW_TITLE_RE = re.compile(r'^\s*PLAN\b', re.I)  # "PLAN", "PLAN SECTION BASE ...", etc.


def _view_band(view, page_views, page_height):
    """
    Approximate the vertical band of the page occupied by ONE view: from
    its own title down to the next detected view's title below it (or the
    bottom of the page, if none). See module note above section N.
    """
    y0 = view["bbox"][1]
    below = sorted(v["bbox"][1] for v in page_views if v is not view and v["bbox"][1] > y0 + 1)
    y1 = below[0] if below else page_height
    return y0, y1


def find_view_section_bubble(page, view, page_views):
    """
    Look for a 'N/SHEETID' section/detail reference bubble (e.g. '3/MW1.1')
    belonging to this specific PLAN / PLAN SECTION view. Only these view
    types are in scope, per spec -- ELEVATION/DETAIL/etc. views are left
    alone here (group-level validation in section J still covers them).
    Returns {"drawing_no": "3", "sheet_id": "MW1.1"} or None.
    """
    if not PLAN_VIEW_TITLE_RE.match(view.get("view_title", "") or ""):
        return None

    y0, y1 = _view_band(view, page_views, page.rect.height)
    words = get_display_words(page)
    band_words = [w for w in words if y0 - 5 <= w[1] <= y1 + 5]

    candidates = []
    for wx0, wy0, wx1, wy1, text, *_ in band_words:
        m = SECTION_BUBBLE_RE.search(text)
        if m:
            candidates.append((wy0, m.group(1), m.group(2).upper()))

    # The bubble's number and sheet-id are sometimes split across separate
    # word boxes ("3", "/", "MW1.1") -- also try adjacent-token joins.
    if not candidates:
        for tok in _build_adjacent_tokens(band_words, max_x_gap=15.0, y_tol=4.0):
            m = SECTION_BUBBLE_RE.search(tok["text"])
            if m:
                candidates.append((tok["y0"], m.group(1), m.group(2).upper()))

    if not candidates:
        return None

    # Prefer whichever candidate sits closest to this view's own title --
    # a page can have more than one PLAN-type view, each with its own bubble.
    candidates.sort(key=lambda c: abs(c[0] - y0))
    _, drawing_no, sheet_id = candidates[0]
    return {"drawing_no": drawing_no, "sheet_id": sheet_id}


def _build_doc_view_index(doc):
    """
    Walk every page ONCE and build:
      pages_info[page_num] = {"sheet_id", "views", "page"}
      target_index[(sheet_id, drawing_no)] = (page_num, view)
    across the WHOLE document -- a bubble's target may be on any sheet, not
    just sheets in the same elevation group.
    """
    pages_info = {}
    target_index = {}

    for pno in range(len(doc)):
        page = doc[pno]
        page_num = pno + 1
        tb = extract_sheet_title_block(page, page_num)
        views = detect_view_titles_from_page(page, page_num)
        pages_info[page_num] = {"sheet_id": tb["sheet_id"], "views": views, "page": page}

        for v in views:
            vid = v.get("view_id")
            if vid:
                target_index[(tb["sheet_id"], str(vid).upper())] = (page_num, v)

    return pages_info, target_index


def _extract_codes_in_band(page, y0, y1):
    """
    Harvest recognizable schedule codes (ALLOWED_CODE_RE) from text sitting
    inside a page band -- i.e. codes actually drawn/labeled in a specific
    view's own area, as opposed to the codes listed in a schedule table.
    Reuses the same single/adjacent/line token building already used for
    drawing-area verification in section J, but to EXTRACT codes rather
    than to check specific expected ones.
    """
    words = get_display_words(page)
    band_words = [w for w in words if y0 - 5 <= w[1] <= y1 + 5]

    tokens = (
        [{"text": w[4], "cx": (w[0] + w[2]) / 2, "cy": (w[1] + w[3]) / 2} for w in band_words]
        + _build_adjacent_tokens(band_words, max_x_gap=25.0, y_tol=5.0)
        + _build_line_tokens(band_words, y_tol=5.0)
    )

    found = {}
    for tok in tokens:
        restored = _restore_dots(tok["text"].upper())
        for m in CODE_RE_LOOSE.finditer(restored):
            code = m.group(1)
            if ALLOWED_CODE_RE.match(code) and code not in found:
                found[code] = {"x": round(tok["cx"]), "y": round(tok["cy"])}
    return found


def validate_plan_view_section_references(pdf_path):
    """
    For every PLAN / PLAN SECTION view in the document that carries a
    'N/SHEETID' bubble: resolve the bubble's target drawing (same sheet OR
    any other sheet), extract the codes visible in that target drawing's
    own band, and check each against BOTH the target sheet's own table and
    the source sheet's table. Purely additive: does not read from or write
    to anything group-level validation (section J) produces.
    """
    if not fitz:
        raise ImportError("PyMuPDF (fitz) is required.")

    doc = fitz.open(pdf_path)
    pages_info, target_index = _build_doc_view_index(doc)

    table_code_cache = {}

    def _table_codes_for_page(page_num):
        if page_num not in table_code_cache:
            page = pages_info[page_num]["page"]
            t_data = extract_table_data_from_page(page)
            codes = set()
            if t_data:
                for sec in t_data[0]:
                    for r in sec["rows"]:
                        if r["code"]:
                            codes.add(r["code"])
            table_code_cache[page_num] = codes
        return table_code_cache[page_num]

    results = []

    for page_num in sorted(pages_info.keys()):
        info = pages_info[page_num]
        source_sheet_id = info["sheet_id"]
        page = info["page"]
        views = info["views"]

        for view in views:
            bubble = find_view_section_bubble(page, view, views)
            if not bubble:
                continue

            record = {
                "source_sheet": source_sheet_id,
                "source_page": page_num,
                "source_view": view["view_title"],
                "bubble": f'{bubble["drawing_no"]}/{bubble["sheet_id"]}',
                "target_resolved": False,
            }

            target = target_index.get((bubble["sheet_id"], bubble["drawing_no"]))
            if target is None:
                record["error"] = (
                    f'Bubble references drawing {bubble["drawing_no"]} on sheet '
                    f'{bubble["sheet_id"]}, but no matching view was found in the PDF.'
                )
                results.append(record)
                continue

            target_page_num, target_view = target
            target_info = pages_info[target_page_num]
            target_page = target_info["page"]
            target_sheet_id = target_info["sheet_id"]

            t_y0, t_y1 = _view_band(target_view, target_info["views"], target_page.rect.height)
            extracted = _extract_codes_in_band(target_page, t_y0, t_y1)

            target_table_codes = _table_codes_for_page(target_page_num)
            source_table_codes = _table_codes_for_page(page_num)

            code_results = {
                code: {
                    "in_target_sheet_table": code in target_table_codes,
                    "in_source_sheet_table": code in source_table_codes,
                }
                for code in extracted
            }
            unresolved_codes = sorted(
                c for c, v in code_results.items()
                if not v["in_target_sheet_table"] and not v["in_source_sheet_table"]
            )

            record.update({
                "target_resolved": True,
                "target_sheet": target_sheet_id,
                "target_page": target_page_num,
                "target_view": target_view["view_title"],
                "codes_in_target_drawing": code_results,
                "unresolved_codes": unresolved_codes,
                "status": "PASS" if not unresolved_codes else "FAIL",
            })
            results.append(record)

    doc.close()
    return results


# ===========================================================================
# M. CLI ENTRY POINT
# ===========================================================================

def _sheets_with_pages(sheet_ids, pages):
    """'MW1.1 (p.1), MW1.2 (p.2), ...' -- pairs each sheet id with its page
    number so the two never have to be cross-referenced by hand."""
    return ", ".join(f"{sid} (p.{pg})" for sid, pg in zip(sheet_ids, pages))


def _print_section(title, char="="):
    print("\n" + char * 88)
    print(f"  {title}")
    print(char * 88)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Drawing vs Scheduling Extractor")
    parser.add_argument("input_file", help="Path to input PDF")
    parser.add_argument("--out", "-o", default="elevation_groups_validation.json",
                        help="Path to write the JSON validation output")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress the per-match [VERIFY] debug lines")
    args = parser.parse_args()

    input_path = args.input_file

    if args.quiet:
        global print
        _real_print = print
        def print(*a, **kw):  # noqa: A001 - intentional local shadow
            if a and isinstance(a[0], str) and a[0].strip().startswith("[VERIFY]"):
                return
            _real_print(*a, **kw)

    GLOBAL_VALIDATION_CACHE.clear()

    print("  [GROUPING] Tracing title blocks, view titles & cross-reference bubbles ...")
    groups = extract_drawing_titles_and_groups(input_path)

    total_groups = len(groups)
    total_sheets = sum(len(g["sheet_ids"]) for g in groups)

    # --------------------------------------------------------------
    # 1. ELEVATION GROUP SUMMARY -- how many groups, how many sheets,
    #    and which page each sheet lives on, all in one place.
    # --------------------------------------------------------------
    _print_section("ELEVATION GROUP SUMMARY")
    print(f"  Elevation groups detected : {total_groups}")
    print(f"  Sheets grouped            : {total_sheets}  (out of {len(fitz.open(input_path))} pages in the PDF)")
    print("-" * 88)
    print(f"  {'#':<4}{'GROUP ID':<16}{'#SHEETS':<9}{'MOD':<10}SCOPE")
    print("-" * 88)
    for idx, g in enumerate(groups, start=1):
        print(f"  {idx:<4}{g['group_id']:<16}{len(g['sheet_ids']):<9}{g['mod_number']:<10}{g['scope']}")
    print("-" * 88)

    print("\n  Sheet -> page number, by group:")
    for idx, g in enumerate(groups, start=1):
        print(f"    {idx}. {g['group_id']}  ({len(g['sheet_ids'])} sheet"
              f"{'s' if len(g['sheet_ids']) != 1 else ''}):")
        for sid, pg in zip(g["sheet_ids"], g["pages"]):
            print(f"         - Sheet {sid:<10} -> Page {pg}")

    # --------------------------------------------------------------
    # 1b. CODES BY SHEET -- what's actually IN each sheet's own
    #     schedule table, listed page by page / sheet by sheet.
    #     (Extraction only -- this is not the group-level drawing
    #     verification, which runs next.)
    # --------------------------------------------------------------
    sheet_lookup = {}  # page_num -> (sheet_id, group_id)
    for g in groups:
        for sid, pg in zip(g["sheet_ids"], g["pages"]):
            sheet_lookup[pg] = (sid, g["group_id"])

    codes_by_sheet = []  # collected for the JSON payload below
    _print_section("CODES BY SHEET  (extracted from each sheet's own schedule table)")
    doc_for_tables = fitz.open(input_path)
    for pg in sorted(sheet_lookup.keys()):
        sid, gid = sheet_lookup[pg]
        page = doc_for_tables[pg - 1]
        table_data = extract_table_data_from_page(page)

        print(f"\n  Page {pg}  |  Sheet {sid}  |  {gid}")

        if not table_data:
            print(f"        (no schedule table detected on this sheet)")
            codes_by_sheet.append({
                "page": pg, "sheet_id": sid, "group_id": gid, "sections": [],
            })
            continue

        sections = table_data[0]
        total_codes = sum(1 for sec in sections for r in sec["rows"] if r["code"])
        print(f"        {total_codes} code(s):")

        sec_payload = []
        for sec in sections:
            hdr = sec["section_header"]
            codes_in_sec = [r["code"] for r in sec["rows"] if r["code"]]
            if codes_in_sec:
                print(f"          [{hdr:<10}] {', '.join(codes_in_sec)}")
            sec_payload.append({"section": hdr, "codes": codes_in_sec})

        codes_by_sheet.append({
            "page": pg, "sheet_id": sid, "group_id": gid, "sections": sec_payload,
        })
    doc_for_tables.close()

    # --------------------------------------------------------------
    # 2. VALIDATION
    # --------------------------------------------------------------
    print("\n  [VALIDATE] Checking each elevation group's schedule tags "
          "against the UNION of its sheets' drawing areas ...")
    GLOBAL_VALIDATION_CACHE.clear()
    group_results = validate_all_elevation_groups(input_path, cache=GLOBAL_VALIDATION_CACHE)

    pass_count = sum(1 for gr in group_results if gr["status"] == "PASS")
    fail_count = total_groups - pass_count

    _print_section("VALIDATION RESULTS  (1 row per elevation group)")
    print(f"  {'#':<4}{'GROUP ID':<16}{'SHEETS (PAGE)':<38}{'RESULT'}")
    print("-" * 88)
    for idx, gr in enumerate(group_results, start=1):
        sheets_pages = _sheets_with_pages(gr["sheet_ids"], gr["pages"])
        status_tag = "PASS" if gr["status"] == "PASS" else "FAIL"
        print(f"  {idx:<4}{gr['group_id']:<16}{sheets_pages:<38}{status_tag}")
        if gr["status"] != "PASS":
            if gr["orphaned_codes"]:
                print(f"         reason: orphaned/unused code(s) -> {', '.join(gr['orphaned_codes'])}")
            for om in gr["schedule_omissions"]:
                print(f"         reason: '{om['code']}' leadered on Page {om['leadered_on_sheet']} "
                      f"but missing from that sheet's own schedule table "
                      f"(scheduled on page(s) {om['scheduled_on_sheets'] or '—'})")
    print("-" * 88)

    # --------------------------------------------------------------
    # 3. GROUP-BY-GROUP REPORT -- for each group: its page range and
    #    sheets, its view titles, the codes extracted from its sheets'
    #    tables (and whether each code is on EVERY sheet in the group
    #    or only some), and the group-level drawing verification.
    # --------------------------------------------------------------
    group_view_titles = {g["group_id"]: g["view_titles"] for g in groups}

    _print_section("GROUP-BY-GROUP REPORT")
    for idx, gr in enumerate(group_results):
        label = chr(65 + idx) if idx < 26 else str(idx + 1)
        view_titles = group_view_titles.get(gr["group_id"], [])
        sheets_pages = _sheets_with_pages(gr["sheet_ids"], gr["pages"])
        pg_min, pg_max = min(gr["pages"]), max(gr["pages"])
        n_sheets = len(gr["sheet_ids"])
        codes_info = gr["codes"]  # code -> {shape, table_sheets, present_in_drawing, occurrences}
        all_codes = sorted(codes_info.keys())

        print(f"\n{'=' * 88}")
        print(f"  GROUP {label}   ({gr['group_id']})   [{gr['status']}]")
        print(f"{'=' * 88}")
        print(f"  Pages     : {pg_min} to {pg_max}   ({n_sheets} sheet{'s' if n_sheets != 1 else ''})")
        print(f"  Sheets    : {sheets_pages}")
        print(f"  MOD       : {gr['mod_number']}")
        print(f"  Scope     : {gr['scope']}")

        print(f"\n  View Titles ({len(view_titles)}):")
        if view_titles:
            for vt in view_titles:
                print(f"    - {vt}")
        else:
            print(f"    (none detected)")

        print(f"\n  Extracted Codes ({len(all_codes)}) -- union across every sheet's table in Group {label}:")
        print(f"    {', '.join(all_codes) if all_codes else '(none)'}")

        print(f"\n  Code presence across sheets in Group {label} "
              f"(does the same code appear on EVERY sheet's own table?):")
        if all_codes:
            print(f"    {'CODE':<12}{'ON SHEETS (table)':<30}{'ON ALL SHEETS?'}")
            for code in all_codes:
                info = codes_info[code]
                sheets_with_code = [sid for sid, pg in zip(gr["sheet_ids"], gr["pages"])
                                     if pg in info["table_sheets"]]
                missing_from = [sid for sid in gr["sheet_ids"] if sid not in sheets_with_code]
                on_all = not missing_from
                tag = "YES" if on_all else f"NO -- missing on: {', '.join(missing_from)}"
                print(f"    {code:<12}{', '.join(sheets_with_code):<30}{tag}")
        else:
            print(f"    (no codes extracted)")

        present_codes = sorted(c for c, i in codes_info.items() if i["present_in_drawing"])
        missing_codes = gr["orphaned_codes"]
        print(f"\n  Drawing verification (group-level, per DVCS §11 -- checked against the "
              f"UNION of all sheets' drawing areas):")
        print(f"    Verified present in drawing ({len(present_codes)}) : "
              f"{', '.join(present_codes) if present_codes else '(none)'}")
        print(f"    NOT found in drawing / orphaned ({len(missing_codes)}) : "
              f"{', '.join(missing_codes) if missing_codes else '(none)'}")
        if gr["schedule_omissions"]:
            print(f"    Schedule omissions ({len(gr['schedule_omissions'])}):")
            for om in gr["schedule_omissions"]:
                print(f"      - '{om['code']}' leadered on Page {om['leadered_on_sheet']} but "
                      f"missing from that sheet's own table (scheduled on page(s) "
                      f"{om['scheduled_on_sheets'] or '—'})")
    print(f"\n{'=' * 88}")

    # --------------------------------------------------------------
    # 3b. PER-VIEW SECTION-REFERENCE VALIDATION (section N -- additive)
    # --------------------------------------------------------------
    print("\n  [VALIDATE] Checking PLAN / PLAN SECTION view bubbles against "
          "their target drawing's own sheet table AND the source sheet's table ...")
    view_ref_results = validate_plan_view_section_references(input_path)

    _print_section("PER-VIEW SECTION-REFERENCE VALIDATION  (additive -- section N)")
    if not view_ref_results:
        print("  (no PLAN / PLAN SECTION view carried a resolvable 'N/SHEETID' bubble)")
    for rec in view_ref_results:
        print(f"\n  Source : {rec['source_sheet']} (p.{rec['source_page']})  "
              f"-- \"{rec['source_view']}\"")
        print(f"  Bubble : {rec['bubble']}")
        if not rec["target_resolved"]:
            print(f"  Result : UNRESOLVED -- {rec['error']}")
            continue
        print(f"  Target : {rec['target_sheet']} (p.{rec['target_page']})  "
              f"-- \"{rec['target_view']}\"")
        codes = rec["codes_in_target_drawing"]
        print(f"  Codes found in target drawing ({len(codes)}): "
              f"{', '.join(sorted(codes)) if codes else '(none)'}")
        for code, v in sorted(codes.items()):
            tags = []
            if v["in_target_sheet_table"]:
                tags.append("in target sheet table")
            if v["in_source_sheet_table"]:
                tags.append("in source sheet table")
            print(f"    {code:<10} {', '.join(tags) if tags else 'NOT found in either table'}")
        print(f"  Result : {rec['status']}"
              + (f"  (unresolved: {', '.join(rec['unresolved_codes'])})" if rec["unresolved_codes"] else ""))

    # --------------------------------------------------------------
    # 4. FINAL SUMMARY
    # --------------------------------------------------------------
    _print_section("VALIDATION SUMMARY")
    print(f"  Total elevation groups : {total_groups}")
    print(f"  Total sheets grouped   : {total_sheets}")
    print(f"  PASS                   : {pass_count}")
    print(f"  FAIL                   : {fail_count}")
    print("=" * 88)

    serializable = []
    for gr in group_results:
        gr2 = dict(gr)
        gr2["sheets"] = [
            {"sheet_id": sid, "page": pg}
            for sid, pg in zip(gr["sheet_ids"], gr["pages"])
        ]
        n_sheets_gr = len(gr["sheet_ids"])
        gr2["codes"] = {
            code: {
                "shape": info["shape"],
                "table_sheets": info["table_sheets"],
                "table_sheet_ids": [sid for sid, pg in zip(gr["sheet_ids"], gr["pages"])
                                     if pg in info["table_sheets"]],
                "on_all_sheets": len(info["table_sheets"]) == n_sheets_gr,
                "present_in_drawing": info["present_in_drawing"],
                "occurrences": info["occurrences"],
            }
            for code, info in gr["codes"].items()
        }
        serializable.append(gr2)

    gr_by_id = {gr["group_id"]: gr for gr in group_results}

    output_payload = {
        "codes_by_sheet": codes_by_sheet,
        "summary": {
            "total_groups": total_groups,
            "total_sheets_grouped": sum(len(g["sheet_ids"]) for g in groups),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "groups": [
                {
                    "group_id": g["group_id"],
                    "sheet_count": len(g["sheet_ids"]),
                    "sheet_ids": g["sheet_ids"],
                    "pages": g["pages"],
                    # explicit sheet_id -> page_number pairing, so page numbers
                    # never have to be cross-referenced against sheet_ids by hand
                    "sheets": [
                        {"sheet_id": sid, "page": pg}
                        for sid, pg in zip(g["sheet_ids"], g["pages"])
                    ],
                    "mod_number": g["mod_number"],
                    "status": gr_by_id[g["group_id"]]["status"],
                    "view_titles": g["view_titles"],
                    "visible_codes": sorted(
                        code for code, info in gr_by_id[g["group_id"]]["codes"].items()
                        if info["present_in_drawing"]
                    ),
                    "missing_codes": gr_by_id[g["group_id"]]["orphaned_codes"],
                }
                for g in groups
            ],
        },
        "groups": serializable,
        # Additive -- per-view section-reference validation (section N).
        # Does not affect "summary"/"groups" above in any way.
        "plan_view_section_references": view_ref_results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\n  Elevation group validation saved -> {args.out}\n")


if __name__ == "__main__":
    main()