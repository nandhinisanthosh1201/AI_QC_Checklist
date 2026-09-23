"""
crop_drawings.py
================
Automatically detects every individual drawing/view on each sheet of a
submittal PDF and draws a bounding box around it (rather than cropping it
out into its own PNG). One annotated image is written per processed page,
with a rectangle + label drawn for every detected drawing.

HOW IT WORKS (see write-up for full explanation)
-------------------------------------------------
1. ANCHOR DETECTION (primary)
   Every drawing view in this drawing set is labeled with a fixed 3-line
   callout, left-aligned to the same x-position:

        <TITLE>                  (font size ~11pt, italic)
        SCALE <ratio>            (font size ~8pt)
        ARCH REF: <sheet/detail> (font size ~8pt)

   We find every "ARCH REF:" span, then walk upward requiring each line to
   stack directly under the previous one (small dy, same x0) with the title
   line being the larger font. This triplet is the anchor for one drawing.

   FALLBACK: on sheets that don't follow this specific 3-line pattern (e.g.
   a different template that only has a "SCALE:" callout with no ARCH REF
   line), we fall back to a more generic title-block detector: it anchors
   on any scale-like keyword (see SCALE_KEYWORDS) and walks upward to
   collect the title lines above it, with no ARCH REF requirement. If that
   also finds nothing (fully rasterized/scanned sheet), and --use-vlm is
   passed, we fall back once more to reading titles directly off a
   rendered image of the page via Qwen-VL (OpenRouter).

   PATCHED (this version): the fallback chain used to be gated behind
   `if not anchors` at the page level -- i.e. it only ran when the PRIMARY
   detector found ZERO anchors anywhere on the page. That silently missed
   individual views on pages where the primary detector found SOME (but
   not all) anchors: e.g. a sheet with 3 drawings where 2 match the tight
   ARCH REF triplet pattern exactly and a 3rd (a different drawing type,
   slightly different title-block layout) narrowly misses the alignment
   tolerance. Because the page as a whole "succeeded" (anchors was
   non-empty), the fallback never ran, and that 3rd view was dropped with
   no error -- it just never entered the pipeline. See PATCH NOTES
   (fallback gating fix) below for the fix: fallback now always runs
   alongside the primary detector, and results are merged with
   overlap-based de-duplication instead of an all-or-nothing gate.

2. SHEET FRAME
   Each sheet has a constant outer border and a vertical divider that
   separates the drawing field from the title block on the right. We detect
   these per-page from the vector line-work (robust to template changes).

3. TESSELLATION (the "formula")
   Anchors are clustered into rows (by y0) and sorted into columns (by x0)
   within each row. For each row:
       top    = bottom of the row above (or sheet top border)
       bottom = this anchor's own bottom + padding
   The LEFT/RIGHT split between two adjacent columns is NOT the midpoint of
   their title text (title text width has nothing to do with how wide the
   actual drawing is -- a title can be much narrower than its drawing's
   dimension lines, which caused real content to get clipped). Instead we
   search the actual ink (vector paths + text) for the true empty gap that
   separates the two drawings:
     - restrict the search to the x-zone between the two titles
     - merge nearby ink into blobs (small gaps within one annotation, e.g.
       a leader line to a callout, get merged together)
     - take the RIGHT-MOST gap that's wide enough to be real (this avoids
       being fooled by a wide gap between a drawing's body and its own
       far-flung callout text, which can be *larger* than the true gap to
       the next drawing, but always sits to the left of it)
   The TOP edge is then tightened to the actual topmost ink found inside
   that column (instead of always reaching up to the sheet border), which
   removes large stretches of blank space above small/low-set drawings.

   PATCHED: row/column tessellation is computed INDEPENDENTLY PER
   COLUMN-GROUP (see find_column_group_bounds / COLUMN-GROUP PARTITIONING
   below), not once globally across the full sheet width. See that
   section's docstring for the bug this fixes.

4. VISUALIZATION (instead of cropping)
   Rather than rendering each computed rectangle as its own cropped PNG,
   we draw every rectangle directly onto a full-page render, along with a
   small text label (the detected title). This produces one annotated
   image per page that you can eyeball to sanity-check the detection /
   tessellation logic, instead of dozens of small crop files.

USAGE
-----
    pip install pymupdf --break-system-packages
    python3 crop_drawings.py input.pdf output_dir/ [--zoom 2.5] [--pages 6,7,15]

    # Enable the generic + VLM fallback for sheets that don't match the
    # primary ARCH REF pattern (requires `pip install openai --break-system-packages`
    # and `pip install pillow --break-system-packages`, plus OPENROUTER_API_KEY
    # set in the environment):
    python3 crop_drawings.py input.pdf output_dir/ --use-vlm

This is a single, self-contained script -- no other project files required.

Each annotated page is saved as:
    output_dir/p{page:03d}_bboxes.png

A manifest.csv is also written with page, title, scale, arch_ref, the
detection method used, and the PDF-space bounding-box rectangle for every
drawing found (the rectangles that got drawn on the page image).

--------------------------------------------------------------------------
PATCH NOTES (fallback gating fix -- this version)
--------------------------------------------------------------------------
BUG: a sheet with several drawings where the PRIMARY detector (the tight
TITLE/SCALE/ARCH REF triplet matcher in find_anchors(), tol_x/tol_y ~2pt)
correctly finds most of them, but one view -- often a different drawing
type on the same sheet (e.g. an ELEVATION block mixed in among PLAN
SECTION blocks) whose title-block text happens to be laid out with a
slightly different indent or line-spacing -- narrowly misses the tight
alignment tolerance and is silently dropped. No error is raised; the page
just ends up with fewer boxes than drawings.

Root cause: the fallback chain (generic scale-keyword detector, then
optional VLM read) was gated with a page-level `if not anchors:` check --
it only ran when find_anchors() found ZERO anchors on the ENTIRE page.
A page where 2 of 3 views matched was "successful" as far as that gate
was concerned, so the fallback -- which might well have caught the 3rd,
oddly-aligned view -- never even ran.

Fix: run BOTH the primary detector and the fallback chain on every page,
unconditionally, then merge the two anchor lists with overlap-based
de-duplication (merge_anchor_sets() / _anchors_overlap() below) so a
fallback-detected anchor is only added when it does NOT spatially overlap
an anchor the primary detector already found. This means:
  - Pages where the primary detector finds everything: fallback anchors
    all overlap existing ones and add nothing (cheap, harmless).
  - Pages where the primary detector finds SOME anchors: fallback can now
    supply the ones it missed.
  - Pages where the primary detector finds NOTHING: behaves exactly as
    before (fallback supplies everything).
Per-page console output now reports how many anchors came from each
source, e.g. "[page 006] anchors: primary=2 fallback_added=1", so a
partial primary-detector miss is visible in the log instead of being
silently absorbed into what looked like a fully successful page.

Secondary hardening: find_anchors()'s default tol_x/tol_y was tightened
from 2.0pt in an earlier version to 3.5pt here, giving a little more
slack for near-miss alignment before a view has to fall through to the
fallback path at all. This is a minor mitigation, not a fix on its own --
the real fix is the always-run-fallback-and-merge behavior above.
--------------------------------------------------------------------------
PATCH NOTES (this version)
--------------------------------------------------------------------------
BUG: a sheet whose content field contains more than one independent
"grid" of views -- e.g. a 2x2 elevation grid PLUS a narrow side column of
unrelated stacked views (a "TILT-UP REVEAL ELEVATION" over a "TYP CAST
CONC. REVEAL JOINT" detail, sitting in their own strip beside the main
grid) -- produced clipped boxes in the main grid.

Root cause: cluster_rows() grouped every anchor ON THE WHOLE PAGE into
rows by y0 alone, then compute_crop_boxes() applied EVERY row's bottom
bound (and, for multi-anchor rows, its ink-tightened top bound) across
the FULL content width -- including under columns whose real row
structure has nothing to do with that row. The side column's own title
(at a y0 that doesn't line up with either row of the main grid) became
its own phantom "row" sandwiched between the main grid's two real rows.
That phantom row's bottom bound then became the TOP bound for the main
grid's second row, sheet-wide -- silently clipping the top strip (parapet
callouts, top dimension strings) off both drawings in that row, even
though visually that content is nowhere near the side column.

Fix: partition the sheet's content field into independent COLUMN-GROUPS
first, using the sheet's own full-height SOLID vertical divider lines
(find_vertical_dividers -- already used elsewhere for single-anchor-row
bounding), then run cluster_rows()/row-bottom computation SEPARATELY
within each column-group. A group's row boundaries can now never be
applied under a different group. See find_column_group_bounds() and the
restructured compute_crop_boxes() below.

Caveat: this partition step relies on a real ruled divider existing
between the groups (as DGS/Turner-style templates typically have). A
sheet where two independent view-clusters sit side by side with no ruled
line between them -- just whitespace -- would still need an ink-gap-based
partition; that's a reasonable next step if it turns out to matter in
practice, but isn't implemented here.
--------------------------------------------------------------------------
PATCH NOTES (rotation fix, v1 -- superseded, kept for context)
--------------------------------------------------------------------------
BUG: on pages with a non-zero /Rotate (e.g. /Rotate 270, a portrait
MediaBox displayed as landscape), every anchor/divider/title detector
found nothing -- "ARCH REF pattern not found" / "No anchors found" on
every page, even though the sheets clearly have title blocks.

First attempt assumed page.get_text() and page.get_drawings() ALWAYS
return coordinates in the RAW, unrotated MediaBox space (while page.rect
/ get_pixmap() apply /Rotate and live in DISPLAY space), and unconditionally
applied page.rotation_matrix to every extracted bbox to normalize into
display space. That assumption turned out to be wrong for the PyMuPDF
version actually in use here -- get_text()/get_drawings() were ALREADY
returning rotation-aware (display-space) coordinates on this install, so
forcing an extra rotation_matrix transform on top of already-correct data
mapped everything into a third, bogus space: this is what produced giant
merged boxes spanning unrelated drawings and sideways-rotated labels once
tested against a real sheet. See PATCH NOTES (rotation fix, v2) below for
the actual fix that replaced this.
--------------------------------------------------------------------------
PATCH NOTES (rotation fix, v2 -- current)
--------------------------------------------------------------------------
PyMuPDF's rotation-awareness for get_text() vs get_drawings() is not
consistent across versions (and the two functions can even differ from
each other on the same version), so hardcoding either "always raw" or
"always display" is unsafe -- whichever guess is wrong silently produces
plausible-looking but corrupted geometry (see v1 above), which is much
harder to notice than an outright crash.

Fix: auto-detect, per page and separately for text vs. drawings, whether
the extracted coordinates already fit inside page.rect (DISPLAY space) or
need page.rotation_matrix applied (RAW space) -- see _auto_matrices()
below. The detection samples the actual extent of what get_text() /
get_drawings() returns and compares it against page.rect: if it already
fits, coordinates are left alone (identity transform); if it doesn't
(e.g. a raw-space portrait extent that overflows a rotated-to-landscape
page.rect), the real rotation matrices are applied. This is checked
independently for get_spans() (text) and get_content_items() /
find_border_and_divider() / find_vertical_dividers() (drawings), since
those two extraction paths aren't guaranteed to agree.

Everything downstream of extraction (find_anchors, group_title_blocks,
cluster_rows, find_column_split, find_column_group_bounds,
compute_crop_boxes) is untouched -- it only ever works with the returned
bbox tuples, so once those tuples are reliably in DISPLAY space (whether
that took a real transform or none at all), the existing heuristics work
unmodified.

draw_boxes_on_page() needs the inverse: page.new_shape() draws directly
into whatever RAW content-stream space get_drawings() reads from, so
final (display-space) rectangles are converted back with the matching
`to_raw` matrix from the same auto-detection used for drawings -- which
also means NO extra rotation is applied when detection found drawings
were already display-space (avoiding the v1 bug in the write-back path
too). Label text's `rotate=` argument is likewise only set to counter-
rotate the glyphs when a real transform was actually detected; otherwise
it's left at 0.

Diagnostics: process_pdf() now prints, once per page, whether a real
rotation transform was applied for text and for drawings, e.g.:
    [page 006] rotation=270  text=identity  drawings=identity
so you can see directly what was detected instead of only inferring it
from the output image. If boxes are still off after this fix, that
printed line tells you which extraction path (text vs. drawings) to
look at -- if it says the OPPOSITE of what's actually true (i.e. it says
"identity" but coordinates really are raw, or vice versa), the sampling
heuristic in _auto_matrices() picked wrong, most likely on a page whose
content happens to cluster in a way that fits inside page.rect either
way (e.g. a mostly-empty sheet, or a near-square one) -- in that case,
override the detection manually for that page rather than trusting the
heuristic blindly.
--------------------------------------------------------------------------
PATCH NOTES (cropped output + title filter -- this version)
--------------------------------------------------------------------------
CHANGE 1: Actual per-drawing crops are now saved, not just the annotated
full-page overview. Each detected drawing is rendered as its own PNG,
named after its detected title, into an output_dir/{pdf_name}/crops/
folder (see crop_and_save_boxes() below). The crop is taken BEFORE the
red bounding boxes / labels are drawn onto the page, so the saved crops
are clean (no annotation overlay) -- only the separate full-page
"p###_bboxes.png" overview still shows the boxes. The manifest.csv now
has a `crop_file` column recording which crop file corresponds to each
detected drawing.

CHANGE 2: A new `--filter TEXT` CLI option restricts processing to only
drawings whose detected title contains TEXT (case-insensitive substring
match), e.g. `--filter elevation` keeps only views like "SOUTH
ELEVATION" or "INTERIOR ELEVATION - LOBBY" and skips PLAN/SECTION/DETAIL
views on the same sheets. Filtering happens AFTER detection and
tessellation (so the full anchor set is still used for correct row/column
math -- filtering the anchors first would corrupt the tessellation of the
drawings that ARE kept, since neighboring anchors are what the column-
split/row-bound logic uses as reference points) and BEFORE both cropping
and the annotated overview render, so pages with no matching drawings are
skipped entirely (no overview PNG, no crops, no manifest rows) and pages
with a partial match only draw/crop the matching subset.
--------------------------------------------------------------------------
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

# --------------------------------------------------------------------------
# Generic title-block fallback config (vector-text grouping + optional VLM
# read). Used to supplement the primary ARCH REF detector on every page --
# see PATCH NOTES (fallback gating fix) at the top of this file.
# --------------------------------------------------------------------------

QWEN_MODEL = "qwen/qwen2.5-vl-72b-instruct"

# These keywords act as the anchor to locate the title block in generic mode.
SCALE_KEYWORDS = ["scale:", "1/4\"", "1/8\"", "1'-", "nts", "n.t.s", "=\"", "1'-0"]

# Bounding-box drawing style.
BBOX_COLOR = (1, 0, 0)       # red
BBOX_WIDTH = 1.2             # PDF points
LABEL_COLOR = (1, 0, 0)      # red
LABEL_SIZE = 7               # PDF points

# --------------------------------------------------------------------------
# ARCH vs SUBMITTAL classification + sheet/view numbering
# --------------------------------------------------------------------------
# Sheet numbers across both doc types look like: S8, MW1.2, AW271, QW345,
# A0.06B, 1.1, 1.12 -- i.e. 0-3 leading letters, digits, optional .decimal,
# optional 0-2 trailing letters. Kept loose on purpose (see PATCH NOTES
# below) so it matches both letter-prefixed codes and pure decimals.
# Also allows an optional underscore, hyphen, or space after the letter prefix (e.g. A_1.21).
SHEET_NUMBER_PATTERN = re.compile(r"^[A-Z]{0,3}[_\-\s]?\d+([.\-_]\d+)?[A-Z]{0,2}$")

# Arch sheets sometimes carry an explicit "SHEET NUMBER" / "DWG NO." label
# next to the value. Submittal sheets (per Muthamil, 2026-09) do NOT --
# it's just a standalone code sitting in the bottom-right title-block cell.
SHEET_NUMBER_LABELS = re.compile(
    r"(?i)^(sheet\s*(no\.?|number)|dwg\.?\s*no\.?|drawing\s*no\.?)\s*:?$"
)

# Values that pass SHEET_NUMBER_PATTERN but are NOT sheet numbers and can
# legitimately sit in the same bottom-right corner region (job #, revision
# numbers, etc.) -- excluded explicitly rather than relying on pattern
# strictness alone, since over-tightening the pattern risks rejecting real
# sheet numbers like "1.1" (see PATCH NOTES below).
SHEET_NUMBER_EXCLUDE = re.compile(r"(?i)^(job|mod|rev|rel)\s*#?\.?$")

# Calendar years (2000-2099) sit in the same bottom-right corner region
# (issuance date, revision table) and pass SHEET_NUMBER_PATTERN as plain
# 4-digit numbers -- exclude them explicitly, since real arch/submittal
# sheet numbers are never a bare 4-digit year with no letters/decimal.
YEAR_PATTERN = re.compile(r"^20\d{2}$")


def classify_pdf_type(pdf_path) -> str:
    """
    Classify a PDF as 'arch' or 'submittal' purely from its filename, so the
    caller knows which of ARCH_CROP_FOLDER / SUBMITTAL_CROP_FOLDER to write
    crops into.

    'submittal' is checked first because real filenames have included both
    words at once (e.g. "..._Submittal_..._DGS_..."), and in that case the
    doc is a submittal package referencing arch sheets, not an arch sheet
    itself.
    """
    stem = Path(pdf_path).stem.lower()
    if "submittal" in stem:
        return "submittal"
    elif "arch" in stem:
        return "arch"
    else:
        print(
            f"[warn] could not classify '{stem}' as arch/submittal from "
            f"filename -- defaulting to 'submittal'. Pass --doc-type to "
            f"override."
        )
        return "submittal"


def find_sheet_number(page, spans, doc_type="submittal"):
    """
    Find this PAGE's own sheet number (e.g. "S8", "A0.06B") -- distinct
    from ARCH REF, which is a per-VIEW cross-reference to a detail
    elsewhere and must never be used as the sheet number.

    doc_type == "arch":   try an explicit "SHEET NUMBER" / "DWG NO." label
                           first (common on arch title blocks), then fall
                           through to the same corner heuristic below.
    doc_type == "submittal": no label to anchor on -- go straight to the
                           bottom-right-corner heuristic.
    """
    pr = page.rect

    if doc_type == "arch":
        label_spans = [s for s in spans if SHEET_NUMBER_LABELS.match(s["text"].strip())]
        for label in label_spans:
            lx0, ly0, lx1, ly1 = label["bbox"]
            candidates = [
                s for s in spans
                if s is not label
                and SHEET_NUMBER_PATTERN.match(s["text"].strip())
                and not SHEET_NUMBER_EXCLUDE.match(s["text"].strip())
                and not YEAR_PATTERN.match(s["text"].strip())
                and (
                    (0 <= s["bbox"][1] - ly1 < 40 and abs(s["bbox"][0] - lx0) < 60)     # below label
                    or (0 <= s["bbox"][0] - lx1 < 150 and abs(s["bbox"][1] - ly0) < 15)  # beside label
                )
            ]
            if candidates:
                candidates.sort(key=lambda s: s["size"], reverse=True)
                return candidates[0]["text"].strip()

    # Bottom-right corner heuristic (default path for submittal, fallback
    # path for arch if no label was found). Biggest-font standalone code
    # in the corner region wins; ties broken by lowest position (title
    # block's sheet-number cell is typically the last row in that corner).
    corner_spans = [
        s for s in spans
        if s["bbox"][0] > pr.width * 0.80 and s["bbox"][1] > pr.height * 0.75
        and SHEET_NUMBER_PATTERN.match(s["text"].strip())
        and not SHEET_NUMBER_EXCLUDE.match(s["text"].strip())
        and not YEAR_PATTERN.match(s["text"].strip())
    ]
    if corner_spans:
        print(
            f"  [sheet_number] corner candidates: "
            f"{[(s['text'].strip(), round(s['size'], 1)) for s in corner_spans]}"
        )
        corner_spans.sort(key=lambda s: (s["size"], s["bbox"][1]), reverse=True)
        return corner_spans[0]["text"].strip()

    return None


def _find_view_number(spans, ref_x0, ref_y0, exclude=(), x_search=100, y_tol=15):
    """
    Find the small circled letter/number (e.g. "A", "B", "1") that sits to
    the LEFT of a view's title/scale/arch_ref stack -- the "view number"
    printed right next to the detail bubble, as opposed to ARCH REF (a
    cross-reference to a different sheet entirely) or the sheet number
    (the whole page's own number, found separately by find_sheet_number).

    ref_x0, ref_y0: top-left corner of the title line for this view --
    the search box extends leftward and is loosely y-aligned to that line.
    """
    candidates = [
        s for s in spans
        if s not in exclude
        and s["bbox"][2] <= ref_x0 + 5
        and (ref_x0 - s["bbox"][2]) < x_search
        and abs(s["bbox"][1] - ref_y0) < y_tol
        and 1 <= len(s["text"].strip()) <= 3
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda s: ref_x0 - s["bbox"][2])  # nearest wins
    return candidates[0]["text"].strip()


def get_api_key() -> str:
    """Reads the OpenRouter API key from the environment. Only needed if
    the --use-vlm fallback is actually triggered."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY environment variable is not set.\n"
            "  PowerShell:  $env:OPENROUTER_API_KEY = \"sk-or-v1-...\"\n"
            "  cmd:         set OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  bash/zsh:    export OPENROUTER_API_KEY=\"sk-or-v1-...\"\n"
            "Never hardcode API keys directly in this script."
        )
    return key


def _b64_encode(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


# --------------------------------------------------------------------------
# 0. Rotation normalization helpers
# --------------------------------------------------------------------------

_IDENTITY = fitz.Matrix(1, 0, 0, 1, 0, 0)


def _sample_extent_text(page):
    """Flat list of (x, y) coordinates from every text span on the page,
    exactly as get_text() returns them -- i.e. whatever space that
    function happens to use, unmodified. Used only to detect that space,
    not for any actual content."""
    xs, ys = [], []
    d = page.get_text("dict")
    for b in d["blocks"]:
        if "lines" not in b:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if s["text"].strip():
                    x0, y0, x1, y1 = s["bbox"]
                    xs += [x0, x1]
                    ys += [y0, y1]
    return xs, ys


def _sample_extent_drawings(page):
    """Flat list of (x, y) coordinates from every line/rect endpoint
    get_drawings() returns, unmodified. Used only to detect that
    function's coordinate space."""
    xs, ys = [], []
    for d in page.get_drawings():
        for it in d["items"]:
            if it[0] == "l":
                p0, p1 = it[1], it[2]
                xs += [p0.x, p1.x]
                ys += [p0.y, p1.y]
            elif it[0] == "re":
                r = it[1]
                xs += [r.x0, r.x1]
                ys += [r.y0, r.y1]
    return xs, ys


def _auto_matrices(page, xs, ys):
    """
    Decide whether coordinates sampled from get_text()/get_drawings() on
    this page are already in DISPLAY space (matching page.rect -- i.e.
    already rotation-aware) or in RAW/unrotated MediaBox space, and return
    (to_display, to_raw, rotated) where `rotated` is False when no real
    transform was needed (identity) and True when page.rotation_matrix /
    page.derotation_matrix were actually applied.

    This is determined empirically per page and per extraction source
    rather than hardcoded, because PyMuPDF's actual rotation-awareness for
    get_text() vs get_drawings() is inconsistent across versions (see
    PATCH NOTES (rotation fix, v2) at the top of this file). If the
    sampled extent already fits inside page.rect (with a little slack),
    the coordinates are treated as already display space and left alone;
    otherwise the real rotation/derotation matrices are applied.
    """
    if page.rotation == 0 or not xs:
        return _IDENTITY, _IDENTITY, False

    disp = page.rect
    ext_w = max(xs) - min(xs)
    ext_h = max(ys) - min(ys)
    slack = 2.0
    if ext_w <= disp.width + slack and ext_h <= disp.height + slack:
        return _IDENTITY, _IDENTITY, False
    return page.rotation_matrix, page.derotation_matrix, True


def _text_matrices(page):
    """(to_display, to_raw, rotated) for whatever space get_text() uses on
    this page -- see _auto_matrices()."""
    xs, ys = _sample_extent_text(page)
    return _auto_matrices(page, xs, ys)


def _drawing_matrices(page):
    """(to_display, to_raw, rotated) for whatever space get_drawings()
    uses on this page -- see _auto_matrices()."""
    xs, ys = _sample_extent_drawings(page)
    return _auto_matrices(page, xs, ys)


def _bbox_to_display(bbox, m):
    """
    Transform a (x0, y0, x1, y1) bbox through matrix m and return the
    axis-aligned bbox of the four transformed corners. Using all four
    corners (not just the two given points) matters for 90/270-degree
    rotations, where a box's width/height effectively swap and a naive
    two-point transform can give a degenerate or wrongly-oriented rect.
    """
    x0, y0, x1, y1 = bbox
    pts = [
        fitz.Point(x0, y0) * m,
        fitz.Point(x1, y1) * m,
        fitz.Point(x0, y1) * m,
        fitz.Point(x1, y0) * m,
    ]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def group_title_blocks(items):
    """Group vector-text spans into cohesive {number, title, scale} blocks.

    Generic fallback used to supplement find_anchors() -- e.g. a sheet
    with only a "SCALE:" callout and no ARCH REF line, OR a single odd
    view on an otherwise-normal sheet whose title-block layout narrowly
    misses find_anchors()'s tight alignment tolerance (see PATCH NOTES
    (fallback gating fix) at the top of this file).

    Multi-line title support
    ------------------------
    Titles are often split across 2-3 lines, e.g.:
        FOOD PREP &          <- line 1
        STAFF LOUNGE -       <- line 2
        NORTH                <- line 3 (closest to the SCALE: text)

    Lines are collected by walking directly upward from the scale one line
    at a time -- each next line must sit immediately above (small gap) and
    left-aligned with the line just found. This is a tight, contiguous
    stack, NOT a loose "anything within N points" search: a wide catch-all
    window pulls in unrelated nearby text (dimension callouts, elevation
    datum labels, sheet reference tags, etc.) that happens to sit above
    the scale but isn't actually part of the title.

    Handles three layouts:
      CASE 1: number is BELOW the scale (DGS / standard format)
      CASE 2: number is to the LEFT of the title (Docusign format)
      CASE 3: fallback -- title+scale found, but no number located

    NOTE: `items` must already be in DISPLAY space (see get_spans()) --
    this function does no rotation handling of its own; on a rotated page
    passing raw-space spans in here means "upward" and "left-aligned"
    don't correspond to how the text actually reads, and this silently
    finds nothing.
    """
    scales = [i for i in items if any(kw in i["text"].lower() for kw in SCALE_KEYWORDS) and len(i["text"]) < 30]

    blocks = []
    seen_scales = set()

    # Tight tolerances for walking up a stacked title: each successive line
    # must sit immediately above the previous one (near-zero gap) and be
    # left-aligned with it. This mirrors the stacking logic find_anchors()
    # already uses for the ARCH REF triplet, instead of a loose bounding
    # window that can catch unrelated text elsewhere on the sheet.
    LINE_GAP_TOL = 4.0      # max gap (pts) between bottom of one line and top of the next
    X_ALIGN_TOL = 6.0       # max x0 drift between stacked lines
    MAX_TITLE_LINES = 3     # safety cap so we don't walk into unrelated content

    for scale in scales:
        sx0, sy0, sx1, sy1 = scale["bbox"]

        raw_title_lines = []
        cursor_y = sy0   # top edge of the line we're currently trying to find something above
        cursor_x = sx0

        for _ in range(MAX_TITLE_LINES):
            candidates = [
                i for i in items
                if i != scale
                and i not in raw_title_lines
                and abs(i["bbox"][0] - cursor_x) < X_ALIGN_TOL
                and 0 <= (cursor_y - i["bbox"][3]) < LINE_GAP_TOL
                and len(i["text"]) > 1  # skip single-char noise
            ]
            if not candidates:
                break
            # If multiple candidates tie, prefer the one closest in x0.
            candidates.sort(key=lambda i: abs(i["bbox"][0] - cursor_x))
            nxt = candidates[0]
            raw_title_lines.append(nxt)
            cursor_y = nxt["bbox"][1]
            cursor_x = nxt["bbox"][0]

        if raw_title_lines:
            raw_title_lines.sort(key=lambda x: x["bbox"][1])
            merged_text = " ".join(l["text"].strip() for l in raw_title_lines)
            mx0 = min(l["bbox"][0] for l in raw_title_lines)
            my0 = min(l["bbox"][1] for l in raw_title_lines)
            mx1 = max(l["bbox"][2] for l in raw_title_lines)
            my1 = max(l["bbox"][3] for l in raw_title_lines)
            title = {"text": merged_text, "bbox": [mx0, my0, mx1, my1], "_lines": len(raw_title_lines)}
        else:
            title = None

        # -- CASE 1: Number is BELOW the scale (DGS / standard format) --------
        search_y0 = title["bbox"][1] - 20 if title else sy0 - 50
        search_y1 = sy1 + 20
        number_candidates_c1 = [
            i for i in items
            if i != scale
            and i not in (raw_title_lines if raw_title_lines else [])
            and i["bbox"][2] < sx0 + 20
            and (sx0 - i["bbox"][2]) < 150
            and i["bbox"][3] > search_y0
            and i["bbox"][1] < search_y1
            and len(i["text"]) <= 5
        ]
        number_candidates_c1.sort(key=lambda x: x["bbox"][2], reverse=True)
        number_c1 = number_candidates_c1[0] if number_candidates_c1 else None

        if number_c1:
            scale_key = (round(sx0), round(sy0))
            if scale_key not in seen_scales:
                seen_scales.add(scale_key)
                blocks.append({"number": number_c1, "title": title, "scale": scale, "case": 1})
            continue

        # -- CASE 2: Number is to the LEFT of the title (Docusign format) -----
        number_c2 = None
        if title:
            tx0, ty0, tx1, ty1 = title["bbox"]
            number_candidates_c2 = [
                i for i in items
                if i != scale
                and i not in (raw_title_lines if raw_title_lines else [])
                and i["bbox"][2] <= tx0 + 5
                and (tx0 - i["bbox"][2]) < 100
                and abs(i["bbox"][1] - ty0) < 20
                and len(i["text"]) <= 5
            ]
            number_candidates_c2.sort(key=lambda x: x["bbox"][2], reverse=True)
            number_c2 = number_candidates_c2[0] if number_candidates_c2 else None

            if number_c2:
                scale_key = (round(sx0), round(sy0))
                if scale_key not in seen_scales:
                    seen_scales.add(scale_key)
                    blocks.append({"number": number_c2, "title": title, "scale": scale, "case": 2})

        # -- CASE 3: Fallback -- title+scale found, number missed -------------
        if title and not number_c1 and not number_c2:
            scale_key = (round(sx0), round(sy0))
            if scale_key not in seen_scales:
                seen_scales.add(scale_key)
                tx0, ty0, tx1, ty1 = title["bbox"]
                dummy_num = {"text": "UNKNOWN", "bbox": [tx0 - 30, ty0, tx0, ty1]}
                blocks.append({"number": dummy_num, "title": title, "scale": scale, "case": 3})

    return blocks

def _parse_json_robust(content: str) -> list:
    """Parse a JSON array from `content`, recovering gracefully if the
    response was truncated mid-stream (common on dense sheets with many
    drawings): find the last complete '}' and close the array there."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    last_brace = content.rfind("}")
    if last_brace != -1:
        recovered = content[: last_brace + 1]
        if "[" not in recovered:
            recovered = "[" + recovered
        recovered = recovered.rstrip() + "]"
        try:
            data = json.loads(recovered)
            print(f"  [WARN] VLM response was truncated -- recovered {len(data)} complete item(s).")
            return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse VLM response as JSON.\nRaw content (first 300 chars): {content[:300]}")


def run_vlm_fallback(client, page):
    """Last-resort title read: renders the full page as an image and asks
    Qwen-VL (via OpenRouter) to find title/number/scale text directly. Used
    when the sheet has no usable vector text at all (e.g. rasterized/scanned).

    get_pixmap() applies /Rotate by default, so the rendered image here is
    already correctly oriented DISPLAY space -- the returned bbox is scaled
    against page.rect (also display space), so no extra rotation handling
    is needed in this function specifically.
    """
    print("  Vector text failed. Falling back to Qwen-VL via OpenRouter (this may take 10-20 seconds)...")

    zoom = 120 / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    from PIL import Image
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64_image = _b64_encode(buf.getvalue())

    prompt = """
You are an expert at reading architectural drawings.

Your task is to extract ONLY the drawing titles from this sheet.

A drawing title satisfies ALL of the following:

1. It is the main label describing a drawing.
2. It is usually located directly BELOW the drawing.
3. It is often ABOVE the "SCALE:" text, but some schematic/diagram drawings do NOT have a scale. Extract them anyway!
4. It is larger and bolder than dimension texts.
5. Ignore dimensions, notes, symbols, callouts, sheet notes, revision tables, project title, company logo, and consultant information.
6. Ignore "SCALE", "SHEET", "G0.3", general page drawing numbers, and coordinates.
7. Return one title per drawing.
8. CRITICAL: NEVER merge adjacent titles! Each drawing panel has EXACTLY ONE title. Whether drawings are side-by-side in a row, stacked vertically in a column, or arranged in a grid, you MUST return one separate JSON object per drawing. For example, if there are 3 columns and 2 rows, you must return 6 separate JSON objects.
9. CRITICAL: DO NOT SKIP ANY DRAWINGS! Scan the entire page exhaustively from top to bottom, left to right. Every single drawing with a title MUST be extracted!
10. CRITICAL: IGNORE material finish legends/callouts (e.g. "WD-01 WOOD FINISH", "CP-02 CARPET", "FL-03 FLOORING"). These are material specifications, NOT drawing titles.
11. CRITICAL: IGNORE general page headers and general sheet titles (e.g. "GENERAL NOTES", "SYMBOLS", "CABLE SCHEDULE"). Valid drawing titles often start with words like "SECTION", "PLAN", "ELEVATION", or "DETAIL". They usually have a drawing number directly next to them, but NOT ALWAYS. Extract the title even if the drawing number is missing.

If you cannot clearly read a title, number, or scale, use an empty string for
that field rather than guessing a plausible value.

Output JSON only:

[
  {
    "number": "The drawing number (e.g. 1, A, 3/A4)",
    "title": "The descriptive title text",
    "scale": "The scale if it exists, otherwise empty string",
    "bbox":[x1,y1,x2,y2]
  }
]
"""

    try:
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                ],
            }],
            temperature=0.1,
            max_tokens=4000,  # Dense sheets (20+ drawings) can exceed 1000 tokens
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        data = _parse_json_robust(content)

        blocks = []
        for item in data:
            if "bbox" in item:
                x1, y1, x2, y2 = item["bbox"]
                ymin, xmin, ymax, xmax = min(y1, y2), min(x1, x2), max(y1, y2), max(x1, x2)
            elif "bbox_1000" in item:
                ymin, xmin, ymax, xmax = item["bbox_1000"]
            else:
                continue

            scale_factor = 1000.0 if xmax > 1.0 else 1.0

            pdf_w, pdf_h = page.rect.width, page.rect.height
            pdf_x0 = (xmin / scale_factor) * pdf_w
            pdf_y0 = (ymin / scale_factor) * pdf_h
            pdf_x1 = (xmax / scale_factor) * pdf_w
            pdf_y1 = (ymax / scale_factor) * pdf_h

            blocks.append({
                "number": {"text": item.get("number", "UNKNOWN"), "bbox": [pdf_x0 - 20, pdf_y0, pdf_x0, pdf_y1]},
                "title": {"text": item.get("title", "UNKNOWN"), "bbox": [pdf_x0, pdf_y0, pdf_x1, pdf_y1]},
                "scale": {"text": item.get("scale", ""), "bbox": [pdf_x0, pdf_y1, pdf_x1, pdf_y1 + 15]},
                "case": "VLM",
            })

        print(f"  VLM extracted {len(blocks)} title blocks.")
        for b in blocks:
            print(f"    -> [{b['number']['text']}] '{b['title']['text'][:45]}'")
        return blocks

    except Exception as e:
        print(f"  [ERROR] VLM extraction failed: {e}")
        return []


# --------------------------------------------------------------------------
# 1. Text extraction
# --------------------------------------------------------------------------

def get_spans(page, matrices=None):
    """Return a flat list of text spans with bbox, font size, and text.

    PATCHED (rotation fix, v2): bboxes are transformed into DISPLAY space
    via an AUTO-DETECTED matrix (see _text_matrices() / _auto_matrices()
    at the top of this file) rather than an assumed one -- get_text() does
    not reliably return raw, unrotated coordinates on every PyMuPDF
    version, and assuming so double-transforms already-correct data (see
    PATCH NOTES (rotation fix, v1) at the top of this file for what that
    looked like). Pass `matrices` (the 3-tuple from _text_matrices()) if
    the caller already computed it for this page, to avoid re-sampling.
    """
    if matrices is None:
        to_display, _, _ = _text_matrices(page)
    else:
        to_display, _, _ = matrices
    d = page.get_text("dict")
    spans = []
    for block in d["blocks"]:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"]
                if text.strip():
                    spans.append(
                        {
                            "bbox": _bbox_to_display(span["bbox"], to_display),
                            "size": round(span["size"], 2),
                            "text": text,
                        }
                    )
    return spans


# --------------------------------------------------------------------------
# 2. Sheet frame detection (outer border + title-block divider)
# --------------------------------------------------------------------------

def find_border_and_divider(page, matrices=None):
    """
    Detect the sheet's outer border rectangle and the vertical divider line
    that separates the drawing field from the title block, using the
    page's vector line-work. Falls back to the page's mediabox if nothing
    is found (e.g. a page with no border drawn).

    PATCHED (rotation fix, v2): line endpoints are transformed into
    DISPLAY space via an AUTO-DETECTED matrix (see _drawing_matrices() at
    the top of this file) before the axis-alignment checks below, instead
    of an assumed one -- see PATCH NOTES (rotation fix, v1/v2) at the top
    of this file for why an assumed transform is unsafe. Pass `matrices`
    (the 3-tuple from _drawing_matrices()) if the caller already computed
    it for this page. The fallback branch returns page.rect directly,
    which is always display space, so it stays consistent with the rest
    of this function regardless of what was detected above.
    """
    if matrices is None:
        to_display, _, _ = _drawing_matrices(page)
    else:
        to_display, _, _ = matrices
    drawings = page.get_drawings()
    long_lines = []
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p0, p1 = item[1] * to_display, item[2] * to_display
                length = ((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5
                if length > 500:
                    long_lines.append((p0, p1, length))

    xs_vert = [p0.x for p0, p1, _ in long_lines if abs(p0.x - p1.x) < 0.1]
    ys_horz = [p0.y for p0, p1, _ in long_lines if abs(p0.y - p1.y) < 0.1]

    rect = page.rect
    if not xs_vert or not ys_horz:
        # No frame detected -- use the raw page bounds as a fallback.
        return rect.x0, rect.y0, rect.x1, rect.y1, rect.x1

    left = min(xs_vert)
    right = max(xs_vert)
    top = min(ys_horz)
    bottom = max(ys_horz)

    xs_sorted = sorted(set(round(x, 1) for x in xs_vert))
    # The divider before the title block is the second-largest distinct
    # vertical line (the largest is the outer-right border itself).
    divider = xs_sorted[-2] if len(xs_sorted) >= 2 else right

    return left, top, right, bottom, divider


# --------------------------------------------------------------------------
# 3. Anchor (title/scale/arch-ref triplet) detection -- PRIMARY
# --------------------------------------------------------------------------

def _nearest_stack_candidate(candidates, ref_x0, ref_y0):
    """
    Among `candidates` (spans already filtered to be within SOME tolerance
    window of a reference x0/y0), return the one that is actually closest
    -- by combined x/y distance -- to that reference point.

    PATCHED (nearest-match fix): find_anchors() used to take cand_list[0],
    i.e. whichever span happened to come first in page.get_text()'s
    natural (top-to-bottom, left-to-right-ish) ordering, rather than
    whichever span was actually closest to the anchor it was being
    matched against. On a sheet where an unrelated callout (e.g. an
    "ARCHITECT NOTE: ..." box) happens to sit inside the same loose
    tolerance window as the real title/scale line, "first in list order"
    can silently pick the WRONG span, or a genuinely correct match can
    lose out to a spurious one purely due to extraction order. Sorting by
    actual distance and taking the closest one is a small, cheap fix that
    removes this class of order-dependent misdetection.
    """
    return min(candidates, key=lambda s: abs(s["bbox"][0] - ref_x0) + abs(s["bbox"][3] - ref_y0))


def find_anchors(spans, tol_y=3.5, tol_x=3.5, title_min_size=9.0,
                  widen_tol_y=12.0, widen_tol_x=6.0, verbose=True):
    """
    Find every (title, scale, arch_ref) triplet stacked at the same x0.
    Returns a list of dicts with the combined bbox and the three text
    strings.

    PATCHED (nearest-match + self-widening fix): two independent
    improvements over the previous version, aimed at the class of bug
    where a real triplet is silently dropped with zero diagnostic:

    1. NEAREST MATCH, NOT FIRST MATCH. Candidate scale/title spans are now
       chosen by actual proximity to the reference span (see
       _nearest_stack_candidate()) instead of "whichever the tolerance
       filter happened to keep first" -- removing an order-dependent
       failure mode where a nearby unrelated callout could out-rank the
       real match, or a real match could simply be skipped because
       something else satisfied the filter first.

    2. SELF-WIDENING RETRY WITH DIAGNOSTICS. If an ARCH REF span fails to
       find a scale/title match at the normal (tol_x, tol_y) tolerance, we
       retry that ONE arch_ref at a wider tolerance (widen_tol_x,
       widen_tol_y) before giving up on it entirely. This catches title
       blocks whose line-pitch is genuinely a bit looser than the sheet's
       usual template (observed: ELEVATION-type blocks on this drawing
       set sit ~5-8pt looser than PLAN/PLAN SECTION blocks, just outside
       the normal tolerance) without loosening the tolerance for every
       anchor on every page, which would raise the risk of the
       cross-talk failure mode nearest-match already had to fix. Any
       arch_ref that still fails even at the widened tolerance -- or
       that fails outright with no scale/title spans found nearby at all
       -- is reported via a printed warning naming the arch_ref text and
       its position, so a future miss shows up in the console log instead
       of vanishing with no trace (this is what made both prior misses on
       this drawing set hard to diagnose from the output alone).

    NOTE: `spans` must already be in DISPLAY space (see get_spans()) --
    this function itself does no rotation handling; that's intentional,
    since it means this and every other geometric function below it never
    needs to know or care whether the page was rotated.
    """
    archrefs = [s for s in spans if re.match(r"(?i)^arch\.?\s*ref", s["text"].strip())]
    scales = [s for s in spans if re.match(r"(?i)^scale", s["text"].strip())]

    def _try_match(ar, tx, ty):
        ax0, ay0, ax1, ay1 = ar["bbox"]
        cand_scale = [
            s for s in scales
            if abs(s["bbox"][0] - ax0) < tx and abs(s["bbox"][3] - ay0) < ty
        ]
        if not cand_scale:
            return None
        sc = _nearest_stack_candidate(cand_scale, ax0, ay0)
        sx0, sy0, sx1, sy1 = sc["bbox"]

        cand_title = [
            s for s in spans
            if abs(s["bbox"][0] - sx0) < tx
            and abs(s["bbox"][3] - sy0) < ty
            and s["size"] > title_min_size
        ]
        if not cand_title:
            return None
        ti = _nearest_stack_candidate(cand_title, sx0, sy0)
        tx0, ty0, tx1, ty1 = ti["bbox"]

        # view_number: the circled letter/number sitting to the left of
        # this title/scale/arch_ref stack (e.g. "A", "B" -- NOT the ARCH
        # REF text, which is a cross-reference to a different sheet).
        view_number = _find_view_number(spans, tx0, ty0, exclude=(ti, sc, ar))

        return {
            "title": ti["text"].strip(),
            "scale": sc["text"].strip(),
            "arch_ref": ar["text"].strip(),
            "view_number": view_number,
            "x0": min(tx0, sx0, ax0),
            "x1": max(tx1, sx1, ax1),
            "y0": ty0,
            "y1": ay1,
            "method": "arch_ref_triplet",
        }

    anchors = []
    for ar in archrefs:
        result = _try_match(ar, tol_x, tol_y)
        if result is not None:
            anchors.append(result)
            continue

        # Normal tolerance failed -- retry this one ARCH REF at a wider
        # tolerance before giving up, and report either way so a miss is
        # visible in the log.
        widened = _try_match(ar, widen_tol_x, widen_tol_y)
        if widened is not None:
            widened["method"] = "arch_ref_triplet_widened"
            anchors.append(widened)
            if verbose:
                ax0, ay0 = ar["bbox"][0], ar["bbox"][1]
                print(
                    f"  [anchor] '{ar['text'].strip()}' at ({ax0:.0f},{ay0:.0f}) "
                    f"only matched at widened tolerance -- title block line "
                    f"pitch here is looser than the sheet default."
                )
        elif verbose:
            ax0, ay0 = ar["bbox"][0], ar["bbox"][1]
            print(
                f"  [anchor] [MISS] '{ar['text'].strip()}' at ({ax0:.0f},{ay0:.0f}) "
                f"found NO scale/title match even at widened tolerance -- "
                f"this drawing will be picked up (if at all) only by the "
                f"generic/VLM fallback, or dropped entirely."
            )

    return anchors


# --------------------------------------------------------------------------
# 3b. Anchor detection -- FALLBACK (generic scale-keyword title blocks,
#     with an optional VLM read as a last resort)
# --------------------------------------------------------------------------

def _generic_blocks_to_anchors(title_blocks, method):
    """
    Convert the {number, title, scale} dicts produced by
    group_title_blocks() / run_vlm_fallback() into the same anchor shape
    find_anchors() produces, so the rest of the pipeline (row/column
    clustering, tessellation) doesn't need to know which detector found
    them.

    PATCHED (sheet/view numbering fix): there's no ARCH REF concept in
    this generic mode, so the detected drawing "number" used to be stored
    directly in the anchor's arch_ref field. That collided with the real
    meaning of arch_ref (a cross-reference to a different sheet, only
    produced by the primary triplet detector) and made downstream sheet
    numbering ambiguous -- see find_sheet_number() / _find_view_number()
    at the top of this file. Now the detected number goes into
    view_number, and arch_ref is left empty for generic-mode anchors
    since no true ARCH REF exists here.
    """
    anchors = []
    for tb in title_blocks:
        title_span = tb.get("title")
        scale_span = tb.get("scale")
        number_span = tb.get("number")

        title_text = (title_span or {}).get("text", "").strip()
        if not title_text or title_text.upper() == "UNKNOWN":
            continue

        bboxes = [s["bbox"] for s in (number_span, title_span, scale_span) if s]
        if not bboxes:
            continue
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)

        number_text = (number_span or {}).get("text", "").strip()
        if number_text.upper() == "UNKNOWN":
            number_text = ""

        anchors.append(
            {
                "title": title_text,
                "scale": (scale_span or {}).get("text", "").strip(),
                "arch_ref": "",              # no true ARCH REF in generic mode
                "view_number": number_text,  # the detected drawing number
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "method": method,
            }
        )
    return anchors


def find_anchors_fallback(page, spans, vlm_client=None):
    """
    PATCHED (fallback gating fix): this used to run ONLY when find_anchors()
    (the ARCH REF triplet pattern) found NOTHING on the page. It is now
    called unconditionally on every page and its results are MERGED with
    find_anchors()'s results via merge_anchor_sets() -- see PATCH NOTES
    (fallback gating fix) at the top of this file. That means this
    function's job is now two-fold:
      1. Sole detector on sheets where find_anchors() finds nothing at all
         (original behavior, unchanged).
      2. Supplementary detector on sheets where find_anchors() found SOME
         anchors but missed one or more views whose title-block layout
         didn't fit the tight triplet pattern -- those extra finds get
         added in by the merge step, while anchors that duplicate what
         find_anchors() already found are dropped as overlaps.

    Tries the generic vector-text title-block grouper first; if that also
    finds nothing and a VLM client was provided, falls back once more to a
    full-page Qwen-VL read.
    """
    generic_blocks = group_title_blocks(spans)
    anchors = _generic_blocks_to_anchors(generic_blocks, method="generic_scale_keyword")
    if anchors:
        return anchors

    if vlm_client is not None:
        vlm_blocks = run_vlm_fallback(vlm_client, page)
        anchors = _generic_blocks_to_anchors(vlm_blocks, method="vlm")
        return anchors

    return []


# --------------------------------------------------------------------------
# 3c. Anchor set merging (NEW -- fallback gating fix)
# --------------------------------------------------------------------------

def _anchors_overlap(a, b, tol=15.0):
    """
    True if anchor bboxes a and b spatially overlap (within `tol` points of
    slack on each side). Used by merge_anchor_sets() to decide whether a
    fallback-detected anchor is a duplicate of one the primary detector
    already found (same physical title block, possibly with a slightly
    different exact bbox depending on which detector found it) versus a
    genuinely different view the primary detector missed.
    """
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    if ax1 + tol < bx0 or bx1 + tol < ax0:
        return False
    if ay1 + tol < by0 or by1 + tol < ay0:
        return False
    return True


def merge_anchor_sets(primary, fallback):
    """
    Combine the primary (ARCH REF triplet) anchors with the fallback
    (generic scale-keyword / VLM) anchors, keeping every primary anchor
    and adding only the fallback anchors that don't spatially overlap one
    already present. This is what lets a view whose title-block layout
    narrowly missed find_anchors()'s tolerance -- but which the looser
    fallback detector did catch -- get added to the page instead of
    silently dropped (see PATCH NOTES (fallback gating fix) at the top of
    this file).

    Primary anchors always win on overlap (their arch_ref field is
    generally more reliable than the fallback's guessed drawing number),
    so this only ever ADDS views, never replaces or removes one the
    primary detector already found.
    """
    merged = list(primary)
    added = 0
    for fb in fallback:
        if not any(_anchors_overlap(fb, p) for p in primary):
            merged.append(fb)
            added += 1
    return merged, added


# --------------------------------------------------------------------------
# 4. Row/column clustering + content-aware tessellation
# --------------------------------------------------------------------------

def cluster_rows(anchors, tol=30.0):
    """Group anchors into rows by similar y0, then sort each row by x0."""
    anchors = sorted(anchors, key=lambda a: a["y0"])
    rows = []
    for a in anchors:
        placed = False
        for row in rows:
            if abs(row[0]["y0"] - a["y0"]) < tol:
                row.append(a)
                placed = True
                break
        if not placed:
            rows.append([a])
    rows.sort(key=lambda r: r[0]["y0"])
    for row in rows:
        row.sort(key=lambda a: a["x0"])
    return rows


def get_content_items(page, exclude_max_dim=450):
    """
    Every piece of "ink" on the page (individual vector-path primitives +
    text span bounding boxes), excluding only genuinely large single
    primitives such as the outer sheet border and full-width table rules,
    which would otherwise swamp the gap-finding logic below.

    PATCHED: page.get_drawings() returns one entry per *path*, and a path
    can be a compound object containing many sub-items (e.g. a whole
    ceiling grid or a multi-room outline drawn as one continuous path).
    The old version filtered using the path's aggregate d["rect"], which
    threw away every sub-item inside a large compound path even though
    none of the individual primitives were actually that big -- silently
    deleting real content (see PATCH NOTES at the top of this file).

    Fix: decompose each path into its individual items (lines, rects,
    curves) and compute each one's own small bbox, then filter at that
    primitive level. Only a single primitive that's actually large (a true
    border line or table rule) gets excluded now.

    PATCHED (rotation fix, v2): every point/bbox is transformed into
    DISPLAY space before being measured/filtered/returned, via
    AUTO-DETECTED matrices (see _drawing_matrices()/_text_matrices() at
    the top of this file) -- one for the get_drawings() primitives, and a
    SEPARATE one for the get_text() spans mixed in below, since those two
    extraction functions aren't guaranteed to share the same rotation
    behavior even on the same PyMuPDF version. This also means
    exclude_max_dim (450pt) is now compared against the correct,
    as-displayed extent of each primitive rather than its raw-space
    extent.
    """
    to_display_draw, _, _ = _drawing_matrices(page)
    to_display_text, _, _ = _text_matrices(page)
    to_display = to_display_draw
    items = []
    for d in page.get_drawings():
        for it in d["items"]:
            op = it[0]
            pts = []
            if op == "l":  # line: (op, p0, p1)
                pts = [it[1] * to_display, it[2] * to_display]
            elif op == "re":  # rect: (op, Rect, ...)
                r = it[1]
                pts = [fitz.Point(r.x0, r.y0) * to_display, fitz.Point(r.x1, r.y1) * to_display]
            elif op in ("c", "qu"):  # curve/quad: bbox over all points
                pts = [p * to_display for p in it[1:] if isinstance(p, fitz.Point)]
            else:
                continue
            if not pts:
                continue
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w, h = x1 - x0, y1 - y0
            if w > exclude_max_dim or h > exclude_max_dim:
                continue  # a genuinely large single primitive -- a border/rule
            if w <= 0 and h <= 0:
                continue  # degenerate point
            items.append((x0, y0, x1, y1))

    d = page.get_text("dict")
    for b in d["blocks"]:
        if "lines" not in b:
            continue
        for l in b["lines"]:
            for s in l["spans"]:
                if s["text"].strip():
                    items.append(_bbox_to_display(s["bbox"], to_display_text))
    return items


def find_column_split(all_items, y0, y1, xa1, xb0, min_gap=30.0, merge_tol=5.0, search_pad=20.0):
    """
    Find the true horizontal boundary between two side-by-side drawings.

    We only look at ink strictly between the two titles (xa1 = right edge
    of the left title, xb0 = left edge of the right title), which is where
    the real separation must live. Nearby ink is merged into blobs first
    (so a multi-word callout doesn't look like several separate objects),
    then we take the RIGHT-MOST gap that clears `min_gap` points. Taking
    the right-most (rather than simply the largest) gap matters because a
    drawing's own far-flung callout text can leave a bigger blank strip
    between itself and its own body than the strip that actually separates
    it from the next drawing -- but that callout gap always occurs before
    the true inter-drawing gap, never after it.
    """
    items_in_zone = [
        it for it in all_items
        if it[1] < y1 and it[3] > y0 and it[0] >= xa1 - search_pad and it[2] <= xb0 + search_pad
    ]
    intervals = sorted((it[0], it[2]) for it in items_in_zone)

    merged = []
    for x0, x1 in intervals:
        if merged and x0 - merged[-1][1] < merge_tol:
            merged[-1] = (merged[-1][0], max(merged[-1][1], x1))
        else:
            merged.append((x0, x1))

    if len(merged) < 2:
        # No internal structure to reason about -- just split the empty zone.
        return (xa1 + xb0) / 2

    gaps = [
        (merged[i + 1][0] - merged[i][1], (merged[i][1] + merged[i + 1][0]) / 2)
        for i in range(len(merged) - 1)
    ]
    qualifying = [g for g in gaps if g[0] >= min_gap]
    if qualifying:
        return qualifying[-1][1]  # right-most gap that's "real"
    gaps.sort(reverse=True)
    return gaps[0][1]  # fallback: nothing cleared the threshold, take the biggest
def find_vertical_dividers(page, sheet_top, sheet_bottom, min_coverage=0.9, tol=0.75, matrices=None):
    """
    Return sorted x-positions of vertical rule lines that span (nearly) the
    full height of the sheet's content area (sheet_top -> sheet_bottom).

    Used for two things now:
      (a) bounding a single-anchor row against neighboring non-drawing
          content that shares the row's vertical band but has no
          title/scale anchor of its own (original use), and
      (b) partitioning the WHOLE content field into independent
          column-groups before row clustering even starts, so an
          unrelated stacked side-column can't inject phantom row
          boundaries into a different grid (see find_column_group_bounds
          and the PATCH NOTES at the top of this file).

    PATCHED: the previous version accepted ANY vertical line overlapping
    the row (min_length=50pt), which is not selective enough -- CAD column-
    grid extension lines (the leader dropped straight down from a grid
    bubble like W1/W2/... at the top of the sheet) run the *entire* height
    of the drawing and are visually indistinguishable from a true divider
    by length/coverage alone. Picking the nearest such line as a boundary
    silently clipped almost the entire drawing (see screenshot: box only
    covered the first ~20% of the sheet width).

    Fix: (1) require near-full SHEET height coverage (not just the row) --
    interior CAD content essentially never runs the complete sheet height,
    so this alone rules out most walls/dimension lines/callouts; (2) only
    consider SOLID lines. Grid extension lines are conventionally drawn
    dashed, while true panel dividers (sheet border, title-block divider,
    the vertical rules separating General Notes / References / Keynotes /
    Legend, or a side-column of independent views) are solid -- this is
    what actually tells the two apart, since both can span the full sheet
    height by design.

    PATCHED (rotation fix, v2): line endpoints are transformed into
    DISPLAY space via an AUTO-DETECTED matrix (see _drawing_matrices() at
    the top of this file) before the near-vertical test and the
    sheet_top/sheet_bottom coverage check, both of which are expressed in
    (and only make sense in) display space. `sheet_top`/`sheet_bottom`
    themselves must be display-space values, which they are as long as
    they came from find_border_and_divider(). Pass `matrices` (the
    3-tuple from _drawing_matrices()) if the caller already computed it
    for this page.
    """
    sheet_height = sheet_bottom - sheet_top
    if sheet_height <= 0:
        return []

    if matrices is None:
        to_display, _, _ = _drawing_matrices(page)
    else:
        to_display, _, _ = matrices
    xs = []
    for d in page.get_drawings():
        dashes = d.get("dashes") or ""
        if dashes not in ("", "[] 0"):
            continue  # dashed/dotted -- treat as a grid line, not a divider
        for it in d["items"]:
            if it[0] != "l":
                continue
            p0, p1 = it[1] * to_display, it[2] * to_display
            if abs(p0.x - p1.x) > tol:
                continue  # not (near-)vertical
            ytop, ybot = min(p0.y, p1.y), max(p0.y, p1.y)
            overlap = min(ybot, sheet_bottom) - max(ytop, sheet_top)
            if overlap < min_coverage * sheet_height:
                continue
            xs.append(round((p0.x + p1.x) / 2, 1))
    return sorted(set(xs))


# --------------------------------------------------------------------------
# 4b. COLUMN-GROUP PARTITIONING
# --------------------------------------------------------------------------

def find_column_group_bounds(page, content_left, content_right, content_top, content_bottom):
    """
    Split the sheet's content field into independent column-groups using
    full-height SOLID vertical divider lines (the same detector used by
    the single-anchor-row fallback below). Each group then gets its OWN
    row/column tessellation pass in compute_crop_boxes(), instead of one
    global pass across the whole sheet width.

    BUG THIS FIXES: cluster_rows() groups every anchor on the page into
    rows by y0 ALONE, across the full sheet width. compute_crop_boxes()
    then applies each row's bottom bound -- and, for multi-anchor rows, its
    ink-tightened top bound -- across the ENTIRE content width, including
    under columns whose real row structure has nothing to do with that
    row. Concretely: a sheet with a 2x2 elevation grid (E1/E2 over E3/E4)
    PLUS a narrow side column of two unrelated stacked views ("TILT-UP
    REVEAL ELEVATION" over "TYP CAST CONC. REVEAL JOINT") produces a
    phantom third "row" for the side column's first view, because its y0
    doesn't match either real row of the main grid. That phantom row's
    bottom bound then became the TOP bound for the main grid's second row
    (E3/E4) -- sheet-wide -- silently clipping the top strip (parapet
    callouts, top dimension strings) off E3 and E4, even though that
    content has nothing to do with the side column.

    Fix: find every full-height solid divider inside the content field and
    use those as hard group boundaries. Anchors are assigned to a group by
    which pair of adjacent divider x-positions their center falls between.
    Row clustering / tessellation then runs once per group, so one group's
    row boundaries can never leak into another's.

    Falls back to a single group (the whole content field, old behavior)
    if no interior dividers are found -- e.g. a plain sheet with one
    grid and no ruled sub-columns.

    Caveat: this depends on a real ruled line existing between groups (as
    DGS/Turner-style templates typically have). Two independent
    view-clusters separated only by whitespace, with no ruled divider,
    would need an ink-gap-based partition instead -- not implemented here.
    """
    dividers = find_vertical_dividers(page, content_top, content_bottom)
    interior = [d for d in dividers if content_left + 5 < d < content_right - 5]
    bounds = sorted(set([content_left] + interior + [content_right]))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def find_row_ink_bounds(all_items, anchor, y0, y1, content_left, content_right,
                         min_gap=30.0, merge_tol=5.0, pad=4.0):
    """
    Fallback for a single-anchor row when find_vertical_dividers finds no
    ruled divider lines to bound against (e.g. a sheet where a small
    drawing sits next to unrelated reference tables -- NOTES, a materials
    schedule, a plumbing schedule -- with no ruled line between them, just
    whitespace).

    Expands outward from the anchor's own x-range through the merged ink
    blobs in this row's y-band, stopping at the first real empty gap
    (>= min_gap) on each side. This keeps a small lone drawing's box tight
    around its own content instead of defaulting to the full sheet width
    and sweeping in unrelated tables that happen to share its row-band but
    actually belong contextually to a DIFFERENT (often larger) drawing
    elsewhere on the sheet -- which the naive full-width default was
    doing, and which then also starved that other drawing of its own
    reference context (see write-up: KEY PLAN row silently absorbing the
    NOTES / MATERIALS / PLUMBING SCHEDULE tables that describe fixtures
    used in the "PLAN - RESTROOM 122" drawing below it, not the key plan).

    Note: this does NOT attempt to reattach that swept-out content to the
    other drawing's box -- it simply keeps it out of the WRONG box. Content
    that sits in one drawing's row-band but belongs to another remains
    outside both boxes after this fix, which is the safer outcome.
    """
    items_in_band = [it for it in all_items if it[1] < y1 and it[3] > y0]
    intervals = sorted((it[0], it[2]) for it in items_in_band)

    merged = []
    for x0, x1 in intervals:
        if merged and x0 - merged[-1][1] < merge_tol:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    if not merged:
        return content_left, content_right

    ax0, ax1 = anchor["x0"], anchor["x1"]

    idx = None
    for i, (bx0, bx1) in enumerate(merged):
        if bx0 <= ax1 and bx1 >= ax0:
            idx = i
            break
    if idx is None:
        idx = min(range(len(merged)), key=lambda i: min(abs(merged[i][0] - ax0), abs(merged[i][1] - ax1)))

    lo, hi = idx, idx
    while lo > 0 and (merged[lo][0] - merged[lo - 1][1]) < min_gap:
        lo -= 1
    while hi < len(merged) - 1 and (merged[hi + 1][0] - merged[hi][1]) < min_gap:
        hi += 1

    left = max(content_left, merged[lo][0] - pad)
    right = min(content_right, merged[hi][1] + pad)
    return left, right


def _compute_group_boxes(page, anchors, content_left, content_top, content_right, content_bottom,
                          all_items, pad=4.0, top_pad=6.0, min_gap=30.0):
    """
    Row/column tessellation for ONE column-group. This is the original
    (pre-patch) compute_crop_boxes() body, now scoped to a single group's
    anchors and x-range and taking the page-wide `all_items` ink list as a
    parameter (computed once per page by compute_crop_boxes, not once per
    group).
    """
    rows = cluster_rows(anchors)
    if not rows:
        return []

    row_bottoms = [content_top] + [max(a["y1"] for a in row) + pad for row in rows]

    boxes = []
    for ri, row in enumerate(rows):
        row_top_bound = row_bottoms[ri]
        row_bottom_bound = row_bottoms[ri + 1]
        n = len(row)

        if n == 1:
            a = row[0]
            dividers = find_vertical_dividers(page, content_top, content_bottom)
            left_candidates = [x for x in dividers if content_left - 2 <= x <= a["x0"] + 2]
            right_candidates = [x for x in dividers if a["x1"] - 2 <= x <= content_right + 2]

            if left_candidates or right_candidates:
                left_bound = max(left_candidates) if left_candidates else content_left
                right_bound = min(right_candidates) if right_candidates else content_right
            else:
                # No ruled divider lines on this sheet -- fall back to
                # ink-blob expansion so unrelated same-row-band content
                # (e.g. reference tables belonging to a different drawing)
                # doesn't get swept into a lone small drawing's box.
                left_bound, right_bound = find_row_ink_bounds(
                    all_items, a, row_top_bound, row_bottom_bound,
                    content_left, content_right, min_gap=min_gap,
                )
            col_lefts = [left_bound, right_bound]
        else:
            col_lefts = [content_left]
            for ci in range(n - 1):
                split = find_column_split(
                    all_items, row_top_bound, row_bottom_bound,
                    row[ci]["x1"], row[ci + 1]["x0"], min_gap=min_gap,
                )
                col_lefts.append(split)
            col_lefts.append(content_right)

        for ci, a in enumerate(row):
            left = col_lefts[ci]
            right = col_lefts[ci + 1]

            if n == 1:
                # Single-anchor row: still nothing to validate a tightened
                # top guess against, so leave the top edge at the row
                # bound (see earlier patch note).
                top = row_top_bound
            else:
                col_items_above = [
                    it for it in all_items
                    if it[0] >= left - 1 and it[2] <= right + 1
                    and it[1] >= row_top_bound - 1 and it[3] <= a["y0"] + 1
                ]
                if col_items_above:
                    ink_top = min(it[1] for it in col_items_above)
                    top = max(row_top_bound, ink_top - top_pad)
                else:
                    top = row_top_bound

            bottom = row_bottom_bound
            boxes.append(
                {
                    "title": a["title"],
                    "scale": a["scale"],
                    "arch_ref": a["arch_ref"],
                    "view_number": a.get("view_number"),
                    "method": a.get("method", "arch_ref_triplet"),
                    "rect": (left, top, right, bottom),
                }
            )
    return boxes


def compute_crop_boxes(page, anchors, content_left, content_top, content_right, content_bottom,
                        pad=4.0, top_pad=6.0, min_gap=30.0):
    """
    Compute one bounding box per anchor.

    First partitions the content field into independent column-groups
    (find_column_group_bounds), then runs row/column tessellation
    SEPARATELY within each group -- see find_column_group_bounds()'s
    docstring for the bug this fixes (a stacked side-column injecting a
    phantom row boundary that clipped an unrelated grid elsewhere on the
    sheet).

    All coordinates in and out of this function (content_left/top/right/
    bottom, anchors, and the returned box rects) are DISPLAY space -- see
    PATCH NOTES (rotation fix) at the top of the file.
    """
    if not anchors:
        return []

    all_items = get_content_items(page)
    group_bounds = find_column_group_bounds(page, content_left, content_right, content_top, content_bottom)

    boxes = []
    for gl, gr in group_bounds:
        group_anchors = [
            a for a in anchors
            if gl - 1 <= (a["x0"] + a["x1"]) / 2 <= gr + 1
        ]
        if not group_anchors:
            continue
        boxes.extend(
            _compute_group_boxes(
                page, group_anchors, gl, content_top, gr, content_bottom,
                all_items, pad=pad, top_pad=top_pad, min_gap=min_gap,
            )
        )
    return boxes

# --------------------------------------------------------------------------
# 5. Bounding-box drawing (replaces the old per-anchor cropping step)
# --------------------------------------------------------------------------

def draw_boxes_on_page(page, boxes, label=True):
    """
    Draw a rectangle for every computed box directly onto the page, plus
    an optional small text label showing the detected title. This mutates
    the in-memory `page` (adds page-content drawing ops) -- fine here since
    each page is only ever visited once and we never save the PDF back,
    only rasterize it afterwards.

    PATCHED (rotation fix, v2): `boxes[i]["rect"]` is in DISPLAY space
    (that's the space every detector/tessellation function above operates
    in), but page.new_shape() draws directly into whatever RAW content-
    stream space get_drawings() reads from -- so this reuses the SAME
    auto-detected drawings matrix (_drawing_matrices()) as
    find_border_and_divider()/get_content_items(), converting each rect
    back with its `to_raw` before drawing. Critically, if that detection
    found get_drawings() was already display-space (`rotated=False`),
    `to_raw` is the identity and nothing extra is applied here either --
    avoiding the double-transform bug described in PATCH NOTES (rotation
    fix, v1) at the top of this file. Label text's `rotate=` is likewise
    only set to counter-rotate the glyphs when `rotated` is True;
    otherwise it stays 0.
    """
    _, to_raw, rotated = _drawing_matrices(page)
    label_rotate = (360 - page.rotation) % 360 if rotated else 0

    shape = page.new_shape()
    for b in boxes:
        disp_rect = fitz.Rect(b["rect"])
        if disp_rect.is_empty or disp_rect.width <= 1 or disp_rect.height <= 1:
            continue

        raw_rect = fitz.Rect(_bbox_to_display(
            (disp_rect.x0, disp_rect.y0, disp_rect.x1, disp_rect.y1), to_raw
        ))
        shape.draw_rect(raw_rect)
        shape.finish(color=BBOX_COLOR, width=BBOX_WIDTH, fill=None)

        if label and b["title"]:
            label_point_disp = fitz.Point(disp_rect.x0 + 2, disp_rect.y0 + LABEL_SIZE + 1)
            label_point_raw = label_point_disp * to_raw
            shape.insert_text(
                label_point_raw,
                b["title"][:60],
                fontsize=LABEL_SIZE,
                color=LABEL_COLOR,
                rotate=label_rotate,
            )
    shape.commit()


# --------------------------------------------------------------------------
# 5b. Per-drawing crop export (NEW)
# --------------------------------------------------------------------------

def _fs_safe(text):
    """Sanitize a value for use as a folder/file name component. Keeps
    dots (sheet numbers like '1.12' rely on them) but strips anything the
    filesystem would choke on."""
    s = re.sub(r'[\\/*?:"<>|]', "-", text.strip())
    s = re.sub(r"\s+", "_", s)
    return s if s else "UNK"


def crop_and_save_boxes(page, boxes, crop_root, pdf_name, sheet_number, pno,
                         zoom=2.5, view_counter=None):
    """
    Render and save one clean PNG per box into:
        {crop_root}/{pdf_name}/{sheet_number}/{sheet_number}_{view_number}.png

    `crop_root` is ARCH_CROP_FOLDER or SUBMITTAL_CROP_FOLDER (decided once
    per PDF by classify_pdf_type()). `sheet_number` is this PAGE's own
    sheet number (found once per page by find_sheet_number()) -- every box
    on this page shares it. `view_number` is per-box (b["view_number"],
    set by find_anchors()/_generic_blocks_to_anchors() -- the circled
    letter/number next to that specific view, NOT the ARCH REF).

    Must be called BEFORE draw_boxes_on_page() mutates the page with the
    red rectangles/labels, so these crops come out clean.

    `view_counter` is a dict passed in by the caller and shared across the
    whole PDF (not just this page) -- used only as a fallback sequential
    counter, keyed by sheet_number, for views where no view_number could
    be found near the anchor (rare -- e.g. a generic/VLM-detected view
    with no legible bubble number at all).

    Returns a list of saved relative paths (sheet_number/filename.png),
    one per box, in the same order as `boxes`.
    """
    if view_counter is None:
        view_counter = {}

    sheet_number_safe = _fs_safe(sheet_number)
    sheet_dir = crop_root / pdf_name / sheet_number_safe
    sheet_dir.mkdir(parents=True, exist_ok=True)

    mat = fitz.Matrix(zoom, zoom)
    rel_paths = []
    for b in boxes:
        rect = fitz.Rect(b["rect"])
        pix = page.get_pixmap(matrix=mat, clip=rect)

        view_number = (b.get("view_number") or "").strip()
        if not view_number:
            view_counter[sheet_number_safe] = view_counter.get(sheet_number_safe, 0) + 1
            view_number = str(view_counter[sheet_number_safe])
            print(
                f"  [crop] '{b['title']}' on sheet {sheet_number_safe} has no "
                f"detected view_number -- using sequential fallback '{view_number}'."
            )
        view_number_safe = _fs_safe(view_number)

        fname = f"{sheet_number_safe}_{view_number_safe}.png"
        fpath = sheet_dir / fname
        n = 1
        while fpath.exists():
            # Two views on the same sheet resolved to the same view_number
            # (shouldn't normally happen, but don't silently overwrite).
            fname = f"{sheet_number_safe}_{view_number_safe}_{n}.png"
            fpath = sheet_dir / fname
            n += 1

        pix.save(str(fpath))
        rel_paths.append(f"{sheet_number_safe}/{fname}")
    return rel_paths


# --------------------------------------------------------------------------
# 6. Driver
# --------------------------------------------------------------------------

def slugify(text, max_len=40):
    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return s[:max_len] if s else "untitled"


def process_pdf(pdf_path, out_dir, zoom=2.5, pages=None, pad=4.0, margin=3.0, min_gap=30.0,
                 use_vlm=False, label=True, filter_str=None, doc_type=None):
    doc = fitz.open(pdf_path)

    # Automatically create a subfolder using the PDF's filename (without extension)
    pdf_name = Path(pdf_path).stem

    # ARCH vs SUBMITTAL classification -- decides whether crops land under
    # ARCH_CROP_FOLDER or SUBMITTAL_CROP_FOLDER. `doc_type` can be forced
    # via --doc-type; otherwise it's inferred from the filename.
    doc_type = doc_type or classify_pdf_type(pdf_path)
    crop_root_name = "arch_crop" if doc_type == "arch" else "submittal_crop"
    print(f"[{pdf_name}] classified as doc_type='{doc_type}' -> {crop_root_name}/")

    crop_root = Path(out_dir) / crop_root_name
    out_dir = crop_root / pdf_name
    
    if doc_type == "arch" and out_dir.exists() and any(out_dir.iterdir()):
        print(f"Skipping {pdf_name}: Architecture crops already exist in {out_dir}. Skipping redundant extraction.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    crop_root.mkdir(parents=True, exist_ok=True)

    # Shared across the whole PDF (not reset per page) -- only used as a
    # fallback sequential suffix for views where no view_number could be
    # detected near the anchor at all. See crop_and_save_boxes().
    view_counter = {}

    # Only spin up an OpenRouter client if the VLM fallback is enabled --
    # avoids requiring `openai` / an API key for the common case.
    vlm_client = None
    if use_vlm:
        from openai import OpenAI
        vlm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=get_api_key())

    manifest_rows = []
    page_indices = pages if pages is not None else range(len(doc))

    for pno in page_indices:
        page = doc[pno]

        # Auto-detect rotation handling once per page and print what was
        # found -- see PATCH NOTES (rotation fix, v2) at the top of this
        # file. This is the fastest way to tell, from the console output
        # alone, whether a given page's text/drawings coordinates needed
        # a real rotation transform or were already display-space.
        text_matrices = _text_matrices(page)
        draw_matrices = _drawing_matrices(page)
        print(
            f"[page {pno:03d}] rotation={page.rotation}  "
            f"text={'rotated' if text_matrices[2] else 'identity'}  "
            f"drawings={'rotated' if draw_matrices[2] else 'identity'}"
        )

        spans = get_spans(page, matrices=text_matrices)

        # This PAGE's own sheet number (e.g. "S8", "A0.06B") -- shared by
        # every view/box found on this page. NOT the same as ARCH REF,
        # which is a per-view cross-reference to a different sheet.
        sheet_number = find_sheet_number(page, spans, doc_type=doc_type)
        if not sheet_number:
            sheet_number = f"UNK_p{pno:03d}"
            print(
                f"[page {pno:03d}] [warn] could not detect a sheet number -- "
                f"crops for this page will go under '{sheet_number}'."
            )
        else:
            print(f"[page {pno:03d}] sheet_number={sheet_number}")

        # PATCHED (fallback gating fix): both detectors ALWAYS run, and
        # their results are merged (not gated on "primary found nothing").
        # See PATCH NOTES (fallback gating fix) at the top of this file
        # for the bug this fixes -- a page where the primary detector
        # found SOME but not all views used to never even try the
        # fallback, silently dropping the views it missed.
        primary_anchors = find_anchors(spans)
        fallback_anchors = find_anchors_fallback(page, spans, vlm_client=vlm_client)
        anchors, fallback_added = merge_anchor_sets(primary_anchors, fallback_anchors)

        print(
            f"[page {pno:03d}] anchors: primary={len(primary_anchors)} "
            f"fallback_added={fallback_added} total={len(anchors)}"
        )

        if not anchors:
            print(f"[page {pno:03d}] no anchors found on this page -- skipping.")
            continue

        left, top, right, bottom, divider = find_border_and_divider(page, matrices=draw_matrices)
        boxes = compute_crop_boxes(
            page,
            anchors,
            content_left=left + margin,
            content_top=top + margin,
            content_right=divider - margin,
            content_bottom=bottom - margin,
            pad=pad,
            min_gap=min_gap,
        )
        boxes = [b for b in boxes if not fitz.Rect(b["rect"]).is_empty and fitz.Rect(b["rect"]).width > 1 and fitz.Rect(b["rect"]).height > 1]
        if not boxes:
            continue

        # NEW: --filter TEXT restricts this page's boxes to only those
        # whose detected title contains TEXT (case-insensitive substring
        # match), e.g. --filter elevation. This runs AFTER detection and
        # tessellation -- filtering the anchors before tessellation would
        # corrupt the row/column math for the drawings that ARE kept,
        # since neighboring anchors are the reference points that math
        # uses (see PATCH NOTES (cropped output + title filter) at the
        # top of this file).
        if filter_str:
            before = len(boxes)
            boxes = [b for b in boxes if filter_str.lower() in b["title"].lower()]
            print(f"[page {pno:03d}] filter '{filter_str}': kept {len(boxes)}/{before} box(es)")
            if not boxes:
                print(f"[page {pno:03d}] no boxes match filter '{filter_str}' -- skipping.")
                continue

        # Save a clean, cropped PNG per drawing -- into
        # {crop_root}/{pdf_name}/{sheet_number}/{sheet_number}_{view_number}.png
        # BEFORE the boxes get drawn onto the page below.
        crop_names = crop_and_save_boxes(
            page, boxes, crop_root, pdf_name, sheet_number, pno,
            zoom=zoom, view_counter=view_counter,
        )
        for b, cname in zip(boxes, crop_names):
            b["crop_file"] = cname
            b["sheet_number"] = sheet_number

        # Draw every rectangle (+ title label) onto the page, then render
        # the whole page once -- this is the "overview" image, separate
        # from the individual crops saved just above.
        draw_boxes_on_page(page, boxes, label=label)

        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        fname = f"p{pno:03d}_bboxes.png"
        fpath = out_dir / fname
        pix.save(str(fpath))
        print(f"[page {pno:03d}] saved {fname}  ({len(boxes)} box(es) drawn, {len(crop_names)} crop(s) saved)")

        for b in boxes:
            rect = fitz.Rect(b["rect"])
            manifest_rows.append(
                {
                    "page": pno,
                    "file": fname,
                    "crop_file": b.get("crop_file", ""),
                    "doc_type": doc_type,
                    "sheet_number": b.get("sheet_number", sheet_number),
                    "view_number": b.get("view_number", ""),
                    "title": b["title"],
                    "scale": b["scale"],
                    "arch_ref": b["arch_ref"],
                    "method": b["method"],
                    "x0": round(rect.x0, 1),
                    "y0": round(rect.y0, 1),
                    "x1": round(rect.x1, 1),
                    "y1": round(rect.y1, 1),
                }
            )

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "page", "file", "crop_file", "doc_type", "sheet_number",
                "view_number", "title", "scale", "arch_ref", "method",
                "x0", "y0", "x1", "y1",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    if doc_type == "submittal":
        meta_base = out_dir.parent.parent.parent / "meta_data_json" / pdf_name
        
        sheets = {}
        for row in manifest_rows:
            crop_file = row.get("crop_file", "").replace("\\", "/")
            if not crop_file:
                continue
                
            parts = crop_file.split("/")
            if len(parts) >= 2:
                sheet = parts[-2]
            else:
                sheet = row.get("sheet_number", "unknown")
                
            if sheet not in sheets:
                sheets[sheet] = []
                
            t_up = row["title"].upper()
            v_type = "View"
            if "KEY PLAN" in t_up: v_type = "Key Plan"
            elif "PLAN SECTION" in t_up: v_type = "Plan Section"
            elif "ELEVATION" in t_up: v_type = "Elevation"
            elif "SECTION" in t_up: v_type = "Section"
            elif "PLAN" in t_up: v_type = "Plan"
            elif "DETAIL" in t_up: v_type = "Detail"
            elif "SCHEDULE" in t_up: v_type = "Schedule"
            
            source_file = f"{crop_root_name}/{pdf_name}/{crop_file}"
            
            clean_arch_ref = row.get("arch_ref", "")
            if clean_arch_ref.upper().startswith("ARCH REF:"):
                clean_arch_ref = clean_arch_ref[9:].strip()
            elif "ARCH REF:" in clean_arch_ref.upper():
                clean_arch_ref = clean_arch_ref.upper().replace("ARCH REF:", "").strip()
            
            sheets[sheet].append({
                "view_name": row["title"],
                "arch_ref": clean_arch_ref,
                "view_type": v_type,
                "source_file": source_file
            })
            
        for sheet, meta_list in sheets.items():
            sheet_meta_dir = meta_base / sheet
            sheet_meta_dir.mkdir(parents=True, exist_ok=True)
            meta_file = sheet_meta_dir / f"{sheet}.json"
            with open(meta_file, "w", encoding="utf-8") as mf:
                json.dump(meta_list, mf, indent=4)

    print(f"\nDone. {len(manifest_rows)} bounding boxes drawn across annotated pages in {out_dir}")
    print(f"Crops saved in: {crop_root / pdf_name}")
    print(f"Manifest: {manifest_path}")


def parse_pages_arg(s):
    if not s:
        return None
    pages = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return sorted(pages)


def main():
    ap = argparse.ArgumentParser(description="Draw a bounding box around every drawing view found on a CAD submittal PDF.")
    ap.add_argument("pdf", help="Path to the input PDF")
    ap.add_argument("out_dir", nargs="?", default="crop", help="Directory to write annotated page PNGs + manifest.csv into")
    ap.add_argument("--zoom", type=float, default=2.5, help="Render zoom factor (default 2.5 = ~180dpi)")
    ap.add_argument("--pages", type=str, default=None, help="0-indexed pages to process, e.g. '6,7,15' or '0-10'. Default: all pages.")
    ap.add_argument("--pad", type=float, default=4.0, help="Padding in PDF points below each title block (default 4.0)")
    ap.add_argument("--margin", type=float, default=3.0, help="Margin in PDF points inside the sheet border (default 3.0)")
    ap.add_argument("--min-gap", type=float, default=30.0, help="Minimum ink-free gap (PDF points) to count as a real column boundary (default 30.0)")
    ap.add_argument("--no-label", action="store_true", help="Don't draw the detected title text next to each bounding box.")
    ap.add_argument("--use-vlm", action="store_true",
                     help="On pages where neither the ARCH REF pattern nor the generic scale-keyword "
                          "detector finds any titles, fall back to a Qwen-VL read via OpenRouter. "
                          "Requires `pip install openai` and OPENROUTER_API_KEY set in the environment.")
    ap.add_argument("--filter", type=str, default=None,
                     help="Only keep drawings whose detected title contains this text "
                          "(case-insensitive substring match), e.g. --filter elevation. "
                          "Pages with no matching drawing are skipped entirely.")
    ap.add_argument("--doc-type", type=str, default=None, choices=["arch", "submittal"],
                     help="Force ARCH_CROP_FOLDER or SUBMITTAL_CROP_FOLDER instead of "
                          "inferring it from the PDF filename (see classify_pdf_type()).")
    args = ap.parse_args()

    process_pdf(
        args.pdf,
        args.out_dir,
        zoom=args.zoom,
        pages=parse_pages_arg(args.pages),
        pad=args.pad,
        margin=args.margin,
        min_gap=args.min_gap,
        use_vlm=args.use_vlm,
        label=not args.no_label,
        filter_str=args.filter,
        doc_type=args.doc_type,
    )


if __name__ == "__main__":
    main()