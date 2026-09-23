#!/usr/bin/env python3
"""
find_grain_arrows.py  (CORRECTED - full-page highlight version, + Style C)

Scans a folder of shop-drawing PDFs for the "grain direction" symbol used in
these submittals: a short straight shaft with a small hooked chevron
flaring off each end - e.g. the "[shaft] = INDICATE GRAIN DIRECTION" legend
glyph on the spec page.

--- Why this version differs from the original ---
Inspecting the actual PDF vector data (page.get_drawings()) shows the CAD
export draws this same symbol in up to THREE different ways depending on
where/how it was placed:

  STYLE A (e.g. the spec-page legend): three separate single-item drawings
    - one straight-line shaft, plus one bezier-curve "hook" at each end,
      whose curve start point coincides (sub-point precision) with the
      shaft's endpoint, because both were placed together as one CAD
      block/symbol instance.

  STYLE B (seen on some elevation/drawing pages): a SINGLE drawing
    containing three connected straight "l" segments - short tick, long
    shaft, short tick - as one continuous open polyline. No bezier curves
    at all. The original script's head-detector only ever looked for
    kind=="c" single-item drawings, so every Style B instance was
    invisible to it.

  STYLE C (confirmed on a real submittal's elevation sheet - the case that
    prompted this revision): the shaft and ONE tick are fused into a
    single 2-item drawing (one continuous 2-segment polyline: tick+shaft
    or shaft+tick), while the OTHER tick is a completely separate,
    independent 1-item drawing whose endpoint just happens to coincide
    with the shaft's free (non-fused) end. Confirmed by direct inspection
    of the PDF content stream: the CAD export emitted these as two
    distinct path-paint operations, not one continuous 3-segment
    polyline (that would be Style B) and with no bezier curves involved
    (that would rule out Style A). Neither existing matcher can see this
    shape: Style A requires lone single-segment shafts and lone
    *bezier-curve* heads, and this "head" is a straight line fused onto
    the shaft rather than standalone; Style B requires all three segments
    to belong to one single drawing, and here they're split across two.
    On the one real submittal page inspected, all THREE grain arrows
    present used this exact split geometry - Style A/B alone found zero
    of them.

All three styles are matched now. For Style B/C, a coincidentally similar
tick-shaft-tick shape can appear from other CAD geometry (a step/jog in a
section line, a wall offset, etc.) that happens to have similar length
ranges. The discriminator that separates real grain arrows from lookalike
jog/step geometry: on every confirmed real grain arrow, each tick bends
away from the shaft at roughly a 45-degree "hook" angle (confirmed by
measuring the real Style C instances directly: consistently ~45 degrees
at both ends), where the lookalike jog/step geometry meets the shaft at
exactly 90 degrees (a right-angle offset, not a hook) or continues
straight through at ~180 degrees. Style B's existing check (tick-to-shaft
angle 110-160 degrees, in that function's own sign convention) encodes
the same discriminator; Style C uses a physically-defined bend angle
(measured consistently as "direction back into the shaft" vs "direction
out along the tick," both taken from the shared point) so it isn't tied
to any particular path traversal order - useful since Style C's two ticks
come from independently-drawn paths that have no fixed order relative to
each other.

--- Output (unchanged from previous revision) ---
Full-page images are still rendered once per page (not per-candidate),
with every candidate on that page circled in red, using the same
draw-in-content-space-then-render-once approach as before (see
save_page_with_all_highlights for why this handles rotated sheets
correctly).

This version:
  1. Reads each page's VECTOR paths (not pixels) with PyMuPDF.
  2. Style A: shortlists a lone straight "shaft" line (15-300pt) plus lone
     bezier-curve "heads" (5-12pt chord, <=10x10pt bbox) anchored to within
     ~1.5pt of a shaft endpoint at BOTH ends.
  3. Style B: shortlists a single drawing made of exactly three connected
     black "l" segments - tick, shaft (15-300pt), tick (each tick 2-12pt) -
     where each tick meets the shaft at a diagonal (110-160 degree, in
     Style B's own convention) angle, not a right angle.
  4. Style C: shortlists a single drawing made of exactly two connected
     black "l" segments - one shaft-length (15-300pt), one tick-length
     (2-12pt) - bending at a diagonal (~25-65 degree physical bend angle)
     at their shared joint, PLUS a separate lone single-segment black "l"
     drawing (also tick-length, also ~25-65 degree physical bend angle)
     anchored within ~1.5pt of the shaft's free (non-fused) end.
  5. Rejects Style A candidates sitting right next to a callout/dimension
     label ("MW1.5", a lone digit, a dimension string like 24" or 2'-6",
     a material code like SS-1) - those are pointer-flag stems or
     dimension lines, not grain arrows. (Style B/C don't need this filter
     - their geometry is already specific enough; see the note in
     scan_pdf.)
  6. Optionally restricts to pages/regions whose title text contains
     "ELEVATION".
  7. Saves ONE FULL PAGE image per page that has any surviving candidates,
     with every candidate on that page highlighted, plus a CSV report, for
     a fast visual pass - geometry alone still isn't 100% certain across
     inconsistent CAD exports.

Usage:
  python3 find_grain_arrows.py /path/to/pdf_folder --out /path/to/report_folder
  python3 find_grain_arrows.py drawing.pdf --out report --elevations-only
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import pymupdf as fitz

# Text patterns that mark "this is a leader/callout/dimension, not a grain
# arrow": view-reference tags (MW1.5), lone index numbers, dimension
# strings (24", 2'-6", 24), and material/finish codes (MM-A, PL-F, SS-1).
CALLOUT_RE = re.compile(
    r'^(MW\d+\.\d+|\d+\'-?\d*"?|\d+"?|[A-Z]{2,3}-[A-Z0-9]{1,2})$'
)

SHAFT_LEN_MIN, SHAFT_LEN_MAX = 15, 300  # pt - was capped at 65pt, which missed longer
                                          # grain arrows (e.g. one drawn the full height
                                          # of a tall panel); now a wider default,
                                          # override further with --shaft-min/--shaft-max
HEAD_CHORD_MIN, HEAD_CHORD_MAX = 5, 12
HEAD_BBOX_MAX = 10
ANCHOR_TOL = 1.5  # pt - how exactly a head's start point must land on a shaft endpoint

# Style B (compound tick-shaft-tick polyline) constants
TICK_LEN_MIN, TICK_LEN_MAX = 1.0, 12
JOIN_TOL = 1.0  # pt - how exactly consecutive/separate segments must share an endpoint
TICK_ANGLE_MIN, TICK_ANGLE_MAX = 110, 160  # degrees off the shaft, in Style B's own
                                             # sign convention; excludes 90 (jogs) and
                                             # 180 (straight-through)
BLACK_TOL = 0.15  # how close an RGB stroke color must be to (0,0,0)

# Style C (split shaft+tick / lone-tick) constants. Measured directly off
# 3 confirmed real grain arrows in a real submittal: every tick bends
# ~45 degrees off the shaft at its joint. Kept as a separate, wider band
# from Style B's TICK_ANGLE_MIN/MAX because Style C's angle is computed
# with a different (orientation-independent) convention - see
# find_split_arrows - not because the real-world angle itself differs.
BEND_ANGLE_MIN, BEND_ANGLE_MAX = 25, 65

# Highlight styling for the full-page output images
HIGHLIGHT_RADIUS = 18  # pt, radius of the circle drawn around each candidate
HIGHLIGHT_COLOR = (1, 0, 0)  # red
HIGHLIGHT_WIDTH = 2  # pt, stroke width of the highlight circle
PAGE_ZOOM = 2  # render zoom factor for the saved full-page PNG


def line_len(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def is_black(color, tol=BLACK_TOL):
    if color is None:
        return False
    return all(abs(v) < tol for v in color)


def angle_between(v1, v2):
    a1 = math.atan2(v1[1], v1[0])
    a2 = math.atan2(v2[1], v2[0])
    d = abs(a1 - a2)
    if d > math.pi:
        d = 2 * math.pi - d
    return math.degrees(d)


def find_shafts_and_heads(drawings):
    """Style A. Return (shafts, heads).
    shaft: drawing whose only item is a straight line 15-300pt long.
    head:  drawing whose only item is a bezier curve with a 5-12pt chord
           and a bounding box no larger than 10x10pt.
    """
    shafts, heads = [], []
    for d in drawings:
        items = d["items"]
        if len(items) != 1:
            continue
        kind = items[0][0]
        if kind == "l":
            p1, p2 = items[0][1], items[0][2]
            length = line_len(p1, p2)
            if SHAFT_LEN_MIN <= length <= SHAFT_LEN_MAX:
                shafts.append({"start": p1, "end": p2, "len": length})
        elif kind == "c":
            p0, p3 = items[0][1], items[0][4]
            chord = line_len(p0, p3)
            r = d["rect"]
            if (HEAD_CHORD_MIN <= chord <= HEAD_CHORD_MAX
                    and r.width <= HEAD_BBOX_MAX and r.height <= HEAD_BBOX_MAX):
                heads.append({"start": p0, "chord": chord})
    return shafts, heads


def head_anchored_at(pt, heads, tol=ANCHOR_TOL):
    """A head only counts if its curve's start point literally coincides
    with this shaft endpoint (within floating-point tolerance) - not just
    'nearby' in a loose radius."""
    for h in heads:
        if line_len(h["start"], pt) <= tol:
            return h
    return None


def find_compound_arrows(drawings):
    """Style B. A single drawing made of exactly 3 connected black "l"
    segments: tick, shaft, tick. Returns a list of shaft-like dicts
    matching the shape returned by find_shafts_and_heads's shafts list,
    so callers can treat both styles uniformly."""
    out = []
    for d in drawings:
        items = d["items"]
        if len(items) != 3:
            continue
        if any(it[0] != "l" for it in items):
            continue
        if not is_black(d.get("color")):
            continue
        p1a, p1b = items[0][1], items[0][2]
        p2a, p2b = items[1][1], items[1][2]
        p3a, p3b = items[2][1], items[2][2]
        # must be one continuous open polyline: seg1 end == seg2 start, etc.
        if line_len(p1b, p2a) > JOIN_TOL or line_len(p2b, p3a) > JOIN_TOL:
            continue
        shaft_len = line_len(p2a, p2b)
        tick1_len = line_len(p1a, p1b)
        tick2_len = line_len(p3a, p3b)
        if not (SHAFT_LEN_MIN <= shaft_len <= SHAFT_LEN_MAX):
            continue
        if not (TICK_LEN_MIN <= tick1_len <= TICK_LEN_MAX):
            continue
        if not (TICK_LEN_MIN <= tick2_len <= TICK_LEN_MAX):
            continue
        shaft_v = (p2b.x - p2a.x, p2b.y - p2a.y)
        tick1_v = (p1b.x - p1a.x, p1b.y - p1a.y)
        tick2_v = (p3b.x - p3a.x, p3b.y - p3a.y)
        a1 = angle_between(shaft_v, tick1_v)
        a2 = angle_between(shaft_v, tick2_v)
        if not (TICK_ANGLE_MIN <= a1 <= TICK_ANGLE_MAX):
            continue
        if not (TICK_ANGLE_MIN <= a2 <= TICK_ANGLE_MAX):
            continue
        out.append({"start": p2a, "end": p2b, "len": shaft_len})
    return out


def _bend_angle(joint, along_far, along_tick):
    """Physical, orientation-independent bend angle at `joint`: the angle
    between the direction from joint TOWARD along_far (i.e. back along
    whatever the shaft continues into) and the direction from joint OUT
    to along_tick. Computed the same way regardless of which point of
    either segment happens to be listed first in the PDF's path data, so
    it works whether the tick's "other end" is p[0] or p[1], and whether
    the shaft is items[0] or items[1] of a fused 2-item drawing."""
    v_shaft = (along_far.x - joint.x, along_far.y - joint.y)
    v_tick = (along_tick.x - joint.x, along_tick.y - joint.y)
    return angle_between(v_shaft, v_tick)


def find_split_arrows(drawings):
    """Style C. The shaft and ONE tick are fused into a single drawing of
    exactly 2 connected black "l" segments (one continuous polyline: the
    two segments share an endpoint, the "joint"). The OTHER tick is a
    completely separate, independent single-item black "l" drawing, whose
    endpoint coincides with the shaft's free (non-joint) end. Confirmed
    against a real submittal's content stream, where PyMuPDF's
    get_drawings() reports these as two distinct path-paint operations -
    not one continuous 3-segment polyline (Style B), and with no bezier
    curve involved (Style A).

    Returns a list of shaft-like dicts (start/end/len), same shape as the
    other two finders, so callers can treat all three uniformly.
    """
    # Lone single-segment black ticks: candidates for the "free" tick that
    # anchors onto a fused shaft+tick drawing's unattached end.
    lone_ticks = []
    for d in drawings:
        items = d["items"]
        if len(items) != 1 or items[0][0] != "l":
            continue
        if not is_black(d.get("color")):
            continue
        p1, p2 = items[0][1], items[0][2]
        length = line_len(p1, p2)
        if TICK_LEN_MIN <= length <= TICK_LEN_MAX:
            lone_ticks.append((p1, p2))

    out = []
    for d in drawings:
        items = d["items"]
        if len(items) != 2 or any(it[0] != "l" for it in items):
            continue
        if not is_black(d.get("color")):
            continue
        p1a, p1b = items[0][1], items[0][2]
        p2a, p2b = items[1][1], items[1][2]
        # must be one continuous open polyline: seg1 end == seg2 start
        if line_len(p1b, p2a) > JOIN_TOL:
            continue

        seg1_len = line_len(p1a, p1b)
        seg2_len = line_len(p2a, p2b)
        joint = p1b  # == p2a, within JOIN_TOL

        # Figure out which segment is the shaft and which is the fused
        # tick; identify the shaft's free (non-joint) end and the fused
        # tick's outer (non-joint) end, regardless of which segment came
        # first in the path.
        if (SHAFT_LEN_MIN <= seg1_len <= SHAFT_LEN_MAX
                and TICK_LEN_MIN <= seg2_len <= TICK_LEN_MAX):
            free_end = p1a          # far end of the shaft segment
            fused_tick_outer = p2b  # far end of the fused tick segment
        elif (SHAFT_LEN_MIN <= seg2_len <= SHAFT_LEN_MAX
                and TICK_LEN_MIN <= seg1_len <= TICK_LEN_MAX):
            free_end = p2b
            fused_tick_outer = p1a
        else:
            continue

        fused_angle = _bend_angle(joint, free_end, fused_tick_outer)
        if not (BEND_ANGLE_MIN <= fused_angle <= BEND_ANGLE_MAX):
            continue

        # Look for a separate lone tick anchored at the shaft's free end.
        match = None
        for lp1, lp2 in lone_ticks:
            if line_len(lp1, free_end) <= JOIN_TOL:
                lone_outer = lp2
            elif line_len(lp2, free_end) <= JOIN_TOL:
                lone_outer = lp1
            else:
                continue
            lone_angle = _bend_angle(free_end, joint, lone_outer)
            if BEND_ANGLE_MIN <= lone_angle <= BEND_ANGLE_MAX:
                match = lone_outer
                break
        if match is None:
            continue

        out.append({"start": joint, "end": free_end, "len": seg1_len if seg1_len > seg2_len else seg2_len})
    return out


def near_callout_text(cx, cy, words, radius=45):
    for w in words:
        if CALLOUT_RE.match(w[4].strip()):
            wx, wy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
            if math.hypot(wx - cx, wy - cy) < radius:
                return True
    return False


def elevation_view_regions(page, pad=400):
    """Rough regions around any title text containing ELEVATION, used to
    restrict matches to elevation drawings only."""
    regions = []
    for b in page.get_text("blocks"):
        if "ELEVATION" in b[4].upper():
            # In shop drawings, the elevation drawing sits ABOVE the title callout bubble
            r = fitz.Rect(b[0] - pad, b[1] - pad * 3.5, b[2] + pad, b[3] + 50)
            regions.append(r)
    return regions


def inside_any(rect_list, x, y):
    return any(r.x0 <= x <= r.x1 and r.y0 <= y <= r.y1 for r in rect_list)


def scan_pdf(path, elevations_only=False):
    doc = fitz.open(path)
    results = []
    for pno in range(len(doc)):
        print(f"  [page {pno+1:02d}/{len(doc):02d}] Scanning sheet for grain arrows...", flush=True)
        page = doc[pno]
        drawings = page.get_drawings()
        words = page.get_text("words")

        elev_regions = elevation_view_regions(page) if elevations_only else None

        # --- Style A: separate shaft + 2 curve-head drawings ---
        shafts, heads = find_shafts_and_heads(drawings)
        style_a_matches = []
        for s in shafts:
            head1 = head_anchored_at(s["start"], heads)
            head2 = head_anchored_at(s["end"], heads)
            if head1 and head2:
                style_a_matches.append(s)

        # --- Style B: single compound tick-shaft-tick polyline ---
        style_b_matches = find_compound_arrows(drawings)

        # --- Style C: split shaft+tick drawing + separate lone tick ---
        style_c_matches = find_split_arrows(drawings)

        # near_callout_text exists to reject a lone straight line that
        # merely LOOKS like a shaft because it happens to sit next to a
        # dimension/leader/material-code label (e.g. "24"", "PL-F") - a
        # real risk for Style A, whose only other requirement is two
        # small anchored curves. Style B/C don't have that ambiguity:
        # their tick-shaft(-tick) geometry with the diagonal hook-angle
        # check is already a much more specific signature than any leader
        # stem would produce, so the callout filter is applied to Style A
        # only. Without this split, a real grain arrow placed close to
        # its own cabinet's material-code label (e.g. "PL-F" sitting
        # ~44pt away) was being discarded even though the geometry match
        # was solid.
        for s in style_a_matches:
            cx = (s["start"].x + s["end"].x) / 2
            cy = (s["start"].y + s["end"].y) / 2
            if near_callout_text(cx, cy, words):
                continue  # dimension string / leader callout, not a grain arrow
            if elevations_only and not inside_any(elev_regions, cx, cy):
                continue
            results.append(
                {
                    "file": path.name,
                    "page": pno + 1,
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "shaft_len": round(s["len"], 1),
                    "x0": s["start"].x, "y0": s["start"].y,
                    "x1": s["end"].x, "y1": s["end"].y,
                    "style": "A",
                }
            )

        for style_label, matches in (("B", style_b_matches), ("C", style_c_matches)):
            for s in matches:
                cx = (s["start"].x + s["end"].x) / 2
                cy = (s["start"].y + s["end"].y) / 2
                if elevations_only and not inside_any(elev_regions, cx, cy):
                    continue
                results.append(
                    {
                        "file": path.name,
                        "page": pno + 1,
                        "x": round(cx, 1),
                        "y": round(cy, 1),
                        "shaft_len": round(s["len"], 1),
                        "x0": s["start"].x, "y0": s["start"].y,
                        "x1": s["end"].x, "y1": s["end"].y,
                        "style": style_label,
                    }
                )
    doc.close()
    return results




def main():
    global SHAFT_LEN_MIN, SHAFT_LEN_MAX

    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="PDF file or folder of PDFs")
    ap.add_argument("--out", default="grain_arrow_report", help="output folder")
    ap.add_argument("--elevations-only", action="store_true",
                     help="only report matches inside views titled ELEVATION...")
    ap.add_argument("--shaft-min", type=float, default=SHAFT_LEN_MIN,
                     help=f"minimum grain-arrow shaft length in pt (default {SHAFT_LEN_MIN})")
    ap.add_argument("--shaft-max", type=float, default=SHAFT_LEN_MAX,
                     help=f"maximum grain-arrow shaft length in pt (default {SHAFT_LEN_MAX}); "
                          "raise this if long grain arrows on tall panels are still being missed")
    args = ap.parse_args()

    SHAFT_LEN_MIN, SHAFT_LEN_MAX = args.shaft_min, args.shaft_max

    in_path = Path(args.input)
    out_dir = Path(args.out)
    pages_dir = out_dir / "highlighted_pages"

    pdfs = [in_path] if in_path.is_file() else sorted(in_path.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found.", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for pdf in pdfs:
        print(f"Scanning {pdf.name} ...")
        rows = scan_pdf(pdf, elevations_only=args.elevations_only)
        if rows:
            annotated_pdf_path = out_dir / f"{pdf.stem}_annotated.pdf"
            
            rows_by_page = {}
            for row in rows:
                rows_by_page.setdefault(row["page"], []).append(row)

            doc = fitz.open(pdf)
            for page_no, page_rows in rows_by_page.items():
                page = doc[page_no - 1]
                shape = page.new_shape()
                for c in page_rows:
                    shape.draw_circle(fitz.Point(c["x"], c["y"]), HIGHLIGHT_RADIUS)
                shape.finish(color=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH, fill=None)
                shape.commit()
            
            doc.save(str(annotated_pdf_path))
            doc.close()
            print(f"  -> Saved annotated PDF: {annotated_pdf_path.name}")

        all_rows.extend(rows)
        print(f"  -> {len(rows)} candidate(s)")

    csv_path = out_dir / "candidates.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "page", "x", "y", "shaft_len", "x0", "y0", "x1", "y1", "style", "page_image"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} total candidate(s).")
    print(f"Report: {csv_path}")
    print(f"Annotated PDFs saved to: {out_dir}")


if __name__ == "__main__":
    main()