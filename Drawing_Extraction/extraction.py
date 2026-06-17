"""
extraction.py — Quade Task: Drawing + Top Tablet Strip Extractor
=================================================================
For each specified page this script produces TWO cropped PDFs:

  1. <sheet>-<drawing_id>.pdf   — the architectural drawing
  2. <sheet>-tablet.pdf         — the exact tablet/keynote table at the TOP of the sheet

Tablet detection strategy
--------------------------
Every architectural table has continuous horizontal border lines separating its
rows. The script uses morphological filtering to isolate these long horizontal
lines, then reads their positions to find:

  - Left edge  (x_left)  : leftmost point of any table row border
  - Right edge (x_right) : rightmost point of any table row border
  - Bottom     (y_bottom) : bottommost row border line

The table crop is then exactly (x_left, 0, x_right, y_bottom).

Fallback: If no long horizontal lines are found in the top half, the top
``--top-pct`` percent of the full page width is used (default 20%).

Usage (CLI)
-----------
    python extraction.py <pdf_path> <page_number> <drawing_id> [options]

Options
-------
    --top-pct <float>   Fallback top-strip height as % of page (default: 20)
    --verbose           Enable DEBUG logging

Examples
--------
    python extraction.py "my_plans.pdf" 8 1
    python extraction.py "my_plans.pdf" 8 1 --top-pct 25 --verbose
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

# ── Add parent directory to path so we can import drawing_extractor ──────────
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import cv2
import fitz          # PyMuPDF
import numpy as np
import pdfplumber
import json

from drawing_extractor import (
    DPI,
    OUTPUT_DIR,
    _BASE_DPI,
    _configure_logging,
    render_page_to_image,
    _build_masks,
    get_sheet_name,
    find_regions_visually,
    _crop_pdf_region,
)

logger = logging.getLogger("extraction")


# ──────────────────────────────────────────────────────────────────────────────
# Tablet Table Detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_table_bounds(
    thresh_bubbles: np.ndarray,
    dpi_scale: float,
    fallback_pct: float = 20.0,
) -> Tuple[int, int, int, int]:
    """
    Detect the exact bounding box of the tablet table using morphological
    horizontal line extraction.

    Core idea
    ---------
    Every row of the tablet table is separated by a continuous horizontal
    border line that spans the full width of the table. These lines are the
    most reliable structural feature of any architectural keynote/legend table.

    Algorithm
    ---------
    1.  Build a morphological kernel that is very wide (at least 5% of page
        width) and 1 pixel tall. When applied with MORPH_OPEN (erode + dilate),
        this keeps ONLY horizontal ink segments that are at least that wide —
        eliminating text, symbols, short lines, and drawing content.

    2.  Find all contours of the surviving horizontal line segments.

    3.  Filter to the top half of the page (the tablet is always in the top
        half). Require each segment to be at least 5% of the page width long.

    4.  From all qualifying segments, compute:
          x_left   = minimum starting x across all row-border lines
          x_right  = maximum ending x across all row-border lines
          y_bottom = maximum bottom edge across all row-border lines

    5.  y_bottom + small padding = precise bottom of the tablet's last row.

    Continuous row detection
    ------------------------
    Because we look at ALL horizontal line segments in the top half (not just
    the first or last), the algorithm sees every row border — even if rows have
    different heights. The leftmost and rightmost extents of these lines
    naturally define the table's left and right boundaries, and the bottommost
    line is the last row's bottom border.

    Args:
        thresh_bubbles:  Binary bubble-detection mask (255=ink, 0=background).
        dpi_scale:       DPI normalisation ratio (current_dpi / 200).
        fallback_pct:    Fallback strip height as % of page height if no lines found.

    Returns:
        (x_left, y_top, x_right, y_bottom) in pixel coordinates.
        Falls back to (0, 0, img_w, fallback_y) if no table lines are found.
    """
    img_h, img_w = thresh_bubbles.shape
    fallback_y   = int(img_h * fallback_pct / 100.0)

    # ── Step 1: Morphological horizontal line isolation ───────────────────────
    # The kernel width is 5% of the page width. Any ink segment shorter than
    # this is eroded away. Only long horizontal table borders survive.
    # At 200 DPI on a 3400px wide page: 5% = 170px ≈ 0.85 inch — easily longer
    # than any drawing annotation but shorter than any table row border.
    min_line_len = max(30, int(img_w * 0.05))
    h_kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_len, 1))

    # MORPH_OPEN = erode then dilate. Erode removes anything shorter than the
    # kernel width. Dilate restores the surviving (long) segments to full length.
    h_lines_mask = cv2.morphologyEx(thresh_bubbles, cv2.MORPH_OPEN, h_kernel)

    logger.debug("Horizontal line mask: %d ink pixels surviving", cv2.countNonZero(h_lines_mask))

    # ── Step 2: Find contours of the surviving horizontal line segments ────────
    contours, _ = cv2.findContours(
        h_lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ── Step 3: Filter contours to the top half of the page ───────────────────
    # The tablet table is always in the top half. Reject anything below.
    search_y_limit = img_h // 2

    # Minimum line length: must span at least 5% of page width
    min_span = int(img_w * 0.05)

    table_segments = []   # List of (x_left, y_top, x_right, y_bottom) per segment
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Must be in the top half
        if y > search_y_limit:
            continue

        # Must be wide enough to be a real table border (not a symbol or dim. line)
        if w < min_span:
            continue

        # Must be thin (a horizontal line, not a filled block)
        # Thickness > 15px at 200DPI would be a filled region, not a line
        if h > int(15 * dpi_scale):
            continue

        table_segments.append((x, y, x + w, y + h))
        logger.debug("Table segment: x=%d y=%d w=%d h=%d", x, y, w, h)

    if not table_segments:
        logger.info(
            "No horizontal table lines found; falling back to top %.0f%% = %d px "
            "(full page width)",
            fallback_pct, fallback_y,
        )
        return (0, 0, img_w, fallback_y)

    # ── Step 4: Compute the table bounding box from all row border lines ───────
    # x_left   = smallest x-start across all row borders  → left edge of table
    # x_right  = largest  x-end   across all row borders  → right edge of table
    # y_bottom = largest  y-end   across all row borders  → bottom of last row
    x_left   = min(seg[0] for seg in table_segments)
    x_right  = max(seg[2] for seg in table_segments)
    y_bottom = max(seg[3] for seg in table_segments)

    # ── Step 5: Add a small padding below the last row border ─────────────────
    # This ensures the bottom line's full stroke width is included in the crop.
    PAD_PX   = int(15 * dpi_scale)
    y_bottom = min(img_h, y_bottom + PAD_PX)

    n = len(table_segments)
    logger.info(
        "Table bounds from %d row-border line(s): "
        "left=%d  right=%d  bottom=%d px",
        n, x_left, x_right, y_bottom,
    )

    return (x_left, 0, x_right, y_bottom)

    
def _extract_tablet_first_col_items(
    pdf_path: str,
    page_index: int,
    thresh_bubbles: np.ndarray,
    tablet_bbox: tuple,
    dpi: int
) -> list[dict]:
    """
    Fixed version: correctly maps pixel coordinates → PDF point coordinates.

    Root cause of original failure
    --------------------------------
    The original code used:
        scale = 72.0 / dpi
        target_x0 = x_left * scale
    
    This ONLY works if the rendered image pixel dimensions match the PDF point
    dimensions scaled by (dpi/72). For a 1224x792 pt PDF at 200 DPI:
        expected image width = 1224 * (200/72) = 3400 px

    The fix: compute scale from actual image size vs actual PDF page size,
    separately for X and Y axes (handles non-square pages and any DPI).
    """
    import logging
    logging.getLogger("pdfminer").setLevel(logging.WARNING)

    x_left, y_top, x_right, y_bottom = tablet_bbox
    img_h, img_w = thresh_bubbles.shape

    items = []

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_index]
        pdf_w = float(page.width)   # e.g. 1224.0 pts
        pdf_h = float(page.height)  # e.g.  792.0 pts

        # ── Correct scaling: pixels → PDF points ──────────────────────────────
        # Instead of a single 72/dpi factor, compute from actual dimensions.
        # This handles any DPI, any page size, landscape or portrait.
        scale_x = pdf_w / img_w   # pts per pixel (x axis)
        scale_y = pdf_h / img_h   # pts per pixel (y axis)

        logger.debug(
            "Page PDF size: %.1f x %.1f pts  |  Image: %d x %d px  |  "
            "scale_x=%.4f  scale_y=%.4f",
            pdf_w, pdf_h, img_w, img_h, scale_x, scale_y
        )

        # ── Convert tablet pixel bounds → PDF points ──────────────────────────
        tab_x0_pt = x_left  * scale_x
        tab_y0_pt = y_top   * scale_y
        tab_x1_pt = x_right * scale_x
        tab_y1_pt = y_bottom * scale_y

        logger.debug(
            "Tablet PDF pts: x(%.1f to %.1f)  y(%.1f to %.1f)",
            tab_x0_pt, tab_x1_pt, tab_y0_pt, tab_y1_pt
        )

        # ── First column = leftmost ~12% of the tablet width ─────────────────
        # The first column in architectural schedules is narrow (code labels
        # like PL-C, H-4.6). We take 12% of the tablet width in PDF pts.
        tab_width_pt  = tab_x1_pt - tab_x0_pt
        col_width_pt  = tab_width_pt * 0.12

        col_x0 = tab_x0_pt
        col_x1 = tab_x0_pt + col_width_pt
        col_y0 = tab_y0_pt
        col_y1 = tab_y1_pt

        logger.debug(
            "First-column search area (PDF pts): x(%.1f to %.1f)  y(%.1f to %.1f)",
            col_x0, col_x1, col_y0, col_y1
        )

        # ── Extract all words from the page ──────────────────────────────────
        words = page.extract_words(x_tolerance=3, y_tolerance=3)
        logger.debug("Total words on page: %d", len(words))

        # Log first 20 words so coordinate space is visible in --verbose mode
        for i, w in enumerate(words[:20]):
            logger.debug(
                "WORD %2d: %-12r  x0=%.1f  top=%.1f  bottom=%.1f",
                i, w["text"], w["x0"], w["top"], w["bottom"]
            )

        skip = {"MATERIALS", "QTY.", "QTY", "HARDWARE"}

        for w in words:
            cx = (w["x0"] + w["x1"]) / 2.0
            cy = (w["top"] + w["bottom"]) / 2.0

            if col_x0 <= cx <= col_x1 and col_y0 <= cy <= col_y1:
                text = w["text"].strip("() ")
                if text and text.upper() not in skip:
                    items.append({
                        "item_id":  text,
                        "bbox_pt": (w["x0"], w["top"], w["x1"], w["bottom"])
                    })
                    logger.debug("  ✓ captured: %r  cx=%.1f  cy=%.1f", text, cx, cy)

    if not items:
        logger.warning(
            "No items found in first column. "
            "Check the DEBUG output above — if word x0 values are far outside "
            "(%.1f to %.1f), the tablet bbox detection may need tuning.",
            col_x0, col_x1
        )

    return items


# ──────────────────────────────────────────────────────────────────────────────
# Main Extraction Function
# ──────────────────────────────────────────────────────────────────────────────

def extract_page(
    pdf_path: str,
    page_index: int,
    drawing_ids: Optional[list[str]] = None,
    dpi: int = DPI,
    output_dir: str = OUTPUT_DIR,
    top_pct: float = 20.0,
) -> None:
    """
    Extract architectural drawing(s) AND the tablet table from a PDF page.

    Produces two types of output PDFs:
      - output/<sheet>/<sheet>-<drawing_id>.pdf   (the drawings)
      - output/<sheet>/<sheet>-tablet.pdf         (the tablet table, exact bounds)

    Args:
        pdf_path:    Path to the source PDF.
        page_index:  Zero-based page index.
        drawing_ids: List of target drawing IDs (e.g. ["1", "W3"]). If None or empty, extracts all.
        dpi:         Render resolution in DPI.
        output_dir:  Root output folder.
        top_pct:     Fallback tablet height as % of page if auto-detection fails.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("─" * 55)
    logger.info("PDF: %s  |  Page: %d  |  IDs: %s", pdf_path, page_index + 1, drawing_ids or "ALL")

    # ── Sheet name and output folder ──────────────────────────────────────────
    sheet_name = get_sheet_name(pdf_path, page_index)
    sheet_dir  = os.path.join(output_dir, sheet_name)
    os.makedirs(sheet_dir, exist_ok=True)
    logger.info("Sheet: %s  →  %s", sheet_name, sheet_dir)

    # ── Step 1: Render the page ───────────────────────────────────────────────
    logger.info("Step 1  — Rendering page to image")
    try:
        img = render_page_to_image(pdf_path, page_index, dpi)
    except Exception as exc:
        logger.error("Page render failed: %s", exc)
        return
    img_h, img_w = img.shape[:2]
    logger.info("Image size: %dx%dpx at %d DPI", img_w, img_h, dpi)

    # Build ink masks once — used for both drawing detection and tablet detection
    thresh_bubbles, thresh_radar = _build_masks(img)
    dpi_scale = dpi / _BASE_DPI

    # ── Step 2–4: Detect all drawings on the page ─────────────────────────────
    logger.info("Step 2  — Detecting drawings on page")
    try:
        regions = find_regions_visually(pdf_path, page_index, img, dpi)
    except Exception as exc:
        logger.error("Drawing detection failed: %s", exc)
        return
    logger.info("Total drawings on page: %d", len(regions))

    # ── Drawing lookup and export ─────────────────────────────────────────────
    if not drawing_ids or "ALL" in (d.upper() for d in drawing_ids):
        target_keys = list(regions.keys())
        if not target_keys:
            logger.warning("No drawings found on page %d.", page_index + 1)
    else:
        target_keys = []
        for did in drawing_ids:
            t_key = next((k for k in regions if k.lower() == did.lower()), None)
            if t_key:
                target_keys.append(t_key)
            else:
                logger.error(
                    "Drawing %r not found on page %d. Available: %s",
                    did, page_index + 1, list(regions.keys()),
                )

    for t_key in target_keys:
        bbox, title, scale = regions[t_key]
        left, top, right, bottom = bbox
        logger.info(
            "Region %r → px (%d,%d) → (%d,%d)", t_key, left, top, right, bottom
        )
        out_drawing = os.path.join(sheet_dir,f"Page_{page_index+1}_{t_key}.pdf")
        try:
            _crop_pdf_region(pdf_path, page_index, bbox, img.shape[:2], out_drawing)
            logger.info("Drawing saved → %s", out_drawing)
        except Exception as exc:
            logger.error("Drawing crop failed: %s", exc)

        logger.info("═" * 55)
        logger.info("  Drawing Number : %s", t_key)
        logger.info("  Sheet          : %s", sheet_name)
        logger.info("  Title          : %s", title)
        logger.info("  Scale          : %s", scale)
        logger.info("  Drawing PDF    : %s", out_drawing)
        logger.info("═" * 55)

    # ── Tablet Table Detection and Crop ───────────────────────────────────────
    # Use morphological line detection to find the exact left, right, and bottom
    # boundaries of the table. Each row's horizontal border line is detected
    # and their combined extents define the exact crop box.
    logger.info("─" * 55)
    logger.info("Detecting tablet table boundaries (horizontal line scan)...")

    x_left, y_top, x_right, y_bottom = _detect_table_bounds(
        thresh_bubbles, dpi_scale, top_pct
    )

    if y_bottom <= 0 or x_right <= x_left:
        logger.warning("Could not determine valid tablet bounds. Skipping tablet crop.")
        return

    tablet_bbox = (x_left, y_top, x_right, y_bottom)
    logger.info(
        "Tablet region → px (%d,%d) → (%d,%d)  [w=%d h=%d]",
        x_left, y_top, x_right, y_bottom,
        x_right - x_left, y_bottom - y_top,
    )

    out_tablet = os.path.join(sheet_dir, f"{sheet_name}-tablet.pdf")
    try:
        _crop_pdf_region(pdf_path, page_index, tablet_bbox, img.shape[:2], out_tablet)
        logger.info("Tablet saved → %s", out_tablet)
        
        # ── Extract First Column Data ─────────────────────────────────────────
        logger.info("Extracting item data from tablet's first column shapes...")
        tablet_items = _extract_tablet_first_col_items(pdf_path, page_index, thresh_bubbles, tablet_bbox, dpi)
        
        if tablet_items:
            items_json_path = os.path.join(sheet_dir, f"{sheet_name}-tablet-items.json")
            with open(items_json_path, "w", encoding="utf-8") as f:
                json.dump(tablet_items, f, indent=2)
            logger.info("Extracted %d items from tablet shapes → %s", len(tablet_items), items_json_path)
            # Print them nicely
            item_names = [i["item_id"] for i in tablet_items]
            logger.info("Found items: %s", ", ".join(item_names))
        else:
            logger.warning("No text extracted from the first column shapes.")
            
    except Exception as exc:
        logger.error("Tablet crop or extraction failed: %s", exc)


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    _args    = sys.argv[1:]
    _verbose = "--verbose" in _args
    _args    = [a for a in _args if a != "--verbose"]

    _top_pct = 20.0
    if "--top-pct" in _args:
        idx = _args.index("--top-pct")
        try:
            _top_pct = float(_args[idx + 1])
            _args.pop(idx)
            _args.pop(idx)
        except (IndexError, ValueError):
            print("ERROR: --top-pct requires a numeric value (e.g. --top-pct 25)")
            sys.exit(1)

    _configure_logging(_verbose)

    _pdf_path    = _args[0]
    _page_num    = int(_args[1]) - 1   # CLI is 1-based; internally 0-based
    _drawing_ids = _args[2:]

    extract_page(
        pdf_path    = _pdf_path,
        page_index  = _page_num,
        drawing_ids = _drawing_ids if _drawing_ids else None,
        top_pct     = _top_pct,
    )
