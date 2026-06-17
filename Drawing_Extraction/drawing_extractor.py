"""
drawing_extractor.py — Architectural PDF Drawing Extractor
===========================================================
Detects and extracts individual drawings from architectural PDF sheets
by visually locating title bubbles (circle + horizontal title line).

Pipeline Stages
---------------
  1. Render      — PDF page → high-resolution numpy image (with OOM fallback)
  2. Detect      — Find candidate circles via OpenCV contour analysis
  3. Validate    — Confirm each circle has a long title line; read drawing ID
                   and title via pdfplumber OCR; verify SCALE label exists
  4. Bound       — Shoot horizontal and vertical pixel-density raycasts to
                   locate the four edges of each drawing's content area
  5. Export      — Crop the PDF (vector-quality) and save as a new PDF file

Usage (CLI)
-----------
    # Extract specific drawings from a page
    python drawing_extractor.py <pdf_path> <page_number> <drawing_id> [<drawing_id> ...]

    # List every drawing detected on a page
    python drawing_extractor.py <pdf_path> <page_number> --list

    # Enable verbose debug output
    python drawing_extractor.py <pdf_path> <page_number> --list --verbose

Examples
--------
    python drawing_extractor.py DGS_Arch.pdf 20 1 2 3
    python drawing_extractor.py DGS_Arch.pdf 35 --list
    python drawing_extractor.py DGS_Arch.pdf 18 W1 W2 W3

Output
------
    output/<SheetName>/<SheetName>-<DrawingID>.pdf
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import fitz          # PyMuPDF
import numpy as np
import pdfplumber


# ──────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────────────────────────

# Module-level logger. Callers configure the level via _configure_logging().
logger = logging.getLogger("drawing_extractor")


def _configure_logging(verbose: bool = False) -> None:
    """
    Configure console logging for the extractor.

    Args:
        verbose: If True, sets level to DEBUG (shows scale-text diagnostics).
                 If False, uses INFO (shows pipeline progress only).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(levelname)s  %(message)s",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(level)


# ──────────────────────────────────────────────────────────────────────────────
# Global Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Page render resolution. Higher = sharper detection but slower and more RAM.
DPI        = 200
# Root folder where all extracted PDF sub-folders are created.
OUTPUT_DIR = "output"

# ── Padding ───────────────────────────────────────────────────────────────────
# After the radar finds the ink edges, extra space is added so annotations,
# grid bubbles, and leader lines at the very edge are never clipped.
# Expressed as multiples of the detected bubble radius so it scales with sheet size.
PAD_BUBBLE_RADII  = 4.0   # Multiplied by bubble radius; applied left/right/bottom.
PAD_TOP_PX        = 150   # Fixed-pixel breathing room added above the topmost ink row.

# ── Radar Thresholds ──────────────────────────────────────────────────────────
# All pixel values below are calibrated for DPI=200 and are normalised at runtime
# by dpi_scale = current_dpi / _BASE_DPI so behaviour is DPI-independent.
_BASE_DPI              = 200
GAP_HORIZONTAL_PX      = 80    # Consecutive empty columns that mark the horizontal edge.
GAP_VERTICAL_PX        = 150   # Consecutive empty rows that mark the top edge.
SOLID_WALL_RATIO       = 0.80  # Column ink fraction above this = structural wall.
MIN_COLUMN_INK         = 10    # Minimum ink pixels in a column to count as "drawing".

# ── Title Bubble Geometry ─────────────────────────────────────────────────────
# Controls what contour sizes are accepted as title bubbles.
MIN_BUBBLE_WIDTH_PX    = 40    # Smallest valid bubble diameter at BASE_DPI.
MAX_BUBBLE_WIDTH_PX    = 150   # Largest valid bubble diameter at BASE_DPI.
CIRCLE_TOLERANCE       = 10    # Max pixel deviation: bounding-box radius vs enclosing circle.
MIN_TITLE_LINE_PX      = 250   # Minimum horizontal title-line length to qualify as a drawing.
LINE_TRACE_WINDOW_PX   = 20    # Vertical pixel window used while tracing the title line.
LINE_GAP_TOLERANCE_PX  = 30    # Gap within which the title line is still considered continuous.

# ── Spatial Exclusion ─────────────────────────────────────────────────────────
TITLE_BLOCK_ZONE       = 0.85  # Bubbles right of this page-width fraction are ignored (legend area).
SIDE_BY_SIDE_DELTA_PX  = 300   # Maximum vertical distance for two bubbles to be "side-by-side".
CEILING_MIN_DELTA_PX   = 200   # Minimum vertical gap for a drawing above to count as a "ceiling".
CEILING_OVERLAP_RATIO  = 0.50  # Ceiling only applies if it covers > 50% of our drawing's width.

# ── Pre-compiled Regex Patterns ───────────────────────────────────────────────
# Compiled once at import time (not inside hot loops) for performance.
_RE_SCALE_WORD  = re.compile(r"SCALE\s*[:\-]?\s*([^\n\r]+)")   # Matches "SCALE: 1/4" = 1'-0""
_RE_SCALE_VALUE = re.compile(r'([\d/\s]+"\']\s*=\s*[\d\'\s"-]+)')  # Matches bare ratio e.g. '3" = 1\'-0"'
_RE_SHEET_ID    = re.compile(r"^[A-Z]{1,5}[-.]?\d{2,4}(\.\d{1,2})?$")  # Matches AE203.4, AB100


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TitleBubble:
    """
    Represents one detected drawing title bubble on the rendered page image.

    All pixel coordinates are in the rendered image space at the configured DPI.
    The bounding box fields (left, top, right, bottom) start at 0 and are
    populated by compute_boundaries() in Stage 4.
    """
    cx:         int           # Circle centre X (pixels)
    cy:         int           # Circle centre Y (pixels)
    r:          int           # Circle radius   (pixels)
    line_end_x: int           # Rightmost X of the horizontal title line (pixels)
    num:        str = ""      # Drawing number read from inside the circle (e.g. "W1", "3")
    title:      str = ""      # Drawing title read from above the title line
    scale:      str = ""      # Scale string, e.g. '1/4" = 1\'-0"'

    # Bounding box — initially zero; set by the boundary radar in Stage 4
    left:   int = 0
    top:    int = 0
    right:  int = 0
    bottom: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# OCR Helper
# ──────────────────────────────────────────────────────────────────────────────

def _clean_ocr(text: str, force_dedup: bool = False) -> str:
    """
    Fix PDF fill+stroke text doubling (e.g. "LLOOBBBBYY" → "LOBBY").

    Some CAD exports render every character twice — once as a filled glyph and
    once as a stroked outline directly on top. pdfplumber reads both layers and
    returns doubled strings like "SSCCAALLEE".

    Detection strategy (word-by-word):
      - A word is doubled if it has an even length AND every other character
        equals the next one (w[0::2] == w[1::2]).
      - Words of length >= 4 are always cleaned when they match this pattern
        (long words are never genuinely doubled in English).
      - Short words (length 2) are only cleaned when force_dedup=True,
        because "11", "22", "AA" can be legitimate drawing IDs.

    force_dedup=True is set when a long doubled word (like "SSCCAALLEE") is
    detected anywhere in the same bubble's text, proving the entire page uses
    the fill+stroke export glitch.

    Args:
        text:        Raw string extracted by pdfplumber.
        force_dedup: When True, also deduplicate short (2-char) words.

    Returns:
        Cleaned string with doubled characters collapsed.
    """
    if not text:
        return ""

    lines = []
    for ln in text.split("\n"):
        words = ln.split(" ")
        clean_words = []
        for w in words:
            if not w:
                # Preserve intentional spacing — skip empty tokens from split
                continue
            # Check if this word is a perfect interleaved duplicate
            if len(w) % 2 == 0 and w[0::2] == w[1::2]:
                # Deduplicate if the word is long (always safe) or force_dedup is on
                if force_dedup or len(w) >= 4:
                    clean_words.append(w[0::2])
                else:
                    # Short word like "11" or "AA" — preserve as-is unless forced
                    clean_words.append(w)
            else:
                # Word is not doubled — keep verbatim
                clean_words.append(w)
        lines.append(" ".join(clean_words))

    return "\n".join(lines).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Page Rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_page_to_image(pdf_path: str, page_index: int, dpi: int = DPI) -> np.ndarray:
    """
    Render a single PDF page to a BGR numpy image at the requested DPI.

    Memory safety: If the system cannot allocate the full-resolution pixmap,
    the renderer retries at 75% of the current DPI until it succeeds, then
    upscales the result so all downstream pixel thresholds remain calibrated.

    Args:
        pdf_path:   Absolute or relative path to the source PDF file.
        page_index: Zero-based page index (page 1 in UI = index 0 here).
        dpi:        Target render resolution. Default 200 DPI.

    Returns:
        BGR uint8 numpy array of shape (H, W, 3).

    Raises:
        FileNotFoundError: If the PDF file does not exist at pdf_path.
        ValueError:        If page_index is beyond the last page.
        MemoryError:       If the page cannot be rendered even at minimum DPI (72).
    """
    # Guard: fail fast with a clear message if the file is missing
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    # Open with a context manager to guarantee the file handle is released
    # even if rendering throws an exception halfway through.
    with fitz.open(pdf_path) as doc:
        if page_index >= len(doc):
            raise ValueError(
                f"PDF has {len(doc)} page(s); index {page_index} is out of range."
            )

        page        = doc[page_index]
        current_dpi = float(dpi)
        pix         = None

        # Retry loop: reduce DPI by 25% each attempt on out-of-memory errors.
        # This allows very large sheets to be processed on memory-constrained machines.
        while current_dpi >= 72:
            try:
                # fitz uses a 72-point coordinate system, so DPI/72 is the scale factor
                mat = fitz.Matrix(current_dpi / 72, current_dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                break   # Render succeeded — exit the retry loop
            except Exception as exc:
                if "malloc" in str(exc).lower() or "memory" in str(exc).lower():
                    # Out-of-memory: log a warning and try a smaller DPI
                    logger.warning(
                        "OOM at %d DPI → retrying at %d DPI",
                        int(current_dpi), int(current_dpi * 0.75)
                    )
                    current_dpi *= 0.75
                else:
                    # Non-memory error (corrupt PDF, etc.) — re-raise immediately
                    raise

    # If the loop exited without a pixmap, every DPI level failed
    if pix is None:
        raise MemoryError("Cannot render page — system out of memory even at minimum DPI.")

    # Convert fitz RGB pixmap → OpenCV BGR numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # If we rendered at a reduced DPI, upscale to the target DPI so that all
    # pixel-count thresholds (gap sizes, line lengths, etc.) remain valid.
    if current_dpi != dpi:
        logger.warning("Upscaling image to target DPI for consistent radar thresholds...")
        scale_factor = dpi / current_dpi
        img = cv2.resize(
            img, (0, 0), fx=scale_factor, fy=scale_factor,
            interpolation=cv2.INTER_LINEAR
        )

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 & 3: Circle Detection + Bubble Validation
# ──────────────────────────────────────────────────────────────────────────────

def _build_masks(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two binary ink masks from the BGR page image for different pipeline stages.

    Mask A — thresh_bubbles (used by: circle detection, title-line tracing):
        Standard luminance grayscale so ALL ink colours (black, blue, green)
        appear dark. Yellow PDF highlighter is surgically erased before
        thresholding so it cannot break the outline of any circle drawn beneath it.
        Threshold: pixels darker than 200 become white (ink present).

    Mask B — thresh_radar (used by: boundary edge detection):
        Channel-minimum grayscale: np.min(R, G, B) per pixel. A pure red pixel
        (0, 0, 255 in BGR) has min=0, so it maps to "ink present". Pure white
        maps to 255 = "no ink". Threshold at 254 means only pure white (255) is
        treated as empty space — every other colour, including faint grey lines,
        counts as drawing content.

    Returns:
        (thresh_bubbles, thresh_radar) — both binary inverse uint8 arrays
        where 255 = ink present and 0 = background.
    """
    # Split BGR channels for per-channel operations
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # ── Mask A: bubble detection ─────────────────────────────────────────────
    # Standard luminance grayscale detects all ink colours uniformly.
    gray_bubbles = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Yellow highlighter signature in BGR: low Blue (<80), high Green (>180), high Red (>180).
    # We blank these pixels to pure white (255) BEFORE thresholding so the highlighter
    # cannot create a gap in a circle outline drawn underneath it.
    is_yellow = (b.astype(np.int16) < 80) & (g > 180) & (r > 180)
    gray_bubbles = gray_bubbles.copy()
    gray_bubbles[is_yellow] = 255

    # Binary-inverse threshold: pixels with luminance < 200 become 255 (ink), rest 0.
    _, thresh_bubbles = cv2.threshold(gray_bubbles, 200, 255, cv2.THRESH_BINARY_INV)

    # ── Mask B: radar / boundary detection ───────────────────────────────────
    # np.min across the colour axis picks the darkest channel per pixel.
    # This catches coloured lines (green structural walls, red annotations,
    # blue dimension lines) that a luminance conversion might lighten.
    gray_radar = np.min(img, axis=2).astype(np.uint8)
    # Only pure white (255) is empty — threshold at 254 makes it maximally sensitive.
    _, thresh_radar = cv2.threshold(gray_radar, 254, 255, cv2.THRESH_BINARY_INV)

    return thresh_bubbles, thresh_radar


def _detect_circles(
    thresh_bubbles: np.ndarray, dpi_scale: float
) -> List[Tuple[int, int, int]]:
    """
    Find all near-circular contours within the expected title-bubble size range.

    Why contours instead of HoughCircles?
        HoughCircles works well for photographic images with gradient edges.
        Architectural PDFs have pure binary, single-pixel-thick strokes with no
        gradients, which causes HoughCircles to be unreliable. Contour analysis
        directly traces the ink paths and tests their geometry mathematically.

    Four-stage geometric filter (applied to every contour found):
        Test 1 — Aspect ratio:   bounding box must be nearly square (0.85–1.15).
        Test 2 — Size range:     width must be within bubble diameter limits.
        Test 3 — Circularity:    enclosing-circle radius must match bounding-box half-width.

    Deduplication:
        CAD software often draws a circle twice (fill layer + stroke layer),
        producing two overlapping contours. Any pair of centres within 20px
        of each other is deduplicated, keeping only the first.

    Args:
        thresh_bubbles: Binary bubble-detection mask (255=ink, 0=background).
        dpi_scale:      Ratio current_dpi / _BASE_DPI, used to scale size limits.

    Returns:
        List of (cx, cy, radius) tuples in pixel coordinates.
    """
    # Scale the fixed pixel thresholds to the actual render DPI
    min_w = int(MIN_BUBBLE_WIDTH_PX * dpi_scale)
    max_w = int(MAX_BUBBLE_WIDTH_PX * dpi_scale)
    tol   = int(CIRCLE_TOLERANCE   * dpi_scale)

    # Find all connected ink regions (contours) in the binary mask
    contours, _ = cv2.findContours(
        thresh_bubbles, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_circles: List[Tuple[int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Guard against degenerate zero-height contours
        if h == 0:
            continue

        # Test 1: Bounding box must be approximately square (circles fit in squares)
        if not (0.85 <= w / h <= 1.15):
            continue

        # Test 2: Diameter must fall within the expected bubble size range
        if not (min_w <= w <= max_w):
            continue

        # Test 3: The smallest circle enclosing the contour should have a radius
        # that closely matches half the bounding-box width.
        # A door-swing arc or D-shape would have a much larger enclosing circle.
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if abs((w / 2) - radius) < tol:
            raw_circles.append((int(cx), int(cy), int(radius)))

    # Deduplicate: remove any centre that is within 20px of an earlier centre.
    # Uses a vectorised numpy approach for efficiency on large contour sets.
    if not raw_circles:
        return []

    pts  = np.array([(cx, cy) for cx, cy, _ in raw_circles], dtype=np.float32)
    kept = np.ones(len(pts), dtype=bool)
    for i in range(len(pts)):
        if not kept[i]:
            continue
        if i + 1 < len(pts):
            dists        = np.hypot(pts[i + 1:, 0] - pts[i, 0], pts[i + 1:, 1] - pts[i, 1])
            kept[i + 1:] &= dists >= 20

    return [raw_circles[i] for i in range(len(raw_circles)) if kept[i]]


def _trace_title_line(
    cx: int, cy: int, r: int,
    thresh_bubbles: np.ndarray,
    dpi_scale: float,
) -> Optional[int]:
    """
    Confirm a circle has a long horizontal title line to its right, and trace it.

    A valid title bubble has a horizontal line extending from its right edge to
    some point further right. This line carries the drawing title text above it.

    Phase 1 — Fast ROI probe (rejection filter):
        Scans a small rectangular region immediately to the right of the circle.
        If no significant horizontal ink exists, the circle is immediately rejected
        without the more expensive column-by-column trace. This filters out
        small circles that appear in hatching, section marks, etc.

    Phase 2 — Full rightward column trace:
        Walks column by column across the full page width.
        Tracks the exact Y-position of the strongest horizontal ink row found in
        Phase 1. Stops when a gap of more than LINE_GAP_TOLERANCE_PX empty
        columns is encountered (the line has ended).

    Args:
        cx, cy, r:      Circle centre coordinates and radius (pixels).
        thresh_bubbles: Binary bubble-detection mask.
        dpi_scale:      DPI normalisation factor.

    Returns:
        The rightmost X pixel of the title line, or None if no valid line found.
    """
    h, w        = thresh_bubbles.shape
    min_line_px = int(MIN_TITLE_LINE_PX  * dpi_scale)
    win_half    = max(1, int(LINE_TRACE_WINDOW_PX * dpi_scale / 2))
    gap_tol     = max(5, int(LINE_GAP_TOLERANCE_PX * dpi_scale))

    # Start probing just to the right of the circle boundary
    x_probe_start = cx + r + 5
    x_probe_end   = min(w, x_probe_start + int(150 * dpi_scale))

    # If the probe start would be at or beyond the page edge, skip this circle
    if x_probe_end >= w:
        return None

    # Phase 1: Crop a small ROI to the right of the circle for fast rejection
    y_start = max(0, cy - r)
    y_end   = min(h, cy + r)
    roi     = thresh_bubbles[y_start:y_end, x_probe_start:x_probe_end]

    if roi.size == 0:
        return None

    # Count ink pixels per row in the ROI to find the dominant horizontal line
    row_sums = np.count_nonzero(roi, axis=1)
    if row_sums.max() <= int(100 * dpi_scale):
        # Not enough horizontal ink — not a title line
        return None

    # Find the exact Y of the strongest (most ink) row; this is the title line
    exact_y = y_start + int(np.argmax(row_sums))

    # Phase 2: Trace the title line rightward across the full page width.
    # Pre-compute the vertical scan window bounds (constant for this circle).
    wt = max(0, exact_y - win_half)
    wb = min(h, exact_y + win_half + 1)

    line_end_x  = x_probe_start
    empty_count = 0
    for x in range(x_probe_start, w):
        if np.any(thresh_bubbles[wt:wb, x]):
            # Found ink at this column — advance the end marker and reset gap counter
            line_end_x  = x
            empty_count = 0
        else:
            empty_count += 1
            if empty_count > gap_tol:
                # Gap is too long — the line has ended
                break

    # Final length check: the line must be long enough to be a real title line
    if (line_end_x - cx) < min_line_px:
        return None

    return line_end_x


def _read_bubble_metadata(
    bubbles: List[TitleBubble],
    pdf_path: str,
    page_index: int,
    img_shape: Tuple[int, int],
    dpi_scale: float,
) -> List[TitleBubble]:
    """
    Use pdfplumber to read the drawing number, title, and scale for each bubble.

    Only bubbles that have a detectable SCALE label below them are returned
    (the SCALE check is the final validation gate that confirms a shape is a
    real drawing title bubble rather than an arbitrary circle on the page).

    Coordinate conversion:
        Pixel positions in the rendered image are divided by the pixel-per-point
        ratio (image_pixels / pdf_points) to convert them to PDF point coordinates,
        which pdfplumber uses for its crop() calls.

    Doubled-text detection:
        Some PDF exports double every character (fill + stroke layers).
        Before cleaning, the code scans all extracted text for a long doubled word
        (len >= 4, perfectly interleaved). If found, it sets force_dedup=True so
        even short 2-char tokens like "11" are correctly halved to "1".

    Args:
        bubbles:     Candidate TitleBubble objects (geometry already set).
        pdf_path:    Path to the source PDF.
        page_index:  Zero-based page index.
        img_shape:   (height, width) of the rendered image for coordinate conversion.
        dpi_scale:   DPI normalisation ratio (not used here directly, kept for API consistency).

    Returns:
        Subset of bubbles that passed SCALE validation, with num/title/scale populated.
    """
    img_h, img_w = img_shape
    validated: List[TitleBubble] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page   = pdf.pages[page_index]
            pw, ph = page.width, page.height

            # Scale factors: divide pixel coords by these to get PDF point coords
            sx = img_w / pw
            sy = img_h / ph

            for b in bubbles:
                cx, cy, r = b.cx, b.cy, b.r

                # ── Step 1: Extract raw text from three regions around the bubble ──
                # We extract all three regions first (before any OCR cleaning) so we
                # can detect the fill+stroke doubling glitch from the combined text.
                raw_num_text   = ""
                raw_title_text = ""
                raw_scale_text = ""

                # Drawing number: tight crop inside the circle.
                # Using r-2 to stay well inside the boundary and avoid picking up
                # text from an adjacent drawing number (e.g. "11" leaking into "1").
                try:
                    num_box = (
                        max(0,  (cx - r + 2) / sx),
                        max(0,  (cy - r + 2) / sy),
                        min(pw, (cx + r - 2) / sx),
                        min(ph, (cy + r - 2) / sy),
                    )
                    raw_num_text = (page.crop(num_box).extract_text(x_tolerance=1) or "").strip()
                except Exception as exc:
                    logger.debug("Number extraction failed for bubble at (%d,%d): %s", cx, cy, exc)

                # Drawing title: region to the right of the circle and above the title line.
                try:
                    x0 = max(0,  (cx + r)       / sx)
                    y0 = max(0,  (cy - r - 100) / sy)
                    x1 = min(pw, b.line_end_x   / sx)
                    y1 = max(0,  cy             / sy)
                    # Ensure the crop box is at least 1 point wide and tall
                    if x1 <= x0: x1 = x0 + 1
                    if y1 <= y0: y1 = y0 + 1
                    raw_title_text = (page.crop((x0, y0, x1, y1)).extract_text() or "").strip()
                except Exception as exc:
                    logger.debug("Title extraction failed for bubble at (%d,%d): %s", cx, cy, exc)

                # Scale label: region below the circle (also the validation gate).
                # Extended horizontally to 12 radii wide to capture typical scale formats.
                try:
                    sx0 = max(0,  (cx - r * 2)  / sx)
                    sy0 = max(0,   cy           / sy)
                    sx1 = min(pw, (cx + r * 12) / sx)
                    sy1 = min(ph, (cy + r * 5)  / sy)
                    raw_scale_text = (page.crop((sx0, sy0, sx1, sy1)).extract_text() or "").upper()
                except Exception as exc:
                    logger.debug("Scale extraction failed for bubble at (%d,%d): %s", cx, cy, exc)

                # ── Step 2: Detect fill+stroke doubling on this page ──────────────
                # Combine all text from this bubble and look for any word >= 4 chars
                # that is perfectly interleaved (e.g. "SSCCAALLEE"). If found, the
                # entire PDF page uses the fill+stroke glitch.
                combined   = f"{raw_num_text} {raw_title_text} {raw_scale_text}"
                is_doubled = False
                for word in combined.split():
                    if len(word) >= 4 and len(word) % 2 == 0 and word[0::2] == word[1::2]:
                        is_doubled = True
                        break

                # ── Step 3: Clean and assign the OCR fields ────────────────────────
                # Number: collapse all whitespace first (handles "1\n1", "1 1", "W1\nW1")
                collapsed = re.sub(r"\s+", "", raw_num_text)
                b.num = _clean_ocr(collapsed, force_dedup=is_doubled)

                # Title: take the LAST non-empty line (closest to the title baseline)
                lines   = [ln.strip() for ln in raw_title_text.split("\n") if ln.strip()]
                b.title = _clean_ocr(lines[-1], force_dedup=is_doubled) if lines else ""

                # ── Step 4: Scale validation gate ─────────────────────────────────
                # A bubble only qualifies as a real drawing title if a SCALE label
                # is visible below it. Silently skip any bubble that lacks one.
                try:
                    clean_scale = _clean_ocr(raw_scale_text, force_dedup=is_doubled)
                    logger.debug("Drawing %s scale text: %r", b.num, clean_scale)

                    match_a = _RE_SCALE_WORD.search(clean_scale)   # "SCALE: 1/4" = 1'-0""
                    match_b = _RE_SCALE_VALUE.search(clean_scale)  # bare ratio fallback

                    if match_a:
                        b.scale = match_a.group(1).strip()
                        validated.append(b)
                    elif match_b:
                        b.scale = match_b.group(1).strip()
                        validated.append(b)
                    # If neither pattern matches, this is not a drawing bubble — skip silently

                except Exception as exc:
                    logger.debug("Scale validation failed for drawing %s: %s", b.num, exc)

    except Exception as exc:
        # If pdfplumber itself fails to open the page, log and return whatever was validated so far
        logger.error("pdfplumber failed to open page %d of %r: %s", page_index, pdf_path, exc)

    return validated


def detect_title_bubbles(
    img: np.ndarray,
    thresh_bubbles: np.ndarray,
    pdf_path: str,
    page_index: int,
    dpi: int = DPI,
) -> List[TitleBubble]:
    """
    Full Stage 2 + 3 pipeline — detect circles and validate them as drawing titles.

    Sub-steps:
      2a — Circle detection via contour analysis (geometry only).
      2b — Title line tracing: each circle must have a long line to its right.
      2c — Spatial filter: exclude bubbles in the right-side title block / legend area.
      3  — OCR validation: read number, title, and SCALE from pdfplumber.
           Only bubbles with a detectable SCALE label survive this stage.

    Args:
        img:            Full-page BGR numpy image.
        thresh_bubbles: Binary bubble-detection mask from _build_masks().
        pdf_path:       Source PDF path.
        page_index:     Zero-based page index.
        dpi:            Render DPI (used to compute dpi_scale for threshold scaling).

    Returns:
        List of fully populated TitleBubble objects — one per valid drawing on the page.
    """
    # Normalisation factor: all pixel thresholds are calibrated at _BASE_DPI.
    dpi_scale = dpi / _BASE_DPI
    _, w      = thresh_bubbles.shape

    # Step 2a: Find all near-circular contours in the binary mask
    circles = _detect_circles(thresh_bubbles, dpi_scale)
    logger.info("Step 2a — %d candidate circle(s) detected", len(circles))

    # Step 2b: For each circle, check whether a long horizontal title line
    # extends to its right. Circles without such a line are rejected here.
    raw_bubbles: List[TitleBubble] = []
    for cx, cy, r in circles:
        line_end = _trace_title_line(cx, cy, r, thresh_bubbles, dpi_scale)
        if line_end is not None:
            raw_bubbles.append(TitleBubble(cx=cx, cy=cy, r=r, line_end_x=line_end))
    logger.info("Step 2b — %d bubble(s) with valid title line", len(raw_bubbles))

    # Step 2c: Remove bubbles in the right-side title block / legend zone.
    # TITLE_BLOCK_ZONE = 0.85 means anything in the rightmost 15% is excluded.
    max_cx   = int(w * TITLE_BLOCK_ZONE)
    filtered = [b for b in raw_bubbles if b.cx < max_cx]
    logger.info("Step 2c — %d bubble(s) after title-block zone exclusion", len(filtered))

    if not filtered:
        # No candidates remain — nothing to validate
        return []

    # Step 3: Open the PDF with pdfplumber to read text inside/near each bubble.
    # Only bubbles that contain a readable SCALE label pass through this gate.
    validated = _read_bubble_metadata(
        filtered, pdf_path, page_index, img.shape[:2], dpi_scale
    )
    logger.info("Step 3  — %d bubble(s) passed SCALE validation", len(validated))

    # Debug: log each validated bubble's position and metadata
    if logger.isEnabledFor(logging.DEBUG):
        for b in sorted(validated, key=lambda x: x.cy):
            logger.debug(
                "  cy=%-5d  id=%-6r  scale=%r  title=%r",
                b.cy, b.num, b.scale, b.title,
            )

    return validated


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4: Boundary Detection (Radar System)
# ──────────────────────────────────────────────────────────────────────────────

def _classify_neighbours(
    bubble: TitleBubble,
    all_bubbles: List[TitleBubble],
) -> Tuple[List[TitleBubble], List[TitleBubble]]:
    """
    Classify sibling bubbles on the same page as left-blockers or right-blockers.

    A bubble is a "side-by-side neighbour" only if its vertical centre (cy) is
    within SIDE_BY_SIDE_DELTA_PX of the current bubble. This prevents a drawing
    far above from acting as a horizontal blocker.

    Left-blockers limit how far the radar scans to the left (they own that territory).
    Right-blockers limit how far the radar scans to the right.

    Args:
        bubble:      The bubble whose neighbours we are classifying.
        all_bubbles: All validated bubbles on the current page.

    Returns:
        (left_blockers, right_blockers) — two lists of TitleBubble objects.
    """
    left_blockers:  List[TitleBubble] = []
    right_blockers: List[TitleBubble] = []

    for ob in all_bubbles:
        if ob is bubble:
            continue   # Skip self

        # Only consider bubbles that are roughly on the same horizontal band
        if abs(ob.cy - bubble.cy) < SIDE_BY_SIDE_DELTA_PX:
            if ob.cx < bubble.cx:
                left_blockers.append(ob)
            else:
                right_blockers.append(ob)

    return left_blockers, right_blockers


def _horizontal_radar(
    bubble: TitleBubble,
    thresh_radar: np.ndarray,
    left_blockers: List[TitleBubble],
    right_blockers: List[TitleBubble],
    dpi_scale: float,
    col_sums: np.ndarray,   # Pre-computed binary ink array (H×W), shape (img_h, img_w)
) -> Tuple[int, int]:
    """
    Shoot horizontal raycasts left and right from the title bubble to locate
    the natural ink edges of the drawing's content.

    How it works:
        The radar walks column by column outward from the bubble centre.
        At each column it sums the ink pixels in a vertical band that spans
        from 35% of the image height above the bubble down to the bubble bottom.

        The scan stops when one of three conditions is met:
          a) A gap of GAP_HORIZONTAL_PX consecutive ink-free columns is found
             (the drawing content has naturally ended).
          b) A solid wall is detected (ink > SOLID_WALL_RATIO × band height).
             When a wall is hit, the gap threshold is reduced to ~150px so the
             radar crosses the wall but stops quickly on the other side,
             capturing any leader lines or callouts attached to the outside of
             structural walls without running all the way to the next drawing.
          c) The scan reaches a neighbour bubble's boundary or the page edge.

    Pre-computed ink maps:
        col_sums is the raw binary ink array (thresh_radar > 0, uint16).
        The band-specific column sums are computed here by slicing along axis=0
        and calling .sum(axis=0), which is fast because numpy slices are views.

    Args:
        bubble:         The drawing whose horizontal bounds we are finding.
        thresh_radar:   Binary radar mask (colour-aware, 255=ink, 0=empty).
        left_blockers:  Bubbles to the left that cap the left scan.
        right_blockers: Bubbles to the right that cap the right scan.
        dpi_scale:      DPI normalisation factor.
        col_sums:       Pre-computed binary ink array for efficient band slicing.

    Returns:
        (left_px, right_px) — column indices of the detected left and right edges.
    """
    img_h, img_w = thresh_radar.shape
    gap_px = int(GAP_HORIZONTAL_PX * dpi_scale)
    pad_px = int(bubble.r * PAD_BUBBLE_RADII)

    # Define the vertical scan band: from 85% of the image height above the bubble to its bottom edge.
    # This band is wide enough to catch tall drawing content above the bubble.
    y_top    = int(max(0,     bubble.cy - img_h * 0.85))
    y_bottom = int(min(img_h, bubble.bottom))
    band_h   = y_bottom - y_top

    # Compute column ink sums within the scan band (one sum per column)
    band_col_sums = col_sums[y_top:y_bottom, :].sum(axis=0)   # shape (img_w,)

    # ── Left scan ─────────────────────────────────────────────────────────────
    # The left edge cannot go beyond the centre of any left-blocker.
    max_left = max((ob.cx for ob in left_blockers), default=0)
    left_px  = max(max_left, bubble.left)

    x_start = int(bubble.cx)
    x_limit = int(max_left)

    if x_start > x_limit and band_h > 0:
        solid_threshold   = SOLID_WALL_RATIO * band_h
        gap               = 0
        found             = False
        current_gap_limit = gap_px   # Dynamic: tightens when a solid wall is crossed

        for x in range(x_start, x_limit, -1):
            ink = float(band_col_sums[x])

            if ink > solid_threshold:
                # Solid structural wall detected — tighten the gap limit so we
                # stop soon after the wall rather than scanning all the way to the
                # next drawing's territory.
                current_gap_limit = min(current_gap_limit, int(150 * dpi_scale))

            if ink > MIN_COLUMN_INK:
                found = True
                gap   = 0
            elif found:
                gap += 1
                if gap > current_gap_limit:
                    # Wide enough gap found — set left edge and stop scanning
                    left_px = max(x_limit, x + gap - pad_px)
                    break
        else:
            # for-else: loop completed without hitting a gap — use neighbour boundary
            left_px = max_left
    else:
        # Not enough room to scan — use the initial geometry-based estimate
        left_px = min(bubble.left, max_left) if max_left else bubble.left

    # ── Right scan ────────────────────────────────────────────────────────────
    # The right edge cannot go beyond the rightmost title line of any right-blocker.
    max_right = min((ob.line_end_x for ob in right_blockers), default=img_w)
    right_px  = min(max_right, bubble.right)

    x_start = int(bubble.line_end_x)
    x_limit = int(max_right)

    if x_limit > x_start and band_h > 0:
        solid_threshold   = SOLID_WALL_RATIO * band_h
        gap               = 0
        found             = False
        current_gap_limit = gap_px

        for x in range(x_start, x_limit):
            ink = float(band_col_sums[x])

            if ink > solid_threshold:
                # Solid wall — reduce gap limit so we stop shortly after crossing it
                current_gap_limit = min(current_gap_limit, int(150 * dpi_scale))

            if ink > MIN_COLUMN_INK:
                found = True
                gap   = 0
            elif found:
                gap += 1
                if gap > current_gap_limit:
                    # Gap found — set right edge and stop
                    right_px = min(x_limit, x - gap + pad_px)
                    break
        else:
            # Loop completed without a gap — use the neighbour boundary
            right_px = max_right
    else:
        right_px = max(bubble.right, max_right)

    return int(left_px), int(right_px)


def _vertical_radar(
    bubble: TitleBubble,
    all_bubbles: List[TitleBubble],
    thresh_radar: np.ndarray,
    dpi_scale: float,
    row_sums: np.ndarray,   # Pre-computed binary ink array (H×W), shape (img_h, img_w)
) -> int:
    """
    Shoot a vertical raycast upward from the title bubble to locate the top edge
    of the drawing's content.

    Hard ceiling logic:
        Before scanning, the function checks all other bubbles on the page.
        If a drawing directly above covers more than CEILING_OVERLAP_RATIO of
        the current drawing's horizontal width, its bubble centre (+ 50px) is
        used as a hard ceiling. The upward scan cannot go above this ceiling,
        preventing the scanner from invading a stacked drawing's territory.

    Upward scan:
        Walks row by row upward within the current drawing's horizontal span.
        Uses pre-computed row sums (sliced to the drawing's x-range) for speed.
        Stops when GAP_VERTICAL_PX consecutive empty rows are found.

    Args:
        bubble:      The drawing whose top boundary we are finding.
        all_bubbles: All validated bubbles on the page (for ceiling detection).
        thresh_radar: Binary radar mask.
        dpi_scale:   DPI normalisation factor.
        row_sums:    Pre-computed binary ink array for efficient band slicing.

    Returns:
        top_px — the row index of the detected top boundary.
    """
    gap_px = int(GAP_VERTICAL_PX * dpi_scale)

    # ── Hard ceiling: find the nearest stacked drawing above ─────────────────
    hard_ceiling = 0   # 0 = top of page (no ceiling applied)
    b_width      = bubble.right - bubble.left

    for ob in all_bubbles:
        if ob is bubble:
            continue

        # Must be genuinely above, not just a side-by-side neighbour
        if ob.cy >= bubble.cy - CEILING_MIN_DELTA_PX:
            continue

        # Only apply a ceiling if the drawing above overlaps a significant fraction
        # of our drawing's horizontal width
        overlap = min(bubble.right, ob.right) - max(bubble.left, ob.left)
        if overlap > 0 and b_width > 0 and (overlap / b_width) > CEILING_OVERLAP_RATIO:
            # Place the ceiling at the stacked drawing's bubble centre + 50px buffer
            hard_ceiling = max(hard_ceiling, ob.cy + 50)

    # ── Upward scan ───────────────────────────────────────────────────────────
    scan_x0 = int(bubble.left)
    scan_x1 = int(bubble.right)
    y_start  = int(max(hard_ceiling, bubble.cy - 150))
    y_limit  = int(hard_ceiling)

    # Default top: hard ceiling minus padding (avoids clipping the drawing above)
    top_px = max(0, hard_ceiling - PAD_TOP_PX)

    if y_start > y_limit and (scan_x1 - scan_x0) > 0:
        # Compute row ink sums within the horizontal span of this drawing
        band_row_sums = row_sums[:, scan_x0:scan_x1].sum(axis=1)  # shape (img_h,)

        gap   = 0
        found = False
        for y in range(y_start, y_limit, -1):
            ink = float(band_row_sums[y])
            if ink > 0:
                # Found ink — this row is still inside the drawing
                found = True
                gap   = 0
            elif found:
                gap += 1
                if gap > gap_px:
                    # Wide enough empty gap found — set top edge and stop
                    top_px = max(y_limit, y + gap_px - PAD_TOP_PX)
                    break

    return int(top_px)


def compute_boundaries(
    bubbles: List[TitleBubble],
    thresh_radar: np.ndarray,
    dpi: int = DPI,
) -> List[TitleBubble]:
    """
    Stage 4 — Run the full boundary radar for every detected drawing on the page.

    Performance optimisation:
        Instead of calling np.sum() on large 2D array slices inside hot per-column
        and per-row loops (which is expensive), we convert thresh_radar to a binary
        uint16 array once and pass it to the radar functions. Each function slices
        the relevant band and sums it with numpy's vectorised .sum(), which is
        significantly faster than Python loops over np.sum() calls.

    Sub-steps:
      4a — Initialise bounds from bubble geometry (radius-proportional padding).
      4b — Horizontal radar: walk left and right to find ink edges.
      4c — Vertical radar:   walk upward to find the top ink edge.

    Args:
        bubbles:      List of validated TitleBubble objects.
        thresh_radar: Binary radar mask (colour-aware).
        dpi:          Render DPI.

    Returns:
        The same list with left/top/right/bottom populated on each bubble.
    """
    dpi_scale    = dpi / _BASE_DPI
    img_h, img_w = thresh_radar.shape

    # Convert the binary mask to uint16 once to avoid repeated dtype conversions
    # in the radar inner loops. shape: (img_h, img_w), values: 0 or 1.
    logger.info("Step 4  — Pre-computing ink maps")
    ink = (thresh_radar > 0).astype(np.uint16)

    # ── Step 4a: Initialise bounding boxes from bubble geometry ───────────────
    # Using radius-proportional padding (PAD_BUBBLE_RADII) means the initial box
    # automatically scales with sheet size: bigger bubbles on bigger sheets get
    # a proportionally wider safety margin.
    logger.info("Step 4a — Initialising bounds from bubble geometry")
    for b in bubbles:
        pad    = int(b.r * PAD_BUBBLE_RADII)
        b.left   = max(0,     b.cx - b.r - pad)
        b.right  = min(img_w, b.line_end_x + pad)
        b.bottom = min(img_h, b.cy + int(b.r * 0.5) + 50)
        b.top    = 0   # Will be overwritten by vertical radar in Step 4c

    # ── Step 4b: Horizontal boundary radar ───────────────────────────────────
    # Each bubble's left and right edges are refined by scanning the ink density
    # outward from the bubble until a clear gap or neighbour boundary is found.
    logger.info("Step 4b — Running horizontal radar (left + right edges)")
    for b in bubbles:
        left_blockers, right_blockers = _classify_neighbours(b, bubbles)
        b.left, b.right = _horizontal_radar(
            b, thresh_radar, left_blockers, right_blockers, dpi_scale,
            col_sums=ink,
        )

    # ── Step 4c: Vertical boundary radar ─────────────────────────────────────
    # The top edge is found by scanning upward from the bubble. This uses the
    # final left/right edges computed in Step 4b (not the initial geometry) so
    # the vertical scan band is accurate.
    logger.info("Step 4c — Running vertical radar (top edge)")
    for b in bubbles:
        b.top = _vertical_radar(b, bubbles, thresh_radar, dpi_scale, row_sums=ink)

    return bubbles


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3b: Sheet Name Extraction
# ──────────────────────────────────────────────────────────────────────────────

def get_sheet_name(pdf_path: str, page_index: int) -> str:
    """
    Extract the architectural sheet identifier from the bottom-right title block.

    Examples of valid sheet identifiers: "AE203.4", "AE406", "AB100", "S101".

    Detection strategy (four fallback levels):
      1. Group all characters in the bottom-right 25% × 20% corner by font size.
         Test each group (largest font first) against the sheet-ID regex.
      2. Word-level fallback: extract_words() on the same corner, sorted by font size.
      3. Raw fallback: return whatever text the largest font group contains.
      4. Return "SHEET" if all else fails (prevents the pipeline from crashing).

    Args:
        pdf_path:   Path to the source PDF.
        page_index: Zero-based page index.

    Returns:
        Sheet name string (e.g. "AE203.4"), or "SHEET" on failure.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page   = pdf.pages[page_index]
            pw, ph = page.width, page.height

            # Crop the bottom-right corner where the title block is conventionally located
            corner = page.crop((pw * 0.75, ph * 0.80, pw, ph))

            chars = corner.chars
            if not chars:
                # No text characters found in the corner — fallback to "SHEET"
                return "SHEET"

            # Group all character glyphs by their font size.
            # The sheet number is typically printed in the largest font.
            size_groups: Dict[float, str] = defaultdict(str)
            for ch in chars:
                size = round(ch.get("size", 0), 1)
                text = ch.get("text", "")
                if text.strip():
                    size_groups[size] += text

            # Test each size group (largest first) against the sheet-ID pattern
            for size in sorted(size_groups, reverse=True):
                # Collapse whitespace before testing (font groups can have spaces between chars)
                text = re.sub(r"\s+", "", size_groups[size]).strip()
                if _RE_SHEET_ID.match(text):
                    return text

            # Fallback 1: word-level extraction, sorted by font size descending
            for word in sorted(
                corner.extract_words(),
                key=lambda w: float(w.get("size", 0) or 0),
                reverse=True,
            ):
                text = word["text"].strip()
                if _RE_SHEET_ID.match(text):
                    return text

            # Fallback 2: return whatever the largest-font group contains (raw)
            if size_groups:
                largest = max(size_groups)
                return re.sub(r"\s+", "", size_groups[largest]).strip()

    except Exception as exc:
        # Log the failure but do not crash the pipeline — return the safe default
        logger.warning("Sheet name extraction failed for page %d: %s", page_index, exc)

    return "SHEET"


# ──────────────────────────────────────────────────────────────────────────────
# Public API: find_regions_visually
# ──────────────────────────────────────────────────────────────────────────────

def find_regions_visually(
    pdf_path: str,
    page_index: int,
    img: np.ndarray,
    dpi: int = DPI,
) -> Dict[str, Tuple[Tuple[int, int, int, int], str, str]]:
    """
    Master detection function — orchestrates the full detection and boundary pipeline.

    Calls (in order):
        _build_masks()          → two binary masks
        detect_title_bubbles()  → Stage 2 + 3 (circle detection + OCR validation)
        compute_boundaries()    → Stage 4 (radar boundary finding)

    Duplicate ID handling:
        If two drawings on the same page share the same ID (rare but possible),
        the second and subsequent occurrences get a numeric suffix (_2, _3, …).

    Args:
        pdf_path:   Source PDF file path.
        page_index: Zero-based page index.
        img:        Rendered BGR page image (from render_page_to_image).
        dpi:        Render DPI used to calibrate pixel thresholds.

    Returns:
        Dict mapping drawing_id → ((left, top, right, bottom), title, scale).
        Returns an empty dict if no drawings are found.
    """
    # Build both binary masks from the page image
    thresh_bubbles, thresh_radar = _build_masks(img)

    # Stage 2 + 3: detect circles and validate them via OCR
    bubbles = detect_title_bubbles(img, thresh_bubbles, pdf_path, page_index, dpi)
    if not bubbles:
        # No valid drawing bubbles found on this page
        return {}

    # Stage 4: run the boundary radar to get accurate crop boxes
    bubbles = compute_boundaries(bubbles, thresh_radar, dpi)

    # Build the output dictionary, handling duplicate drawing IDs gracefully
    regions: Dict[str, Tuple[Tuple[int, int, int, int], str, str]] = {}
    seen:    Dict[str, int] = {}

    for b in bubbles:
        key = b.num or f"unknown_{b.cy}"

        # If this key was already used, append an incrementing suffix
        if key in seen:
            seen[key] += 1
            key = f"{key}_{seen[key]}"
        else:
            seen[key] = 0

        regions[key] = (
            (b.left, b.top, b.right, b.bottom),
            b.title,
            b.scale,
        )

    return regions


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5: PDF Export
# ──────────────────────────────────────────────────────────────────────────────

def _crop_pdf_region(
    pdf_path: str,
    page_index: int,
    bbox_px: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    out_path: str,
) -> None:
    """
    Crop a rectangular region from a PDF page (vector-quality) and save as a new PDF.

    Mechanism — CropBox (vector-lossless):
        fitz.open() opens the source. insert_pdf() copies the full original page
        into a new document. set_cropbox() sets a PDF-standard CropBox rectangle
        that tells all viewers to display only the specified area.
        The underlying vector drawing data is completely untouched — no rasterisation.

    Coordinate conversion:
        bbox_px is in rendered image pixel space. We convert to PDF point space by
        multiplying by (page_points / image_pixels), i.e. dividing by the render
        scale factor used during image creation.

    Args:
        pdf_path:   Source PDF path.
        page_index: Zero-based page index.
        bbox_px:    (left, top, right, bottom) in rendered image pixel space.
        img_shape:  (height, width) of the rendered image used for coordinate scaling.
        out_path:   Destination file path for the cropped PDF.

    Raises:
        Any fitz exception if the PDF cannot be read or written.
    """
    left, top, right, bottom = bbox_px
    img_h, img_w = img_shape

    # Use a context manager for the source document to guarantee it is closed
    with fitz.open(pdf_path) as src_doc:
        src_page = src_doc[page_index]

        # Compute pixel → PDF point scale factors
        sx = src_page.rect.width  / img_w
        sy = src_page.rect.height / img_h

        # Build the PDF-space crop rectangle
        crop_rect = fitz.Rect(left * sx, top * sy, right * sx, bottom * sy)

        # Create a new single-page document and apply the CropBox
        out_doc = fitz.open()
        out_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
        out_page = out_doc[0]
        out_page.set_cropbox(crop_rect)

        # Save to disk and close
        out_doc.save(out_path)
        out_doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# Public API: extract_drawing
# ──────────────────────────────────────────────────────────────────────────────

def extract_drawing(
    pdf_path: str,
    drawing_ids: Optional[list[str]] = None,
    page_index: int = 0,
    dpi: int = DPI,
    output_dir: str = OUTPUT_DIR,
) -> List[Dict]:
    """
    Full end-to-end pipeline for drawing(s): detect → bound → export.

    Drawing ID lookup is case-insensitive ("w1" matches "W1") so CLI users
    don't need to worry about capitalisation.

    Args:
        pdf_path:   Source PDF path.
        drawing_ids: Target drawing identifier(s) (e.g. ["1", "W3"]). If None or empty, extracts all.
        page_index: Zero-based page index (default 0 = first page).
        dpi:        Render resolution (default 200 DPI).
        output_dir: Root output directory; sub-folders are created per sheet.

    Returns:
        List of Dicts with keys: drawing_number, sheet, title, scale, drawing_path, region_px.
    """
    # Ensure the root output directory exists before doing any heavy work
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %r: %s", output_dir, exc)
        return []

    logger.info("─" * 55)
    logger.info("PDF: %s  |  Page: %d  |  IDs: %s", pdf_path, page_index + 1, drawing_ids or "ALL")

    # ── Sheet identification ───────────────────────────────────────────────────
    # Determine the architectural sheet name (e.g. "AE409") from the title block.
    # All output files for this page are saved under output/<sheet_name>/.
    sheet_name = get_sheet_name(pdf_path, page_index)
    sheet_dir  = os.path.join(output_dir, sheet_name)
    try:
        os.makedirs(sheet_dir, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create sheet directory %r: %s", sheet_dir, exc)
        return []
    logger.info("Sheet: %s  →  %s", sheet_name, sheet_dir)

    # ── Step 1: Render the full page to a high-resolution image ──────────────
    logger.info("Step 1  — Rendering page to image")
    try:
        img = render_page_to_image(pdf_path, page_index, dpi)
    except (FileNotFoundError, ValueError, MemoryError) as exc:
        logger.error("Page render failed: %s", exc)
        return []
    logger.info("Image size: %d×%dpx at %d DPI", img.shape[1], img.shape[0], dpi)

    # ── Steps 2–4: Detect all drawings on the page ───────────────────────────
    # find_regions_visually() runs the full detection + boundary pipeline and
    # returns a dict of all drawing regions found on this page.
    logger.info("Step 2  — Detecting title bubbles")
    try:
        regions = find_regions_visually(pdf_path, page_index, img, dpi)
    except Exception as exc:
        logger.error("Drawing detection failed: %s", exc)
        return []
    logger.info("Total drawings on page: %d", len(regions))

    # ── Drawing ID lookup (case-insensitive) ──────────────────────────────────
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

    results = []
    for t_key in target_keys:
        bbox, drawing_title, drawing_scale = regions[t_key]
        left, top, right, bottom = bbox
        logger.info("Region %r → px (%d,%d) → (%d,%d)", t_key, left, top, right, bottom)

        # ── Step 5: Export the cropped PDF ───────────────────────────────────────
        logger.info("Step 5  — Exporting PDF crop")
        file_stem = f"{sheet_name}-{t_key}"
        out_pdf   = os.path.join(sheet_dir, f"{file_stem}.pdf")
        try:
            _crop_pdf_region(pdf_path, page_index, bbox, img.shape[:2], out_pdf)
        except Exception as exc:
            logger.error("PDF crop/save failed for drawing %r: %s", t_key, exc)
            continue
        logger.info("Saved → %s", out_pdf)

        # ── Result summary ────────────────────────────────────────────────────────
        logger.info("═" * 55)
        logger.info("  Drawing Number : %s", t_key)
        logger.info("  Sheet          : %s", sheet_name)
        logger.info("  Title          : %s", drawing_title)
        logger.info("  Scale          : %s", drawing_scale)
        logger.info("  Drawing PDF    : %s", out_pdf)
        logger.info("═" * 55)

        results.append({
            "drawing_number": t_key,
            "sheet":          sheet_name,
            "title":          drawing_title,
            "scale":          drawing_scale,
            "drawing_path":   out_pdf,
            "region_px":      bbox,
        })

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Public API: list_drawings
# ──────────────────────────────────────────────────────────────────────────────

def list_drawings(pdf_path: str, page_index: int, dpi: int = DPI) -> None:
    """
    Detect and print all drawings on a page without extracting any files.

    Useful for discovering available drawing IDs before running extract_drawing().
    Uses a context manager when reading the page count to prevent file-handle leaks.

    Args:
        pdf_path:   Source PDF path.
        page_index: Zero-based page index.
        dpi:        Render DPI (should match the DPI you plan to use for extraction).
    """
    # Read total page count safely with a context manager
    try:
        with fitz.open(pdf_path) as doc:
            total = len(doc)
    except Exception as exc:
        logger.error("Cannot open PDF %r: %s", pdf_path, exc)
        return

    logger.info("═" * 55)
    logger.info("PDF         : %s  (%d pages)", pdf_path, total)
    logger.info("Scanning    : Page %d", page_index + 1)
    logger.info("═" * 55)

    # Render the page and run the full detection pipeline
    try:
        img = render_page_to_image(pdf_path, page_index, dpi)
    except Exception as exc:
        logger.error("Page render failed: %s", exc)
        return

    sheet_name = get_sheet_name(pdf_path, page_index)

    try:
        regions = find_regions_visually(pdf_path, page_index, img, dpi)
    except Exception as exc:
        logger.error("Drawing detection failed: %s", exc)
        return

    # Print the summary table
    logger.info("─" * 55)
    logger.info("Sheet: %s  |  Drawings found: %d", sheet_name, len(regions))
    logger.info("─" * 55)
    for d_id, (bbox, title, scale) in sorted(regions.items(), key=lambda x: str(x[0])):
        l, t, r, b = bbox
        logger.info("  %-8s | %-40s | %d×%dpx", d_id, title, r - l, b - t)
    logger.info("─" * 55)
    logger.info(
        'To extract:  python drawing_extractor.py "%s" %d <drawing_id>',
        pdf_path, page_index + 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Require at least: <pdf_path> <page_number>
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    # Strip --verbose from the argument list and configure logging accordingly
    _args    = sys.argv[1:]
    _verbose = "--verbose" in _args
    _args    = [a for a in _args if a != "--verbose"]

    _configure_logging(_verbose)

    # Parse positional arguments
    _pdf_path    = _args[0]
    _page_num    = int(_args[1]) - 1   # CLI is 1-based; internally 0-based
    _drawing_ids = _args[2:]

    # Route to list_drawings or extract_drawing based on the first ID argument
    if _drawing_ids and _drawing_ids[0] == "--list":
        list_drawings(_pdf_path, _page_num)
    else:
        extract_drawing(_pdf_path, _drawing_ids if _drawing_ids else None, _page_num)
