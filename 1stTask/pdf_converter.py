"""
pdf_converter.py
----------------
Core module for Stage 1: converts a single PDF into per-page PNG images.

Public API
----------
convert_pdf(pdf_path, output_dir, dpi, colorspace, image_format)
    → list[dict]   (page-level metadata records)
"""

from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import Any

from config import RENDER_DPI, COLORSPACE, IMAGE_FORMAT
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_matrix(dpi: int) -> fitz.Matrix:
    """
    Convert a DPI value into a PyMuPDF transformation matrix.

    PyMuPDF's native resolution is 72 DPI, so the scale factor is dpi/72.
    """
    scale = dpi / 72.0
    return fitz.Matrix(scale, scale)


def _fitz_colorspace(colorspace: str) -> fitz.Colorspace:
    """Return the fitz.Colorspace object matching a colour-space name string."""
    cs_map = {
        "RGB": fitz.csRGB,
        "GRAY": fitz.csGRAY,
        "CMYK": fitz.csCMYK,
    }
    cs = cs_map.get(colorspace.upper())
    if cs is None:
        raise ValueError(
            f"Unsupported colorspace '{colorspace}'. Choose from: {list(cs_map)}"
        )
    return cs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = RENDER_DPI,
    colorspace: str = COLORSPACE,
    image_format: str = IMAGE_FORMAT,
) -> list[dict[str, Any]]:
    """
    Convert every page of a PDF into a raster image.

    The output folder is created automatically as::

        <output_dir>/<pdf_stem>_pages/

    For example, a file named ``arch_drawing.pdf`` produces images in::

        output/arch_drawing_pages/arch_drawing_page_1.png
        output/arch_drawing_pages/arch_drawing_page_2.png
        ...

    Parameters
    ----------
    pdf_path : Path
        Absolute or relative path to the source PDF.
    output_dir : Path
        Root output directory.  A sub-folder named ``<stem>_pages`` is
        created inside it automatically.
    dpi : int
        Render resolution in dots-per-inch.
    colorspace : str
        ``"RGB"``, ``"GRAY"``, or ``"CMYK"``.
    image_format : str
        Image extension without a leading dot, e.g. ``"png"``.

    Returns
    -------
    list[dict]
        One record per page::

            {
                "pdf_name":    "arch_drawing.pdf",
                "page_number": 1,
                "image_path":  "C:\\...\\arch_drawing_page_1.png"
            }

    Raises
    ------
    FileNotFoundError
        If the PDF does not exist at the given path.
    fitz.FileDataError
        If the file cannot be opened as a valid PDF.
    """
    pdf_path = Path(pdf_path)
    pdf_stem = pdf_path.stem                          # filename without extension
    file_prefix = "page"

    # ── Validate input ──────────────────────────────────────────────────────
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path.resolve()}\n"
            "Ensure the file is inside the 'input/' directory."
        )

    # ── Prepare output directory  (<output_dir>/<stem>_pages/) ─────────────
    output_folder = Path(output_dir) / f"{pdf_stem}_pages"
    output_folder.mkdir(parents=True, exist_ok=True)
    log.info("[%s] Output folder ready -> %s", pdf_stem, output_folder.resolve())

    # ── Open PDF ────────────────────────────────────────────────────────────
    matrix = _build_matrix(dpi)
    cs = _fitz_colorspace(colorspace)
    metadata_records: list[dict[str, Any]] = []

    log.info(
        "[%s] Opening PDF: %s  (%d DPI, %s colour space)",
        pdf_stem, pdf_path.name, dpi, colorspace,
    )

    with fitz.open(str(pdf_path)) as doc:
        total_pages = doc.page_count
        log.info("[%s] Total pages: %d", pdf_stem, total_pages)

        for page_index in range(total_pages):
            page_number = page_index + 1          # 1-based numbering
            page = doc.load_page(page_index)

            # Render page to a Pixmap
            pixmap: fitz.Pixmap = page.get_pixmap(
                matrix=matrix, colorspace=cs, alpha=False
            )

            # e.g. page1.png
            filename = f"{file_prefix}{page_number}.{image_format}"
            image_path = output_folder / filename
            pixmap.save(str(image_path))

            log.info(
                "[%s] Page %d/%d saved -> %s",
                pdf_stem, page_number, total_pages, filename,
            )

            metadata_records.append(
                {
                    "pdf_name": pdf_path.name,
                    "page_number": page_number,
                    "image_path": str(image_path.resolve()),
                }
            )

    log.info(
        "[%s] Conversion complete. %d image(s) generated.",
        pdf_stem, total_pages,
    )
    return metadata_records
