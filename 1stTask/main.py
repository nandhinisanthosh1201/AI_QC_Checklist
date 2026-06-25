"""
main.py
-------
Entry point for Stage 1: PDF to High-Resolution Page Images.

Usage
-----
    py main.py

How it works
------------
Drop ANY number of PDF files into the ``input/`` folder and run this script.
Every PDF is discovered automatically — no filenames need to be configured.

Output layout (example for "arch_plan.pdf"):
    output/
    ├── arch_plan_pages/
    │   ├── arch_plan_page_1.png
    │   ├── arch_plan_page_2.png
    │   └── ...
    └── metadata/
        └── stage1_metadata.json
"""

from __future__ import annotations

import sys
from pathlib import Path

from config import INPUT_DIR, OUTPUT_DIR, RENDER_DPI, COLORSPACE, IMAGE_FORMAT
from config import METADATA_DIR, METADATA_FILENAME
from pdf_converter import convert_pdf
from metadata_writer import save_metadata
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# PDF discovery
# ---------------------------------------------------------------------------

def discover_pdfs(input_dir: Path) -> list[Path]:
    """
    Return all PDF files found directly inside *input_dir* (non-recursive).

    Parameters
    ----------
    input_dir : Path
        Directory to scan.

    Returns
    -------
    list[Path]
        Sorted list of PDF paths.

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist.
    RuntimeError
        If no PDF files are found.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir.resolve()}\n"
            "Create the 'input/' folder and place your PDFs inside it."
        )

    pdfs = sorted(input_dir.glob("*.pdf"))

    if not pdfs:
        raise RuntimeError(
            f"No PDF files found in: {input_dir.resolve()}\n"
            "Place at least one PDF file inside the 'input/' folder and retry."
        )

    return pdfs


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage1() -> Path:
    """
    Discover all PDFs in ``input/``, convert each to per-page images,
    and write a consolidated metadata JSON.

    Returns
    -------
    Path
        Path to the generated ``stage1_metadata.json`` file.

    Raises
    ------
    SystemExit
        On any unrecoverable error (missing directory, no PDFs, etc.).
    """
    log.info("=" * 60)
    log.info("STAGE 1 — PDF to High-Resolution Page Images")
    log.info("  Input dir  : %s", INPUT_DIR.resolve())
    log.info("  Output dir : %s", OUTPUT_DIR.resolve())
    log.info("  DPI        : %d", RENDER_DPI)
    log.info("  Colour     : %s", COLORSPACE)
    log.info("  Format     : %s", IMAGE_FORMAT.upper())
    log.info("=" * 60)

    # ── Discover PDFs ───────────────────────────────────────────────────────
    try:
        pdf_files = discover_pdfs(INPUT_DIR)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info("Found %d PDF(s) to process:", len(pdf_files))
    for pdf in pdf_files:
        log.info("  • %s", pdf.name)

    # ── Convert each PDF ────────────────────────────────────────────────────
    all_records: list[dict] = []
    errors: list[str] = []

    for pdf_path in pdf_files:
        try:
            records = convert_pdf(
                pdf_path=pdf_path,
                output_dir=OUTPUT_DIR,
                dpi=RENDER_DPI,
                colorspace=COLORSPACE,
                image_format=IMAGE_FORMAT,
            )
            all_records.extend(records)
        except FileNotFoundError as exc:
            log.error(str(exc))
            errors.append(str(exc))
        except Exception as exc:                         # noqa: BLE001
            log.exception("[%s] Unexpected error: %s", pdf_path.name, exc)
            errors.append(f"[{pdf_path.name}] {exc}")

    if errors:
        log.error("-" * 60)
        log.error("Stage 1 finished with %d error(s):", len(errors))
        for err in errors:
            log.error("  x %s", err)
        sys.exit(1)

    # ── Persist metadata ────────────────────────────────────────────────────
    metadata_path = save_metadata(
        records=all_records,
        output_dir=METADATA_DIR,
        filename=METADATA_FILENAME,
    )

    log.info("=" * 60)
    log.info("Stage 1 complete.")
    log.info("  PDFs processed : %d", len(pdf_files))
    log.info("  Total images   : %d", len(all_records))
    log.info("  Metadata file  : %s", metadata_path)
    log.info("=" * 60)

    return metadata_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_stage1()
