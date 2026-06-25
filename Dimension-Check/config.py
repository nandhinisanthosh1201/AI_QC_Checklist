"""
config.py
---------
Central configuration for Stage 1: PDF → Image Conversion.

No PDF filenames are hardcoded here.
The pipeline auto-discovers every *.pdf inside INPUT_DIR at runtime.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

# Folder scanned for PDF files.
# Set to Path(".") to pick up PDFs from the project root itself,
# or Path("input") if you prefer a dedicated subfolder.
INPUT_DIR = Path(".")

# Root folder for all generated images and metadata
OUTPUT_DIR = Path("output")

# Metadata JSON is written here
METADATA_DIR = OUTPUT_DIR / "metadata"
METADATA_FILENAME = "stage1_metadata.json"

# ---------------------------------------------------------------------------
# Rendering settings
# ---------------------------------------------------------------------------

# DPI for rasterisation (300 is print-quality; raise for finer detail)
RENDER_DPI: int = 300

# Output image format — PNG keeps lossless quality for downstream OCR / CV
IMAGE_FORMAT: str = "png"

# PyMuPDF colour space: "RGB" for colour, "GRAY" for grayscale
COLORSPACE: str = "RGB"
