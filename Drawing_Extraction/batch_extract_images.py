"""
Batch Drawing Image Extractor
==============================
Scans EVERY page of a PDF, auto-detects ALL drawings on each page,
and saves each drawing as a PNG image into a single flat output folder.

Usage:
    python batch_extract_images.py <pdf_path> [output_folder]

Examples:
    python batch_extract_images.py DGS_Arch.pdf
    python batch_extract_images.py DGS_Arch.pdf my_output_images

Output naming format:
    <OutputFolder>/<SheetName>-<DrawingID>.png
    e.g.  all_drawings/AE203.4-W1.png
          all_drawings/AE406-1.png
"""

import sys
import os
import cv2
import fitz  # PyMuPDF

# ─────────────────────────────────────────────────────
# CONFIG — edit these as needed
# ─────────────────────────────────────────────────────
PDF_PATH      = "DGS_Arch.pdf"   # ← Change this to your PDF filename
OUTPUT_FOLDER = "all_drawings"   # ← All PNGs land in this single folder
IMAGE_DPI     = 100              # Render DPI (higher = sharper image, slower)
# ─────────────────────────────────────────────────────

# Import the core logic from the existing extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drawing_extractor import render_page_to_image, find_regions_visually, get_sheet_name


def extract_all_as_images(pdf_path: str, output_folder: str):
    """
    Iterate every page, detect all drawings, crop each one from the
    rendered image, and save it as a PNG file.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get total page count
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    print(f"\n{'═'*60}")
    print(f"  PDF           : {pdf_path}")
    print(f"  Total Pages   : {total_pages}")
    print(f"  Output Folder : {output_folder}")
    print(f"  Render DPI    : {IMAGE_DPI}")
    print(f"{'═'*60}\n")

    total_saved   = 0
    total_skipped = 0
    failed_pages  = []

    for page_index in range(total_pages):
        page_num = page_index + 1
        print(f"─── Page {page_num}/{total_pages} ───────────────────────────────────")

        try:
            # 1. Render the full page to a numpy image
            img = render_page_to_image(pdf_path, page_index, IMAGE_DPI)

            # 2. Detect all drawings on this page
            sheet_name = get_sheet_name(pdf_path, page_index)
            regions    = find_regions_visually(pdf_path, page_index, img, dpi=IMAGE_DPI)

            if not regions:
                print(f"   ⚠️  No drawings found on page {page_num} ({sheet_name}). Skipping.")
                total_skipped += 1
                continue

            print(f"   Sheet: {sheet_name}  |  Drawings found: {len(regions)}")

            # 3. For each detected drawing, crop and save as PNG
            for drawing_id, (bbox, title, scale) in regions.items():
                left, top, right, bottom = bbox

                # Guard against degenerate bounding boxes
                if right <= left or bottom <= top:
                    print(f"   ⚠️  Drawing {drawing_id}: invalid bbox {bbox}, skipping.")
                    total_skipped += 1
                    continue

                # Crop from the rendered image
                cropped = img[top:bottom, left:right]

                # Build output filename:  Pg14_AE203.4-W1.png
                safe_id    = str(drawing_id).replace("/", "_").replace("\\", "_")
                filename   = f"Pg{page_num:02d}_{sheet_name}-{safe_id}.png"
                out_path   = os.path.join(output_folder, filename)

                # If the same name already exists, use an incrementing suffix
                original_out_path = out_path
                counter = 1
                while os.path.exists(out_path):
                    base, ext = os.path.splitext(original_out_path)
                    out_path  = f"{base}_dup{counter}{ext}"
                    filename  = os.path.basename(out_path)
                    counter  += 1

                cv2.imwrite(out_path, cropped)
                print(f"   ✅  Saved: {filename}  [{right-left}×{bottom-top}px]  \"{title}\"")
                total_saved += 1

        except Exception as e:
            print(f"   ❌  Page {page_num} failed: {e}")
            failed_pages.append((page_num, str(e)))

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅  Saved    : {total_saved} drawing image(s)")
    print(f"  ⚠️   Skipped  : {total_skipped} page(s) with no drawings")
    if failed_pages:
        print(f"  ❌  Failed   : {len(failed_pages)} page(s)")
        for pn, err in failed_pages:
            print(f"       Page {pn}: {err}")
    print(f"\n  Output folder : {os.path.abspath(output_folder)}")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Allow overriding PDF path and output folder from the command line
    pdf    = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    folder = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_FOLDER

    if not os.path.isfile(pdf):
        print(f"\n❌  File not found: '{pdf}'")
        print("Usage: python batch_extract_images.py <pdf_path> [output_folder]\n")
        sys.exit(1)

    extract_all_as_images(pdf, folder)
