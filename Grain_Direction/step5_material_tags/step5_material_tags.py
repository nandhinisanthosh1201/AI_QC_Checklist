"""
step5_material_tags.py
======================
Production-ready material tag detector.

Reads the material tags extracted in Step 1, then finds and marks every
occurrence of those tags on the Step 4 YOLO-annotated elevation images.

Works with ANY project — no hardcoded paths or tags.

Usage:
    python step5_material_tags.py \
        --pdf       <original PDF path> \
        --manifest  <Step 3 manifest.csv path> \
        --step1     <Step 1 output dir with materials_cache.json> \
        --step4     <Step 4 output dir with detected_*.png images> \
        --output    <Step 5 output dir to save results>

Example:
    python step5_material_tags.py \
        --pdf      "inputs/Submittal_26022 - Docusign.pdf" \
        --manifest "outputs/Step3/manifest.csv" \
        --step1    "outputs/Step1_MaterialScraping/Docusign" \
        --step4    "outputs/Step4_ComponentDetection/Docusign" \
        --output   "outputs/Step5_MaterialTags/Docusign"
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import fitz  # PyMuPDF

# MUST match the zoom used in test1 1.py (default=2.5)
ZOOM = 2.5

# Pattern: 2-4 uppercase letters, hyphen, 1-2 alphanumeric chars
# Matches: WV-A, MM-A, PL-C, SS-1, QZ-A, LAM-A, MWT-A, etc.
TAG_PATTERN = re.compile(r"\b([A-Z]{2,4}-[A-Z0-9]{1,2})\b")

# Minimum pixel size of a valid tag box (filters ghost detections)
MIN_PX_SIZE = 8

# Tight boundary tolerance (PDF points) around the crop area
CROP_PAD = 5.0


def load_material_tags(step1_dir: Path) -> set:
    """
    Load the set of material tags from Step 1's materials_cache.json.
    If the cache is missing or empty, returns an empty set (meaning
    the detector will match ALL tag-like patterns in the drawing).
    """
    cache_path = step1_dir / "materials_cache.json"
    if not cache_path.exists():
        print(f"[WARN] materials_cache.json not found at {cache_path}.")
        print("[WARN] Will detect ALL tag-like patterns (no Step 1 filter applied).")
        return set()

    with open(cache_path, "r", encoding="utf-8") as f:
        materials = json.load(f)

    tags = set(m["tag"].strip().upper() for m in materials if "tag" in m and m["tag"].strip())
    print(f"[INFO] Step 1 material tags loaded ({len(tags)}): {sorted(tags)}")
    return tags


def process(pdf_path: Path, manifest_path: Path, step1_dir: Path,
            step4_dir: Path, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    material_tags = load_material_tags(step1_dir)

    doc = fitz.open(str(pdf_path))

    total_images = 0
    total_tags = 0
    tags_data = []

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        # page_no in manifest is 0-indexed (written by test1 1.py's `pno` loop var)
        page_no = int(row["page"])
        crop_file_name = row["crop_file"]
        print(f"  [crop {idx+1:02d}/{len(rows):02d}] Scanning {crop_file_name} for material tags...", flush=True)

        step4_img_name = f"detected_{crop_file_name}"
        step4_img_path = step4_dir / step4_img_name

        if not step4_img_path.exists():
            print(f"[SKIP] {step4_img_name} not found in Step 4 directory.")
            continue

        img = cv2.imread(str(step4_img_path))
        if img is None:
            print(f"[SKIP] Cannot read image: {step4_img_path}")
            continue

        # Crop bounding box in original PDF points
        x0 = float(row["x0"])
        y0 = float(row["y0"])
        x1 = float(row["x1"])
        y1 = float(row["y1"])

        # Page access: 0-indexed directly
        if page_no >= len(doc):
            print(f"[SKIP] Page {page_no} out of range for this PDF.")
            continue
        page = doc[page_no]
        words = page.get_text("words")

        # Create exclusion zones to avoid detecting tags in material tables, schedules, or title blocks
        exclusion_zones = []
        for w in words:
            w_text = w[4].strip().upper()
            if w_text in ['MATERIAL', 'MATERIALS', 'MATL', 'FINISH', 'HARDWARE', 'HDWE', 'SCHEDULE', 'LEGEND', 'CRESTMARK']:
                zx0, zy0 = w[0], w[1]
                # Store anchor coordinates (zx0, zy0) to verify if it lies outside the crop
                exclusion_zones.append((zx0 - 30, zy0 - 10, zx0 + 350, zy0 + 150, zx0, zy0))

        tags_found = 0

        for w in words:
            wx0, wy0, wx1, wy1, text = w[0], w[1], w[2], w[3], w[4]
            text_clean = text.strip().upper()

            match = TAG_PATTERN.search(text_clean)
            if not match:
                continue

            # Check word is within the crop boundary (with small tolerance) FIRST
            if not (wx0 >= x0 - CROP_PAD and wy0 >= y0 - CROP_PAD and
                    wx1 <= x1 + CROP_PAD and wy1 <= y1 + CROP_PAD):
                continue

            # Check if tag is inside an exclusion zone (e.g. material table or title block)
            in_zone = False
            for zx0, zy0, zx1, zy1, hx0, hy0 in exclusion_zones:
                # If the table header was outside the crop, it must NEVER suppress tags inside the crop!
                if hy0 < y0 and wy0 >= y0:
                    continue
                if hy0 > y1 and wy1 <= y1:
                    continue
                if hx0 < x0 and wx0 >= x0:
                    continue
                if hx0 > x1 and wx1 <= x1:
                    continue
                if wx0 >= zx0 and wy0 >= zy0 and wx1 <= zx1 and wy1 <= zy1:
                    in_zone = True
                    break
            
            if in_zone:
                continue

            found_tag = match.group(1)

            # Filter: if Step 1 tags available, only mark those.
            # If Step 1 is empty, mark ALL pattern-matched tags.
            if material_tags and found_tag not in material_tags:
                continue

            # Check word is within the crop boundary (with small tolerance)
            if not (wx0 >= x0 - CROP_PAD and wy0 >= y0 - CROP_PAD and
                    wx1 <= x1 + CROP_PAD and wy1 <= y1 + CROP_PAD):
                continue

            # Convert PDF coords to crop-relative pixel coords
            px0 = int((wx0 - x0) * ZOOM)
            py0 = int((wy0 - y0) * ZOOM)
            px1 = int((wx1 - x0) * ZOOM)
            py1 = int((wy1 - y0) * ZOOM)

            # Clamp to image bounds
            h_img, w_img = img.shape[:2]
            px0 = max(0, min(px0, w_img))
            py0 = max(0, min(py0, h_img))
            px1 = max(0, min(px1, w_img))
            py1 = max(0, min(py1, h_img))

            if px1 <= px0 or py1 <= py0:
                continue

            if (px1 - px0) < MIN_PX_SIZE or (py1 - py0) < MIN_PX_SIZE:
                continue

            # Clamp again after expansion
            px1 = min(px1, w_img)
            py1 = min(py1, h_img)

            # Draw orange bounding box
            cv2.rectangle(img, (px0, py0), (px1, py1), (0, 140, 255), 2)

            # Draw filled label background + white text
            label = f"MAT: {found_tag}"
            label_y = max(18, py0 - 5)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (px0, label_y - th - 4), (px0 + tw + 4, label_y + 2),
                          (0, 140, 255), -1)
            cv2.putText(img, label, (px0 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            tags_found += 1
            total_tags += 1
            
            # Save tag data for Step 6
            tags_data.append({
                "file": step4_img_name,
                "tag": found_tag,
                "x0": px0,
                "y0": py0,
                "x1": px1,
                "y1": py1
            })

        out_path = out_dir / step4_img_name
        ok = cv2.imwrite(str(out_path), img)
        if not ok:
            print(f"[ERROR] Failed to write: {out_path}")
        else:
            total_images += 1
            print(f"[OK] {step4_img_name}: {tags_found} tag(s) marked")

    doc.close()
    
    # Save the tags JSON for Step 6
    import json
    json_path = out_dir / "tags_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tags_data, f, indent=4)
        
    print(f"\nDone. {total_tags} tag(s) across {total_images} image(s).")
    print(f"Step 5 outputs saved to: {out_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Step 5: Detect and mark material tags on Step 4 elevation images."
    )
    ap.add_argument("--pdf",      required=True, help="Path to the original input PDF")
    ap.add_argument("--manifest", required=True, help="Path to Step 3 manifest.csv")
    ap.add_argument("--step1",    required=True, help="Step 1 output dir (contains materials_cache.json)")
    ap.add_argument("--step4",    required=True, help="Step 4 output dir (contains detected_*.png)")
    ap.add_argument("--output",   required=True, help="Step 5 output dir to save results")
    args = ap.parse_args()

    process(
        pdf_path=Path(args.pdf),
        manifest_path=Path(args.manifest),
        step1_dir=Path(args.step1),
        step4_dir=Path(args.step4),
        out_dir=Path(args.output),
    )


if __name__ == "__main__":
    main()
