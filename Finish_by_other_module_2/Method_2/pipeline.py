import os
import fitz
import json
import base64
import requests
import re
from PIL import Image

import main as main_module # Import main to reuse its existing extraction logic

# ====================================================
# CONFIGURATION
# ====================================================
SUBMITTAL_PDF_PATH = r"C:\Users\Muthamil\view_detect\(Submittal Mods 1-10)_26034 - DGS SREOC Costa Mesa - MOD 1-10 Rev-Axxxx -05-20-2026 (57).pdf"
ARCH_PDF_PATH = r"C:\Users\Muthamil\view_detect\DGS_Arch.pdf"

SUBMITTAL_CROP_DIR = "Submittal_crop"
ARCH_CROP_DIR = "Arch_crop"
from config import OPENROUTER_API_KEY
API_KEY = OPENROUTER_API_KEY
# We use main.py's global constants implicitly if we call its functions, 
# but for the parts we reimplement, we might need some of them.
EXTRA_PADDING = 50

# ====================================================
# HELPER: ARCHITECTURAL SHEET INDEXER
# ====================================================
def build_arch_sheet_index(pdf_path):
    print(f"Building Architectural Sheet Index for {pdf_path}...")
    doc = fitz.open(pdf_path)
    index = {}
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Look for sheet names in the bottom right corner usually
        items = main_module.extract_vector_text(page)
        
        # A simple heuristic to find the sheet number (e.g. AE406)
        # It's usually a short string with uppercase letters and numbers, in the bottom right
        page_w = page.rect.width
        page_h = page.rect.height
        
        candidates = []
        for item in items:
            text = item["text"].strip()
            # Basic sheet number pattern: 1-3 letters followed by numbers (and optional decimal/letter)
            # Examples: A101, AE406, M-201, AE101.1
            if re.match(r'^[A-Z]{1,3}-?\d{1,3}(\.\d+)?[A-Z]?$', text) and len(text) >= 2:
                # Check if it's in the bottom right quadrant
                bbox = item["bbox"]
                if bbox[0] > page_w * 0.7 and bbox[1] > page_h * 0.7:
                    candidates.append((text, bbox))
                    
        # If we found candidates, sort by how bottom-right they are
        if candidates:
            # Sort by sum of x and y coordinates (largest means most bottom-right)
            candidates.sort(key=lambda c: c[1][0] + c[1][1], reverse=True)
            sheet_name = candidates[0][0]
            index[sheet_name] = page_num
            print(f"  Found Sheet {sheet_name} on Page {page_num + 1}")
    
    print(f"Indexed {len(index)} sheets.")
    return index

# ====================================================
# HELPER: SUBMITTAL SHEET EXTRACTOR
# ====================================================
def get_submittal_sheet_number(page):
    items = main_module.extract_vector_text(page)
    page_w = page.rect.width
    page_h = page.rect.height
    candidates = []
    for item in items:
        text = item["text"].strip()
        # Submittal sheets often look like "1.1", "1.4", "A1", etc.
        if re.match(r'^[A-Z0-9\.-]+$', text) and 2 <= len(text) <= 8:
            bbox = item["bbox"]
            # Look in bottom right quadrant
            if bbox[0] > page_w * 0.7 and bbox[1] > page_h * 0.7:
                candidates.append((text, bbox))
                
    if candidates:
        candidates.sort(key=lambda c: c[1][0] + c[1][1], reverse=True)
        return candidates[0][0]
    return ""

# ====================================================
# STEP 1: SUBMITTAL CROP
# ====================================================
def crop_drawings_from_page(page, page_num, model, processor, device, target_drawing="", is_arch=False, arch_sheet="", pdf_basename=""):
    """
    Reuses main.py's detection logic to find bounding boxes, but crops the image
    and saves it into the target folder structure.
    """
    items = main_module.extract_vector_text(page)
    title_blocks = main_module.group_title_blocks(items)
    
    if not title_blocks:
        print(f"  No title blocks found on page {page_num}. Skipping.")
        return []
        
    drawing_clusters = main_module.get_dino_clusters(page, model, processor, device)
    
    final_drawings_pdf_bboxes = []
    
    # Same logic as main.py process_page to map titles to clusters
    if len(title_blocks) == 1:
        all_x0, all_y0, all_x1, all_y1 = [], [], [], []
        t = title_blocks[0]
        num_bbox = t["number"]["bbox"]
        tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
        sca_bbox = t["scale"]["bbox"]
        for x in [num_bbox, tit_bbox, sca_bbox]:
            all_x0.append(x[0]); all_y0.append(x[1]); all_x1.append(x[2]); all_y1.append(x[3])
            
        for drawing in page.get_drawings():
            r = drawing["rect"]
            if r.width < 1 or r.height < 1: continue
            if (r.x1 - r.x0) > page.rect.width * 0.3 or (r.y1 - r.y0) > page.rect.height * 0.5: continue
            all_x0.append(r.x0); all_y0.append(r.y0); all_x1.append(r.x1); all_y1.append(r.y1)
            
        for i in items:
            if i["bbox"][0] > page.rect.width * 0.85: continue
            all_x0.append(i["bbox"][0]); all_y0.append(i["bbox"][1]); all_x1.append(i["bbox"][2]); all_y1.append(i["bbox"][3])
            
        if all_x0:
            fx0 = min(all_x0) - EXTRA_PADDING
            fy0 = min(all_y0) - EXTRA_PADDING
            fx1 = max(all_x1) + EXTRA_PADDING
            fy1 = max(all_y1) + EXTRA_PADDING
            final_drawings_pdf_bboxes.append({
                "number_text": title_blocks[0]["number"]["text"],
                "title_text": title_blocks[0]["title"]["text"] if title_blocks[0]["title"] else "",
                "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)]
            })
    else:
        for t in title_blocks:
            num_bbox = t["number"]["bbox"]
            tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
            sca_bbox = t["scale"]["bbox"]
            
            all_x0 = [x[0] for x in [num_bbox, tit_bbox, sca_bbox]]
            all_y0 = [x[1] for x in [num_bbox, tit_bbox, sca_bbox]]
            all_x1 = [x[2] for x in [num_bbox, tit_bbox, sca_bbox]]
            all_y1 = [x[3] for x in [num_bbox, tit_bbox, sca_bbox]]
            tx0, ty0, tx1, ty1 = min(all_x0), min(all_y0), max(all_x1), max(all_y1)
            tx_c = (tx0 + tx1) / 2.0
            
            best_cluster = None
            min_score = float('inf')
            
            for cx0, cy0, cx1, cy1 in drawing_clusters:
                ccx = (cx0 + cx1) / 2.0
                ccy = (cy0 + cy1) / 2.0
                if ccy > ty0: continue
                dx = abs(ccx - tx_c)
                dy = abs(ty0 - cy1)
                
                # Must be reasonably close horizontally and vertically
                if dx > page.rect.width * 0.15: continue
                #if dy > page.rect.height * 0.25: continue # Reject drawings that are too far above the title
                
                score = (dx * 10.0) + dy
                if score < min_score:
                    min_score = score
                    best_cluster = [cx0, cy0, cx1, cy1]
            
            if best_cluster:
                fx0 = min(best_cluster[0], tx0) - EXTRA_PADDING
                fy0 = min(best_cluster[1], ty0) - EXTRA_PADDING
                fx1 = max(best_cluster[2], tx1) + EXTRA_PADDING
                fy1 = max(best_cluster[3], ty1) + EXTRA_PADDING
            else:
                fx0, fy0, fx1, fy1 = tx0 - EXTRA_PADDING, ty0 - EXTRA_PADDING, tx1 + EXTRA_PADDING, ty1 + EXTRA_PADDING
                
            final_drawings_pdf_bboxes.append({
                "number_text": t["number"]["text"],
                "title_text": t["title"]["text"] if t["title"] else "",
                "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)]
            })

    # Target drawing filter (used for Arch PDF cropping)
    if target_drawing:
        # Strip and normalize to improve matching
        norm_target = target_drawing.strip().lower()
        final_drawings_pdf_bboxes = [d for d in final_drawings_pdf_bboxes if d["number_text"].strip().lower() == norm_target]
        print(f"  Filtered to {len(final_drawings_pdf_bboxes)} matching '{target_drawing}'")

    if not final_drawings_pdf_bboxes:
        return []

    # Get 300 DPI image
    scale_factor = 300.0 / 72.0
    mat = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    cropped_paths = []
    
    # Save the crops
    for i, data in enumerate(final_drawings_pdf_bboxes):
        pdf_bbox = data["pdf_bbox"]
        
        # Scale bbox to 300 DPI
        px_pdf = [int(b * scale_factor) for b in pdf_bbox]
        
        cropped_img = img.crop((px_pdf[0], px_pdf[1], px_pdf[2], px_pdf[3]))
        
        bubble_label = "".join(c for c in data['number_text'] if c.isalnum() or c in " -_").strip()
        if not bubble_label:
            bubble_label = str(i + 1)
            
        if not is_arch:
            submittal_sheet = get_submittal_sheet_number(page)
            if not submittal_sheet:
                submittal_sheet = str(page_num)
                
            # Include the PDF name in the folder path to avoid overwriting previous PDFs
            page_folder = os.path.join(SUBMITTAL_CROP_DIR, pdf_basename, submittal_sheet)
            os.makedirs(page_folder, exist_ok=True)
            out_filename = f"{submittal_sheet}_{bubble_label}.png"
            out_path = os.path.join(page_folder, out_filename)
        else:
            # Save to Arch_crop/AE406/AE406_A.png
            sheet_folder = os.path.join(ARCH_CROP_DIR, arch_sheet)
            os.makedirs(sheet_folder, exist_ok=True)
            out_filename = f"{arch_sheet}_{bubble_label}.png"
            out_path = os.path.join(sheet_folder, out_filename)
            
        cropped_img.save(out_path, format="PNG")
        print(f"  Saved crop: {out_path}")
        cropped_paths.append({
            "path": out_path,
            "title_text": data.get("title_text", "")
        })
        
    return cropped_paths


# ====================================================
# STEP 2: METADATA EXTRACTION (LLM)
# ====================================================
def extract_metadata_from_crop(image_path, title_text=""):
    if not API_KEY:
        print("Warning: API_KEY is missing. Skipping Qwen extraction.")
        return None
        
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are analyzing a cropped architectural drawing.

The exact title written on this drawing is: "{title_text}"

Extract the following metadata:
1. view_name: The title/name of the view. CRITICAL: You MUST output the exact title text provided above! Do not guess!
2. arch_ref: The architectural reference (e.g. 2/AE406), if present. Sometimes written as "SEE 2/AE406" or just "2/AE406". Leave empty string if none.
3. view_type: The type of view (e.g. Plan, Section, Elevation, Detail)

Return ONLY a JSON object:
{{
  "view_name": "{title_text if title_text else 'SECTION A'}",
  "arch_ref": "2/AE406",
  "view_type": "Section"
}}"""

    with open(image_path, "rb") as image_file:
        b64_string = base64.b64encode(image_file.read()).decode("utf-8")
        
    payload = {
        "model": "qwen/qwen3-vl-32b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}}
                ]
            }
        ],
        "max_tokens": 512
    }
    
    print(f"  Querying Qwen for {os.path.basename(image_path)}...")
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return data
        else:
            print("  API Error:", response.text)
    except Exception as e:
        print("  Failed to extract:", e)
        
    return None

# ====================================================
# PIPELINE ORCHESTRATOR
# ====================================================
def run_full_pipeline(submittal_pdf_path, arch_pdf_path):
    # Setup Grounding DINO
    model, processor, device = main_module.load_dino_model()
    
    # 1. Crop Submittal PDF
    print(f"\n--- STEP 1: Cropping Submittal PDF ---")
    submittal_doc = fitz.open(submittal_pdf_path)
    pdf_basename = os.path.splitext(os.path.basename(submittal_pdf_path))[0]
    all_submittal_crops = []
    
    for i in range(len(submittal_doc)):
        page_num = i + 1
        print(f"\nProcessing Submittal Page {page_num}/{len(submittal_doc)}")
        crops = crop_drawings_from_page(submittal_doc[i], page_num, model, processor, device, pdf_basename=pdf_basename)
        all_submittal_crops.extend(crops)
        
    if not all_submittal_crops:
        print("No drawings found in Submittal PDF.")
        return

    # Build Index for Arch PDF
    print(f"\n--- BUILDING ARCHITECTURAL INDEX ---")
    arch_index = build_arch_sheet_index(arch_pdf_path)
    arch_doc = fitz.open(arch_pdf_path)
    
    # 2 & 3. Extract Metadata and Crop Arch References
    print(f"\n--- STEP 2 & 3: Extracting Metadata & Cropping Arch Refs ---")
    extracted_metadata = []
    
    for crop_info in all_submittal_crops:
        crop_path = crop_info["path"]
        title_text = crop_info["title_text"]
        meta = extract_metadata_from_crop(crop_path, title_text)
        if meta:
            meta["source_file"] = crop_path
            extracted_metadata.append(meta)
            print(f"  Extracted: {meta}")
            
            arch_ref = meta.get("arch_ref", "")
            if arch_ref and "/" in arch_ref:
                parts = arch_ref.split("/")
                drawing_number = parts[0].strip()
                sheet_name = parts[1].strip()
                
                print(f"  --> Identified Arch Ref: Sheet {sheet_name}, Drawing {drawing_number}")
                
                if sheet_name in arch_index:
                    page_idx = arch_index[sheet_name]
                    print(f"  Locating Sheet {sheet_name} on Page {page_idx + 1}...")
                    
                    arch_page = arch_doc[page_idx]
                    crop_drawings_from_page(
                        arch_page, page_idx + 1, model, processor, device, 
                        target_drawing=drawing_number, 
                        is_arch=True, arch_sheet=sheet_name
                    )
                else:
                    print(f"  Warning: Sheet {sheet_name} not found in index.")
            else:
                print(f"  No valid arch_ref found for {crop_path}.")

    # Save metadata JSON
    out_json = f"submittal_metadata_{pdf_basename}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(extracted_metadata, f, indent=2)
    print(f"\n✅ Pipeline complete. Metadata saved to {out_json}.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        sub_pdf = sys.argv[1]
        arch_pdf = sys.argv[2]
        run_full_pipeline(sub_pdf, arch_pdf)
    else:
        print("Usage: python pipeline.py <submittal_pdf_path> <arch_pdf_path>")
        print(f"Falling back to default configured paths...")
        run_full_pipeline(SUBMITTAL_PDF_PATH, ARCH_PDF_PATH)
