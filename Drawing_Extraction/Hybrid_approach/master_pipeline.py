import fitz
import json
import base64
import requests
import io
import os
from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ====================================================
# CONFIG
# ====================================================

MODEL_ID = "IDEA-Research/grounding-dino-base"
TEXT_PROMPT = "countertop . sink . cabinet . casework . shelf ."
BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10

API_KEY = ""
PDF_PATH = r"D:\Drawing_Extraction\VERSION2\DGS_Arch.pdf"
PAGE_NUM = 20  # 0-indexed (Page 1)

VERIFICATION_DIR = "outputnew"
EXTRA_PADDING = 50# Points

# Set to a specific drawing number (e.g., "10") to focus ONLY on that drawing. 
# Set to "" to process all drawings on the page.
# TARGET_DRAWING = "A"
TARGET_DRAWING = ""

# Add a custom question you want to ask Qwen about every drawing!
# Leave it as an empty string "" if you don't want to ask anything.
USER_QUESTION = "can you detect all the by others notes in this drawing, Answer briefly."

# ----------------------------------------------------
# STANDARD FORMAT RULES (CONFIGURE PER PROJECT)
# ----------------------------------------------------
# These keywords act as the "anchor" to locate the Title Block.
SCALE_KEYWORDS = ["scale:", "1/4\"", "1/8\"", "1'-", "nts", "n.t.s", "=\""]

# ====================================================
# 1. EXTRACT VECTOR TEXT
# ====================================================
def extract_vector_text(page):
    print("Extracting vector text from PDF...")
    page_dict = page.get_text("dict")
    items = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    items.append({"text": text, "bbox": span["bbox"]})
    return items

# ====================================================
# 2. GROUP INTO TITLE BLOCKS
# ====================================================
def group_title_blocks(items):
    print("Grouping text into cohesive Title Blocks...")
    # Using the globally configured rule set
    scales = [i for i in items if any(kw in i["text"].lower() for kw in SCALE_KEYWORDS) and len(i["text"]) < 30]
    
    blocks = []
    for scale in scales:
        sx0, sy0, sx1, sy1 = scale["bbox"]
        
        # Find Title
        title_candidates = [i for i in items if i != scale and i["bbox"][3] <= sy0 + 5 and (sy0 - i["bbox"][3]) < 80 and abs(i["bbox"][0] - sx0) < 150]
        title_candidates.sort(key=lambda x: x["bbox"][3], reverse=True)
        title = title_candidates[0] if title_candidates else None
        
        # Find Number
        search_y0 = title["bbox"][1] - 20 if title else sy0 - 50
        search_y1 = sy1 + 20
        number_candidates = [i for i in items if i != scale and i != title and i["bbox"][2] < sx0 + 20 and (sx0 - i["bbox"][2]) < 150 and i["bbox"][3] > search_y0 and i["bbox"][1] < search_y1 and len(i["text"]) <= 5]
        number_candidates.sort(key=lambda x: x["bbox"][2], reverse=True)
        number = number_candidates[0] if number_candidates else None
        
        if number:
            blocks.append({"number": number, "title": title, "scale": scale})
            
    print(f"Successfully identified {len(blocks)} title blocks.")
    return blocks

# ====================================================
# 3. GROUNDING DINO CLUSTERS
# ====================================================
def load_dino_model():
    print(f"Loading Grounding DINO: {MODEL_ID}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    print(f"Model loaded on {device}\n")
    return model, processor, device

def get_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iom = inter_area / float(min(box1_area, box2_area))
    if iom > 0.5:
        return 1.0
        
    return inter_area / float(box1_area + box2_area - inter_area)

def merge_boxes(boxes_with_info, iou_thresh=0.05):
    merged = []
    for current in boxes_with_info:
        matched = False
        for m in merged:
            if get_iou(current["box"], m["box"]) > iou_thresh:
                m["box"][0] = min(m["box"][0], current["box"][0])
                m["box"][1] = min(m["box"][1], current["box"][1])
                m["box"][2] = max(m["box"][2], current["box"][2])
                m["box"][3] = max(m["box"][3], current["box"][3])
                labels = set(m["label"].split(" | "))
                labels.add(current["label"])
                m["label"] = " | ".join(labels)
                m["score"] = max(m["score"], current["score"])
                matched = True
                break
        if not matched:
            merged.append(current)
    return merged

def get_dino_clusters(page, model, processor, device):
    print("Extracting drawing bounds using Grounding DINO...")
    # Lowered DPI to 100 to prevent MemoryErrors (DINO resizes internally anyway)
    scale_factor = 100.0 / 72.0
    mat = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=mat)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    
    results = processor.image_processor.post_process_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        target_sizes=target_sizes
    )[0]

    detected_objects = []
    scores = results["scores"].tolist()
    labels = results["labels"]
    boxes = results["boxes"].tolist()

    img_width, img_height = image.size

    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        if (box_width * box_height) > (0.40 * img_width * img_height):
            continue
            
        detected_objects.append({
            "label": str(label),
            "score": round(float(score), 3),
            "box": [float(x1), float(y1), float(x2), float(y2)]
        })
        
    final_objects = merge_boxes(detected_objects)
    
    merged_clusters = []
    for obj in final_objects:
        px1, py1, px2, py2 = obj["box"]
        # Convert from 300 DPI pixels back to PDF points
        pdf_x1 = px1 / scale_factor
        pdf_y1 = py1 / scale_factor
        pdf_x2 = px2 / scale_factor
        pdf_y2 = py2 / scale_factor
        merged_clusters.append([pdf_x1, pdf_y1, pdf_x2, pdf_y2])
        
    print(f"Found {len(merged_clusters)} major drawing clusters with DINO.")
    return merged_clusters

# ====================================================
# MAIN PIPELINE
# ====================================================
def process_page(page, page_num, doc_len, model, processor, device, results, headers, prompt_text):
    print(f"\n--- Processing Page {page_num}/{doc_len} ---")
    
    # 1. Text & Titles
    items = extract_vector_text(page)
    title_blocks = group_title_blocks(items)
    
    if not title_blocks:
        print("  No title blocks found. Skipping page.")
        return
        
    # 2. Grounding DINO Clusters
    drawing_clusters = get_dino_clusters(page, model, processor, device)
    
    # 3. Match Titles to Drawing Clusters
    print("Matching Titles to Drawing Clusters...")
    final_drawings_pdf_bboxes = []
    
    if len(title_blocks) == 1:
        print("  Exactly 1 title found! Creating a single bounding box for the entire drawing area.")
        all_x0, all_y0, all_x1, all_y1 = [], [], [], []
        
        t = title_blocks[0]
        num_bbox = t["number"]["bbox"]
        tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
        sca_bbox = t["scale"]["bbox"]
        t_all_x0 = [x[0] for x in [num_bbox, tit_bbox, sca_bbox]]
        t_all_y0 = [x[1] for x in [num_bbox, tit_bbox, sca_bbox]]
        t_all_x1 = [x[2] for x in [num_bbox, tit_bbox, sca_bbox]]
        t_all_y1 = [x[3] for x in [num_bbox, tit_bbox, sca_bbox]]
        title_bbox_final = [min(t_all_x0), min(t_all_y0), max(t_all_x1), max(t_all_y1)]
        
        dino_bbox_final = None
        if drawing_clusters:
            d_x0 = min([c[0] for c in drawing_clusters])
            d_y0 = min([c[1] for c in drawing_clusters])
            d_x1 = max([c[2] for c in drawing_clusters])
            d_y1 = max([c[3] for c in drawing_clusters])
            dino_bbox_final = [d_x0, d_y0, d_x1, d_y1]

        for drawing in page.get_drawings():
            r = drawing["rect"]
            if r.width < 1 or r.height < 1: continue
            if (r.x1 - r.x0) > page.rect.width * 0.3 or (r.y1 - r.y0) > page.rect.height * 0.5: continue
            all_x0.append(r.x0); all_y0.append(r.y0); all_x1.append(r.x1); all_y1.append(r.y1)
            
        for i in items:
            if i["bbox"][0] > page.rect.width * 0.85: continue
            all_x0.append(i["bbox"][0]); all_y0.append(i["bbox"][1]); all_x1.append(i["bbox"][2]); all_y1.append(i["bbox"][3])
            
        if all_x0:
            fx0, fy0, fx1, fy1 = min(all_x0) - EXTRA_PADDING, min(all_y0) - EXTRA_PADDING, max(all_x1) + EXTRA_PADDING, max(all_y1) + EXTRA_PADDING
            final_drawings_pdf_bboxes.append({
                "number_text": title_blocks[0]["number"]["text"],
                "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)],
                "title_bbox": title_bbox_final,
                "dino_bbox": dino_bbox_final
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
                
                if ccy > ty0: continue  # Drawing must be above the title
                dx = abs(ccx - tx_c)
                dy = abs(ty0 - cy1)
                
                if dx > page.rect.width * 0.15: continue
                
                score = (dx * 10.0) + dy
                if score < min_score:
                    min_score = score
                    best_cluster = [cx0, cy0, cx1, cy1]
            
            # Combine the cluster bbox and the title bbox into one MASSIVE bbox for the whole drawing
            if best_cluster:
                fx0 = min(best_cluster[0], tx0) - EXTRA_PADDING
                fy0 = min(best_cluster[1], ty0) - EXTRA_PADDING
                fx1 = max(best_cluster[2], tx1) + EXTRA_PADDING
                fy1 = max(best_cluster[3], ty1) + EXTRA_PADDING
            else:
                fx0, fy0, fx1, fy1 = tx0 - EXTRA_PADDING, ty0 - EXTRA_PADDING, tx1 + EXTRA_PADDING, ty1 + EXTRA_PADDING
                
            final_drawings_pdf_bboxes.append({
                "number_text": t["number"]["text"],
                "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)],
                "title_bbox": [tx0, ty0, tx1, ty1],
                "dino_bbox": best_cluster
            })
            
    # Filter by target drawing if specified
    if TARGET_DRAWING:
        final_drawings_pdf_bboxes = [d for d in final_drawings_pdf_bboxes if d["number_text"] == TARGET_DRAWING]
        print(f"Filtered down to {len(final_drawings_pdf_bboxes)} drawing(s) matching number '{TARGET_DRAWING}'")
        
    if not final_drawings_pdf_bboxes:
        return
        
    # 4. Render 300 DPI Image
    print("\nRendering high-res 300 DPI image in memory...")
    scale_factor = 100.0 / 72.0
    mat = fitz.Matrix(scale_factor, scale_factor)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    draw = ImageDraw.Draw(img)

    for i, data in enumerate(final_drawings_pdf_bboxes):
        pdf_bbox = data["pdf_bbox"]
        title_bbox = data.get("title_bbox")
        dino_bbox = data.get("dino_bbox")
        
        def scale_box(b):
            if b is None: return None
            return [int(b[0] * scale_factor), int(b[1] * scale_factor), int(b[2] * scale_factor), int(b[3] * scale_factor)]
            
        px_pdf = scale_box(pdf_bbox)
        px_title = scale_box(title_bbox)
        px_dino = scale_box(dino_bbox)
        
        # Create a copy of the FULL 300 DPI image
        full_img_copy = img.copy()
        draw_copy = ImageDraw.Draw(full_img_copy)
        
        # Draw all 3 boxes for debugging: title (yellow), dino drawing (blue), final merged (green)
        if px_title: draw_copy.rectangle(px_title, outline="yellow", width=8)
        if px_dino: draw_copy.rectangle(px_dino, outline="blue", width=8)
        if px_pdf: draw_copy.rectangle(px_pdf, outline="green", width=15)
        
        # Save locally for verification
        safe_number_text = "".join(c for c in data['number_text'] if c.isalnum() or c in " -_").strip()
        if not safe_number_text: safe_number_text = f"unknown_{i}"
        box_path = os.path.join(VERIFICATION_DIR, f"page_{page_num}_drawing_{safe_number_text}.jpg")
        full_img_copy.save(box_path, format="JPEG", quality=85)
        print(f"  Saved visual bounding box to {box_path}")
        
        # Uncomment below to enable Qwen AI Extraction
        '''
        # Convert FULL SHEET with bounding box to Base64 (Use JPEG to save API payload size)
        buf = io.BytesIO()
        full_img_copy.save(buf, format="JPEG", quality=85)
        b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        print(f"\\nSending FULL SHEET with Box for Drawing {i+1} (Number: {data['number_text']}) to Qwen-VL...")
        
        payload = {
            "model": "qwen/qwen3-vl-32b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}}
                    ]
                }
            ],
            "max_tokens": 512
        }
        
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"]
                if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
                
                result_data = json.loads(content)
                print("  Qwen Output:", json.dumps(result_data, indent=2).replace('\\n', '\\n  '))
                results.append(result_data)
            else:
                print("  API Error:", res.text)
        except Exception as e:
            print("  Failed to process:", e)
        '''
        
        # Draw Verification Box on the main page image
        if px_title: draw.rectangle(px_title, outline="yellow", width=8)
        if px_dino: draw.rectangle(px_dino, outline="blue", width=8)
        if px_pdf: draw.rectangle(px_pdf, outline="green", width=15)
        
    # Save final verification image for the entire page
    final_img_path = os.path.join(VERIFICATION_DIR, f"FINAL_PAGE_{page_num}.png")
    img.save(final_img_path)
    print(f"\n✅ Created Verification Image for Page {page_num}: {final_img_path}")
    
def run_pipeline():
    print(f"Loading {PDF_PATH}...")
    doc = fitz.open(PDF_PATH)
    
    if not os.path.exists(VERIFICATION_DIR): os.makedirs(VERIFICATION_DIR)
    
    # Load DINO once
    model, processor, device = load_dino_model()
    
    results = []
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt_text = """You are analyzing a FULL architectural drawing sheet.

I have drawn a THICK RED BOUNDING BOX around one specific drawing (the Region of Interest).
Please look at the entire sheet to understand the context, but focus your extraction ONLY on the drawing inside the RED BOUNDING BOX.

Extract the following for the drawing inside the red box:
1. drawing_number: The circled/bubbled number at the bottom
2. drawing_title: The title text at the bottom
3. scale: The scale text at the bottom
4. fractions: Any circular callout bubbles located INSIDE the drawing that contain a FRACTION (top value over bottom value separated by a line). 
   CRITICAL: Do NOT include grid column bubbles which only have a single value.\n"""

    if USER_QUESTION:
        prompt_text += f"5. custom_answer: Please answer this specific question based ONLY on this drawing: '{USER_QUESTION}'\n"

    prompt_text += """
Return ONLY a JSON object:
{
  "drawing_number": "1",
  "drawing_title": "ENLARGED PLAN", 
  "scale": "1/4\\" = 1'-0\\"",
  "fractions": [
    {"top": "1", "bottom": "AE512"}
  ]"""
    if USER_QUESTION:
        prompt_text += ",\n  \"custom_answer\": \"Yes, there is a sink on the left.\"\n}"
    else:
        prompt_text += "\n}"
        
    for page_num in range(len(doc)):
        page = doc[page_num]
        process_page(page, page_num + 1, len(doc), model, processor, device, results, headers, prompt_text)
        
    # Uncomment to save AI results to JSON
    '''
    out_json = "final_qwen_extracted_details.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved Qwen results to {out_json}")
    '''
    
    print(f"✅ End-to-End Pipeline Complete! Checked {len(doc)} pages.")

if __name__ == "__main__":
    run_pipeline()
