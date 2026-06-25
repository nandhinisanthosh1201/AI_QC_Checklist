"""
stage3_dino.py
--------------
Runs Grounding DINO on architectural drawings to detect countertops.
Supports processing a single image or an entire directory.

Usage:
  # Process a single image
  py stage3_dino.py --image "output/path/to/page13.png"

  # Process all pages for a PDF
  py stage3_dino.py --pdf_dir "output/(Submittal Mods 1-10)..._pages"
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_ID = "IDEA-Research/grounding-dino-base"
# The prompt format requires a dot (.) at the end of each phrase/class
TEXT_PROMPT = "countertop . sink . cabinet . casework . shelf ."
BOX_THRESHOLD = 0.10
TEXT_THRESHOLD = 0.10

BASE_DIR = Path("C:/1stTask")
RESULTS_DIR = BASE_DIR / "output" / "stage3_dino"
VISUALS_DIR = RESULTS_DIR / "visualizations"

# ── Model ─────────────────────────────────────────────────────────────────────

def load_dino_model():
    print(f"Loading Grounding DINO: {MODEL_ID}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Run in standard float32 to avoid grid_sample type errors on Windows
    dtype = torch.float32

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
        
    print(f"Model loaded on {device}\n")
    return model, processor, device

# ── Box Merging and Padding ───────────────────────────────────────────────────

def get_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Also calculates Intersection over Minimum (IoM) to force merging 
    when a smaller box is completely nested inside a larger one.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0: return 0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # NESTED MERGING LOGIC:
    # If the smaller box is mostly (>50%) inside the larger box, force a merge!
    # This prevents duplicate crops when Grounding DINO detects both the overall 
    # cabinet and the sink inside it as two separate objects.
    iom = inter_area / float(min(box1_area, box2_area))
    if iom > 0.5:
        return 1.0
        
    return inter_area / float(box1_area + box2_area - inter_area)

def merge_boxes(boxes_with_info, iou_thresh=0.05):
    # 1. Merge overlapping boxes (Union)
    merged = []
    for current in boxes_with_info:
        matched = False
        for m in merged:
            if get_iou(current["box"], m["box"]) > iou_thresh:
                m["box"][0] = min(m["box"][0], current["box"][0])
                m["box"][1] = min(m["box"][1], current["box"][1])
                m["box"][2] = max(m["box"][2], current["box"][2])
                m["box"][3] = max(m["box"][3], current["box"][3])
                
                # Combine labels uniquely
                labels = set(m["label"].split(" | "))
                labels.add(current["label"])
                m["label"] = " | ".join(labels)
                m["score"] = max(m["score"], current["score"])
                matched = True
                break
        if not matched:
            merged.append(current)
            
    return merged

# ── Inference ─────────────────────────────────────────────────────────────────

def process_image(model, processor, device, image_path: Path):
    """Run inference on a single image and draw boxes."""
    print(f"Processing: {image_path.name}")
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Cannot open image: {e}")
        return None

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt")
    # Move to GPU
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    
    # post_process_object_detection returns list of dicts:
    # [{'scores': tensor, 'labels': list[str], 'boxes': tensor(x1,y1,x2,y2)}]
    results = processor.image_processor.post_process_object_detection(
        outputs,
        threshold=BOX_THRESHOLD,
        target_sizes=target_sizes
    )[0]

    detected_objects = []
    scores = results["scores"].tolist()
    labels = results["labels"]
    boxes = results["boxes"].tolist()

    if not boxes:
        print("  No countertops found.")
        return []

    print(f"  Found {len(boxes)} candidates.")
    
    draw = ImageDraw.Draw(image)
    
    for score, label, box in zip(scores, labels, boxes):
        x1, y1, x2, y2 = box
        
        # Calculate box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        img_width, img_height = image.size
        
        # FILTER: Ignore boxes that cover more than 40% of the total page area.
        # This prevents massive hallucinated boxes (like the entire page) from being processed.
        if (box_width * box_height) > (0.40 * img_width * img_height):
            print(f"    [Filtered] Ignored massive box ({label}) taking up too much area.")
            continue
            
        detected_objects.append({
            "label": str(label),
            "score": round(float(score), 3),
            "box": [float(x1), float(y1), float(x2), float(y2)]
        })
        
    # Merge overlapping boxes
    final_objects = merge_boxes(detected_objects)
        
    # Draw the final merged & padded boxes
    draw = ImageDraw.Draw(image)
    for obj in final_objects:
        x1, y1, x2, y2 = obj["box"]
        label = obj["label"]
        score = obj["score"]
        
        # Round for final output JSON
        obj["box"] = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
        
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw.text((x1 + 5, y1 + 5), f"{label} {score:.2f}", fill="red")
        
    # Save visualization
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    out_img_path = VISUALS_DIR / f"{image_path.stem}_dino.png"
    image.save(out_img_path)
    print(f"  Saved visualization -> {out_img_path}")
    
    return final_objects

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Grounding DINO Stage 3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image to test")
    group.add_argument("--pdf_dir", type=str, help="Path to a folder of extracted PDF pages")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    model, processor, device = load_dino_model()
    
    all_results = []
    
    if args.image:
        images = [Path(args.image)]
    else:
        images = sorted(Path(args.pdf_dir).glob("*.png"))
        print(f"Found {len(images)} images in {args.pdf_dir}\n")

    for img_path in images:
        if not img_path.exists():
            print(f"[SKIP] Not found: {img_path}")
            continue

        objects = process_image(model, processor, device, img_path)
        if objects is not None:
            all_results.append({
                "filename": img_path.name,
                "objects": objects
            })
        print()

    # Save summary
    summary_path = RESULTS_DIR / "stage3_dino_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
        
    print(f"Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()
