# import fitz
# import os
# import cv2
# import numpy as np
# from collections import deque
# import re
# import argparse

# # ====================================================
# # CONFIG
# # ====================================================

# # PDF_PATH = r"D:\Drawing_Extraction\Quade-task\DGS_(Submittal Mod 1-10)_26034 - DGS SREOC Costa Mesa - MOD 1-10 Rev-Axx - 05-05-2026.pdf"
# PDF_PATH = r"D:\Drawing_Extraction\VERSION2\DGS_Arch.pdf"
# OUTPUT_DIR = r"D:\Drawing_Extraction\VERSION2\verification_output"

# # ====================================================
# # 1. EXTRACT VECTOR TEXT
# # ====================================================
# def extract_vector_text(page):
#     page_dict = page.get_text("dict")
#     items = []
#     for block in page_dict.get("blocks", []):
#         for line in block.get("lines", []):
#             for span in line.get("spans", []):
#                 text = span["text"].strip()
#                 if text:
#                     items.append({"text": text, "bbox": span["bbox"]})
#     return items

# import re

# # ====================================================
# # 2. GROUP INTO TITLE BLOCKS
# # ====================================================
# def extract_scale_candidates(items):
#     scales = []
#     for i in items:
#         text = i["text"].strip().lower()
#         # Allow up to 150 chars because PyMuPDF sometimes merges "SCALE 1:8" and "ARCH REF: 2/AE583" into one block.
#         if len(text) > 150: continue
        
#         is_scale = False
#         if "scale" in text or "n.t.s" in text or re.search(r'\bnts\b', text):
#             is_scale = True
            
#         if is_scale:
#             scales.append(i)
#         elif "=" in text and ("\"" in text or "'" in text):
#             scales.append(i)
#     return scales

# title_keywords = ["plan", "section", "detail", "elevation", "schedule", "diagram", "view"]

# def group_title_blocks(items):
#     scales = extract_scale_candidates(items)

#     blocks = []
#     for scale in scales:
#         sx0, sy0, sx1, sy1 = scale["bbox"]
#         font_h = max(5.0, sy1 - sy0) # Use the scale font size as a dynamic page-relative metric
        
#         # Title is usually just above the scale
#         title_candidates = [i for i in items if i is not scale and i["bbox"][3] <= sy0 + font_h * 2 and (sy0 - i["bbox"][3]) < font_h * 15 and abs(i["bbox"][0] - sx0) < font_h * 30]
#         title_candidates.sort(key=lambda x: x["bbox"][3], reverse=True)
        
#         title = None
#         for tc in title_candidates:
#             tc_text = tc["text"].strip().lower()
#             # Reference notes are not titles!
#             if "see " in tc_text or "refer " in tc_text:
#                 continue
#             if any(kw in tc_text for kw in title_keywords):
#                 title = tc
#                 break
        
#         if not title:
#             continue
            
#         search_y0 = title["bbox"][1] - 30 if title else sy0 - 50
        
#         # Number is usually to the left of the title/scale, and shouldn't be "NA" (Not Applicable)
#         number_candidates = [
#             i for i in items 
#             if i is not scale and i != title 
#             and i["bbox"][2] < sx0 + font_h * 5 
#             and (sx0 - i["bbox"][2]) < font_h * 30 
#             and i["bbox"][3] > search_y0 
#             and i["bbox"][1] < sy1 
#             and len(i["text"].strip()) <= 8 # Support longer tags like "10.1." or "D1A"
#             and i["text"].strip().upper() not in ["NA", "N/A", "SEE", "REF", "TYP", "SIM"]
#         ]
        
#         title_y_center = (title["bbox"][1] + title["bbox"][3]) / 2.0
        
#         # Sort by vertical alignment with title text, then proximity to title, then top-most wins ties.
#         # This prevents stray text (like dimensions) sitting above the title block from stealing the number!
#         number_candidates.sort(key=lambda x: (
#             round(abs((x["bbox"][1] + x["bbox"][3])/2.0 - title_y_center) / 5.0),
#             -round(x["bbox"][2] / 5.0),
#             x["bbox"][1]
#         ))
#         number = number_candidates[0] if number_candidates else None
        
#         if number and title:
#             blocks.append({"number": number, "title": title, "scale": scale, "font_h": font_h})
            
#     # Deduplicate blocks based on number + title text to prevent multiple scales assigning to the same title
#     unique_blocks = []
#     seen = set()
#     for b in blocks:
#         t_text = b["title"]["text"].strip()
#         n_text = b["number"]["text"].strip()
#         key = f"{n_text}_{t_text}"
#         if key not in seen:
#             seen.add(key)
#             unique_blocks.append(b)
            
#     return unique_blocks

# def get_obstacle_clusters(items):
#     obstacles = []
    
#     exact_pattern = re.compile(r'^(materials|general notes|notes|note|legend|finish schedule|hardware)$', re.IGNORECASE)
    
#     for i in items:
#         text = i["text"].strip().lower()
#         if exact_pattern.match(text):
#             obstacles.append(i)
#         elif re.search(r'\bhardware\b', text) and len(text) < 15:
#             if i not in obstacles: obstacles.append(i)
#         elif re.search(r'\bschedule\b', text) and len(text) < 20:
#             if i not in obstacles: obstacles.append(i)

#     blocks = []
#     for obs in obstacles:
#         b = obs["bbox"]
#         blocks.append({
#             "is_obstacle": True,
#             "title_data": {"text": obs["text"]},
#             "bounds": [b[0], b[1], b[2], b[3]],
#             "title_y0": b[1],
#             "title_y1": b[3],
#             "lines": [[b[0], b[1], b[2], b[3]]]
#         })
#     return blocks

# # ====================================================
# # 3. SEMANTIC REGION GROWING (TITLES AS SEEDS)
# # ====================================================
# def semantic_region_growing(titles, obstacles, page, items):
#     # 1. Get all valid vector lines
#     paths = page.get_drawings()
#     boxes = []
    
#     page_w = page.rect.width
#     page_h = page.rect.height
    
#     # Dynamically find global page separators
#     vertical_separators = []
#     horizontal_separators = []
#     for drawing in paths:
#         r = drawing["rect"]
#         px0, py0, px1, py1 = r.x0, r.y0, r.x1, r.y1
#         if (py1 - py0) > page_h * 0.75 and (px1 - px0) < 20:
#             vertical_separators.append(px0)
#         if (px1 - px0) > page_w * 0.75 and (py1 - py0) < 20:
#             horizontal_separators.append(py0)
            
#     max_title_x = 0
#     max_title_y = 0
#     min_title_x = page_w
#     min_title_y = page_h
#     for t in titles:
#         num_bbox = t["number"]["bbox"]
#         tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
#         sca_bbox = t["scale"]["bbox"]
#         tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
#         ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
#         tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
#         ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])
#         if tx0 < min_title_x: min_title_x = tx0
#         if ty0 < min_title_y: min_title_y = ty0
#         if tx1 > max_title_x: max_title_x = tx1
#         if ty1 > max_title_y: max_title_y = ty1
        
#     # Global page borders should only exist in the extreme margins!
#     # This prevents long dimension lines inside the drawing from being mistaken for page borders.
#     valid_vwalls_left = [x for x in vertical_separators if x < min_title_x and x < page_w * 0.05]
#     left_wall = max(valid_vwalls_left) if valid_vwalls_left else 0
        
#     valid_vwalls_right = [x for x in vertical_separators if x > max_title_x and x > page_w * 0.8]
#     right_wall = min(valid_vwalls_right) if valid_vwalls_right else page_w
    
#     valid_hwalls_top = [y for y in horizontal_separators if y < min_title_y and y < page_h * 0.05]
#     top_wall = max(valid_hwalls_top) if valid_hwalls_top else 0
    
#     valid_hwalls_bottom = [y for y in horizontal_separators if y > max_title_y and y > page_h * 0.8]
#     bottom_wall = min(valid_hwalls_bottom) if valid_hwalls_bottom else page_h
    
#     for drawing in paths:
#         r = drawing["rect"]
#         # Filter out tiny 1x1 pixel artifacts/dots, but KEEP pure vertical (width=0) and horizontal (height=0) lines!
#         if r.width < 1 and r.height < 1: continue
#         px0, py0, px1, py1 = r.x0, r.y0, r.x1, r.y1
        
#         # PRODUCTION RULE 1: Crop out global title blocks and page margins using dynamic separators
#         # Anything outside the main drawing boundaries is a border/printer-mark and should be dropped.
#         if px1 < left_wall + 10 or px0 > right_wall - 10 or py1 < top_wall + 10 or py0 > bottom_wall - 10:
#             continue
            
#         # PRODUCTION RULE 2: Filter out full-page rectangular paths
#         # If a border is drawn as a single continuous rectangle path, it bypassed the horizontal/vertical separator logic.
#         # We delete any single path that is massive in BOTH width and height.
#         if (px1 - px0) > page_w * 0.5 and (py1 - py0) > page_h * 0.5:
#             continue
            
#         boxes.append([px0, py0, px1, py1])

#     # PRODUCTION RULE 5: Text as Bridges
#     # Dimension lines are often broken by text (e.g., |--- 98 VIF ---|).
#     # If we only grow regions along vector lines, the text creates a massive gap that halts growth.
#     # By adding text bounding boxes to the graph, they act as perfect bridges!
#     for item in items:
#         b = item["bbox"]
#         if b[2] < left_wall + 10 or b[0] > right_wall - 10 or b[3] < top_wall + 10 or b[1] > bottom_wall - 10:
#             continue
#         boxes.append([b[0], b[1], b[2], b[3]])

#     # 2. Initialize each Title Block as a "Seed Cluster"
#     clusters = []
#     for t in titles:
#         num_bbox = t["number"]["bbox"]
#         tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
#         sca_bbox = t["scale"]["bbox"]
        
#         tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
#         ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
#         tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
#         ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])
        
#         clusters.append({
#             "is_obstacle": False,
#             "title_data": t,
#             "font_h": t["font_h"], # Save font size for dynamic growth metrics
#             "bounds": [tx0, ty0, tx1, ty1],
#             "title_x0": tx0,
#             "title_x1": tx1,
#             "title_y0": ty0, # Used for distance to title
#             "title_y1": ty1, # Used to penalize downward growth
#             "lines": [[tx0, ty0, tx1, ty1]] # Seed with title block bounding box
#         })
        
#     for obs in obstacles:
#         clusters.append(obs)

#     # Wrap geometric boxes into stateful dicts for O(N) BFS tracking
#     all_geometric_boxes = [{"bbox": box, "assigned_cluster": None} for box in boxes]
    
#     # Spatial Hashing Grid for O(N) neighbor lookups
#     grid = {}
#     CELL_SIZE = 200
#     for idx, obj in enumerate(all_geometric_boxes):
#         cx0, cx1 = int(obj["bbox"][0] // CELL_SIZE), int(obj["bbox"][2] // CELL_SIZE)
#         cy0, cy1 = int(obj["bbox"][1] // CELL_SIZE), int(obj["bbox"][3] // CELL_SIZE)
#         for cx in range(cx0, cx1 + 1):
#             for cy in range(cy0, cy1 + 1):
#                 if (cx, cy) not in grid:
#                     grid[(cx, cy)] = []
#                 grid[(cx, cy)].append(idx)
                
#     # Initialize BFS queues for each cluster
#     for c in clusters:
#         c["queue"] = deque()
#         # Seed the queue with the initial title block bounds
#         seed_box = {"bbox": c["bounds"]}
#         c["queue"].append(seed_box)
        
#     active_clusters = True
#     while active_clusters:
#         active_clusters = False
        
#         # Grow each cluster by 1 hop (BFS layer) to ensure fair competition
#         for c in clusters:
#             if not c["queue"]: continue
#             active_clusters = True
            
#             # Process one layer of the BFS queue
#             layer_size = len(c["queue"])
#             for _ in range(layer_size):
#                 curr_box = c["queue"].popleft()
#                 x0, y0, x1, y1 = curr_box["bbox"]
                
#                 # Check neighbors in the spatial grid
#                 cx0, cx1 = int(max(0, x0 - 500) // CELL_SIZE), int((x1 + 500) // CELL_SIZE)
#                 cy0, cy1 = int(max(0, y0 - 500) // CELL_SIZE), int((y1 + 500) // CELL_SIZE)
                
#                 for cx in range(cx0, cx1 + 1):
#                     for cy in range(cy0, cy1 + 1):
#                         if (cx, cy) not in grid: continue
                        
#                         for idx in grid[(cx, cy)]:
#                             neighbor = all_geometric_boxes[idx]
#                             if neighbor["assigned_cluster"] is not None: continue
                            
#                             nx0, ny0, nx1, ny1 = neighbor["bbox"]
                            
#                             # Calculate geometric distance
#                             dx = max(0, x0 - nx1, nx0 - x1)
#                             dy = max(0, y0 - ny1, ny0 - y1)
#                             dist = max(dx, dy)
                            
#                             if c.get("is_obstacle"):
#                                 effective_gap = 100
#                             else:
#                                 font_h = c["font_h"]
#                                 base_gap = font_h * 10
                                
#                                 if ny0 > c["title_y1"] + font_h:
#                                     dist += font_h * 100 
                                    
#                                 is_above = ny1 < c["bounds"][1] + font_h * 5
#                                 horizontal_overlap = max(0, min(nx1 + font_h * 5, c["title_x1"]) - max(nx0 - font_h * 5, c["title_x0"]))
                                
#                                 if is_above and horizontal_overlap > 0:
#                                     effective_gap = font_h * 50
#                                 else:
#                                     effective_gap = base_gap
                                    
#                                 stick_out_left = max(0, c["bounds"][0] - nx0)
#                                 stick_out_right = max(0, nx1 - c["bounds"][2])
#                                 if stick_out_left > font_h * 15 or stick_out_right > font_h * 15:
#                                     dist += font_h * 50
                                    
#                             if dist <= effective_gap:
#                                 neighbor["assigned_cluster"] = c
#                                 c["lines"].append(neighbor["bbox"])
#                                 c["queue"].append(neighbor)
                                
#                                 # Update cluster bounds
#                                 c["bounds"][0] = min(c["bounds"][0], nx0)
#                                 c["bounds"][1] = min(c["bounds"][1], ny0)
#                                 c["bounds"][2] = max(c["bounds"][2], nx1)
#                                 c["bounds"][3] = max(c["bounds"][3], ny1)

#     # Filter out obstacle clusters before returning!
#     valid_clusters = [c for c in clusters if not c.get("is_obstacle")]

#     # 4. Calculate perfect Rectangular Bounding Box for each valid cluster
#     for c in valid_clusters:
#         pts = []
        
#         # Add the Title Block corners
#         t = c["title_data"]
#         num_bbox = t["number"]["bbox"]
#         tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
#         sca_bbox = t["scale"]["bbox"]
#         tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
#         ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
#         tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
#         ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])
        
#         pts.extend([[tx0, ty0], [tx1, ty0], [tx0, ty1], [tx1, ty1]])
        
#         # Add all assigned vector lines
#         for r in c["lines"]:
#             pts.extend([
#                 [r[0], r[1]],
#                 [r[2], r[1]],
#                 [r[0], r[3]],
#                 [r[2], r[3]]
#             ])
            
#         # Get perfect min/max bounding box
#         xs = [p[0] for p in pts]
#         ys = [p[1] for p in pts]
#         min_x, max_x = min(xs), max(xs)
#         min_y, max_y = min(ys), max(ys)
        
#         # Dynamic Padding: 3% of the drawing size, constrained between 5 and 30 points.
#         dw = max_x - min_x
#         dh = max_y - min_y
#         pad_x = min(30, max(5, dw * 0.03))
#         pad_y = min(30, max(5, dh * 0.03))
        
#         # Define the 4 corners of the perfect rectangle
#         c["hull_points"] = [
#             fitz.Point(min_x - pad_x, min_y - pad_y),
#             fitz.Point(max_x + pad_x, min_y - pad_y),
#             fitz.Point(max_x + pad_x, max_y + pad_y),
#             fitz.Point(min_x - pad_x, max_y + pad_y)
#         ]
#         c["label_pos"] = (tx0, ty0)

#     return valid_clusters

# # ====================================================
# # PROCESS FULL PDF
# # ====================================================
# def process_full_pdf(start_page=None, end_page=None, pdf_path=None, output_dir=None):
#     if pdf_path is None:
#         pdf_path = PDF_PATH
#     if output_dir is None:
#         output_dir = OUTPUT_DIR
        
#     print(f"Loading {pdf_path}...")
#     try:
#         doc = fitz.open(pdf_path)
#     except Exception as e:
#         print(f"❌ Critical Error: Could not open PDF {pdf_path}. Exception: {e}")
#         return
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     for page_num in range(len(doc)):
#         current_page = page_num + 1
#         if start_page and current_page < start_page: continue
#         if end_page and current_page > end_page: continue
        
#         try:
#             print(f"--- Processing Page {current_page}/{len(doc)} ---")
#             page = doc[page_num]
            
#             items = extract_vector_text(page)
#             title_blocks = group_title_blocks(items)
#             if not title_blocks:
#                 print("  No titles found on this page. Skipping.")
#                 continue
                
#             obstacle_blocks = get_obstacle_clusters(items)
                
#             print(f"  Found {len(title_blocks)} title blocks. Semantically growing regions...")
            
#             # Region Growing directly outputs the finished semantic clusters!
#             clusters = semantic_region_growing(title_blocks, obstacle_blocks, page, items)
            
#             print("  Drawing Perfect Rectangular Bounding Boxes...")
#             for c in clusters:
#                 if c.get("is_obstacle"):
#                     continue # Do not draw bounding boxes for notes, legends, or margins!
                    
#                 hull_points = c["hull_points"]
                
#                 # Ensure polygon is fully closed without mutating the original list
#                 if hull_points:
#                     closed_points = hull_points + [hull_points[0]]
#                 else:
#                     closed_points = hull_points
                    
#                 shape = page.new_shape()
#                 shape.draw_polyline(closed_points)
#                 shape.finish(color=(1, 0, 0), width=4)
#                 shape.commit()
                
#                 # Draw label
#                 t = c["title_data"]
#                 lx, ly = c["label_pos"]
#                 label_text = f" Drawing: {t['number']['text']} "
#                 label_width = len(label_text) * 10 
#                 label_rect = fitz.Rect(lx, max(0, ly-20), lx + label_width, max(20, ly))
                
#                 page.draw_rect(label_rect, color=(1, 0, 0), fill=(1, 1, 1))
#                 page.insert_text((lx + 4, max(14, ly-6)), label_text, color=(1, 0, 0), fontsize=14)
    
#             # RENDER TO IMAGE
#             print(f"  Rendering Page {current_page} to Image...")
#             mat = fitz.Matrix(2.0, 2.0)
#             pix = page.get_pixmap(matrix=mat)
#             img_path = os.path.join(output_dir, f"page_{current_page}_highlighted.png")
#             pix.save(img_path)
#             print(f"  ✅ Saved {img_path}")
            
#         except Exception as e:
#             print(f"❌ Error processing Page {current_page}: {e}")
#             continue
        
#     print("\n✅ Finished saving Semantic Polygon images!")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Process architectural PDFs and draw bounding boxes.")
#     parser.add_argument("--page", type=int, help="Run a specific page (e.g., --page 3)")
#     parser.add_argument("--range", type=str, help="Run a range of pages (e.g., --range 3-5)")
#     parser.add_argument("--pdf", type=str, help="Path to input PDF file", default=PDF_PATH)
#     parser.add_argument("--out", type=str, help="Path to output directory", default=OUTPUT_DIR)
#     args = parser.parse_args()
    
#     start = None
#     end = None
#     if args.page:
#         start = args.page
#         end = args.page
#     elif args.range:
#         try:
#             parts = args.range.split("-")
#             start = int(parts[0])
#             end = int(parts[1])
#         except Exception:
#             print("❌ Invalid range format. Use --range start-end (e.g., --range 3-5)")
#             exit(1)
            

#     process_full_pdf(start, end, args.pdf, args.out)

import fitz
import os
import re
import logging
import argparse
from collections import deque

# ====================================================
# LOGGING
# ====================================================
def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("drawing_extractor")

log = logging.getLogger("drawing_extractor")  # module-level fallback before setup_logging runs

# ====================================================
# 1. EXTRACT VECTOR TEXT
# ====================================================
def extract_vector_text(page):
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
def extract_scale_candidates(items):
    scales = []
    for i in items:
        text = i["text"].strip().lower()
        # Allow up to 150 chars because PyMuPDF sometimes merges "SCALE 1:8" and "ARCH REF: 2/AE583" into one block.
        if len(text) > 150:
            continue

        is_scale = False
        if "scale" in text or "n.t.s" in text or re.search(r'\bnts\b', text):
            is_scale = True

        if is_scale:
            scales.append(i)
        elif "=" in text and ("\"" in text or "'" in text):
            scales.append(i)
    return scales


title_keywords = ["plan", "section", "detail", "elevation", "schedule", "diagram", "view"]


def group_title_blocks(items, page_num=None):
    scales = extract_scale_candidates(items)
    if not scales:
        log.debug(f"Page {page_num}: no scale candidates found.")

    blocks = []
    no_title_count = 0
    no_number_count = 0

    for scale in scales:
        sx0, sy0, sx1, sy1 = scale["bbox"]
        font_h = max(5.0, sy1 - sy0)  # Use the scale font size as a dynamic page-relative metric

        # Title is usually just above the scale
        title_candidates = [
            i for i in items
            if i is not scale
            and i["bbox"][3] <= sy0 + font_h * 2
            and (sy0 - i["bbox"][3]) < font_h * 15
            and abs(i["bbox"][0] - sx0) < font_h * 30
        ]
        title_candidates.sort(key=lambda x: x["bbox"][3], reverse=True)

        title = None
        for tc in title_candidates:
            tc_text = tc["text"].strip().lower()
            # Reference notes are not titles!
            if "see " in tc_text or "refer " in tc_text:
                continue
            if any(kw in tc_text for kw in title_keywords):
                title = tc
                break

        if not title:
            no_title_count += 1
            continue

        search_y0 = title["bbox"][1] - 30 if title else sy0 - 50

        # Number is usually to the left of the title/scale, and shouldn't be "NA" (Not Applicable)
        number_candidates = [
            i for i in items
            if i is not scale and i != title
            and i["bbox"][2] < sx0 + font_h * 5
            and (sx0 - i["bbox"][2]) < font_h * 30
            and i["bbox"][3] > search_y0
            and i["bbox"][1] < sy1
            and len(i["text"].strip()) <= 8  # Support longer tags like "10.1." or "D1A"
            and i["text"].strip().upper() not in ["NA", "N/A", "SEE", "REF", "TYP", "SIM"]
        ]

        title_y_center = (title["bbox"][1] + title["bbox"][3]) / 2.0

        # Sort by vertical alignment with title text, then proximity to title, then top-most wins ties.
        number_candidates.sort(key=lambda x: (
            round(abs((x["bbox"][1] + x["bbox"][3]) / 2.0 - title_y_center) / 5.0),
            -round(x["bbox"][2] / 5.0),
            x["bbox"][1]
        ))
        number = number_candidates[0] if number_candidates else None

        if not number:
            no_number_count += 1
            continue

        blocks.append({"number": number, "title": title, "scale": scale, "font_h": font_h})

    if no_title_count or no_number_count:
        log.info(
            f"Page {page_num}: {no_title_count} scale(s) had no matching title, "
            f"{no_number_count} had a title but no number."
        )

    # Deduplicate blocks based on number + title text to prevent multiple scales assigning to the same title
    unique_blocks = []
    seen = set()
    for b in blocks:
        t_text = b["title"]["text"].strip()
        n_text = b["number"]["text"].strip()
        key = f"{n_text}_{t_text}"
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks


# ====================================================
# OBSTACLE DETECTION (tightened to word-boundary regex)
# ====================================================
OBSTACLE_PATTERNS = [
    re.compile(r'^\s*materials?\s*$', re.IGNORECASE),
    re.compile(r'^\s*general\s+notes?\s*$', re.IGNORECASE),
    re.compile(r'^\s*notes?\s*:?\s*$', re.IGNORECASE),
    re.compile(r'^\s*legend\s*$', re.IGNORECASE),
    re.compile(r'^\s*finish\s+schedule\s*$', re.IGNORECASE),
    re.compile(r'\bhardware\s+schedule\b', re.IGNORECASE),
    re.compile(r'\bhardware\s+legend\b', re.IGNORECASE),
    re.compile(r'^\s*door\s+hardware\s*$', re.IGNORECASE),
]


def get_obstacle_clusters(items):
    obstacles = []

    for i in items:
        text = i["text"].strip()
        if not text or len(text) > 40:
            continue
        if any(p.search(text) for p in OBSTACLE_PATTERNS):
            obstacles.append(i)

    blocks = []
    for obs in obstacles:
        b = obs["bbox"]
        blocks.append({
            "is_obstacle": True,
            "title_data": {"text": obs["text"]},
            "bounds": [b[0], b[1], b[2], b[3]],
            "title_y0": b[1],
            "title_y1": b[3],
            "lines": [[b[0], b[1], b[2], b[3]]]
        })
    return blocks


# ====================================================
# 3. SEMANTIC REGION GROWING (TITLES AS SEEDS)
# ====================================================
def semantic_region_growing(titles, obstacles, page, items):
    paths = page.get_drawings()
    boxes = []

    page_w = page.rect.width
    page_h = page.rect.height

    vertical_separators = []
    horizontal_separators = []
    for drawing in paths:
        r = drawing["rect"]
        px0, py0, px1, py1 = r.x0, r.y0, r.x1, r.y1
        if (py1 - py0) > page_h * 0.75 and (px1 - px0) < 20:
            vertical_separators.append(px0)
        if (px1 - px0) > page_w * 0.75 and (py1 - py0) < 20:
            horizontal_separators.append(py0)

    max_title_x = 0
    max_title_y = 0
    min_title_x = page_w
    min_title_y = page_h
    for t in titles:
        num_bbox = t["number"]["bbox"]
        tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
        sca_bbox = t["scale"]["bbox"]
        tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
        ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
        tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
        ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])
        if tx0 < min_title_x: min_title_x = tx0
        if ty0 < min_title_y: min_title_y = ty0
        if tx1 > max_title_x: max_title_x = tx1
        if ty1 > max_title_y: max_title_y = ty1

    valid_vwalls_left = [x for x in vertical_separators if x < min_title_x and x < page_w * 0.05]
    left_wall = max(valid_vwalls_left) if valid_vwalls_left else 0

    valid_vwalls_right = [x for x in vertical_separators if x > max_title_x and x > page_w * 0.8]
    right_wall = min(valid_vwalls_right) if valid_vwalls_right else page_w

    valid_hwalls_top = [y for y in horizontal_separators if y < min_title_y and y < page_h * 0.05]
    top_wall = max(valid_hwalls_top) if valid_hwalls_top else 0

    valid_hwalls_bottom = [y for y in horizontal_separators if y > max_title_y and y > page_h * 0.8]
    bottom_wall = min(valid_hwalls_bottom) if valid_hwalls_bottom else page_h

    for drawing in paths:
        r = drawing["rect"]
        if r.width < 1 and r.height < 1:
            continue
        px0, py0, px1, py1 = r.x0, r.y0, r.x1, r.y1

        if px1 < left_wall + 10 or px0 > right_wall - 10 or py1 < top_wall + 10 or py0 > bottom_wall - 10:
            continue

        if (px1 - px0) > page_w * 0.5 and (py1 - py0) > page_h * 0.5:
            continue

        boxes.append([px0, py0, px1, py1])

    for item in items:
        b = item["bbox"]
        if b[2] < left_wall + 10 or b[0] > right_wall - 10 or b[3] < top_wall + 10 or b[1] > bottom_wall - 10:
            continue
        boxes.append([b[0], b[1], b[2], b[3]])

    clusters = []
    for t in titles:
        num_bbox = t["number"]["bbox"]
        tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
        sca_bbox = t["scale"]["bbox"]

        tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
        ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
        tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
        ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])

        clusters.append({
            "is_obstacle": False,
            "title_data": t,
            "font_h": t["font_h"],
            "bounds": [tx0, ty0, tx1, ty1],
            "title_x0": tx0,
            "title_x1": tx1,
            "title_y0": ty0,
            "title_y1": ty1,
            "lines": [[tx0, ty0, tx1, ty1]]
        })

    for obs in obstacles:
        clusters.append(obs)

    all_geometric_boxes = [{"bbox": box, "assigned_cluster": None} for box in boxes]

    grid = {}
    CELL_SIZE = 200
    for idx, obj in enumerate(all_geometric_boxes):
        cx0, cx1 = int(obj["bbox"][0] // CELL_SIZE), int(obj["bbox"][2] // CELL_SIZE)
        cy0, cy1 = int(obj["bbox"][1] // CELL_SIZE), int(obj["bbox"][3] // CELL_SIZE)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                if (cx, cy) not in grid:
                    grid[(cx, cy)] = []
                grid[(cx, cy)].append(idx)

    for c in clusters:
        c["queue"] = deque()
        seed_box = {"bbox": c["bounds"]}
        c["queue"].append(seed_box)

    active_clusters = True
    while active_clusters:
        active_clusters = False

        for c in clusters:
            if not c["queue"]:
                continue
            active_clusters = True

            layer_size = len(c["queue"])
            for _ in range(layer_size):
                curr_box = c["queue"].popleft()
                x0, y0, x1, y1 = curr_box["bbox"]

                cx0, cx1 = int(max(0, x0 - 500) // CELL_SIZE), int((x1 + 500) // CELL_SIZE)
                cy0, cy1 = int(max(0, y0 - 500) // CELL_SIZE), int((y1 + 500) // CELL_SIZE)

                for cx in range(cx0, cx1 + 1):
                    for cy in range(cy0, cy1 + 1):
                        if (cx, cy) not in grid:
                            continue

                        for idx in grid[(cx, cy)]:
                            neighbor = all_geometric_boxes[idx]
                            if neighbor["assigned_cluster"] is not None:
                                continue

                            nx0, ny0, nx1, ny1 = neighbor["bbox"]

                            dx = max(0, x0 - nx1, nx0 - x1)
                            dy = max(0, y0 - ny1, ny0 - y1)
                            dist = max(dx, dy)

                            if c.get("is_obstacle"):
                                effective_gap = 100
                            else:
                                font_h = c["font_h"]
                                base_gap = font_h * 10

                                if ny0 > c["title_y1"] + font_h:
                                    dist += font_h * 100

                                is_above = ny1 < c["bounds"][1] + font_h * 5
                                horizontal_overlap = max(0, min(nx1 + font_h * 5, c["title_x1"]) - max(nx0 - font_h * 5, c["title_x0"]))

                                if is_above and horizontal_overlap > 0:
                                    effective_gap = font_h * 50
                                else:
                                    effective_gap = base_gap

                                stick_out_left = max(0, c["bounds"][0] - nx0)
                                stick_out_right = max(0, nx1 - c["bounds"][2])
                                if stick_out_left > font_h * 15 or stick_out_right > font_h * 15:
                                    dist += font_h * 50

                            if dist <= effective_gap:
                                neighbor["assigned_cluster"] = c
                                c["lines"].append(neighbor["bbox"])
                                c["queue"].append(neighbor)

                                c["bounds"][0] = min(c["bounds"][0], nx0)
                                c["bounds"][1] = min(c["bounds"][1], ny0)
                                c["bounds"][2] = max(c["bounds"][2], nx1)
                                c["bounds"][3] = max(c["bounds"][3], ny1)

    valid_clusters = [c for c in clusters if not c.get("is_obstacle")]

    for c in valid_clusters:
        pts = []

        t = c["title_data"]
        num_bbox = t["number"]["bbox"]
        tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
        sca_bbox = t["scale"]["bbox"]
        tx0 = min(num_bbox[0], tit_bbox[0], sca_bbox[0])
        ty0 = min(num_bbox[1], tit_bbox[1], sca_bbox[1])
        tx1 = max(num_bbox[2], tit_bbox[2], sca_bbox[2])
        ty1 = max(num_bbox[3], tit_bbox[3], sca_bbox[3])

        pts.extend([[tx0, ty0], [tx1, ty0], [tx0, ty1], [tx1, ty1]])

        for r in c["lines"]:
            pts.extend([
                [r[0], r[1]],
                [r[2], r[1]],
                [r[0], r[3]],
                [r[2], r[3]]
            ])

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        dw = max_x - min_x
        dh = max_y - min_y
        pad_x = min(30, max(5, dw * 0.03))
        pad_y = min(30, max(5, dh * 0.03))

        c["hull_points"] = [
            fitz.Point(min_x - pad_x, min_y - pad_y),
            fitz.Point(max_x + pad_x, min_y - pad_y),
            fitz.Point(max_x + pad_x, max_y + pad_y),
            fitz.Point(min_x - pad_x, max_y + pad_y)
        ]
        c["label_pos"] = (tx0, ty0)

    return valid_clusters


# ====================================================
# PROCESS FULL PDF
# ====================================================
def process_full_pdf(pdf_path, output_dir, start_page=None, end_page=None):
    if not os.path.isfile(pdf_path):
        log.error(f"PDF not found: {pdf_path}")
        return

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.error(f"Failed to open PDF '{pdf_path}': {e}")
        return

    if doc.is_encrypted:
        log.error(f"PDF is encrypted and could not be processed: {pdf_path}")
        doc.close()
        return

    os.makedirs(output_dir, exist_ok=True)

    pages_processed = 0
    pages_skipped = 0
    pages_failed = 0

    for page_num in range(len(doc)):
        current_page = page_num + 1
        if start_page and current_page < start_page:
            continue
        if end_page and current_page > end_page:
            continue

        log.info(f"--- Processing Page {current_page}/{len(doc)} ---")

        try:
            page = doc[page_num]

            items = extract_vector_text(page)
            title_blocks = group_title_blocks(items, page_num=current_page)
            if not title_blocks:
                log.info(f"  No titles found on page {current_page}. Skipping.")
                pages_skipped += 1
                continue

            obstacle_blocks = get_obstacle_clusters(items)

            log.info(f"  Found {len(title_blocks)} title blocks. Semantically growing regions...")
            clusters = semantic_region_growing(title_blocks, obstacle_blocks, page, items)

            log.info("  Drawing bounding boxes...")
            for c in clusters:
                if c.get("is_obstacle"):
                    continue

                hull_points = list(c["hull_points"])
                if hull_points:
                    hull_points.append(hull_points[0])

                shape = page.new_shape()
                shape.draw_polyline(hull_points)
                shape.finish(color=(1, 0, 0), width=4)
                shape.commit()

                t = c["title_data"]
                lx, ly = c["label_pos"]
                label_text = f" Drawing: {t['number']['text']} "
                label_width = len(label_text) * 10
                label_rect = fitz.Rect(lx, max(0, ly - 20), lx + label_width, max(20, ly))

                page.draw_rect(label_rect, color=(1, 0, 0), fill=(1, 1, 1))
                page.insert_text((lx + 4, max(14, ly - 6)), label_text, color=(1, 0, 0), fontsize=14)

            log.info(f"  Rendering page {current_page} to image...")
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_path = os.path.join(output_dir, f"page_{current_page}_highlighted.png")
            pix.save(img_path)
            log.info(f"  Saved {img_path}")
            pages_processed += 1

        except Exception as e:
            log.error(f"  Failed to process page {current_page}: {e}", exc_info=True)
            pages_failed += 1
            continue

    doc.close()
    log.info(
        f"Finished. Processed: {pages_processed}, Skipped (no titles): {pages_skipped}, "
        f"Failed: {pages_failed}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process architectural PDFs and draw bounding boxes.")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the input PDF file.")
    parser.add_argument("--output", type=str, required=True, help="Directory to save output images and logs.")
    parser.add_argument("--page", type=int, help="Run a specific page (e.g., --page 3)")
    parser.add_argument("--range", type=str, help="Run a range of pages (e.g., --range 3-5)")
    args = parser.parse_args()

    log = setup_logging(args.output)

    start = None
    end = None
    if args.page:
        start = args.page
        end = args.page
    elif args.range:
        try:
            parts = args.range.split("-")
            start = int(parts[0])
            end = int(parts[1])
        except Exception:
            log.error("Invalid range format. Use --range start-end (e.g., --range 3-5)")
            exit(1)

    process_full_pdf(args.pdf, args.output, start, end)