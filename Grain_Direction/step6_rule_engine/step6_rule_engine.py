import json
import pandas as pd
import cv2
import numpy as np
import os
import glob
from pathlib import Path
from collections import defaultdict
import fitz


def find_project_pdf(proj_name):
    """Locate the original input PDF for the given project name."""
    patterns = [
        f"inputs/**/*{proj_name}*.pdf",
        f"inputs/**/{proj_name}*.pdf",
        f"inputs/*.pdf",
        f"**/*{proj_name}*.pdf"
    ]
    for pat in patterns:
        matches = glob.glob(pat, recursive=True)
        for m in matches:
            if proj_name.lower().split()[0] in os.path.basename(m).lower():
                return m
    return None


def trace_leader_arrow_tip(page, tag_rect_pdf, max_reach=250):
    """
    Traces vector leader lines in PDF starting near a tag to find where the arrowhead / tip points.
    Filters out the closed tag bubble and follows outward line chains to the arrowhead.
    Returns (target_x, target_y, vec_dx, vec_dy) in PDF points, or None if no leader line found.
    """
    tx = (tag_rect_pdf.x0 + tag_rect_pdf.x1) / 2.0
    ty = (tag_rect_pdf.y0 + tag_rect_pdf.y1) / 2.0

    try:
        drawings = page.get_drawings()
    except Exception:
        return None

    all_lines = []
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == 'l':
                all_lines.append((item[1], item[2]))

    # Step 1: Find lines starting at tag bubble (< 25 pt) and extending outward away from tag center
    outward_starts = []
    for p1, p2 in all_lines:
        d1 = ((p1.x - tx)**2 + (p1.y - ty)**2)**0.5
        d2 = ((p2.x - tx)**2 + (p2.y - ty)**2)**0.5
        if d1 < 25.0 and d2 > d1 + 4.0:
            outward_starts.append((p1, p2, d2))
        elif d2 < 25.0 and d1 > d2 + 4.0:
            outward_starts.append((p2, p1, d1))

    if not outward_starts:
        return None

    # Step 2: Follow line chain outward to the farthest endpoint (arrowhead tip)
    outward_starts.sort(key=lambda x: x[2], reverse=True)
    best_target = None
    best_vec_dx = 0.0
    best_vec_dy = 0.0
    max_travel = 0

    for start_pt, end_pt, _ in outward_starts[:3]:
        curr_tip = end_pt
        visited = {(round(start_pt.x, 1), round(start_pt.y, 1)), (round(end_pt.x, 1), round(end_pt.y, 1))}
        for hop in range(6):
            extended = False
            for p1, p2 in all_lines:
                dist1 = ((p1.x - curr_tip.x)**2 + (p1.y - curr_tip.y)**2)**0.5
                dist2 = ((p2.x - curr_tip.x)**2 + (p2.y - curr_tip.y)**2)**0.5
                if ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5 > 2.0:
                    if dist1 < 3.0:
                        key = (round(p2.x, 1), round(p2.y, 1))
                        if key not in visited:
                            visited.add(key)
                            curr_tip = p2
                            extended = True
                            break
                    elif dist2 < 3.0:
                        key = (round(p1.x, 1), round(p1.y, 1))
                        if key not in visited:
                            visited.add(key)
                            curr_tip = p1
                            extended = True
                            break
            if not extended:
                break

        total_dist = ((curr_tip.x - tx)**2 + (curr_tip.y - ty)**2)**0.5
        if total_dist >= 18.0 and total_dist > max_travel and total_dist <= max_reach:
            max_travel = total_dist
            best_target = curr_tip
            best_vec_dx = curr_tip.x - start_pt.x
            best_vec_dy = curr_tip.y - start_pt.y

    return (best_target.x, best_target.y, best_vec_dx, best_vec_dy) if best_target else None


def arrow_points_into_component(tgt_x, tgt_y, comp, vec_dx=0.0, vec_dy=0.0):
    """
    Returns True if the leader arrow tip (tgt_x, tgt_y) points INTO the component face.
    If the arrow tip lands near a boundary edge, it only matches if the arrow direction
    vector is pointing TOWARDS the interior of the component, NOT away from it
    (e.g. an arrow pointing downwards onto the bottom edge of a wall panel is pointing
     at the countertop below it, NOT at the wall panel).
    """
    cx0, cy0, cx1, cy1 = comp["x0"], comp["y0"], comp["x1"], comp["y1"]
    w = cx1 - cx0
    h = cy1 - cy0

    margin_x = min(20.0, w * 0.10)
    margin_y = min(20.0, h * 0.10)

    # 1. Deep interior match (well inside the component face)
    if (cx0 + margin_x) <= tgt_x <= (cx1 - margin_x) and (cy0 + margin_y) <= tgt_y <= (cy1 - margin_y):
        return True

    # 2. Boundary proximity check: must not be pointing OUT of the component
    if (cx0 - 3) <= tgt_x <= (cx1 + 3) and (cy0 - 3) <= tgt_y <= (cy1 + 3):
        # If moving downwards (vec_dy > 5) and tip is near the bottom edge (cy1 - 25)
        # it is pointing DOWNWARDS out of the component (e.g. onto countertop / floor)
        if vec_dy > 5.0 and tgt_y >= (cy1 - 25):
            return False
        # If moving upwards (vec_dy < -5) and tip is near the top edge (cy0 + 25)
        if vec_dy < -5.0 and tgt_y <= (cy0 + 25):
            return False
        # If moving right (vec_dx > 5) and tip is near right edge (cx1 - 25)
        if vec_dx > 5.0 and tgt_x >= (cx1 - 25):
            return False
        # If moving left (vec_dx < -5) and tip is near left edge (cx0 + 25)
        if vec_dx < -5.0 and tgt_x <= (cx0 + 25):
            return False

        return True

    return False


def load_material_rules(step1_dir):
    """
    Load dynamic material rules from Step 1 JSON files.
    Returns dict: tag -> {grain_required: bool, expected_direction: str, name: str, mfg: str}
    """
    material_rules = {}
    if not os.path.isdir(step1_dir):
        print(f"[WARN] Step 1 dir not found: {step1_dir}")
        return material_rules

    for filename in os.listdir(step1_dir):
        if filename.endswith(".json") and filename != "materials_cache.json":
            json_path = os.path.join(step1_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                tag = m_data.get("tag", "").upper().strip()
                if tag:
                    grain_val = str(m_data.get("grain", "")).upper().strip()
                    grain_required = grain_val in ["YES", "Y", "TRUE"]
                    direction = str(m_data.get("direction", "VERTICAL")).upper().strip()
                    if not direction or direction == "NONE":
                        direction = "VERTICAL" if grain_required else "NONE"

                    material_rules[tag] = {
                        "grain_required": grain_required,
                        "expected_direction": direction,
                        "name": m_data.get("name", ""),
                        "manufacturer": m_data.get("manufacturer", "")
                    }
            except Exception as e:
                print(f"[WARN] Failed to read {json_path}: {e}")

    return material_rules


def normalize_component_class(raw_cls):
    """Normalize raw detected class into exactly 4 architectural components: Cabinet, Toe kick, Panel, Soffit."""
    raw_lower = str(raw_cls).strip().lower()
    if "toe" in raw_lower or "kick" in raw_lower:
        return "Toe kick"
    elif "panel" in raw_lower:
        return "Panel"
    elif "soffit" in raw_lower:
        return "Soffit"
    else:
        return "Cabinet"


def is_table_or_titleblock_fp(comp, manifest_row, w_img, h_img, zoom=2.5):
    """
    Detect and reject false positive boxes detected on sheet title blocks,
    revision tables, hardware schedules, and grid tables outside the drawing field.
    """
    cx0, cy0, cx1, cy1 = comp["x0"], comp["y0"], comp["x1"], comp["y1"]
    cw = cx1 - cx0
    ch = cy1 - cy0

    if manifest_row is not None:
        try:
            crop_x0 = float(manifest_row.get("x0", 0))
            crop_x1 = float(manifest_row.get("x1", 0))
            comp_pdf_x0 = crop_x0 + (cx0 / zoom)
            comp_pdf_x1 = crop_x0 + (cx1 / zoom)

            # Standard CAD drawing sheets: x > 2180 is the right-hand title block & revision table strip
            if comp_pdf_x0 > 2180 or comp_pdf_x1 > 2450:
                return True

            # If crop extends to sheet boundary and component is in the far right 15% strip
            if crop_x1 > 2300 and cx0 > (w_img * 0.82):
                return True
        except Exception:
            pass

    # Aspect ratio check for thin table row slices sitting on the right edge
    if ch < 120 and (cw / max(1.0, ch)) > 3.5 and cx0 > (w_img * 0.75):
        return True

    return False


def filter_duplicate_detections(comps):
    """
    Non-Maximum Suppression (NMS) and container-box suppression to eliminate
    duplicate or overlapping boxes (e.g. YOLO detecting both Panel and Cabinet
    on the exact same door, or an overarching box covering a bank of 3 cabinets).
    """
    if not comps or len(comps) <= 1:
        return comps

    # 1. Container box filtering: filter out any casework box that encloses >= 2 smaller casework boxes
    filtered = []
    for i, c in enumerate(comps):
        if "toe" in c.get("class", "").lower() or "kick" in c.get("class", "").lower():
            filtered.append(c)
            continue
        cx0, cy0, cx1, cy1 = c["x0"], c["y0"], c["x1"], c["y1"]
        c_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
        contained_count = 0
        for j, o in enumerate(comps):
            if i == j or "toe" in o.get("class", "").lower() or "kick" in o.get("class", "").lower():
                continue
            ox0, oy0, ox1, oy1 = o["x0"], o["y0"], o["x1"], o["y1"]
            if ox0 >= cx0 - 15 and ox1 <= cx1 + 15 and oy0 >= cy0 - 15 and oy1 <= cy1 + 15:
                o_area = max(1.0, (ox1 - ox0) * (oy1 - oy0))
                if o_area < 0.70 * c_area:
                    contained_count += 1
        if contained_count >= 2:
            continue
        filtered.append(c)

    sorted_comps = sorted(filtered, key=lambda c: float(c.get("confidence", 0.5)), reverse=True)
    kept = []

    for c in sorted_comps:
        cx0, cy0, cx1, cy1 = c["x0"], c["y0"], c["x1"], c["y1"]
        c_area = max(1.0, (cx1 - cx0) * (cy1 - cy0))
        is_dup = False

        for k in kept:
            kx0, ky0, kx1, ky1 = k["x0"], k["y0"], k["x1"], k["y1"]
            k_area = max(1.0, (kx1 - kx0) * (ky1 - ky0))

            ix0 = max(cx0, kx0)
            iy0 = max(cy0, ky0)
            ix1 = min(cx1, kx1)
            iy1 = min(cy1, ky1)

            if ix1 > ix0 and iy1 > iy0:
                i_area = (ix1 - ix0) * (iy1 - iy0)
                iou = i_area / (c_area + k_area - i_area)
                min_ratio = i_area / min(c_area, k_area)
                if iou > 0.60 or min_ratio > 0.80:
                    is_dup = True
                    break

        if not is_dup:
            kept.append(c)

    return kept


def cluster_components_into_runs(comps):
    """
    Group casework into architectural banks/runs of adjacent, similar components
    (Cabinet banks, Panel runs, Soffit runs).
    Toe kicks are separated out.
    """
    toe_kicks = []
    casework = []

    for c in comps:
        cls_lower = c["class"].lower()
        if "toe" in cls_lower or "kick" in cls_lower:
            toe_kicks.append(c)
        else:
            casework.append(c)

    if not casework:
        return [], toe_kicks

    n = len(casework)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        c1 = casework[i]
        w1, h1 = c1["x1"] - c1["x0"], c1["y1"] - c1["y0"]
        cls1 = normalize_component_class(c1["class"])

        for j in range(i + 1, n):
            c2 = casework[j]
            w2, h2 = c2["x1"] - c2["x0"], c2["y1"] - c2["y0"]
            cls2 = normalize_component_class(c2["class"])

            # Must be compatible component class (Cabinet with Cabinet, Panel with Panel)
            if cls1 != cls2:
                continue

            # A. Horizontal adjacency (adjacent doors / panels in a row)
            v_overlap = max(0.0, min(c1["y1"], c2["y1"]) - max(c1["y0"], c2["y0"]))
            min_h, max_h = min(h1, h2), max(h1, h2)
            h_gap = max(0.0, max(c1["x0"], c2["x0"]) - min(c1["x1"], c2["x1"]))
            is_horiz_adj = (v_overlap / min_h >= 0.50) and (h_gap <= 50) and ((max_h - min_h) / max_h <= 0.40)

            # B. Vertical adjacency (stacked drawers / doors in same column/bank)
            h_overlap = max(0.0, min(c1["x1"], c2["x1"]) - max(c1["x0"], c2["x0"]))
            min_w, max_w = min(w1, w2), max(w1, w2)
            v_gap = max(0.0, max(c1["y0"], c2["y0"]) - min(c1["y1"], c2["y1"]))
            is_vert_adj = (h_overlap / min_w >= 0.50) and (v_gap <= 40) and ((max_w - min_w) / max_w <= 0.35)

            if is_horiz_adj or is_vert_adj:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(casework[i])

    runs = []
    for cluster_id, cluster_comps in clusters.items():
        min_x0 = min(c["x0"] for c in cluster_comps)
        max_x1 = max(c["x1"] for c in cluster_comps)
        min_y0 = min(c["y0"] for c in cluster_comps)
        max_y1 = max(c["y1"] for c in cluster_comps)
        avg_cy = sum((c["y0"] + c["y1"]) / 2.0 for c in cluster_comps) / len(cluster_comps)
        avg_h = sum(c["y1"] - c["y0"] for c in cluster_comps) / len(cluster_comps)

        runs.append({
            "components": cluster_comps,
            "x0": min_x0,
            "y0": min_y0,
            "x1": max_x1,
            "y1": max_y1,
            "avg_cy": avg_cy,
            "avg_h": avg_h,
            "type": "Casework Bank"
        })

    runs.sort(key=lambda r: (r["avg_cy"], r["x0"]))
    return runs, toe_kicks


def run_step6(outputs_dir, project_name):
    """
    Step 6: Hierarchical Rule Engine with Run-Level Grouping & Tag/Grain Inheritance.
    """
    proj = os.path.join(outputs_dir, project_name)

    step1_dir  = os.path.join(proj, "Step1_MaterialScraping")
    step2_csv  = os.path.join(proj, "Step2_GrainSymbols", "candidates.csv")
    step3_dir  = os.path.join(proj, "Step3_ElevationCrops")
    step4_json = os.path.join(proj, "Step4_ComponentDetection", "components_data.json")
    step5_dir  = os.path.join(proj, "Step5_MaterialTags")
    step5_json = os.path.join(step5_dir, "tags_data.json")

    print(f"\n{'='*70}")
    print(f"   STEP 6: RULE ENGINE VALIDATION — {project_name}")
    print(f"{'='*70}")

    # 1. Load Step 1 Material Rules
    material_rules = load_material_rules(step1_dir)
    print(f"[INFO] Loaded {len(material_rules)} dynamic material rules from Step 1:")
    for t, info in material_rules.items():
        req_str = f"Grain: {info['expected_direction']}" if info['grain_required'] else "No grain"
        print(f"       - {t:6s} : {req_str} ({info['name']})")

    # 2. Load Step 2 Grain Candidates CSV
    grains_df = pd.DataFrame()
    if os.path.exists(step2_csv):
        try:
            grains_df = pd.read_csv(step2_csv)
            print(f"[INFO] Loaded {len(grains_df)} grain symbol candidates from Step 2")
        except Exception as e:
            print(f"[WARN] Failed to read candidates.csv: {e}")
    else:
        print(f"[INFO] No candidates.csv found at {step2_csv}")

    # 3. Locate Step 3 Manifest CSV
    manifest_path = None
    for root, dirs, files in os.walk(step3_dir):
        if "manifest.csv" in files:
            manifest_path = os.path.join(root, "manifest.csv")
            break

    if manifest_path is None or not os.path.exists(manifest_path):
        print(f"[ERROR] manifest.csv not found under: {step3_dir}")
        print("        Please ensure Step 3 elevation cropping has been executed.")
        return

    print(f"[INFO] Found manifest at: {manifest_path}")
    manifest_df = pd.read_csv(manifest_path)

    # 4. Load Step 4 Components JSON
    if not os.path.exists(step4_json):
        print(f"[ERROR] Step 4 components_data.json not found at: {step4_json}")
        return

    with open(step4_json, "r", encoding="utf-8") as f:
        components = json.load(f)
    print(f"[INFO] Loaded {len(components)} detected components from Step 4")

    # 5. Load Step 5 Material Tags JSON
    if not os.path.exists(step5_json):
        print(f"[ERROR] Step 5 tags_data.json not found at: {step5_json}")
        return

    with open(step5_json, "r", encoding="utf-8") as f:
        tags = json.load(f)
    print(f"[INFO] Loaded {len(tags)} detected material tags from Step 5")

    # Index manifest by crop_file
    manifest_by_crop = {}
    for _, row in manifest_df.iterrows():
        cfile = str(row["crop_file"]).strip()
        if cfile:
            manifest_by_crop[cfile] = row

    # Group components and tags by normalized crop file name
    comps_by_crop = defaultdict(list)
    for c in components:
        clean_name = c["file"].replace("detected_", "")
        comps_by_crop[clean_name].append(c)

    tags_by_crop = defaultdict(list)
    for t in tags:
        clean_name = t["file"].replace("detected_", "")
        tags_by_crop[clean_name].append(t)

    ZOOM = 2.5
    validation_results = []
    os.makedirs(os.path.join(proj, "Step6_FinalValidation"), exist_ok=True)
    step6_out_dir = os.path.join(proj, "Step6_FinalValidation")

    print(f"[INFO] Processing {len(comps_by_crop)} elevation views with hierarchical grouping...")

    for crop_file, img_comps in comps_by_crop.items():
        step5_filename = f"detected_{crop_file}"
        step5_img_path = os.path.join(step5_dir, step5_filename)

        if not os.path.exists(step5_img_path):
            continue

        img = cv2.imread(step5_img_path)
        if img is None:
            continue

        h_img, w_img = img.shape[:2]

        # ------------------------------------------------------------- #
        # Step A: Identify Grain Direction Arrows for this Crop
        # ------------------------------------------------------------- #
        # 1. From HSV red circle detection in the image
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask  = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv_circles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            if 0.75 <= aspect_ratio <= 1.35 and 15 < w < 250:
                hsv_circles.append({
                    "x": x + w / 2.0,
                    "y": y + h / 2.0,
                    "w": w,
                    "h": h,
                    "dir": "UNKNOWN"
                })

        # 2. From Step 2 candidates.csv with exact line segment orientations
        step2_arrows = []
        manifest_row = manifest_by_crop.get(crop_file)
        if manifest_row is not None and not grains_df.empty:
            pno = int(manifest_row["page"])
            crop_x0 = float(manifest_row["x0"])
            crop_y0 = float(manifest_row["y0"])
            crop_x1 = float(manifest_row["x1"])
            crop_y1 = float(manifest_row["y1"])

            matching_cands = grains_df[
                (grains_df["page"] == pno) &
                (grains_df["x"] >= crop_x0 - 15) & (grains_df["x"] <= crop_x1 + 15) &
                (grains_df["y"] >= crop_y0 - 15) & (grains_df["y"] <= crop_y1 + 15)
            ]

            for _, cand in matching_cands.iterrows():
                # Convert PDF point to crop image pixel
                gx_img = (float(cand["x"]) - crop_x0) * ZOOM
                gy_img = (float(cand["y"]) - crop_y0) * ZOOM
                dx = abs(float(cand["x1"]) - float(cand["x0"]))
                dy = abs(float(cand["y1"]) - float(cand["y0"]))
                shaft_dir = "VERTICAL" if dy >= dx else "HORIZONTAL"
                step2_arrows.append({
                    "x": gx_img,
                    "y": gy_img,
                    "dir": shaft_dir
                })

        # Combine arrows: attach orientation to circles
        grain_arrows = []
        for arr in step2_arrows:
            grain_arrows.append(arr)

        for circ in hsv_circles:
            # Check if already matched to a step2 arrow
            matched = False
            for arr in step2_arrows:
                dist = np.hypot(circ["x"] - arr["x"], circ["y"] - arr["y"])
                if dist < 60.0:
                    matched = True
                    break
            if not matched:
                # Infer orientation from circle's inner ink or default to Vertical
                grain_arrows.append({
                    "x": circ["x"],
                    "y": circ["y"],
                    "dir": "VERTICAL"
                })

        # ------------------------------------------------------------- #
        # Step B: Filter False Positives & Deduplicate Components
        # ------------------------------------------------------------- #
        manifest_row = manifest_by_crop.get(crop_file)
        cleaned_comps = [c for c in img_comps if not is_table_or_titleblock_fp(c, manifest_row, w_img, h_img, ZOOM)]
        cleaned_comps = filter_duplicate_detections(cleaned_comps)

        runs, toe_kicks = cluster_components_into_runs(cleaned_comps)
        crop_tags = tags_by_crop.get(crop_file, [])

        # ------------------------------------------------------------- #
        # Step C: Trace Tag Leader Arrows & Match to Components/Banks
        # ------------------------------------------------------------- #
        # Open PDF page if available to trace vector leader arrows
        pdf_path = find_project_pdf(project_name)
        pdf_page = None
        if pdf_path and manifest_row is not None:
            try:
                pdf_doc = fitz.open(pdf_path)
                pno = int(manifest_row["page"])
                if pno < len(pdf_doc):
                    pdf_page = pdf_doc[pno]
            except Exception:
                pdf_page = None

        crop_x0 = float(manifest_row["x0"]) if manifest_row is not None else 0.0
        crop_y0 = float(manifest_row["y0"]) if manifest_row is not None else 0.0

        # Determine target point and direction for each tag (leader arrow tip vs tag box)
        for t in crop_tags:
            t_midx = (t["x0"] + t["x1"]) / 2.0
            t_midy = (t["y0"] + t["y1"]) / 2.0
            t["target_x"] = t_midx
            t["target_y"] = t_midy
            t["vec_dx"] = 0.0
            t["vec_dy"] = 0.0
            t["has_leader"] = False

            if pdf_page is not None:
                tx_pdf = crop_x0 + t["x0"] / ZOOM
                ty_pdf = crop_y0 + t["y0"] / ZOOM
                target_pdf = trace_leader_arrow_tip(pdf_page, fitz.Rect(tx_pdf - 12, ty_pdf - 10, tx_pdf + 12, ty_pdf + 10))
                if target_pdf is not None:
                    t["target_x"] = (target_pdf[0] - crop_x0) * ZOOM
                    t["target_y"] = (target_pdf[1] - crop_y0) * ZOOM
                    t["vec_dx"] = target_pdf[2] * ZOOM
                    t["vec_dy"] = target_pdf[3] * ZOOM
                    t["has_leader"] = True

        all_comps_with_context = []

        # 1. Casework components within Banks (Cabinet, Panel, Soffit)
        for run in runs:
            b_x0, b_y0, b_x1, b_y1 = run["x0"], run["y0"], run["x1"], run["y1"]

            # First, check direct tag matching for each component in this bank:
            # A tag only matches a component if the leader arrow tip lands strictly INSIDE the component face,
            # or if a standalone label sits entirely inside the component face.
            bank_matched_tags = []
            comp_direct_tags = {}

            for idx, comp in enumerate(run["components"]):
                direct_tag = None
                for t in crop_tags:
                    if t.get("has_leader", False):
                        if arrow_points_into_component(t["target_x"], t["target_y"], comp, t.get("vec_dx", 0.0), t.get("vec_dy", 0.0)):
                            direct_tag = t["tag"]
                            break
                    else:
                        t_x0, t_y0, t_x1, t_y1 = t["x0"], t["y0"], t["x1"], t["y1"]
                        if comp["x0"] <= t_x0 and t_x1 <= comp["x1"] and comp["y0"] <= t_y0 and t_y1 <= comp["y1"]:
                            direct_tag = t["tag"]
                            break

                if direct_tag:
                    comp_direct_tags[idx] = direct_tag
                    if direct_tag not in bank_matched_tags:
                        bank_matched_tags.append(direct_tag)

            # Bank only inherits a common tag if at least one component in this bank was explicitly tagged
            common_bank_tag = bank_matched_tags[0] if bank_matched_tags else None

            # Find all grain arrows within this bank
            bank_grains = []
            for g in grain_arrows:
                if (b_x0 - 25) <= g["x"] <= (b_x1 + 25) and (b_y0 - 25) <= g["y"] <= (b_y1 + 25):
                    bank_grains.append(g)
            common_bank_grain = bank_grains[0]["dir"] if bank_grains else None

            for idx, comp in enumerate(run["components"]):
                c_cls = normalize_component_class(comp.get("class", "Cabinet"))
                cx0, cy0, cx1, cy1 = comp["x0"], comp["y0"], comp["x1"], comp["y1"]

                direct_tag = comp_direct_tags.get(idx)

                # Direct grain arrow check on this specific component
                direct_grain = None
                for g in grain_arrows:
                    if (cx0 - 20) <= g["x"] <= (cx1 + 20) and (cy0 - 20) <= g["y"] <= (cy1 + 20):
                        direct_grain = g["dir"]
                        break

                eff_tag = direct_tag or common_bank_tag
                eff_tag_source = "Direct Leader / Tag" if direct_tag else ("Bank Common Tag" if common_bank_tag else "None")

                eff_grain = direct_grain or common_bank_grain
                eff_grain_source = "Direct Arrow" if direct_grain else ("Bank Common Arrow" if common_bank_grain else "None")
                eff_grain_present = eff_grain is not None

                all_comps_with_context.append({
                    "comp": comp,
                    "component_class": c_cls,
                    "run_type": run["type"],
                    "tag": eff_tag,
                    "tag_source": eff_tag_source,
                    "grain_dir": eff_grain,
                    "grain_source": eff_grain_source,
                    "grain_present": eff_grain_present,
                    "is_toe_kick": False
                })

        # 2. Toe kick components
        for comp in toe_kicks:
            cx0, cy0, cx1, cy1 = comp["x0"], comp["y0"], comp["x1"], comp["y1"]

            # Direct tag on toe kick (via leader arrow tip strictly inside or direct label)
            direct_tag = None
            for t in crop_tags:
                if t.get("has_leader", False):
                    if arrow_points_into_component(t["target_x"], t["target_y"], comp, t.get("vec_dx", 0.0), t.get("vec_dy", 0.0)):
                        direct_tag = t["tag"]
                        break
                else:
                    t_x0, t_y0, t_x1, t_y1 = t["x0"], t["y0"], t["x1"], t["y1"]
                    if cx0 <= t_x0 and t_x1 <= cx1 and cy0 <= t_y0 and t_y1 <= cy1:
                        direct_tag = t["tag"]
                        break

            # Direct grain on toe kick
            direct_grain = None
            for g in grain_arrows:
                if (cx0 - 20) <= g["x"] <= (cx1 + 20) and (cy0 - 15) <= g["y"] <= (cy1 + 15):
                    direct_grain = g["dir"]
                    break

            all_comps_with_context.append({
                "comp": comp,
                "component_class": "Toe kick",
                "run_type": "Toe Kick",
                "tag": direct_tag,
                "tag_source": "Direct Leader / Tag" if direct_tag else "None",
                "grain_dir": direct_grain,
                "grain_source": "Direct Arrow" if direct_grain else "None",
                "grain_present": direct_grain is not None,
                "is_toe_kick": True
            })

        # ------------------------------------------------------------- #
        # Step E: Step 6 Material Tag & Grain Presence Validation
        # ------------------------------------------------------------- #
        for item in all_comps_with_context:
            comp = item["comp"]
            c_cls = item["component_class"]
            cx0, cy0, cx1, cy1 = comp["x0"], comp["y0"], comp["x1"], comp["y1"]
            tag = item["tag"]
            g_dir = item["grain_dir"]
            g_src = item["grain_source"]
            g_present = item["grain_present"]

            status = "PASS"
            reason = ""

            # Check material rules from Step 1 Web Scraping JSON
            mat_info = material_rules.get(tag, {})
            grain_required = mat_info.get("grain_required", False)

            if not tag:
                # Completely missing material tag after run and elevation inheritance
                if g_present:
                    status = "FAIL"
                    reason = f"{c_cls}: Grain arrow marked, but missing material tag callout"
                else:
                    status = "FAIL"
                    reason = f"{c_cls}: Missing material tag callout"

            elif not grain_required:
                # Material does not require grain (e.g. solid color laminate, stainless steel, quartz, rubber/metal base)
                if g_present and g_src in ["Direct Component Arrow", "Run Arrow", "Direct Arrow"]:
                    status = "REVIEW REQUIRED"
                    reason = f"{c_cls} ({tag}): Non-grain material, but grain arrow marked in CAD"
                else:
                    status = "PASS"
                    reason = f"{c_cls} ({tag}): Non-grain material verified (no grain required)"

            else:
                # Material requires grain (e.g. wood grain laminate / veneer)
                if g_present:
                    status = "PASS"
                    reason = f"{c_cls} ({tag}): Grain required by spec & grain symbol present ({g_src})"
                else:
                    status = "GRAIN MISSING"
                    reason = f"{c_cls} ({tag}): Grain required by spec, but arrow missing in CAD"

            validation_results.append({
                "Image": crop_file,
                "Component": c_cls,
                "Run_Type": item["run_type"],
                "Tag": tag if tag else "UNTAGGED",
                "Tag_Source": item["tag_source"],
                "Grain_Required": grain_required,
                "Grain_Present": g_present,
                "Grain_Direction": g_dir if g_dir else ("None" if not grain_required else "MISSING"),
                "Grain_Source": g_src if g_src != "None" else ("N/A" if not grain_required else "Missing in CAD"),
                "Status": status,
                "Reason": reason,
                "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1
            })

    df_results = pd.DataFrame(validation_results)

    # ------------------------------------------------------------------ #
    # Step F: Save JSON Report & Generate Annotated Visuals
    # ------------------------------------------------------------------ #
    json_out_path = os.path.join(step6_out_dir, "validation_report.json")
    presence_json_path = os.path.join(step6_out_dir, "presence_validation_report.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4)
    with open(presence_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=4)
    print(f"\n[OK] Saved structured Step 6 validation report to: {json_out_path}")

    # Generate Marked-up Images
    if not df_results.empty:
        for crop_name, group in df_results.groupby("Image"):
            step5_filename = f"detected_{crop_name}"
            step5_img_path = os.path.join(step5_dir, step5_filename)
            if not os.path.exists(step5_img_path):
                continue

            img = cv2.imread(step5_img_path)
            if img is None:
                continue

            for _, r in group.iterrows():
                status = r["Status"]
                reason = r["Reason"]
                cx0, cy0, cx1, cy1 = int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])
                tag_label = r["Tag"]
                g_req = r["Grain_Required"]
                g_pres = r["Grain_Present"]

                # Step 6 Labels: ONLY show Grain Requirement & Presence (NO direction!)
                if status == "PASS":
                    color = (0, 190, 0)      # Green
                    grain_label = "Grain: YES" if g_req else "Grain: NO"
                    label_text = f"PASS: {tag_label} [{grain_label}]"
                elif status == "GRAIN MISSING":
                    color = (0, 165, 255)    # Orange
                    label_text = f"GRAIN MISSING: {tag_label}"
                elif "Grain arrow marked" in reason or (g_pres and tag_label in ["UNTAGGED", "TAG MISSING"]):
                    color = (0, 0, 240)      # Red
                    label_text = f"FAIL: Tag Missing [Grain: YES]"
                elif tag_label in ["UNTAGGED", "TAG MISSING"] or status == "FAIL":
                    color = (0, 0, 240)      # Red
                    label_text = f"FAIL: Tag Missing"
                else:
                    color = (0, 165, 255)    # Orange
                    label_text = f"REVIEW: {tag_label} [No Grain Req]"

                cv2.rectangle(img, (cx0, cy0), (cx1, cy1), color, 3)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

                # Draw label background
                lbl_y = max(cy0, th + 8)
                cv2.rectangle(img, (cx0, lbl_y - th - 6), (cx0 + tw + 6, lbl_y + 2), color, -1)
                text_color = (255, 255, 255) if status not in ["REVIEW REQUIRED", "GRAIN MISSING"] else (0, 0, 0)
                cv2.putText(img, label_text, (cx0 + 3, lbl_y - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

            out_img_path = os.path.join(step6_out_dir, f"final_{crop_name}")
            cv2.imwrite(out_img_path, img)

        print(f"[OK] Generated marked-up validation images in: {step6_out_dir}")

    # ------------------------------------------------------------------ #
    # Step G: Summary Metrics Table
    # ------------------------------------------------------------------ #
    if not df_results.empty:
        total   = len(df_results)
        passes  = (df_results["Status"] == "PASS").sum()
        fails   = (df_results["Status"] == "FAIL").sum()
        reviews = (df_results["Status"] == "REVIEW REQUIRED").sum()
        pass_rate = (passes / total) * 100.0 if total > 0 else 0.0

        print(f"\n{'='*70}")
        print(f"   STEP 6 VALIDATION SUMMARY — {project_name}")
        print(f"{'='*70}")
        print(f"   Total Components Verified : {total:,}")
        print(f"   PASS                      : {passes:,} ({pass_rate:.1f}%)")
        print(f"   FAIL                      : {fails:,}")
        print(f"   REVIEW REQUIRED           : {reviews:,}")
        print(f"{'-'*70}")

        print("\n   Breakdown by Material Tag:")
        tag_summary = df_results.groupby(["Tag", "Status"]).size().unstack(fill_value=0)
        for tag_name, row in tag_summary.iterrows():
            p = row.get("PASS", 0)
            f = row.get("FAIL", 0)
            r = row.get("REVIEW REQUIRED", 0)
            print(f"     * {tag_name:12s} -> PASS: {p:3d} | FAIL: {f:3d} | REVIEW: {r:3d}")

        print("\n   Breakdown by Component Run Type:")
        run_summary = df_results.groupby(["Run_Type", "Status"]).size().unstack(fill_value=0)
        for r_name, row in run_summary.iterrows():
            p = row.get("PASS", 0)
            f = row.get("FAIL", 0)
            r = row.get("REVIEW REQUIRED", 0)
            print(f"     * {r_name:24s} -> PASS: {p:3d} | FAIL: {f:3d} | REVIEW: {r:3d}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 6: Rule Engine Validation with Hierarchical Grouping")
    parser.add_argument("--outputs-dir", default=r"c:\Grain direction final\outputs_production_final",
                        help="Path to outputs_production_final folder")
    parser.add_argument("--project", required=True,
                        help="Project folder name (e.g. 'Sutter MBCC', 'Docusign')")
    args = parser.parse_args()
    run_step6(args.outputs_dir, args.project)
