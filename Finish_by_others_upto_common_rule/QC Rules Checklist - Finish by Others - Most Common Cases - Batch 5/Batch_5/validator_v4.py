import os
import json
import cv2
from src.utils.llm_client import call_vision_llm

def extract_json_from_text(raw_text: str) -> dict:
    """Helper to parse JSON from VLM responses."""
    try:
        raw_text = raw_text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error parsing JSON from VLM: {e}")
        return {}

def stage_1_view_gate(submittal_view_type: str, views_to_check: list) -> bool:
    """Stage 1: Check if the current view type is in the required views for this row."""
    if not views_to_check:
        return False
    return submittal_view_type.lower() in [v.lower().strip() for v in views_to_check]

def stage_2_graphic_detection(sub_crop_path: str, arch_crop_path: str, geometry_logic: dict) -> tuple:
    """Stage 2: Check submittal and arch for the required graphic."""
    prompt = geometry_logic.get("boundary_detection", "Look for the relevant graphic.")
    full_prompt = (
        f"{prompt}\n"
        "If you find the graphic, provide its bounding box coordinates [x1, y1, x2, y2] normalized between 0 and 1000.\n"
        "Respond with a valid JSON strictly following this schema:\n"
        "{\"graphic_found\": true/false, \"coordinates\": [0, 0, 0, 0], \"reasoning\": \"string\"}"
    )

    s_g = False
    s_g_coords = [0, 0, 0, 0]
    a_g = False

    # Check Submittal Graphic
    if os.path.exists(sub_crop_path):
        sub_resp = call_vision_llm(prompt=full_prompt, image_path=sub_crop_path)
        sub_data = extract_json_from_text(sub_resp)
        s_g = sub_data.get("graphic_found", False)
        if s_g:
            s_g_coords = sub_data.get("coordinates", [0, 0, 0, 0])

    # Check Arch Graphic Unconditionally
    if arch_crop_path and arch_crop_path != "NA" and os.path.exists(arch_crop_path):
        arch_resp = call_vision_llm(prompt=full_prompt, image_path=arch_crop_path)
        arch_data = extract_json_from_text(arch_resp)
        a_g = arch_data.get("graphic_found", False)

    return s_g, a_g, s_g_coords

def stage_3_tag_note_detection(arch_tags_path: str, active_code_tag: str, sub_crop_path: str, expected_notes: list, anti_hallucination_prompt: str = "") -> tuple:
    """Stage 3: Check Arch Tags via JSON, Check Submittal Note via VLM."""
    a_t = False
    s_n = False
    matched_tags_list = []

    # Check Arch Tags
    if arch_tags_path and os.path.exists(arch_tags_path):
        with open(arch_tags_path, 'r') as f:
            arch_tags_data = json.load(f)
            
        visible_tags = arch_tags_data.get("visible_tags", [])
        extracted_tags = [item.get("tag", "").upper().strip() for item in visible_tags if isinstance(item, dict)]
        
        target_tags = [t.strip().upper() for t in active_code_tag.replace("/", ",").split(",") if t.strip()] if active_code_tag else []
        matched_tags_list = []
        for extracted in extracted_tags:
            for target in target_tags:
                if target in extracted:
                    matched_tags_list.append(target)
        matched_tags_list = list(set(matched_tags_list))
        
        if matched_tags_list:
            a_t = True

    # Check Submittal Note
    if os.path.exists(sub_crop_path) and expected_notes:
        notes_str = ", ".join([f'"{n}"' for n in expected_notes])
        strict_text = f"\n{anti_hallucination_prompt}\n" if anti_hallucination_prompt else "\n"
        
        prompt = (
            f"Analyze the image and check if EXACTLY ONE of the following notes is present (ignoring minor typos or line breaks): {notes_str}.\n"
            "Do NOT assume the note is present just because a graphic is present. You must read the text."
            f"{strict_text}"
            "If found, provide its bounding box coordinates [x1, y1, x2, y2] normalized between 0 and 1000.\n"
            "Respond with a valid JSON strictly following this schema:\n"
            "{\"note_found\": true/false, \"detected_text\": \"string\", \"coordinates\": [0, 0, 0, 0], \"reasoning\": \"string\"}"
        )
        
        sub_resp = call_vision_llm(prompt=prompt, image_path=sub_crop_path)
        sub_data = extract_json_from_text(sub_resp)
        s_n = sub_data.get("note_found", False)
        coords = sub_data.get("coordinates", [0, 0, 0, 0])
        is_zero_coords = False
        import re
        coord_str = str(coords)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", coord_str)
        if len(numbers) >= 4 and all(float(n) == 0 for n in numbers):
            is_zero_coords = True
            
        if s_n and is_zero_coords:
            s_n = False
            sub_data["note_found"] = False
            sub_data["reasoning"] = "System Overridden: AI returned note_found=true but provided [0,0,0,0] coordinates, which indicates a hallucination."
    return a_t, s_n, sub_data if 'sub_data' in locals() else {}, matched_tags_list

def stage_3b_exclusion_check(arch_crop_path: str, exclusion_logic: dict) -> tuple[bool, str]:
    """Stage 3b: Check Arch drawing for semantic exclusion notes."""
    e_n = False
    
    prompt = exclusion_logic.get("detection_prompt", "")
    if not prompt or not arch_crop_path or arch_crop_path == "NA" or not os.path.exists(arch_crop_path):
        return False, ""
        
    full_prompt = (
        f"{prompt}\n"
        "Respond with a valid JSON strictly following this schema:\n"
        "{\"exclusion_found\": true/false, \"detected_text\": \"string\", \"reasoning\": \"string\"}"
    )
    
    arch_resp = call_vision_llm(prompt=full_prompt, image_path=arch_crop_path)
    arch_data = extract_json_from_text(arch_resp)
    
    e_n = arch_data.get("exclusion_found", False)
    
    reasoning_str = ""
    if e_n:
        reasoning_str = f"Exclusion Found: {arch_data.get('detected_text')} - {arch_data.get('reasoning')}"
        print(f"  [!] {reasoning_str}")
        
    return e_n, reasoning_str

def evaluate_truth_table(truth_table: list, a_g: bool, a_t: bool, s_g: bool, s_n: bool) -> str:
    """Stage 4: Resolve Final Status via Truth Table matching."""
    for row in truth_table:
        if (row.get("A_G") == a_g and 
            row.get("A_T") == a_t and 
            row.get("S_G") == s_g and 
            row.get("S_N") == s_n):
            return row.get("result", "FAIL")
    return "FAIL"

def run_pipeline(
    sub_crop_path: str,
    arch_crop_path: str,
    rule: object,
    output_dir: str,
    arch_ref: str,
    schedule_json_path: str,
    arch_tags_json: str,
    view: object
):
    print(f"\n--- [V4] Running 4-Stage Truth Table Pipeline for Tag: {rule.active_code_tag} ---")
    
    os.makedirs(output_dir, exist_ok=True)
    sub_base = os.path.splitext(os.path.basename(sub_crop_path))[0]
    output_path = os.path.join(output_dir, f"result.json")
    
    truth_table = getattr(rule.decision_logic, "truth_table", []) if hasattr(rule, "decision_logic") and rule.decision_logic else []
    exclusion_logic = getattr(rule, "exclusion_logic", {}) if hasattr(rule, "exclusion_logic") else {}
    if not truth_table and not exclusion_logic:
        print("ERROR: No Truth Table or Exclusion Logic found in rule definition!")
        return

    # STAGE 1: View Gate
    views_to_check = getattr(rule.view_applicability, "allowed_view_types", []) if hasattr(rule, "view_applicability") and rule.view_applicability else []
    if not stage_1_view_gate(view.view_type, views_to_check):
        print(f"Skipping: View Type '{view.view_type}' not in {views_to_check}.")
        result_payload = {
            "sheet_status": "OMITTED",
            "reasoning": f"View type '{view.view_type}' is not applicable."
        }
        with open(output_path, 'w') as f:
            json.dump(result_payload, f, indent=4)
        return

    expected_notes = getattr(rule.submittal_validation, "expected_notes", []) if hasattr(rule, "submittal_validation") and rule.submittal_validation else []
    anti_hallucination = getattr(rule.submittal_validation, "anti_hallucination_prompt", "") if hasattr(rule, "submittal_validation") and rule.submittal_validation else ""
    
    if exclusion_logic:
        print("--- Running Sequential Short-Circuit Flow ---")
        a_t, s_n, sub_note_data, matched_tags = stage_3_tag_note_detection(arch_tags_json, rule.active_code_tag, sub_crop_path, expected_notes, anti_hallucination)
        
        # Initialize skipped stage variables
        a_g, s_g, s_g_coords = False, False, [0, 0, 0, 0]
        e_n_reasoning = ""
        
        if a_t:
            final_status = "PASS" if s_n else "FAIL"
            print("  -> Short-circuit: Tag found. Skipping Exclusion and Graphic checks.")
        else:
            e_n, e_n_reasoning = stage_3b_exclusion_check(arch_crop_path, exclusion_logic)
            if e_n:
                final_status = "OMITTED"
                print("  -> Short-circuit: Exclusion note found. Skipping Graphic check.")
            else:
                geometry_logic = getattr(rule, "geometry_logic", {})
                if not geometry_logic and hasattr(rule, "model_extra") and rule.model_extra:
                    geometry_logic = rule.model_extra.get("geometry_logic", {})
                if hasattr(geometry_logic, "model_dump"):
                    geometry_logic = geometry_logic.model_dump()
                elif not isinstance(geometry_logic, dict):
                    geometry_logic = {"boundary_detection": str(geometry_logic)}
                    
                s_g, a_g, s_g_coords = stage_2_graphic_detection(sub_crop_path, arch_crop_path, geometry_logic)
                if a_g:
                    final_status = "PASS" if s_n else "FAIL"
                else:
                    final_status = "OMITTED"
                    
        print(f"SEQUENTIAL INPUTS -> A_T: {a_t}, E_N: {locals().get('e_n', False)}, A_G: {a_g}, S_N: {s_n} => STATUS: {final_status}")
    else:
        print("--- Running Standard Truth Table Flow ---")
        geometry_logic = getattr(rule, "geometry_logic", {})
        if not geometry_logic and hasattr(rule, "model_extra") and rule.model_extra:
            geometry_logic = rule.model_extra.get("geometry_logic", {})
        if hasattr(geometry_logic, "model_dump"):
            geometry_logic = geometry_logic.model_dump()
        elif not isinstance(geometry_logic, dict):
            geometry_logic = {"boundary_detection": str(geometry_logic)}
            
        s_g, a_g, s_g_coords = stage_2_graphic_detection(sub_crop_path, arch_crop_path, geometry_logic)
        a_t, s_n, sub_note_data, matched_tags = stage_3_tag_note_detection(arch_tags_json, rule.active_code_tag, sub_crop_path, expected_notes, anti_hallucination)
        final_status = evaluate_truth_table(truth_table, a_g, a_t, s_g, s_n)
        
        print(f"TRUTH TABLE INPUTS -> A_G: {a_g}, A_T: {a_t}, S_G: {s_g}, S_N: {s_n} => STATUS: {final_status}")
    
    result_payload = {
        "rule_id": rule.rule_id,
        "rule_name": getattr(rule, "rule_name", ""),
        "common_rule_group": getattr(rule, "common_rule_group", ""),
        "active_code_tag": rule.active_code_tag,
        "active_description": rule.active_description,
        "sheet_status": final_status,
        "drawing_context": {
            "arch_reference": arch_ref,
            "architectural_view_name": getattr(view, "view_name", ""),
            "submittal_view_name": getattr(view, "view_name", "")
        },
        "view_results": [
            {
                "view_id": sub_base,
                "view_type": view.view_type,
                "view_applicable": True,
                "active_code_tag": rule.active_code_tag,
                "matched_tags": matched_tags,
                "truth_table_inputs": {
                    "arch_exclusion": locals().get('e_n', False),
                    "arch_graphic": a_g,
                    "arch_tag": a_t,
                    "sub_graphic": s_g,
                    "sub_note": s_n
                },
                "status": final_status,
                "confidence_score": 90.0,
                "reasoning": locals().get('e_n_reasoning', "") if locals().get('e_n_reasoning') else sub_note_data.get("reasoning", "Resolved via Truth Table.")
            }
        ]
    }
    
    if sub_note_data.get("detected_text"):
        result_payload["view_results"][0]["detected_entities"] = [
            {
                "entity_name": "By Others Note",
                "detected_text": sub_note_data.get("detected_text", ""),
                "coordinates": {"left": 0, "top": 0, "right": 0, "bottom": 0}
            }
        ]

    # Needs markup flag
    if hasattr(rule, "markup") and rule.markup:
        statuses_to_mark = getattr(rule.markup, "statuses", ["FAIL", "REVIEW_REQUIRED", "RFI_REQUIRED"])
        if not statuses_to_mark:
            statuses_to_mark = ["FAIL", "REVIEW_REQUIRED", "RFI_REQUIRED"]
        result_payload["needs_markup"] = final_status in statuses_to_mark
    else:
        result_payload["needs_markup"] = False

    with open(output_path, 'w') as f:
        json.dump(result_payload, f, indent=4)
        
    print(f"Result saved to {output_path}")
    
    # V4 Standalone Markup Logic
    if result_payload["needs_markup"]:
        markup_image_path = os.path.join(output_dir, "markup.png")
        
        # Determine Color (BGR)
        color = (0, 0, 255) # Red for FAIL
        if final_status == "RFI_REQUIRED":
            color = (0, 165, 255) # Orange
        elif final_status == "REVIEW_REQUIRED":
            color = (0, 255, 255) # Yellow
            
        # Determine Label Text
        rule_desc = getattr(rule, "active_description", None) or getattr(rule, "rule_name", "UNKNOWN_RULE")
        label_text = f"{final_status} - {rule_desc}"
        
        # Determine Target Bounding Box
        bbox = [0, 0, 0, 0]
        if s_g and s_g_coords and s_g_coords != [0, 0, 0, 0]:
            bbox = s_g_coords
        elif s_n and sub_note_data.get("coordinates"):
            bbox = sub_note_data.get("coordinates")
            
        try:
            img = cv2.imread(sub_crop_path)
            if img is not None:
                h, w = img.shape[:2]
                
                # If both are missing, draw full border
                if not bbox or bbox == [0, 0, 0, 0]:
                    print("Both Graphic and Note missing. Drawing full-border alert.")
                    bbox_pixels = [10, 10, w - 10, h - 10]
                else:
                    # Convert normalized (0-1000) coords to actual pixels
                    bbox_pixels = [
                        int(bbox[0] * w / 1000.0),
                        int(bbox[1] * h / 1000.0),
                        int(bbox[2] * w / 1000.0),
                        int(bbox[3] * h / 1000.0)
                    ]
                
                x1, y1, x2, y2 = bbox_pixels
                
                # Draw Bounding Box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(3, int(w / 150.0)))
                
                # Draw Label Background and Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.6, w / 1000.0)
                thickness = max(2, int(w / 400.0))
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                
                t_x = x1
                t_y = max(0, y1 - text_size[1] - 20)
                
                cv2.rectangle(img, (t_x, t_y), (t_x + text_size[0] + 10, t_y + text_size[1] + 10), color, -1)
                
                text_color = (0, 0, 0) if final_status == "REVIEW_REQUIRED" else (255, 255, 255)
                cv2.putText(img, label_text, (t_x + 5, t_y + text_size[1] + 5), font, font_scale, text_color, thickness, cv2.LINE_AA)
                
                cv2.imwrite(markup_image_path, img)
                print(f"V4 Standalone Markup saved to {markup_image_path}")
        except Exception as e:
            print(f"Error generating V4 markup: {e}")
            
    return None
