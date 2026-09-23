"""
CAD Submittal QC Pipeline
=========================
Validates millwork CAD drawing submittals against a JSON rule engine.

Usage:
    python cad_qc_pipeline.py

Or import and call directly:
    from cad_qc_pipeline import init_openrouter, run_pipeline
    init_openrouter(api_key="sk-or-...")
    run_pipeline(sub_crop_path, arch_crop_path, rule_json_path, output_dir)
"""

# ── Standard library ─────────────────────────────────────────────────────────
import json
import base64
import os
import re
import cv2
import requests
from PIL import Image


# ============================================================
# CONFIGURATION — OpenRouter
# ============================================================
from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = "qwen/qwen3-vl-32b-instruct"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
MAX_TOKENS         = 1012


# ============================================================
# TEXT / JSON HELPERS
# ============================================================

def normalize_text(value) -> str:
    if value is None:
        return ""
    value = str(value).upper()
    for old, new in {
        "|": "I", "—": "-", "–": "-", "_": " ", ".": " ",
        ",": " ", ":": " ", ";": " ", "°": "", "′": "'", "″": '"',
    }.items():
        value = value.replace(old, new)
    value = re.sub(r"[^A-Z0-9/\-\s&@']", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_arch_ref(raw) -> dict:
    """
    Parse arch ref string like '9/AE406' into components.
    Handles common OCR typos: 1AE → /AE, |AE → /AE, IAE → /AE
    """
    text = normalize_text(raw) if raw else ""
    if text in {"", "NA", "N A", "N/A", "NONE", "NULL", "NOT AVAILABLE"}:
        return {
            "arch_ref_raw": "N/A", "arch_ref_status": "N/A",
            "arch_view_no": None,  "arch_sheet_no": None,
        }
    # OCR typo corrections
    text = (text.replace("IAE", "/AE")
                .replace("1AE", "/AE")
                .replace("|AE", "/AE"))
    m = re.search(r"\b(\d{1,3})\s*/\s*([A-Z]{1,6}\s*-?\s*\d{2,6})\b", text)
    if not m:
        return {
            "arch_ref_raw": text, "arch_ref_status": "INVALID",
            "arch_view_no": None, "arch_sheet_no": None,
        }
    return {
        "arch_ref_raw":    text,
        "arch_ref_status": "VALID",
        "arch_view_no":    m.group(1).strip(),
        "arch_sheet_no":   m.group(2).replace(" ", "").strip(),
    }


def _local_match_arch_ref(arch_ref: str, filenames: list) -> str | None:
    """
    Match arch_ref (e.g. '9/AE406') to the best filename in a list.
    Strategies (in order):
      1. Full ref alphanum contained in filename  ('9AE406' in 'AE406__9.png')
      2. Sheet number only                        ('AE406'  in 'AE406__9.png')
      3. Sheet number with ≤1 character OCR typo
    """
    parsed   = parse_arch_ref(arch_ref)
    sheet    = (parsed["arch_sheet_no"] or "").upper()
    ref_alph = re.sub(r"[^A-Z0-9]", "", arch_ref.upper())

    # Strategy 1 — full ref alphanum
    for fn in filenames:
        fn_n = re.sub(r"[^A-Z0-9]", "", fn.upper())
        if ref_alph and ref_alph in fn_n:
            return fn

    # Strategy 2 — sheet number only
    if sheet:
        sh_n = re.sub(r"[^A-Z0-9]", "", sheet)
        for fn in filenames:
            fn_n = re.sub(r"[^A-Z0-9]", "", fn.upper())
            if sh_n and sh_n in fn_n:
                return fn

    # Strategy 3 — sheet number with 1-char typo tolerance
    if sheet and len(sheet) >= 4:
        sh_n = re.sub(r"[^A-Z0-9]", "", sheet)
        for fn in filenames:
            fn_n = re.sub(r"[^A-Z0-9]", "", fn.upper())
            for start in range(max(0, len(fn_n) - len(sh_n) + 1)):
                window = fn_n[start:start + len(sh_n)]
                if len(window) == len(sh_n):
                    diffs = sum(a != b for a, b in zip(sh_n, window))
                    if diffs <= 1:
                        return fn
    return None


def extract_json_from_text(text: str) -> dict | None:
    """
    Robust JSON extractor:
      - Strips ```json fences
      - Strips Qwen3 <think>...</think> blocks
      - Tries strict json.loads first
      - Falls back to brace-trimming repair
    """
    if not text:
        return None
    # strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$",        "", text,         flags=re.MULTILINE)
    # strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e < s:
        return None
    candidate = text[s:e + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # brace-trim repair
        for cut in range(len(candidate) - 1, max(0, len(candidate) - 200), -1):
            try:
                return json.loads(candidate[:cut] + "}")
            except json.JSONDecodeError:
                continue
    print(f"  [JSON Parse Failed] raw snippet: {text[:300]}")
    return None


# ============================================================
# IMAGE ENCODE
# ============================================================

def _encode_image_b64(image_path: str) -> tuple[str, str]:
    """Return (base64_string, mime_type) for a given image path."""
    ext = os.path.splitext(image_path)[1].lower()
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
    }.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


# ============================================================
# QWEN VL API  (OpenRouter)
# ============================================================

def call_qwen_vl_api(prompt: str, image_path, expected_format_desc="") -> dict:
    if isinstance(image_path, list):
        print(f"  [Inference] {OPENROUTER_MODEL} ← {len(image_path)} images")
    else:
        print(f"  [Inference] {OPENROUTER_MODEL} ← {os.path.basename(image_path)}")
        image_path = [image_path]

    try:
        system_content = "You are a specialized AI validating engineering/architectural drawings. "
        if expected_format_desc:
            system_content += f"Strictly return JSON matching this format: {json.dumps(expected_format_desc)}"

        content_list = []
        for p in image_path:
            b64, mime = _encode_image_b64(p)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        content_list.append({
            "type": "text",
            "text": prompt + "\nProvide output strictly as a valid JSON object.",
        })

        payload = {
            "model": OPENROUTER_MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": content_list},
            ],
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type":  "application/json",
            "HTTP-Referer":  "https://kaggle.com",
            "X-Title":       "CAD-QC-Pipeline",
        }

        import time
        max_retries = 5
        for attempt in range(max_retries):
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    wait_time = 10 * (attempt + 1)
                    print(f"  [Inference] HTTP 429 Rate Limited. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    resp.raise_for_status()
            
            resp.raise_for_status()
            break
            
        data = resp.json()

        raw_text = data["choices"][0]["message"]["content"].strip()

        parsed = extract_json_from_text(raw_text)
        if not parsed:
            print(f"  [Inference Error] Could not parse JSON — raw: {raw_text[:300]}")
            return {}
        return parsed

    except requests.exceptions.HTTPError as e:
        print(f"  [Inference Error] HTTP {e.response.status_code}: {e.response.text[:200]}")
        return {}
    except Exception as e:
        print(f"  [Inference Error] {e}")
        return {}


# ============================================================
# STAGES
# ============================================================

def stage_1_input_load(submittal_crop_path, arch_crop_path, rule_json_path):
    print("\n--- Stage 1: Input Load ---")
    with open(rule_json_path, 'r') as f:
        rule = json.load(f)
    submittal_loaded = os.path.exists(submittal_crop_path)
    arch_loaded      = os.path.exists(arch_crop_path) if arch_crop_path else False
    result = {
        "submittal_crop_loaded": submittal_loaded,
        "arch_crop_loaded":      arch_loaded,
        "rule_loaded":           True,
    }
    print(json.dumps(result, indent=2))
    return submittal_loaded, arch_loaded, rule


def stage_2_rule_source_gate(rule):
    print("\n--- Stage 2: Rule Source Gate ---")
    required_sources      = rule.get("required_sources", [])
    run_arch_vs_submittal = (
        "Architectural Drawing" in required_sources or
        "Arch Drawing"          in required_sources
    )
    print(json.dumps({"run_arch_vs_submittal": run_arch_vs_submittal}, indent=2))
    return run_arch_vs_submittal


def stage_3_view_applicability(submittal_crop_path, rule):
    print("\n--- Stage 3: View Applicability ---")
    prompt = f"""
    Analyze the submittal crop image to determine if this view is APPLICABLE to the rule.

    CRITICAL ASSUMPTION:
    - This cropped image represents exactly ONE view, as named by its own title block / view label inside this image.
    - Do NOT assume or infer that any other separate view (e.g. a plan view) exists elsewhere within this same crop.
    - Even if a shape (circle, box, outline, etc.) near the top or any region of the image visually resembles a "plan-style" element, you must treat it as belonging to THIS view (the one named in this crop's title block), NOT as part of a different, separate view.
    - There is only one view boundary in this image. All geometry found anywhere within this image belongs to that single view.

    Check ONLY these two things:
    1. VIEW TYPE: Does the view match any of these types? {json.dumps(rule.get('views_need_to_check', []))}
       - View matching logic: {json.dumps(rule.get('view_matching_logic', {}))}
       - Check the view title/label written in the drawing (e.g. "ELEVATION WEST", "PLAN VIEW")

    2. GEOMETRY: Does the drawing contain the physical element/shape described?
       - Geometry logic: {json.dumps(rule.get('geometry_logic', {}))}
       - Look for the PRESENCE of the physical object or dashed box/outline (even if it has no text label)
       - Do NOT check for any notes or text annotations inside the element
       - Any such shape found anywhere in this image belongs to THIS single view — do not discard it as belonging to a different, unseen view.

    IMPORTANT:
    - geometry_applicable = true if the physical shape/element EXISTS in the drawing
    - geometry_applicable = false ONLY if the element shape is completely absent
    - Missing text/notes inside the element does NOT make geometry_applicable = false

    Also identify the view name/title written in the drawing.
    """
    expected_format = {
        "view_applicable":     True,
        "geometry_applicable": True,
        "matched_rule_view":   "string or null",
        "submittal_view_name": "string",
        "confidence":          90,
        "reasoning":           "Explanation",
    }
    res = call_qwen_vl_api(prompt, submittal_crop_path, expected_format)
    print(json.dumps(res, indent=2))
    return res


def tile_image(image_path: str, out_dir: str) -> list[str]:
    """Splits an image into 4 overlapping quadrants for high-res scanning."""
    img = Image.open(image_path)
    w, h = img.size
    overlap_x, overlap_y = int(w * 0.1), int(h * 0.1)
    boxes = [
        (0, 0, w//2 + overlap_x, h//2 + overlap_y), # Top-Left
        (w//2 - overlap_x, 0, w, h//2 + overlap_y), # Top-Right
        (0, h//2 - overlap_y, w//2 + overlap_x, h), # Bottom-Left
        (w//2 - overlap_x, h//2 - overlap_y, w, h)  # Bottom-Right
    ]
    paths = []
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    for i, box in enumerate(boxes):
        crop = img.crop(box)
        p = os.path.join(out_dir, f"{name}_tile{i}{ext}")
        crop.save(p)
        paths.append(p)
    return paths

def stage_4_arch_trigger_validation(arch_crop_path, rule, arch_ref="NA"):
    print("\n--- Stage 4: ARCH Trigger Validation ---")
    
    import tempfile
    import shutil
    
    drawing_focus = ""
    if arch_ref and arch_ref != "NA" and "/" in arch_ref:
        drawing_num = arch_ref.split('/')[0].strip()
        if drawing_num:
            drawing_focus = f"\n    CRITICAL: This image may contain multiple drawings. You MUST locate and extract the specific view name/title for Drawing Number {drawing_num}. Do not extract the title for any other drawing number."

    # Generic icon/shape-based detection instruction.
    # Only present for rules that define it (e.g. ADA Sign). Empty string for all
    # other rules — zero impact on their existing text-variant-only behavior.
    icon_detection_logic = rule.get("arch_trigger_detection_logic", "")
    icon_focus = ""
    if icon_detection_logic:
        icon_focus = f"""
    ICON-BASED DETECTION (in addition to text matching below):
    - {icon_detection_logic}
    - Do NOT require literal matching text to confirm the trigger if a matching icon/symbol is visually present.
    """

    # Tile the image to simulate zooming in, store in temp folder so it doesn't pollute the project
    temp_dir = tempfile.mkdtemp()
    tiles = tile_image(arch_crop_path, temp_dir)
    image_paths_to_send = [arch_crop_path] + tiles
    print(f"  [Tile Engine] Sliced image into {len(tiles)} overlapping zooming tiles.")

    coord_keys = rule.get("coordinate_keys", {"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"})
    example_coords = {
        coord_keys.get("x1", "x1"): 100,
        coord_keys.get("y1", "y1"): 150,
        coord_keys.get("x2", "x2"): 200,
        coord_keys.get("y2", "y2"): 250
    }
    
    prompt = f"""
    Analyze the ARCH crop image very carefully. The text might be extremely small, so please scan the entire image meticulously, as if you are zooming in. The resolution is high enough to read small tags.{drawing_focus}
    {icon_focus}
    Look specifically for any of these trigger variants/keys or optional entities (as TEXT, if present):
    - Arch trigger variants: {json.dumps(rule.get('arch_trigger_variants', []))}
    - Optional entities: {json.dumps(rule.get('optional_entities', []))}

    EXACT MATCH REQUIRED:
    - A trigger is found ONLY if the text in the drawing EXACTLY matches one of the trigger variants listed above (ignoring case/spacing only).
    - Do NOT treat a similar-looking tag as a match just because it shares a common prefix or partial text with a listed trigger variant.
    - Pay close attention to the exact number/suffix/character differences in any tag — tags that differ only by a number or suffix are distinct, unrelated tags, not the same trigger.
    - If you see a different but similar tag that does not exactly match any listed trigger variant, report arch_trigger_found = false for that tag and mention the similar-but-non-matching tag you saw in the reasoning, so it is not lost.

    Pay special attention to small text, leader lines, and callouts inside or near the room elevations.
    Also identify the specific view name or title written in the architectural drawing.
    CRITICAL: Provide coordinates as a simple JSON array of 4 integers: [x1, y1, x2, y2]. Do NOT use a dictionary for coordinates.
    """
    expected_format = {
        "arch_trigger_found":  True,
        "arch_detected_text":  "string",
        "arch_view_name":      "string",
        "coordinates": [0, 0, 0, 0],
        "confidence": 95,
        "reasoning":  "Explanation",
    }
    res = call_qwen_vl_api(prompt, image_paths_to_send, expected_format)
    
    # Convert array to dict
    coords = res.get("coordinates", [])
    if isinstance(coords, list) and len(coords) >= 4:
        res["coordinates"] = {
            coord_keys.get("x1", "x1"): coords[0], coord_keys.get("y1", "y1"): coords[1],
            coord_keys.get("x2", "x2"): coords[2], coord_keys.get("y2", "y2"): coords[3],
        }
    elif not isinstance(coords, dict):
        res["coordinates"] = {coord_keys.get("x1", "x1"): 0, coord_keys.get("y1", "y1"): 0, coord_keys.get("x2", "x2"): 0, coord_keys.get("y2", "y2"): 0}
    print(json.dumps(res, indent=2))
    
    # Clean up temp tiles
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        pass
        
    return res

def stage_5_submittal_note_validation(submittal_crop_path, rule):
    print("\n--- Stage 5: Submittal Note Validation ---")
    coord_keys = rule.get("coordinate_keys", {"x1": "x1", "y1": "y1", "x2": "x2", "y2": "y2"})
    prompt = f"""
    Analyze the submittal crop image. Check if any of these required notes or entities are present:
    - Detected text variants: {json.dumps(rule.get('detected_text_variants', []))}
    - Required entities: {json.dumps(rule.get('required_entities', []))}

    CRITICAL: Provide coordinates as a simple JSON array of 4 integers: [x1, y1, x2, y2]. Do NOT use a dictionary for coordinates.
    """
    expected_format = {
        "submittal_note_found":    True,
        "submittal_detected_text": "string",
        "coordinates": [0, 0, 0, 0],
        "confidence": 88,
        "reasoning":  "Explanation",
    }
    res = call_qwen_vl_api(prompt, submittal_crop_path, expected_format)
    
    # Convert array to dict
    coords = res.get("coordinates", [])
    if isinstance(coords, list) and len(coords) >= 4:
        res["coordinates"] = {
            coord_keys.get("x1", "x1"): coords[0], coord_keys.get("y1", "y1"): coords[1],
            coord_keys.get("x2", "x2"): coords[2], coord_keys.get("y2", "y2"): coords[3],
        }
    elif not isinstance(coords, dict):
        res["coordinates"] = {coord_keys.get("x1", "x1"): 0, coord_keys.get("y1", "y1"): 0, coord_keys.get("x2", "x2"): 0, coord_keys.get("y2", "y2"): 0}
    print(json.dumps(res, indent=2))
    return res


def stage_6_decision_logic(view_res, arch_res, sub_res, rule):
    print("\n--- Stage 6: Decision Logic ---")
    view_app   = view_res.get("view_applicable",   False)
    geom_app   = view_res.get("geometry_applicable", False)
    arch_found = arch_res.get("arch_trigger_found", False)
    sub_found  = sub_res.get("submittal_note_found", False)
    app_logic  = rule.get("applicability_logic", {})

    if not view_app:
        status = app_logic.get("if_view_not_applicable", "OMITTED")
    elif not geom_app:
        status = app_logic.get("if_geometry_not_applicable", "OMITTED")
    elif arch_found and sub_found:
        status = "PASS"
    elif arch_found and not sub_found:
        status = "FAIL"
    elif not arch_found and not sub_found:
        status = app_logic.get("arch_missing_action", "OMITTED")
    else:
        status = "REVIEW_REQUIRED"

    result = {"status": status}
    print(json.dumps(result, indent=2))
    return result


def stage_7_markup_decision(decision_res, rule):
    print("\n--- Stage 7: Markup Decision ---")
    status = decision_res.get("status")
    if status == "FAIL":
        markup_required = rule.get("markup_required", True)
        markup_color    = rule.get("markup_color",    "red")
        markup_label    = rule.get("markup_label",    "NOTE MISSING")
    elif status == "REVIEW_REQUIRED":
        markup_required, markup_color, markup_label = True, "yellow", "REVIEW REQUIRED"
    else:
        markup_required, markup_color, markup_label = False, None, None

    result = {
        "markup_required": markup_required,
        "markup_color":    markup_color,
        "markup_label":    markup_label,
    }
    print(json.dumps(result, indent=2))
    return result


def stage_8_final_json(rule, view_res, arch_res, sub_res, decision_res, markup_res,
                        output_path="output/final_result.json"):
    print("\n--- Stage 8: Final JSON Save ---")
    final_output = {
        "rule_id":      rule.get("rule_id",   "UNKNOWN_RULE"),
        "rule_name":    rule.get("rule_name",  "Unknown Rule"),
        "sheet_status": decision_res.get("status"),
        "drawing_context": {
            "architectural_view_name": arch_res.get("arch_view_name", "Unknown Arch View"),
            "submittal_view_name":     view_res.get("submittal_view_name", "Unknown Submittal View"),
        },
        "view_results": [{
            "view_id":             "V1",
            "matched_rule_view":   view_res.get("matched_rule_view"),
            "view_applicable":     view_res.get("view_applicable"),
            "geometry_applicable": view_res.get("geometry_applicable"),
            "required_note_found": sub_res.get("submittal_note_found"),
            "status":              decision_res.get("status"),
            "confidence_score":    min(
                view_res.get("confidence", 100),
                arch_res.get("confidence", 100),
                sub_res.get("confidence",  100),
            ),
            "reasoning": decision_res.get("reasoning", (
                f"ARCH trigger found: {arch_res.get('arch_trigger_found')}. "
                f"Submittal note found: {sub_res.get('submittal_note_found')}."
            )),
            "detected_entities": [],
        }],
        "markup_required": markup_res.get("markup_required"),
        "markup_color":    markup_res.get("markup_color"),
        "markup_label":    markup_res.get("markup_label"),
        "severity": rule.get("severity", "Major"),
        "priority": rule.get("priority", "High"),
    }

    if arch_res.get("arch_trigger_found"):
        final_output["view_results"][0]["detected_entities"].append({
            "entity_name":   "Arch Trigger",
            "detected_text": arch_res.get("arch_detected_text", ""),
            "coordinates":   arch_res.get("coordinates", {}),
        })
    if sub_res.get("submittal_note_found"):
        final_output["view_results"][0]["detected_entities"].append({
            "entity_name":   "Submittal Note",
            "detected_text": sub_res.get("submittal_detected_text", ""),
            "coordinates":   sub_res.get("coordinates", {}),
        })

    with open(output_path, 'w') as f:
        json.dump(final_output, f, indent=2)
    print(json.dumps(final_output, indent=2))
    print(f"\nSaved final result to {output_path}")


# ============================================================
# INIT
# ============================================================

def init_openrouter(api_key: str,
                    model: str      = "qwen/qwen3-vl-32b-instruct:free",
                    max_tokens: int = 1500):
    """Configure OpenRouter credentials. Call this before run_pipeline()."""
    global OPENROUTER_API_KEY, OPENROUTER_MODEL, MAX_TOKENS
    OPENROUTER_API_KEY = api_key
    OPENROUTER_MODEL   = model
    MAX_TOKENS         = max_tokens
    print(f"✅ OpenRouter configured: {OPENROUTER_MODEL}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(sub_crop_path: str, arch_crop_path: str, rule: dict, output_dir: str, arch_ref: str = "NA"):
    """
    Validation Flow V3:
    Uses exact matrix logic for rules.
    No early exits for missing geometry.
    """
    arch_na = (not arch_crop_path) or (arch_crop_path.strip().upper() == "NA")

    if not os.path.exists(sub_crop_path):
        print(f"Error: Submittal crop not found at {sub_crop_path}")
        return

    if not arch_na and not os.path.exists(arch_crop_path):
        print(f"Error: ARCH crop not found at {arch_crop_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    sub_base = os.path.splitext(os.path.basename(sub_crop_path))[0]
    rule_base = rule.get("rule_id", "UnknownRule")
    output_path = os.path.join(output_dir, f"{rule_base}_{sub_base}_result.json")
    markup_image_path = os.path.join(output_dir, f"{rule_base}_{sub_base}_markup.png")

    print("\n--- [V3] Stage 3: View Applicability & Geometry Check ---")
    view_res = stage_3_view_applicability(sub_crop_path, rule)
    
    # Early Exit for View Type only
    if not view_res.get("view_applicable", False):
        status = "OMITTED"
        print(f"View not applicable (e.g., not Elevation). Status: {status}")
        stage_8_final_json(rule, view_res, {}, {}, {"status": status, "reasoning": "View type mismatch."},
                           {"markup_required": False}, output_path=output_path)
        return

    sub_geom_app = view_res.get("geometry_applicable", False)

    print("\n--- [V3] Stage 4: ARCH Trigger Validation ---")
    if arch_na:
        print("[SKIPPED — ARCH REF: NA]")
        arch_res = {"arch_trigger_found": False, "arch_view_name": "NA", "confidence": 100}
    else:
        arch_res = stage_4_arch_trigger_validation(arch_crop_path, rule, arch_ref)
        
    arch_sym = arch_res.get("arch_trigger_found", False)

    print("\n--- [V3] Stage 5: Submittal Note Validation ---")
    sub_res = stage_5_submittal_note_validation(sub_crop_path, rule)
    sub_note = sub_res.get("submittal_note_found", False)

    print("\n--- [V3] Stage 6: Matrix Logic (The 5 Cases) ---")
    if arch_sym:
        if sub_note and sub_geom_app:
            status = "PASS"
            reasoning = "Case 1: ARCH has required symbol, SUBMITTAL has note and symbol."
        elif sub_note and not sub_geom_app:
            status = "FAIL"
            reasoning = "Case 3: ARCH has symbol, SUBMITTAL has note, but required symbol is missing in submittal."
        elif not sub_note:
            status = "FAIL"
            reasoning = "Case 2: ARCH has symbol, but SUBMITTAL is missing required note."
    else:
        if not sub_note and not sub_geom_app:
            status = "OMITTED"
            reasoning = "Case 4: No evidence found in ARCH or SUBMITTAL."
        else:
            status = "REVIEW_REQUIRED"
            reasoning = "Case 5: Submittal has indication, but Arch lacks supporting symbol evidence."

    decision_res = {"status": status, "reasoning": reasoning}
    print(f"Final Status: {status} | Reason: {reasoning}")

    print("\n--- [V3] Stage 7: Markup Decision ---")
    if status == "FAIL":
        markup_required = True
        markup_color = "red"
        markup_label = rule.get("markup_label", "NOTE MISSING")
    elif status == "REVIEW_REQUIRED":
        markup_required = True
        markup_color = "yellow"
        markup_label = "REVIEW REQUIRED"
    else:
        markup_required = False
        markup_color = None
        markup_label = None

    markup_res = {"markup_required": markup_required, "markup_color": markup_color, "markup_label": markup_label}

    print("\n--- [V3] Stage 8: Final JSON Save ---")
    stage_8_final_json(rule, view_res, arch_res, sub_res, decision_res, markup_res, output_path=output_path)

    # Draw Markup
    if markup_required:
        print("\n--- Generating Markup Image ---")
        try:
            img = cv2.imread(sub_crop_path)
            if img is not None:
                h, w = img.shape[:2]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = max(0.8, w / 800.0)
                thickness_txt = max(2, int(w / 400.0))
                color = (0, 255, 255) if markup_color == "yellow" else (0, 0, 255)
                text = f"[{status}] {markup_label}"
                
                bi = int(w * 0.10)
                cv2.rectangle(img, (bi, bi), (w - bi, h - bi), color, thickness=max(3, int(w / 150.0)))
                cv2.putText(img, text, (bi, max(30, bi - 10)), font, font_scale, color, thickness_txt)
                cv2.imwrite(markup_image_path, img)
                print(f"Markup saved to {markup_image_path}")
        except Exception as e:
            print(f"Error generating markup: {e}")
