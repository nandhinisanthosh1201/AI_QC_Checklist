"""
Architectural Submittal Room Verification  –  ARCH-ROOM-001
===========================================================
Rule: Room Name and Room Number must match the architectural reference.

Stack
-----
  • Qwen-VL 32B  (via OpenRouter)   — view extraction, room extraction
  • PaddleOCR 2.6.1 / paddlepaddle 2.6.2  — text bbox location for markup
  • OpenCV  — markup drawing

Usage
-----
Edit TEST_CASES at the bottom and run:
    python Roomnn-1.py
"""

# ── Standard library ─────────────────────────────────────────────────────────
import re
import json
import base64
import io
import sys
import os
import logging
import warnings
import traceback
from pathlib import Path
from datetime import datetime

# ── Suppress PaddlePaddle oneDNN/MKL before any paddle import ────────────────
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_onednn_passes", "0")
os.environ["GLOG_minloglevel"]  = "3"
os.environ["PADDLE_LOG_LEVEL"] = "3"

# ── Force UTF-8 on Windows ───────────────────────────────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Third-party ──────────────────────────────────────────────────────────────
import requests
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # allow very large architectural sheets

# ── NumPy 2.x compatibility for imgaug 0.4.0 ─────────────────────────────────
# imgaug uses np.sctypes which was removed in NumPy 2.0.
# Restore it so PaddleOCR 2.6.x (which depends on imgaug) can import cleanly.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int":     [np.int8,    np.int16,   np.int32,    np.int64],
        "uint":    [np.uint8,   np.uint16,  np.uint32,   np.uint64],
        "float":   [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others":  [bool, object, bytes, str],
    }

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARNING] opencv-python not installed — markup drawing disabled.")

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

# Module-level OCR engine — set in __main__
ocr_engine = None


# ============================================================
# OPENROUTER / QWEN CONFIG
# ============================================================
OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "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")

OPENROUTER_MODEL    = "qwen/qwen3-vl-32b-instruct"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# GLOBAL CONFIG
# ============================================================
CONFIG = {
    "rule_code": "ARCH-ROOM-001",
    "rule_name": "Room Name and Room Number Verification",

    # Qwen image resize (longest side, pixels)
    "max_image_side": 1400,

    # Token budgets
    "max_new_tokens_submittal": 2000,
    "max_new_tokens_arch_crop": 1500,

    # Room-name fuzzy match thresholds (0–100)
    "name_match_threshold":   80,
    "name_partial_threshold": 55,

    # PaddleOCR
    "ocr_min_confidence":      0.20,
    "ocr_target_tile_px":      800,  # Target size for adaptive tiling
    "ocr_search_overlap":      0.25,
    "ocr_dedup_iou_threshold": 0.50,
    "bbox_match_score_threshold": 60,

    # Markup drawing
    "markup": {
        "pass_color_bgr":            (0, 200, 0),
        "fail_color_bgr":            (0, 0, 220),
        "review_required_color_bgr": (0, 140, 255),
        "line_thickness":            4,
        "bbox_padding_px":           30,
    },
}


# ============================================================
# TEXT HELPERS
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


def safe_str(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    return "" if s.upper() in {"NULL", "NONE", "NAN", "NA", "N/A", ""} else s


def clean_room_number(value, fallback: str = "") -> str:
    text = normalize_text(value)
    
    # 1. First, try to extract explicit ROOM designations (e.g. ROOM 112)
    m = re.search(r"\b(?:RM|ROOM)\s*(\d{2,5})\b", text)
    if m:
        return m.group(1)
        
    # 2. Match standard room numbers (2-5 digits, optional single letter)
    # Exclude typical dimensions/scales by ensuring it's not followed by " or '
    m2 = re.search(r"\b(\d{2,5}[A-Z]?)\b(?![\"'\s]*(?:IN|FT|MM|CM))", text)
    if m2:
        return m2.group(1)
        
    return fallback


def extract_room_name_from_title(title: str, room_num: str = "") -> str:
    if not title:
        return ""
    
    # Remove common architectural view/drawing prefixes and suffixes
    stop_pattern = r"\b(PLAN|SECTION|ELEVATION|DETAIL|ENLARGED|KEY|KEYPLAN|NORTH|SOUTH|EAST|WEST|WALL|BASE|TALL|CABINET|VIEW|SCALE|TYP|NOTE|REF)\b"
    cleaned = re.sub(stop_pattern, "", title)
    
    # Remove room numbers or stray numbers
    if room_num:
        cleaned = cleaned.replace(room_num, "")
    cleaned = re.sub(r"\b\d+\b", "", cleaned)
    
    return normalize_text(cleaned)


def infer_view_type(title: str) -> str:
    t = normalize_text(title)
    if "KEYPLAN" in t or "KEY PLAN" in t: return "KEYPLAN"
    if "SECTION"   in t: return "SECTION"
    if "DETAIL"    in t: return "DETAIL"
    if "ELEVATION" in t: return "ELEVATION"
    if "PLAN"      in t: return "PLAN"
    return "UNKNOWN"


def normalize_confidence(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def parse_arch_ref(raw) -> dict:
    text = normalize_text(raw)
    if text in {"", "NA", "N A", "N/A", "NONE", "NULL", "NOT AVAILABLE"}:
        return {"arch_ref_raw": "N/A", "arch_ref_status": "N/A",
                "arch_view_no": None, "arch_sheet_no": None}
    text = (text.replace("IAE", "/AE")
                .replace("1AE", "/AE")
                .replace("|AE", "/AE"))
    m = re.search(r"\b(\d{1,3})\s*/\s*([A-Z]{1,6}\s*-?\s*\d{2,6})\b", text)
    if not m:
        return {"arch_ref_raw": text, "arch_ref_status": "INVALID",
                "arch_view_no": None, "arch_sheet_no": None}
    return {"arch_ref_raw": text, "arch_ref_status": "VALID",
            "arch_view_no": m.group(1).strip(),
            "arch_sheet_no": m.group(2).replace(" ", "").strip()}


def room_name_similarity(a: str, b: str) -> int:
    a, b = normalize_text(a), normalize_text(b)
    if not a or not b:
        return 0
    if RAPIDFUZZ_AVAILABLE:
        return max(fuzz.token_sort_ratio(a, b), fuzz.partial_ratio(a, b))
    if a == b: return 100
    if a in b or b in a: return 65
    return 0


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_text(text: str):
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e < s:
        return None
    candidate = text[s:e + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        for cut in range(len(candidate) - 1, max(0, len(candidate) - 200), -1):
            try:
                return json.loads(candidate[:cut] + "}")
            except json.JSONDecodeError:
                continue
    print(f"  [DEBUG] JSON Parse Failed. Raw text was:\n{text[:500]}")
    return None
# ============================================================
# QWEN-VL API
# ============================================================

def _image_to_data_url(image_path, max_side: int) -> str:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def qwen_vl(image_path: str, prompt: str, max_tokens: int = 500, max_side: int = 1400) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY. Please provide your API key.")
        
    data_url = _image_to_data_url(image_path, max_side)
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text",      "text": prompt},
    ]}]
    
    import time
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "https://github.com/arch-room-rule",
        "X-Title":       "Arch Room Rule Engine",
    }
    payload = {"model": OPENROUTER_MODEL, "max_tokens": max_tokens,
               "messages": messages}
    last_exc = None
    for attempt, delay in enumerate([0, 5, 15, 30]):
        if delay:
            print(f"    [API] Retry {attempt}/3 — waiting {delay}s...")
            time.sleep(delay)
        try:
            resp = requests.post(OPENROUTER_BASE_URL, headers=headers,
                                 json=payload, timeout=180)
            if resp.status_code == 429:
                print("    [API] Rate-limited (429), will retry.")
                continue
            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP {resp.status_code}: {resp.text[:300]}")
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_exc = exc
            print(f"    [API] Error attempt {attempt}: {exc}")
    raise RuntimeError(f"OpenRouter API failed: {last_exc}")


# ============================================================
# ARCH-REF → FILENAME MATCHING
# ============================================================

def _local_match_arch_ref(arch_ref: str, filenames: list) -> str | None:
    """
    Pure-Python heuristic: match arch-ref (e.g. '1/AE480') to a filename.
    Strategies (in order):
      1. Full ref alphanum contained in filename  ('1AE480' in 'ARCHAE480JPG')
      2. Sheet number only contained in filename  ('AE480' in 'ARCH AE480.jpg')
      3. Sheet number with ≤1 character OCR typo tolerance
    """
    parsed   = parse_arch_ref(arch_ref)
    sheet    = (parsed["arch_sheet_no"] or "").upper()
    ref_alph = re.sub(r"[^A-Z0-9]", "", arch_ref.upper())

    # Strategy 1 – full ref
    for fn in filenames:
        fn_n = re.sub(r"[^A-Z0-9]", "", fn.upper())
        if ref_alph and ref_alph in fn_n:
            return fn

    # Strategy 2 – sheet number only
    if sheet:
        sh_n = re.sub(r"[^A-Z0-9]", "", sheet)
        for fn in filenames:
            fn_n = re.sub(r"[^A-Z0-9]", "", fn.upper())
            if sh_n and sh_n in fn_n:
                return fn

    # Strategy 3 – sheet number with 1-char typo
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


def match_arch_ref_to_filename(arch_ref: str, filenames: list) -> str | None:
    """Match an arch-ref to the best filename. Local heuristics first."""
    if not filenames:
        return None
    local = _local_match_arch_ref(arch_ref, filenames)
    if local:
        return local
    # Qwen fallback only when local fails
    if not OPENROUTER_API_KEY:
        return None
    prompt = (
        "Match the Arch Ref to the closest filename. "
        "Minor OCR errors are possible. "
        "Return ONLY the exact filename, or 'NONE'.\n\n"
        f"Arch Ref: {arch_ref}\n"
        "Filenames:\n" + "\n".join(filenames)
    )
    try:
        content = _qwen_call(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=100
        ).strip()
    except Exception:
        return None
    return content if content in filenames else None


# ============================================================
# STEP 1 — SUBMITTAL VIEW EXTRACTION
# ============================================================

_SUBMITTAL_PROMPT = """\
You are an expert architectural drawing analyst.

TASK: Identify ALL distinct DRAWING VIEWS on this sheet and return their details in JSON.

A DRAWING VIEW has its own formal title block, which typically contains:
  • A circled or boxed view number  (e.g. ①)
  • A view name  (e.g. "PLAN - RECEPTION" or "ENLARGED PLAN")
  • A scale notation  (e.g. SCALE: 1/8" = 1'-0")
  • Optionally an Arch Ref  (e.g. ARCH REF: 1/AE480)

DO NOT treat the following as drawing views:
  • Section-cut arrows inside a floor plan  (triangle arrow + number/sheet callout)
  • Interior elevation targets  (circle split by a line, numbers inside)
  • Detail call-out bubbles pointing to walls/floors within a drawing
  • Door tags, window tags, north arrows, grid bubbles, column marks
  • Material schedules, hardware schedules, notes columns, revision clouds
  • Title block, revision table, address block, company logo

For each legitimate drawing view extract:
  1. view_id            — integer, 1-based, reading order (left→right, top→bottom)
  2. view_title_raw     — exact text of the view title as printed
  3. view_type          — PLAN / ELEVATION / SECTION / DETAIL / KEYPLAN / UNKNOWN
  4. room_name          — primary room name shown INSIDE the drawing (null if absent)
  5. room_number        — primary room number shown INSIDE the drawing (null if absent)
  6. arch_ref_raw       — arch ref printed IN the title block (e.g. "1/AE480").
                          NOT the internal callout bubbles. Return "N/A" if absent.
  7. title_bbox_percent — tight bbox around the VIEW TITLE LABEL ONLY
                          (the number + name + scale line, NOT the full drawing area)
                          as [x1_pct, y1_pct, x2_pct, y2_pct]  (0.0–100.0 % of image)
  8. extraction_confidence — integer 0–100
  9. evidence_text      — verbatim text you read to support this entry

Rules:
  • room_name / room_number → null if genuinely absent or ambiguous.
  • arch_ref_raw → "N/A" if not printed in the title block.
  • Return ONLY valid JSON. No prose, no markdown, no commentary.

JSON schema:
{
  "submittal_views": [
    {
      "view_id": 1,
      "view_title_raw": "",
      "view_type": "",
      "room_name": null,
      "room_number": null,
      "arch_ref_raw": "N/A",
      "title_bbox_percent": [0.0, 0.0, 50.0, 10.0],
      "extraction_confidence": 0,
      "evidence_text": ""
    }
  ]
}
"""


def _standardize_view(raw: dict, index: int) -> dict:
    title     = normalize_text(raw.get("view_title_raw"))
    pref      = parse_arch_ref(raw.get("arch_ref_raw"))
    room_num  = clean_room_number(raw.get("room_number"), fallback="")
    rn_raw    = safe_str(raw.get("room_name"))
    room_name = (extract_room_name_from_title(rn_raw, room_num)
                 if rn_raw else
                 extract_room_name_from_title(title, room_num))
    bbox_raw  = raw.get("title_bbox_percent") or raw.get("bbox_percent")
    return {
        "view_id":               raw.get("view_id") or index,
        "view_title":            title,
        "view_type":             infer_view_type(title),
        "room_name":             normalize_text(room_name),
        "room_number":           room_num,
        "arch_ref":              pref["arch_ref_raw"],
        "arch_ref_status":       pref["arch_ref_status"],
        "arch_view_no":          pref["arch_view_no"],
        "arch_sheet_no":         pref["arch_sheet_no"],
        "bbox_percent":          bbox_raw,
        "extraction_confidence": normalize_confidence(
                                     raw.get("extraction_confidence")),
        "evidence_text":         normalize_text(raw.get("evidence_text")),
    }


def extract_submittal_views(submittal_path: str) -> dict:
    """
    Call Qwen-VL on the submittal and return standardised view list.
    Returns {"success", "submittal_views", "raw_output", "error"}.
    """
    print(f"  [Qwen] Analysing submittal: {Path(submittal_path).name}")
    try:
        raw = qwen_vl(submittal_path, _SUBMITTAL_PROMPT,
                      max_tokens=CONFIG["max_new_tokens_submittal"],
                      max_side=CONFIG["max_image_side"])
    except Exception as exc:
        return {"success": False, "submittal_views": [],
                "raw_output": "", "error": str(exc)}

    parsed = extract_json_from_text(raw)
    if parsed is None:
        return {"success": False, "submittal_views": [],
                "raw_output": raw,
                "error": "Could not parse JSON from Qwen response."}

    seen, results = set(), []
    for idx, v in enumerate(parsed.get("submittal_views") or [], start=1):
        sv  = _standardize_view(v, idx)
        key = (sv["view_title"], sv["view_type"],
               sv["room_name"], sv["room_number"], sv["arch_ref"])
        if key in seen:
            continue
        seen.add(key)
        results.append(sv)

    print(f"  [Qwen] {len(results)} view(s) detected: "
          + ", ".join(f"'{v['view_title']}'" for v in results))
    return {"success": True, "submittal_views": results,
            "raw_output": raw, "error": None}


# ============================================================
# STEP 2 — ARCH CROP EXTRACTION
# ============================================================

def _build_arch_prompt(arch_view_no: str = None,
                       arch_sheet_no: str = None) -> str:
    """
    Build a targeted arch-crop prompt that tells Qwen exactly which
    view number to look for when the arch ref is known.
    """
    if arch_view_no and arch_sheet_no:
        hint = (
            f"\nIMPORTANT: You are specifically looking for VIEW NUMBER {arch_view_no} "
            f"on sheet {arch_sheet_no} (arch ref: {arch_view_no}/{arch_sheet_no}). "
            f"If the image contains multiple numbered views, focus ONLY on the view "
            f"whose title block shows the number '{arch_view_no}'. "
            f"Extract rooms from THAT view only."
        )
    elif arch_view_no:
        hint = (
            f"\nIMPORTANT: Focus on the view labelled '{arch_view_no}' "
            f"and extract rooms from that view only."
        )
    else:
        hint = ""

    return f"""\
You are an expert architectural drawing analyst.

This image is a portion of an architectural reference sheet.
Extract the room name(s) and room number(s) labelled in this image.{hint}

A single view can contain MULTIPLE labelled rooms — extract ALL of them.

Extract:
  1. arch_view_title — full title text of the view as printed, verbatim
                       (e.g. "PLAN - RECEPTION" or "1 REFLECTED CEILING PLAN")
  2. rooms — list of objects, one per distinct room:
             {{"room_name": "<name as printed>", "room_number": "<number as printed>"}}
             Maximum 20 entries. Use null for a field if it is genuinely missing.
  3. extraction_confidence — integer 0–100
  4. evidence_text — exact text you read to populate the fields

Rules:
  • Read the ACTUAL text. Do NOT infer or guess room names.
  • Room numbers are typically 3–5 digit codes inside or adjacent to rooms.
  • If NO rooms are labelled, return [] for rooms.
  • Return ONLY valid JSON — no prose, no markdown, no commentary.

JSON schema:
{{
  "arch_view_title": "",
  "rooms": [
    {{"room_name": "", "room_number": ""}}
  ],
  "extraction_confidence": 0,
  "evidence_text": ""
}}
"""


def extract_arch_info(arch_path: str,
                      arch_view_no: str = None,
                      arch_sheet_no: str = None) -> dict:
    """
    Call Qwen-VL on an arch image.
    arch_view_no / arch_sheet_no are injected into the prompt so Qwen
    knows exactly which view number to focus on.

    Returns dict with keys:
        mapping_status, arch_view_title, rooms, room_name, room_number,
        extraction_confidence, evidence_text, arch_crop_path, raw_output.
    """
    ref_label = (
        f"{arch_view_no}/{arch_sheet_no}" if arch_view_no and arch_sheet_no
        else arch_view_no or arch_sheet_no or "?"
    )
    print(f"  [Qwen] Arch extraction: {Path(arch_path).name}  "
          f"(target view: {ref_label})")

    prompt = _build_arch_prompt(arch_view_no, arch_sheet_no)
    try:
        raw = qwen_vl(arch_path, prompt,
                      max_tokens=CONFIG["max_new_tokens_arch_crop"],
                      max_side=CONFIG["max_image_side"])
    except Exception as exc:
        return _blank_arch("EXTRACTION_FAILED", str(exc), arch_path)

    parsed = extract_json_from_text(raw)
    if parsed is None:
        return _blank_arch("EXTRACTION_FAILED",
                           "Could not parse JSON from Qwen response.", arch_path)

    title = normalize_text(parsed.get("arch_view_title"))
    rooms = []
    for r in (parsed.get("rooms") or []):
        if not isinstance(r, dict):
            continue
        # Pass empty string as fallback, not the title
        r_num  = clean_room_number(r.get("room_number"), fallback="")
        r_name = safe_str(r.get("room_name"))
        if not r_name and not r_num:
            continue
        if not r_name:
            r_name = extract_room_name_from_title(title, r_num)
        rooms.append({"room_name": normalize_text(r_name),
                      "room_number": r_num})

    # Fallback: top-level room fields (some Qwen responses)
    if not rooms:
        r_num  = clean_room_number(parsed.get("room_number"), fallback="")
        r_name = (safe_str(parsed.get("room_name"))
                  or extract_room_name_from_title(title, r_num))
        if r_name or r_num:
            rooms.append({"room_name": normalize_text(r_name),
                          "room_number": r_num})

    rooms_str = (", ".join(
        f"{r['room_name']} {r['room_number']}".strip() for r in rooms
    ) if rooms else "(none)")
    print(f"  [Qwen] Arch result: title='{title}'  rooms=[{rooms_str}]")

    return {
        "mapping_status":        "FOUND",
        "arch_view_title":       title,
        "rooms":                 rooms,
        "room_name":             rooms[0]["room_name"]   if rooms else None,
        "room_number":           rooms[0]["room_number"] if rooms else None,
        "extraction_confidence": normalize_confidence(
                                     parsed.get("extraction_confidence")),
        "evidence_text":         normalize_text(parsed.get("evidence_text")),
        "arch_crop_path":        str(arch_path),
        "raw_output":            raw,
    }


def _blank_arch(status: str, reason: str, path=None) -> dict:
    return {
        "mapping_status": status, "arch_view_title": None,
        "rooms": [], "room_name": None, "room_number": None,
        "extraction_confidence": 0, "evidence_text": None,
        "arch_crop_path": str(path) if path else None,
        "reason": reason,
    }


# ============================================================
# STEP 3 — RULE VALIDATION
# ============================================================

def validate_room_rule(sv: dict, av: dict) -> dict:
    """
    Compare submittal view (sv) against arch view info (av).
    Returns a rule_result dict with: status, confidence_score, reason, …
    """
    sub_name   = sv.get("room_name")   or ""
    sub_number = sv.get("room_number") or ""

    # ── early exits ──────────────────────────────────────────────────────────
    if sv["arch_ref_status"] == "N/A":
        return _rr(sub_name, sub_number, None, None,
                   nm=None, num_m=None, score=0, conf=40,
                   status="REVIEW_REQUIRED",
                   reason="ARCH REF is N/A — no architectural comparison possible.")

    if sv["arch_ref_status"] == "INVALID":
        return _rr(sub_name, sub_number, None, None,
                   nm=None, num_m=None, score=0, conf=35,
                   status="REVIEW_REQUIRED",
                   reason="ARCH REF format is invalid or unreadable.")

    if av.get("mapping_status") != "FOUND":
        return _rr(sub_name, sub_number, None, None,
                   nm=None, num_m=None, score=0, conf=45,
                   status="REVIEW_REQUIRED",
                   reason=av.get("reason") or "Arch extraction failed.")

    # ── build room list ───────────────────────────────────────────────────────
    arch_rooms = av.get("rooms") or []
    if not arch_rooms and (av.get("room_name") or av.get("room_number")):
        arch_rooms = [{"room_name": av.get("room_name") or "",
                       "room_number": av.get("room_number") or ""}]

    rooms_str = ("; ".join(
        f"{r.get('room_name') or ''} {r.get('room_number') or ''}".strip()
        for r in arch_rooms
    ) or "(none extracted)")

    # ── best-match room ───────────────────────────────────────────────────────
    best = None
    if arch_rooms:
        exact = [r for r in arch_rooms
                 if sub_number and r.get("room_number") == sub_number]
        best = exact[0] if exact else max(
            arch_rooms,
            key=lambda r: room_name_similarity(sub_name, r.get("room_name") or "")
        )

    arch_name   = (best or {}).get("room_name")   or ""
    arch_number = (best or {}).get("room_number") or ""

    number_match = (bool(sub_number) and bool(arch_number)
                    and sub_number == arch_number)
    name_score   = room_name_similarity(sub_name, arch_name)
    name_match   = name_score >= CONFIG["name_match_threshold"]

    base_conf = min(80
        + min(int(av.get("extraction_confidence", 0) * 0.10), 10)
        + min(int(sv.get("extraction_confidence", 0) * 0.10), 10), 100)

    # ── missing room numbers ──────────────────────────────────────────────────
    if not sub_number or not arch_number:
        arch_title = av.get("arch_view_title") or ""
        sub_title  = sv.get("view_title") or ""
        sub_has    = bool(sub_name or sub_number)
        arch_has   = bool(arch_name or arch_number)

        if (not sub_has and arch_title and sub_title
                and room_name_similarity(sub_title, arch_title) >= 70):
            return _rr(sub_title, "", arch_title, "",
                       nm=True, num_m=None,
                       score=room_name_similarity(sub_title, arch_title),
                       conf=60, status="PASS",
                       reason=(f"No room numbers; view titles match: "
                               f"'{sub_title}' ↔ '{arch_title}'."),
                       arch_rooms=arch_rooms, rooms_str=rooms_str)

        if sub_has and not arch_has:
            return _rr(sub_name, sub_number, None, None,
                       nm=False, num_m=False, score=0, conf=70,
                       status="FAIL",
                       reason=(f"Submittal has room '{sub_name} {sub_number}' "
                               f"but arch view '{arch_title}' has no room data. "
                               f"All arch rooms: {rooms_str}."),
                       arch_rooms=arch_rooms, rooms_str=rooms_str)

        if not sub_number and not arch_number and name_match:
            return _rr(sub_name, sub_number, arch_name, arch_number,
                       nm=True, num_m=None, score=name_score, conf=80,
                       status="PASS",
                       reason="Both room numbers are missing, but room names match.",
                       arch_rooms=arch_rooms, rooms_str=rooms_str)

        status = "REVIEW_REQUIRED" if name_match else "FAIL"
        return _rr(sub_name, sub_number, arch_name, arch_number,
                   nm=name_match, num_m=None, score=name_score, conf=50,
                   status=status,
                   reason=(f"Room number missing on one side. "
                           f"Name {'matches' if name_match else 'differs'}. "
                           f"Arch title: '{arch_title}'. Arch rooms: {rooms_str}."),
                   arch_rooms=arch_rooms, rooms_str=rooms_str)

    # ── both sides have numbers ───────────────────────────────────────────────
    if number_match and name_match:
        return _rr(sub_name, sub_number, arch_name, arch_number,
                   nm=True, num_m=True, score=name_score,
                   conf=base_conf, status="PASS",
                   reason="Room name and number both match the architectural reference.",
                   arch_rooms=arch_rooms, rooms_str=rooms_str)

    if number_match and not name_match:
        s = "FAIL" if name_score < CONFIG["name_partial_threshold"] else "REVIEW_REQUIRED"
        return _rr(sub_name, sub_number, arch_name, arch_number,
                   nm=False, num_m=True, score=name_score, conf=70, status=s,
                   reason=(f"Room number matches ({sub_number}), "
                           f"but name differs: '{sub_name}' vs '{arch_name}' "
                           f"(similarity {name_score}). Arch rooms: {rooms_str}."),
                   arch_rooms=arch_rooms, rooms_str=rooms_str)

    if not number_match and name_match:
        return _rr(sub_name, sub_number, arch_name, arch_number,
                   nm=True, num_m=False, score=name_score, conf=70,
                   status="FAIL",
                   reason=(f"Name matches but numbers differ: "
                           f"'{sub_number}' vs '{arch_number}'. "
                           f"Arch rooms: {rooms_str}."),
                   arch_rooms=arch_rooms, rooms_str=rooms_str)

    return _rr(sub_name, sub_number, arch_name, arch_number,
               nm=False, num_m=False, score=name_score, conf=75,
               status="FAIL",
               reason=(f"Neither name nor number match. "
                       f"Submittal: '{sub_name} {sub_number}'. "
                       f"Arch rooms: {rooms_str}."),
               arch_rooms=arch_rooms, rooms_str=rooms_str)


def _rr(sub_name, sub_number, arch_name, arch_number,
        *, nm, num_m, score, conf, status, reason,
        arch_rooms=None, rooms_str="") -> dict:
    return {
        "submittal_room_name":       sub_name,
        "submittal_room_number":     sub_number,
        "architectural_room_name":   arch_name,
        "architectural_room_number": arch_number,
        "architectural_rooms_all":   rooms_str,
        "room_name_match":           nm,
        "room_number_match":         num_m,
        "name_similarity_score":     score,
        "confidence_score":          conf,
        "status":                    status,
        "reason":                    reason,
    }


# ============================================================
# STEP 4 — OCR (PaddleOCR 2.6.x)  for markup bbox location
# ============================================================

def _img_size(path: str) -> tuple:
    return Image.open(path).size   # (width, height)


def _paddle_predict(img_arr: np.ndarray) -> list:
    """
    Run PaddleOCR 2.6.x on a uint8 numpy array.
    Returns list of (text, polygon, score).
    """
    if ocr_engine is None:
        return []
    img_arr = np.ascontiguousarray(img_arr.astype(np.uint8))
    try:
        result = ocr_engine.ocr(img_arr, cls=True)
    except Exception as exc:
        print(f"  [OCR] Engine error: {exc}")
        return []
    if not result or result[0] is None:
        return []
    items = []
    for line in result[0]:
        try:
            poly, (text, score) = line[0], line[1]
            items.append((text, poly, float(score)))
        except (TypeError, ValueError, IndexError):
            continue
    return items


def run_ocr_tiled(image_path: str) -> list:
    """
    Tile the image and run OCR on each tile, mapping detections back
    to full-resolution pixel coordinates.
    Returns list of {"text", "bbox": [x1,y1,x2,y2], "conf"}.
    Falls back to single full-image pass if tiling yields nothing.
    """
    if ocr_engine is None:
        return []

    img  = Image.open(image_path).convert("RGB")
    W, H = img.size
    import math
    tile_px = CONFIG["ocr_target_tile_px"]
    rows = max(1, math.ceil(H / tile_px))
    cols = max(1, math.ceil(W / tile_px))
    ovlp = CONFIG["ocr_search_overlap"]
    tw, th = W / cols, H / rows
    min_conf = CONFIG["ocr_min_confidence"]

    items, seen = [], set()

    for r in range(rows):
        for c in range(cols):
            tx1 = max(0, int(c * tw - tw * ovlp))
            ty1 = max(0, int(r * th - th * ovlp))
            tx2 = min(W, int((c + 1) * tw + tw * ovlp))
            ty2 = min(H, int((r + 1) * th + th * ovlp))
            tile = np.array(img.crop((tx1, ty1, tx2, ty2)))
            try:
                dets = _paddle_predict(tile)
            except Exception:
                continue
            for text, poly, score in dets:
                if score < min_conf:
                    continue
                pts = np.array(poly, dtype=float)
                gx1 = tx1 + float(pts[:, 0].min())
                gy1 = ty1 + float(pts[:, 1].min())
                gx2 = tx1 + float(pts[:, 0].max())
                gy2 = ty1 + float(pts[:, 1].max())
                key = (round(gx1), round(gy1), round(gx2), round(gy2), text)
                if key in seen:
                    continue
                seen.add(key)
                items.append({"text": text,
                              "bbox": [gx1, gy1, gx2, gy2],
                              "conf": score})
    if items:
        return items

    # Full-image fallback
    full = np.array(img)
    for text, poly, score in _paddle_predict(full):
        if score < min_conf:
            continue
        pts = np.array(poly, dtype=float)
        items.append({"text": text,
                      "bbox": [float(pts[:, 0].min()), float(pts[:, 1].min()),
                               float(pts[:, 0].max()), float(pts[:, 1].max())],
                      "conf": score})
    return items


def _qwen_bbox_to_pixel(bp, img_w, img_h, qwen_w, qwen_h) -> list | None:
    """
    Convert Qwen's title_bbox_percent to full-resolution pixel coords.
    Handles both 0-100% format and raw-pixel-on-resized-image format.
    Clamps and pads the result.
    """
    try:
        bx1, by1, bx2, by2 = (float(v) for v in bp[:4])
    except (TypeError, ValueError):
        return None

    pad = CONFIG["markup"]["bbox_padding_px"]

    if max(bx1, by1, bx2, by2) <= 100.0:
        # Percentage format
        x1, y1 = int(bx1 / 100 * img_w), int(by1 / 100 * img_h)
        x2, y2 = int(bx2 / 100 * img_w), int(by2 / 100 * img_h)
    elif max(bx1, by1, bx2, by2) <= 1000.0:
        # Normalized 1000-based scale (Qwen-VL default format)
        x1, y1 = int(bx1 / 1000 * img_w), int(by1 / 1000 * img_h)
        x2, y2 = int(bx2 / 1000 * img_w), int(by2 / 1000 * img_h)
    else:
        # Raw pixel on resized image — clamp then scale
        bx1, bx2 = min(bx1, qwen_w), min(bx2, qwen_w)
        by1, by2 = min(by1, qwen_h), min(by2, qwen_h)
        x1, y1 = int(bx1 / qwen_w * img_w), int(by1 / qwen_h * img_h)
        x2, y2 = int(bx2 / qwen_w * img_w), int(by2 / qwen_h * img_h)

    x1, y1 = max(0, x1 - pad),     max(0, y1 - pad)
    x2, y2 = min(img_w, x2 + pad), min(img_h, y2 + pad)

    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def locate_view_bbox(sv: dict, submittal_path: str,
                     ocr_items: list, used_bboxes: list = None) -> list | None:
    """
    Locate pixel bbox for a view's title label on the submittal.
    Strategy:
      1. OCR fuzzy match on view title / room name / room number
      2. Fall back to Qwen's title_bbox_percent
    Returns [x1,y1,x2,y2] or None.
    """
    if used_bboxes is None:
        used_bboxes = []
        
    vt = normalize_text(sv.get("view_title") or "")
    
    # Pre-calculate Qwen bbox to use as a spatial filter for OCR
    qwen_pixel_bbox = None
    bp = sv.get("bbox_percent")
    if bp and isinstance(bp, (list, tuple)) and len(bp) == 4:
        img_w, img_h = _img_size(submittal_path)
        qw = CONFIG["max_image_side"]
        qs = min(1.0, qw / max(img_w, img_h))
        qwen_pixel_bbox = _qwen_bbox_to_pixel(
            bp, img_w, img_h,
            int(img_w * qs), int(img_h * qs)
        )

    # 1 — OCR search (strict search on view title to prevent random false positives)
    best_score, best_bbox = 0, None
    for it in ocr_items:
        candidate_bbox = it["bbox"]
        
        # Spatial filtering: If Qwen predicted a location, ignore OCR tokens that are way off vertically
        if qwen_pixel_bbox:
            cy = (candidate_bbox[1] + candidate_bbox[3]) / 2
            qy1, qy2 = qwen_pixel_bbox[1], qwen_pixel_bbox[3]
            img_h = _img_size(submittal_path)[1]
            if abs(cy - (qy1 + qy2) / 2) > img_h * 0.20:
                continue
                
        # Prevent picking duplicate boxes from overlapping OCR tiles
        is_duplicate = False
        for ub in used_bboxes:
            if _compute_iou(candidate_bbox, ub) > CONFIG["ocr_dedup_iou_threshold"]:
                is_duplicate = True
                break
        if is_duplicate:
            continue
            
        tn = normalize_text(it["text"])
        if not tn or len(tn) < 2:
            continue
            
        score = 0
        if vt:
            # Require the OCR text to be a substantial chunk of the view title
            if len(tn) >= min(7, int(len(vt) * 0.3)):
                if RAPIDFUZZ_AVAILABLE:
                    score = max(score, fuzz.token_sort_ratio(tn, vt))
                else:
                    if tn == vt: score = max(score, 100)
                    elif len(tn) >= len(vt) * 0.5 and (tn in vt or vt in tn): score = max(score, 85)
                    elif tn in vt or vt in tn: score = max(score, 65)
            
            
        if score > best_score:
            best_score = score
            best_bbox  = candidate_bbox

    if best_bbox and best_score >= CONFIG["bbox_match_score_threshold"]:
        used_bboxes.append(best_bbox)
        return best_bbox

    # 2 — Qwen bbox fallback
    if qwen_pixel_bbox:
        return qwen_pixel_bbox

    return None


# ============================================================
# STEP 5 — MARKUP DRAWING
# ============================================================

def _status_color(status: str):
    m = CONFIG["markup"]
    return {"PASS":            m["pass_color_bgr"],
            "FAIL":            m["fail_color_bgr"],
            "REVIEW_REQUIRED": m["review_required_color_bgr"]}.get(status)


def draw_markups(submittal_path: str,
                 items: list,
                 out_dir: Path) -> str | None:
    """
    Draw coloured boxes + labels on the submittal image.
    items: [{"view_id", "bbox":[x1,y1,x2,y2], "status", "label"}, ...]
    Returns saved image path or None.
    """
    if not items or not CV2_AVAILABLE:
        return None

    pil   = Image.open(submittal_path).convert("RGB")
    img   = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W  = img.shape[:2]
    thick = max(2, CONFIG["markup"]["line_thickness"])
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fsc   = max(0.5, W / 3500.0)

    for it in items:
        bbox   = it.get("bbox")
        status = it.get("status", "")
        label  = it.get("label", f"View {it.get('view_id')}: {status}")
        color  = _status_color(status)
        if color is None or bbox is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        (tw, th), base = cv2.getTextSize(label, font, fsc, thick)
        ly = y1 - 8 if y1 - 8 - th >= 0 else y2 + th + 8
        cv2.rectangle(img, (x1, ly - th - base),
                      (x1 + tw + 6, ly + base), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 3, ly),
                    font, fsc, (255, 255, 255), thick, cv2.LINE_AA)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submittal_markup_all_views.png"
    cv2.imwrite(str(out_path), img)
    return str(out_path)


def display_inline(path: str):
    try:
        from IPython.display import display, Image as IPy
        display(IPy(filename=path))
    except Exception:
        print(f"  (markup: {path})")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(sv: dict, av: dict, rule: dict):
    W = 62
    print("\n" + "─" * W)
    print(f"  View {sv.get('view_id'):>2}  |  {sv.get('view_title')}")
    print(f"  Type        : {sv.get('view_type')}")
    print(f"  Room Name   : {sv.get('room_name') or '(none)'}")
    print(f"  Room Number : {sv.get('room_number') or '(none)'}")
    print(f"  Arch Ref    : {sv.get('arch_ref')}")
    arch_title = av.get("arch_view_title")
    if arch_title:
        print(f"  Arch Title  : {arch_title}")
    arch_rooms = av.get("rooms") or []
    if arch_rooms:
        rs = " | ".join(
            f"{r.get('room_name') or ''} {r.get('room_number') or ''}".strip()
            for r in arch_rooms
        )
        print(f"  Arch Rooms  : {rs}")
    else:
        print(f"  Arch Rooms  : (not extracted)")
    print(f"  ► STATUS    : {rule['status']}  |  Confidence: {rule.get('confidence_score','?')}")
    print(f"  ► Reason    : {rule['reason']}")
    print("─" * W)


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def process_submittal(submittal_path: str,
                      available_arch_paths: list,
                      output_dir: str = "outputs",
                      _arch_cache: dict = None) -> tuple:
    """
    Full pipeline for one submittal image:
      1  Extract views with Qwen-VL
      2  Match arch refs → provided arch image files
      3  Run OCR on submittal for markup bbox location
      4  For each view: extract arch info (Qwen), validate rule
      5  Draw combined markup + save JSON result
    Returns (result_dict, json_path).
    """
    submittal_path = str(submittal_path)
    job_dir    = Path(output_dir) / Path(submittal_path).stem
    markup_dir = job_dir / "submittal_markups"
    vjson_dir  = job_dir / "per_view_json"
    job_dir.mkdir(parents=True, exist_ok=True)

    if _arch_cache is None:
        _arch_cache = {}

    # ── 1. Qwen view extraction ───────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Step 1  —  Extracting views  [{Path(submittal_path).name}]")
    sub_result = extract_submittal_views(submittal_path)
    views = sub_result["submittal_views"]

    if not views:
        error_msg = sub_result.get("error") or "No views detected."
        print(f"  [ERROR] {error_msg}")
        result = {
            "rule_code": CONFIG["rule_code"], "rule_name": CONFIG["rule_name"],
            "submittal_path": submittal_path,
            "run_timestamp": datetime.now().isoformat(),
            "view_validation_results": [],
            "error": error_msg,
        }
        _write_json(result, job_dir / "arch_room_rule_result.json")
        return result, str(job_dir / "arch_room_rule_result.json")

    print(f"  → {len(views)} view(s) detected.")

    # ── 2. Match arch refs → files ────────────────────────────────────────────
    print(f"\n  Step 2  —  Matching arch refs  "
          f"({len(available_arch_paths)} arch file(s) available)")
    arch_for_view: dict[str, str] = {}
    if available_arch_paths:
        filenames    = [Path(p).name for p in available_arch_paths]
        name_to_path = {Path(p).name: str(p) for p in available_arch_paths}
        for sv in views:
            ref = sv.get("arch_ref")
            vid = str(sv["view_id"])
            if not ref or ref == "N/A" or ref.strip().upper() == "NA":
                if arch:
                    arch_for_view[vid] = arch[0]
                    print(f"    View {sv['view_id']} ({ref})  →  [Fallback to {Path(arch[0]).name}]")
                continue
            fn  = match_arch_ref_to_filename(ref, filenames)
            if fn:
                arch_for_view[vid] = name_to_path[fn]
                print(f"    View {sv['view_id']} ({ref})  →  {fn}")
            else:
                if arch:
                    arch_for_view[vid] = arch[0]
                    print(f"    View {sv['view_id']} ({ref})  →  [Fallback to {Path(arch[0]).name}]")
                else:
                    print(f"    View {sv['view_id']} ({ref})  →  [no match]")

    # ── 3. OCR for markup bbox ────────────────────────────────────────────────
    print(f"\n  Step 3  —  OCR for markup locations...")
    ocr_items = run_ocr_tiled(submittal_path)
    print(f"  → {len(ocr_items)} OCR token(s) detected.")

    view_bboxes: dict[str, list | None] = {}
    used_ocr_bboxes = []
    for sv in views:
        vid = str(sv["view_id"])
        bb  = locate_view_bbox(sv, submittal_path, ocr_items, used_ocr_bboxes)
        view_bboxes[vid] = bb
        status_str = f"{[int(v) for v in bb]}" if bb else "NOT LOCATED"
        print(f"    View {sv['view_id']} bbox: {status_str}")

    # ── 4. Per-view validation ────────────────────────────────────────────────
    print(f"\n  Step 4  —  Validating {len(views)} view(s)...")
    view_results   = []
    markup_pending = []

    for sv in views:
        vid      = str(sv["view_id"])
        bbox     = view_bboxes.get(vid)
        arch_ref = sv.get("arch_ref")

        print(f"\n  View {sv['view_id']}: '{sv.get('view_title')}'")

        if vid not in arch_for_view:
            if not arch_ref or arch_ref == "N/A" or arch_ref.strip().upper() == "NA":
                av   = _blank_arch("REVIEW_REQUIRED", "ARCH REF is N/A and no fallback arch file available.", None)
                rule = _quick_rule("REVIEW_REQUIRED", 40,
                                   "ARCH REF is N/A and no fallback arch file available.", sv)
            else:
                av   = _blank_arch("REVIEW_REQUIRED", f"ARCH REF '{arch_ref}' found but no matching arch file was provided.", None)
                rule = _quick_rule("REVIEW_REQUIRED", 40,
                                   f"ARCH REF '{arch_ref}' found but no matching "
                                   f"arch file was provided.", sv)

        elif not Path(arch_for_view[vid]).exists():
            av   = {}
            rule = _quick_rule("REVIEW_REQUIRED", 40,
                               f"Arch file does not exist: {arch_for_view[vid]}", sv)

        else:
            arch_path = arch_for_view[vid]
            cache_key = (arch_path, arch_ref)
            if cache_key in _arch_cache:
                av = _arch_cache[cache_key]
                print(f"  [Cache] Using cached arch result")
            else:
                av = extract_arch_info(
                    arch_path,
                    arch_view_no=sv.get("arch_view_no"),
                    arch_sheet_no=sv.get("arch_sheet_no"),
                )
                _arch_cache[cache_key] = av
            rule = validate_room_rule(sv, av)

        print_summary(sv, av, rule)

        v_res = {
            "view_id":               int(sv["view_id"]),
            "submittal_room_name":   sv.get("room_name"),
            "submittal_room_number": sv.get("room_number"),
            "submittal_view_title":  sv.get("view_title"),
            "submittal_view_type":   sv.get("view_type"),
            "arch_view_title":       av.get("arch_view_title"),
            "arch_crop_path":        av.get("arch_crop_path"),
            "arch_rooms_found":      av.get("rooms", []),
            "rule_result":           rule,
        }
        view_results.append(v_res)

        # Per-view JSON
        vjson_dir.mkdir(parents=True, exist_ok=True)
        _write_json(v_res, vjson_dir / f"view_{vid}_result.json")

        if rule["status"] != "PASS":
            markup_pending.append({
                "view_id": int(sv["view_id"]),
                "bbox":    bbox,
                "status":  rule["status"],
                "label":   f"View {sv['view_id']}: {rule['status']}",
            })

    # ── 5. Draw markup ────────────────────────────────────────────────────────
    print(f"\n  Step 5  —  Drawing markup...")
    mp = draw_markups(submittal_path, markup_pending, markup_dir)
    if mp:
        print(f"  Markup saved  →  {mp}")
        display_inline(mp)
    else:
        print("  (no markup drawn)")

    # ── aggregate ─────────────────────────────────────────────────────────────
    pass_n = sum(1 for r in view_results if r["rule_result"]["status"] == "PASS")
    fail_n = sum(1 for r in view_results if r["rule_result"]["status"] == "FAIL")
    rr_n   = sum(1 for r in view_results if r["rule_result"]["status"] == "REVIEW_REQUIRED")

    result = {
        "rule_code":   CONFIG["rule_code"],
        "rule_name":   CONFIG["rule_name"],
        "submittal_path":        submittal_path,
        "run_timestamp":         datetime.now().isoformat(),
        "total_views":           len(views),
        "pass_count":            pass_n,
        "fail_count":            fail_n,
        "review_required_count": rr_n,
        "markup_image":          mp,
        "view_validation_results": view_results,
        "submittal_raw_qwen":    sub_result,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = job_dir / f"arch_room_rule_result_{ts}.json"
    _write_json(result, jp)
    print(f"\n  Result JSON  →  {jp}")
    print(f"  Summary: PASS={pass_n}  FAIL={fail_n}  REVIEW={rr_n}  "
          f"(total {len(views)} views)")
    return result, str(jp)


# ── helpers ────────────────────────────────────────────────────────────────────

def _quick_rule(status, conf, reason, sv) -> dict:
    return {
        "submittal_room_name":       sv.get("room_name"),
        "submittal_room_number":     sv.get("room_number"),
        "architectural_room_name":   None,
        "architectural_room_number": None,
        "architectural_rooms_all":   "",
        "room_name_match":    None,
        "room_number_match":  None,
        "name_similarity_score": 0,
        "confidence_score":   conf,
        "status":             status,
        "reason":             reason,
    }


def _write_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Silence all PaddleOCR/Paddle loggers before import
    for _lg in ["ppocr", "ppdet", "paddle", "paddlex", "paddleocr",
                "PaddleOCR", "PIL", "urllib3"]:
        _l = logging.getLogger(_lg)
        _l.setLevel(logging.CRITICAL)
        if not _l.handlers:
            _l.addHandler(logging.NullHandler())
    logging.disable(logging.CRITICAL)

    # ── NumPy 2.x / imgaug / paddleocr compatibility (must be set before paddleocr import) ──
    import numpy as _np
    if not hasattr(_np, "sctypes"):
        _np.sctypes = {
            "int":     [_np.int8,     _np.int16,    _np.int32,    _np.int64],
            "uint":    [_np.uint8,    _np.uint16,   _np.uint32,   _np.uint64],
            "float":   [_np.float16,  _np.float32,  _np.float64],
            "complex": [_np.complex64, _np.complex128],
            "others":  [bool, object, bytes, str],
        }
    if not hasattr(_np, "int"):
        _np.int = int
    if not hasattr(_np, "float"):
        _np.float = float
    if not hasattr(_np, "bool"):
        _np.bool = bool

    # ── Mock missing native C++ modules (Polygon, lanms) to allow import ──
    import sys as _sys, types as _types
    class _DummyModule(_types.ModuleType):
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    _sys.modules['Polygon'] = _DummyModule('Polygon')
    _sys.modules['lanms'] = _DummyModule('lanms')

    # ── PaddleOCR 2.6.x init ─────────────────────────────────────────────────
    try:
        import paddleocr as _poc
        _ver = getattr(_poc, "__version__", "unknown")
        print(f"[INFO] PaddleOCR {_ver} / paddlepaddle detected.")
    except Exception as exc:
        print(f"[ERROR] Failed to load PaddleOCR. Error: {exc}")
        print("Make sure you ran: pip install paddleocr==2.6.1 paddlepaddle==2.6.2")
        traceback.print_exc()
        sys.exit(1)

    from paddleocr import PaddleOCR

    print("Loading PaddleOCR engine...")
    try:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except TypeError:
        # Some 2.6.x builds may not accept show_log
        ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

    logging.disable(logging.NOTSET)
    print("PaddleOCR ready.\n")

    import argparse
    parser = argparse.ArgumentParser(description="Arch Room Validation")
    parser.add_argument("--submittal", type=str, help="Path to submittal image")
    parser.add_argument("--arch", type=str, nargs="*", default=[], help="Paths to architectural images")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    if args.submittal:
        TEST_CASES = [{"submittal": args.submittal, "arch_images": args.arch}]
        OUTPUT_DIR = args.output_dir
    else:
        # Fallback to hardcoded cases if no arguments provided
        TEST_CASES = [
            {
                "submittal": "inputs/mod 4_page-0001.jpg",
                "arch_images": ["inputs/AE101.1 - 1.png", "inputs/AE406 -10.png"],
            }
        ]
        OUTPUT_DIR = "outputs"

    n = len(TEST_CASES)
    print(f"Running {n} test case(s)...\n")

    for i, tc in enumerate(TEST_CASES, start=1):
        sub  = tc.get("submittal", "")
        arch = tc.get("arch_images", [])

        print(f"\n{'#'*62}")
        print(f"# Test {i}/{n}: {sub}")
        print(f"{'#'*62}")

        if not sub:
            print("[ERROR] No submittal path specified."); continue
        if not os.path.isfile(sub):
            print(f"[ERROR] Submittal not found: {sub}"); continue

        missing = [p for p in arch if not os.path.isfile(p)]
        if missing:
            print("[WARNING] Arch files not found (will be skipped):")
            for p in missing:
                print(f"    {p}")
            arch = [p for p in arch if os.path.isfile(p)]

        try:
            process_submittal(sub, arch, output_dir=OUTPUT_DIR)
        except Exception as exc:
            print(f"[ERROR] Test {i} failed: {exc}")
            traceback.print_exc()
            sys.exit(1)

        print(f"\n{'─'*62}\n")