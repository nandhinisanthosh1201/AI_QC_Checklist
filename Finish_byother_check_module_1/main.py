import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'requests', 'Pillow', 'opencv-python-headless'])


# CELL 2 — Imports
import os, re, json, cv2, gc, glob
import requests, base64
from PIL import Image
from pathlib import Path

# torch is optional — kept so existing gc/cuda cleanup lines in the pipeline
# continue to work without any modification to the pipeline code.
try:
    import torch
except ImportError:
    class _FakeTorch:
        """No-op stand-in when torch is not installed."""
        class cuda:
            empty_cache  = staticmethod(lambda: None)
            ipc_collect  = staticmethod(lambda: None)
        inference_mode = staticmethod(lambda: __import__('contextlib').nullcontext())
    torch = _FakeTorch()

print('✅ Imports done')

# CELL 3 — OpenRouter API configuration
OPENROUTER_MODEL   = 'qwen/qwen3-vl-32b-instruct'
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

_api_key = os.environ.get('OPENROUTER_API_KEY', '').strip()
if not _api_key:
    raise RuntimeError(
        'OPENROUTER_API_KEY environment variable is not set or is empty.\n'
        'Set it before running:\n'
        '  export OPENROUTER_API_KEY=<your_key>   # Linux / macOS\n'
        '  set    OPENROUTER_API_KEY=<your_key>   # Windows CMD\n'
        '  $env:OPENROUTER_API_KEY="<your_key>"   # Windows PowerShell'
    )

print(f'✅ OpenRouter API configured  (model: {OPENROUTER_MODEL})')

DRAWING_DIR    = r'C:\Finish_byother\test_images'
 
# Your rule engine JSON file
RULE_JSON_PATH = r'C:\Finish_byother\rules\d.json'
 
# Where results and markup images will be saved
OUTPUT_DIR     = r'C:\Finish_byother\safe_o'
 
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DRAWING_DIR, exist_ok=True)
 
DRAWING_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
 
def get_drawing_paths(directory):
    paths = []
    for ext in DRAWING_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, f'*{ext}')))
        paths.extend(glob.glob(os.path.join(directory, f'*{ext.upper()}')))
    return sorted(set(paths))
 
print(f'✅ Paths configured')
print(f'   Drawings : {DRAWING_DIR}')
print(f'   Rules    : {RULE_JSON_PATH}')
print(f'   Output   : {OUTPUT_DIR}')
 
# CELL 5 — Utility functions (with robust multi-stage JSON parser)
def resize_image(input_path, output_dir, max_size=2000):
    stem     = Path(str(input_path)).stem
    out_path = os.path.join(output_dir, f'{stem}_resized.png')
    img      = Image.open(str(input_path)).convert('RGB')
    orig_w, orig_h = img.size
    print(f'  [{stem[:45]}] Original: {orig_w}x{orig_h}', end='')
    if max(orig_w, orig_h) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
        res_w, res_h = img.size
        scale_x = res_w / orig_w
        scale_y = res_h / orig_h
        print(f' -> resized to {res_w}x{res_h}  scale=({scale_x:.3f},{scale_y:.3f})')
    else:
        scale_x = scale_y = 1.0
        print(' -> no resize needed')
    img.save(out_path, format='PNG')
    return out_path, scale_x, scale_y, orig_w, orig_h

print('✅ resize_image ready (with scale tracking)')

def load_rules(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data

# ── JSON repair helpers ──────────────────────────────────────────────────────

def _strip_markdown_fences(text):
    """Remove ```json ... ``` and bare ``` fences."""
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def _fix_python_literals(text):
    """Convert Python True/False/None to JSON true/false/null (whole-word only)."""
    text = re.sub(r'\bTrue\b',  'true',  text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b',  'null',  text)
    return text

# Known nullable string fields that the model often returns as "null" (string)
_NULLABLE_FIELDS = {'matched_rule_view', 'matched_text', 'detected_text'}

def _fix_string_null_for_nullable_fields(text):
    """Convert "field": "null"  ->  "field": null  for known nullable fields."""
    def repl(m):
        field = m.group(1)
        if field in _NULLABLE_FIELDS:
            return f'"{field}": null'
        return m.group(0)
    return re.sub(r'"([^"]+)":\s*"null"', repl, text)

def _fix_single_quoted_values(text):
    """Convert :  'value'  ->  : "value"  after JSON separators."""
    def repl(match):
        content = match.group(1).replace('"', '\\"')
        return ': "' + content + '"' + match.group(2)
    return re.sub(r':\s*\'((?:[^\'\\]|\\.)*)\'\s*([,\n\}])', repl, text)

def _fix_single_quoted_array_elements(text):
    """Convert [ 'a', 'b' ]  ->  [ "a", "b" ] (iterative)."""
    def repl(match):
        content = match.group(2).replace('"', '\\"')
        return match.group(1) + '"' + content + '"' + match.group(3)
    pattern = r'([\[,\s])\'((?:[^\'\\]|\\.)*)\'\s*([,\s\]])'
    old = ''
    while old != text:
        old = text
        text = re.sub(pattern, repl, text)
    return text

def _remove_single_quotes_inside_double_quoted_strings(text):
    """
    Inside an already double-quoted JSON string value, strip single-quote
    characters used as inner quotation marks.
    E.g.  "The note 'DOOR BY OTHERS' is visible"
       -> "The note DOOR BY OTHERS is visible"
    Only touches the value side of  "key": "value with 'quotes'".
    """
    result = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        # Detect start of a JSON string value after ": "
        if ch == '"':
            result.append(ch)
            i += 1
            # Consume string content, removing stray single quotes
            while i < n:
                c = text[i]
                if c == '\\':
                    result.append(c)
                    i += 1
                    if i < n:
                        result.append(text[i])
                        i += 1
                elif c == '"':
                    result.append(c)
                    i += 1
                    break
                elif c == "'":
                    # Drop the single quote (inner quotation mark removal)
                    i += 1
                else:
                    result.append(c)
                    i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)

def _fix_unquoted_strings(text):
    """Quote bare string values that are not true/false/null/numbers."""
    def repl(match):
        key_part = match.group(1)
        value    = match.group(2).strip()
        if value.lower() in ('true', 'false', 'null') or re.match(r'^-?\d+(?:\.\d+)?$', value):
            return key_part + match.group(2)
        return key_part + '"' + value.replace('"', '\\"') + '"'
    return re.sub(r'("[^"]+"\s*:\s*)([^\'"`,\{\}\[\]\n\s][^,\n]*)', repl, text)

def _fix_unescaped_inner_quotes(text):
    """Escape double-quotes that appear inside a JSON string value."""
    result, in_string, escape_next = [], False, False
    for i, ch in enumerate(text):
        if escape_next:
            result.append(ch); escape_next = False; continue
        if ch == '\\':
            result.append(ch); escape_next = True; continue
        if ch == '"':
            if not in_string:
                in_string = True; result.append(ch)
            else:
                rest = text[i+1:].lstrip()
                if not rest or rest[0] in (':', ',', '}', ']', '\n', '\r', ' '):
                    in_string = False; result.append(ch)
                else:
                    result.append('\\"')
        else:
            result.append(ch)
    return ''.join(result)

def _close_open_braces(text):
    """Append missing closing brackets/braces if the JSON is truncated."""
    ob  = text.count('{') - text.count('}')
    ob2 = text.count('[') - text.count(']')
    return text + ']' * max(ob2, 0) + '}' * max(ob, 0)

def _regex_status_recovery(raw_text):
    """
    Last-resort: recover status fields from raw text using regex so the
    pipeline can continue without manufacturing a REVIEW_REQUIRED.
    Returns a minimal dict or raises ValueError.
    """
    # Try to find the outermost status
    m_sheet = re.search(r'"sheet_status"\s*:\s*"(PASS|FAIL|REVIEW_REQUIRED|OMITTED)"', raw_text)
    m_stage = re.search(r'"stage"\s*:\s*"([^"]+)"', raw_text)
    m_rule  = re.search(r'"rule_id"\s*:\s*"([^"]+)"', raw_text)

    # Collect view-level statuses
    view_statuses = re.findall(r'"status"\s*:\s*"(PASS|FAIL|REVIEW_REQUIRED|OMITTED|APPLICABLE)"', raw_text)
    view_ids      = re.findall(r'"view_id"\s*:\s*"([^"]+)"', raw_text)

    if not view_statuses and not m_sheet:
        raise ValueError('regex recovery: no status fields found')

    view_results = []
    for idx, vs in enumerate(view_statuses):
        view_results.append({
            'view_id'            : view_ids[idx] if idx < len(view_ids) else f'V{idx+1}',
            'status'             : vs,
            'confidence_score'   : 0.0,
            'required_note_found': vs == 'PASS',
            'reasoning'          : 'Recovered via regex — original JSON was malformed.',
            'detected_entities'  : [],
        })

    stage = m_stage.group(1) if m_stage else 'unknown'
    rule_id = m_rule.group(1) if m_rule else ''

    recovered = {
        'rule_id'     : rule_id,
        'stage'       : stage,
        'view_results': view_results,
        '_recovered'  : True,
    }
    if m_sheet:
        recovered['sheet_status'] = m_sheet.group(1)
    return recovered


def extract_json(raw_text):
    """
    Parse JSON from model output with multi-stage repair.

    Repair order:
      1. Strip markdown fences.
      2. Convert Python literals (True/False/None → true/false/null).
      3. Fix "null" string for known nullable fields.
      4. Convert single-quoted JSON values to double-quoted.
      5. Remove single quotes inside already double-quoted string values.
      6. Fix unescaped inner double-quotes.
      7. Quote bare unquoted string values.
      8. Close open braces/brackets.
      9. Global single-quote → double-quote sweep.
     10. Regex status recovery — never produces REVIEW_REQUIRED from parse failure.
    """
    # ── Step 1: strip fences ──
    text = _strip_markdown_fences(raw_text)

    # Locate the first JSON object
    start = text.find('{')
    if start == -1:
        # Maybe the raw output contains a recoverable status
        try:
            return _regex_status_recovery(raw_text)
        except ValueError:
            raise ValueError('No JSON object found in model output')
    text = text[start:]

    # ── Fast path ──
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # ── Step 2-8: progressive repair ──
    fixed = _fix_python_literals(text)
    fixed = _fix_string_null_for_nullable_fields(fixed)
    fixed = _fix_single_quoted_values(fixed)
    fixed = _fix_single_quoted_array_elements(fixed)
    fixed = _remove_single_quotes_inside_double_quoted_strings(fixed)
    fixed = _fix_unescaped_inner_quotes(fixed)
    fixed = _fix_unquoted_strings(fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # ── Step 8: close open braces ──
    closed = _close_open_braces(fixed)
    try:
        return json.loads(closed)
    except json.JSONDecodeError:
        pass

    # ── Step 9: aggressive global single-quote sweep ──
    try:
        global_fixed = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', closed)
        return json.loads(global_fixed)
    except Exception:
        pass

    # ── Step 10: regex recovery — must NOT produce REVIEW_REQUIRED from parse failure ──
    print(f'[extract_json] All parse strategies failed — attempting regex status recovery')
    try:
        recovered = _regex_status_recovery(raw_text)
        print(f'[extract_json] Regex recovery succeeded: stage={recovered.get("stage")} '
              f'statuses={[v.get("status") for v in recovered.get("view_results", [])]}')
        return recovered
    except ValueError as ve:
        raise ValueError(
            f'extract_json: all repair strategies failed.\n'
            f'  Last error   : {ve}\n'
            f'  Raw output   :\n{raw_text[:500]}'
        )

print('✅ load_rules + extract_json (robust 10-stage recovery parser) ready')

def _to_bool(val, default=False):
    """Safe boolean coercion. Default False prevents accidental applicability."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ('true', '1', 'yes', 'y', 't', 'applicable'):
        return True
    if s in ('false', '0', 'no', 'n', 'f', 'omitted'):
        return False
    return default

def clean_text(t):
    return str(t or '').upper().replace('-', ' ').replace('_', ' ').strip()

def get_view_name(v):
    name = v.get('view_type') or v.get('detected_view_type') or 'Unknown'
    bad  = {'exact visible drawing title', 'actual view title from drawing',
            'same actual view title from applicable_views', '<exact title from drawing>',
            'same matched rule view from applicable_views'}
    return 'Unknown' if str(name).strip().lower() in bad else name

def get_matched_rule_view(view_name, rule):
    vc = clean_text(view_name)
    if not vc or vc == 'UNKNOWN':
        return None
    best, best_score = None, 0
    for rv in rule.get('views_need_to_check', []):
        rc = clean_text(rv)
        if vc == rc:
            return rv
        if vc.startswith(rc):
            sfx = vc[len(rc):]
            if sfx and sfx[0] in (' ', '-', '/', '.', *'0123456789'):
                if len(rc) > best_score:
                    best_score, best = len(rc), rv
    return best

def is_cached_views_reliable(cached_views, img_w, img_h):
    """Return False when global view detection is too weak to constrain applicability."""
    if not cached_views:
        return False
    sheet_area = max(float(img_w) * float(img_h), 1.0)
    unknown_titles = {'', 'unknown', 'main drawing', '<exact title from drawing>'}
    unknown_count = tiny_count = 0
    for v in cached_views:
        title = str(v.get('detected_view_type', '')).strip().lower()
        vclass = str(v.get('view_type_classification') or v.get('view_type') or '').strip().lower()
        if title in unknown_titles or vclass == 'unknown':
            unknown_count += 1
        box = v.get('scope_coordinates', {})
        if valid_box(box):
            area = (float(box['x2']) - float(box['x1'])) * (float(box['y2']) - float(box['y1']))
            if area < sheet_area * 0.08:
                tiny_count += 1
        else:
            tiny_count += 1
        if normalize_confidence(v.get('confidence_score', 75)) < 40:
            return False
    if unknown_count == len(cached_views):
        return False
    if tiny_count == len(cached_views):
        return False
    return True

def cached_views_to_hints(cached_views):
    hints = []
    for v in cached_views or []:
        hints.append({
            'view_id'   : v.get('view_id'),
            'view_title': get_view_name(v),
            'view_type' : v.get('view_type_classification') or v.get('view_type') or 'Unknown',
            'arch_ref'  : v.get('arch_ref', ''),
        })
    return hints

def normalize_confidence(s):
    try:
        s = float(s)
    except:
        return 0.0
    if 0 < s <= 1:
        s *= 100
    return round(min(max(s, 0.0), 100.0), 2)

def normalize_status(st, rule):
    allowed = rule.get('allowed_status', ['PASS', 'FAIL', 'REVIEW_REQUIRED', 'OMITTED'])
    return st if st in allowed else 'REVIEW_REQUIRED'

def valid_box(box):
    try:
        x1, y1 = float(box.get('x1', 0)), float(box.get('y1', 0))
        x2, y2 = float(box.get('x2', 0)), float(box.get('y2', 0))
        return x2 > x1 and y2 > y1
    except:
        return False

def get_box_ints(box):
    return (int(float(box['x1'])), int(float(box['y1'])),
            int(float(box['x2'])), int(float(box['y2'])))

def clamp_box(box, img_w, img_h):
    return {
        'x1': max(0, min(float(box.get('x1', 0)), img_w)),
        'y1': max(0, min(float(box.get('y1', 0)), img_h)),
        'x2': max(0, min(float(box.get('x2', 0)), img_w)),
        'y2': max(0, min(float(box.get('y2', 0)), img_h)),
    }

print('✅ Utility functions ready')

def ask_qwen(image_path, prompt, max_new_tokens=512):
    """Send image + prompt to OpenRouter Qwen2.5-VL-32B and return the generated text.

    Signature is identical to the original local-model version so the entire
    pipeline works without any further modification.
    """
    # ── 1. Read image and encode as base64 ───────────────────────────────────
    with open(str(image_path), 'rb') as _f:
        _image_bytes = _f.read()
    _b64_image = base64.b64encode(_image_bytes).decode('utf-8')

    _mime_map = {
        '.png': 'image/png',  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.bmp': 'image/bmp',  '.tif': 'image/tiff', '.tiff': 'image/tiff',
    }
    _mime = _mime_map.get(Path(str(image_path)).suffix.lower(), 'image/png')

    # ── 2. Build OpenAI-compatible multimodal payload ─────────────────────────
    _payload = {
        'model'     : OPENROUTER_MODEL,
        'max_tokens': max_new_tokens,
        'messages'  : [{
            'role': 'user',
            'content': [
                {
                    'type'     : 'image_url',
                    'image_url': {'url': f'data:{_mime};base64,{_b64_image}'}
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        }]
    }

    _headers = {
        'Authorization': f'Bearer {_api_key}',
        'Content-Type' : 'application/json',
    }

    # ── 3. Call OpenRouter API ────────────────────────────────────────────────
    try:
        _resp = requests.post(
            OPENROUTER_API_URL,
            headers=_headers,
            json=_payload,
            timeout=180,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            '[ask_qwen] Request timed out after 180 s — '
            'try reducing max_new_tokens or image size'
        )
    except requests.exceptions.RequestException as _e:
        raise RuntimeError(f'[ask_qwen] Network error contacting OpenRouter: {_e}')

    # ── 4. Validate response ──────────────────────────────────────────────────
    if _resp.status_code != 200:
        raise RuntimeError(
            f'[ask_qwen] OpenRouter API returned HTTP {_resp.status_code}.\n'
            f'Response: {_resp.text[:600]}'
        )

    try:
        return _resp.json()['choices'][0]['message']['content']
    except (KeyError, IndexError) as _e:
        raise RuntimeError(
            f'[ask_qwen] Unexpected API response structure: {_e}\n'
            f'Response: {str(_resp.json())[:400]}'
        )

print(f'✅ ask_qwen ready  (OpenRouter API · {OPENROUTER_MODEL} · base64 vision)')


_JSON_OUTPUT_RULES = """
STRICT JSON OUTPUT RULES — follow exactly:
- Return valid JSON only. Start directly with { and end with }.
- Use double quotes for all JSON keys and string values.
- Do not use single quotes anywhere in the JSON response.
- Do not use quotes of any kind inside string values. For example, write DOOR BY OTHERS not 'DOOR BY OTHERS'.
- Use null, true, and false as JSON literals. Do not write "null", "true", "false", True, False, or None.
- Do not wrap output in ```json or markdown code fences under any condition.
- Do not add any explanation, preamble, or text before or after the JSON.
""".strip()

def build_global_detection_prompt(img_w, img_h):
    """Generates prompt to detect all drawing views on the page once."""
    lines = [
        "You are a strict Millwork CAD View Detector.",
        "TASK: Detect all individual drawing views (e.g. Elevation, Section, Plan Section, Plan, Detail, or Unknown) on the page.",
        "For each detected view, extract:",
        "  - view_id: unique identifier starting from V1 (V1, V2, V3, etc.)",
        "  - detected_view_type: the exact visible drawing title or view name as written on the sheet (including rotated text).",
        "  - view_type_classification: classify as one of 'Elevation', 'Section', 'Plan Section', 'Plan', 'Detail', or 'Unknown'.",
        f"  - scope_coordinates: pixel coordinates (x1, y1, x2, y2) bounding box of the view. x2 <= {img_w}, y2 <= {img_h}.",
        "",
        _JSON_OUTPUT_RULES,
        "",
        "Return ONLY a valid JSON object in the following format (no markdown):",
        "{",
        '  "view_results": [',
        "    {",
        '      "view_id": "V1",',
        '      "detected_view_type": "<exact title from drawing>",',
        '      "view_type_classification": "<Elevation|Section|Plan Section|Plan|Detail|Unknown>",',
        '      "scope_coordinates": {"x1": 100, "y1": 150, "x2": 600, "y2": 800}',
        "    }",
        "  ]",
        "}"
    ]
    return '\n'.join(lines)

def run_global_view_detection(resized_path, img_w, img_h):
    """Runs global view detection once per drawing; clamps boxes and normalizes titles."""
    prompt = build_global_detection_prompt(img_w, img_h)
    raw = ask_qwen(resized_path, prompt)
    print(f'[GLOBAL DETECTION RAW]\n{raw[:600]}')
    result = extract_json(raw)
    views = result.get('view_results', [])
    clean_views = []
    for idx, v in enumerate(views):
        title = v.get('detected_view_type')
        if not title or title == '<exact title from drawing>':
            title = v.get('view_type_classification', 'Unknown')
        box = clamp_box(v.get('scope_coordinates', {}), img_w, img_h)
        if not valid_box(box):
            box = {'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h}
        clean_views.append({
            'view_id'                 : f'V{idx+1}',
            'detected_view_type'      : title,
            'view_type_classification': v.get('view_type_classification', 'Unknown'),
            'confidence_score'        : normalize_confidence(v.get('confidence_score', 75)),
            'scope_coordinates'       : box,
        })
    if not clean_views:
        clean_views = [{
            'view_id': 'V1', 'detected_view_type': 'Unknown',
            'view_type_classification': 'Unknown', 'confidence_score': 0.0,
            'scope_coordinates': {'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h},
        }]
    return clean_views

print('✅ Global view detection functions ready')

# ===== CELL: Updated Applicability Stage =====

def build_applicability_prompt(rule, img_w, img_h, cached_views=None, use_cached_hints=False):
    """Full-sheet applicability scan; optional cached-view hints when global detection is reliable."""
    views_list = ', '.join(rule.get('views_need_to_check', []))
    geom       = rule.get('geometry_logic', {})
    geom_conds = (geom.get('wall_interaction_conditions')
                  or geom.get('applicable_conditions')
                  or [geom.get('boundary_detection', 'See geometry_logic')])
    omit       = rule.get('omit_condition', [])

    hint_lines = []
    if use_cached_hints and cached_views:
        for v in cached_views:
            hint_lines.append(
                f"  View {v.get('view_id')}: {get_view_name(v)} "
                f"(type={v.get('view_type_classification', 'Unknown')})"
            )
    views_hint = '\n'.join(hint_lines) if hint_lines else None

    lines = [
        'You are a strict Millwork CAD QC Auditor.',
        'TASK: determine if this rule is APPLICABLE to the drawing.',
        '',
        f'RULE ID   : {rule.get("rule_id")}',
        f'RULE NAME : {rule.get("rule_name")}',
        f'VIEWS TO CHECK: {views_list}',
        '',
    ]
    if views_hint:
        lines += [
            'PRE-DETECTED VIEWS IN THIS DRAWING (use these exact titles when possible):',
            views_hint,
            '(Match detected view titles to VIEWS TO CHECK. Use detected titles in detected_view_type.)',
            '',
        ]
    lines += [
        'GEOMETRY CONDITIONS (any one = applicable):',
        json.dumps(geom_conds, indent=2),
        '',
        'OMIT CONDITIONS:',
        json.dumps(omit, indent=2),
        '',
        f'IMAGE SIZE: {img_w}x{img_h} pixels',
        '',
        'INSTRUCTIONS:',
        '1. Read ALL view titles (including rotated text).',
        '2. Does each view match VIEWS TO CHECK? YES/NO',
        '3. Does the required visible physical geometry for this rule clearly exist in the matched view? YES/NO',
        f'4. scope_coordinates: pixel (x1,y1,x2,y2). x2<={img_w}, y2<={img_h}.',
        '5. DO NOT search for note text here. Geometry check only.',
        '6. Set status=OMITTED when view scope does not match OR geometry is absent.',
        '7. Set status=REVIEW_REQUIRED only when geometry/view match is genuinely unclear.',
        '8. Do NOT set geometry_applicable=true only because the detected view type matches VIEWS TO CHECK.',
        '9. Set geometry_applicable=true ONLY when the required visible geometry condition is clearly present in that same view.',
        '',
        _JSON_OUTPUT_RULES,
        '',
        'Return ONLY valid JSON (no markdown):',
        '{',
        f'  "rule_id": "{rule.get("rule_id")}",',
        '  "stage": "applicability",',
        '  "view_results": [{',
        '    "view_id": "V1",',
        '    "detected_view_type": "<exact title from drawing>",',
        f'   "matched_rule_view": "<one of: {views_list} OR null>",',
        '    "view_applicable": true,',
        '    "geometry_applicable": <true only if required visible physical geometry exists; otherwise false>,,',
        '    "status": "<APPLICABLE|OMITTED|REVIEW_REQUIRED>",',
        '    "confidence_score": 85.0,',
        '    "reasoning":"The required visible physical geometry for this rule is clearly present in this view.",',
        '    "scope_coordinates": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}',
        '  }]',
        '}',
    ]
    return '\n'.join(lines)

def build_note_prompt(rule, applicable_views, img_w, img_h):
    variants  = rule.get('detected_text_variants', [])[:8]
    required  = rule.get('required_entities', [])
    pass_cond = rule.get('pass_condition', [])
    fail_cond = rule.get('fail_condition', [])
    view_ctxs = []
    for v in applicable_views:
        sc = v.get('scope_coordinates', {})
        view_ctxs.append({
            'view_id'             : v.get('view_id'),
            'view_type'           : get_view_name(v),
            'matched_rule_view'   : v.get('matched_rule_view'),
            'search_region_pixels': {
                'x1': int(sc.get('x1', 0)),    'y1': int(sc.get('y1', 0)),
                'x2': int(sc.get('x2', img_w)), 'y2': int(sc.get('y2', img_h))
            }
        })
    lines = [
        'You are a strict Millwork CAD QC Auditor. Validate whether a required note exists.',
        '',
        f'RULE ID   : {rule.get("rule_id")}',
        f'RULE NAME : {rule.get("rule_name")}',
        '',
        'REQUIRED TEXT — find ANY ONE of these variants:',
        json.dumps(variants, indent=2),
        '',
        'REQUIRED ENTITIES:',
        json.dumps(required, indent=2),
        '',
        f'PASS CONDITIONS: {json.dumps(pass_cond)}',
        f'FAIL CONDITIONS: {json.dumps(fail_cond)}',
        '',
        'VIEWS TO VALIDATE (with pixel search regions):',
        json.dumps(view_ctxs, indent=2),
        '',
        f'IMAGE SIZE: {img_w}x{img_h} pixels',
        '',
        'MANDATORY 3-STEP PROCESS:',
        '',
        'STEP 1 - OCR SCAN of search_region_pixels:',
        '  Read EVERY text inside the bounding box.',
        '  Include: annotations, callouts, leader lines, rotated text, small labels.',
        '  List all text strings you can read.',
        '',
        'STEP 2 - STRICT MATCH CHECK:',
        '  Compare your Step-1 text list against REQUIRED TEXT variants.',
        '  Partial match is NOT allowed.',
        '  Extra prefixes or different note intent are NOT allowed.',
        '  Do not treat CENTERLINE OF IN-WALL STRAPPING as IN-WALL STRAPPING BY OTHERS.',
        '  Found -> PASS only when detected text matches one REQUIRED TEXT variant after normalizing spaces, hyphens, and line breaks.',
        '  Not found -> FAIL. Unclear -> REVIEW_REQUIRED.',
        '',
        'STEP 3 - COORDINATES:',
        f'  Give pixel (x1,y1,x2,y2) of the text location. x2<={img_w}, y2<={img_h}.',
        '  If not found: use search_region_pixels as fallback.',
        '',
        'ANTI-HALLUCINATION: NEVER say PASS unless you can actually read the text.',
        '',
        _JSON_OUTPUT_RULES,
        '  - detected_text field: write plain text only, no surrounding quotes of any kind.',
        '',
        'Return ONLY valid JSON (no markdown):',
        '{',
        f'  "rule_id": "{rule.get("rule_id")}",',
        '  "stage": "note_validation",',
        '  "view_results": [{',
        '    "view_id": "V1",',
        '    "detected_view_type": "<same as input>",',
        '    "matched_rule_view": "<same as input>",',
        '    "status": "<PASS|FAIL|REVIEW_REQUIRED>",',
        '    "confidence_score": 0.0,',
        '    "required_note_found": false,',
        '    "reasoning": "<Step1 text list + Step2 match decision, single-quotes only>",',
        '    "detected_entities": [{',
        '      "entity_name": "<from required_entities>",',
        '      "detected_text": "<plain text OR NOT FOUND>",',
        '      "coordinates": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}',
        '    }]',
        '  }]',
        '}',
    ]
    return '\n'.join(lines)


def build_binary_fallback_prompt(rule, view, img_w, img_h):
    variants = rule.get('detected_text_variants', [])[:5]
    sc       = view.get('scope_coordinates', {})
    x1, y1   = int(sc.get('x1', 0)),    int(sc.get('y1', 0))
    x2, y2   = int(sc.get('x2', img_w)), int(sc.get('y2', img_h))
    lines = [
        'Look at this CAD drawing carefully.',
        f'Search ONLY in pixel region: x={x1} to {x2}, y={y1} to {y2}',
        '',
        'Is ANY of this text visible in that region?',
        json.dumps(variants, indent=2),
        '',
        'Read all text carefully: small labels, rotated text, leader-line annotations.',
        '',
        _JSON_OUTPUT_RULES,
        '',
        'Reply with ONLY this JSON and nothing else:',
        '{"found": true, "matched_text": "EXACT TEXT YOU SAW OR null", '
        f'"confidence": 85.0, "x1": {x1}, "y1": {y1}, "x2": {x2}, "y2": {y2}}}',
    ]
    return '\n'.join(lines)

print('✅ All 3 prompts ready (applicability + note_validation + binary_fallback)')

from difflib import SequenceMatcher

import re

def _normalize_text(text):
    """Normalize for exact variant matching.

    Allowed normalizations only:
      - uppercase
      - newlines  → space
      - hyphens   → space
      - collapse multiple spaces
      - strip non-alphanumeric chars (brackets, parentheses, punctuation)

    NOT allowed: prefix/suffix/substring tolerance — that is handled by
    word-count enforcement in _fuzzy_match.
    """
    text = str(text or "").upper()
    text = text.replace("\n", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fuzzy_match(expected, detected, threshold=0.90):
    expected = _normalize_text(expected)
    detected = _normalize_text(detected)

    if not expected or not detected:
        return False

    if expected == detected:
        return True

    # No substring match. Prevents:
    # STRAPPING BY OTHERS matching CENTERLINE OF IN-WALL STRAPPING STRAPPING BY OTHERS
    if len(expected.split()) != len(detected.split()):
        return False

    return SequenceMatcher(None, expected, detected).ratio() >= threshold

def post_process_note_view(nv, rule, app_view=None):
    """Stable post-process with exact applicability gating and conservative PASS handling."""
    entities  = nv.get('detected_entities', [])
    status    = nv.get('status', 'REVIEW_REQUIRED')
    reasoning = nv.get('reasoning', '').upper()

    texts_to_check = []
    for key in ('detected_text', 'evidence_text', 'matched_text'):
        if key in nv:
            texts_to_check.append(nv[key])
    for e in entities:
        if 'detected_text' in e:
            texts_to_check.append(e.get('detected_text'))

    variant_matched = False
    for text in texts_to_check:
        text_str = str(text or '').strip().upper()
        if text_str and text_str not in ('NOT FOUND', 'NULL', 'NONE'):
            for var in rule.get('detected_text_variants', []):
                if _fuzzy_match(var, text_str):
                    variant_matched = True
                    break
        if variant_matched:
            break

    # ── STRICT VARIANT ENFORCEMENT ────────────────────────────────────────────
    # When no entity text exactly matches a rule variant (after normalization),
    # the detected text is irrelevant — it is a different note, a prefix variant,
    # or a partial match.  Overwrite unconditionally.
    #
    # Allowed normalization: uppercase, newline→space, hyphen→space,
    #                        collapse spaces, strip punctuation.
    # NOT allowed: prefix words, suffix words, substring, intent-based match.
    #
    # Example that MUST stay FAIL:
    #   detected  : "CENTERLINE OF IN-WALL STRAPPING (STRAPPING BY OTHERS)"
    #   variant   : "IN-WALL STRAPPING BY OTHERS"
    #   → word counts differ after normalization → _fuzzy_match returns False
    #   → variant_matched stays False → enforcement fires below
    if not variant_matched:
        # Determine whether the view is applicable (FAIL is meaningful only then).
        is_applicable = (
            app_view is not None
            and app_view.get('view_applicable') is True
            and app_view.get('geometry_applicable') is True
            and app_view.get('matched_rule_view') is not None
        )
        # Always clear detected_text so the output never shows the wrong note.
        for ent in entities:
            if ent.get('detected_text') and ent['detected_text'].upper() not in ('NOT FOUND', 'NULL', 'NONE', ''):
                print(f"  [STRICT-REJECT] {nv.get('view_id')} detected_text "
                      f"'{ent['detected_text'][:60]}' "
                      f"does not match any variant → overwriting to NOT FOUND")
                ent['detected_text'] = 'NOT FOUND'
        # Force required_note_found off.
        nv['required_note_found'] = False
        # Force FAIL (not PASS, not REVIEW_REQUIRED) when the view is applicable
        # and the note is simply absent / wrong.
        if is_applicable and status in ('PASS', 'REVIEW_REQUIRED'):
            nv['status'] = 'FAIL'
            nv['_strict_rejected'] = f'variant_matched=False; original_status={status}'
            status = 'FAIL'
            print(f"  [STRICT-REJECT] {nv.get('view_id')} status forced FAIL "
                  f"(no exact variant match; was {nv.get('_strict_rejected', '')})") 

    found_kws    = ['WAS FOUND', 'IS PRESENT', 'IS CLEARLY VISIBLE', 'IS VISIBLE',
                    'CLEARLY VISIBLE', 'NOTE IS FOUND', 'TEXT IS FOUND',
                    'STEP 2: MATCH FOUND', 'STEP 2: FOUND', 'MATCH FOUND']
    negative_kws = ['DOES NOT CONTAIN', 'NOT MATCHING', 'INCORRECT', 'WRONG TEXT',
                    'MISSING REQUIRED', 'NOT THE REQUIRED', 'NOT PRESENT', 'ABSENT',
                    'STEP 2: NOT FOUND', 'STEP 2: NO MATCH', 'NO MATCH']
    confirms    = any(k in reasoning for k in found_kws)
    flags_wrong = any(k in reasoning for k in negative_kws)

    # REVIEW_REQUIRED -> FAIL only when view and geometry are clearly applicable and note is clearly missing.
    if status == 'REVIEW_REQUIRED' and app_view is not None:
        clearly_applicable = (
            app_view.get('view_applicable') is True
            and app_view.get('geometry_applicable') is True
            and app_view.get('matched_rule_view') is not None
        )
        clearly_missing = (
            flags_wrong and not confirms and not variant_matched
            and not nv.get('required_note_found', False)
        )
        if clearly_applicable and clearly_missing:
            nv.update({'status': 'FAIL', 'required_note_found': False,
                       '_auto_downgraded': 'REVIEW_REQUIRED->FAIL(clearly_missing)'})
            status = 'FAIL'
            print(f"  [AUTO-DOWNGRADE] {nv.get('view_id')} REVIEW_REQUIRED->FAIL (clearly missing)")

    return nv


def combine_stage_results(rule, app_result, note_result, scale_x=1.0, scale_y=1.0):
    """Merge applicability + note-validation results with strict view/geometry gating.

    mod_condition_active is intentionally not used. A view proceeds only when:
    view_applicable=True and geometry_applicable=True and matched_rule_view is not None.
    """
    final_views = []
    app_views   = app_result.get('view_results', [])
    note_views  = (note_result or {}).get('view_results', [])
    app_logic   = rule.get('applicability_logic', {})

    # Exact view_id lookup only — no fallback by view type.
    note_by_id = {
        str(nv.get('view_id')): nv
        for nv in note_views
        if nv.get('view_id') is not None
    }

    def unscale(box, img_w=99999, img_h=99999):
        if not valid_box(box):
            return box
        return {
            'x1': box.get('x1', 0) / scale_x,
            'y1': box.get('y1', 0) / scale_y,
            'x2': box.get('x2', 0) / scale_x,
            'y2': box.get('y2', 0) / scale_y,
        }

    for idx, av in enumerate(app_views):
        app_status   = str(av.get('status', 'REVIEW_REQUIRED')).upper().strip()
        view_name    = get_view_name(av)
        matched_view = av.get('matched_rule_view') or get_matched_rule_view(view_name, rule)
        scope_raw    = av.get('scope_coordinates', {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0})
        scope        = unscale(scope_raw)
        vid          = str(av.get('view_id') or f'V{idx+1}')

        # Default false; never infer from status string.
        view_app = False
        geom_app = False

        if matched_view is None:
            app_status = 'OMITTED'
        else:
            view_app = True
            geom_app = _to_bool(av.get('geometry_applicable'), default=False)
            if not geom_app:
                app_status = 'OMITTED'
                geom_app = False

        base = {
            'view_id'             : vid,
            'view_type'           : view_name,
            'matched_rule_view'   : matched_view,
            'view_applicable'     : view_app,
            'geometry_applicable' : geom_app,
            'scope_coordinates'   : scope,
            'detected_entities'   : [],
        }

        if app_status == 'OMITTED':
            final_views.append({**base,
                'required_note_found': False,
                'status'             : 'OMITTED',
                'confidence_score'   : normalize_confidence(av.get('confidence_score', 0)),
                'reasoning'          : av.get('reasoning', 'Rule scope/geometry not applicable.')})
            continue

        if app_status == 'REVIEW_REQUIRED':
            final_views.append({**base,
                'required_note_found': False,
                'status'             : app_logic.get('if_condition_unclear', 'REVIEW_REQUIRED'),
                'confidence_score'   : normalize_confidence(av.get('confidence_score', 0)),
                'reasoning'          : av.get('reasoning', 'Applicability unclear.')})
            continue

        # Before note validation result can produce FAIL/PASS, re-check applicability flags.
        if not view_app or not geom_app:
            final_views.append({**base,
                'required_note_found': False,
                'status'             : 'OMITTED',
                'confidence_score'   : normalize_confidence(av.get('confidence_score', 0)),
                'reasoning'          : av.get('reasoning', 'Rule scope/geometry not applicable.')})
            continue

        nv = note_by_id.get(vid)
        if nv:
            nv = post_process_note_view(nv, rule, app_view={**av,
                'view_applicable': view_app,
                'geometry_applicable': geom_app,
                'matched_rule_view': matched_view,
            })
            st = normalize_status(nv.get('status', 'REVIEW_REQUIRED'), rule)

            if st == 'FAIL' and (not view_app or not geom_app):
                st = 'OMITTED'

            entities = nv.get('detected_entities', [])
            for ent in entities:
                box = ent.get('coordinates', {})
                if valid_box(box):
                    ent['coordinates'] = unscale(box)
                else:
                    ent['coordinates'] = scope
                    ent['_coords_fallback'] = 'scope'

            final_views.append({**base,
                'required_note_found': nv.get('required_note_found', False),
                'status'             : st,
                'confidence_score'   : normalize_confidence(nv.get('confidence_score', 0)),
                'reasoning'          : nv.get('reasoning', ''),
                'detected_entities'  : entities})
        else:
            final_views.append({**base,
                'required_note_found': False,
                'status'             : 'REVIEW_REQUIRED',
                'confidence_score'   : 0.0,
                'reasoning'          : 'Applicable view found but note validation result missing.'})

    statuses = [v['status'] for v in final_views]
    sheet    = ('FAIL' if 'FAIL' in statuses else
                'REVIEW_REQUIRED' if 'REVIEW_REQUIRED' in statuses else
                'PASS' if 'PASS' in statuses else 'OMITTED')
    return {'rule_id': rule.get('rule_id'), 'rule_name': rule.get('rule_name'),
            'sheet_status': sheet, 'view_results': final_views,
            'markup_required': sheet in ('FAIL', 'REVIEW_REQUIRED'),
            'markup_color': rule.get('markup_color', 'red'),
            'markup_label': rule.get('markup_label', 'CHECK'),
            'severity': rule.get('severity', 'Major'), 'priority': rule.get('priority', 'High')}


print('✅ post_process_note_view + combine_stage_results ready')

STATUS_COLORS = {'FAIL': (0, 0, 255), 'REVIEW_REQUIRED': (0, 140, 255), 'PASS': (0, 200, 0)}
DRAW_STATUSES = {'FAIL', 'REVIEW_REQUIRED'}

def draw_markup_box(img, x1, y1, x2, y2, label, status, thickness=3, fscale=0.55):
    color  = STATUS_COLORS.get(status, (0, 0, 255))
    H, W   = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W - 1, int(x2)), min(H - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return False
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    lines = [f'[{status}]', label[:55]]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    lh    = int(fscale * 30 + 8)
    ty0   = max(y1 - len(lines) * lh - 4, 22)
    for i, line in enumerate(lines):
        ty = ty0 + i * lh
        (tw, th), bl = cv2.getTextSize(line, font, fscale, 2)
        cv2.rectangle(img, (x1, ty - th - 4), (min(x1 + tw + 8, W - 1), ty + bl + 4), color, -1)
        cv2.putText(img, line, (x1 + 4, ty), font, fscale, (255, 255, 255), 2, cv2.LINE_AA)
    return True

def default_fail_box(img, rule_idx=0):
    H, W = img.shape[:2]
    col  = rule_idx % 2
    row  = rule_idx // 2
    x1   = int(W * 0.03) + col * int(W * 0.50)
    y1   = int(H * 0.05) + row * 60
    return x1, y1, x1 + int(W * 0.45), y1 + 50

def markup_results(original_image_path, results, output_path):
    img = cv2.imread(str(original_image_path))
    if img is None:
        raise FileNotFoundError(f'Image not found: {original_image_path}')
    H, W = img.shape[:2]
    print(f'  Markup on original: {W}x{H}')
    for rule_idx, rr in enumerate(results):
        if rr.get('sheet_status') not in DRAW_STATUSES:
            continue
        label = rr.get('markup_label', rr.get('rule_name', rr.get('rule_id', 'CHECK')))
        for view in rr.get('view_results', []):
            vstatus = view.get('status')
            if vstatus not in DRAW_STATUSES:
                continue
            drew = False
            for ent in view.get('detected_entities', []):
                box = ent.get('coordinates', {})
                if valid_box(box):
                    box = clamp_box(box, W, H)
                    if valid_box(box):
                        x1, y1, x2, y2 = get_box_ints(box)
                        drew = draw_markup_box(img, x1, y1, x2, y2, label, vstatus) or drew
            if not drew:
                scope = view.get('scope_coordinates', {})
                if valid_box(scope):
                    scope = clamp_box(scope, W, H)
                    if valid_box(scope):
                        x1, y1, x2, y2 = get_box_ints(scope)
                        drew = draw_markup_box(img, x1, y1, x2, y2, label, vstatus)
            if not drew:
                x1, y1, x2, y2 = default_fail_box(img, rule_idx)
                draw_markup_box(img, x1, y1, x2, y2, label, vstatus)
                print(f'  [FALLBACK BOX] {label[:40]} — no valid coords')
    cv2.imwrite(str(output_path), img)
    print(f'  ✅ Markup saved: {output_path}')
    return output_path

print('✅ markup_results ready')


def _run_applicability_stage(resized_path, rule, img_w, img_h, cached_views=None, use_cached_hints=False):
    """Stage 1 applicability. Uses cached hints only when global detection is reliable."""
    raw = ask_qwen(
        resized_path,
        build_applicability_prompt(rule, img_w, img_h, cached_views, use_cached_hints)
    )
    print(f'[APP] {raw[:200]}')
    result = extract_json(raw)

    pre_matched = {}
    if use_cached_hints and cached_views:
        for cv in cached_views:
            rv = get_matched_rule_view(get_view_name(cv), rule)
            if rv:
                pre_matched[str(cv.get('view_id'))] = (rv, get_view_name(cv))
        if pre_matched:
            print(f'[APP] Pre-matched views from global cache: {list(pre_matched.items())}')

    for v in result.get('view_results', []):
        vid = str(v.get('view_id', ''))
        rv = get_matched_rule_view(get_view_name(v), rule)
        if rv is None and vid in pre_matched:
            rv, detected_title = pre_matched[vid]
            v['detected_view_type'] = detected_title
            print(f'[APP] View {vid}: using cached title "{detected_title}" -> matched="{rv}"')
        v['matched_rule_view'] = rv

        geom_app = _to_bool(v.get('geometry_applicable'), default=False)
        vstatus = str(v.get('status', 'APPLICABLE')).upper().strip()

        if rv is None:
            # Existing view-not-matched behavior preserved.
            v['status'] = 'OMITTED'
            v['view_applicable'] = False
            v['geometry_applicable'] = False
        elif not geom_app:
            # View type matched, but rule geometry is inactive.
            v['status'] = 'OMITTED'
            v['view_applicable'] = True
            v['geometry_applicable'] = False
        else:
            v['status'] = 'APPLICABLE'
            v['view_applicable'] = True
            v['geometry_applicable'] = True

        # Remove deprecated field if model returns it.
        v.pop('mod_condition_active', None)
    return result


def _run_note_stage(resized_path, rule, applicable_views, img_w, img_h):
    raw = ask_qwen(resized_path, build_note_prompt(rule, applicable_views, img_w, img_h))
    print(f'[NOTE] {raw[:200]}')
    try:
        return extract_json(raw)
    except Exception as e:
        print(f'[NOTE ERROR] {e} -> binary fallback...')
        fb_views = []
        for av in applicable_views:
            raw_fb = ask_qwen(resized_path,
                              build_binary_fallback_prompt(rule, av, img_w, img_h),
                              max_new_tokens=300)
            print(f'[FALLBACK] {raw_fb[:120]}')
            try:
                fb    = extract_json(raw_fb)
                found = bool(fb.get('found', False))
                sc    = av.get('scope_coordinates', {'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h})
                coords = {'x1': fb.get('x1', sc['x1']), 'y1': fb.get('y1', sc['y1']),
                          'x2': fb.get('x2', sc['x2']), 'y2': fb.get('y2', sc['y2'])}
                if not valid_box(coords):
                    coords = sc
                fb_views.append({
                    'view_id'           : av.get('view_id'),
                    'detected_view_type': get_view_name(av),
                    'matched_rule_view' : av.get('matched_rule_view'),
                    'status'            : 'PASS' if found else 'FAIL',
                    'confidence_score'  : float(fb.get('confidence', 50.0)),
                    'required_note_found': found,
                    'reasoning'         : f'Binary fallback: found={found} text={fb.get("matched_text")}',
                    'detected_entities' : [{
                        'entity_name'  : (rule.get('required_entities') or ['Note'])[0],
                        'detected_text': str(fb.get('matched_text') or 'NOT FOUND'),
                        'coordinates': coords
                    }]
                })
                print(f'[FALLBACK] {av.get("view_id")} found={found}')
            except Exception as fe:
                print(f'[FALLBACK ERROR] {fe}')
                sc = av.get('scope_coordinates', {'x1': 0, 'y1': 0, 'x2': img_w, 'y2': img_h})
                fb_views.append({
                    'view_id'           : av.get('view_id'),
                    'detected_view_type': get_view_name(av),
                    'matched_rule_view' : av.get('matched_rule_view'),
                    'status'            : 'REVIEW_REQUIRED',
                    'confidence_score'  : 0.0,
                    'required_note_found': False,
                    'reasoning'         : f'Both main and binary fallback failed: {fe}',
                    'detected_entities' : []
                })
        if fb_views:
            return {'rule_id': rule.get('rule_id'), 'stage': 'note_validation', 'view_results': fb_views}
        return None

# ===== CELL: Updated Pipeline Run Execution =====

def run_pipeline_for_drawing(image_path, rules, output_dir):
    stem = Path(str(image_path)).stem
    resized_path, scale_x, scale_y, orig_w, orig_h = resize_image(image_path, output_dir)
    res_img = Image.open(resized_path)
    img_w, img_h = res_img.size
    res_img.close()
    print(f'  Resized for model: {img_w}x{img_h}  scale=({scale_x:.3f},{scale_y:.3f})')

    output_json   = os.path.join(output_dir, f'{stem}_results.json')
    output_markup = os.path.join(output_dir, f'{stem}_marked.png')
    all_results   = []

    print('\n[GLOBAL VIEW DETECTION] Analyzing drawing to detect all views...')
    use_cached_hints = False
    cached_views = None
    try:
        cached_views = run_global_view_detection(resized_path, img_w, img_h)
        use_cached_hints = is_cached_views_reliable(cached_views, img_w, img_h)
        print(f'[GLOBAL VIEW DETECTION] Found {len(cached_views)} view(s): '
              f'{[v.get("view_id") for v in cached_views]} | reliable={use_cached_hints}')
        if not use_cached_hints:
            print('[GLOBAL VIEW DETECTION] Unreliable cache (Unknown/tiny/low confidence) '
                  '-> per-rule full-sheet applicability fallback')
    except Exception as e:
        print(f'[GLOBAL VIEW DETECTION ERROR] {e} -> per-rule full-sheet applicability fallback')

    for rule in rules:
        rule_id = rule.get('rule_id')
        print(f'\n{"="*60}\n[{stem[:40]}] {rule_id}')

        try:
            app_result = _run_applicability_stage(
                resized_path, rule, img_w, img_h,
                cached_views=cached_views,
                use_cached_hints=use_cached_hints,
            )
        except Exception as e:
            print(f'[APP ERROR] {e}')
            all_results.append({'rule_id': rule_id, 'rule_name': rule.get('rule_name'),
                'sheet_status': 'REVIEW_REQUIRED', 'view_results': [],
                'markup_required': False, 'markup_color': rule.get('markup_color', 'red'),
                'markup_label': rule.get('markup_label', 'CHECK'),
                'severity': rule.get('severity', 'Major'), 'priority': rule.get('priority', 'High')})
            gc.collect(); torch.cuda.empty_cache(); continue

        applicable_views = [
            v for v in app_result.get('view_results', [])
            if v.get('view_applicable') is True
            and v.get('geometry_applicable') is True
            and v.get('matched_rule_view') is not None
        ]
        dtypes = [v.get('detected_view_type') for v in app_result.get('view_results', [])]
        print(f'[APP] Detected: {dtypes} | Applicable: {[v.get("view_id") for v in applicable_views]}')

        if not applicable_views:
            final = combine_stage_results(rule, app_result, None, scale_x, scale_y)
            print(f'[RESULT] {rule_id} -> {final["sheet_status"]} (no applicable views)')
            all_results.append(final)
            gc.collect(); torch.cuda.empty_cache(); continue

        note_result = _run_note_stage(resized_path, rule, applicable_views, img_w, img_h)
        final = combine_stage_results(rule, app_result, note_result, scale_x, scale_y)
        print(f'[RESULT] {rule_id} -> {final["sheet_status"]}')
        all_results.append(final)
        gc.collect(); torch.cuda.empty_cache()

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n✅ Results JSON: {output_json}')
    markup_results(image_path, all_results, output_markup)
    return all_results, output_json, output_markup

rules = load_rules(RULE_JSON_PATH)
print(f'✅ {len(rules)} rules loaded')

drawing_paths = get_drawing_paths(DRAWING_DIR)
print(f'✅ {len(drawing_paths)} drawing(s) found in {DRAWING_DIR}')

if not drawing_paths:
    print(f'⚠️  No drawings found. Add CAD images to {DRAWING_DIR}')
else:
    all_drawing_results = {}
    for dp in drawing_paths:
        print(f'\n{"#"*70}\n# {Path(dp).name}\n{"#"*70}')
        res, jp, mp = run_pipeline_for_drawing(dp, rules, OUTPUT_DIR)
        all_drawing_results[dp] = {'results': res, 'json_path': jp, 'markup_path': mp}
    print(f'\n{"="*70}\n✅ Done — {len(drawing_paths)} drawing(s). Outputs: {OUTPUT_DIR}')
 