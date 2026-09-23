"""
dino_lab.py — Grounding DINO parameter research harness for architectural sheets
================================================================================

A CLI sweeper for the DETECTION half of your drawing-extraction pipeline
(no UI — backend only). Point it at a PDF, pick pages, and sweep:

    mode        : single-pass (whole sheet, no tiling)  vs  tiled
    render DPI  : 100 / 150 / 200 / 300 ...
    proc size   : the processor's internal resize (THE variable that actually
                  controls "high resolution" — see NOTE 1 below)
    box_thr     : box (logit) threshold
    text_thr    : per-token phrase threshold (see NOTE 2)
    tile size   : 1400 / 2000 / 2800 ...
    overlap     : 0.10 / 0.20 / 0.30
    nms_iou     : cross-tile duplicate merge threshold
    prompt      : the text query itself

and scores every run automatically against vector-text SCALE anchors pulled
from the PDF itself, so you get recall / spurious / merged numbers instead of
eyeballing overlays.

--------------------------------------------------------------------------------
NOTE 1 — why "high resolution single sheet" usually does nothing
--------------------------------------------------------------------------------
GroundingDinoImageProcessor resizes every input to shortest_edge=800,
longest_edge=1333 by default. Render a 34x22" sheet at 600 DPI, hand it to the
model whole, and the processor immediately squashes it back to ~1333px wide.
A 1/4"=1'-0" elevation ends up ~40px tall either way. That is the real reason
tiling helped you: tiling is a resolution hack, not a detection hack — each
2800px tile arrives at the backbone at a far higher effective px-per-inch.

So "single sheet at high res" is only a real experiment if you also raise the
processor size. This harness exposes `proc_short` / `proc_long` and reports the
ACTUAL tensor shape fed to the backbone (`input_px`) plus px-per-PDF-point, so
you can compare apples to apples:

    tiled  2800px tile @150dpi  -> effective ~1333/2800 * 2800 ... see table
    single 1536/2560 proc size  -> effective backbone res on the full sheet

Swin + DINO's sine positional encodings are resolution-agnostic, so large
proc sizes work — but attention cost is ~O(n^2) in tokens. 2560 long edge is
roughly 3.7x the FLOPs of 1333. Expect VRAM blowups past ~3000; the harness
catches OOM and records it as a failed run rather than dying.

--------------------------------------------------------------------------------
NOTE 2 — text_threshold is dead config in your current pipeline
--------------------------------------------------------------------------------
Your v2.1 script calls:

    processor.image_processor.post_process_object_detection(...)

That path takes only `threshold` (box) and ignores text entirely; TEXT_THRESHOLD
in your CONFIG block has never affected a single run. The phrase-grounding path
is:

    processor.post_process_grounded_object_detection(
        outputs, input_ids, threshold=..., text_threshold=..., target_sizes=...)

which additionally decides WHICH prompt phrase each box is labelled with (it
keeps tokens whose logit > text_threshold). This harness uses the grounded path
and falls back to the image_processor path only if your transformers version is
too old — and it tells you which one it used, per run.

Consequence for your sweep: box_threshold moves recall/precision; text_threshold
mostly moves LABELS, which matters only if you plan to filter by label
("floor plan" vs "detail drawing"). Sweep box_thr first.

--------------------------------------------------------------------------------
NOTE 3 — scoring without hand-labels
--------------------------------------------------------------------------------
Anchors = vector-text spans that pass a real-scale regex (same `_is_real_scale`
logic as your v2.1 fix), deduped by position. Each anchor marks "a drawing
exists here, and its body sits ABOVE this point". Per run we compute:

    anchors        : number of scale anchors on the page
    hit            : anchors claimed by >=1 cluster
    recall         : hit / anchors
    spurious       : clusters claiming 0 anchors
    merged         : clusters claiming >=2 anchors  (under-segmentation)
    clean          : clusters claiming exactly 1    (what you want)

This is a proxy, not ground truth — an anchor can be "hit" by a box that is
badly cropped. Use recall/spurious/merged to shortlist 3-4 configs, then look
at the overlays for those only.

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    pip install pymupdf transformers torch pillow pandas

    python dino_lab.py sheet.pdf --pages 1,5 --sweep-box 0.05,0.08,0.12 \\
        --modes single,tiled --dpi 150,300 --proc 800x1333,1536x2560

Outputs land in ./dino_lab_runs/<timestamp>/ :
    <tag>.jpg             Grounding DINO output image (boxes scored vs anchors),
                          with box_thr / text_thr / anchor count in its caption
    results.csv / .json   one row per run (every threshold + anchor count + metrics)
    run_labels.csv        human verdict per run: correct / wrong (from app.py)
    research_dataset.csv  results.csv + run_labels.csv joined on `tag`

--------------------------------------------------------------------------------
NOTE 4 — Grounding DINO output only
--------------------------------------------------------------------------------
Earlier versions also ran a second "drawing box" stage (title blocks -> territories
-> red final bboxes) and saved a <tag>_bbox.jpg per run. That whole stage has been
removed: the ONLY image a run produces is the Grounding DINO overlay.

--------------------------------------------------------------------------------
NOTE 5 — human labeling (app.py "Label" tab)
--------------------------------------------------------------------------------
Every completed run's DINO image is listed on one page. Mark each image
correct / wrong; the thresholds and anchor count come from that run's results
row automatically, so nothing is typed by hand. Verdicts are written to
run_labels.csv (one row per run, latest verdict wins) and build_research_dataset()
joins them onto results.csv.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Sequence

import fitz  # PyMuPDF
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ======================================================================
# DEFAULTS  (mirrors your v2.1 CONFIG so results transfer 1:1)
# ======================================================================
MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_PROMPT = (
    "floor plan . elevation drawing . 3d isometric view . structural drawing . "
    "architectural section . detail drawing . building structure ."
)
D = dict(
    box_thr=0.08,
    text_thr=0.10,
    dpi=150,
    tile=2800,
    overlap=0.20,
    nms_iou=0.45,
    min_w=200,
    min_h=200,
    max_aspect=8.0,
    min_area_frac=0.005,
    max_area_frac=0.40,
)
OUT_ROOT = "dino_lab_runs"

_MODEL: dict[str, Any] = {}


# ======================================================================
# MODEL
# ======================================================================
def get_model(model_id: str = MODEL_ID):
    if _MODEL.get("id") == model_id:
        return _MODEL["model"], _MODEL["processor"], _MODEL["device"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {model_id} -> {device}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    _MODEL.clear()
    _MODEL.update(id=model_id, model=model, processor=processor, device=device)
    return model, processor, device


def vram_str() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    free, total = torch.cuda.mem_get_info()
    return f"{(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB"


# ======================================================================
# GEOMETRY / FILTERS  (lifted from your v2.1 so behaviour matches)
# ======================================================================
def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def tiled_nms(dets, iou_thresh):
    """Union-merge true cross-tile duplicates, then greedy NMS. Same as v2.1."""
    if not dets:
        return []
    merged = list(dets)
    changed = True
    while changed:
        changed = False
        out, used = [], [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            box, score, label = list(a["box"]), a["score"], a["label"]
            for j, b in enumerate(merged):
                if i == j or used[j]:
                    continue
                if box_iou(box, b["box"]) >= iou_thresh:
                    box[0] = min(box[0], b["box"][0])
                    box[1] = min(box[1], b["box"][1])
                    box[2] = max(box[2], b["box"][2])
                    box[3] = max(box[3], b["box"][3])
                    score = max(score, b["score"])
                    used[j] = True
                    changed = True
            out.append({"label": label, "score": score, "box": box})
        merged = out
    dets2 = sorted(merged, key=lambda d: d["score"], reverse=True)
    kept = []
    while dets2:
        best = dets2.pop(0)
        kept.append(best)
        dets2 = [d for d in dets2 if box_iou(best["box"], d["box"]) < iou_thresh]
    return kept


def quality_filter(dets, img_w, img_h, cfg, log: list[str]):
    page_area = float(img_w * img_h)
    keep = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        if w < cfg["min_w"] or h < cfg["min_h"]:
            log.append(f"drop small {w:.0f}x{h:.0f}")
            continue
        aspect = max(w / h, h / w)
        frac = (w * h) / page_area
        if aspect > cfg["max_aspect"]:
            log.append(f"drop aspect {aspect:.1f}")
            continue
        if frac < cfg["min_area_frac"]:
            log.append(f"drop tiny {frac*100:.2f}%")
            continue
        if frac > cfg["max_area_frac"]:
            log.append(f"drop whole-sheet {frac*100:.1f}%")
            continue
        keep.append(d)
    return keep


def generate_tiles(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles, y = [], 0
    while y < height:
        x = 0
        while x < width:
            x1, y1 = x, y
            x2, y2 = min(x1 + tile_size, width), min(y1 + tile_size, height)
            if x2 - x1 < tile_size and width >= tile_size:
                x1, x2 = width - tile_size, width
            if y2 - y1 < tile_size and height >= tile_size:
                y1, y2 = height - tile_size, height
            t = (x1, y1, x2, y2)
            if t not in tiles:
                tiles.append(t)
            if x2 >= width:
                break
            x += stride
        if y2 >= height:
            break
        y += stride
    return tiles


# ======================================================================
# VECTOR-TEXT ANCHORS  (auto ground-truth proxy — see NOTE 3)
# ======================================================================
_SCALE_VALUE_RE = re.compile(
    r'(\b\d+(\s+\d+/\d+|\s*/\s*\d+)?\s*["\']?\s*=\s*\d+[\'"]?|\b1\s*:\s*\d+'
    r'|\b(nts|n\.t\.s\.?|no\s*scale|as\s*noted)\b|\bscale\b)',
    re.IGNORECASE,
)


def _is_real_scale(text: str) -> bool:
    t = text.strip()
    if not t or len(t) >= 50:
        return False
    if re.match(r'^\d+[\'"]?\s*-\s*\d+', t):  # plain dimension string
        return False
    return bool(_SCALE_VALUE_RE.search(t))


def find_scale_anchors(page) -> list[dict]:
    """Scale/ARCH REF spans -> one anchor per drawing, in PDF points."""
    rot = page.rotation_matrix if page.rotation != 0 else None
    spans = []
    for b in page.get_text("dict").get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                t = s["text"].strip()
                if not t:
                    continue
                r = fitz.Rect(s["bbox"])
                if rot:
                    r = r * rot
                spans.append({"text": t, "bbox": [r.x0, r.y0, r.x1, r.y1]})

    raw = []
    for s in spans:
        if _is_real_scale(s["text"]):
            raw.append(s)
        elif re.match(r'^arch\s*ref\b', s["text"], re.IGNORECASE):
            x0, y0 = s["bbox"][0], s["bbox"][1]
            if not any(
                _is_real_scale(o["text"])
                and abs(o["bbox"][0] - x0) < 50
                and 0 < (y0 - o["bbox"][3]) < 25
                for o in spans if o is not s
            ):
                raw.append(s)

    # dedupe: collapse spans within 40pt x / 14pt y of each other
    anchors = []
    for s in sorted(raw, key=lambda s: (s["bbox"][1], s["bbox"][0])):
        cx = (s["bbox"][0] + s["bbox"][2]) / 2.0
        cy = (s["bbox"][1] + s["bbox"][3]) / 2.0
        if any(abs(cx - a["cx"]) < 40 and abs(cy - a["cy"]) < 14 for a in anchors):
            continue
        anchors.append({"cx": cx, "cy": cy, "text": s["text"], "bbox": s["bbox"]})
    return anchors


def score_against_anchors(clusters_pdf, anchors, up_tol=40.0, side_tol=30.0):
    """A cluster claims an anchor if the anchor sits inside its x-range and at/below
    its body (drawings sit above their title block)."""
    claims = []
    for c in clusters_pdf:
        x0, y0, x1, y1 = c
        got = [
            i for i, a in enumerate(anchors)
            if (x0 - side_tol) <= a["cx"] <= (x1 + side_tol) and y0 <= a["cy"] <= (y1 + up_tol)
        ]
        claims.append(got)
    hit = {i for g in claims for i in g}
    return dict(
        anchors=len(anchors),
        hit=len(hit),
        recall=round(len(hit) / len(anchors), 3) if anchors else None,
        clusters=len(clusters_pdf),
        clean=sum(1 for g in claims if len(g) == 1),
        merged=sum(1 for g in claims if len(g) > 1),
        spurious=sum(1 for g in claims if len(g) == 0),
    ), claims


# ======================================================================
# PER-PAGE ANCHOR CACHE
# ======================================================================
_anchor_cache: dict[tuple, list] = {}


def _page_key(page) -> tuple:
    """Cache key that survives re-opening the same file (doc.name = path)."""
    parent = getattr(page, "parent", None)
    return (getattr(parent, "name", "") or "", page.number)


def get_page_anchors(page) -> list:
    """Scale anchors for a page (independent of sweep settings -> cached)."""
    key = _page_key(page)
    if key not in _anchor_cache:
        if len(_anchor_cache) > 16:
            _anchor_cache.clear()
        _anchor_cache[key] = find_scale_anchors(page)
    return _anchor_cache[key]


# ======================================================================
# DETECTION
# ======================================================================
def _dino_forward(image, prompt, model, processor, device, box_thr, text_thr,
                  proc_short=None, proc_long=None):
    """One forward pass. Returns (detections_in_image_px, meta)."""
    kwargs = {}
    if proc_short and proc_long:
        kwargs["size"] = {"shortest_edge": int(proc_short), "longest_edge": int(proc_long)}
    inputs = processor(images=image, text=prompt, return_tensors="pt", **kwargs)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    in_h, in_w = inputs["pixel_values"].shape[-2:]

    ctx = torch.autocast("cuda") if device == "cuda" else _null()
    with torch.inference_mode(), ctx:
        outputs = model(**inputs)

    path = "grounded"
    try:
        res = processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"],
            threshold=box_thr, text_threshold=text_thr,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        try:
            res = processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"],
                box_threshold=box_thr, text_threshold=text_thr,
                target_sizes=[image.size[::-1]],
            )[0]
        except Exception:
            path = "image_processor(text_thr IGNORED)"
            res = processor.image_processor.post_process_object_detection(
                outputs, threshold=box_thr, target_sizes=[image.size[::-1]]
            )[0]

    labels = res.get("labels", res.get("text_labels", []))
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    dets = []
    for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
        dets.append({"box": [float(v) for v in box], "score": float(score), "label": str(lab)})

    del inputs, outputs
    if device == "cuda":
        torch.cuda.empty_cache()
    return dets, {"input_px": f"{in_w}x{in_h}", "postproc": path}


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def detect_single(image, prompt, cfg, model, processor, device):
    dets, meta = _dino_forward(
        image, prompt, model, processor, device,
        cfg["box_thr"], cfg["text_thr"], cfg.get("proc_short"), cfg.get("proc_long"),
    )
    meta["tiles"] = 1
    meta["raw"] = len(dets)
    return dets, meta


def detect_tiled(image, prompt, cfg, model, processor, device):
    W, H = image.size
    tiles = generate_tiles(W, H, int(cfg["tile"]), float(cfg["overlap"]))
    raw, meta = [], {}
    for (tx1, ty1, tx2, ty2) in tiles:
        crop = image.crop((tx1, ty1, tx2, ty2))
        dets, meta = _dino_forward(
            crop, prompt, model, processor, device,
            cfg["box_thr"], cfg["text_thr"], cfg.get("proc_short"), cfg.get("proc_long"),
        )
        for d in dets:
            b = d["box"]
            raw.append({
                "label": d["label"], "score": d["score"],
                "box": [b[0] + tx1, b[1] + ty1, b[2] + tx1, b[3] + ty1],
            })
    out = tiled_nms(raw, float(cfg["nms_iou"]))
    meta = dict(meta)
    meta.update(tiles=len(tiles), raw=len(raw), after_nms=len(out))
    return out, meta


# ======================================================================
# RENDER / OVERLAY
# ======================================================================
_render_cache: dict[tuple, Image.Image] = {}


def clear_caches() -> None:
    """Free page renders and cached per-page anchors."""
    _render_cache.clear()
    _anchor_cache.clear()


def render_page(page, dpi: int) -> Image.Image:
    key = _page_key(page) + (dpi,)
    if key in _render_cache:
        return _render_cache[key]
    sf = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(sf, sf))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if len(_render_cache) > 8:
        _render_cache.clear()
    _render_cache[key] = img
    return img


def _get_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_overlay(image, dets, anchors, sf, claims=None, caption=""):
    """DINO overlay scored against anchors. sf = pixels per PDF point."""
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    font = _get_font(max(14, img.width // 90))

    for a in anchors:  # magenta X at every scale anchor
        x, y = a["cx"] * sf, a["cy"] * sf
        r = max(8, img.width // 260)
        d.line([x - r, y - r, x + r, y + r], fill="magenta", width=4)
        d.line([x - r, y + r, x + r, y - r], fill="magenta", width=4)

    for i, det in enumerate(dets):
        n = len(claims[i]) if claims else 1
        color = {0: "red", 1: "#00C000"}.get(n, "orange")  # 0=spurious 1=clean 2+=merged
        b = [int(v) for v in det["box"]]
        d.rectangle(b, outline=color, width=max(4, img.width // 400))
        d.text((b[0] + 6, max(0, b[1] - 26)),
               f'{det["score"]:.2f} {det.get("label","")[:22]}', fill=color, font=font)

    if caption:
        bar = font.size + 20 if hasattr(font, "size") else 40
        d.rectangle([0, 0, img.width, bar], fill="black")
        d.text((10, 10), caption, fill="white", font=font)
    return img


# ======================================================================
# RUN LABELS  (one human verdict per DINO output image)
# ======================================================================
LABEL_VALUES = ("correct", "wrong")

# Everything below is copied from the run's own results row — the human only
# supplies the last column.
RUN_LABEL_COLUMNS = [
    "timestamp", "tag", "page", "mode", "dpi", "proc", "input_px",
    "box_thr", "text_thr", "tile", "overlap", "nms_iou",
    "anchors", "kept", "hit", "recall", "clean", "merged", "spurious",
    "label",
]


def save_run_labels(out_dir, rows):
    """Write run verdicts to <out_dir>/run_labels.csv.

    `rows` = results-rows (dicts from run_one) each carrying a "label" key.
    Existing labels are kept; a re-labeled run REPLACES its earlier row
    (keyed on `tag`), so clicking correct -> wrong never leaves duplicates.
    Rows whose label is empty/None remove that run's verdict.
    """
    path = os.path.join(out_dir, "run_labels.csv")
    ts = datetime.now().isoformat(timespec="seconds")
    new = []
    for r in rows:
        rec = {c: r.get(c, "") for c in RUN_LABEL_COLUMNS}
        rec["timestamp"] = ts
        rec["label"] = r.get("label") or ""
        new.append(rec)
    df_new = pd.DataFrame(new, columns=RUN_LABEL_COLUMNS)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df_new], ignore_index=True)
        df = df.drop_duplicates("tag", keep="last")
    else:
        df = df_new
    df = df[df["label"].astype(str).str.strip() != ""]
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(path, index=False)
    return path, len(df)


def build_research_dataset(out_dir):
    """results.csv (config + automatic metrics) joined with run_labels.csv
    (human verdict) on `tag` -> research_dataset.csv, one row per run.
    Adds `human_label` ("correct" / "wrong" / "unlabeled") and `human_correct`
    (1 / 0 / blank) columns."""
    res_path = os.path.join(out_dir, "results.csv")
    lab_path = os.path.join(out_dir, "run_labels.csv")
    if not os.path.exists(res_path):
        raise FileNotFoundError("results.csv not found in this run directory — wait for the sweep to finish.")
    if not os.path.exists(lab_path):
        raise FileNotFoundError("No labels saved yet — mark at least one run correct/wrong first.")
    res = pd.read_csv(res_path)
    lab = pd.read_csv(lab_path).sort_values("timestamp").drop_duplicates("tag", keep="last")
    res = res.merge(lab[["tag", "label"]].rename(columns={"label": "human_label"}), on="tag", how="left")
    res["human_label"] = res["human_label"].fillna("unlabeled")
    res["human_correct"] = res["human_label"].map({"correct": 1, "wrong": 0})
    out_path = os.path.join(out_dir, "research_dataset.csv")
    res.to_csv(out_path, index=False)
    return res, out_path


def _cap_and_save(img: Image.Image, path: str, max_w: int = 2600) -> Image.Image:
    """Cap overlay size so the gallery stays responsive, then save."""
    if img.width > max_w:
        k = max_w / img.width
        img = img.resize((max_w, int(img.height * k)))
    img.save(path, quality=85)
    return img


# ======================================================================
# RUNNER
# ======================================================================
@dataclass
class Cfg:
    mode: str = "tiled"
    dpi: int = D["dpi"]
    box_thr: float = D["box_thr"]
    text_thr: float = D["text_thr"]
    tile: int = D["tile"]
    overlap: float = D["overlap"]
    nms_iou: float = D["nms_iou"]
    proc_short: int | None = None
    proc_long: int | None = None
    min_w: int = D["min_w"]
    min_h: int = D["min_h"]
    max_aspect: float = D["max_aspect"]
    min_area_frac: float = D["min_area_frac"]
    max_area_frac: float = D["max_area_frac"]


def run_tag(page_no, cfg: dict) -> str:
    return (f"p{page_no}_{cfg['mode']}_dpi{cfg['dpi']}_box{cfg['box_thr']}"
            f"_txt{cfg['text_thr']}"
            + (f"_tile{cfg['tile']}_ov{cfg['overlap']}_nms{cfg['nms_iou']}" if cfg["mode"] == "tiled" else "")
            + (f"_proc{cfg['proc_short']}x{cfg['proc_long']}" if cfg.get("proc_short") else "_procDEFAULT"))


def run_one(page, page_no, cfg: dict, prompt: str, out_dir: str, save_overlay=True):
    """One (page, config) run.

    Returns (row, dino_overlay, label_ctx):
        row          dict of config + metrics (thresholds, anchor count, ...)
        dino_overlay PIL image: the Grounding DINO output, boxes scored vs
                     anchors, with thresholds + anchor count in the caption
                     (None if save_overlay=False)
        label_ctx    {"img": dino_overlay, "row": row} — what the Label tab
                     needs to show this run and save a verdict for it.
    """
    model, processor, device = get_model(cfg.get("model_id", MODEL_ID))
    img = render_page(page, int(cfg["dpi"]))
    sf = int(cfg["dpi"]) / 72.0
    anchors = get_page_anchors(page)

    t0 = time.time()
    err = None
    try:
        if cfg["mode"] == "single":
            dets, meta = detect_single(img, prompt, cfg, model, processor, device)
        else:
            dets, meta = detect_tiled(img, prompt, cfg, model, processor, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        dets, meta, err = [], {"input_px": "-", "postproc": "-", "tiles": 0, "raw": 0}, "CUDA_OOM"
    except Exception as e:
        dets, meta, err = [], {"input_px": "-", "postproc": "-", "tiles": 0, "raw": 0}, f"{type(e).__name__}: {e}"
        traceback.print_exc()
    dt = time.time() - t0

    log: list[str] = []
    dets = quality_filter(dets, img.width, img.height, cfg, log)
    clusters_pdf = [[b / sf for b in d["box"]] for d in dets]
    metrics, claims = score_against_anchors(clusters_pdf, anchors)

    tag = run_tag(page_no, cfg)

    overlay_path = None
    over_img = None
    if save_overlay:
        os.makedirs(out_dir, exist_ok=True)
        cap = (f"{tag}  |  box_thr={cfg['box_thr']}  text_thr={cfg['text_thr']}  "
               f"anchors={metrics['anchors']}  boxes={len(dets)}  "
               f"R={metrics['recall']}  sp={metrics['spurious']}  mg={metrics['merged']}")
        over_img = draw_overlay(img, dets, anchors, sf, claims, cap)
        overlay_path = os.path.join(out_dir, tag + ".jpg")
        over_img = _cap_and_save(over_img, overlay_path)

    row = {
        "page": page_no, "mode": cfg["mode"], "dpi": cfg["dpi"],
        "proc": f"{cfg.get('proc_short') or 'def'}x{cfg.get('proc_long') or 'def'}",
        "input_px": meta.get("input_px"),
        "box_thr": cfg["box_thr"], "text_thr": cfg["text_thr"],
        "tile": cfg["tile"] if cfg["mode"] == "tiled" else "",
        "overlap": cfg["overlap"] if cfg["mode"] == "tiled" else "",
        "nms_iou": cfg["nms_iou"] if cfg["mode"] == "tiled" else "",
        "tiles": meta.get("tiles"), "raw": meta.get("raw"),
        "kept": len(dets),
        **metrics,
        "mean_score": round(sum(d["score"] for d in dets) / len(dets), 3) if dets else None,
        "sec": round(dt, 1), "postproc": meta.get("postproc"), "err": err,
        "overlay": overlay_path, "tag": tag,
    }
    label_ctx = {"img": over_img, "row": row}
    gc.collect()
    return row, over_img, label_ctx


def build_grid(modes, dpis, boxes, texts, tiles, overlaps, nmss, procs, base: dict):
    combos = []
    for mode, dpi, bt, tt, proc in itertools.product(modes, dpis, boxes, texts, procs):
        if mode == "single":
            c = dict(base, mode="single", dpi=dpi, box_thr=bt, text_thr=tt,
                     proc_short=proc[0], proc_long=proc[1])
            combos.append(c)
        else:
            for tl, ov, nm in itertools.product(tiles, overlaps, nmss):
                combos.append(dict(base, mode="tiled", dpi=dpi, box_thr=bt, text_thr=tt,
                                   tile=tl, overlap=ov, nms_iou=nm,
                                   proc_short=proc[0], proc_long=proc[1]))
    # dedupe
    seen, out = set(), []
    for c in combos:
        k = json.dumps(c, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# ======================================================================
# PARSERS
# ======================================================================
def plist(s, cast=float, default=None):
    if s is None or str(s).strip() == "":
        return list(default or [])
    return [cast(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_procs(s):
    """'800x1333,1536x2560,default' -> [(800,1333),(1536,2560),(None,None)]"""
    if not str(s).strip():
        return [(None, None)]
    out = []
    for chunk in str(s).split(","):
        c = chunk.strip().lower()
        if not c:
            continue
        if c in ("default", "def", "none"):
            out.append((None, None))
        else:
            a, b = c.split("x")
            out.append((int(a), int(b)))
    return out


def parse_pages(s, n):
    s = str(s).strip().lower()
    if s in ("", "all", "*"):
        return list(range(1, n + 1))
    pages = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            pages += list(range(int(a), int(b) + 1))
        elif chunk.isdigit():
            pages.append(int(chunk))
    return [p for p in pages if 1 <= p <= n]


# ======================================================================
# CORE SWEEP
# ======================================================================
def sweep(pdf_path, pages_str, prompt, modes, dpis, boxes, texts, tiles, overlaps,
          nmss, procs, filt, out_dir=None, progress=None):
    doc = fitz.open(pdf_path)
    pages = parse_pages(pages_str, len(doc))
    if not pages:
        raise ValueError("No valid pages selected.")

    base = dict(min_w=filt["min_w"], min_h=filt["min_h"], max_aspect=filt["max_aspect"],
                min_area_frac=filt["min_area_frac"], max_area_frac=filt["max_area_frac"],
                tile=D["tile"], overlap=D["overlap"], nms_iou=D["nms_iou"],
                proc_short=None, proc_long=None)
    grid = build_grid(modes, dpis, boxes, texts, tiles, overlaps, nmss, procs, base)

    out_dir = out_dir or os.path.join(OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    rows, imgs = [], []
    total = len(grid) * len(pages)
    i = 0
    for pno in pages:
        page = doc[pno - 1]
        for cfg in grid:
            i += 1
            if progress:
                progress(i / total, desc=f"[{i}/{total}] p{pno} {cfg['mode']} dpi{cfg['dpi']} box{cfg['box_thr']}")
            print(f"[{i}/{total}] p{pno} {cfg['mode']} dpi={cfg['dpi']} box={cfg['box_thr']} "
                  f"tile={cfg['tile']} proc={cfg['proc_short']} vram={vram_str()}")
            row, im, _label_ctx = run_one(page, pno, cfg, prompt, out_dir)
            rows.append(row)
            if im is not None:
                imgs.append((im, row["tag"]))
    doc.close()
    clear_caches()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "results.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return df, imgs, out_dir


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """Shortlist: maximise recall, then clean boxes, then penalise spurious+merged."""
    if df.empty:
        return df
    d = df.copy()
    d["recall"] = d["recall"].fillna(0)
    d["score_"] = d["recall"] * 2.0 + d["clean"] * 0.05 - d["spurious"] * 0.08 - d["merged"] * 0.15
    cols = ["score_", "mode", "dpi", "proc", "input_px", "box_thr", "text_thr", "tile",
            "overlap", "nms_iou", "tiles", "raw", "kept", "anchors", "hit", "recall",
            "clean", "merged", "spurious", "mean_score", "sec",
            "err", "page"]
    cols = [c for c in cols if c in d.columns]
    return d.sort_values("score_", ascending=False)[cols].round(3)


# ======================================================================
# CLI
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Grounding DINO research harness (backend / CLI only)")
    ap.add_argument("pdf", metavar="PDF", help="path to the drawing set PDF")
    ap.add_argument("--pages", default="1", help="e.g. 1,5,8 or 1-4 or 'all'")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--modes", default="tiled", help="comma-separated: tiled,single")
    ap.add_argument("--dpi", default="150")
    ap.add_argument("--proc", default="default", help="e.g. default,1536x2560")
    ap.add_argument("--sweep-box", default="0.08")
    ap.add_argument("--sweep-text", default="0.10")
    ap.add_argument("--tiles", default="2800")
    ap.add_argument("--overlaps", default="0.20")
    ap.add_argument("--nms", default="0.45")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    filt = dict(min_w=D["min_w"], min_h=D["min_h"], max_aspect=D["max_aspect"],
                min_area_frac=D["min_area_frac"], max_area_frac=D["max_area_frac"])
    df, _, out = sweep(a.pdf, a.pages, a.prompt, [m.strip() for m in a.modes.split(",")],
                       plist(a.dpi, int), plist(a.sweep_box, float), plist(a.sweep_text, float),
                       plist(a.tiles, int), plist(a.overlaps, float), plist(a.nms, float),
                       parse_procs(a.proc), filt, out_dir=a.out)
    print("\n" + rank(df).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()