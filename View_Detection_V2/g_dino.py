"""
dino_lab_adaptive.py

Grounding DINO architectural drawing detector with empirical adaptive
confidence thresholding.

SHEET-ADAPTIVE PROFILE (default --mode auto)
  The number of drawing views on the sheet is estimated from the PDF
  scale anchors located in the drawing area (title-block strip excluded).
    views <= 5  -> SINGLE pass, box threshold 0.12
    views >= 6  -> TILED  pass, box threshold 0.08
  If a sheet has no text anchors (scanned), a single-pass probe counts
  the detections instead.
  ZERO-BOX ESCALATION: if an auto-routed SINGLE pass keeps no boxes at all
  (after the quality filter), the sheet is re-run TILED at threshold 0.10. The old score-gap rule is still available via
  --threshold-strategy score_gap.

VIEW-AWARE CROP BOXES
  After detection, a second "crop box" is drawn per drawing view (blue).
  DINO boxes are tight and often clip grid bubbles, level markers and the
  view title, so the crop is built from the PDF's own geometry:
    1. every view = one scale anchor + its title rule (long horizontal line)
    2. DINO boxes are assigned to the view whose title rule is nearest below
    3. the union of those boxes grows over connected vector/text content
       (bounded by the title-rule width and the view above)
    4. the title strip + margin are added
  Views DINO missed get a content-only crop (cyan); boxes with no title
  (notes, legends) get a margin-only crop (purple); boxes inside the
  title-block strip are dropped.

  FIX 1 (title/drawing merge, vertical clip): the seed for each view is the
  union of its assigned DINO box(es) and its title bbox (see
  `vinfo[...]["seed"]` below). Previously the vertical top bound (`ytop`,
  from `_view_ytop`, which only reasons about neighboring rows) could sit
  BELOW this view's own title bbox and clip the title off the top of the
  crop even though it was unioned into the seed/region. Fixed by clamping
  `ytop` to never exceed this view's own title top.

  FIX 2 (title/drawing merge, assignment): `assign_boxes_to_views` refused
  to attach a DINO box to a view when the gap between the box's bottom and
  the view's title line exceeded `max_title_gap` (a fixed 350pt default).
  On a sheet with a page-spanning drawing whose title sits far below the
  content (e.g. a single large "OVERALL" plan sharing the sheet with small
  enlarged details), the top of that drawing landed farther from its own
  title than max_title_gap allows, so it was never assigned -> its crop
  stopped partway down instead of reaching the title. Fixed by giving each
  view an effective gap tolerance of max(max_title_gap, its own vertical
  room up to the next view above) so a page-spanning drawing gets a
  page-spanning tolerance while small, tightly-packed views keep the
  original tight tolerance.

Based on the existing dino_lab.py pipeline:
- Grounding DINO
- single / tiled inference
- quality filtering
- cross-tile merge + NMS
- PDF scale-anchor scoring
- overlay output
- CSV results

Adaptive threshold design from the 33-run experiment:
- DINO first collects candidates at a low floor of 0.08.
- Raw confidence scores are analyzed after inference.
- A meaningful confidence gap is used when one exists.
- Otherwise the empirical fallback 0.12 is used.
- Threshold is constrained to [0.08, 0.15].
- For tiled mode, all tile candidates are collected first and ONE
  page-level threshold is selected before cross-tile merge/NMS.

Important:
The 33-run Excel contains aggregate results, not raw confidence scores.
Therefore the adaptive rule must be evaluated on raw DINO scores at runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import traceback
from datetime import datetime
from typing import Any

import fitz
import pandas as pd
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-base"

DEFAULT_PROMPT = (
    "floor plan . elevation drawing . 3d isometric view . structural drawing . "
    "architectural section . detail drawing . building structure ."
)

D = dict(
    # Empirical adaptive-threshold settings
    adaptive_floor=0.08,
    adaptive_fallback=0.12,
    adaptive_ceiling=0.15,
    min_score_gap=0.05,

    # Sheet-adaptive profile (mode + threshold chosen per drawing sheet)
    max_single_views=5,       # <=5 views -> single, >=6 -> tiled
    single_thr=0.12,          # box threshold for single mode
    tiled_thr=0.08,           # box threshold for tiled mode
    zero_box_fallback=True,   # single pass found nothing -> retry tiled
    fallback_tiled_thr=0.10,  # threshold used for that tiled retry
    title_block_x_frac=0.86,  # anchors right of this x-fraction = title block
    threshold_strategy="profile",  # "profile" | "score_gap"

    # View-aware crop boxes (all distances in PDF points, 72 pt = 1 inch)
    crop_margin=18.0,         # padding around the grown region
    view_pad=15.0,            # tolerance beyond the title-rule width
    max_title_gap=350.0,      # max gap between a box bottom and its title rule
    crop_link_gap=6.0,        # vertical: content within this gap is "connected"
    crop_link_gap_x=40.0,     # horizontal: side labels / level markers sit farther out
    crop_dpi=200,             # resolution of saved crop images
    single_view_max_area_frac=0.85,  # a 1-view sheet's drawing may fill most of the page

    # DINO text threshold
    text_thr=0.10,

    # Rendering / tiling
    dpi=150,
    tile=2800,
    overlap=0.20,
    nms_iou=0.45,

    # Detection quality filters
    min_w=200,
    min_h=200,
    max_aspect=8.0,
    min_area_frac=0.005,
    max_area_frac=0.40,
)

OUT_ROOT = "dino_adaptive_runs"

_MODEL: dict[str, Any] = {}
_anchor_cache: dict[tuple, list] = {}
_render_cache: dict[tuple, Image.Image] = {}


# ============================================================
# MODEL
# ============================================================

def get_model(model_id: str = MODEL_ID):
    if _MODEL.get("id") == model_id:
        return _MODEL["model"], _MODEL["processor"], _MODEL["device"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[load] {model_id} -> {device}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(model_id)
        .to(device)
        .eval()
    )

    _MODEL.clear()
    _MODEL.update(
        id=model_id,
        model=model,
        processor=processor,
        device=device,
    )
    return model, processor, device


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ============================================================
# ADAPTIVE THRESHOLD
# ============================================================

def adaptive_box_threshold(
    scores,
    mode="single",
    floor=D["adaptive_floor"],
    fallback=D["adaptive_fallback"],
    ceiling=D["adaptive_ceiling"],
    min_score_gap=D["min_score_gap"],
):
    """
    Select the confidence threshold from raw DINO scores.

    Logic:
      1. Never go below 0.08.
      2. If too few candidates exist, use the empirical 0.12 fallback.
      3. Sort confidence scores.
      4. Find the largest meaningful confidence gap.
      5. Put the threshold at the midpoint of that gap.
      6. Clamp to [0.08, 0.15].
      7. If no meaningful gap exists, use 0.12.

    Tiled mode intentionally uses the same page-level rule rather than
    selecting a different threshold independently for every tile. This
    avoids unstable tile-to-tile thresholds.
    """
    scores = np.asarray(scores, dtype=float)

    scores = scores[np.isfinite(scores)]
    scores = scores[(scores >= floor) & (scores <= 1.0)]

    if len(scores) < 4:
        return float(fallback), {
            "method": "fallback_few_candidates",
            "candidate_count": int(len(scores)),
            "gap": None,
        }

    scores = np.sort(scores)
    gaps = np.diff(scores)

    if len(gaps) == 0:
        return float(fallback), {
            "method": "fallback_no_gap",
            "candidate_count": int(len(scores)),
            "gap": None,
        }

    idx = int(np.argmax(gaps))
    largest_gap = float(gaps[idx])

    if largest_gap < min_score_gap:
        return float(fallback), {
            "method": "fallback_no_meaningful_gap",
            "candidate_count": int(len(scores)),
            "gap": largest_gap,
        }

    threshold = float((scores[idx] + scores[idx + 1]) / 2.0)
    threshold = float(np.clip(threshold, floor, ceiling))

    return threshold, {
        "method": "score_gap",
        "candidate_count": int(len(scores)),
        "gap": largest_gap,
        "lower_score": float(scores[idx]),
        "upper_score": float(scores[idx + 1]),
    }


# ============================================================
# GEOMETRY / FILTERS
# ============================================================

def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    ua = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - inter
    )

    return inter / ua if ua > 0 else 0.0


def tiled_nms(dets, iou_thresh):
    """Merge true cross-tile duplicates, then greedy NMS."""
    if not dets:
        return []

    merged = list(dets)

    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(merged)

        for i, a in enumerate(merged):
            if used[i]:
                continue

            box = list(a["box"])
            score = a["score"]
            label = a["label"]

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

            out.append({
                "label": label,
                "score": score,
                "box": box,
            })

        merged = out

    dets2 = sorted(merged, key=lambda d: d["score"], reverse=True)
    kept = []

    while dets2:
        best = dets2.pop(0)
        kept.append(best)
        dets2 = [
            d for d in dets2
            if box_iou(best["box"], d["box"]) < iou_thresh
        ]

    return kept


def quality_filter(dets, img_w, img_h, cfg, log):
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
            log.append(f"drop tiny {frac * 100:.2f}%")
            continue

        if frac > cfg["max_area_frac"]:
            log.append(f"drop whole-sheet {frac * 100:.1f}%")
            continue

        keep.append(d)

    return keep


def generate_tiles(width, height, tile_size, overlap):
    stride = max(1, int(tile_size * (1.0 - overlap)))

    tiles = []
    y = 0

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


# ============================================================
# SCALE ANCHORS
# ============================================================

_SCALE_VALUE_RE = re.compile(
    r'(\b\d+(\s+\d+/\d+|\s*/\s*\d+)?\s*["\']?\s*=\s*\d+[\'"]?'
    r'|\b1\s*:\s*\d+'
    r'|\b(nts|n\.t\.s\.?|no\s*scale|as\s*noted)\b'
    r'|\bscale\b)',
    re.IGNORECASE,
)


def _is_real_scale(text: str) -> bool:
    t = text.strip()

    if not t or len(t) >= 50:
        return False

    if re.match(r'^\d+[\'"]?\s*-\s*\d+', t):
        return False

    return bool(_SCALE_VALUE_RE.search(t))


def find_scale_anchors(page):
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

                spans.append({
                    "text": t,
                    "bbox": [r.x0, r.y0, r.x1, r.y1],
                })

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
                for o in spans
                if o is not s
            ):
                raw.append(s)

    anchors = []

    for s in sorted(raw, key=lambda s: (s["bbox"][1], s["bbox"][0])):
        cx = (s["bbox"][0] + s["bbox"][2]) / 2.0
        cy = (s["bbox"][1] + s["bbox"][3]) / 2.0

        if any(
            abs(cx - a["cx"]) < 40
            and abs(cy - a["cy"]) < 14
            for a in anchors
        ):
            continue

        anchors.append({
            "cx": cx,
            "cy": cy,
            "text": s["text"],
            "bbox": s["bbox"],
        })

    return anchors


def get_page_anchors(page):
    key = (
        getattr(getattr(page, "parent", None), "name", "") or "",
        page.number,
    )

    if key not in _anchor_cache:
        if len(_anchor_cache) > 16:
            _anchor_cache.clear()

        _anchor_cache[key] = find_scale_anchors(page)

    return _anchor_cache[key]


def score_against_anchors(clusters_pdf, anchors, up_tol=40.0, side_tol=30.0):
    claims = []

    for c in clusters_pdf:
        x0, y0, x1, y1 = c

        got = [
            i
            for i, a in enumerate(anchors)
            if (
                (x0 - side_tol) <= a["cx"] <= (x1 + side_tol)
                and y0 <= a["cy"] <= (y1 + up_tol)
            )
        ]

        claims.append(got)

    hit = {i for g in claims for i in g}

    return (
        dict(
            anchors=len(anchors),
            hit=len(hit),
            recall=round(len(hit) / len(anchors), 3)
            if anchors else None,
            clusters=len(clusters_pdf),
            clean=sum(1 for g in claims if len(g) == 1),
            merged=sum(1 for g in claims if len(g) > 1),
            spurious=sum(1 for g in claims if len(g) == 0),
        ),
        claims,
    )


# ============================================================
# DINO INFERENCE
# ============================================================

def _dino_forward(
    image,
    prompt,
    model,
    processor,
    device,
    text_thr,
    proc_short=None,
    proc_long=None,
):
    """
    IMPORTANT:
    DINO post-processing is intentionally performed at the LOW FLOOR
    threshold (0.08). We then manually apply the adaptive threshold.

    This is the key change from the original fixed-threshold pipeline.
    """
    kwargs = {}

    if proc_short and proc_long:
        kwargs["size"] = {
            "shortest_edge": int(proc_short),
            "longest_edge": int(proc_long),
        }

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
        **kwargs,
    )

    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v)
        for k, v in inputs.items()
    }

    in_h, in_w = inputs["pixel_values"].shape[-2:]

    ctx = torch.autocast("cuda") if device == "cuda" else _null()

    with torch.inference_mode(), ctx:
        outputs = model(**inputs)

    path = "grounded"

    try:
        res = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=D["adaptive_floor"],
            text_threshold=text_thr,
            target_sizes=[image.size[::-1]],
        )[0]

    except TypeError:
        try:
            res = processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=D["adaptive_floor"],
                text_threshold=text_thr,
                target_sizes=[image.size[::-1]],
            )[0]

        except Exception:
            path = "image_processor(text_thr IGNORED)"

            res = processor.image_processor.post_process_object_detection(
                outputs,
                threshold=D["adaptive_floor"],
                target_sizes=[image.size[::-1]],
            )[0]

    labels = res.get("labels", res.get("text_labels", []))

    if hasattr(labels, "tolist"):
        labels = labels.tolist()

    dets = []

    for box, score, lab in zip(
        res["boxes"].tolist(),
        res["scores"].tolist(),
        labels,
    ):
        dets.append({
            "box": [float(v) for v in box],
            "score": float(score),
            "label": str(lab),
        })

    del inputs, outputs

    if device == "cuda":
        torch.cuda.empty_cache()

    return dets, {
        "input_px": f"{in_w}x{in_h}",
        "postproc": path,
    }


def select_threshold(scores, cfg):
    """Profile strategy: fixed per-mode threshold. Legacy: score-gap."""
    if cfg.get("threshold_strategy", "profile") == "score_gap":
        return adaptive_box_threshold(scores, mode=cfg["mode"])

    thr = float(cfg.get("fixed_thr", D["adaptive_fallback"]))
    return thr, {
        "method": f"profile_{cfg['mode']}",
        "candidate_count": int(len(scores)),
        "gap": None,
    }


def count_view_anchors(anchors, page_width, title_x_frac):
    """Scale anchors inside the drawing area (title-block strip excluded)."""
    limit = page_width * title_x_frac
    return [a for a in anchors if a["cx"] < limit]


def choose_profile(page, anchors, img, cfg, prompt, model, processor, device):
    """
    Decide single vs tiled + threshold for THIS sheet.
    Returns (profile_dict, probe) where probe is the raw single-pass result
    (reused by detect_single) or None.
    """
    tb_x = title_block_x(page, cfg)
    n_views = len([a for a in anchors if a["cx"] < tb_x])
    probe = None
    source = "anchors"

    if n_views == 0:
        # No usable text anchors (scanned / outlined text): probe with a
        # single pass and count detections at the single-mode threshold.
        source = "probe"
        dets, meta = _dino_forward(
            img, prompt, model, processor, device,
            cfg["text_thr"], cfg.get("proc_short"), cfg.get("proc_long"),
        )
        probe = (dets, meta)

        cand = [d for d in dets if d["score"] >= cfg["single_thr"]]
        cand = tiled_nms(cand, float(cfg["nms_iou"]))
        cand = quality_filter(cand, img.width, img.height, cfg, [])
        n_views = len(cand)

    if n_views <= cfg["max_single_views"]:
        mode, thr = "single", cfg["single_thr"]
    else:
        mode, thr = "tiled", cfg["tiled_thr"]
        probe = None  # tiled re-runs on tiles; probe not reusable

    return {
        "mode": mode,
        "threshold": float(thr),
        "views": n_views,
        "source": source,
    }, probe


def apply_adaptive_threshold(dets, threshold):
    return [
        d for d in dets
        if float(d["score"]) >= threshold
    ]


def detect_single(image, prompt, cfg, model, processor, device, precomputed=None):
    if precomputed is not None:
        dets, meta = precomputed
        meta = dict(meta)
    else:
        dets, meta = _dino_forward(
            image,
            prompt,
            model,
            processor,
            device,
            cfg["text_thr"],
            cfg.get("proc_short"),
            cfg.get("proc_long"),
        )

    scores = [d["score"] for d in dets]

    threshold, threshold_meta = select_threshold(scores, cfg)

    final = apply_adaptive_threshold(dets, threshold)

    meta.update(
        tiles=1,
        raw=len(dets),
        adaptive_threshold=threshold,
        adaptive_method=threshold_meta["method"],
        score_gap=threshold_meta["gap"],
        candidates_for_threshold=threshold_meta["candidate_count"],
    )

    return final, meta


def detect_tiled(image, prompt, cfg, model, processor, device):
    """
    Tiled inference:
      tile inference at 0.08 floor
      -> collect ALL raw scores
      -> choose ONE page-level adaptive threshold
      -> filter
      -> cross-tile merge/NMS
    """
    W, H = image.size

    tiles = generate_tiles(
        W,
        H,
        int(cfg["tile"]),
        float(cfg["overlap"]),
    )

    raw = []
    last_meta = {}

    for tx1, ty1, tx2, ty2 in tiles:
        crop = image.crop((tx1, ty1, tx2, ty2))

        dets, meta = _dino_forward(
            crop,
            prompt,
            model,
            processor,
            device,
            cfg["text_thr"],
            cfg.get("proc_short"),
            cfg.get("proc_long"),
        )

        last_meta = meta

        for d in dets:
            b = d["box"]

            raw.append({
                "label": d["label"],
                "score": d["score"],
                "box": [
                    b[0] + tx1,
                    b[1] + ty1,
                    b[2] + tx1,
                    b[3] + ty1,
                ],
            })

    scores = [d["score"] for d in raw]

    threshold, threshold_meta = select_threshold(scores, cfg)

    filtered = apply_adaptive_threshold(raw, threshold)

    out = tiled_nms(
        filtered,
        float(cfg["nms_iou"]),
    )

    meta = dict(last_meta)

    meta.update(
        tiles=len(tiles),
        raw=len(raw),
        after_adaptive=len(filtered),
        after_nms=len(out),
        adaptive_threshold=threshold,
        adaptive_method=threshold_meta["method"],
        score_gap=threshold_meta["gap"],
        candidates_for_threshold=threshold_meta["candidate_count"],
    )

    return out, meta


# ============================================================
# PDF RENDER / OVERLAY
# ============================================================

def render_page(page, dpi):
    key = (
        getattr(getattr(page, "parent", None), "name", "") or "",
        page.number,
        dpi,
    )

    if key in _render_cache:
        return _render_cache[key]

    sf = dpi / 72.0

    pix = page.get_pixmap(
        matrix=fitz.Matrix(sf, sf)
    )

    img = Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )

    if len(_render_cache) > 8:
        _render_cache.clear()

    _render_cache[key] = img

    return img


def _get_font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass

    return ImageFont.load_default()


def draw_overlay(
    image,
    dets,
    anchors,
    sf,
    claims=None,
    caption="",
    crops=None,
):
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    font = _get_font(max(14, img.width // 90))

    for a in anchors:
        x, y = a["cx"] * sf, a["cy"] * sf
        r = max(8, img.width // 260)

        d.line(
            [x - r, y - r, x + r, y + r],
            fill="magenta",
            width=4,
        )
        d.line(
            [x - r, y + r, x + r, y - r],
            fill="magenta",
            width=4,
        )

    crop_colors = {
        "view": "#0066FF",
        "content_fallback": "#00B7EB",
        "box_only": "#8A2BE2",
    }

    for c in crops or []:
        col = crop_colors.get(c["source"], "#0066FF")
        cb = [int(v) for v in c["bbox_px"]]

        d.rectangle(
            cb,
            outline=col,
            width=max(3, img.width // 500),
        )

        d.text(
            (cb[0] + 6, cb[3] - font.size - 8 if hasattr(font, "size") else cb[3] - 30),
            c["id"],
            fill=col,
            font=font,
        )

    for i, det in enumerate(dets):
        n = len(claims[i]) if claims else 1

        color = {
            0: "red",
            1: "#00C000",
        }.get(n, "orange")

        b = [int(v) for v in det["box"]]

        d.rectangle(
            b,
            outline=color,
            width=max(4, img.width // 400),
        )

        d.text(
            (b[0] + 6, max(0, b[1] - 26)),
            f'{det["score"]:.3f} {det.get("label", "")[:22]}',
            fill=color,
            font=font,
        )

    if caption:
        bar = font.size + 20 if hasattr(font, "size") else 40

        d.rectangle(
            [0, 0, img.width, bar],
            fill="black",
        )

        d.text(
            (10, 10),
            caption,
            fill="white",
            font=font,
        )

    return img


def _cap_and_save(img, path, max_w=2600):
    if img.width > max_w:
        k = max_w / img.width
        img = img.resize(
            (max_w, int(img.height * k))
        )

    img.save(path, quality=85)

    return img


# ============================================================
# VIEW-AWARE CROP BOXES
# ============================================================

_geom_cache: dict[tuple, dict] = {}


def _page_key(page):
    return (
        getattr(getattr(page, "parent", None), "name", "") or "",
        page.number,
    )


def get_page_geometry(page, min_hline=120.0):
    """
    Cached vector + text geometry of a page, in the same (rotated) PDF-point
    space as the scale anchors.

      hlines : (N,3) long horizontal lines  [x0, x1, y]
      items  : (M,4) bounding boxes of every vector path and text span
      spans  : text spans [(x0,y0,x1,y1,text), ...]
    """
    key = _page_key(page)

    if key in _geom_cache:
        return _geom_cache[key]

    rot = page.rotation_matrix if page.rotation != 0 else None

    def tr_rect(r):
        r = fitz.Rect(r)

        if rot:
            r = r * rot
            r.normalize()

        return r

    hlines, rects, vlines = [], [], []

    for d in page.get_drawings():
        r = d.get("rect")

        if r is not None:
            rr = tr_rect(r)
            rects.append((rr.x0, rr.y0, rr.x1, rr.y1))

        for it in d.get("items", []):
            kind = it[0]

            if kind == "l":
                p1, p2 = it[1], it[2]

                if rot:
                    p1, p2 = p1 * rot, p2 * rot

                if abs(p1.y - p2.y) <= 1.0 and abs(p1.x - p2.x) >= min_hline:
                    hlines.append((
                        min(p1.x, p2.x),
                        max(p1.x, p2.x),
                        (p1.y + p2.y) / 2.0,
                    ))

                elif (
                    abs(p1.x - p2.x) <= 1.0
                    and abs(p1.y - p2.y) >= 0.6 * page.rect.height
                ):
                    vlines.append((p1.x + p2.x) / 2.0)

            elif kind == "re":
                rr = tr_rect(it[1])

                if rr.height <= 2.5 and rr.width >= min_hline:
                    hlines.append((
                        rr.x0,
                        rr.x1,
                        (rr.y0 + rr.y1) / 2.0,
                    ))

                elif rr.width <= 2.5 and rr.height >= 0.6 * page.rect.height:
                    vlines.append((rr.x0 + rr.x1) / 2.0)

    spans = []

    for b in page.get_text("dict").get("blocks", []):
        for l in b.get("lines", []):
            for sp in l.get("spans", []):
                t = sp["text"].strip()

                if not t:
                    continue

                r = tr_rect(sp["bbox"])
                spans.append((r.x0, r.y0, r.x1, r.y1, t))
                rects.append((r.x0, r.y0, r.x1, r.y1))

    # Title-block left edge = right-most full-height vertical rule that lies
    # between 75% and 95% of the page width (inner separators are further
    # left, the outer frame further right).
    Wp = float(page.rect.width)
    cand = [x for x in vlines if 0.75 * Wp <= x <= 0.95 * Wp]
    title_x = (max(cand) - 1.0) if cand else None

    geom = {
        "title_x": title_x,
        "hlines": np.array(hlines, dtype=float).reshape(-1, 3),
        "items": np.array(rects, dtype=float).reshape(-1, 4),
        "spans": spans,
    }

    if len(_geom_cache) > 8:
        _geom_cache.clear()

    _geom_cache[key] = geom

    return geom


def title_block_x(page, cfg, geom=None):
    """x (pt) where the title block starts: detected, else fraction of width."""
    try:
        geom = geom or get_page_geometry(page)
        tx = geom.get("title_x")
    except Exception:
        tx = None

    return float(tx) if tx else float(page.rect.width) * cfg["title_block_x_frac"]


def build_views(anchors, geom, page_w, cfg):
    """
    One 'view' per drawing title: scale anchor + the long horizontal rule
    printed with it (the rule's x-extent is the view's viewport width).
    Anchors in the title-block strip are ignored.
    """
    xmax = geom.get("title_x") or page_w * cfg["title_block_x_frac"]
    hl = geom["hlines"]
    views = {}

    for a in anchors:
        if a["cx"] >= xmax:
            continue

        ax0, ay0, ax1, ay1 = a["bbox"]
        line, src = None, "fallback"

        if len(hl):
            m = (
                (hl[:, 2] >= ay0 - 14)
                & (hl[:, 2] <= ay1 + 8)
                & (hl[:, 0] <= a["cx"])
                & (hl[:, 1] >= a["cx"])
                & ((hl[:, 1] - hl[:, 0]) < xmax)
            )

            idx = np.where(m)[0]

            if len(idx):
                best = min(
                    idx,
                    key=lambda i: (
                        round(abs(hl[i, 2] - ay0), 1),
                        -(hl[i, 1] - hl[i, 0]),
                    ),
                )
                line = (float(hl[best, 0]), float(hl[best, 1]), float(hl[best, 2]))
                src = "vector"

        if line is None:
            lx0 = ax0 - 25.0
            line = (lx0, lx0 + 0.35 * page_w, ay0 - 2.0)

        key = (round(line[2]), round(line[0]))

        if key in views:
            v = views[key]
            v["anchor"] = [
                min(v["anchor"][0], ax0), min(v["anchor"][1], ay0),
                max(v["anchor"][2], ax1), max(v["anchor"][3], ay1),
            ]
        else:
            views[key] = {
                "line": line,
                "line_source": src,
                "anchor": [ax0, ay0, ax1, ay1],
            }

    out = []

    for v in views.values():
        lx0, lx1, ly = v["line"]

        title = [
            sp for sp in geom["spans"]
            if (ly - 22) <= (sp[1] + sp[3]) / 2 <= (ly + 1)
            and sp[0] >= lx0 - 4
            and sp[2] <= lx1 + 4
        ]

        tb = list(v["anchor"])

        for sp in title:
            tb = [
                min(tb[0], sp[0]), min(tb[1], sp[1]),
                max(tb[2], sp[2]), max(tb[3], sp[3]),
            ]

        v["title_bbox"] = tb
        v["title"] = " ".join(
            sp[4] for sp in sorted(title, key=lambda z: z[0])
        )
        v["bottom"] = tb[3]

        # Some sheets underline only the title text (rule ~ title width), so
        # the rule does NOT tell us how wide the drawing is.
        v["title_only"] = (lx1 - lx0) <= max(
            (tb[2] - tb[0]) + 60.0,   # rule ~ as wide as the title text
            0.25 * xmax,              # or simply short vs. the drawing area
        )
        out.append(v)

    out.sort(key=lambda v: (round(v["line"][2] / 60.0), v["line"][0]))
    _resolve_viewports(out, xmax)

    return out


def _resolve_viewports(views, xmax, row_tol=40.0):
    """
    v["xr"] = horizontal extent of the view.
      - normal rule (spans the viewport)      -> the rule's x-range
      - title-only rule (as wide as the text) -> stretch sideways to the
        nearest same-row neighbour, or to the drawing-area edge if the view
        is alone in its row.
    """
    for v in views:
        lx0, lx1, ly = v["line"]

        if not v.get("title_only"):
            v["xr"] = (lx0, lx1)
            continue

        left, right = 0.0, xmax

        for w in views:
            if w is v:
                continue

            wx0, wx1, wy = w["line"]

            if abs(wy - ly) > row_tol:
                continue

            if wx1 <= lx0 + 1.0:
                left = max(left, wx1)
            elif wx0 >= lx1 - 1.0:
                right = min(right, wx0)

        v["xr"] = (min(left, lx0), max(right, lx1))


def assign_boxes_to_views(boxes, views, cfg, ytops=None):
    """
    Each DINO box -> the view whose title sits nearest BELOW it.

    Evidence, strongest first:
      strong : the title starts inside the box's horizontal span (titles are
               left-aligned to their own drawing)
      weak   : the box overlaps the view's viewport (a guess when the title
               rule is only as wide as the title text)
    A weak candidate only beats a strong one if it is far closer.

    ytops[vi], if given, is view vi's own vertical room (from
    _view_ytop: the bottom of the nearest view above it, or 0 if none).
    A box may sit anywhere between that and the view's title line and
    still be "this view's box", even farther above the title than
    cfg["max_title_gap"] alone would allow. Without this, a single tall
    drawing whose title sits far below a large block of content loses
    whatever DINO boxes land far from its title line, and the resulting
    crop gets cut off partway through the drawing instead of growing
    down to the title. Small, tightly-packed views (little room above
    them) keep the original tight max_title_gap tolerance.
    """
    pad = cfg["view_pad"]
    assign = []

    for b in boxes:
        bw = max(1e-6, b[2] - b[0])
        bh = max(1e-6, b[3] - b[1])

        # DINO often includes the title strip, so allow the box to dip below
        # the rule by up to a quarter of its height
        neg = -max(20.0, 0.25 * bh)

        strong, weak = [], []

        for vi, v in enumerate(views):
            title_y = v["line"][2]
            gap = title_y - b[3]

            room = (title_y - ytops[vi]) if ytops is not None else 0.0
            eff_max_gap = max(cfg["max_title_gap"], room)

            if gap < neg or gap > eff_max_gap:
                continue

            tl = v["title_bbox"][0]

            if (b[0] - 60.0) <= tl <= (b[2] - 0.1 * bw):
                strong.append((gap, vi))
                continue

            lx0, lx1 = v["xr"]
            ov = min(b[2], lx1 + pad) - max(b[0], lx0 - pad)

            if ov >= 0.5 * bw:
                weak.append((gap, vi))

        best = None

        if strong and weak:
            g_s, v_s = min(strong)
            g_w, v_w = min(weak)
            best = v_w if (g_w >= 0 and g_w < 0.5 * g_s) else v_s
        elif strong:
            best = min(strong)[1]
        elif weak:
            best = min(weak)[1]

        assign.append(best)

    return assign


def _view_ytop(v, views):
    """Upper limit of a view = bottom of the nearest view above it."""
    lx0, lx1 = v["xr"]
    ly = v["line"][2]
    top = 0.0

    for w in views:
        if w is v:
            continue

        wx0, wx1 = w["xr"]
        wy = w["line"][2]

        # only views clearly ABOVE (title lines a few pt apart = same row)
        if wy >= ly - 60.0:
            continue

        ov = min(lx1, wx1) - max(lx0, wx0)

        if ov >= 0.2 * min(lx1 - lx0, wx1 - wx0):
            top = max(top, w["bottom"] + 3.0)

    return top


def _eligible_items(items, xr, yr, ly):
    """Content that may be pulled into a view's crop."""
    if len(items) == 0:
        return items

    w = items[:, 2] - items[:, 0]
    h = items[:, 3] - items[:, 1]

    m = (
        (items[:, 0] >= xr[0]) & (items[:, 2] <= xr[1])
        & (items[:, 1] >= yr[0]) & (items[:, 3] <= yr[1])
        & (w <= 0.98 * (xr[1] - xr[0]))
        & (h <= 0.98 * (yr[1] - yr[0]))
    )

    # the title rule itself would stretch the crop to the full viewport
    title_rule = (h < 3) & (np.abs((items[:, 1] + items[:, 3]) / 2 - ly) <= 4)

    return items[m & ~title_rule]


def _grow(seed, items, gap, max_iter=8):
    """Grow a region over content near it. gap = (gap_x, gap_y)."""
    gx, gy = gap
    R = list(seed)

    for _ in range(max_iter):
        if len(items) == 0:
            break

        hit = (
            (items[:, 0] <= R[2] + gx) & (items[:, 2] >= R[0] - gx)
            & (items[:, 1] <= R[3] + gy) & (items[:, 3] >= R[1] - gy)
        )

        if not hit.any():
            break

        new = [
            min(R[0], items[hit, 0].min()),
            min(R[1], items[hit, 1].min()),
            max(R[2], items[hit, 2].max()),
            max(R[3], items[hit, 3].max()),
        ]

        if np.allclose(new, R):
            break

        R = new

    return R


def _union(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]


def _merge_contained(items, thr=0.8):
    """Merge boxes where the smaller one is >= thr inside the other."""
    items = list(items)
    changed = True

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    while changed:
        changed = False

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i]["box"], items[j]["box"]

                inter = (
                    max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
                )

                sm = min(area(a), area(b))

                if sm > 0 and inter / sm >= thr:
                    top = items[i] if items[i]["score"] >= items[j]["score"] else items[j]
                    items[i] = {
                        "box": _union([a, b]),
                        "score": top["score"],
                        "label": top["label"],
                    }
                    del items[j]
                    changed = True
                    break

            if changed:
                break

    return items


def compute_crops(page, dets, anchors, cfg, sf):
    """
    Build one crop box per drawing view.

    Returns (crops, info). Each crop:
      id, source (view | content_fallback | box_only), bbox_pt, bbox_px,
      score, label, n_boxes, title
    """
    W, H = float(page.rect.width), float(page.rect.height)
    geom = get_page_geometry(page)
    views = build_views(anchors, geom, W, cfg)
    tbx = title_block_x(page, cfg, geom)

    if len(views) == 1:
        # Only one drawing on the sheet: the title rule can be as short as the
        # title text, so use the whole drawing area as the viewport.
        v0 = views[0]
        v0["line"] = (0.0, tbx, v0["line"][2])
        v0["xr"] = (0.0, tbx)
        cfg = {**cfg, "max_title_gap": H}

    # Each view's own vertical room (bottom of the view above it, or 0).
    # Used both to bound growth (as before) and, now, to give each view a
    # per-view gap tolerance in assign_boxes_to_views so a page-spanning
    # drawing whose title sits far below its content still claims that
    # content instead of leaving it unassigned. See FIX 2 above.
    ytops = [_view_ytop(v, views) for v in views]

    boxes = [[c / sf for c in d["box"]] for d in dets]
    assign = assign_boxes_to_views(boxes, views, cfg, ytops)

    m = float(cfg["crop_margin"])
    pad = float(cfg["view_pad"])
    gap = (float(cfg["crop_link_gap_x"]), float(cfg["crop_link_gap"]))
    xmax = tbx

    crops, counters = [], {"V": 0, "F": 0, "U": 0}

    def finish(region, source, prefix, score, label, n_boxes, title,
               ytop=0.0, xlim=(0.0, None), ylim=(0.0, None)):
        xl0 = max(0.0, xlim[0])
        xl1 = xmax if xlim[1] is None else min(xmax, xlim[1])
        yl0 = max(ytop, ylim[0])
        yl1 = H if ylim[1] is None else min(H, ylim[1])

        x0, y0 = max(xl0, region[0] - m), max(yl0, region[1] - m)
        x1, y1 = min(xl1, region[2] + m), min(yl1, region[3] + m)

        if x1 - x0 < 10 or y1 - y0 < 10:
            return

        counters[prefix] += 1

        crops.append({
            "id": f"{prefix}{counters[prefix]}",
            "source": source,
            "bbox_pt": [round(float(v), 1) for v in (x0, y0, x1, y1)],
            "bbox_px": [round(float(v) * sf, 1) for v in (x0, y0, x1, y1)],
            "score": round(float(score), 3),
            "label": label,
            "n_boxes": n_boxes,
            "title": title,
        })

    missed = 0

    # ---- pass 1: seed of every view (assigned boxes + title strip)
    #
    # The seed already unions each view's DINO box(es) with its own title
    # bbox — this is the "merge title + drawing" step. The vertical clip
    # applied later (ytop) must never cut back INTO that seed; see the
    # ytop clamp in pass 2 below (FIX 1).
    vinfo = []

    for vi, v in enumerate(views):
        idxs = [i for i, a in enumerate(assign) if a == vi]

        strip = list(v["title_bbox"]) if cfg.get("crop_include_title", True) \
            else list(v["anchor"])

        vinfo.append({
            "idxs": idxs,
            "ytop": ytops[vi],
            "strip": strip,
            "seed": _union([boxes[i] for i in idxs] + [strip]) if idxs else None,
        })

    def hbounds(vi):
        """
        Horizontal room a view may use: halfway to the seed of the nearest
        neighbour in the same row. Stops one view growing into another when
        drawings are packed tightly.
        """
        me, v = vinfo[vi], views[vi]
        own_y = (me["seed"][1], me["seed"][3]) if me["seed"] \
            else (me["ytop"], v["bottom"])
        own_x = (me["seed"][0], me["seed"][2]) if me["seed"] else None
        own_cx = (own_x[0] + own_x[1]) / 2.0 if own_x \
            else (v["xr"][0] + v["xr"][1]) / 2.0

        bl, br = 0.0, xmax

        for wi, other in enumerate(vinfo):
            if wi == vi or other["seed"] is None:
                continue

            sd = other["seed"]
            ov = min(own_y[1], sd[3]) - max(own_y[0], sd[1])

            if ov < 0.3 * min(own_y[1] - own_y[0], sd[3] - sd[1]):
                continue

            if (sd[0] + sd[2]) / 2.0 < own_cx:
                edge = min((sd[2] + own_x[0]) / 2.0, own_x[0]) if own_x else sd[2]
                bl = max(bl, edge)
            else:
                edge = max((own_x[1] + sd[0]) / 2.0, own_x[1]) if own_x else sd[0]
                br = min(br, edge)

        return bl, br

    def vbounds(vi):
        """Vertical room: halfway to the nearest view above / below it."""
        me = vinfo[vi]

        if me["seed"] is None:
            return 0.0, None

        sd0 = me["seed"]
        bt, bb = 0.0, None

        for wi, other in enumerate(vinfo):
            if wi == vi or other["seed"] is None:
                continue

            sd = other["seed"]
            ov = min(sd0[2], sd[2]) - max(sd0[0], sd[0])

            if ov < 0.3 * min(sd0[2] - sd0[0], sd[2] - sd[0]):
                continue

            if (sd[1] + sd[3]) / 2.0 < (sd0[1] + sd0[3]) / 2.0:      # above
                if sd[3] <= sd0[1]:
                    bt = max(bt, (sd[3] + sd0[1]) / 2.0)
            else:                                                     # below
                if sd[1] >= sd0[3]:
                    edge = (sd0[3] + sd[1]) / 2.0
                    bb = edge if bb is None else min(bb, edge)

        return bt, bb

    # ---- pass 2: grow each view inside its own horizontal room
    for vi, v in enumerate(views):
        me = vinfo[vi]
        idxs, ytop, strip = me["idxs"], me["ytop"], me["strip"]

        # FIX 1: ytop is computed purely from neighboring-row geometry and
        # can sit below this view's own title bbox on tightly packed
        # sheets. If that happens, the later vertical clip
        # (yl0 = max(ytop, ...)) would slice the title off the top of the
        # crop even though it was already unioned into the seed/region
        # above. Clamp ytop so it can never cut into this view's own
        # title.
        ytop = min(ytop, v["title_bbox"][1])

        bl, br = hbounds(vi)

        lx0, lx1 = v["xr"]
        ly = v["line"][2]

        if idxs:  # never clip content inside the boxes assigned to this view
            lx0 = min(lx0, min(boxes[i][0] for i in idxs))
            lx1 = max(lx1, max(boxes[i][2] for i in idxs))

        xr = (max(0.0, bl, lx0 - pad), min(xmax, br, lx1 + pad))
        yr = (ytop, v["bottom"] + 3.0)

        elig = _eligible_items(geom["items"], xr, yr, ly)

        if idxs:
            region = _grow(me["seed"], elig, gap)
            best = max(idxs, key=lambda i: dets[i]["score"])

            finish(
                region, "view", "V",
                dets[best]["score"], dets[best]["label"],
                len(idxs), v["title"], ytop, (bl, br), vbounds(vi),
            )

        else:
            missed += 1

            if not cfg.get("crop_missed_views", True) or len(elig) == 0:
                continue

            region = _union(
                [[elig[:, 0].min(), elig[:, 1].min(),
                  elig[:, 2].max(), elig[:, 3].max()], strip]
            )

            finish(region, "content_fallback", "F", 0.0, "missed_view",
                   0, v["title"], ytop, (bl, br))

    un_items, dropped_tb = [], 0

    for i, a in enumerate(assign):
        if a is not None:
            continue

        if (boxes[i][0] + boxes[i][2]) / 2.0 >= xmax:
            dropped_tb += 1
            continue

        un_items.append({
            "box": boxes[i],
            "score": dets[i]["score"],
            "label": dets[i]["label"],
        })

    if not cfg.get("crop_drop_unanchored", False):
        for it in sorted(
            _merge_contained(un_items),
            key=lambda z: (round(z["box"][1] / 60.0), z["box"][0]),
        ):
            finish(it["box"], "box_only", "U", it["score"], it["label"], 1, "")

    n_views = len(views)
    hit = n_views - missed

    return crops, {
        "views_total": n_views,
        "views_with_box": hit,
        "views_missed": missed,
        "view_recall": round(hit / n_views, 3) if n_views else None,
        "crops_view": counters["V"],
        "crops_fallback": counters["F"],
        "crops_unanchored": counters["U"],
        "dropped_title_block": dropped_tb,
        "line_fallback_views": sum(
            1 for v in views if v["line_source"] != "vector"
        ),
    }


def save_crops(page, img, crops, out_dir, tag, cfg, sf):
    """Save each crop as an image (sharper re-render when page is unrotated)."""
    cdir = os.path.join(out_dir, "crops")
    os.makedirs(cdir, exist_ok=True)

    z = float(cfg["crop_dpi"]) / 72.0

    for c in crops:
        path = os.path.join(cdir, f"{tag}_{c['id']}.jpg")

        try:
            if page.rotation == 0:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(z, z),
                    clip=fitz.Rect(*c["bbox_pt"]),
                )
                im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            else:
                im = img.crop(tuple(int(v) for v in c["bbox_px"]))

            im.save(path, quality=90)
            c["file"] = path

        except Exception as e:
            c["file"] = f"ERROR {type(e).__name__}: {e}"


# ============================================================
# RUN ONE PAGE
# ============================================================

def run_one(
    page,
    page_no,
    cfg,
    prompt,
    out_dir,
    save_overlay=True,
):
    model, processor, device = get_model(
        cfg.get("model_id", MODEL_ID)
    )

    img = render_page(
        page,
        int(cfg["dpi"]),
    )

    sf = int(cfg["dpi"]) / 72.0
    anchors = get_page_anchors(page)

    t0 = time.time()
    err = None
    prof = {}
    probe = None
    was_auto = cfg["mode"] == "auto"
    escalated = ""

    try:
        if cfg["mode"] == "auto":
            prof, probe = choose_profile(
                page, anchors, img, cfg, prompt, model, processor, device
            )
            cfg = {
                **cfg,
                "mode": prof["mode"],
                "fixed_thr": prof["threshold"],
            }

            if prof["views"] <= 1:
                cfg["max_area_frac"] = max(
                    cfg["max_area_frac"], cfg["single_view_max_area_frac"]
                )
            print(
                f"      [auto] views={prof['views']} ({prof['source']}) "
                f"-> {prof['mode']} thr={prof['threshold']}"
            )

        if cfg["mode"] == "single":
            dets, meta = detect_single(
                img,
                prompt,
                cfg,
                model,
                processor,
                device,
                precomputed=probe,
            )
        else:
            dets, meta = detect_tiled(
                img,
                prompt,
                cfg,
                model,
                processor,
                device,
            )

    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dets = []
        meta = {
            "input_px": "-",
            "postproc": "-",
            "tiles": 0,
            "raw": 0,
            "adaptive_threshold": D["adaptive_fallback"],
            "adaptive_method": "CUDA_OOM",
            "score_gap": None,
            "candidates_for_threshold": 0,
        }
        err = "CUDA_OOM"

    except Exception as e:
        dets = []

        meta = {
            "input_px": "-",
            "postproc": "-",
            "tiles": 0,
            "raw": 0,
            "adaptive_threshold": D["adaptive_fallback"],
            "adaptive_method": "ERROR",
            "score_gap": None,
            "candidates_for_threshold": 0,
        }

        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    dt = time.time() - t0

    log = []

    dets = quality_filter(
        dets,
        img.width,
        img.height,
        cfg,
        log,
    )

    # ---- zero-box escalation: single found nothing -> tiled @ fallback_tiled_thr
    if (
        was_auto
        and cfg["mode"] == "single"
        and not err
        and len(dets) == 0
        and cfg.get("zero_box_fallback", True)
    ):
        single_raw = meta.get("raw", 0)
        print(
            f"      [fallback] single kept 0 boxes (raw={single_raw}) "
            f"-> retry tiled thr={cfg['fallback_tiled_thr']}"
        )

        try:
            cfg_t = {
                **cfg,
                "mode": "tiled",
                "fixed_thr": cfg["fallback_tiled_thr"],
            }
            dets_t, meta_t = detect_tiled(
                img, prompt, cfg_t, model, processor, device
            )
            log_t = []
            dets_t = quality_filter(
                dets_t, img.width, img.height, cfg_t, log_t
            )

            log.append(
                f"single kept 0 (raw={single_raw}); "
                f"tiled@{cfg_t['fixed_thr']} kept {len(dets_t)}"
            )

            if dets_t:
                dets, meta, cfg = dets_t, meta_t, cfg_t
                log.extend(log_t)
                escalated = "single_to_tiled"
            else:
                escalated = "tiled_also_empty"

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            escalated = "tiled_failed"
            log.append(f"tiled fallback failed: {type(e).__name__}: {e}")

        dt = time.time() - t0

    crops, crop_info = [], {}

    if cfg.get("enable_crops", True):
        try:
            crops, crop_info = compute_crops(page, dets, anchors, cfg, sf)
        except Exception as e:
            traceback.print_exc()
            crops, crop_info = [], {"error": f"{type(e).__name__}: {e}"}

    clusters_pdf = [
        [b / sf for b in d["box"]]
        for d in dets
    ]

    metrics, claims = score_against_anchors(
        clusters_pdf,
        anchors,
    )

    tag = (
        f"p{page_no}_{cfg['mode']}"
        f"_adaptive"
        f"_thr{meta.get('adaptive_threshold', D['adaptive_fallback']):.2f}"
    )

    overlay_path = None

    if save_overlay:
        os.makedirs(
            out_dir,
            exist_ok=True,
        )

        cap = (
            f"{tag} | adaptive_thr="
            f"{meta.get('adaptive_threshold', D['adaptive_fallback']):.3f} | "
            f"method={meta.get('adaptive_method', '')} | "
            f"anchors={metrics['anchors']} | "
            f"views={prof.get('views', '-')} | "
            f"{('esc=' + escalated + ' | ') if escalated else ''}"
            f"boxes={len(dets)} | "
            f"crops={len(crops)} | "
            f"R={metrics['recall']} | "
            f"sp={metrics['spurious']} | "
            f"mg={metrics['merged']}"
        )

        over_img = draw_overlay(
            img,
            dets,
            anchors,
            sf,
            claims,
            cap,
            crops=crops,
        )

        overlay_path = os.path.join(
            out_dir,
            tag + ".jpg",
        )

        _cap_and_save(
            over_img,
            overlay_path,
        )

        if crops:
            if cfg.get("save_crops"):
                save_crops(page, img, crops, out_dir, tag, cfg, sf)

            with open(
                os.path.join(out_dir, tag + "_crops.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {"page": page_no, "info": crop_info, "crops": crops},
                    f,
                    indent=2,
                )

    row = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "page": page_no,
        "mode": cfg["mode"],
        "views": prof.get("views", ""),
        "views_source": prof.get("source", ""),
        "escalated": escalated,
        "dpi": cfg["dpi"],
        "floor_thr": D["adaptive_floor"],
        "fallback_thr": D["adaptive_fallback"],
        "ceiling_thr": D["adaptive_ceiling"],
        "adaptive_thr": meta.get(
            "adaptive_threshold",
            D["adaptive_fallback"],
        ),
        "adaptive_method": meta.get(
            "adaptive_method",
            "",
        ),
        "score_gap": meta.get(
            "score_gap",
            "",
        ),
        "candidate_count": meta.get(
            "candidates_for_threshold",
            0,
        ),
        "text_thr": cfg["text_thr"],
        "tile": cfg["tile"],
        "overlap": cfg["overlap"],
        "nms_iou": cfg["nms_iou"],
        "input_px": meta.get(
            "input_px",
            "",
        ),
        "anchors": metrics["anchors"],
        "kept": len(dets),
        "crops": len(crops),
        "filter_log": "; ".join(log[:12]),
        "views_total": crop_info.get("views_total", ""),
        "views_with_box": crop_info.get("views_with_box", ""),
        "views_missed": crop_info.get("views_missed", ""),
        "view_recall": crop_info.get("view_recall", ""),
        "crops_view": crop_info.get("crops_view", ""),
        "crops_fallback": crop_info.get("crops_fallback", ""),
        "crops_unanchored": crop_info.get("crops_unanchored", ""),
        "dropped_title_block": crop_info.get("dropped_title_block", ""),
        "hit": metrics["hit"],
        "recall": metrics["recall"],
        "clean": metrics["clean"],
        "merged": metrics["merged"],
        "spurious": metrics["spurious"],
        "seconds": round(dt, 2),
        "error": err or "",
        "overlay": overlay_path or "",
    }

    return row


# ============================================================
# CLI
# ============================================================

def parse_pages(value, total_pages):
    if not value:
        return list(range(1, total_pages + 1))

    pages = []

    for part in value.split(","):
        part = part.strip()

        if "-" in part:
            a, b = part.split("-", 1)

            for p in range(int(a), int(b) + 1):
                if 1 <= p <= total_pages:
                    pages.append(p)

        else:
            p = int(part)

            if 1 <= p <= total_pages:
                pages.append(p)

    return sorted(set(pages))


def main():
    parser = argparse.ArgumentParser(
        description="Grounding DINO with empirical adaptive confidence threshold."
    )

    parser.add_argument(
        "pdf",
        help="Input PDF",
    )

    parser.add_argument(
        "--pages",
        default=None,
        help="Pages, e.g. 1,5,10-15",
    )

    parser.add_argument(
        "--mode",
        choices=["auto", "single", "tiled", "both"],
        default="auto",
        help="auto = choose single/tiled per sheet (default)",
    )

    parser.add_argument(
        "--max-single-views",
        type=int,
        default=D["max_single_views"],
        help="Sheets with <= this many views use single mode",
    )

    parser.add_argument(
        "--single-thr",
        type=float,
        default=D["single_thr"],
    )

    parser.add_argument(
        "--tiled-thr",
        type=float,
        default=D["tiled_thr"],
    )

    parser.add_argument(
        "--title-block-x",
        type=float,
        default=D["title_block_x_frac"],
        help="Anchors right of this page-width fraction are ignored when counting views",
    )

    parser.add_argument(
        "--threshold-strategy",
        choices=["profile", "score_gap"],
        default=D["threshold_strategy"],
    )

    parser.add_argument("--no-zero-box-fallback", action="store_true",
                        help="Do not retry tiled when the single pass keeps 0 boxes")
    parser.add_argument("--fallback-tiled-thr", type=float,
                        default=D["fallback_tiled_thr"],
                        help="Threshold for the tiled retry (default 0.10)")

    parser.add_argument("--no-crops", action="store_true",
                        help="Disable the crop-box step")
    parser.add_argument("--save-crops", action="store_true",
                        help="Also save each crop as an image (crops/ folder)")
    parser.add_argument("--crop-margin", type=float, default=D["crop_margin"],
                        help="Margin around each crop, PDF points (72 = 1 inch)")
    parser.add_argument("--crop-dpi", type=int, default=D["crop_dpi"])
    parser.add_argument("--no-crop-title", action="store_true",
                        help="Do not extend crops to include the view title strip")
    parser.add_argument("--no-crop-missed-views", action="store_true",
                        help="No content-based crop for views DINO missed")
    parser.add_argument("--crop-drop-unanchored", action="store_true",
                        help="Drop boxes that have no view title (notes/legends)")

    parser.add_argument(
        "--dpi",
        type=int,
        default=D["dpi"],
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=D["tile"],
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=D["overlap"],
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=D["nms_iou"],
    )

    parser.add_argument(
        "--text-thr",
        type=float,
        default=D["text_thr"],
    )

    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
    )

    parser.add_argument(
        "--proc-size",
        default=None,
        help="Optional processor size: 1536x2560",
    )

    parser.add_argument(
        "--output",
        default=None,
    )

    args = parser.parse_args()

    doc = fitz.open(args.pdf)

    pages = parse_pages(
        args.pages,
        len(doc),
    )

    if args.output:
        out_dir = args.output
    else:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        out_dir = os.path.join(
            OUT_ROOT,
            stamp,
        )

    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    proc_short = None
    proc_long = None

    if args.proc_size:
        proc_short, proc_long = map(
            int,
            args.proc_size.lower().split("x"),
        )

    modes = (
        ["single", "tiled"]
        if args.mode == "both"
        else [args.mode]
    )

    rows = []

    print()
    print("=" * 72)
    print("GROUNDING DINO — ADAPTIVE THRESHOLD")
    print("=" * 72)
    print(f"floor      : {D['adaptive_floor']}")
    print(f"fallback   : {D['adaptive_fallback']}")
    print(f"ceiling    : {D['adaptive_ceiling']}")
    print(f"min gap    : {D['min_score_gap']}")
    print(f"strategy   : {args.threshold_strategy}")
    print(f"single     : <= {args.max_single_views} views, thr {args.single_thr}")
    print(f"tiled      : >  {args.max_single_views} views, thr {args.tiled_thr}")
    print(f"pages      : {pages}")
    print(f"modes      : {modes}")
    print("=" * 72)

    for page_no in pages:
        page = doc[page_no - 1]

        for mode in modes:
            cfg = {
                "mode": mode,
                "dpi": args.dpi,
                "text_thr": args.text_thr,
                "tile": args.tile,
                "overlap": args.overlap,
                "nms_iou": args.nms_iou,
                "proc_short": proc_short,
                "proc_long": proc_long,
                "max_single_views": args.max_single_views,
                "single_thr": args.single_thr,
                "tiled_thr": args.tiled_thr,
                "title_block_x_frac": args.title_block_x,
                "threshold_strategy": args.threshold_strategy,
                "zero_box_fallback": not args.no_zero_box_fallback,
                "fallback_tiled_thr": args.fallback_tiled_thr,
                "enable_crops": not args.no_crops,
                "save_crops": args.save_crops,
                "crop_margin": args.crop_margin,
                "crop_dpi": args.crop_dpi,
                "crop_include_title": not args.no_crop_title,
                "crop_missed_views": not args.no_crop_missed_views,
                "crop_drop_unanchored": args.crop_drop_unanchored,
                "single_view_max_area_frac": D["single_view_max_area_frac"],
                "view_pad": D["view_pad"],
                "max_title_gap": D["max_title_gap"],
                "crop_link_gap": D["crop_link_gap"],
                "crop_link_gap_x": D["crop_link_gap_x"],
                "fixed_thr": (
                    args.single_thr if mode == "single" else args.tiled_thr
                ),
                "min_w": D["min_w"],
                "min_h": D["min_h"],
                "max_aspect": D["max_aspect"],
                "min_area_frac": D["min_area_frac"],
                "max_area_frac": D["max_area_frac"],
            }

            print(
                f"\n[run] page={page_no} mode={mode}"
            )

            row = run_one(
                page,
                page_no,
                cfg,
                args.prompt,
                out_dir,
                save_overlay=True,
            )

            rows.append(row)

            print(
                f"      adaptive_thr={row['adaptive_thr']:.3f} "
                f"method={row['adaptive_method']} "
                f"boxes={row['kept']} "
                f"recall={row['recall']} "
                f"spurious={row['spurious']} "
                f"merged={row['merged']}"
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    csv_path = os.path.join(
        out_dir,
        "adaptive_results.csv",
    )

    json_path = os.path.join(
        out_dir,
        "adaptive_results.json",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rows,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"CSV : {csv_path}")
    print(f"JSON: {json_path}")
    print(f"OUT : {out_dir}")


if __name__ == "__main__":
    main()