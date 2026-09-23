"""
app.py — Gradio front-end for dino_lab.py
==========================================

Thin UI layer over the existing detection backend. No detection logic lives
here — this only collects inputs, runs `dino_lab.run_one()` per (page, config)
combo, and streams the results back one at a time (ranked shortlist, full run
table, overlay gallery) as they complete, then writes CSV/JSON once the whole
grid is done. All scoring, tiling, NMS and anchor logic stays in dino_lab.py
so CLI and UI runs stay numerically identical.

Results tabs:
    Overlays          the Grounding DINO output image for every run — the only
                       image this harness produces (the old per-title-block
                       "drawing bbox" stage is gone)
    Ranked shortlist  configs sorted by recall / clean / spurious / merged
    All runs          full run table

Label tab:
    Every completed run's DINO image, listed as cards on ONE page. Each card
    shows the image plus its box_thr / text_thr / anchor count (pulled from the
    run's own row — nothing to type) and a ✅ Correct / ❌ Wrong pair of buttons.
    Every click auto-saves to run_labels.csv.

Run with:
    pip install gradio pymupdf transformers torch pillow pandas
    python app.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import fitz  # PyMuPDF
import gradio as gr
import pandas as pd

from dino_lab import (
    D,
    DEFAULT_PROMPT,
    OUT_ROOT,
    build_grid,
    build_research_dataset,
    clear_caches,
    draw_overlay,
    find_scale_anchors,
    parse_pages,
    parse_procs,
    plist,
    rank,
    render_page,
    run_one,
    save_run_labels,
)


# ======================================================================
# SWEEP
# ======================================================================
def run_sweep(
    pdf_file,
    pages,
    prompt,
    modes,
    dpi_s,
    box_s,
    text_s,
    tile_s,
    overlap_s,
    nms_s,
    proc_s,
    min_w,
    min_h,
    max_aspect,
    min_area_frac,
    max_area_frac,
    progress=gr.Progress(),
):
    """Generator: runs one (page, config) combo at a time and yields after
    each one, so the gallery/tables fill in live instead of waiting for the
    whole grid — including when 'Pages' is set to 'all'.

    Every yield is an 8-tuple, in this order (must match `outputs=` below):
        ranked_df, raw_df, overlay_gallery, results.csv, results.json,
        status, label_cache, out_dir

    label_cache is {tag: {img, row, label}} — everything the Label tab's
    gallery needs: the overlay image, the full config+metrics row (for the
    caption and for saving), and the current human label (None until set).
    Built up as runs complete so the Label tab's "Refresh gallery" can pick
    up runs before the sweep finishes.
    """
    if not pdf_file:
        raise gr.Error("Upload a PDF first.")
    if not modes:
        raise gr.Error("Pick at least one mode (tiled and/or single).")

    try:
        dpis = plist(dpi_s, int)
        boxes = plist(box_s, float)
        texts = plist(text_s, float)
        tiles = plist(tile_s, int) or [D["tile"]]
        overlaps = plist(overlap_s, float) or [D["overlap"]]
        nmss = plist(nms_s, float) or [D["nms_iou"]]
        procs = parse_procs(proc_s)
    except Exception as e:
        raise gr.Error(f"Couldn't parse a sweep field — check the comma-separated values ({e})")

    if not dpis or not boxes or not texts:
        raise gr.Error("DPI, box_thr, and text_thr each need at least one value.")

    filt = dict(
        min_w=int(min_w),
        min_h=int(min_h),
        max_aspect=float(max_aspect),
        min_area_frac=float(min_area_frac),
        max_area_frac=float(max_area_frac),
    )
    base = dict(
        min_w=filt["min_w"], min_h=filt["min_h"], max_aspect=filt["max_aspect"],
        min_area_frac=filt["min_area_frac"], max_area_frac=filt["max_area_frac"],
        tile=D["tile"], overlap=D["overlap"], nms_iou=D["nms_iou"],
        proc_short=None, proc_long=None,
    )
    grid = build_grid(modes, dpis, boxes, texts, tiles, overlaps, nmss, procs, base)
    if not grid:
        raise gr.Error("The sweep grid is empty — check your parameter values.")

    empty = pd.DataFrame()
    doc = fitz.open(pdf_file)
    try:
        page_nums = parse_pages(pages, len(doc))
        if not page_nums:
            raise gr.Error("No valid pages selected — check the 'Pages' field.")

        out_dir = os.path.join(OUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)

        total = len(grid) * len(page_nums)
        rows, gallery_items = [], []
        label_cache: dict = {}
        i = 0
        yield (
            empty, empty, [], None, None,
            f"Starting sweep — **{total} run(s)** across **{len(page_nums)} page(s)** ({', '.join(str(p) for p in page_nums)})...",
            label_cache, out_dir,
        )

        for pno in page_nums:
            page = doc[pno - 1]
            for cfg in grid:
                i += 1
                progress(i / total, desc=f"[{i}/{total}] page {pno} — {cfg['mode']} dpi{cfg['dpi']} box{cfg['box_thr']}")

                row, im, label_ctx = run_one(page, pno, cfg, prompt, out_dir)
                rows.append(row)
                if im is not None:
                    gallery_items.append((im, row["tag"]))
                    label_cache[row["tag"]] = {**label_ctx, "label": None}

                df_partial = pd.DataFrame(rows)
                ranked_partial = rank(df_partial)
                pct = i / total * 100
                status = (
                    f"Processing… **{i}/{total}** ({pct:.1f}%) — just finished page **{pno}** "
                    f"({cfg['mode']}, dpi={cfg['dpi']}, box_thr={cfg['box_thr']}) — "
                    f"{row['anchors']} anchor(s), recall={row['recall']}"
                    + (f" — ⚠️ {row['err']}" if row.get("err") else "")
                    + "  \nSwitch to the **Results** tab to see overlays fill in live, "
                      "or **④ Label** to start marking finished runs correct/wrong."
                )
                yield (
                    ranked_partial, df_partial, list(gallery_items), None, None,
                    status, label_cache, out_dir,
                )
    finally:
        doc.close()
        clear_caches()

    if not rows:
        yield empty, empty, [], None, None, "No runs completed — check the page range and PDF.", {}, None
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "results.csv")
    json_path = os.path.join(out_dir, "results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    ranked = rank(df)
    best = ranked.iloc[0] if not ranked.empty else None
    n_err = int(df["err"].notna().sum()) if "err" in df.columns else 0
    status_lines = [
        f"**Done** — {len(df)} runs across {df['page'].nunique()} page(s) -> `{out_dir}`",
    ]
    if best is not None:
        status_lines.append(
            f"Top config: **{best['mode']}**, dpi={best['dpi']}, proc={best['proc']}, "
            f"box_thr={best['box_thr']}, text_thr={best['text_thr']}"
            + (f", tile={best['tile']}, overlap={best['overlap']}, nms={best['nms_iou']}"
               if best['mode'] == "tiled" else "")
            + f" — anchors={best.get('anchors')}, recall={best['recall']}, "
              f"clean={best['clean']}, spurious={best['spurious']}, merged={best['merged']}"
        )
    if n_err:
        status_lines.append(f"⚠️ {n_err} run(s) failed (see `err` column in the full table) — often CUDA OOM at large proc/tile sizes.")

    status_lines.append(
        "Open **④ Label** to mark each DINO output image correct/wrong, "
        "then **Build research dataset** once you've labeled the runs you care about."
    )
    yield (
        ranked, df, gallery_items, csv_path, json_path,
        "\n\n".join(status_lines), label_cache, out_dir,
    )


def preview_page(pdf_file, page_no, dpi):
    if not pdf_file:
        raise gr.Error("Upload a PDF first.")
    doc = fitz.open(pdf_file)
    try:
        page_no = max(1, min(int(page_no), len(doc)))
        page = doc[page_no - 1]
        img = render_page(page, int(dpi))
        anchors = find_scale_anchors(page)
        sf = int(dpi) / 72.0
        over = draw_overlay(
            img, [], anchors, sf, None,
            caption=f"page {page_no}/{len(doc)} — {len(anchors)} scale anchor(s), rendered @ {int(dpi)} dpi ({img.width}x{img.height}px)",
        )
    finally:
        doc.close()
    return over, f"Found **{len(anchors)}** scale/title-block anchor(s) on page {page_no}. These are the ground-truth-proxy points every sweep run is scored against."


def goto_sweep(pdf_file, pages):
    """Gate step 1 -> step 2: make sure there's something to sweep first."""
    if not pdf_file:
        raise gr.Error("Upload a PDF before continuing.")
    if not str(pages).strip():
        raise gr.Error("Enter a page range (e.g. 1,5,8 or 1-4 or all) before continuing.")
    return gr.Tabs(selected=1)


def goto_step(idx: int):
    return gr.Tabs(selected=idx)


def pdf_uploaded(pdf_file):
    """Shows the PDF's real page count as soon as it's uploaded, so you can
    catch a mismatch (e.g. sheet numbering vs. actual PDF pages) immediately."""
    if not pdf_file:
        return ""
    try:
        doc = fitz.open(pdf_file)
        n = doc.page_count
        doc.close()
        return f"📄 This file has **{n} page(s)** (per PyMuPDF)."
    except Exception as e:
        return f"⚠️ Couldn't read this PDF: {e}"


def preview_total(pdf_file, pages, modes, dpi_s, box_s, text_s, tile_s, overlap_s, nms_s, proc_s):
    """Live breakdown of where the final run count comes from: the PDF's
    actual page count, how many of those pages your 'Pages' field selects,
    and how many sweep configs each selected page will run."""
    if not pdf_file:
        return "_Upload a PDF in step ① to see the run-count breakdown here._"
    try:
        doc = fitz.open(pdf_file)
        n_pages_doc = doc.page_count
        doc.close()
        page_nums = parse_pages(pages, n_pages_doc)
    except Exception as e:
        return f"⚠️ Couldn't read this PDF: {e}"
    if not modes:
        return "_Pick at least one mode to see the run-count breakdown._"

    try:
        dpis = plist(dpi_s, int)
        boxes = plist(box_s, float)
        texts = plist(text_s, float)
        tiles = plist(tile_s, int) or [D["tile"]]
        overlaps = plist(overlap_s, float) or [D["overlap"]]
        nmss = plist(nms_s, float) or [D["nms_iou"]]
        procs = parse_procs(proc_s)
    except Exception:
        return "⚠️ Check the comma-separated sweep values above — one of them doesn't parse."

    if not dpis or not boxes or not texts:
        return "_DPI, box_thr, and text_thr each need at least one value._"

    base = dict(
        min_w=D["min_w"], min_h=D["min_h"], max_aspect=D["max_aspect"],
        min_area_frac=D["min_area_frac"], max_area_frac=D["max_area_frac"],
        tile=D["tile"], overlap=D["overlap"], nms_iou=D["nms_iou"],
        proc_short=None, proc_long=None,
    )
    n_grid = len(build_grid(modes, dpis, boxes, texts, tiles, overlaps, nmss, procs, base))
    n_pg = len(page_nums)

    return (
        f"📄 PDF has **{n_pages_doc} page(s)** — `{pages or '1'}` selects **{n_pg}** of them.  \n"
        f"⚙️ Your sweep values expand to **{n_grid} config(s)** per page.  \n"
        f"➡️ **Total runs: {n_grid * n_pg}**"
        + (f" ({n_grid} × {n_pg})" if n_grid != 1 else "")
    )


# ======================================================================
# LABEL TAB — DINO output images as cards; ✅ / ❌ under each one
# ======================================================================
FILTERS = ["All", "Unlabeled", "Correct", "Wrong"]
_BADGE = {
    "correct": "### ✅ CORRECT",
    "wrong": "### ❌ WRONG",
    None: "### ⬜ not labeled yet",
}


def _fmt(v, nd=None):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return "—"
    if nd is not None and isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _card_info(row) -> str:
    """Everything the run already knows about itself — shown under the image
    and also what gets exported to run_labels.csv."""
    lines = [
        f"**`{row['tag']}`**",
        f"box_thr **{_fmt(row['box_thr'])}** · text_thr **{_fmt(row['text_thr'])}** · "
        f"anchors **{_fmt(row.get('anchors'))}** · DINO boxes **{_fmt(row.get('kept'))}**",
        f"{row['mode']} · dpi {_fmt(row['dpi'])} · proc {_fmt(row.get('proc'))}"
        + (f" · tile {_fmt(row['tile'])} · overlap {_fmt(row['overlap'])} · nms {_fmt(row['nms_iou'])}"
           if row["mode"] == "tiled" else ""),
        f"recall **{_fmt(row.get('recall'), 2)}** · clean {_fmt(row.get('clean'))} · "
        f"merged {_fmt(row.get('merged'))} · spurious {_fmt(row.get('spurious'))}",
    ]
    if row.get("err"):
        lines.append(f"⚠️ {row['err']}")
    return "  \n".join(lines)


def _label_summary(cache) -> str:
    n = len(cache)
    ok = sum(1 for c in cache.values() if c.get("label") == "correct")
    bad = sum(1 for c in cache.values() if c.get("label") == "wrong")
    return f"**{n}** run(s) · ✅ **{ok}** correct · ❌ **{bad}** wrong · ⬜ **{n - ok - bad}** unlabeled"


def _persist_label(cache, tag, out_dir):
    """Auto-save one run's verdict (or clear it) to run_labels.csv."""
    if not out_dir:
        return
    r = dict(cache[tag]["row"])
    r["label"] = cache[tag].get("label")
    save_run_labels(out_dir, [r])


def _make_label_handler(tag: str, value: str):
    def _handler(cache, out_dir):
        if not cache or tag not in cache:
            raise gr.Error("This run is no longer in memory — start a new sweep.")
        # clicking the active verdict again clears it (undo a mis-click)
        cache[tag]["label"] = None if cache[tag].get("label") == value else value
        _persist_label(cache, tag, out_dir)
        return _BADGE[cache[tag]["label"]], cache, _label_summary(cache)
    return _handler


def _passes(filter_value, label) -> bool:
    return (
        filter_value == "All"
        or (filter_value == "Unlabeled" and label is None)
        or (filter_value == "Correct" and label == "correct")
        or (filter_value == "Wrong" and label == "wrong")
    )


def build_dataset_click(out_dir_state):
    if not out_dir_state:
        raise gr.Error("Run a sweep first — no run directory to build a dataset from yet.")
    try:
        _df, path = build_research_dataset(out_dir_state)
    except FileNotFoundError as e:
        raise gr.Error(str(e))
    return path, f"🧪 Built **research_dataset.csv** → `{path}`"


# ======================================================================
# UI  —  3-step wizard: Setup -> Sweep parameters -> Run & Results
# ======================================================================
CSS = """
.run-card { border: 1px solid var(--border-color-primary); border-radius: 12px;
            padding: 10px; background: var(--background-fill-secondary); }
.run-card .prose h3 { margin: 4px 0 0 0; }
"""

with gr.Blocks(title="Grounding DINO Parameter Lab", theme=gr.themes.Soft()) as demo:
    gr.HTML(f"<style>{CSS}</style>")
    gr.Markdown(
        """
        # Grounding DINO Parameter Lab
        Sweep detection settings across an architectural sheet PDF and score every run
        against auto-extracted scale-note anchors — Grounding DINO output only.
        """
    )

    label_cache_state = gr.State({})
    out_dir_state = gr.State(None)

    with gr.Tabs() as tabs:
        # ---------------------------------------------------------- STEP 1
        with gr.Tab("① Setup", id=0):
            gr.Markdown("Upload the drawing set, pick which pages to test, and describe what to detect.")
            with gr.Row():
                with gr.Column():
                    pdf_file = gr.File(label="Drawing set (PDF)", file_types=[".pdf"], type="filepath")
                    pdf_info = gr.Markdown("")
                    pages = gr.Textbox(label="Pages", value="1", placeholder="1,5,8  or  1-4  or  all")
                    prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=3)
                with gr.Column():
                    gr.Markdown("**Quick anchor preview** — renders a page and marks the scale-note anchors used for scoring, with no model run.")
                    with gr.Row():
                        prev_page = gr.Number(label="Page", value=1, precision=0)
                        prev_dpi = gr.Number(label="DPI", value=150, precision=0)
                    prev_btn = gr.Button("Preview anchors")
                    prev_status = gr.Markdown("")
                    prev_img = gr.Image(label="Anchors on page", height=360)

            step1_next = gr.Button("Next: sweep parameters →", variant="primary")

        # ---------------------------------------------------------- STEP 2
        with gr.Tab("② Sweep parameters", id=1):
            gr.Markdown("Every comma-separated value below is one axis of the grid — the more values, the more runs.")
            modes = gr.CheckboxGroup(["tiled", "single"], value=["tiled"], label="Modes")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Used by both modes:**")
                    dpi_s = gr.Textbox(label="Render DPI", value="150")
                    proc_s = gr.Textbox(
                        label="Processor size (default,1536x2560,...)", value="default"
                    )
                    box_s = gr.Textbox(label="box_thr", value="0.08")
                    text_s = gr.Textbox(label="text_thr (labels only — see NOTE 2)", value="0.10")
                with gr.Column():
                    gr.Markdown("**Tiled-mode only:**")
                    tile_s = gr.Textbox(label="Tile size (px)", value="2800")
                    overlap_s = gr.Textbox(label="Overlap (0-1)", value="0.20")
                    nms_s = gr.Textbox(label="NMS IoU", value="0.45")

            with gr.Accordion("Quality filters (advanced)", open=False):
                with gr.Row():
                    min_w = gr.Number(label="min_w (px)", value=D["min_w"])
                    min_h = gr.Number(label="min_h (px)", value=D["min_h"])
                    max_aspect = gr.Number(label="max_aspect", value=D["max_aspect"])
                    min_area_frac = gr.Number(label="min_area_frac", value=D["min_area_frac"])
                    max_area_frac = gr.Number(label="max_area_frac", value=D["max_area_frac"])

            grid_preview = gr.Markdown("_Upload a PDF in step ① to see the run-count breakdown here._")

            with gr.Row():
                step2_back = gr.Button("← Back")
                step2_next = gr.Button("Next: run →", variant="primary")

        # ---------------------------------------------------------- STEP 3
        with gr.Tab("③ Run & Results", id=2):
            with gr.Tabs() as run_tabs:
                with gr.Tab("▶ Processing", id=0):
                    gr.Markdown("Kick off the sweep here and watch progress. Switch to **Results** any time — it fills in live and isn't blocked by the loading state on this tab.")
                    with gr.Row():
                        step3_back = gr.Button("← Back", scale=0)
                        run_btn = gr.Button("▶ Run sweep", variant="primary", scale=1)
                    status = gr.Markdown("")
                with gr.Tab("🖼 Results", id=1):
                    with gr.Tabs():
                        with gr.Tab("Overlays"):
                            gr.Markdown("Magenta ✕ = scale anchor · Green box = clean · Orange = merged · Red = spurious. Fills in live as each run finishes.")
                            gallery = gr.Gallery(label=None, columns=3, height=650, object_fit="contain")
                        with gr.Tab("Ranked shortlist"):
                            gr.Markdown("Sorted by recall, then clean boxes, penalised for spurious/merged.")
                            ranked_df = gr.Dataframe(label=None, wrap=True)
                        with gr.Tab("All runs"):
                            raw_df = gr.Dataframe(label=None, wrap=True)
                    with gr.Row():
                        csv_out = gr.File(label="results.csv")
                        json_out = gr.File(label="results.json")

        # ---------------------------------------------------------- STEP 4
        with gr.Tab("④ Label", id=3):
            gr.Markdown(
                "Every completed run's **Grounding DINO output image** is listed below on one page. "
                "Look at the image, then press **✅ Correct** or **❌ Wrong** under it (press again to undo). "
                "Thresholds and the anchor count are read from each run automatically — nothing to type. "
                "Each click is saved straight to `run_labels.csv`."
            )
            with gr.Row():
                label_refresh_btn = gr.Button("🔄 Load / refresh runs", variant="primary", scale=1)
                label_filter = gr.Radio(FILTERS, value="All", label="Show", scale=2)
                label_cols = gr.Radio([1, 2, 3], value=2, label="Images per row", scale=1)
            label_summary = gr.Markdown("_Run a sweep, then press **Load / refresh runs**._")

            @gr.render(
                inputs=[label_cache_state, label_filter, label_cols],
                triggers=[label_refresh_btn.click, label_filter.change, label_cols.change],
            )
            def render_label_cards(cache, flt, ncols):
                if not cache:
                    return
                items = [(t, c) for t, c in cache.items() if _passes(flt, c.get("label"))]
                if not items:
                    gr.Markdown("_No runs match this filter._")
                    return
                ncols = int(ncols or 2)
                for start in range(0, len(items), ncols):
                    with gr.Row(equal_height=False):
                        for tag, ctx in items[start:start + ncols]:
                            with gr.Column(elem_classes="run-card"):
                                gr.Image(
                                    value=ctx["img"], show_label=False, interactive=False,
                                    container=False, format="jpeg",
                                )
                                gr.Markdown(_card_info(ctx["row"]))
                                badge = gr.Markdown(_BADGE[ctx.get("label")])
                                with gr.Row():
                                    ok_btn = gr.Button("✅ Correct", variant="primary")
                                    bad_btn = gr.Button("❌ Wrong", variant="stop")
                                io = dict(
                                    inputs=[label_cache_state, out_dir_state],
                                    outputs=[badge, label_cache_state, label_summary],
                                )
                                ok_btn.click(_make_label_handler(tag, "correct"), **io)
                                bad_btn.click(_make_label_handler(tag, "wrong"), **io)

            gr.Markdown("---")
            build_dataset_btn = gr.Button("🧪 Build research dataset (results.csv + run_labels.csv)")
            dataset_out = gr.File(label="research_dataset.csv")
            dataset_status = gr.Markdown("")

    # ---------------------------------------------------------- wiring
    prev_btn.click(preview_page, inputs=[pdf_file, prev_page, prev_dpi], outputs=[prev_img, prev_status])
    pdf_file.change(pdf_uploaded, inputs=pdf_file, outputs=pdf_info)
    step1_next.click(goto_sweep, inputs=[pdf_file, pages], outputs=tabs)

    grid_preview_inputs = [pdf_file, pages, modes, dpi_s, box_s, text_s, tile_s, overlap_s, nms_s, proc_s]
    for comp in grid_preview_inputs:
        comp.change(preview_total, inputs=grid_preview_inputs, outputs=grid_preview)

    step2_back.click(lambda: goto_step(0), outputs=tabs)
    step2_next.click(lambda: goto_step(2), outputs=tabs)

    step3_back.click(lambda: goto_step(1), outputs=tabs)
    run_btn.click(
        run_sweep,
        inputs=[
            pdf_file, pages, prompt, modes,
            dpi_s, box_s, text_s, tile_s, overlap_s, nms_s, proc_s,
            min_w, min_h, max_aspect, min_area_frac, max_area_frac,
        ],
        # order must match the 8-tuple yielded by run_sweep()
        outputs=[
            ranked_df, raw_df, gallery, csv_out, json_out,
            status, label_cache_state, out_dir_state,
        ],
        show_progress="hidden",  # our own `status` text carries progress; this stops the
                                  # built-in spinner/dim overlay from covering the gallery
    )

    # ---- ④ Label tab wiring ----
    # (the ✅/❌ buttons are wired inside render_label_cards above)
    label_refresh_btn.click(lambda c: _label_summary(c) if c else "_Run a sweep first, then refresh._",
                            inputs=label_cache_state, outputs=label_summary)
    build_dataset_btn.click(build_dataset_click, inputs=out_dir_state, outputs=[dataset_out, dataset_status])


if __name__ == "__main__":
    demo.queue().launch()