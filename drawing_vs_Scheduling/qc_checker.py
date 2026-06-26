"""
QC System v4 — Geometry-first, no hardcoded assumptions
========================================================

APPROACH:
  1. LOCATE TABLE dynamically: find the 'MATERIALS' anchor word, then use
     the vertical divider line at its right edge to extract exact row y-bands
     from vector geometry. Row bboxes span the full table width.
  2. EXTRACT TABLE CODES: words in the code column (left of divider),
     within each row y-band.
  3. EXTRACT DRAWING CODES: every code-like word on the page that is
     NOT inside the table region and NOT inside the title block (x > title_x).
  4. ANNOTATE: red full-row border for missing, orange exact-word border for unknown.

This works regardless of where the table appears on the page.
"""

import re, json, shutil, ctypes, argparse
import pdfplumber
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_r

# ── Patterns ──────────────────────────────────────────────
CODE_RE  = re.compile(r'^([A-Z]{1,4})-([A-Z0-9]+(?:\.[0-9]+)?)$', re.IGNORECASE)
EMBEDDED = re.compile(r'\b([A-Z]{1,4})-([A-Z0-9]+(?:\.[0-9]+)?)\b', re.IGNORECASE)

STOPWORDS = {
    'BY-OTHERS','NTS','VIF','ADJ','CLR','SIM','REV-A','REV-B','REV-C',
    'IN-WALL','IN-FIELD','GT-METAL','L-ANGLE','P-LAM',
    'Z-CLIP','Z-CLIPS','E-MAIL','U-SHAPE','V-GROOVE',
    'PRE-FIN','PLY-WD','SS-2','SF-1','FF-1','A-FF',
}

# ── Colors ────────────────────────────────────────────────
RED    = (220, 26,  26)
GREEN  = (20,  160, 20)
ORANGE = (230, 115, 0)
GREY   = (120, 120, 120)

# ── PDF geometry helpers ──────────────────────────────────

def pdf_y(h, top):
    return h - top


def rect_path(page_raw, h, x0, top, x1, bot):
    py0, py1 = pdf_y(h, bot), pdf_y(h, top)
    p = pdfium_r.FPDFPageObj_CreateNewPath(x0, py0)
    pdfium_r.FPDFPath_LineTo(p, x1, py0)
    pdfium_r.FPDFPath_LineTo(p, x1, py1)
    pdfium_r.FPDFPath_LineTo(p, x0, py1)
    pdfium_r.FPDFPath_Close(p)
    return p


def stroke_rect(page_raw, h, x0, top, x1, bot, rgb, alpha=255, lw=2.0):
    p = rect_path(page_raw, h, x0, top, x1, bot)
    pdfium_r.FPDFPath_SetDrawMode(p, pdfium_r.FPDF_FILLMODE_NONE, True)
    pdfium_r.FPDFPageObj_SetStrokeColor(p, *rgb, alpha)
    pdfium_r.FPDFPageObj_SetStrokeWidth(p, lw)
    pdfium_r.FPDFPage_InsertObject(page_raw, p)


def fill_rect(page_raw, h, x0, top, x1, bot, fill, stroke=None, alpha=220, lw=1.0):
    p = rect_path(page_raw, h, x0, top, x1, bot)
    pdfium_r.FPDFPath_SetDrawMode(p, pdfium_r.FPDF_FILLMODE_ALTERNATE, True)
    pdfium_r.FPDFPageObj_SetFillColor(p, *fill, alpha)
    s = stroke or fill
    pdfium_r.FPDFPageObj_SetStrokeColor(p, *s, 255)
    pdfium_r.FPDFPageObj_SetStrokeWidth(p, lw)
    pdfium_r.FPDFPage_InsertObject(page_raw, p)


def put_text(page_raw, h, x, top, text, size, rgb):
    obj = pdfium_r.FPDFPageObj_NewTextObj(page_raw.pdf.raw, b"Helvetica-Bold", size)
    pdfium_r.FPDFPageObj_SetFillColor(obj, *rgb, 255)
    enc = (text + "\0").encode("utf-16-le")
    ptr = ctypes.cast(enc, ctypes.POINTER(ctypes.c_ushort))
    pdfium_r.FPDFText_SetText(obj, ptr)
    pdfium_r.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, pdf_y(h, top) - size * 0.3)
    pdfium_r.FPDFPage_InsertObject(page_raw, obj)


def draw_line(page_raw, h, x0, t0, x1, t1, rgb, alpha=255, lw=1.2):
    p = pdfium_r.FPDFPageObj_CreateNewPath(x0, pdf_y(h, t0))
    pdfium_r.FPDFPath_SetDrawMode(p, pdfium_r.FPDF_FILLMODE_NONE, True)
    pdfium_r.FPDFPageObj_SetStrokeColor(p, *rgb, alpha)
    pdfium_r.FPDFPageObj_SetStrokeWidth(p, lw)
    pdfium_r.FPDFPath_LineTo(p, x1, pdf_y(h, t1))
    pdfium_r.FPDFPage_InsertObject(page_raw, p)


def draw_arrow(page_raw, h, x0, t0, x1, t1, rgb, alpha=255, lw=1.2):
    import math
    draw_line(page_raw, h, x0, t0, x1, t1, rgb, alpha, lw)
    dx, dy = x1-x0, t1-t0
    L = math.hypot(dx, dy)
    if L < 0.1: return
    ux, uy = dx/L, dy/L
    hl = 4.0
    for sx in (-0.5, 0.5):
        draw_line(page_raw, h, x1, t1,
                  x1 - hl*ux - hl*sx*(-uy),
                  t1 - hl*uy - hl*sx*ux,
                  rgb, alpha, lw)


# ── Table geometry detection ──────────────────────────────

def find_table_geometry(page):
    """
    Dynamically locate the MATERIALS table using vector geometry.

    Strategy:
      1. Find the 'MATERIALS' header word → gives table_x0 and table_top
      2. Find the vertical divider line just right of the code column
         (the first vertical edge whose x0 > MATERIALS.x0 + a few pts)
         → gives code_col_x1 and the list of row y-breakpoints
      3. The table right edge is the large vertical separator line (title block left)
      4. The table bottom is the last row breakpoint

    Returns a dict with:
      table_x0, table_x1, table_top, table_bot,
      code_col_x1,
      row_bands: list of (y_top, y_bot) for each code row
    """
    words = [w for w in page.extract_words(x_tolerance=3, y_tolerance=3) if w.get("upright", True)]

    # Find MATERIALS anchor
    mat_words = [w for w in words if w['text'].strip().upper() == 'MATERIALS']
    if not mat_words:
        return None
    mat = mat_words[0]
    mat_x0  = mat['x0']
    mat_top  = mat['top']

    # The table occupies y from ~mat_top-5 to ~mat_top+100 (heuristic upper bound)
    table_y0 = mat_top - 6
    table_y1 = mat_top + 110   # generous; will be tightened by row bands

    # Find the vertical divider segment: first 'v' edge at x slightly right of mat_x0,
    # within the table y-range, that has multiple stacked segments
    # (the row-separator pattern: many short vertical stubs forming the cell wall)
    edges = page.edges
    v_edges_in_table = [
        e for e in edges
        if e['orientation'] == 'v'
        and mat_x0 - 35 <= e['x0'] <= mat_x0 + 10   # close to code column right edge
        and table_y0 <= e['top'] <= table_y1
    ]

    # Collect unique x positions of those vertical edges
    xs = sorted(set(round(e['x0'], 0) for e in v_edges_in_table))
    if not xs:
        return None

    # The divider we want is the first one to the right of mat_x0 - 5
    # (code col is to the LEFT of mat_x0, divider is just right of code words)
    # Actually from data: code words at x≈706, divider at x≈730.9
    # Pick the x with the most edges (most row segments = the true divider)
    from collections import Counter
    x_counts = Counter(round(e['x0'], 0) for e in v_edges_in_table)
    divider_x = max(x_counts, key=x_counts.__getitem__)

    # Collect all segments at that divider x, sorted by top
    divider_segs = sorted(
        [e for e in v_edges_in_table if abs(e['x0'] - divider_x) < 1.5],
        key=lambda e: e['top']
    )

    # Build list of row boundary y-values from the segment endpoints
    y_breaks = []
    for s in divider_segs:
        for y in (s['top'], s['bottom']):
            if not y_breaks or abs(y - y_breaks[-1]) > 0.5:
                y_breaks.append(y)
    y_breaks.sort()

    # The first interval is the header (MATERIALS), rest are data rows
    # We skip the header row (y_breaks[0]→y_breaks[1])
    # and skip any interval that has no code-like word in the code column

    # Find table right edge: the large vertical separator between drawing and title block
    # It's the tallest vertical edge in the page
    all_v = [e for e in edges if e['orientation'] == 'v']
    if all_v:
        tallest = max(all_v, key=lambda e: e['bottom'] - e['top'])
        table_x1 = tallest['x0']
    else:
        table_x1 = page.width - 120  # fallback

    table_x0 = mat_x0 - 40    # left edge of table (a bit left of MATERIALS word)

    # Build row bands: pairs of (y_top, y_bot) excluding the header
    # and excluding any "separator" rows (e.g. the QTY header divider)
    row_bands = []
    for i in range(1, len(y_breaks) - 1):
        row_bands.append((y_breaks[i], y_breaks[i+1]))

    return {
        'table_x0':    table_x0,
        'table_x1':    table_x1,
        'table_top':   y_breaks[0] if y_breaks else table_y0,
        'table_bot':   y_breaks[-1] if y_breaks else table_y1,
        'code_col_x1': divider_x,
        'row_bands':   row_bands,       # list of (y_top, y_bot)
    }


# ── Code extraction ───────────────────────────────────────

def extract_codes(page):
    """
    Returns:
        table_codes  : dict  code → {word_bbox, row_bbox}
        drawing_codes: dict  code → first occurrence {bbox, text}
        all_occ      : list  every drawing occurrence
        geom         : the table geometry dict (for debug)
    """
    words = [w for w in page.extract_words(x_tolerance=3, y_tolerance=3) if w.get("upright", True)]
    geom  = find_table_geometry(page)

    table_codes   = {}
    drawing_codes = {}
    all_occ       = []

    if geom is None:
        # No table found — treat everything as drawing
        for w in words:
            text = w['text'].strip()
            for m in EMBEDDED.finditer(text):
                code = f"{m.group(1).upper()}-{m.group(2).upper()}"
                if code in STOPWORDS: continue
                rec = {'code': code, 'bbox': [w['x0'], w['top'], w['x1'], w['bottom']], 'text': text}
                all_occ.append(rec)
                drawing_codes.setdefault(code, rec)
        return table_codes, drawing_codes, all_occ, geom

    tx0   = geom['table_x0']
    tx1   = geom['table_x1']
    ttop  = geom['table_top']
    tbot  = geom['table_bot']
    ccx1  = geom['code_col_x1']
    bands = geom['row_bands']

    for w in words:
        text = w['text'].strip()
        wx0, wtop, wx1, wbot = w['x0'], w['top'], w['x1'], w['bottom']

        # ── Is this word inside the table region? ──────────
        in_table = (tx0 - 5 <= wx0 <= tx1 and ttop - 2 <= wtop <= tbot + 2)

        if in_table:
            # Only extract codes from the CODE COLUMN (left of divider)
            if wx0 <= ccx1 + 2 and CODE_RE.match(text) and text.upper() not in STOPWORDS:
                code = text.upper()
                # Find which row band this word falls in
                row_bbox = None
                for (rb_top, rb_bot) in bands:
                    if rb_top - 1 <= wtop <= rb_bot + 1:
                        row_bbox = [tx0, rb_top, tx1, rb_bot]
                        break
                if row_bbox is None:
                    row_bbox = [wx0, wtop, wx1, wbot]  # fallback: word bbox

                if code not in table_codes:
                    table_codes[code] = {
                        'code':     code,
                        'word_bbox': [wx0, wtop, wx1, wbot],
                        'row_bbox':  row_bbox,
                    }
        else:
            # Drawing body: skip title block (x >= tx1)
            if wx0 >= tx1:
                continue
            for m in EMBEDDED.finditer(text):
                code = f"{m.group(1).upper()}-{m.group(2).upper()}"
                if code in STOPWORDS: continue
                rec = {'code': code, 'bbox': [wx0, wtop, wx1, wbot], 'text': text}
                all_occ.append(rec)
                drawing_codes.setdefault(code, rec)

    return table_codes, drawing_codes, all_occ, geom


# ── QC compare ────────────────────────────────────────────

def qc_compare(table_codes, drawing_codes):
    t, d = set(table_codes), set(drawing_codes)
    return {
        'status':        'PASS' if t <= d else 'FAIL',
        'matched':       sorted(t & d),
        'missing':       sorted(t - d),
        'unknown':       sorted(d - t),
        'table_codes':   sorted(t),
        'drawing_codes': sorted(d),
    }


# ── Annotations ───────────────────────────────────────────

PAD = 2.0

def annotate_missing_row(page_raw, h, pw, row_bbox):
    """Red border around the full table row."""
    x0, top, x1, bot = row_bbox
    stroke_rect(page_raw, h, x0-PAD, top-PAD, x1+PAD, bot+PAD, RED, lw=2.0)

    label  = "Missing"
    lw_est = len(label) * 4.8 + 6
    mid_y  = (top + bot) / 2

    # Try left of row; fall back to right
    lx = x0 - PAD - 4 - lw_est
    if lx < 2:
        lx   = x1 + PAD + 4
        tipx = x1 + PAD + 1
    else:
        tipx = x0 - PAD - 1

    fill_rect(page_raw, h, lx, top, lx+lw_est, bot+1, (255,255,255), alpha=230)
    put_text(page_raw, h, lx+3, bot-1, label, 5.5, RED)

    if tipx < lx:
        draw_arrow(page_raw, h, lx,        mid_y, tipx, mid_y, RED, lw=1.0)
    else:
        draw_arrow(page_raw, h, lx+lw_est, mid_y, tipx, mid_y, RED, lw=1.0)


def annotate_unknown(page_raw, h, pw, bbox):
    """Orange border around the exact word in the drawing."""
    x0, top, x1, bot = bbox
    stroke_rect(page_raw, h, x0-PAD, top-PAD, x1+PAD, bot+PAD, ORANGE, lw=2.0)

    label  = "Unknown"
    lw_est = len(label) * 4.8 + 6
    lh     = 10
    MARGIN = 10

    if top - lh - MARGIN > 10:
        ly_top, ly_bot = top - MARGIN - lh, top - MARGIN
        tail_y, tip_y  = ly_bot, top - PAD - 1
    else:
        ly_top, ly_bot = bot + MARGIN, bot + MARGIN + lh
        tail_y, tip_y  = ly_top, bot + PAD + 1

    lx0 = max(2.0, x0 - 2)
    lx1 = lx0 + lw_est
    fill_rect(page_raw, h, lx0, ly_top, lx1, ly_bot, (255,255,255), alpha=230)
    put_text(page_raw, h, lx0+3, ly_bot-1, label, 5.5, ORANGE)
    draw_arrow(page_raw, h, (lx0+lx1)/2, tail_y, (x0+x1)/2, tip_y, ORANGE, lw=1.0)


def draw_summary(page_raw, h, pw, ph, matched, missing, unknown):
    lines = [
        ("QC CHECK", GREY),
        (f"V Matched : {len(matched)}  {', '.join(matched) or '-'}", GREEN),
        (f"X Missing : {len(missing)}  {', '.join(missing) or '-'}", RED    if missing else GREY),
        (f"? Unknown : {len(unknown)}  {', '.join(unknown) or '-'}", ORANGE if unknown else GREY),
    ]
    lh = 8
    bw = min(420, pw - 10)
    bh = len(lines) * lh + 8
    sx, sy = 6, ph - bh - 6
    fill_rect(page_raw, h, sx, sy, sx+bw, sy+bh, (248,248,248), stroke=GREY, alpha=230, lw=0.8)
    for i, (txt, col) in enumerate(lines):
        put_text(page_raw, h, sx+4, sy+bh-4-i*lh, txt, 6.0, col)


# ── Main pipeline ─────────────────────────────────────────

def run_qc(pdf_path, output_pdf="annotated_qc.pdf"):
    print(f"\n=== QC v4: {pdf_path} ===\n")

    all_table    = {}
    all_drawing  = {}
    occ_by_page  = {}
    tbl_by_page  = {}
    qc_by_page   = {}

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            t, d, occ, geom = extract_codes(page)
            tbl_by_page[i]  = t
            occ_by_page[i]  = occ
            qc_by_page[i]   = qc_compare(t, d)
            for k, v in t.items(): all_table.setdefault(k, v)
            for k, v in d.items(): all_drawing.setdefault(k, v)

            pg = qc_by_page[i]
            print(f"Page {i+1}:")
            if geom:
                print(f"  Table region: x=[{geom['table_x0']:.0f},{geom['table_x1']:.0f}]  "
                      f"y=[{geom['table_top']:.1f},{geom['table_bot']:.1f}]  "
                      f"code_col_x1={geom['code_col_x1']:.1f}  rows={len(geom['row_bands'])}")
            print(f"  Table  : {sorted(t)}")
            print(f"  Drawing: {sorted(d)}")
            print(f"  Missing: {pg['missing']}")
            print(f"  Unknown: {pg['unknown']}")

    global_qc = qc_compare(all_table, all_drawing)
    global_qc['pages'] = {str(k): v for k, v in qc_by_page.items()}
    print(f"\nGlobal {global_qc['status']}  matched={global_qc['matched']}  "
          f"missing={global_qc['missing']}  unknown={global_qc['unknown']}")

    # ── Annotate ──
    shutil.copy(pdf_path, output_pdf)
    doc = pdfium.PdfDocument(output_pdf)

    with pdfplumber.open(pdf_path) as pdf:
        for pi, plumb_page in enumerate(pdf.pages):
            ph, pw = plumb_page.height, plumb_page.width
            raw    = doc[pi]
            pg     = qc_by_page[pi]
            miss   = set(pg['missing'])
            unk    = set(pg['unknown'])

            for code, rec in tbl_by_page[pi].items():
                if code in miss:
                    annotate_missing_row(raw, ph, pw, rec['row_bbox'])

            for occ in occ_by_page[pi]:
                if occ['code'] in unk:
                    annotate_unknown(raw, ph, pw, occ['bbox'])

            draw_summary(raw, ph, pw, ph,
                         pg['matched'], pg['missing'], pg['unknown'])

            pdfium_r.FPDFPage_GenerateContent(raw.raw)

    doc.save(output_pdf)
    doc.close()

    json_path = output_pdf.replace('.pdf', '_report.json')
    with open(json_path, 'w') as f:
        json.dump(global_qc, f, indent=4)

    print(f"\nOutput: {output_pdf}")
    print(f"Report: {json_path}")
    return global_qc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf',    required=True)
    ap.add_argument('--output', default='annotated_qc.pdf')
    args = ap.parse_args()
    result = run_qc(args.pdf, args.output)
    print('\n' + json.dumps(result, indent=4))