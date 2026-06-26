import pdfplumber
import json
import os
import re
import argparse
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
ROLE_MARKERS = [
    "FIXTURE:", "SUPPORT CARRIER:", "FLUSH VALVE:", "SEAT:",
    "FAUCET:", "TEMPERING VALVE:", "DISPOSER:", "VALVE:"
]

# Which column names from the header row hold numeric engineering specs.
# These get extracted as a flat SPECS dict on the TAG, not into COMPONENTS.
SPEC_COLUMN_KEYWORDS = [
    "TRAP", "WASTE", "VENT", "HW", "CW",
    "PRESS", "GPF", "GPM", "FLOW", "MOUNTING"
]

# Which column names belong to the "basis of design" region → COMPONENTS
DESIGN_COLUMN_KEYWORDS = [
    "MANUFACTURER", "MAKE", "MODEL", "DESCRIPTION", "REMARKS", "BASIS"
]

# The primary tag column (leftmost)
TAG_COLUMN_KEYWORD = "TAG"
FIXTURE_COLUMN_KEYWORD = "FIXTURE"


def column_type(header_text):
    """Returns 'TAG', 'FIXTURE', 'SPEC', 'DESIGN', or 'OTHER'."""
    h = header_text.upper()
    if TAG_COLUMN_KEYWORD in h and len(h) < 10:
        return "TAG"
    if FIXTURE_COLUMN_KEYWORD in h:
        return "FIXTURE"
    for k in SPEC_COLUMN_KEYWORDS:
        if k in h:
            return "SPEC"
    for k in DESIGN_COLUMN_KEYWORDS:
        if k in h:
            return "DESIGN"
    return "OTHER"


def get_lines(word_list):
    """Groups words into horizontal text lines, returns [(y_band, text), ...]."""
    lines = defaultdict(list)
    for w in word_list:
        y_band = round(w['top'] / 3) * 3
        lines[y_band].append(w)
    result = []
    for y_band in sorted(lines.keys()):
        line_words = sorted(lines[y_band], key=lambda w: w['x0'])
        result.append((y_band, " ".join(w['text'] for w in line_words)))
    return result


def join_lines(lines_list):
    return re.sub(r'\s+', ' ', " ".join(t for _, t in lines_list)).strip()


def extract_table(pdf_path, out_path, config):
    print(f"\n[Extractor] Reading: {pdf_path}")
    schedule_title = config.get("title", "PLUMBING FIXTURES")

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[0]
        words = page.extract_words(keep_blank_chars=False, x_tolerance=2, y_tolerance=3)

        # ── Step 1: Find the header row by locating 'TAG' ────────────────────
        tag_word = next((w for w in words if w['text'].upper() == TAG_COLUMN_KEYWORD), None)
        if not tag_word:
            print("[ERROR] Could not find 'TAG' column header.")
            return

        header_y = tag_word['top']
        header_bottom_y = tag_word['bottom']

        # Collect all words on the header row (within ±25 px of TAG's Y)
        # Note: headers are multi-line so we scan a wider Y band to get all header text
        header_words = [w for w in words if abs(w['top'] - header_y) < 25]
        header_words.sort(key=lambda w: w['x0'])

        # Update header_bottom_y to the max bottom of any header word
        for w in header_words:
            if w['bottom'] > header_bottom_y:
                header_bottom_y = w['bottom']

        # ── Step 2: Use vertical PDF lines to define column X boundaries ─────
        # This is more reliable than merging adjacent words for multi-line headers.
        v_lines = []
        for edge in page.edges:
            # Vertical lines: width < 2, height > 10, crossing the header Y band
            if edge.get('width', 0) < 2 and edge.get('height', 0) > 10:
                if edge['top'] <= header_y + 10 and edge['bottom'] >= header_y - 10:
                    v_lines.append(round(edge['x0']))

        v_lines = sorted(set(v_lines))
        print(f"[Extractor] Vertical grid lines: {v_lines}")

        # If we found enough vertical lines (one per column divider), use them.
        # Otherwise fall back to building boundaries from the first and last header words.
        if len(v_lines) >= 3:
            # Build column spans from consecutive vertical lines
            col_spans = [(v_lines[i], v_lines[i+1]) for i in range(len(v_lines)-1)]
            # Add the last column from last line to a safe right edge
            col_spans.append((v_lines[-1], v_lines[-1] + 500))
        else:
            # Fallback: no reliable grid lines found, use midpoints between header words
            print("[Extractor] Warning: few vertical lines found, falling back to header midpoints.")
            uniq_xs = sorted(set(round(w['x0']) for w in header_words))
            col_spans = []
            for i in range(len(uniq_xs)):
                x_s = max(0, uniq_xs[i] - 15)
                x_e = (uniq_xs[i] + uniq_xs[i+1]) / 2 if i < len(uniq_xs)-1 else uniq_xs[i] + 500
                col_spans.append((x_s, x_e))

        # ── Step 3: Assign header words → columns, build col_defs ────────────
        # For each column span, collect all header words whose x-center falls inside it.
        col_defs = []
        for x_s, x_e in col_spans:
            col_words = [w for w in header_words
                         if x_s <= (w['x0'] + w['x1']) / 2 < x_e]
            if not col_words:
                continue
            col_text = " ".join(w['text'] for w in sorted(col_words, key=lambda w: (w['top'], w['x0'])))
            col_text = re.sub(r'\s+', ' ', col_text).strip()

            # Cap the last design column so it doesn't bleed into the title block
            ctype = column_type(col_text)
            if ctype == "DESIGN" and x_e > x_s + 400:
                x_e = x_s + 450

            col_defs.append({
                "header": col_text,
                "ctype": ctype,
                "x_start": x_s,
                "x_end": x_e
            })

        for c in col_defs:
            print(f"  Col [{c['ctype']:7s}] '{c['header']}' x={c['x_start']:.0f}→{c['x_end']:.0f}")

        # ── Step 3: Find TAG values to define row Y-boundaries ───────────────
        tag_col = next(c for c in col_defs if c['ctype'] == "TAG")

        raw_tag_words = [
            w for w in words
            if w['top'] > header_bottom_y
            and tag_col['x_start'] <= w['x0'] < tag_col['x_end']
        ]
        # Group into tag lines
        all_tag_lines = get_lines(raw_tag_words)

        # Filter: only keep lines whose text contains a valid tag code (e.g. WC-1, HS-2, KS-1)
        # This eliminates spurious rows from fixture-name words bleeding into the TAG column.
        TAG_PATTERN = re.compile(r'[A-Z]+-\d+')
        tag_lines = [(y, t) for y, t in all_tag_lines if TAG_PATTERN.search(t)]

        print(f"[Extractor] Raw tag lines: {len(all_tag_lines)}, valid tags: {len(tag_lines)}")
        print(f"[Extractor] Tags found: {[t for _, t in tag_lines]}")

        # ── Step 4: Build row boundaries using full-width horizontal grid lines ─
        # In Revit, true row boundaries often span the entire table width (~2800px).
        # Partial boundaries (separating sub-components) only span a few columns (< 1600px).
        # Because lines can be drawn in multiple segments, we sum the widths of all segments at each Y-coordinate.
        h_lines_by_y = {}
        for edge in page.edges:
            if edge.get('height', 0) < 2:
                y = round(edge['top'])
                w = edge.get('width', 0)
                # Find if we already have a line at this Y (within 2 pixels)
                found_y = None
                for ey in h_lines_by_y:
                    if abs(ey - y) <= 2:
                        found_y = ey
                        break
                if found_y is not None:
                    h_lines_by_y[found_y] += w
                else:
                    h_lines_by_y[y] = w

        # Keep only lines whose TOTAL accumulated width across the page is > 2500px
        major_h_lines = [y for y, total_w in h_lines_by_y.items() if total_w > 2500 and y > header_bottom_y - 10]
        major_h_lines.sort()

        row_defs = []
        for ty, tt in tag_lines:
            y_starts = [h for h in major_h_lines if h <= ty]
            y_start = y_starts[-1] if y_starts else header_bottom_y

            y_ends = [h for h in major_h_lines if h > ty + 10]
            y_end = y_ends[0] if y_ends else ty + 250

            row_defs.append({
                "tag": tt.strip(),
                "y_start": y_start,
                "y_end": y_end,
                "cells": {c['header']: [] for c in col_defs}
            })


        # ── Step 5: Assign every word → (row, column) ───────────────────────
        for w in words:
            if w['top'] < header_bottom_y:
                continue
            # Find row
            target_row = None
            for r in row_defs:
                if r['y_start'] <= w['top'] < r['y_end']:
                    target_row = r
                    break
            if target_row is None:
                continue
            # Find column
            xc = (w['x0'] + w['x1']) / 2
            for c in col_defs:
                if c['x_start'] <= xc < c['x_end']:
                    target_row['cells'][c['header']].append(w)
                    break

        # ── Step 6: Build structured output ──────────────────────────────────
        # Identify the single FIXTURE column (fixture name like "WATER CLOSET")
        fixture_col = next((c for c in col_defs if c['ctype'] == "FIXTURE"), None)
        # Spec columns → SPECS dict
        spec_cols = [c for c in col_defs if c['ctype'] == "SPEC"]
        # Design columns → feed into COMPONENTS parser
        design_cols = [c for c in col_defs if c['ctype'] == "DESIGN"]

        final_items = []
        for r in row_defs:
            # --- Clean TAG ------------------------------------------------
            tag_text = r['tag']
            # Strip leading numeric tokens (e.g. "1 1/4 X 1 1/2 2 1 1/2 WC-1" → "WC-1")
            match = re.search(r'([A-Z]+-\d+(?:\s*\(.*?\))?)', tag_text)
            if match:
                tag_text = match.group(1).strip()

            # --- FIXTURE name --------------------------------------------
            fixture_name = ""
            if fixture_col:
                lines = get_lines(r['cells'][fixture_col['header']])
                fixture_name = join_lines(lines)

            # --- SPECS ---------------------------------------------------
            specs = {}
            for sc in spec_cols:
                lines = get_lines(r['cells'][sc['header']])
                val = join_lines(lines)
                key = sc['header'].lower().replace(" ", "_").replace("&", "and")
                specs[key] = val

            # ── Get per-column lines for manufacturer and description ─────────
            mfg_col = next((c for c in design_cols
                            if "MANUFACTURER" in c['header'].upper()
                            or "BASIS" in c['header'].upper()), None)
            desc_col = next((c for c in design_cols
                             if "DESCRIPTION" in c['header'].upper()
                             or "REMARKS" in c['header'].upper()), None)

            mfg_lines = get_lines(r['cells'][mfg_col['header']]) if mfg_col else []
            desc_lines = get_lines(r['cells'][desc_col['header']]) if desc_col else []

            # ── Find role boundaries by scanning manufacturer-column lines ────
            sub_boundaries = []
            for i, (y, text) in enumerate(mfg_lines):
                text_up = text.strip().upper()
                for marker in ROLE_MARKERS:
                    m_key = marker.rstrip(':').upper()
                    if text_up.startswith(m_key):
                        # Intelligent boundary: manufacturer is often printed on the line
                        # ABOVE the role label (e.g. JR SMITH \n SUPPORT CARRIER: FIGURE 0210)
                        # If line i-1 doesn't have a role and doesn't have MODEL/FIGURE, include it!
                        bound_y = y
                        # Intelligent boundary look-back
                        # Only look back if the previous line isn't the VERY FIRST line of the cell (i > 1)
                        # because the first line almost always belongs to the implicit FIXTURE.
                        if i > 1:
                            prev_y, prev_text = mfg_lines[i-1]
                            prev_up = prev_text.upper()
                            has_role = any(prev_up.startswith(rm.rstrip(':').upper()) for rm in ROLE_MARKERS)
                            is_model = "MODEL" in prev_up or "FIGURE" in prev_up
                            if not has_role and not is_model:
                                bound_y = prev_y
                        sub_boundaries.append((bound_y, m_key))
                        break

            # Deduplicate consecutive same-role entries at identical Y
            seen_ys = set()
            clean_bounds = []
            for y, role in sorted(sub_boundaries):
                if y not in seen_ys:
                    clean_bounds.append((y, role))
                    seen_ys.add(y)
            sub_boundaries = clean_bounds

            # Always ensure an implicit FIXTURE role exists at the top of the row if not explicitly labeled
            has_fixture = any(r == "FIXTURE" for _, r in sub_boundaries)
            if not has_fixture:
                sub_boundaries.insert(0, (int(r['y_start']), "FIXTURE"))

            # Count expected role labels from the raw mfg text for validation
            all_mfg_text = " ".join(t.upper() for _, t in mfg_lines)
            expected_count = sum(
                1 for marker in ROLE_MARKERS
                if marker.rstrip(':').upper() in all_mfg_text
            )
            found_count = len(sub_boundaries)
            status = "OK" if found_count >= expected_count else f"WARN: expected≥{expected_count}"
            print(f"  [ROW {tag_text}] {found_count} role labels [{status}]: "
                  f"{[role for _, role in sub_boundaries]}")

            def clean_text(text):
                """Strip role markers, PUA bullet glyphs, excess whitespace."""
                for marker in ROLE_MARKERS:
                    text = text.replace(marker, "").replace(marker.title(), "")
                text = text.replace('\uf06c', '')  # Strip PUA bullet glyph
                text = re.sub(r'\s+', ' ', text).strip()
                return text

            def lines_in_band(lines, y_s, y_e):
                return [(y, t) for y, t in lines if y_s <= y < y_e]

            components = []
            for idx, (y_bound, role) in enumerate(sub_boundaries):
                # Sub-row spans from this role's Y to the next role's Y (or row end)
                y_s = y_bound
                y_e = sub_boundaries[idx + 1][0] if idx < len(sub_boundaries) - 1 else r['y_end']

                comp_mfg = ""
                comp_model = ""
                comp_desc = ""

                for _, text in lines_in_band(mfg_lines, y_s, y_e):
                    text = clean_text(text)
                    if not text:
                        continue
                    if re.match(r'^(MODEL|FIGURE)\b', text, re.IGNORECASE):
                        comp_model += text + " "
                    else:
                        comp_mfg += text + " "

                for _, text in lines_in_band(desc_lines, y_s, y_e):
                    comp_desc += clean_text(text) + " "

                components.append({
                    "role": role,
                    "manufacturer": clean_text(comp_mfg),
                    "model": clean_text(comp_model),
                    "description": clean_text(comp_desc)
                })

            # ── Validation: flag if component count mismatches role count ─────
            if len(components) != len(sub_boundaries):
                print(f"  [WARN] {tag_text}: emitted {len(components)} components but found "
                      f"{len(sub_boundaries)} role labels — possible merge/drop bug!")

            final_items.append({
                "TAG": tag_text,
                "FIXTURE": fixture_name.strip(),
                "SPECS": {k: v.replace('\uf06c', '').strip() for k, v in specs.items()},
                "COMPONENTS": components
            })

        # ── Step 7: Forward-fill "ditto" rows (e.g. KS-2 inheriting from KS-1) ─
        # In engineering schedules, if a variant row leaves all specs blank and 
        # only puts a modifier like "(ACCESSIBLE)" in the fixture name, it implies
        # inheriting the base values from the row directly above it.
        for i in range(1, len(final_items)):
            curr = final_items[i]
            prev = final_items[i-1]

            is_variant = False
            # Check if fixture name is missing or just a modifier
            if not curr["FIXTURE"]:
                curr["FIXTURE"] = prev["FIXTURE"]
                is_variant = True
            elif curr["FIXTURE"].startswith("("):
                # E.g. prev="KITCHEN SINK", curr="(ACCESSIBLE)" -> "KITCHEN SINK (ACCESSIBLE)"
                base_fixture = prev["FIXTURE"].split("(")[0].strip()
                curr["FIXTURE"] = f"{base_fixture} {curr['FIXTURE']}"
                is_variant = True

            # ONLY forward-fill specs if this is a variant/ditto row
            curr_specs_joined = "".join(curr["SPECS"].values()).strip()
            if not curr_specs_joined and is_variant:
                curr["SPECS"] = prev["SPECS"].copy()

        output_data = {schedule_title: {"GENERAL": final_items}}

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)

        print(f"\n[SUCCESS] Saved to {out_path}")
        print(f"Preview ({len(final_items)} tags):")
        for item in final_items[:4]:
            print(f"  TAG={item['TAG']}  FIXTURE={item['FIXTURE']}")
            print(f"    SPECS: {item['SPECS']}")
            for c in item['COMPONENTS'][:2]:
                print(f"    [{c['role']}] mfg={c['manufacturer'][:50]} model={c['model'][:40]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vertically-centered plumbing schedule with full column awareness")
    parser.add_argument("--pdf", required=True, help="Path to source PDF")
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--title", default="PLUMBING FIXTURES",
                        help="Schedule title to use as root key in output JSON")
    args = parser.parse_args()

    extract_table(args.pdf, args.out, {"title": args.title})
