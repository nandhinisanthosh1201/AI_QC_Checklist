"""
run_all_submittals.py
=====================
Batch-processes every PDF in the `All-submittals` folder through
tablet_extractor.py, saving each project's output in its own sub-folder
named after the PDF (without the .pdf extension).

Usage:
    python run_all_submittals.py

All outputs go to:
    All-submittals-output/<project-name>/
        <page>_results.json
        <page>_vector_debug.png
"""

import os
import sys
import json

# ── resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
SUBMITTALS_DIR = os.path.join(SCRIPT_DIR, "All-submittals")
OUTPUT_ROOT    = os.path.join(SCRIPT_DIR, "All-submittals-output")

# ── import extractor (same directory) ───────────────────────────────────────
sys.path.insert(0, SCRIPT_DIR)
import tablet_extractor
from tablet_extractor import TableValidationCache

# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    if not os.path.isdir(SUBMITTALS_DIR):
        print(f"[ERROR] Folder not found: {SUBMITTALS_DIR}")
        sys.exit(1)

    pdfs = sorted(
        f for f in os.listdir(SUBMITTALS_DIR)
        if f.lower().endswith(".pdf")
    )

    if not pdfs:
        print("[INFO] No PDF files found in All-submittals folder.")
        return

    print(f"Found {len(pdfs)} PDF(s) to process.\n")

    summary = []   # collect per-project outcomes for final report

    for pdf_name in pdfs:
        pdf_path     = os.path.join(SUBMITTALS_DIR, pdf_name)
        project_name = os.path.splitext(pdf_name)[0]          # strip .pdf
        out_dir      = os.path.join(OUTPUT_ROOT, project_name)
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 70)
        print(f"  PROJECT : {project_name}")
        print(f"  OUTPUT  : {out_dir}")
        print("=" * 70)

        # ── discover page count ───────────────────────────────────────────
        try:
            import fitz
            doc        = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
        except Exception as e:
            print(f"  [ERROR] Cannot open PDF: {e}\n")
            summary.append({"project": project_name, "status": "error", "error": str(e)})
            continue

        print(f"  Pages   : {page_count}\n")

        # ── One cache per PDF — activates cross-sheet awareness ──────────────
        # The cache pre-indexes ALL scheduled codes across every page of this PDF.
        # When a code is found in a drawing whose table only lives on another
        # sheet, it is correctly recognised as scheduled (not UNSCHEDULED).
        pdf_cache = TableValidationCache()
        pdf_cache.reset_if_new_pdf(pdf_path)

        project_found  = 0
        project_missed = 0
        pages_with_table = []

        # ── run extractor on every page ───────────────────────────────────
        for page_idx in range(page_count):
            page_num_1based = page_idx + 1

            debug_png = os.path.join(out_dir, f"page_{page_num_1based:03d}_debug.png")
            json_path = os.path.join(out_dir, f"page_{page_num_1based:03d}_results.json")

            try:
                result = tablet_extractor.extract(
                    pdf_path,
                    debug_path=debug_png,
                    page_num=page_idx,
                    cache=pdf_cache,       # ← cross-sheet cache
                )
            except Exception as e:
                print(f"  [ERROR] Page {page_num_1based}: {e}")
                continue

            if not result or not result.get("sections"):
                continue   # no table on this page

            # ── build output dict ────────────────────────────────────────
            all_rows = [r for sec in result["sections"] for r in sec["rows"]]
            found    = sum(1 for r in all_rows if r.get("present_in_drawing"))
            missing  = len(all_rows) - found

            project_found  += found
            project_missed += missing
            pages_with_table.append(page_num_1based)

            output = {
                "project":           project_name,
                "pdf":               pdf_name,
                "page":              page_num_1based,
                "validation_status": result.get("validation_status", "VALIDATED"),
                "reused_from_sheet": result.get("reused_from_sheet"),
                "tag_signature":     result.get("tag_signature", []),
                "method":            result.get("method", "unknown"),
                "total_sections":    len(result["sections"]),
                "total_rows":        len(all_rows),
                "found":             found,
                "missing":           missing,
                "sections":          result["sections"],
                "unscheduled_codes_in_drawing": result.get("unscheduled_codes_in_drawing", []),
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=4, ensure_ascii=False)

            status_note = f" [REUSED Sheet {result.get('reused_from_sheet')}]" if result.get("validation_status") == "REUSED_FROM_PREVIOUS_TABLE" else ""
            print(f"  ✓ Page {page_num_1based:>3d}{status_note}  |  "
                  f"rows={len(all_rows)}  found={found}  missing={missing}  "
                  f"→ {os.path.basename(json_path)}")

        # ── project-level summary ────────────────────────────────────────
        if pages_with_table:
            status = "ok" if project_missed == 0 else "missing_codes"
            print(f"\n  SUMMARY: {len(pages_with_table)} page(s) with table  "
                  f"| found={project_found}  missing={project_missed}\n")
        else:
            status = "no_table_found"
            print(f"  SUMMARY: No schedule table found in any page.\n")

        summary.append({
            "project":           project_name,
            "status":            status,
            "pages_with_table":  pages_with_table,
            "total_found":       project_found,
            "total_missing":     project_missed,
        })

    # ── write master summary report ───────────────────────────────────────
    summary_path = os.path.join(OUTPUT_ROOT, "_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  BATCH COMPLETE")
    print("=" * 70)
    for s in summary:
        status_icon = "✓" if s["status"] == "ok" else ("⚠" if s["status"] == "missing_codes" else "✗")
        print(f"  {status_icon}  {s['project']}")
        if s.get("pages_with_table"):
            print(f"       pages={s['pages_with_table']}  "
                  f"found={s['total_found']}  missing={s['total_missing']}")
        else:
            print(f"       No table found")
    print(f"\n  Master report → {summary_path}\n")


if __name__ == "__main__":
    run_all()