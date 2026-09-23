#!/usr/bin/env python3
"""
step7_direction_engine.py
=========================
Step 7: Grain Direction Compliance QC (Vertical vs Horizontal).
Evaluates direction compliance for the 4 architectural millwork components:
  1. Cabinet   -> Standard: VERTICAL
  2. Toe kick  -> Standard: HORIZONTAL (for wood grain)
  3. Panel     -> Standard: VERTICAL (feature horizontal panels accepted)
  4. Soffit    -> Standard: CAD callout / Spec (HORIZONTAL / VERTICAL)
"""

import os
import sys
import json
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import cv2


def load_material_rules(step1_dir):
    """
    Load dynamic material rules from Step 1 JSON files.
    Returns dict: tag -> {grain_required: bool, expected_direction: str, name: str, mfg: str}
    """
    material_rules = {}
    if not os.path.isdir(step1_dir):
        return material_rules

    for filename in os.listdir(step1_dir):
        if filename.endswith(".json") and filename != "materials_cache.json":
            json_path = os.path.join(step1_dir, filename)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                tag = m_data.get("tag", "").upper().strip()
                if tag:
                    grain_val = str(m_data.get("grain", "")).upper().strip()
                    grain_required = grain_val in ["YES", "Y", "TRUE"]
                    direction = str(m_data.get("direction", "VERTICAL")).upper().strip()
                    if not direction or direction == "NONE":
                        direction = "VERTICAL" if grain_required else "NONE"

                    material_rules[tag] = {
                        "grain_required": grain_required,
                        "expected_direction": direction,
                        "name": m_data.get("name", ""),
                        "manufacturer": m_data.get("manufacturer", "")
                    }
            except Exception as e:
                pass

    return material_rules


def normalize_component_class(raw_cls):
    """Normalize into exactly 4 architectural components: Cabinet, Toe kick, Panel, Soffit."""
    raw_lower = str(raw_cls).strip().lower()
    if "toe" in raw_lower or "kick" in raw_lower:
        return "Toe kick"
    elif "panel" in raw_lower:
        return "Panel"
    elif "soffit" in raw_lower:
        return "Soffit"
    else:
        return "Cabinet"


def draw_direction_arrow(img, cx0, cy0, cx1, cy1, direction, color):
    """Draw a visible double-headed arrow inside the bounding box indicating grain direction."""
    mid_x = int((cx0 + cx1) / 2)
    mid_y = int((cy0 + cy1) / 2)
    w = cx1 - cx0
    h = cy1 - cy0

    if direction == "VERTICAL":
        arrow_len = max(16, min(int(h * 0.45), 60))
        pt1 = (mid_x, mid_y - arrow_len // 2)
        pt2 = (mid_x, mid_y + arrow_len // 2)
        cv2.arrowedLine(img, pt1, pt2, color, 2, tipLength=0.35)
        cv2.arrowedLine(img, pt2, pt1, color, 2, tipLength=0.35)
    elif direction == "HORIZONTAL":
        arrow_len = max(16, min(int(w * 0.45), 60))
        pt1 = (mid_x - arrow_len // 2, mid_y)
        pt2 = (mid_x + arrow_len // 2, mid_y)
        cv2.arrowedLine(img, pt1, pt2, color, 2, tipLength=0.35)
        cv2.arrowedLine(img, pt2, pt1, color, 2, tipLength=0.35)


def run_step7(outputs_dir, project_name):
    """
    Step 7: Grain Direction Compliance Validation.
    """
    proj = os.path.join(outputs_dir, project_name)
    step1_dir = os.path.join(proj, "Step1_MaterialScraping")
    step5_dir = os.path.join(proj, "Step5_MaterialTags")
    step6_dir = os.path.join(proj, "Step6_FinalValidation")
    step6_json = os.path.join(step6_dir, "validation_report.json")
    step7_out_dir = os.path.join(proj, "Step7_DirectionValidation")
    os.makedirs(step7_out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"   STEP 7: GRAIN DIRECTION COMPLIANCE QC — {project_name}")
    print(f"{'='*70}")

    if not os.path.exists(step6_json):
        print(f"[ERROR] Step 6 validation_report.json not found: {step6_json}")
        print("        Please run Step 6 first.")
        return False

    with open(step6_json, "r", encoding="utf-8") as f:
        step6_records = json.load(f)

    material_rules = load_material_rules(step1_dir)
    print(f"[INFO] Loaded {len(step6_records)} components from Step 6 for direction verification.")

    direction_results = []

    for item in step6_records:
        crop_file = item.get("Image", "")
        c_cls = normalize_component_class(item.get("Component", "Cabinet"))
        run_type = item.get("Run_Type", "")
        tag = item.get("Tag", "UNTAGGED")
        tag_src = item.get("Tag_Source", "None")
        grain_req = item.get("Grain_Required", False)
        g_present = item.get("Grain_Present", False)
        detected_dir = item.get("Grain_Direction", "NONE")
        g_src = item.get("Grain_Source", "None")
        step6_status = item.get("Status", "UNKNOWN")

        cx0, cy0, cx1, cy1 = int(item["x0"]), int(item["y0"]), int(item["x1"]), int(item["y1"])
        cw = cx1 - cx0
        ch = cy1 - cy0

        mat_info = material_rules.get(tag, {})
        spec_expected_dir = mat_info.get("expected_direction", "VERTICAL")

        status = "PASS"
        reason = ""
        expected_direction = "VERTICAL"

        # ------------------------------------------------------------- #
        # Direction Compliance Evaluation
        # ------------------------------------------------------------- #
        if not grain_req:
            status = "N/A"
            expected_direction = "NONE"
            reason = f"{c_cls} ({tag}): Non-grain material (no grain required)"

        elif not g_present or detected_dir in ["NONE", "MISSING"]:
            status = "REVIEW REQUIRED"
            expected_direction = "HORIZONTAL" if c_cls == "Toe kick" else "VERTICAL"
            reason = f"{c_cls} ({tag}): Grain arrow missing in CAD, standard requires {expected_direction}"

        else:
            # Grain is required and arrow was detected
            norm_detected = detected_dir.upper().strip()

            if c_cls == "Toe kick":
                expected_direction = "HORIZONTAL"
                if norm_detected == "HORIZONTAL":
                    status = "PASS"
                    reason = f"Toe kick ({tag}): Horizontal grain compliant (AWI standard)"
                elif norm_detected == "VERTICAL":
                    status = "FAIL"
                    reason = f"Toe kick ({tag}): Detected Vertical grain, AWI standard is Horizontal"
                else:
                    status = "REVIEW REQUIRED"
                    reason = f"Toe kick ({tag}): Detected {norm_detected}, expected Horizontal"

            elif c_cls == "Cabinet":
                expected_direction = "VERTICAL"
                if norm_detected == "VERTICAL":
                    status = "PASS"
                    reason = f"Cabinet ({tag}): Vertical grain compliant"
                elif norm_detected == "HORIZONTAL":
                    if spec_expected_dir == "HORIZONTAL":
                        status = "PASS"
                        reason = f"Cabinet ({tag}): Horizontal grain matches spec override"
                    else:
                        status = "FAIL"
                        reason = f"Cabinet ({tag}): Detected Horizontal grain, standard is Vertical"
                else:
                    status = "REVIEW REQUIRED"
                    reason = f"Cabinet ({tag}): Ambiguous direction {norm_detected}"

            elif c_cls == "Panel":
                expected_direction = "VERTICAL"
                if norm_detected == "VERTICAL":
                    status = "PASS"
                    reason = f"Panel ({tag}): Vertical grain compliant"
                elif norm_detected == "HORIZONTAL":
                    # Wide feature panels (aspect ratio >= 1.6) may intentionally run horizontal
                    if cw >= 1.6 * ch or spec_expected_dir == "HORIZONTAL":
                        status = "PASS"
                        reason = f"Panel ({tag}): Horizontal grain compliant for feature panel"
                    else:
                        status = "REVIEW REQUIRED"
                        reason = f"Panel ({tag}): Detected Horizontal grain, standard is Vertical"
                else:
                    status = "REVIEW REQUIRED"
                    reason = f"Panel ({tag}): Ambiguous direction {norm_detected}"

            elif c_cls == "Soffit":
                expected_direction = norm_detected if norm_detected in ["VERTICAL", "HORIZONTAL"] else spec_expected_dir
                if norm_detected in ["VERTICAL", "HORIZONTAL"]:
                    status = "PASS"
                    reason = f"Soffit ({tag}): {norm_detected} grain compliant with CAD callout"
                else:
                    status = "REVIEW REQUIRED"
                    reason = f"Soffit ({tag}): Ambiguous grain direction"

            else:
                # Default fallback for casework
                expected_direction = "VERTICAL"
                if norm_detected == "VERTICAL":
                    status = "PASS"
                    reason = f"{c_cls} ({tag}): Vertical grain compliant"
                else:
                    status = "FAIL"
                    reason = f"{c_cls} ({tag}): Detected {norm_detected}, expected Vertical"

        direction_results.append({
            "Image": crop_file,
            "Component": c_cls,
            "Run_Type": run_type,
            "Tag": tag,
            "Tag_Source": tag_src,
            "Grain_Required": grain_req,
            "Detected_Direction": detected_dir,
            "Expected_Direction": expected_direction,
            "Direction_Source": g_src,
            "Status": status,
            "Reason": reason,
            "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1
        })

    df_results = pd.DataFrame(direction_results)

    # ------------------------------------------------------------------ #
    # Save JSON Report
    # ------------------------------------------------------------------ #
    json_out_path = os.path.join(step7_out_dir, "direction_validation_report.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(direction_results, f, indent=4)
    print(f"\n[OK] Saved Step 7 Direction validation report to: {json_out_path}")

    # ------------------------------------------------------------------ #
    # Generate Visual Annotated Drawings with Directional Arrows
    # ------------------------------------------------------------------ #
    if not df_results.empty:
        for crop_name, group in df_results.groupby("Image"):
            step5_filename = f"detected_{crop_name}"
            step5_img_path = os.path.join(step5_dir, step5_filename)
            if not os.path.exists(step5_img_path):
                # Fallback to step6 final image if available
                step5_img_path = os.path.join(step6_dir, f"final_{crop_name}")
                if not os.path.exists(step5_img_path):
                    continue

            img = cv2.imread(step5_img_path)
            if img is None:
                continue

            for _, r in group.iterrows():
                status = r["Status"]
                cx0, cy0, cx1, cy1 = int(r["x0"]), int(r["y0"]), int(r["x1"]), int(r["y1"])
                det_dir = r["Detected_Direction"]
                exp_dir = r["Expected_Direction"]
                c_cls = r["Component"]
                tag = r["Tag"]

                if status == "PASS":
                    color = (0, 190, 0)      # Vibrant Green
                elif status == "FAIL":
                    color = (0, 0, 230)      # Crimson Red
                elif status == "REVIEW REQUIRED":
                    color = (0, 165, 255)    # Orange
                else:
                    color = (180, 180, 180)  # Neutral Gray for N/A

                # Draw component outline
                cv2.rectangle(img, (cx0, cy0), (cx1, cy1), color, 3)

                # Draw directional arrow icon inside component if grain is detected
                if det_dir in ["VERTICAL", "HORIZONTAL"]:
                    draw_direction_arrow(img, cx0, cy0, cx1, cy1, det_dir, color)

                # Direction badge label text
                if status == "PASS":
                    label_text = f"PASS: {c_cls} [{det_dir}]"
                elif status == "FAIL":
                    label_text = f"FAIL: {c_cls} [{det_dir} != {exp_dir}]"
                elif status == "N/A":
                    label_text = f"N/A: {c_cls} ({tag})"
                else:
                    label_text = f"REVIEW: {c_cls} [Exp {exp_dir}]"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.52
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

                lbl_y = max(cy0, th + 8)
                cv2.rectangle(img, (cx0, lbl_y - th - 6), (cx0 + tw + 6, lbl_y + 2), color, -1)
                text_color = (0, 0, 0) if status in ["REVIEW REQUIRED", "N/A"] else (255, 255, 255)
                cv2.putText(img, label_text, (cx0 + 3, lbl_y - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

            out_img_path = os.path.join(step7_out_dir, f"direction_{crop_name}")
            cv2.imwrite(out_img_path, img)

        print(f"[OK] Generated Step 7 marked-up directional images in: {step7_out_dir}")

    # ------------------------------------------------------------------ #
    # Summary Metrics Table
    # ------------------------------------------------------------------ #
    if not df_results.empty:
        total   = len(df_results)
        passes  = (df_results["Status"] == "PASS").sum()
        fails   = (df_results["Status"] == "FAIL").sum()
        reviews = (df_results["Status"] == "REVIEW REQUIRED").sum()
        nas     = (df_results["Status"] == "N/A").sum()
        active_total = total - nas
        compliance_rate = (passes / active_total * 100.0) if active_total > 0 else 100.0

        print(f"\n{'='*70}")
        print(f"   STEP 7 DIRECTION COMPLIANCE SUMMARY — {project_name}")
        print(f"{'='*70}")
        print(f"   Total Evaluated Components   : {total:,}")
        print(f"   PASS (Direction Compliant)   : {passes:,} ({compliance_rate:.1f}% of grain items)")
        print(f"   FAIL (Direction Mismatch)    : {fails:,}")
        print(f"   REVIEW REQUIRED              : {reviews:,}")
        print(f"   N/A (Non-grain materials)    : {nas:,}")
        print(f"{'-'*70}")

        print("\n   Breakdown by 4 Millwork Components:")
        comp_summary = df_results.groupby(["Component", "Status"]).size().unstack(fill_value=0)
        for c_name, row in comp_summary.iterrows():
            p = row.get("PASS", 0)
            f = row.get("FAIL", 0)
            r = row.get("REVIEW REQUIRED", 0)
            n = row.get("N/A", 0)
            print(f"     * {c_name:12s} -> PASS: {p:3d} | FAIL: {f:3d} | REVIEW: {r:3d} | N/A: {n:3d}")

        print("\n   Breakdown by Material Tag:")
        tag_summary = df_results.groupby(["Tag", "Status"]).size().unstack(fill_value=0)
        for t_name, row in tag_summary.iterrows():
            p = row.get("PASS", 0)
            f = row.get("FAIL", 0)
            r = row.get("REVIEW REQUIRED", 0)
            n = row.get("N/A", 0)
            print(f"     * {t_name:12s} -> PASS: {p:3d} | FAIL: {f:3d} | REVIEW: {r:3d} | N/A: {n:3d}")
        print(f"{'='*70}\n")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Step 7: Grain Direction Compliance QC (Vertical vs Horizontal)")
    parser.add_argument("--outputs-dir", default=r"c:\Grain direction final\outputs_production_final",
                        help="Path to outputs_production_final folder")
    parser.add_argument("--project", required=True,
                        help="Project folder name (e.g. 'Sutter MBCC', 'Docusign')")
    args = parser.parse_args()
    run_step7(args.outputs_dir, args.project)
