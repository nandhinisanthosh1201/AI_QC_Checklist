import sys
import os

with open('c:/nwe_flow/validator.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find where 'def run_pipeline(' starts
idx = content.find('def run_pipeline(')
if idx != -1:
    base_content = content[:idx]
else:
    print('Error: run_pipeline not found')
    sys.exit(1)

# Modify stage_8_final_json in base_content to handle decision_res.get("reasoning")
# Find: "reasoning": (
# Replace with: "reasoning": decision_res.get("reasoning", (
old_reasoning = '''            "reasoning": (
                f"ARCH trigger found: {arch_res.get('arch_trigger_found')}. "
                f"Submittal note found: {sub_res.get('submittal_note_found')}."
            ),'''
new_reasoning = '''            "reasoning": decision_res.get("reasoning", (
                f"ARCH trigger found: {arch_res.get('arch_trigger_found')}. "
                f"Submittal note found: {sub_res.get('submittal_note_found')}."
            )),'''
base_content = base_content.replace(old_reasoning, new_reasoning)

# Append our V3 run_pipeline logic
v3_logic = '''def run_pipeline(sub_crop_path: str, arch_crop_path: str, rule: dict, output_dir: str, arch_ref: str = "NA"):
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

    print("\\n--- [V3] Stage 3: View Applicability & Geometry Check ---")
    view_res = stage_3_view_applicability(sub_crop_path, rule)
    
    # Early Exit for View Type only
    if not view_res.get("view_applicable", False):
        status = "OMITTED"
        print(f"View not applicable (e.g., not Elevation). Status: {status}")
        stage_8_final_json(rule, view_res, {}, {}, {"status": status, "reasoning": "View type mismatch."},
                           {"markup_required": False}, output_path=output_path)
        return

    sub_geom_app = view_res.get("geometry_applicable", False)

    print("\\n--- [V3] Stage 4: ARCH Trigger Validation ---")
    if arch_na:
        print("[SKIPPED — ARCH REF: NA]")
        arch_res = {"arch_trigger_found": False, "arch_view_name": "NA", "confidence": 100}
    else:
        arch_res = stage_4_arch_trigger_validation(arch_crop_path, rule, arch_ref)
        
    arch_sym = arch_res.get("arch_trigger_found", False)

    print("\\n--- [V3] Stage 5: Submittal Note Validation ---")
    sub_res = stage_5_submittal_note_validation(sub_crop_path, rule)
    sub_note = sub_res.get("submittal_note_found", False)

    print("\\n--- [V3] Stage 6: Matrix Logic (The 5 Cases) ---")
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

    print("\\n--- [V3] Stage 7: Markup Decision ---")
    markup_required = status in ["FAIL", "REVIEW_REQUIRED"]
    markup_color = "red" if status == "FAIL" else "yellow"
    markup_label = rule.get("markup_label", "REVIEW")
    markup_res = {"markup_required": markup_required, "markup_color": markup_color, "markup_label": markup_label}

    print("\\n--- [V3] Stage 8: Final JSON Save ---")
    stage_8_final_json(rule, view_res, arch_res, sub_res, decision_res, markup_res, output_path=output_path)

    # Draw Markup
    if markup_required:
        print("\\n--- Generating Markup Image ---")
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
'''

with open('c:/nwe_flow/validator_v3.py', 'w', encoding='utf-8') as f:
    f.write(base_content + v3_logic)

print('Successfully generated validator_v3.py')
