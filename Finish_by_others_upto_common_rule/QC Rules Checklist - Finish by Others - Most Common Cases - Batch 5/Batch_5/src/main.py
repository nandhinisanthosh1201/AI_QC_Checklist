import os
import json
import sys
import re
from typing import List

def clean_sheet_name(raw: str) -> str:
    """Universal CAD sheet cleaner: strips parentheses/brackets, modifiers (SIM, TYP, OPP, REF), and trailing descriptions."""
    if not raw:
        return ""
    # 1. Remove anything inside parentheses (...), brackets [...], or braces {...}
    cleaned = re.sub(r"[\(\[\{].*?[\)\]\}]", "", raw)
    # 2. Strip common standalone CAD modifier keywords and anything following them
    cleaned = re.sub(r"\b(SIM|SIMILAR|TYP|TYPICAL|OPP|OPPOSITE|HAND|REF|REFERENCE|REV|REVISED|NTS|SEE\s+NOTE)\.?\b.*$", "", cleaned, flags=re.IGNORECASE)
    # 3. Strip trailing descriptions separated by a spaced hyphen or slash (e.g., "AE582 - FLOOR PLAN" -> "AE582")
    cleaned = re.sub(r"\s+[\-\/]\s+.*$", "", cleaned)
    # 4. Clean up any remaining trailing whitespace, hyphens, slashes, or punctuation
    return re.sub(r"[\s\-\/,]+$", "", cleaned).strip()

# Ensure the project root is in sys.path so 'from src...' imports work regardless of how it is executed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.data_models import RuleDefinition, SubmittalMetadataItem, SubmittalValidation
from src.pipeline.stage2_arch_preprocessing import convert_schedule_excel_to_json, extract_visible_tags
from src.pipeline.stage3_rule_engine import execute_arch_rule
from src.pipeline.stage4_markup import apply_markup
from src.pipeline.stage5_output_handler import save_result
from validator import run_pipeline as run_v2_pipeline
from validator_v3 import run_pipeline as run_v3_pipeline
from validator_v4 import run_pipeline as run_v4_pipeline
from submittal import run_pipeline_single_rule as run_submittal_pipeline
import copy

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def resolve_active_rule(base_rule: RuleDefinition, schedule_item: dict, match_field: str) -> RuleDefinition:
    """Flattens a generic rule by substituting the active code tag and description."""
    rule = copy.deepcopy(base_rule)
    
    item_lower = {k.lower(): v for k, v in schedule_item.items()}
    tag = item_lower.get("tag") or item_lower.get("code / tag")
    desc = item_lower.get("description", "")
    
    rule.active_code_tag = str(tag).strip() if tag else ""
    rule.active_description = str(desc).strip()
    
    has_note_template = getattr(rule, "note_template", None) is not None
    if has_note_template and rule.note_template.get("expected_notes_pattern"):
        patterns = rule.note_template["expected_notes_pattern"]
        resolved_notes = []
        import re
        tag_parts = [t.strip().upper() for t in re.split(r'[,/]', rule.active_code_tag) if t.strip()] if rule.active_code_tag else []
        
        for pattern in patterns:
            note_with_desc = pattern
            if rule.active_description:
                note_with_desc = note_with_desc.replace("{DESCRIPTION}", rule.active_description.upper())
                
            if "{TAG}" in note_with_desc and tag_parts:
                for t in tag_parts:
                    resolved_notes.append(note_with_desc.replace("{TAG}", t))
            else:
                resolved_notes.append(note_with_desc)
                
        resolved_notes = list(set(resolved_notes))
                
        if not rule.submittal_validation:
            rule.submittal_validation = SubmittalValidation(validation_type="note_presence", match_type="semantic", expected_notes=[])
            
        if not rule.submittal_validation.expected_notes:
            rule.submittal_validation.expected_notes = []
            
        rule.submittal_validation.expected_notes.extend(resolved_notes)
        
    views_override = item_lower.get("views_to_check")
    
    # If it's a string, convert it to a list
    if isinstance(views_override, str):
        views_override = [v.strip() for v in views_override.split(",") if v.strip()]
        
    if views_override and isinstance(views_override, list) and len(views_override) > 0:
        if hasattr(rule.view_applicability, "allowed_view_types"):
            rule.view_applicability.allowed_view_types = views_override
        else:
            rule.view_applicability["allowed_view_types"] = views_override
    else:
        print(f"WARNING: No valid 'views_to_check' found in schedule for tag '{rule.active_code_tag}'. Views for this item will remain empty and may be skipped.")
        
    return rule

def run_pipeline():
    print("Starting AI CAD QC Pipeline...")
    
    # Define Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir) # c:\nwe_flow
    
    rules_dir = os.path.join(project_root, "rules")
    schedule_json_path = os.path.join(project_root, "schedule.json") # Project-wide file, so keep at root
    output_dir = os.path.join(project_root, "Output")
    
    # ---------------------------------------------------------
    # Initial Loading & Argument Parsing
    # ---------------------------------------------------------
    target_rule = None
    custom_excel = None
    target_tag = None
    target_desc = None
    arch_pdf_name = None
    target_source_file = None
    
    # Parse any extra arguments (like specific rule, specific schedule excel, or target item)
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.lower().endswith(".json"):
            target_rule = os.path.basename(arg)
            i += 1
        elif arg.lower().endswith(".xlsx"):
            custom_excel = arg
            i += 1
        elif arg == "--target-tag" and i + 1 < len(args):
            target_tag = args[i+1]
            i += 2
        elif arg == "--target-desc" and i + 1 < len(args):
            target_desc = args[i+1]
            i += 2
        elif arg == "--arch-pdf-name" and i + 1 < len(args):
            arch_pdf_name = args[i+1]
            i += 2
        elif arg == "--target-source-file" and i + 1 < len(args):
            target_source_file = args[i+1]
            i += 2
        else:
            i += 1
            
    schedule_excel_path = os.path.join(project_root, custom_excel) if custom_excel else os.path.join(project_root, "schedule.xlsx")

    # Load Rules
    rules: List[RuleDefinition] = []
    for rule_file in os.listdir(rules_dir):
        if rule_file.endswith(".json"):
            if target_rule and rule_file.lower() != target_rule.lower():
                continue
            rule_data = load_json(os.path.join(rules_dir, rule_file))
            rules.append(RuleDefinition(**rule_data))
            
    if target_rule:
        print(f"Loaded {len(rules)} specific rule(s) matching: {target_rule}")
    else:
        print(f"Loaded {len(rules)} rules.")
    
    # Load Submittal Views Metadata
    submittal_views: List[SubmittalMetadataItem] = []
    
    # Check if a specific file was passed via command line (e.g., python src/main.py MW1.2.json)
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        json_path = os.path.join(project_root, target_file)
        if os.path.exists(json_path) and target_file.endswith(".json"):
            print(f"Loading specific metadata file: {target_file}")
            views_raw = load_json(json_path)
            for view in views_raw:
                submittal_views.append(SubmittalMetadataItem(**view))
        else:
            print(f"Error: Specified file {target_file} not found.")
            return
    else:
        # Dynamic: scans project root for any .json files if no argument is passed
        print("No specific file provided. Scanning project root for metadata files...")
        excluded_jsons = ["schedule.json", "package.json"]
        for item in os.listdir(project_root):
            if item.endswith(".json") and item not in excluded_jsons:
                json_path = os.path.join(project_root, item)
                views_raw = load_json(json_path)
                for view in views_raw:
                    submittal_views.append(SubmittalMetadataItem(**view))
                
    print(f"Loaded {len(submittal_views)} views from metadata.")
    
    if target_source_file:
        normalized_target = target_source_file.replace("\\", "/")
        submittal_views = [v for v in submittal_views if normalized_target in v.source_file.replace("\\", "/")]
        print(f"Filtered to {len(submittal_views)} view(s) matching --target-source-file")
    
    # ---------------------------------------------------------
    # Stage 2: Arch Preprocessing (Done once)
    # ---------------------------------------------------------
    
    # 1. Convert Schedule (Always convert to catch Excel updates, if Excel exists)
    if os.path.exists(schedule_excel_path):
        convert_schedule_excel_to_json(schedule_excel_path, schedule_json_path)
    else:
        print(f"Skipping schedule conversion: Excel file {schedule_excel_path} not found. Using existing schedule.json.")
        
    # ---------------------------------------------------------
    # Stage 3, 4, 5: Rule Execution
    # ---------------------------------------------------------
    for view in submittal_views:
        # Resolve absolute path for view image
        original_source = view.source_file
        view.source_file = os.path.join(project_root, original_source)
        if not os.path.exists(view.source_file):
            # Try with 'crop/' prefix if crop.py default was used
            fallback_source = os.path.join(project_root, "crop", original_source)
            if os.path.exists(fallback_source):
                view.source_file = fallback_source
        
        # Dynamically resolve arch_crop_image from arch_ref
        arch_tags_json = ""
        arch_crop_image = ""
        
        arch_crop_base = os.path.join(project_root, "arch_crop")
        crop_arch_crop = os.path.join(project_root, "crop", "arch_crop")
        if arch_pdf_name and os.path.exists(os.path.join(crop_arch_crop, arch_pdf_name)):
            arch_crop_base = crop_arch_crop
        elif not os.path.exists(arch_crop_base) and os.path.exists(crop_arch_crop):
            arch_crop_base = crop_arch_crop
        
        if view.arch_ref and str(view.arch_ref).strip() != "NA" and str(view.arch_ref).strip() != "":
            import glob
            
            if "/" in view.arch_ref:
                drawing_num_raw, sheet_name_raw = view.arch_ref.split("/", 1)
                drawing_num = drawing_num_raw.strip()
                sheet_name = clean_sheet_name(sheet_name_raw)
            else:
                drawing_num = None
                sheet_name = clean_sheet_name(view.arch_ref)
                
            if arch_pdf_name:
                # STRICT search inside the specific pdf name
                search_pattern = os.path.join(arch_crop_base, arch_pdf_name, sheet_name, f"{sheet_name}*.png")
            else:
                # DYNAMIC search across any pdf name if not provided
                search_pattern = os.path.join(arch_crop_base, "*", sheet_name, f"{sheet_name}*.png")
                
            matched_images = glob.glob(search_pattern)
            
            if drawing_num and matched_images:
                # Try to find the exact drawing number match
                exact_image = None
                stripped_num = drawing_num.lstrip('0') or '0'
                padded_num = drawing_num.zfill(2)
                for img in matched_images:
                    base = os.path.basename(img)
                    if f"_{drawing_num}.png" in base or f"_{stripped_num}.png" in base or f"_{padded_num}.png" in base:
                        exact_image = img
                        break
                arch_crop_image = exact_image if exact_image else matched_images[0]
            elif matched_images:
                arch_crop_image = matched_images[0]
                
            # Resolve the tags JSON relative to the found image
            if arch_crop_image:
                base_img = os.path.basename(arch_crop_image).replace(".png", "")
                parent_dir = os.path.dirname(arch_crop_image)
                arch_tags_json = os.path.join(parent_dir, f"{base_img}_tags.json")
            else:
                # Fallback to legacy path logic if nothing found
                if drawing_num:
                    arch_crop_image = os.path.join(arch_crop_base, sheet_name, f"{sheet_name}_{drawing_num}.png")
                    arch_tags_json = os.path.join(arch_crop_base, sheet_name, f"{sheet_name}_{drawing_num}_tags.json")
                else:
                    arch_crop_image = os.path.join(arch_crop_base, sheet_name, f"{sheet_name}.png")
                    arch_tags_json = os.path.join(arch_crop_base, sheet_name, f"{sheet_name}_tags.json")
                
            # 2. Extract Visible Tags from Arch Drawing (Done once per arch view)
            # Only do this if at least one rule requires the v1 or v4 flow (which uses these tags)
            needs_arch_tags = any(
                getattr(r, "validation_flow", "arch_vs_submittal_v1") in ["arch_vs_submittal_v1", "tag_lookup_with_rfi_fallback"]
                for r in rules
            )
            if needs_arch_tags and not os.path.exists(arch_tags_json):
                extract_visible_tags(arch_crop_image, arch_tags_json)
        
        for rule in rules:
            active_rules = []
            if rule.common_rule_group:
                with open(schedule_json_path, 'r') as f:
                    schedule_data = json.load(f)
                    
                target_schedule = []
                schedule_name = rule.scope_identification.schedule_name if rule.scope_identification else "Schedule"
                if schedule_name in schedule_data:
                    target_schedule = schedule_data[schedule_name]
                elif len(schedule_data) > 0:
                    target_schedule = list(schedule_data.values())[0]
                    
                match_field = rule.scope_identification.match_field if rule.scope_identification else "code / tag"
                
                for item in target_schedule:
                    item_lower = {k.lower(): v for k, v in item.items()}
                    tag = item_lower.get("tag") or item_lower.get("code / tag")
                    desc = item_lower.get("description", "")
                    
                    if target_tag:
                        import re
                        raw_tag = str(tag).strip().upper()
                        split_tags = [t.strip() for t in re.split(r'[,/]', raw_tag) if t.strip()]
                        if str(target_tag).strip().upper() not in split_tags:
                            continue
                    if target_desc and str(desc).strip().upper() != str(target_desc).strip().upper():
                        continue
                        
                    item_group = item_lower.get("schedule_category") or item_lower.get("common_rule_group")
                    if item_group and str(item_group).strip().lower() == rule.common_rule_group.lower():
                        active_rules.append(resolve_active_rule(rule, item, match_field))
            else:
                active_rules.append(rule)
                
            for active_rule in active_rules:
                path_parts = os.path.normpath(view.source_file).split(os.sep)
                submittal_id = path_parts[-2] if len(path_parts) >= 2 else "UnknownSubmittal"
                view_sheet_id = os.path.splitext(path_parts[-1])[0]
                
                # Append active_code_tag to output directory to separate different instances of the same rule
                rule_dir_name = active_rule.rule_id
                if active_rule.active_code_tag:
                    rule_dir_name = f"{active_rule.rule_id}_{active_rule.active_code_tag}"
                    
                rule_output_dir = os.path.join(output_dir, submittal_id, view_sheet_id, rule_dir_name)
    
                validation_flow = getattr(active_rule, "validation_flow", "arch_vs_submittal_v1")
    
                if validation_flow == "arch_vs_submittal_v1":
                    if "Architectural Drawing" in active_rule.required_sources:
                        result = execute_arch_rule(
                            view=view,
                            rule=active_rule,
                            arch_tags_path=arch_tags_json,
                            schedule_path=schedule_json_path,
                            target_tag=target_tag,
                            target_desc=target_desc
                        )
                    else:
                        print(f"Skipping {active_rule.rule_id}: Alternative non-arch flow not implemented yet.")
                        continue
    
                    markup_path = ""
                    if result.needs_markup:
                        markup_path = apply_markup(view, result, active_rule, rule_output_dir)
                        
                    arch_view_title = "Unknown View"
                    if os.path.exists(arch_tags_json):
                        with open(arch_tags_json, 'r') as f:
                            arch_tags_data = json.load(f)
                            arch_view_title = arch_tags_data.get("view_title") or "Unknown View"
                            
                    save_result(
                        submittal_id=submittal_id,
                        view_sheet_id=view_sheet_id, 
                        rule=active_rule, 
                        result=result, 
                        markup_path=markup_path, 
                        view_metadata=view,
                        arch_view_title=arch_view_title,
                        output_base_dir=output_dir
                    )

                elif validation_flow == "arch_vs_submittal_v2":
                    print(f"Executing {active_rule.rule_id} for View {view.view_name} using arch_vs_submittal_v2...")
                    run_v2_pipeline(view.source_file, arch_crop_image, active_rule.model_dump(), rule_output_dir, view.arch_ref)
    
                elif validation_flow == "arch_vs_submittal_v3":
                    print(f"Executing {active_rule.rule_id} for View {view.view_name} using arch_vs_submittal_v3...")
                    run_v3_pipeline(view.source_file, arch_crop_image, active_rule.model_dump(), rule_output_dir, view.arch_ref)

                elif validation_flow == "direct_submittal":
                    print(f"Executing {active_rule.rule_id} for View {view.view_name} using submittal.py (direct_submittal)...")
                    
                    # Send the exact view from our JSON to bypass AI view guessing
                    cached_views = [{
                        'view_id': 'V1',
                        'detected_view_type': view.view_name,
                        'view_type_classification': view.view_type,
                        'confidence_score': 100,
                        'scope_coordinates': {'x1': 0, 'y1': 0, 'x2': 10000, 'y2': 10000}
                    }]
                    
                    run_submittal_pipeline(view.source_file, active_rule.model_dump(), cached_views, rule_output_dir)
                    
                elif validation_flow == "tag_lookup_with_rfi_fallback":
                    print(f"Executing {active_rule.rule_id} for View {view.view_name} using validator_v4 (4-Stage Truth Table)...")
                    v4_result = run_v4_pipeline(
                        sub_crop_path=view.source_file,
                        arch_crop_path=arch_crop_image,
                        rule=active_rule,
                        output_dir=rule_output_dir,
                        arch_ref=view.arch_ref,
                        schedule_json_path=schedule_json_path,
                        arch_tags_json=arch_tags_json,
                        view=view
                    )
            
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
