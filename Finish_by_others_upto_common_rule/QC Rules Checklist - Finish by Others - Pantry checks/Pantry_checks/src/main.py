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

from src.models.data_models import RuleDefinition, SubmittalMetadataItem
from src.pipeline.stage2_arch_preprocessing import convert_schedule_excel_to_json, extract_visible_tags
from src.pipeline.stage3_rule_engine import execute_arch_rule
from src.pipeline.stage4_markup import apply_markup
from src.pipeline.stage5_output_handler import save_result
from validator import run_pipeline as run_v2_pipeline
from submittal import run_pipeline_single_rule as run_submittal_pipeline

def load_json(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

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
    
    # Parse any extra arguments (like specific rule or specific schedule excel)
    for arg in sys.argv[2:]:
        if arg.lower().endswith(".json"):
            target_rule = arg
        elif arg.lower().endswith(".xlsx"):
            custom_excel = arg
            
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
    
    # ---------------------------------------------------------
    # Stage 2: Arch Preprocessing (Done once)
    # ---------------------------------------------------------
    
    # 1. Convert Schedule (Always convert to catch Excel updates)
    convert_schedule_excel_to_json(schedule_excel_path, schedule_json_path)
        
    # ---------------------------------------------------------
    # Stage 3, 4, 5: Rule Execution
    # ---------------------------------------------------------
    for view in submittal_views:
        # Resolve absolute path for view image
        view.source_file = os.path.join(project_root, view.source_file)
        
        # Dynamically resolve arch_crop_image from arch_ref
        arch_tags_json = ""
        arch_crop_image = ""
        
        if view.arch_ref != "NA":
            if "/" in view.arch_ref:
                drawing_num_raw, sheet_name_raw = view.arch_ref.split("/", 1)
                drawing_num = drawing_num_raw.strip()
                sheet_name = clean_sheet_name(sheet_name_raw)
                
                # Default expected path (with whatever padding was in JSON)
                arch_crop_image = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{drawing_num}.png")
                arch_tags_json = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{drawing_num}_tags.json")
                
                # Fallback logic to handle mismatch in zeros (e.g. "09" vs "9", or "2" vs "02")
                if not os.path.exists(arch_crop_image):
                    stripped_num = drawing_num.lstrip('0') or '0'
                    padded_num = drawing_num.zfill(2)
                    
                    # Try stripped version
                    fallback_image_1 = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{stripped_num}.png")
                    # Try padded version
                    fallback_image_2 = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{padded_num}.png")
                    
                    if os.path.exists(fallback_image_1):
                        arch_crop_image = fallback_image_1
                        arch_tags_json = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{stripped_num}_tags.json")
                    elif os.path.exists(fallback_image_2):
                        arch_crop_image = fallback_image_2
                        arch_tags_json = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_{padded_num}_tags.json")
            else:
                # Fallback for when arch_ref is just a sheet name (e.g., "A_1.11") without drawing number
                sheet_name = clean_sheet_name(view.arch_ref)
                arch_crop_image = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}.png")
                arch_tags_json = os.path.join(project_root, "arch_crop", sheet_name, f"{sheet_name}_tags.json")
                
            # 2. Extract Visible Tags from Arch Drawing (Done once per arch view)
            # Only do this if at least one rule requires the v1 flow (which uses these tags)
            needs_arch_tags = any(getattr(r, "validation_flow", "arch_vs_submittal_v1") == "arch_vs_submittal_v1" for r in rules)
            if needs_arch_tags and not os.path.exists(arch_tags_json):
                extract_visible_tags(arch_crop_image, arch_tags_json)
        
        for rule in rules:
            path_parts = os.path.normpath(view.source_file).split(os.sep)
            submittal_id = path_parts[-2] if len(path_parts) >= 2 else "UnknownSubmittal"
            view_sheet_id = os.path.splitext(path_parts[-1])[0]
            rule_output_dir = os.path.join(output_dir, submittal_id, view_sheet_id, rule.rule_id)

            validation_flow = getattr(rule, "validation_flow", "arch_vs_submittal_v1")

            if validation_flow == "arch_vs_submittal_v1":
                if "Architectural Drawing" in rule.required_sources:
                    result = execute_arch_rule(
                        view=view,
                        rule=rule,
                        arch_tags_path=arch_tags_json,
                        schedule_path=schedule_json_path
                    )
                else:
                    print(f"Skipping {rule.rule_id}: Alternative non-arch flow not implemented yet.")
                    continue

                markup_path = ""
                if result.needs_markup:
                    markup_path = apply_markup(view, result, rule, rule_output_dir)
                    
                arch_view_title = "Unknown View"
                if os.path.exists(arch_tags_json):
                    with open(arch_tags_json, 'r') as f:
                        arch_tags_data = json.load(f)
                        arch_view_title = arch_tags_data.get("view_title") or "Unknown View"
                        
                save_result(
                    submittal_id=submittal_id,
                    view_sheet_id=view_sheet_id, 
                    rule=rule, 
                    result=result, 
                    markup_path=markup_path, 
                    view_metadata=view,
                    arch_view_title=arch_view_title,
                    output_base_dir=output_dir
                )

            elif validation_flow == "arch_vs_submittal_v2":
                print(f"Executing {rule.rule_id} for View {view.view_name} using arch_vs_submittal_v2...")
                run_v2_pipeline(view.source_file, arch_crop_image, rule.model_dump(), rule_output_dir, view.arch_ref)

            elif validation_flow == "direct_submittal":
                print(f"Executing {rule.rule_id} for View {view.view_name} using submittal.py (direct_submittal)...")
                
                # Send the exact view from our JSON to bypass AI view guessing
                cached_views = [{
                    'view_id': 'V1',
                    'detected_view_type': view.view_name,
                    'view_type_classification': view.view_type,
                    'confidence_score': 100,
                    'scope_coordinates': {'x1': 0, 'y1': 0, 'x2': 10000, 'y2': 10000}
                }]
                
                run_submittal_pipeline(view.source_file, rule.model_dump(), rule_output_dir, cached_views=cached_views)
            
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
