import os
import json
from typing import List, Dict, Any
from src.models.data_models import (
    RuleDefinition, 
    SubmittalMetadataItem, 
    ViewResult, 
    ValidationResultFlags,
    DetectedEntity,
    BoundingBox
)
from src.utils.llm_client import call_vision_llm

def load_json(filepath: str) -> Any:
    with open(filepath, 'r') as f:
        return json.load(f)

def run_gate_1_view_applicability(view_type: str, allowed_types: List[str]) -> bool:
    """Gate 1: Check if the submittal view type is in the rule's allowed view types."""
    return view_type.lower() in [t.lower() for t in allowed_types]

def run_gate_2_arch_scope(rule: RuleDefinition, arch_tags_path: str, schedule_path: str, target_tag: str = None, target_desc: str = None) -> tuple[bool, List[str], List[str], List[str]]:
    """
    Gate 2: Intersect required tags from schedule with visible tags in the architectural view.
    Returns: (is_applicable, required_tags, matched_tags, matched_descriptions)
    """
    required_tags = []
    required_descriptions = {} # tag -> description

    # If the rule has already been flattened with an active_code_tag, use it directly!
    if getattr(rule, "active_code_tag", None):
        import re
        split_tags = [t.strip() for t in re.split(r'[,/]', str(rule.active_code_tag)) if t.strip()]
        required_tags = split_tags
        if getattr(rule, "active_description", None):
            for t in split_tags:
                required_descriptions[t.upper()] = rule.active_description
    else:
        # Fallback to older lookup behavior for non-flattened rules
        # 1. Extract required tags from Schedule
        schedule_data = load_json(schedule_path)
        
        if rule.scope_identification:
            schedule_name = rule.scope_identification.schedule_name
            match_field = rule.scope_identification.match_field
            keywords = [kw.lower() for kw in rule.scope_identification.search_keywords] if rule.scope_identification.search_keywords else []
            
            # Fallback logic for sheet name mismatch
            target_schedule = []
            if schedule_name in schedule_data:
                target_schedule = schedule_data[schedule_name]
            elif len(schedule_data) > 0:
                first_key = list(schedule_data.keys())[0]
                target_schedule = schedule_data[first_key]
                
            for item in target_schedule:
                # Use case insensitive matching for dictionary keys
                item_lower_keys = {k.lower(): v for k, v in item.items()}
                
                # If target_tag or target_desc is specified (UI / targeted execution), filter down!
                extracted_tag = item_lower_keys.get("tag") or item_lower_keys.get("code / tag")
                extracted_desc = item_lower_keys.get("description", "")
                
                if target_tag and str(extracted_tag).strip().upper() != str(target_tag).strip().upper():
                    continue # Skip items that are not our target tag
                if target_desc and str(extracted_desc).strip().upper() != str(target_desc).strip().upper():
                    continue # Skip items that are not our target description
                    
                field_value = item_lower_keys.get(match_field.lower(), "")
                # Match if we have keywords, or if we have no keywords but we have a target_tag or target_desc
                if (keywords and any(kw in str(field_value).lower() for kw in keywords)) or (not keywords and (target_tag or target_desc)):
                    if extracted_tag:
                        import re
                        split_tags = [t.strip() for t in re.split(r'[,/]', str(extracted_tag)) if t.strip()]
                        required_tags.extend(split_tags)
                        for t in split_tags:
                            required_descriptions[t.upper()] = item_lower_keys.get("description", "")
                        
        if not required_tags:
            # If no such items in schedule, rule might not apply to this project
            return False, required_tags, [], []
        
    # 2. Extract visible tags from Arch Crop
    if not os.path.exists(arch_tags_path):
        return False, required_tags, [], []
        
    arch_tags_data = load_json(arch_tags_path)
    visible_tags = [t.get("tag") for t in arch_tags_data.get("visible_tags", [])]
    
    # 3. Intersection (Case Insensitive)
    required_tags_upper = [str(t).strip().upper() for t in required_tags]
    
    req_descriptions_upper = []
    for t in required_tags:
        desc = required_descriptions.get(t.upper(), "")
        if desc:
            desc_upper = str(desc).strip().upper()
            if desc_upper not in req_descriptions_upper:
                req_descriptions_upper.append(desc_upper)
            
    visible_tags_upper = [str(t).strip().upper() for t in visible_tags if t]
    
    # Exact match for tags
    matched_tags = list(set(required_tags_upper) & set(visible_tags_upper))
    
    # Substring match for tags >= 4 chars (if not already matched)
    for req_tag in required_tags_upper:
        if req_tag not in matched_tags and len(req_tag) >= 4:
            for v_tag in visible_tags_upper:
                if req_tag in v_tag:
                    matched_tags.append(req_tag)
                    break
    
    # Match for descriptions (Exact first, then Substring)
    for desc_upper in req_descriptions_upper:
        if desc_upper in visible_tags_upper and desc_upper not in matched_tags:
            matched_tags.append(desc_upper)
            continue
            
        # Substring match for descriptions >= 4 chars to prevent false positives on short words
        if desc_upper not in matched_tags and len(desc_upper) >= 4:
            for v_tag in visible_tags_upper:
                if desc_upper in v_tag:
                    matched_tags.append(desc_upper)
                    break
    matched_descriptions = [required_descriptions[t] for t in matched_tags if t in required_descriptions]
    
    # If any matched tags found, scope is applicable
    is_applicable = len(matched_tags) > 0
    
    return is_applicable, required_tags, matched_tags, matched_descriptions


def run_gate_3_submittal_validation(rule: RuleDefinition, submittal_image_path: str) -> dict:
    """
    Gate 3: Call Vision LLM to check for the required note in the submittal drawing.
    """
    expected_notes = []
    if rule.submittal_validation and rule.submittal_validation.expected_notes:
        expected_notes.extend(rule.submittal_validation.expected_notes)
        
    if not expected_notes:
        expected_notes = ["NOT SPECIFIED (Fallback)"]
        
    notes = ", ".join(expected_notes)
    
    prompt = f"""
    You are a QC inspector for architectural submittals.
    Please scan this submittal drawing. 
    TASK: Check if the exact text of ANY of the following Expected Notes is present in the drawing.
    Expected Notes: {notes}
    
    RULES:
    1. If you find the EXACT literal text (e.g. "ABC BY OTHERS" or "XYZ BY OTHERS") from the Expected Notes list in the drawing, you MUST set "note_found": true. Minor spacing, punctuation, or multi-line breaks (e.g. "ABC\nBY OTHERS" or "10 28 23. UM" vs "10 28 23.UM") STILL count as an exact match. Do NOT fail for minor space differences.
    2. Do NOT accept completely different items. If the drawing has "UNRELATED ITEM BY OTHERS", it does NOT match "ABC BY OTHERS".
    3. NO ABBREVIATION EXPANSION: Do NOT attempt to guess what an abbreviation stands for. If the Expected Note is "ABC BY OTHERS", you MUST find the literal letters A-B-C. Finding "ALPHABET BY OTHERS" is a FAILURE. For example, do not assume "FF" means "Finish" and accept "WALL FINISH". You must find the exact string.
    4. If none of the Expected Notes are found literally, set "note_found": false.
    
    OUTPUT FORMAT (JSON):
    {{
      "reasoning": "Brief explanation of what you found and whether it matches.",
      "note_found": true/false,
      "detected_text": "The exact text you found (leave empty if false)",
      "bbox": [x1, y1, x2, y2], 
      "confidence": 0.0 to 100.0
    }}
    
    IMPORTANT for bbox: Provide coordinates in a 1000x1000 normalized scale, where [0,0] is top-left and [1000,1000] is bottom-right.
    If 'note_found' is false, try to find the physical object itself in the drawing and return its 'bbox' instead! Only set 'bbox' to [0,0,0,0] if both the note AND the physical object are entirely missing.
    Do not include any markdown formatting like ```json in your response.
    """
    
    try:
        response_text = call_vision_llm(prompt=prompt, image_path=submittal_image_path, response_format="json_object")
        
        # Clean response
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json").strip("```").strip()
        elif response_text.startswith("```"):
            response_text = response_text.strip("```").strip()
            
        return json.loads(response_text)
    except Exception as e:
        print(f"Error in Submittal Validation LLM call: {e}")
        return {
            "note_found": False,
            "detected_text": "",
            "bbox": [0,0,0,0],
            "reasoning": f"LLM Error: {str(e)}",
            "confidence": 0.0
        }

def resolve_decision(rule: RuleDefinition, arch_scope_found: bool, submittal_note_found: bool) -> str:
    """Resolve the final status based on the decision logic matrix in the rule."""
    logic = rule.decision_logic
    
    if arch_scope_found and submittal_note_found:
        return logic.arch_found_submittal_found
    elif arch_scope_found and not submittal_note_found:
        return logic.arch_found_submittal_not_found
    elif not arch_scope_found and submittal_note_found:
        return logic.arch_not_found_submittal_found
    else:
        return logic.arch_not_found_submittal_not_found

def execute_arch_rule(view: SubmittalMetadataItem, rule: RuleDefinition, arch_tags_path: str, schedule_path: str, target_tag: str = None, target_desc: str = None) -> ViewResult:
    """
    Executes the 3-Gate Rule Engine for a single submittal view and a single rule,
    assuming 'Architectural Drawing' is in required_sources.
    """
    print(f"Executing Rule {rule.rule_id} for View {view.view_name}...")
    
    # Init Result
    result = ViewResult(
        view_id=view.view_name,
        view_type=view.view_type,
        view_applicable=False,
        arch_scope_applicable=False,
        active_code_tag=getattr(rule, "active_code_tag", None),
        validation_result=ValidationResultFlags(arch_scope_found=False, submittal_note_found=False),
        status="OMITTED"
    )
    
    # Gate 1: View Applicability
    if rule.execution_flow.check_view_applicability:
        if not run_gate_1_view_applicability(view.view_type, rule.view_applicability.allowed_view_types):
            result.reasoning = f"View type '{view.view_type}' not in allowed types: {rule.view_applicability.allowed_view_types}"
            return result
    
    result.view_applicable = True
    
    # If the Architectural Reference is NA, manual review is required
    if view.arch_ref == "NA":
        result.status = "REVIEW_REQUIRED"
        result.reasoning = "Architectural Reference is NA. Manual review required."
        return result
        
    # Gate 2: Arch Scope
    matched_descriptions = []
    is_applicable = False
    if rule.execution_flow.check_arch_scope:
        is_applicable, required_tags, matched_tags, matched_descriptions = run_gate_2_arch_scope(rule, arch_tags_path, schedule_path, target_tag, target_desc)
        result.required_tags = required_tags
        result.matched_tags = matched_tags
        result.validation_result.arch_scope_found = is_applicable
        
        if not is_applicable:
            can_early_exit = True
            if rule.decision_logic:
                if rule.decision_logic.arch_not_found_submittal_found != "OMITTED" or \
                   rule.decision_logic.arch_not_found_submittal_not_found != "OMITTED":
                    can_early_exit = False
            
            if can_early_exit:
                result.status = "OMITTED"
                result.reasoning = "No matching tags found in Architectural Scope (Early Exit)."
                return result
                
    result.arch_scope_applicable = is_applicable
            
    # Gate 3: Submittal Validation
    if rule.execution_flow.check_submittal:
        llm_output = run_gate_3_submittal_validation(rule, view.source_file)
        
        result.validation_result.submittal_note_found = llm_output.get("note_found", False)
        result.confidence_score = llm_output.get("confidence", 0.0)
        result.reasoning = llm_output.get("reasoning", "")
        
        if result.validation_result.submittal_note_found:
            bbox = llm_output.get("bbox", [0,0,0,0])
            result.detected_entities.append(DetectedEntity(
                entity_name=f"{rule.rule_name} Note",
                detected_text=llm_output.get("detected_text", ""),
                coordinates=BoundingBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
            ))
            
    # Resolve Decision
    result.status = resolve_decision(
        rule, 
        result.validation_result.arch_scope_found, 
        result.validation_result.submittal_note_found
    )
    
    # Check if Markup is needed
    if rule.markup.enabled and result.status in rule.markup.statuses:
        result.needs_markup = True
        
    return result
