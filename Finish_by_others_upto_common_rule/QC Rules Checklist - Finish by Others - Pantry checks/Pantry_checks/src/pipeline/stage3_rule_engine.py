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

def run_gate_2_arch_scope(rule: RuleDefinition, arch_tags_path: str, schedule_path: str) -> tuple[bool, List[str], List[str]]:
    """
    Gate 2: Intersect required tags from schedule with visible tags in the architectural view.
    Returns: (is_applicable, required_tags, matched_tags)
    """
    # 1. Extract required tags from Schedule
    schedule_data = load_json(schedule_path)
    schedule_name = rule.scope_identification.schedule_name
    match_field = rule.scope_identification.match_field
    keywords = [kw.lower() for kw in rule.scope_identification.search_keywords]
    
    required_tags = []
    if schedule_name in schedule_data:
        for item in schedule_data[schedule_name]:
            # Use case insensitive matching for dictionary keys to handle 'Description' vs 'description'
            item_lower_keys = {k.lower(): v for k, v in item.items()}
            
            field_value = item_lower_keys.get(match_field.lower(), "")
            if field_value and any(kw in str(field_value).lower() for kw in keywords):
                # Handle 'tag' or 'code / tag'
                extracted_tag = item_lower_keys.get("tag") or item_lower_keys.get("code / tag")
                if extracted_tag:
                    import re
                    split_tags = [t.strip() for t in re.split(r'[,/]', str(extracted_tag)) if t.strip()]
                    required_tags.extend(split_tags)
                
    if not required_tags:
        # If no such items in schedule, rule might not apply to this project
        return False, required_tags, []
        
    # 2. Extract visible tags from Arch Crop
    if not os.path.exists(arch_tags_path):
        return False, required_tags, []
        
    arch_tags_data = load_json(arch_tags_path)
    visible_tags = [t.get("tag") for t in arch_tags_data.get("visible_tags", [])]
    
    # 3. Intersection (Case Insensitive)
    required_tags_upper = [str(t).strip().upper() for t in required_tags]
    visible_tags_upper = [str(t).strip().upper() for t in visible_tags if t]
    
    matched_tags = list(set(required_tags_upper) & set(visible_tags_upper))
    
    # If any matched tags found, scope is applicable
    is_applicable = len(matched_tags) > 0
    
    return is_applicable, required_tags, matched_tags


def run_gate_3_submittal_validation(rule: RuleDefinition, submittal_image_path: str) -> dict:
    """
    Gate 3: Call Vision LLM to check for the required note in the submittal drawing.
    """
    notes = ", ".join(rule.submittal_validation.expected_notes)
    
    prompt = f"""
    You are a QC inspector for architectural submittals.
    Please scan this submittal drawing. 
    Check if the following note (or a very close semantic match/abbreviation) is present:
    Expected Notes: {notes}
    
    Return the output strictly in JSON format matching this schema:
    {{
      "note_found": true/false,
      "detected_text": "The exact text you found",
      "bbox": [x1, y1, x2, y2], 
      "reasoning": "Brief explanation of why you decided true or false",
      "confidence": 0.0 to 100.0
    }}
    
    If 'note_found' is false, you can set 'bbox' to [0,0,0,0] and 'detected_text' to "".
    Do not include any markdown formatting like ```json in your response, just the raw JSON object.
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

def execute_arch_rule(view: SubmittalMetadataItem, rule: RuleDefinition, arch_tags_path: str, schedule_path: str) -> ViewResult:
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
    if rule.execution_flow.check_arch_scope:
        is_applicable, required_tags, matched_tags = run_gate_2_arch_scope(rule, arch_tags_path, schedule_path)
        result.required_tags = required_tags
        result.matched_tags = matched_tags
        result.validation_result.arch_scope_found = is_applicable
        
        # We don't return early here, because the decision logic might dictate REVIEW_REQUIRED 
        # if arch scope is missing BUT the note is present in the submittal.
        # BUT the plan says: "Intersection. If no match -> OMITTED. This prevents checking microwave in every PLAN."
        # Wait, user's decision logic: arch_not_found_submittal_not_found = OMITTED. 
        # arch_not_found_submittal_found = REVIEW_REQUIRED. 
        # So we actually DO need to check the submittal even if arch scope is missing? 
        # User said: "If no intersection -> OMITTED. Reason: Microwave is not used in this Arch View. This prevents checking microwave in every PLAN."
        # So if we skip LLM call here, we can't detect "arch_not_found_submittal_found".
        # Let's optimize: If they want to skip LLM call to save cost when arch scope is false, 
        # we just return OMITTED. But the user decision matrix handles all 4 cases.
        # Let's execute Submittal Validation regardless to fulfill the truth table, or 
        # strictly follow the 'OMITTED early exit' mentioned in the plan description.
        # Let's do early exit to save LLM calls, but if they want the full matrix, we must call it.
        # I'll implement early exit if they want, but since they have a matrix, let's call LLM.
        # Wait, user explicitly said:
        # "Step 3: If no intersection -> OMITTED... prevents checking microwave in every PLAN."
        # This means we skip Gate 3. So `arch_not_found_submittal_found` is a theoretical edge case, 
        # but the early exit overrides it. Let's implement early exit.
        if not is_applicable:
            # Smart Early Exit: Check if we can safely skip Gate 3 based on the decision matrix!
            # If BOTH outcomes when arch is NOT found lead to "OMITTED", then checking submittal is useless.
            # We can skip Gate 3 and save API calls!
            can_early_exit = True
            if rule.decision_logic:
                if rule.decision_logic.arch_not_found_submittal_found != "OMITTED" or \
                   rule.decision_logic.arch_not_found_submittal_not_found != "OMITTED":
                    can_early_exit = False
            
            if can_early_exit:
                result.status = "OMITTED"
                result.reasoning = "No matching tags found in Architectural Scope (Early Exit)."
                return result
            else:
                # We cannot early exit! We must continue to Gate 3 to check if submittal has it.
                pass
                
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
