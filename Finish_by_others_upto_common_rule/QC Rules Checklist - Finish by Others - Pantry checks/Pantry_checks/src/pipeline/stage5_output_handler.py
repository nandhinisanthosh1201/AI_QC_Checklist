import os
import json
from typing import Any
from src.models.data_models import ViewResult, RuleEvaluationResult, DrawingContext, RuleDefinition

def save_result(submittal_id: str, view_sheet_id: str, rule: RuleDefinition, result: ViewResult, markup_path: str, view_metadata: Any, arch_view_title: str, output_base_dir: str = "Output") -> None:
    """
    Saves the final result to Output/{submittal_id}/{view_sheet_id}/{rule_id}/result.json
    """
    # Create directory structure
    rule_dir = os.path.join(output_base_dir, submittal_id, view_sheet_id, rule.rule_id)
    os.makedirs(rule_dir, exist_ok=True)
    
    # Construct final JSON output matching the requested schema in microwave.json
    final_output = RuleEvaluationResult(
        rule_id=rule.rule_id,
        rule_name=rule.rule_name,
        sheet_status=result.status,
        drawing_context=DrawingContext(
            arch_reference=view_metadata.arch_ref,
            architectural_view_name=arch_view_title,
            submittal_view_name=result.view_id
        ),
        view_results=[result],
        markup_required=result.needs_markup,
        severity=rule.severity,
        priority=rule.priority
    )
    
    # If markup was saved, we might want to include the path in the output JSON.
    # The schema doesn't strictly have a markup_path field, but we can dump it.
    output_dict = final_output.model_dump()
    if markup_path:
        output_dict["markup_path"] = markup_path
        
    result_json_path = os.path.join(rule_dir, "result.json")
    
    with open(result_json_path, "w") as f:
        json.dump(output_dict, f, indent=4)
        
    print(f"Result saved to {result_json_path}")
