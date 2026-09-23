from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# Rule Definition Models (Based strictly on microwave.json)
# ---------------------------------------------------------

class ViewApplicability(BaseModel):
    allowed_view_types: List[str]
    view_matching_logic: str

class ArchReference(BaseModel):
    required: bool

class ScopeIdentification(BaseModel):
    method: str
    schedule_name: str
    match_field: str
    search_keywords: Optional[List[str]] = None

class ArchScopeValidation(BaseModel):
    match_type: str
    description: str

class SubmittalValidation(BaseModel):
    validation_type: str
    expected_notes: Optional[List[str]] = None
    match_type: str

class DecisionLogic(BaseModel):
    arch_found_submittal_found: str
    arch_found_submittal_not_found: str
    arch_not_found_submittal_found: str
    arch_not_found_submittal_not_found: str

class ExecutionFlow(BaseModel):
    check_view_applicability: bool
    check_arch_scope: bool
    check_submittal: bool

class ConfidenceScore(BaseModel):
    minimum_pass_threshold: float

class MarkupInfo(BaseModel):
    enabled: bool
    statuses: List[str]

class OutputRequired(BaseModel):
    confidence_score: bool
    reasoning: bool
    coordinates: bool
    detected_text: bool

class RuleDefinition(BaseModel):
    rule_id: str
    rule_name: str
    common_rule_group: Optional[str] = None
    active_code_tag: Optional[str] = None
    active_description: Optional[str] = None
    validation_flow: Optional[str] = "arch_vs_submittal_v1"
    type_of_rule: str
    category: str
    discipline: str
    required_sources: List[str]
    view_applicability: Optional[ViewApplicability] = None
    arch_reference: Optional[ArchReference] = None
    scope_identification: Optional[ScopeIdentification] = None
    arch_scope_validation: Optional[ArchScopeValidation] = None
    submittal_validation: Optional[SubmittalValidation] = None
    decision_logic: Optional[DecisionLogic] = None
    execution_flow: Optional[ExecutionFlow] = None
    confidence_score: Optional[ConfidenceScore] = None
    markup: Optional[MarkupInfo] = None
    severity: Optional[str] = None
    priority: Optional[str] = None
    output_required: Optional[OutputRequired] = None
    output_json_format: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

# ---------------------------------------------------------
# Input Pipeline Models
# ---------------------------------------------------------

class SubmittalMetadataItem(BaseModel):
    view_name: str
    arch_ref: str
    view_type: str
    source_file: str

class ScheduleItem(BaseModel):
    tag: str
    description: Optional[str] = None
    # We allow extra fields since Excel columns can vary
    model_config = {"extra": "allow"}

class VisibleTag(BaseModel):
    tag: str
    bbox: List[int] # [x1, y1, x2, y2]
    nearby_text: Optional[str] = ""

class VisibleTagsOutput(BaseModel):
    view_title: Optional[str] = None
    visible_tags: List[VisibleTag]

# ---------------------------------------------------------
# Evaluation Result Models
# ---------------------------------------------------------

class BoundingBox(BaseModel):
    left: int
    top: int
    right: int
    bottom: int

class DetectedEntity(BaseModel):
    entity_name: str
    detected_text: str
    coordinates: BoundingBox

class ValidationResultFlags(BaseModel):
    arch_scope_found: bool
    submittal_note_found: bool

class ViewResult(BaseModel):
    view_id: str
    view_type: str
    view_applicable: bool
    arch_scope_applicable: bool
    active_code_tag: Optional[str] = None
    required_tags: List[str] = Field(default_factory=list)
    matched_tags: List[str] = Field(default_factory=list)
    validation_result: ValidationResultFlags
    status: str
    confidence_score: float = 0.0
    reasoning: str = ""
    detected_entities: List[DetectedEntity] = Field(default_factory=list)
    
    # Internal flag to signal the orchestrator to draw markup
    needs_markup: bool = False

class DrawingContext(BaseModel):
    arch_reference: str
    architectural_view_name: str
    submittal_view_name: str

class RuleEvaluationResult(BaseModel):
    rule_id: str
    rule_name: str
    active_code_tag: Optional[str] = None
    active_description: Optional[str] = None
    sheet_status: str
    drawing_context: DrawingContext
    view_results: List[ViewResult]
    markup_required: bool
    severity: str
    priority: str
