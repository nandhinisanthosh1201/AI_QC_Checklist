# Dynamic Schedule-Based View Validation (Implementation Plan)

This plan outlines the changes required to transform our Schedule-based JSON rules into pure templates, shifting the control of "Which view to check?" entirely to the `schedule.json` data.

## Context: Current Execution Flows
- **ADA.json** → Executed via `arch_vs_submittal_v3` (Matrix Logic).
- **Sanitary / Display / Kitchen Schedule rules** → Executed via `arch_vs_submittal_v1` (Standard AI Tag mapping).

## Proposed Changes

### 1. Template Conversion (JSON Rules)
We will convert the 3 generic Schedule JSON rules into "Pure Templates" by clearing out the hardcoded default views.

**Files to modify:**
- `rules/Sanitary_accessory_schedule.json`
- `rules/Display_systems_schedule.json`
- `rules/kitchen_equipment_schedule.json`

**Action:**
Set `allowed_view_types` to an empty list `[]` to act as a blank template waiting for schedule data.
```json
"view_applicability": {
    "allowed_view_types": []
}
```

### 2. Schedule Row Overrides (`src/main.py`)
We will modify the `resolve_active_rule` function in `src/main.py`. This function is the exact junction where the `schedule.json` row meets the `base_rule` copy.

**Action:**
Extract the `views_to_check` array from the schedule row. If it exists, override the rule's `allowed_view_types`. 
If it is empty or missing, log a clear warning message and let the `allowed_view_types` remain empty (which means no views will be processed for that specific tag).

```python
#### [MODIFY] src/main.py (inside resolve_active_rule)
views_override = schedule_item.get("views_to_check")
if views_override and isinstance(views_override, list):
    # Override with Schedule logic
    rule.view_applicability["allowed_view_types"] = views_override
else:
    print(f"WARNING: No 'views_to_check' found in schedule for tag {tag}. Skipping execution for this item.")
```

## Verification Plan
1. Edit a dummy `schedule.json` locally to give one item `["Plan"]` and another item `["Elevation", "Section"]`.
2. Run the pipeline with one of the schedule rules (e.g., `Sanitary_accessory_schedule.json`).
3. Verify via console logs that the AI dynamically skips or executes views exactly according to the schedule's `views_to_check` array, and prints a warning if missing.
