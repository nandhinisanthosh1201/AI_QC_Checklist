import os
from src.models.data_models import ViewResult, SubmittalMetadataItem, RuleDefinition
from src.utils.image_utils import draw_markup

def apply_markup(view: SubmittalMetadataItem, result: ViewResult, rule: RuleDefinition, output_dir: str) -> str:
    """
    Applies markup (bounding box) to the submittal image if the rule execution
    resulted in FAIL or REVIEW_REQUIRED.
    Saves the markup image in the rule's output folder.
    Returns the path to the saved markup image.
    """
    if not result.needs_markup:
        return ""
        
    print(f"Applying markup for View: {view.view_name}, Rule: {result.status}")
    
    markup_filename = "markup.png"
    markup_output_path = os.path.join(output_dir, markup_filename)
    
    bbox = [0, 0, 0, 0]
    if result.detected_entities:
        entity = result.detected_entities[0]
        bbox = [
            entity.coordinates.left,
            entity.coordinates.top,
            entity.coordinates.right,
            entity.coordinates.bottom
        ]
    
    # If bbox is [0,0,0,0] or empty (not found), draw a box around the entire view to indicate a missing note
    if bbox == [0, 0, 0, 0]:
        print(f"Note missing. Drawing full-border alert on {markup_filename}")
        # Get image dimensions to draw a full border
        try:
            import cv2
            img = cv2.imread(view.source_file)
            if img is not None:
                h, w, _ = img.shape
                # Leave a small 10-pixel margin
                bbox = [10, 10, w - 10, h - 10]
            else:
                return ""
        except Exception as e:
            print(f"Failed to load image for default bbox: {e}")
            return ""
        
    try:
        draw_markup(
            image_path=view.source_file,
            output_path=markup_output_path,
            bbox=bbox,
            status=result.status,
            rule_name=rule.rule_name
        )
        return markup_output_path
    except Exception as e:
        print(f"Failed to apply markup: {e}")
        return ""
