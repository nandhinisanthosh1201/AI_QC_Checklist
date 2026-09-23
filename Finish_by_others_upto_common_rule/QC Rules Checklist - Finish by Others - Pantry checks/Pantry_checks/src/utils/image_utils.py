import cv2
import os

def draw_markup(image_path: str, output_path: str, bbox: list, status: str, rule_name: str = ""):
    """
    Draws a rectangle on the image based on the status.
    bbox: [x1, y1, x2, y2]
    status: 'FAIL' (Red) or 'REVIEW_REQUIRED' (Yellow)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Define colors in BGR
    color = (0, 0, 255) # Default Red for FAIL
    if status == "REVIEW_REQUIRED":
        color = (0, 255, 255) # Yellow

    # Extract coordinates
    x1, y1, x2, y2 = bbox
    
    # Draw Rectangle (Thickness = 3)
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    
    # Put Text Label
    display_text = f"{status} - {rule_name}" if rule_name else status
    cv2.putText(img, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Markup saved to {output_path}")

