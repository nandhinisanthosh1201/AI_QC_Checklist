import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Custom High-Visibility CAD Color Palette (BGR format)
CLASS_COLORS = {
    'Cabinet':  (255, 120, 0),    # Bright Cyan / Blue
    'Soffit':   (0, 215, 255),    # Vibrant Amber / Gold
    'Toe kick': (0, 0, 255),      # Bright Vivid Red / Crimson (High Contrast)
    'Panel':    (40, 200, 40)     # Neon Emerald Green
}

CLASS_TEXT_COLORS = {
    'Cabinet':  (255, 255, 255),
    'Soffit':   (0, 0, 0),
    'Toe kick': (255, 255, 255),
    'Panel':    (255, 255, 255)
}

def draw_custom_boxes(img, boxes, class_names):
    """
    Renders clean, high-contrast bounding boxes with crisp semi-transparent labels.
    """
    annotated = img.copy()
    class_counts = {}
    
    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls_id >= len(class_names):
            continue
        
        cls_name = class_names[cls_id]
        
        # fallback if name is slightly different case
        mapped_name = cls_name
        if cls_name.lower() == 'panel':
            mapped_name = 'Panel'
        elif cls_name.lower() == 'toe kick':
            mapped_name = 'Toe kick'
            
        class_counts[mapped_name] = class_counts.get(mapped_name, 0) + 1
        instance_name = f"{mapped_name} {class_counts[mapped_name]}"
            
        color = CLASS_COLORS.get(mapped_name, (0, 255, 0))
        text_color = CLASS_TEXT_COLORS.get(mapped_name, (255, 255, 255))
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # 1. Draw crisp bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        
        # 2. Draw label tag badge
        label = f"{instance_name} {conf*100:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        tag_y1 = max(0, y1 - th - 6)
        tag_y2 = y1
        tag_x1 = x1
        tag_x2 = x1 + tw + 6
        
        # Draw tag background
        cv2.rectangle(annotated, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        # Draw label text
        cv2.putText(annotated, label, (tag_x1 + 3, tag_y2 - 3), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
    return annotated

def detect_components(input_dir, output_dir, model_path):
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)
    
    # We retrieve the actual class names from the model
    class_names = model.names
    print(f"Model Class Names: {class_names}")
    
    # Converting dict to list if needed
    if isinstance(class_names, dict):
        class_names = [class_names[i] for i in range(len(class_names))]

    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_files.sort()

    print("\n" + "=" * 80)
    print(f" PROCESSING FOLDER: {input_dir} (Total: {len(img_files)} drawings)")
    print("=" * 80)

    total_counts = {cls: 0 for cls in class_names}
    
    components_data = []

    for idx, fname in enumerate(img_files, 1):
        img_path = os.path.join(input_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model.predict(
            source=img_path,
            imgsz=1024,
            conf=0.25,      # Lowered to 0.25 to catch low-confidence cabinets
            iou=0.40,       # Decreased from 0.65 to remove overlapping boxes
            augment=False,  # Disabled to prevent duplicate bounding boxes
            device='0' if torch.cuda.is_available() else 'cpu',
            save=False,
            verbose=False
        )

        res = results[0]
        boxes = res.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            if cls_id < len(class_names):
                total_counts[class_names[cls_id]] += 1
                
                # Save coordinate data for Step 6
                cls_name = class_names[cls_id]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                components_data.append({
                    "file": fname,
                    "class": cls_name,
                    "confidence": conf,
                    "x0": x1,
                    "y0": y1,
                    "x1": x2,
                    "y1": y2
                })

        annotated = draw_custom_boxes(img, boxes, class_names)
        out_path = os.path.join(output_dir, f"detected_{fname}")
        cv2.imwrite(out_path, annotated)
        print(f"Processed: {fname}")

    # Save the components JSON for Step 6
    import json
    json_path = os.path.join(output_dir, "components_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(components_data, f, indent=4)

    print(f"\n✓ Completed Processing!")
    for cls_name, count in total_counts.items():
        print(f"  • {cls_name}: {count}")
    print(f"  • Saved in  : {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run Component Detection on Cropped Images')
    parser.add_argument('--input', type=str, required=True, help='Path to the input folder containing cropped images')
    parser.add_argument('--output', type=str, default='detected_components_output', help='Path to the output folder')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights = os.path.join(base_dir, "best_production_model.pt")
    
    # Make input and output paths absolute relative to CWD
    in_dir = os.path.abspath(args.input)
    out_dir = os.path.abspath(args.output)
    
    detect_components(in_dir, out_dir, model_weights)
