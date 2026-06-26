import json
import os
from PIL import Image

def crop_detail_blocks(image_path="page1_300dpi_blocks.png", json_path="detail_blocks_300dpi.json", out_dir="cropped_blocks"):
    # Create output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Loading 300 DPI image from {image_path}...")
    img = Image.open(image_path)

    print(f"Loading coordinates from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    print(f"Cropping {len(blocks)} detail blocks...")
    
    for i, block in enumerate(blocks):
        # The full bounding box is [x0, y0, x1, y1]
        bbox = block.get("full_block_bbox_300dpi")
        if not bbox:
            continue
        
        # We add a 30-pixel padding so the text isn't right on the very edge of the image
        padding = 30
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img.width, bbox[2] + padding)
        y1 = min(img.height, bbox[3] + padding)
        
        # Crop the image using Pillow
        cropped_img = img.crop((x0, y0, x1, y1))
        
        # Clean up the drawing number to be used in a filename
        safe_num = "".join(c for c in block['number_text'] if c.isalnum())
        
        out_filename = os.path.join(out_dir, f"detail_{safe_num}.png")
        cropped_img.save(out_filename)
        print(f"Saved: {out_filename}")

    print(f"\n✅ All {len(blocks)} blocks cropped successfully into the '{out_dir}' folder!")
    print("You can now feed these individual images into Qwen (or any other Vision Model).")

if __name__ == "__main__":
    crop_detail_blocks()
