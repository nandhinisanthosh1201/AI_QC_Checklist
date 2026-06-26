import fitz  # PyMuPDF
import json

def generate_image_with_bboxes(pdf_path, json_path, out_image="page1_with_boxes.png"):
    print(f"Loading PDF from {pdf_path}...")
    doc = fitz.open(pdf_path)
    page = doc[0]  # First page
    
    print("Loading extracted JSON data...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("Drawing bounding boxes for scale text...")
    
    keywords = ["scale", "1/4", "1/8", "1'-", "nts", "n.t.s", "=\""]
    box_count = 0
    
    # The first page data is at index 0 in the json
    page_data = data[0]
    for item in page_data.get("items", []):
        text = item.get("text", "")
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in keywords):
            bbox = item.get("bbox")
            if bbox:
                # Create a rectangle object
                rect = fitz.Rect(*bbox)
                
                # Draw a red rectangle (color is RGB: 1, 0, 0)
                page.draw_rect(rect, color=(1, 0, 0), width=2)
                box_count += 1
                
    print(f"Drew {box_count} red boxes on the page.")
    
    print("Rendering page to image...")
    # The matrix determines the resolution. 2.0 zoom gives a higher quality image
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    
    pix.save(out_image)
    print(f"✅ Image saved successfully as: {out_image}")

if __name__ == "__main__":
    pdf_path = r"D:\Drawing_Extraction\DGS_Arch-37.pdf"
    json_path = "vector_text_output.json"
    generate_image_with_bboxes(pdf_path, json_path)
