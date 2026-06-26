import fitz
import json

def extract_detail_blocks(pdf_path, json_path, out_image="page1_detail_blocks.png"):
    print("Loading extracted JSON data...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Keywords for finding scale text
    keywords = ["scale:", "1/4\"", "1/8\"", "1'-", "nts", "n.t.s", "=\""]
    
    page_data = data[0]
    items = page_data.get("items", [])
    
    # 1. Identify all Scale texts
    scales = []
    for item in items:
        text = item.get("text", "")
        if any(kw in text.lower() for kw in keywords):
            # Ignore long paragraphs that just happen to contain the word "scale"
            if len(text) < 30:
                scales.append(item)
                
    print(f"Found {len(scales)} scale labels.")
    
    blocks = []
    for scale in scales:
        sx0, sy0, sx1, sy1 = scale["bbox"]
        
        # 2. Find the Title
        # Title is usually above the scale text (y is smaller) and roughly aligned on the left (x0 is similar)
        title_candidates = []
        for item in items:
            if item == scale: continue
            ix0, iy0, ix1, iy1 = item["bbox"]
            
            # Check if it's directly above
            if iy1 <= sy0 + 5 and (sy0 - iy1) < 80:
                # Check if it's roughly aligned horizontally
                if abs(ix0 - sx0) < 150:
                    title_candidates.append(item)
                    
        # Pick the text closest to the scale (largest y1)
        title_candidates.sort(key=lambda x: x["bbox"][3], reverse=True)
        title = title_candidates[0] if title_candidates else None
        
        # 3. Find the Drawing Number (inside the circle)
        # Drawing numbers are to the left of the title/scale block, and usually 1-3 characters
        number_candidates = []
        search_y0 = title["bbox"][1] - 20 if title else sy0 - 50
        search_y1 = sy1 + 20
        
        for item in items:
            if item == scale or item == title: continue
            ix0, iy0, ix1, iy1 = item["bbox"]
            
            # Check if it's to the left
            if ix1 < sx0 + 20 and (sx0 - ix1) < 150:
                # Check if it's vertically aligned with the title/scale block
                if iy1 > search_y0 and iy0 < search_y1:
                    # Usually drawing numbers are short (like "1", "A", "12")
                    if len(item["text"]) <= 5:
                        number_candidates.append(item)
                        
        # Pick the one closest to the text (largest x1)
        number_candidates.sort(key=lambda x: x["bbox"][2], reverse=True)
        number = number_candidates[0] if number_candidates else None
        
        blocks.append({
            "scale": scale,
            "title": title,
            "number": number
        })

    # --- Print extracted info ---
    print("\n--- Extracted Detail Blocks ---")
    for i, b in enumerate(blocks):
        n_text = b['number']['text'] if b['number'] else "[NOT FOUND]"
        t_text = b['title']['text'] if b['title'] else "[NOT FOUND]"
        s_text = b['scale']['text']
        print(f"Block {i+1}:")
        print(f"  Drawing Number : {n_text}")
        print(f"  Title          : {t_text}")
        print(f"  Scale          : {s_text}")
        print("-" * 40)

    # --- Draw boxes on the PDF and save an image ---
    print(f"\nDrawing bounding boxes on PDF to {out_image}...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    for b in blocks:
        # Scale = Red Box
        page.draw_rect(fitz.Rect(*b["scale"]["bbox"]), color=(1, 0, 0), width=2)
        
        # Title = Blue Box
        if b["title"]:
            page.draw_rect(fitz.Rect(*b["title"]["bbox"]), color=(0, 0, 1), width=2)
            
        # Number = Green Box
        if b["number"]:
            page.draw_rect(fitz.Rect(*b["number"]["bbox"]), color=(0, 1, 0), width=3)
            
    # Render and save
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    pix.save(out_image)
    print("✅ Visualization saved!")

if __name__ == "__main__":
    pdf_path = r"D:\Drawing_Extraction\DGS_Arch-37.pdf"
    json_path = "vector_text_output.json"
    extract_detail_blocks(pdf_path, json_path)
