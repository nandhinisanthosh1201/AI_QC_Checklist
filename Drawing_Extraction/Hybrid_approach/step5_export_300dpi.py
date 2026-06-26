import fitz
import json

def export_300dpi_coordinates(pdf_path, json_path, out_json="detail_blocks_300dpi.json", out_image="page1_300dpi_blocks.png"):
    print("Loading extracted JSON data...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    keywords = ["scale:", "1/4\"", "1/8\"", "1'-", "nts", "n.t.s", "=\""]
    page_data = data[0]
    items = page_data.get("items", [])
    
    scales = [i for i in items if any(kw in i.get("text", "").lower() for kw in keywords) and len(i.get("text", "")) < 30]
    
    blocks = []
    for scale in scales:
        sx0, sy0, sx1, sy1 = scale["bbox"]
        
        # Find Title
        title_candidates = [i for i in items if i != scale and i["bbox"][3] <= sy0 + 5 and (sy0 - i["bbox"][3]) < 80 and abs(i["bbox"][0] - sx0) < 150]
        title_candidates.sort(key=lambda x: x["bbox"][3], reverse=True)
        title = title_candidates[0] if title_candidates else None
        
        # Find Drawing Number
        number_candidates = []
        search_y0 = title["bbox"][1] - 20 if title else sy0 - 50
        search_y1 = sy1 + 20
        
        for item in items:
            if item == scale or item == title: continue
            ix0, iy0, ix1, iy1 = item["bbox"]
            if ix1 < sx0 + 20 and (sx0 - ix1) < 150 and iy1 > search_y0 and iy0 < search_y1 and len(item["text"]) <= 5:
                number_candidates.append(item)
                        
        number_candidates.sort(key=lambda x: x["bbox"][2], reverse=True)
        number = number_candidates[0] if number_candidates else None
        
        # Only keep valid blocks where we found a number
        if number:
            blocks.append({
                "number_text": number["text"],
                "number_bbox_pdf": number["bbox"],
                "title_text": title["text"] if title else "",
                "title_bbox_pdf": title["bbox"] if title else [],
                "scale_text": scale["text"],
                "scale_bbox_pdf": scale["bbox"]
            })

    # Convert coordinates to 300 DPI
    # PDF coordinates are based on 72 DPI. To get 300 DPI pixels, multiply by (300 / 72)
    SCALE_FACTOR = 300.0 / 72.0
    
    def to_300dpi(bbox):
        if not bbox: return []
        # Return rounded integer pixels: [x0, y0, x1, y1]
        return [int(round(coord * SCALE_FACTOR)) for coord in bbox]

    for b in blocks:
        b["number_bbox_300dpi"] = to_300dpi(b["number_bbox_pdf"])
        b["title_bbox_300dpi"] = to_300dpi(b["title_bbox_pdf"])
        b["scale_bbox_300dpi"] = to_300dpi(b["scale_bbox_pdf"])
        
        # Let's also calculate a "Master Bounding Box" that encloses the whole title block
        all_x0 = [x[0] for x in [b["number_bbox_300dpi"], b["title_bbox_300dpi"], b["scale_bbox_300dpi"]] if x]
        all_y0 = [x[1] for x in [b["number_bbox_300dpi"], b["title_bbox_300dpi"], b["scale_bbox_300dpi"]] if x]
        all_x1 = [x[2] for x in [b["number_bbox_300dpi"], b["title_bbox_300dpi"], b["scale_bbox_300dpi"]] if x]
        all_y1 = [x[3] for x in [b["number_bbox_300dpi"], b["title_bbox_300dpi"], b["scale_bbox_300dpi"]] if x]
        
        b["full_block_bbox_300dpi"] = [min(all_x0), min(all_y0), max(all_x1), max(all_y1)]

    # Save to JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)
    print(f"\n✅ Saved exactly {len(blocks)} blocks with 300 DPI coordinates to {out_json}")

    # Generate 300 DPI Image for Verification
    print(f"Generating 300 DPI image: {out_image}")
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    # matrix for 300 DPI
    mat = fitz.Matrix(SCALE_FACTOR, SCALE_FACTOR)
    pix = page.get_pixmap(matrix=mat)
    pix.save(out_image)
    print("✅ 300 DPI Image generated successfully!")

if __name__ == "__main__":
    pdf_path = r"D:\Drawing_Extraction\DGS_Arch-37.pdf"
    json_path = "vector_text_output.json"
    export_300dpi_coordinates(pdf_path, json_path)
