import fitz  # PyMuPDF
import base64
import requests
import json
import os
import io
from PIL import Image

API_KEY = ""

def extract_plumbing16_with_qwen(pdf_path, output_path):
    print(f"Loading {pdf_path}...")
    doc = fitz.open(pdf_path)
    page = doc[0] # First page
    
    # Render page to high-res image
    print("Rendering page to image...")
    mat = fitz.Matrix(300 / 72, 300 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    # Convert to base64
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64_string = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    prompt = """You are an expert architectural data extractor. 
I am providing an image of a Plumbing Fixtures Schedule. 
This table is complex: the 'TAG' is vertically centered, and the manufacturer/model/descriptions are multi-line and stacked.

Please extract ALL the items in the table.
For each item, return the primary TAG (e.g. "WC-1", "WC-2 (BARRIER FREE)").
Then, for that tag, extract the Manufacturer and Model Number. If there are multiple components (like Fixture, Flush Valve, Seat) under one tag, combine their details or just capture the primary Fixture Manufacturer and Model.

Also capture the "REMARKS" or "DESCRIPTION" for the comments column. 
Since there isn't a dedicated "FINISH" column, leave it as an empty string.

Return ONLY a valid JSON object matching this schema so it integrates directly with our QC cross-checker:

{
  "PLUMBING FIXTURES": {
    "GENERAL": [
      {
        "TAG": "WC-1",
        "MANUFACTURER": "AMERICAN STANDARD",
        "MODEL NO.": "3351.101EC",
        "FINISH": "",
        "REMARKS": "WALL HUNG, VITREOUS CHINA..."
      }
    ]
  }
}
"""
    
    print("Sending to Qwen-VL API for visual table extraction (this may take 15-30 seconds)...")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen/qwen3-vl-32b-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_string}"}}
                ]
            }
        ],
        "max_tokens": 2048
    }
    
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
    
    if res.status_code == 200:
        content = res.json()["choices"][0]["message"]["content"]
        
        # Clean up Markdown formatting if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"\n[SUCCESS] Extracted Schedule via Vision AI! Saved to {output_path}")
        try:
            parsed = json.loads(content)
            print("Preview of extracted tags:")
            for item in parsed.get("PLUMBING FIXTURES", {}).get("GENERAL", []):
                print(f" - {item.get('TAG')}: {item.get('MANUFACTURER')} {item.get('MODEL NO.')}")
        except:
            pass
    else:
        print(f"[ERROR] API Error: {res.status_code} - {res.text}")

if __name__ == "__main__":
    import sys
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "Plumbing16.pdf"
    out_path = r"table extraction output\plumbing16\extracted_schedules.json"
    extract_plumbing16_with_qwen(pdf_path, out_path)
