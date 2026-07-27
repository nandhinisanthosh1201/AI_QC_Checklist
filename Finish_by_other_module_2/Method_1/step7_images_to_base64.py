import os
import json
import base64
import requests

API_KEY = ""

def extract_details_with_qwen(in_dir="cropped_blocks", out_json="qwen_extracted_details.json"):
    print(f"Reading images from {in_dir}...")
    
    if not os.path.exists(in_dir):
        print(f"Error: Directory {in_dir} does not exist.")
        return

    results = []
    
    # Sort files so they are in a nice order
    files = sorted([f for f in os.listdir(in_dir) if f.endswith(".png")])
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = """You are analyzing a single cropped architectural drawing title block.

Extract the following:
1. drawing_number: The circled/bubbled number
2. drawing_title: The title text
3. scale: The scale text

Return ONLY a JSON object:
{
  "drawing_number": "1",
  "drawing_title": "ENLARGED PLAN", 
  "scale": "1/4\\" = 1'-0\\""
}"""

    for filename in files:
        filepath = os.path.join(in_dir, filename)
        print(f"\nProcessing {filename}...")
        
        # Read the raw bytes of the PNG and convert to Base64
        with open(filepath, "rb") as image_file:
            b64_string = base64.b64encode(image_file.read()).decode("utf-8")
            
        payload = {
            "model": "qwen/qwen3-vl-32b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_string}"}}
                    ]
                }
            ],
            "max_tokens": 512
        }
        
        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                print("API Error:", response.text)
                continue
            
            content = response.json()["choices"][0]["message"]["content"]
            
            # Clean JSON markdown if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            data["source_file"] = filename
            print("Extracted Data:", json.dumps(data, indent=2))
            
            results.append(data)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            
    # Save the extracted data to a JSON file
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✅ Successfully processed {len(results)} images and saved extracted data to {out_json}")

if __name__ == "__main__":
    extract_details_with_qwen()
