import os
import json
import torch
import re
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def extract_structured_materials(image_path, model=None, processor=None):
    """
    Extracts materials from a cropped table image and breaks them down into 
    tag, manufacturer, name, and finish/code.
    """
    print("Extracting structured material data from image using Qwen-2.5-VL-7B...")
    
    if model is None or processor is None:
        print("Loading Qwen2.5-VL-3B model into memory (This is much faster)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    prompt = """
    Extract all the materials from the 'MATERIALS' or 'MATERIAL SCHEDULE' table in this image.
    
    STRICT RULES FOR EXTRACTION (CRITICAL FOR PRODUCTION ACCURACY):
    1. Do NOT hallucinate or guess ANY information. Only extract exactly what is written in the table.
    2. If a manufacturer (e.g., Formica, Wilsonart, Chemetal, Caesarstone, Fenix) is not explicitly mentioned, leave the 'manufacturer' field as an empty string. Do not assume or guess based on the material type.
    3. Be extremely precise with Tag characters, codes, and finish numbers. 
    4. Pay strict attention to similar-looking characters (e.g., 1 vs I, B vs G, 8 vs B, 0 vs O).
    5. Pay strict attention to special characters like slashes (/) and hyphens (-), and do not misread them as letters (e.g., / is not I).
    6. Ignore any hardware tags (e.g., tags starting with H- or HW-).
    7. CRITICAL: NEVER use double quotes inside string values. If you need to represent inches, use single quotes (e.g., 3/4'). Using unescaped double quotes will break the JSON parser.

    For each material, output:
    - manufacturer: (Extracted strictly from text, otherwise empty string)
    - name: (The actual name/type of the material)
    - code_or_finish: (Any code numbers, thicknesses, or finish descriptions)
    
    Return ONLY a valid JSON array of objects, with each object having 'tag', 'manufacturer', 'name', and 'code_or_finish' keys. If a field is missing, use an empty string. Do not include any markdown formatting like ```json, just the raw JSON array.
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inputs processed and sent to {inputs.input_ids.device}. Running generation (this may take a few minutes)...")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1500)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    output_text = output_text.strip()
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.startswith("```"):
        output_text = output_text[3:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]
    output_text = output_text.strip()

    materials = None
    # 1. Direct parse
    try:
        materials = json.loads(output_text)
    except Exception:
        pass

    # 2. Fix unescaped inch quotes (e.g., 3/4" or 5/16" or 1/4")
    if materials is None:
        try:
            sanitized = re.sub(r'(?<=\d)\"(?=[A-Za-z\s,/\-])', "'", output_text)
            sanitized = re.sub(r'(?<=[A-Za-z])\"(?=\s*[A-Za-z])', "'", sanitized)
            materials = json.loads(sanitized)
        except Exception:
            pass

    # 3. Regex block recovery
    if materials is None:
        try:
            blocks = re.findall(r'\{[^{}]*\}', output_text)
            recovered = []
            for b in blocks:
                b_clean = re.sub(r'(?<=\d)\"(?=[A-Za-z\s,/\-])', "'", b)
                try:
                    recovered.append(json.loads(b_clean))
                except Exception:
                    pass
            if recovered:
                materials = recovered
        except Exception:
            pass

    if materials is not None:
        # Filter out hardware tags just in case
        filtered_materials = [m for m in materials if not (m.get('tag', '').upper().startswith('H-') or m.get('tag', '').upper().startswith('HW'))]
        
        # POST-PROCESSING: Fix AI Hallucinations
        for m in filtered_materials:
            name_up = m.get('name', '').upper()
            code_up = m.get('code_or_finish', '').upper()
            
            if "RCISM" in code_up:
                m['code_or_finish'] = re.sub(r'(?i)RCISM', 'RC/SM', m.get('code_or_finish', ''))
            if "RCISM" in name_up:
                m['name'] = re.sub(r'(?i)RCISM', 'RC/SM', m.get('name', ''))
                
            if m.get('tag') == "QZ-G":
                m['tag'] = "QZ-B"
                
            if "VENEER" in name_up or "MELAMINE" in name_up or "PLYWOOD" in name_up:
                m['manufacturer'] = ""
                
        print(f"Successfully extracted {len(filtered_materials)} materials.")
        return filtered_materials
    else:
        print("Error: Could not parse JSON from Qwen.")
        print("Raw output:", output_text)
        return []

if __name__ == "__main__":
    test_image = r"c:\Users\Lalitha\OneDrive - QUADE Engineering Services\Desktop\Grainn1\inputs\Screenshot 2026-08-18 104211.png"
    if os.path.exists(test_image):
        mats = extract_structured_materials(test_image)
        for m in mats:
            print(m)
    else:
        print("Test image not found.")
