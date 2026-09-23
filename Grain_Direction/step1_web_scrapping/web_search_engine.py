import os
import sys
import shutil
import re
import urllib.parse
import requests
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from ddgs import DDGS
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

def get_manufacturer_domain(manufacturer):
    """Map manufacturer name to their official domain."""
    m = manufacturer.upper()
    if "FORMICA" in m: return "formica.com"
    if "WILSONART" in m: return "wilsonart.com"
    if "CAESARSTONE" in m: return "caesarstone.us"
    if "FENIX" in m: return "fenixforinteriors-na.com"
    if "PIONITE" in m: return "pionite.com"
    if "KVADRAT" in m: return "kvadrat.dk"
    if "CORIAN" in m: return "corian.com"
    if "CHEMETAL" in m: return "chemetal.com"
    if "NEVAMAR" in m: return "nevamar.com"
    if "LAMINART" in m: return "laminart.com"
    if "ARBORITE" in m: return "arborite.com"
    if "PANOLAM" in m: return "panolam.com"
    return None

def verify_image_with_ai(image_path, model, processor):
    """
    Uses Qwen to verify if the downloaded image is actually a material swatch.
    Returns True if valid, False if it's spam (logo, person, animal, etc.).
    """
    # RESIZE IMAGE TO PREVENT CUDA OOM!
    try:
        with Image.open(image_path) as img:
            # Convert to RGB in case it's RGBA or P
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.thumbnail((512, 512))
            img.save(image_path)
    except Exception as e:
        print(f"Warning: Could not resize image {image_path}: {e}")
        
    prompt = "Is this image a SINGLE, clean close-up material swatch (like wood grain, stone, metal, or solid color surface)? It MUST NOT contain multiple different colors, text labels, logos, or catalog grids. NOTE: A completely blank, solid colored image (e.g. pure white, solid black, solid bronze, solid gray, etc.) is perfectly valid for solid color surfaces. Answer strictly YES or NO."
    
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

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip().upper()
    
    return "YES" in answer


def clean_search_query(name, finish):
    combined = f"{name} {finish}".upper()
    
    # Remove anything inside parentheses, e.g. (WD-2)
    combined = re.sub(r'\(.*?\)', '', combined)
    # Remove quotes, commas, semicolons
    combined = re.sub(r'[\'\",;]', ' ', combined)
    
    stop_words = ["FINISH", "GRAIN", "NATURAL", "MATTE", "POLISHED", "TEXTURE", "GLOSS", "2CM", "3CM", "THICK", "RCISM", "RC/SM", "POLY", "STAIN", "CUSTOM", "NG"]
    for w in stop_words:
        combined = re.sub(rf'\b{w}\b', '', combined)
        
    combined = re.sub(r'[\d/\.\s]+(?:cm|mm|"|THICK|ga\.)', ' ', combined, flags=re.IGNORECASE)
    
    # Replace hyphens with spaces to prevent search engines from treating them as exclusion operators (e.g. 5794-NG)
    combined = combined.replace('-', ' ')
    combined = re.sub(r'\s+', ' ', combined)
    
    return combined.strip()

def search_images_playwright(query):
    candidate_urls = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            url = "https://www.bing.com/images/search?q=" + urllib.parse.quote(query)
            page.goto(url, timeout=15000)
            imgs = page.query_selector_all('img.mimg')
            for im in imgs:
                src = im.get_attribute('src') or im.get_attribute('data-src')
                if src and src.startswith('http'):
                    candidate_urls.append(src)
            browser.close()
    except Exception as e:
        print(f"Playwright fallback failed: {e}")
    return candidate_urls[:10]

def scrape_material_image(material, output_dir, qwen_model, qwen_processor):
    tag = material.get('tag', '')
    manufacturer = material.get('manufacturer', '')
    name = material.get('name', '')
    finish = material.get('code_or_finish', '')
    
    clean_name = clean_search_query(name, finish)
    domain = get_manufacturer_domain(manufacturer)
    
    queries_to_try = []
    if manufacturer:
        queries_to_try.append(f"{manufacturer} {clean_name} swatch".strip())
        # Try extracting code from name AND finish (e.g., 5016-38, 7953-38, 1570-38)
        codes = re.findall(r'\b[A-Z]?\d{3,5}(?:K|[-]\d+)?\b', f"{name} {finish}".upper())
        for c in codes:
            queries_to_try.append(f"{manufacturer} {c} swatch")
            # Direct Wilsonart CDN check if Wilsonart
            if "WILSONART" in manufacturer.upper():
                queries_to_try.append(f"{manufacturer} {c} site:wilsonart.com")
        if domain:
            queries_to_try.append(f"{manufacturer} {clean_name} site:{domain}".strip())
    else:
        queries_to_try.append(f"{clean_name} surface material swatch".strip())
        queries_to_try.append(f"{clean_name} swatch".strip())
        # For wood veneer / solid wood materials, add more specific fallback queries
        name_upper = name.upper()
        if "VENEER" in name_upper or "ASH" in name_upper or "OAK" in name_upper or "WALNUT" in name_upper or "MAPLE" in name_upper:
            wood_type = clean_name.split()[0] if clean_name else "wood"
            queries_to_try.append(f"{wood_type} veneer wood grain swatch closeup site:ampleveneer.com OR site:oakwoodveneer.com OR site:veneersupplies.com")
            queries_to_try.append(f"{wood_type} plain sliced veneer wood grain texture swatch")
        if "MELAMINE" in name_upper or "LAMINATE" in name_upper:
            queries_to_try.append(f"{clean_name} laminate swatch site:wilsonart.com OR site:formica.com OR site:pionite.com")
    
    temp_img_path = os.path.join(output_dir, f"temp_{tag}.jpg")
    best_candidate_path = os.path.join(output_dir, f"candidate_{tag}.jpg")
    has_candidate = False
    
    for q in queries_to_try:
        print(f"[{tag}] Query: {q}")
        candidate_urls = []
        try:
            results = list(DDGS().images(q, max_results=5))
            candidate_urls = [r['image'] for r in results if r.get('image')]
        except Exception as e:
            print(f"[{tag}] DuckDuckGo direct search failed: {e}. Using Playwright Fallback (Browser Scraping)...")
            candidate_urls = search_images_playwright(q)
            
        if not candidate_urls:
            continue
            
        for img_url in candidate_urls:
            if 'pinterest.' in img_url.lower(): continue
            print(f"[{tag}] Downloading: {img_url[:60]}...")
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                response = requests.get(img_url, timeout=10, headers=headers)
                if response.status_code == 200 and len(response.content) > 1000:
                    with open(temp_img_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Keep first valid image as candidate fallback
                    if not has_candidate:
                        try:
                            with Image.open(temp_img_path) as c_img:
                                if c_img.mode in ('RGBA', 'P'):
                                    c_img = c_img.convert('RGB')
                                c_img.save(best_candidate_path, 'JPEG')
                                has_candidate = True
                        except Exception:
                            pass
                    
                    is_valid = verify_image_with_ai(temp_img_path, qwen_model, qwen_processor)
                    if is_valid:
                        final_path = os.path.join(output_dir, f"{tag}.jpg")
                        shutil.move(temp_img_path, final_path)
                        if os.path.exists(best_candidate_path):
                            os.remove(best_candidate_path)
                        print(f"[{tag}] [AI VERIFIED] Image is a valid material swatch.")
                        return (final_path, "Success")
                    else:
                        if os.path.exists(temp_img_path):
                            os.remove(temp_img_path)
                        print(f"[{tag}] [AI REJECTED] candidate image.")
                        last_reason = "AI Rejected: Identified as spam, logo, or catalog."
            except Exception as e:
                print(f"[{tag}] Image download failed: {str(e)[:50]}")
                last_reason = f"Image download failed."
                    
    # If AI rejected all candidates but we have a valid downloaded candidate image, use it!
    if has_candidate and os.path.exists(best_candidate_path):
        final_path = os.path.join(output_dir, f"{tag}.jpg")
        shutil.move(best_candidate_path, final_path)
        print(f"[{tag}] [FALLBACK CANDIDATE] Saved best candidate swatch image.")
        return (final_path, "Candidate Swatch Saved")
        
    print(f"[{tag}] All fallback queries and URLs failed.")
    return (None, last_reason if 'last_reason' in locals() else "Could not find any relevant URLs or valid images.")
