import os
import sys
import json
import torch

# Fix Windows cp1252 encoding issue and stdout buffering
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    else:
        sys.stdout.reconfigure(line_buffering=True)
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    else:
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Import our modular pipeline steps
from extractor import extract_structured_materials
from grain_verifier import verify_grain_expected, description_based_grain_decision
from web_search_engine import scrape_material_image
from analyze_grain import analyze_grain_direction

def run_pipeline(image_path, output_dir):
    print("\n" + "="*50, flush=True)
    print("[START] STARTING NEW MATERIAL SCRAPING PIPELINE", flush=True)
    print("="*50, flush=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    vram_gb = 0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct" if vram_gb < 10 else "Qwen/Qwen2.5-VL-7B-Instruct"

    print(f"\n[INIT] Loading {model_id} Vision AI (VRAM: {vram_gb:.1f}GB)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("[INIT] AI Loaded successfully.\n", flush=True)

    cache_path = os.path.join(output_dir, 'materials_cache.json')
    if os.path.exists(cache_path):
        print(f"[INIT] Found cached materials at {cache_path}. Loading...", flush=True)
        with open(cache_path, 'r') as f:
            materials = json.load(f)
    else:
        materials = extract_structured_materials(image_path, model, processor)
        if not materials:
            print("No materials extracted. Exiting with error.", flush=True)
            sys.exit(1)
            
        with open(cache_path, 'w') as f:
            json.dump(materials, f, indent=4)
        print(f"[INIT] Saved extracted materials to cache: {cache_path}", flush=True)
        
    manual_review_list = []
    success_list = []

    for mat in materials:
        tag = mat.get('tag', '')
        print(f"\n--- Processing {tag} ---")
        
        expected_grain = verify_grain_expected(mat)
        print(f"[{tag}] Expected Grain based on description: {expected_grain}")
        
        result = scrape_material_image(mat, output_dir, model, processor)
        if isinstance(result, tuple):
            image_path_res, reason = result
        else:
            image_path_res, reason = result, "Could not find/verify official image"
            
        if not image_path_res:
            # ── AUTO FALLBACK: web scrape failed ──────────────────────────
            # Instead of flagging for manual review, use description keywords
            # to automatically decide grain YES/NO. No manual fix needed!
            fallback = description_based_grain_decision(mat)
            print(f"[{tag}] [FALLBACK] Web scrape failed. Auto-deciding from description...")
            print(f"[{tag}] [FALLBACK] Decision: grain={fallback['grain']} | {fallback['analysis']}")

            mat['status']    = "SUCCESS (description-based fallback)"
            mat['grain']     = fallback['grain']
            mat['direction'] = fallback['direction']
            mat['analysis']  = fallback['analysis']
            mat['image_path'] = None
            success_list.append(mat)
        else:
            if expected_grain == "YES":
                print(f"[{tag}] Image verified. Running precise Grain Direction Analysis...")
                analysis_result = analyze_grain_direction(image_path_res, model, processor)
            elif expected_grain == "NO":
                analysis_result = "Grain: No | Direction: None (Verified by description)"
            else:
                print(f"[{tag}] Grain unknown. Running Visual Grain Analysis to check...")
                analysis_result = analyze_grain_direction(image_path_res, model, processor)
                
            print(f"[{tag}] [OK] Final Analysis: {analysis_result}")
            
            mat['analysis'] = analysis_result
            mat['image_path'] = image_path_res
            mat['status'] = "SUCCESS"
            success_list.append(mat)
            
            if "Grain: Yes" in analysis_result:
                mat['grain'] = "YES"
                mat['direction'] = "VERTICAL" if "Vertical" in analysis_result else "HORIZONTAL" if "Horizontal" in analysis_result else "UNKNOWN"
            elif "Grain: No" in analysis_result:
                mat['grain'] = "NO"
                mat['direction'] = "NONE"
            else:
                mat['grain'] = "UNKNOWN"
                mat['direction'] = "UNKNOWN"
        
        json_path = os.path.join(output_dir, f"{tag}.json")
        with open(json_path, 'w') as f:
            json.dump(mat, f, indent=4)
            
    html_report = "<html><body><h2>Pipeline Execution Report</h2><h3>Successful Verifications</h3><ul>"
    for s in success_list:
        html_report += f"<li><b>{s['tag']}</b>: {s['analysis']}</li>"
    html_report += "</ul><h3>Manual Review Required</h3><ul>"
    for r in manual_review_list:
        html_report += f"<li><b>{r['tag']}</b> (Query: {r['query']}) - {r['status']}</li>"
    html_report += "</ul></body></html>"
    
    report_path = os.path.join(output_dir, "PIPELINE_REPORT.html")
    with open(report_path, 'w') as f:
        f.write(html_report)
        
    print(f"\nPipeline finished. Report saved to {report_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python main_pipeline.py <image_path> <output_dir>")
    else:
        run_pipeline(sys.argv[1], sys.argv[2])
