import json
import argparse
import os
import requests

API_KEY = ""

def verify_with_qwen(tag, drawing_inst, master_comp):
    """
    Uses Qwen to intelligently determine if a deterministic text mismatch is a true 
    discrepancy or just semantic/phrasing differences.
    """
    prompt = f"""You are an expert plumbing engineer doing Quality Control.
We are comparing a component for item '{tag}'.

DRAWING CALLOUT:
- Make: {drawing_inst.get('make', '')}
- Model: {drawing_inst.get('model', '')}
- Comments: {drawing_inst.get('comments', '')}

MASTER SCHEDULE SPECIFICATION:
- Role/Type: {master_comp.get('role', master_comp.get('type', ''))}
- Make: {master_comp.get('manufacturer', master_comp.get('make', ''))}
- Model: {master_comp.get('model', '')}
- Description: {master_comp.get('description', master_comp.get('remarks', ''))}

Does the Drawing Callout genuinely CONFLICT with the Master Schedule?
Rules:
- If the Drawing has a brief summary (e.g. "UNDERCOUNTER ICE MAKER") and the Master has detailed specs (e.g. "500 LBS / 24 HR CAPACITY"), this is a MATCH, NOT a discrepancy.
- Minor spelling differences (e.g. "FRIDGIDAIRE" vs "FRIGIDAIRE") are MATCHES.
- Missing prefixes/suffixes in model numbers are often MATCHES.
- A genuine discrepancy is when they specify fundamentally different brands, completely contradictory models, or conflicting functions.

Return ONLY a valid JSON object with no markdown:
{{
  "is_discrepancy": true/false,
  "reasoning": "Brief 1-sentence explanation"
}}"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "qwen/qwen2.5-vl-72b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128
    }
    
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            content = res.json()["choices"][0]["message"]["content"].strip()
            if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        else:
            print(f"      [!] API returned {res.status_code}: {res.text}")
    except Exception as e:
        print(f"      [!] Request exception for {tag}: {e}")
        
    # Default to assuming it's a discrepancy if API fails
    return {"is_discrepancy": True, "reasoning": "Fallback to deterministic matching due to API failure."}

def check_schedules(all_pages_path, extracted_paths):
    print(f"[QC] Loading Source Details (All Pages): {all_pages_path}")
    with open(all_pages_path, 'r', encoding='utf-8') as f:
        all_pages_data = json.load(f)
        
    # 1. Parse Master Schedules (Extracted) from multiple files
    master_tags = {}
    
    for ext_path in extracted_paths:
        print(f"[QC] Loading Master Schedule: {ext_path}")
        with open(ext_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
            
        for sched_name, sections in extracted_data.items():
            if type(sections) is list:
                # Flat schema
                for item in sections:
                    tag = item.get("TAG") or item.get("ARCH. TAG")
                    if tag:
                        master_tags[tag] = item
                continue

            for section, items in sections.items():
                for item in items:
                    tag = item.get("TAG") or item.get("ARCH. TAG")
                    if tag:
                        master_tags[tag] = {
                            **item, # Preserve all original keys like 'ITEM' and 'DESCRIPTION'
                            "make": item.get("MANUFACTURER", ""),
                            "model": item.get("MODEL NO.", ""),
                            "finish": item.get("FINISH", ""),
                            "remarks": item.get("REMARKS", item.get("DESCRIPTION", "")),
                            "components": item.get("COMPONENTS", [])
                        }
                    
    # 2. Parse Drawing Schedules (All Pages)
    drawing_tags = {}
    tables_by_page = all_pages_data.get("tables_by_page", {})
    for page_num, schedules in tables_by_page.items():
        for sched_type, sched_data in schedules.items():
            rows = sched_data.get("rows", [])
            for row in rows:
                tag = row.get("ARCH. TAG")
                if not tag or "ARCH. TAG" in tag or "TYPICAL" in tag:
                    continue
                
                # Combine values across multiple lines if needed
                if tag not in drawing_tags:
                    drawing_tags[tag] = []
                    
                drawing_tags[tag].append({
                    "page": page_num,
                    "make": row.get("MAKE", ""),
                    "model": row.get("MODEL", ""),
                    "finish": row.get("FINISH", ""),
                    "comments": row.get("COMMENTS", "")
                })

    # 3. Cross-Check
    report = {
        "missing_from_master": [],
        "information_mismatches": [],
        "successfully_matched": []
    }
    
    print("\n" + "="*50)
    print("           QC CROSS-CHECK SUMMARY")
    print("="*50)
    
    for tag, instances in drawing_tags.items():
        if tag not in master_tags:
            pages = list(set([str(inst['page']) for inst in instances]))
            report["missing_from_master"].append({"tag": tag, "pages": pages})
            continue
            
        master = master_tags[tag]
        mismatch_groups = {} # Group identical issues across multiple pages
        ai_cache = {} # Cache AI verification results per unique deterministic mismatch
        
        def add_mismatch(issue_text, page):
            if issue_text not in mismatch_groups:
                mismatch_groups[issue_text] = []
            mismatch_groups[issue_text].append(str(page))

        for inst in instances:
            page = inst["page"]
            drawing_make = inst.get("make", "").strip().upper()
            drawing_model = inst.get("model", "").strip().upper()
            drawing_finish = inst.get("finish", "").strip().upper()
            drawing_comments = inst.get("comments", "").strip().upper()
            
            master_finish = master.get("finish", "").strip().upper()
            master_remarks = master.get("remarks", "").strip().upper()

            master_components = master.get("components", [])
            
            # --- INTELLIGENT MATCHING PRE-CHECK ---
            # Instead of purely strict checking, let's see if we can find ANY perfect deterministic match first.
            best_comp_mismatches = None
            best_comp = None
            
            if master_components:
                for comp in master_components:
                    comp_make = comp.get("manufacturer", "").strip().upper()
                    comp_model = comp.get("model", "").strip().upper()
                    comp_desc = comp.get("description", "").strip().upper()
                    
                    comp_role = comp.get("role", comp.get("type", "")).strip().upper()
                    comp_context = f"{comp_role} {comp_desc}".strip()
                    
                    comp_mismatches = []
                    if drawing_make and comp_make and drawing_make not in comp_make and comp_make not in drawing_make:
                        comp_mismatches.append(f"Make mismatch: Drawing says '{inst['make']}', Master [{comp.get('role')}] says '{comp.get('manufacturer')}'")
                    if drawing_model and comp_model and drawing_model not in comp_model and comp_model not in drawing_model:
                        comp_mismatches.append(f"Model mismatch: Drawing says '{inst['model']}', Master [{comp.get('role')}] says '{comp.get('model')}'")
                    
                    has_positive_match = False
                    if drawing_make and (drawing_make in comp_make or comp_make in drawing_make): has_positive_match = True
                    if drawing_model and (drawing_model in comp_model or comp_model in drawing_model): has_positive_match = True
                    if drawing_comments and (drawing_comments in comp_context or comp_context in drawing_comments): has_positive_match = True
                    
                    if not has_positive_match and (drawing_make or drawing_model):
                        comp_mismatches.append(f"Component mismatch")
                        
                    if best_comp_mismatches is None or len(comp_mismatches) < len(best_comp_mismatches):
                        best_comp_mismatches = comp_mismatches
                        best_comp = comp
                        
                    if len(comp_mismatches) == 0 and has_positive_match:
                        break # Found a perfect matching component!
                        
                # IF DETERMINISTIC MATCHING FAILS, USE QWEN TO SEMANTICALLY VERIFY THE BEST CANDIDATE
                if best_comp_mismatches:
                    cache_key = str(best_comp_mismatches)
                    if cache_key not in ai_cache:
                        print(f"  [AI] Verifying deterministic mismatch for '{tag}' with Qwen...")
                        ai_cache[cache_key] = verify_with_qwen(tag, inst, best_comp)
                        
                    qwen_result = ai_cache[cache_key]
                    
                    if not qwen_result.get("is_discrepancy", True):
                        # Qwen says it's functionally matching!
                        best_comp_mismatches = []
                        if cache_key not in ai_cache.get('_logged', []):
                            print(f"       -> AI Resolved: {qwen_result.get('reasoning')}")
                            ai_cache.setdefault('_logged', []).append(cache_key)
                    else:
                        if cache_key not in ai_cache.get('_logged', []):
                            print(f"       -> AI Confirmed Discrepancy: {qwen_result.get('reasoning')}")
                            ai_cache.setdefault('_logged', []).append(cache_key)
                        # Append the AI's reasoning to the mismatch to explain it to the user
                        ai_note = f" (AI Note: {qwen_result.get('reasoning')})"
                        best_comp_mismatches = [m + ai_note for m in best_comp_mismatches]
                        
                if best_comp_mismatches:
                    for m in best_comp_mismatches:
                        add_mismatch(m, page)
            else:
                # Flat schema fallback
                master_make = master.get("make", "").strip().upper()
                if not master_make: master_make = master.get("manufacturer", master.get("MANUFACTURER", "")).strip().upper()
                master_model = master.get("model", master.get("MODEL NO.", "")).strip().upper()
                
                master_item = master.get("ITEM", "").strip().upper()
                master_desc = master.get("DESCRIPTION", "").strip().upper()
                master_context = f"{master_item} {master_desc} {master_remarks}".strip()
                
                flat_mismatches = []
                if drawing_make and master_make and drawing_make not in master_make and master_make not in drawing_make:
                    flat_mismatches.append(f"Make mismatch: Drawing says '{inst['make']}', Master says '{master_make}'")
                if drawing_model and master_model and drawing_model not in master_model and master_model not in drawing_model:
                    flat_mismatches.append(f"Model mismatch: Drawing says '{inst['model']}', Master says '{master_model}'")
                if drawing_comments and master_context and drawing_comments not in master_context and master_context not in drawing_comments:
                    flat_mismatches.append(f"Comments mismatch: Drawing says '{inst['comments']}', Master specifies '{master_item}' (Desc: {master_desc})")
                
                if flat_mismatches:
                    cache_key = str(flat_mismatches)
                    if cache_key not in ai_cache:
                        print(f"  [AI] Verifying deterministic mismatch for '{tag}' with Qwen...")
                        ai_cache[cache_key] = verify_with_qwen(tag, inst, master)
                        
                    qwen_result = ai_cache[cache_key]
                    
                    if not qwen_result.get("is_discrepancy", True):
                        flat_mismatches = []
                        if cache_key not in ai_cache.get('_logged', []):
                            print(f"       -> AI Resolved: {qwen_result.get('reasoning')}")
                            ai_cache.setdefault('_logged', []).append(cache_key)
                    else:
                        if cache_key not in ai_cache.get('_logged', []):
                            print(f"       -> AI Confirmed Discrepancy: {qwen_result.get('reasoning')}")
                            ai_cache.setdefault('_logged', []).append(cache_key)
                        ai_note = f" (AI Note: {qwen_result.get('reasoning')})"
                        flat_mismatches = [m + ai_note for m in flat_mismatches]
                
                for m in flat_mismatches:
                    add_mismatch(m, page)
        
        if mismatch_groups:
            report["information_mismatches"].append({
                "tag": tag,
                "issues": mismatch_groups
            })
        else:
            report["successfully_matched"].append(tag)
                
    # 4. Save Report
    out_name = "Deterministic_QC_Report.json"
    out_path = os.path.join(os.path.dirname(extracted_paths[0]), out_name)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"Total Unique Tags Found on Drawings: {len(drawing_tags)}")
    print(f"Tags Missing from Master Schedule:   {len(report['missing_from_master'])}")
    print(f"Tags with Information Mismatches:    {len(report['information_mismatches'])}")
    print(f"Tags Successfully Matched:           {len(report['successfully_matched'])}")
    
    if report["successfully_matched"]:
        print("\n[✔] SUCCESSFULLY MATCHED:")
        print("    " + ", ".join(report["successfully_matched"]))

    if report["information_mismatches"]:
        print("\n[!] INFORMATION MISMATCHES:")
        for mm in report["information_mismatches"]:
            print(f"  - {mm['tag']}:")
            for issue, pages in mm["issues"].items():
                print(f"      * {issue} (Found on pages: {', '.join(pages)})")
                
    if report["missing_from_master"]:
        print("\n[X] MISSING FROM MASTER SCHEDULE:")
        for miss in report["missing_from_master"]:
            print(f"  - {miss['tag']} (Found on pages: {', '.join(miss['pages'])})")
            
    print(f"\n[SUCCESS] Detailed JSON report saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drawing_json", required=True, help="Path to ALL_PAGES_summary.json")
    parser.add_argument("--table_json", required=True, nargs="+", help="Paths to extracted_schedules.json files")
    args = parser.parse_args()
    check_schedules(args.drawing_json, args.table_json)
