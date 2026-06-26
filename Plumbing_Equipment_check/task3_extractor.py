import pdfplumber
import json
import re
import argparse
from collections import defaultdict
import os

def extract_tables_tier1(pdf_path, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    all_schedules = {}
    
    # Load configuration
    titles = config.get('titles', [])
    header_keywords = config.get('header_keywords', ['TAG'])
    section_keyword = config.get('section_keyword', 'SECTION')
    skip_keyword = config.get('skip_keyword', 'NOT USED')
    tag_columns = config.get('tag_columns', ['TAG', 'ARCH. TAG'])
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text or len(text) < 50:
                print(f"[Tier 2 Route] Page {page_num+1} has low text density. Falling back to Qwen-VL.")
                # Fallback to Qwen-VL would happen here
                continue
            
            words = page.extract_words(keep_blank_chars=False, x_tolerance=2, y_tolerance=3)
            
            # Group words by Y coordinate (rows)
            y_groups = defaultdict(list)
            for w in words:
                y = round(w['top'] / 2) * 2 # Group by 2px bands
                y_groups[y].append(w)
                
            sorted_ys = sorted(y_groups.keys())
            
            # Locate schedules
            current_schedule = None
            current_headers = None
            current_col_bounds = None
            
            # Track state
            current_section = None
            schedule_items = []
            
            def save_current_schedule():
                if current_schedule and schedule_items:
                    if current_schedule not in all_schedules:
                        all_schedules[current_schedule] = {}
                    
                    # Group by section
                    for item in schedule_items:
                        sec = item.get('_section', 'GENERAL')
                        if sec not in all_schedules[current_schedule]:
                            all_schedules[current_schedule][sec] = []
                        # Clean internal fields
                        cleaned_item = {k: v for k, v in item.items() if not k.startswith('_')}
                        all_schedules[current_schedule][sec].append(cleaned_item)

            for i, y in enumerate(sorted_ys):
                row_words = sorted(y_groups[y], key=lambda x: x['x0'])
                row_text = " ".join([w['text'] for w in row_words]).upper()
                
                # Check for titles (Dynamic regex if titles list is empty, otherwise match provided titles)
                found_title = None
                if titles:
                    for t in titles:
                        if t.upper() in row_text:
                            found_title = t
                            break
                else:
                    if "SCHEDULE" in row_text and len(row_text.split()) < 8:
                        found_title = row_text.strip()
                        
                if found_title:
                    save_current_schedule()
                    current_schedule = found_title
                    current_headers = None
                    current_col_bounds = None
                    current_used_exact_bounds = False
                    current_section = None
                    schedule_items = []
                    continue
                
                if current_schedule and not current_headers:
                    # Look for headers based on keywords
                    is_header = any(hk.upper() in row_text for hk in header_keywords)
                    if is_header:
                        current_headers = []
                        current_col_bounds = []
                        current_used_exact_bounds = False
                        merged_headers = []
                        curr_hdr = row_words[0]
                        for w in row_words[1:]:
                            if w['x0'] - curr_hdr['x1'] < 15: # Same header cell
                                curr_hdr['text'] += " " + w['text']
                                curr_hdr['x1'] = w['x1']
                            else:
                                merged_headers.append(curr_hdr)
                                curr_hdr = w
                        merged_headers.append(curr_hdr)
                        
                        # Find vertical lines on the page that intersect this header's Y band
                        table_v_lines = []
                        for edge in page.edges:
                            # If it's a vertical line (width < 2, height > 15) and intersects the header's Y
                            if edge.get('width', 0) < 2 and edge.get('height', 0) > 15:
                                if edge['top'] <= y + 5 and edge['bottom'] >= y - 5:
                                    table_v_lines.append(edge['x0'])
                                    
                        table_v_lines = sorted(list(set([round(x) for x in table_v_lines])))
                        current_used_exact_bounds = len(table_v_lines) > len(merged_headers)
                        
                        for h in merged_headers:
                            current_headers.append(h['text'])
                            
                            # Determine boundaries for this header using vertical lines if possible
                            if current_used_exact_bounds:
                                # Snap to the nearest vertical line to the left of the header text
                                left_lines = [lx for lx in table_v_lines if lx <= h['x0'] + 5]
                                right_lines = [lx for lx in table_v_lines if lx >= h['x1'] - 5]
                                
                                x_start = left_lines[-1] if left_lines else h['x0'] - 5
                                x_end = right_lines[0] if right_lines else page.width
                                current_col_bounds.append((x_start, x_end))
                            else:
                                current_col_bounds.append((h['x0'], h['x1']))
                        
                        # Add a fake boundary for the end of the page if using heuristic
                        if not current_used_exact_bounds:
                            current_col_bounds.append((page.width, page.width))
                        continue
                        
                if current_schedule and current_headers:
                    # Look for sections
                    if section_keyword and section_keyword.upper() in row_text:
                        current_section = row_text
                        continue
                        
                    # Process data row
                    row_data = {}
                    for idx, h in enumerate(current_headers):
                        if current_used_exact_bounds:
                            # We used vertical lines for exact bounds
                            x_start, x_end = current_col_bounds[idx]
                        else:
                            # Use midpoint between headers for boundaries to handle centered text
                            if idx == 0:
                                x_start = 0
                            else:
                                x_start = (current_col_bounds[idx-1][1] + current_col_bounds[idx][0]) / 2
                                
                            if idx < len(current_headers) - 1:
                                x_end = (current_col_bounds[idx][1] + current_col_bounds[idx+1][0]) / 2
                            else:
                                x_end = page.width
                            
                        cell_words = [w['text'] for w in row_words if x_start <= w['x0'] < x_end]
                        row_data[h] = " ".join(cell_words).strip()
                        
                    row_data['_section'] = current_section or 'GENERAL'
                    
                    if skip_keyword and skip_keyword.upper() in row_text:
                        row_data['_status'] = 'SKIP'
                    
                    # Only add if it has a valid tag column
                    has_tag = False
                    for tag_col in tag_columns:
                        if row_data.get(tag_col) or row_data.get(tag_col.upper()):
                            has_tag = True
                            break
                            
                    if has_tag:
                        schedule_items.append(row_data)
                    else:
                        # Might be a wrapped line, append to previous row
                        if schedule_items:
                            for idx, h in enumerate(current_headers):
                                if row_data.get(h):
                                    if schedule_items[-1].get(h):
                                        schedule_items[-1][h] += " " + row_data[h]
                                    else:
                                        schedule_items[-1][h] = row_data[h]
                                        
            save_current_schedule()

    # Validation Pass
    print("\n--- Validation Pass ---")
    for sched_name, sections in all_schedules.items():
        all_tags = set()
        for sec, items in sections.items():
            for item in items:
                # Find tag key
                tag_key = None
                for tc in tag_columns:
                    if tc in item:
                        tag_key = tc
                        break
                        
                if not tag_key:
                    print(f"[Warning] {sched_name} missing TAG column.")
                    continue
                
                tag = item[tag_key]
                if not tag:
                    continue
                    
                if tag in all_tags:
                    print(f"[Warning] Duplicate tag found in {sched_name}: {tag}")
                all_tags.add(tag)
                
                if item.get('_status') == 'SKIP':
                    print(f"[Info] {tag} is marked to be skipped ({skip_keyword}).")

    out_file = os.path.join(output_dir, "extracted_schedules.json")
    with open(out_file, "w") as f:
        json.dump(all_schedules, f, indent=2)
    print(f"Extraction complete. Saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract schedule tables from PDF")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="table extraction output", help="Output directory")
    parser.add_argument("--config", help="Path to JSON configuration file (optional)")
    parser.add_argument("--titles", nargs="+", help="Explicit schedule titles to look for")
    parser.add_argument("--header_keywords", nargs="+", default=["TAG"], help="Keywords to identify header row")
    parser.add_argument("--section_keyword", default="SECTION", help="Keyword to identify section breaks")
    parser.add_argument("--skip_keyword", default="NOT USED", help="Keyword to flag items to skip")
    parser.add_argument("--tag_columns", nargs="+", default=["TAG", "ARCH. TAG"], help="Column names that identify a valid item")
    args = parser.parse_args()
    
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
            
    # Override config with CLI args if provided
    if args.titles:
        config['titles'] = args.titles
    if args.header_keywords:
        config['header_keywords'] = args.header_keywords
    if args.section_keyword:
        config['section_keyword'] = args.section_keyword
    if args.skip_keyword:
        config['skip_keyword'] = args.skip_keyword
    if args.tag_columns:
        config['tag_columns'] = args.tag_columns
        
    extract_tables_tier1(args.pdf, args.out, config)
