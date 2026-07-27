import gradio as gr
import pandas as pd
import os
import json
import glob
import zipfile
from pipeline import run_full_pipeline
from validator import run_pipeline as run_validator

# ==========================================
# CUSTOM CSS FOR DARK THEME
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* Default Light Mode Variables */
:root {
    --bg-main: #F9FAFB;
    --bg-panel: #FFFFFF;
    --bg-card: #F3F4F6;
    --border-color: #E5E7EB;
    --accent-blue: #3B82F6;
    --text-primary: #111827;
    --text-secondary: #6B7280;
    --status-pass: #16A34A;
    --status-fail: #DC2626;
    --status-warn: #D97706;
    --status-omitted: #4B5563;
    --font-sans: 'Inter', 'Space Grotesk', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* Dark Mode Variables */
.dark {
    --bg-main: #0B0E14;
    --bg-panel: #11151F;
    --bg-card: #1A2030;
    --border-color: #252B3A;
    --accent-blue: #3B82F6;
    --text-primary: #F5F7FA;
    --text-secondary: #9AA4B2;
    --status-pass: #22C55E;
    --status-fail: #EF4444;
    --status-warn: #F59E0B;
    --status-omitted: #6B7280;
}

body, .gradio-container {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* Panel styling */
.custom-panel {
    background-color: var(--bg-panel) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Inputs and Cards */
.gr-box, .gr-input, .gr-dropdown, .gr-file {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}

/* Primary Button */
.primary-btn {
    background-color: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 10px !important;
    transition: all 0.2s ease;
}
.primary-btn:hover {
    background-color: #2563EB !important;
}

/* Toolbar buttons */
.icon-btn {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    font-size: 14px !important;
    padding: 5px 10px !important;
    box-shadow: none !important;
}
.icon-btn:hover {
    color: var(--text-primary) !important;
}

/* Toggle link */
.toggle-link {
    color: var(--accent-blue) !important;
    text-decoration: none;
    font-weight: 500;
    background: none;
    border: none;
    cursor: pointer;
    font-size: 14px;
}

/* Status Badges */
.status-badge {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    display: inline-block;
    min-width: 70px;
    text-align: center;
}
.status-PASS { background-color: var(--status-pass); }
.status-FAIL { background-color: var(--status-fail); }
.status-WARN { background-color: var(--status-warn); }
.status-REVIEW_REQUIRED { background-color: var(--status-warn); }
.status-OMITTED { background-color: var(--status-omitted); }

/* Table styling */
.gr-dataframe table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    background-color: transparent !important;
}
.gr-dataframe th {
    color: var(--text-secondary) !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid var(--border-color) !important;
    text-align: left !important;
}
.gr-dataframe td {
    color: var(--text-primary) !important;
    font-size: 14px !important;
    padding: 16px !important;
    border-bottom: 1px solid var(--border-color) !important;
    vertical-align: middle !important;
}

/* Monospace for specific columns */
.mono-text {
    font-family: var(--font-mono) !important;
    font-size: 13px !important;
    color: #E2E8F0 !important;
}

/* Image viewer overrides */
#viewer_img {
    overflow: auto !important;
    display: flex;
    justify-content: center;
    align-items: center;
}
#viewer_img img {
    transition: transform 0.2s ease-in-out;
}
"""

def format_status(status):
    if not status: return ""
    return f'<span class="status-badge status-{status}">{status}</span>'

def format_mono(text):
    return f'<span class="mono-text">{text}</span>'

# Dummy data for initial table
dummy_data = pd.DataFrame(columns=[
    "Rule ID", "Rule Name", "View ID", "View Name", "View Type", "Status", "Confidence", "Reasoning"
])

def process_documents(sub_pdf, arch_pdf):
    empty_df = pd.DataFrame(columns=["Rule ID", "Rule Name", "View ID", "View Name", "View Type", "Status", "Confidence", "Reasoning"])
    if not sub_pdf or not arch_pdf:
        return "Upload both PDFs first.", gr.update(choices=[]), gr.update(choices=[]), None, [], empty_df
        
    try:
        sub_path = sub_pdf.name if hasattr(sub_pdf, 'name') else sub_pdf
        arch_path = arch_pdf.name if hasattr(arch_pdf, 'name') else arch_pdf
        
        # Run pipeline
        print(f"Running pipeline with {sub_path} and {arch_path}")
        run_full_pipeline(sub_path, arch_path)
        
        pdf_basename = os.path.splitext(os.path.basename(sub_path))[0]
        meta_file = f"submittal_metadata_{pdf_basename}.json"
        
        view_options = []
        gallery_images = []
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for item in metadata:
                vname = item.get('view_name', 'Unknown')
                src = item.get('source_file', '')
                if src:
                    # use the crop filename as a unique identifier
                    crop_name = os.path.basename(src)
                    view_options.append(f"{crop_name} | {vname}")
                    if os.path.exists(src):
                        gallery_images.append(src)
        
        msg = f"Processing completed successfully<br><span style='color: var(--text-secondary); font-size: 13px;'>Views detected: {len(view_options)}</span>"
        
        # Load rules from rules/ directory
        rules_files = glob.glob(os.path.join("rules", "*.json"))
        rules_options = []
        for rf in rules_files:
            try:
                with open(rf, 'r') as f:
                    rdata = json.load(f)
                    if isinstance(rdata, list):
                        for r in rdata:
                            rules_options.append(f"{r.get('rule_id', 'Unknown')} | {r.get('rule_name', os.path.basename(rf))}")
                    else:
                        rules_options.append(f"{rdata.get('rule_id', 'Unknown')} | {rdata.get('rule_name', os.path.basename(rf))}")
            except:
                pass
                
        return msg, gr.update(choices=view_options, interactive=True), gr.update(choices=rules_options, interactive=True), pdf_basename, gallery_images, empty_df
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing documents: {e}", gr.update(choices=[]), gr.update(choices=[]), None, [], empty_df

def run_validation(view_selection, rule_selections, current_df, pdf_basename):
    if not rule_selections or not view_selection or not pdf_basename:
        return current_df, None, None, False, None, None, None
        
    if not isinstance(rule_selections, list):
        rule_selections = [rule_selections]
        
    try:
        crop_name = view_selection.split("|")[0].strip()
        view_name = view_selection.split("|")[1].strip() if "|" in view_selection else "Unknown View"
        
        meta_file = f"submittal_metadata_{pdf_basename}.json"
        with open(meta_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        target_meta = next((m for m in metadata if os.path.basename(m.get('source_file', '')) == crop_name), None)
        if not target_meta:
            raise Exception("View metadata not found")
            
        sub_crop_path = target_meta.get("source_file")
        arch_ref = target_meta.get("arch_ref", "")
        view_type = target_meta.get("view_type", "Unknown")
        
        arch_crop_path = "NA"
        if arch_ref and arch_ref != "NA" and "/" in arch_ref:
            parts = arch_ref.split("/")
            dr_num = parts[0].strip()
            sh_name = parts[1].strip()
            expected_arch_crop = os.path.join("Arch_crop", sh_name, f"{sh_name}_{dr_num}.png")
            if os.path.exists(expected_arch_crop):
                arch_crop_path = expected_arch_crop
                
        sheet_name = crop_name.split('_')[0] if '_' in crop_name else os.path.splitext(crop_name)[0]
        output_dir = os.path.join("results", sheet_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        ret_img = sub_crop_path
        
        for rule_selection in rule_selections:
            rule_id = rule_selection.split("|")[0].strip()
            rule_name = rule_selection.split("|")[1].strip() if "|" in rule_selection else "Unknown Rule"
            
            target_rule = None
            rules_files = glob.glob(os.path.join("rules", "*.json"))
            for rf in rules_files:
                with open(rf, 'r') as f:
                    rdata = json.load(f)
                    if isinstance(rdata, list):
                        for r in rdata:
                            if r.get('rule_id') == rule_id:
                                target_rule = r
                                break
                    else:
                        if rdata.get('rule_id') == rule_id:
                            target_rule = rdata
                if target_rule:
                    break
                    
            if not target_rule:
                continue
                
            # Check if rule requires arch
            req_sources = [r.lower() for r in target_rule.get("required_sources", [])]
            requires_arch = any("arch" in r or "architectural drawing" in r for r in req_sources)
            
            # Run validation
            if requires_arch:
                from config import OPENROUTER_API_KEY
                from validator import init_openrouter, run_pipeline as run_validator
                init_openrouter(
                    api_key    = OPENROUTER_API_KEY,
                    model      = "qwen/qwen3-vl-32b-instruct",
                    max_tokens = 1024,
                )
                run_validator(sub_crop_path, arch_crop_path, target_rule, output_dir, arch_ref)
            else:
                from submittal import run_pipeline_single_rule
                try:
                    from PIL import Image
                    with Image.open(sub_crop_path) as tmp_img:
                        img_w, img_h = tmp_img.size
                except Exception:
                    img_w, img_h = 9999, 9999
                    
                cached_views = [{
                    "view_id": "V1",
                    "detected_view_type": target_meta.get("view_name", "Unknown"),
                    "view_type_classification": target_meta.get("view_type", "Unknown"),
                    "confidence_score": 100.0,
                    "scope_coordinates": {"x1": 0, "y1": 0, "x2": img_w, "y2": img_h}
                }]
                run_pipeline_single_rule(sub_crop_path, target_rule, output_dir, cached_views=cached_views)
            
            # Parse output JSON
            sub_base = os.path.splitext(os.path.basename(sub_crop_path))[0]
            result_json_path = os.path.join(output_dir, f"{rule_id}_{sub_base}_result.json")
            markup_image_path = os.path.join(output_dir, f"{rule_id}_{sub_base}_markup.jpg")
            
            status = "FAIL"
            confidence = "0%"
            reasoning = "Validation failed to produce result."
            json_view_id = crop_name
            
            if os.path.exists(result_json_path):
                with open(result_json_path, 'r') as f:
                    res_data = json.load(f)
                    status = res_data.get("sheet_status", "FAIL")
                    v_results = res_data.get("view_results", [{}])[0]
                    conf = v_results.get("confidence_score", 0)
                    confidence = f"{int(conf)}%" if conf else "-"
                    reasoning = v_results.get("reasoning", "")
                    
                    vid = v_results.get("view_id")
                    if vid:
                        json_view_id = vid
            
            new_row = {
                "Rule ID": format_mono(rule_id),
                "Rule Name": rule_name,
                "View ID": format_mono(json_view_id),
                "View Name": format_mono(view_name),
                "View Type": view_type,
                "Status": format_status(status),
                "Confidence": confidence,
                "Reasoning": reasoning
            }
            
            current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)
            if os.path.exists(markup_image_path):
                ret_img = markup_image_path

        # Zipping and PDF compilation
        json_zip = os.path.join(output_dir, f"jsons_{pdf_basename}.zip")
        img_zip = os.path.join(output_dir, f"images_{pdf_basename}.zip")
        
        with zipfile.ZipFile(json_zip, 'w') as jzip, zipfile.ZipFile(img_zip, 'w') as izip:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('.zip') or file == f"markups_{pdf_basename}.pdf":
                        continue
                    fpath = os.path.join(root, file)
                    if file.endswith('.json'):
                        jzip.write(fpath, os.path.relpath(fpath, output_dir))
                    elif file.endswith('.jpg') or file.endswith('.png'):
                        izip.write(fpath, os.path.relpath(fpath, output_dir))
                        
        pdf_out = None
        try:
            import fitz
            pdf_out = os.path.join(output_dir, f"markups_{pdf_basename}.pdf")
            
            pdf_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith('_markup.pdf') and file != f"markups_{pdf_basename}.pdf":
                        pdf_files.append(os.path.join(root, file))
            
            if len(pdf_files) == 1:
                pdf_out = pdf_files[0]
            elif len(pdf_files) > 1:
                doc = fitz.open()
                for pdf_file in pdf_files:
                    src_doc = fitz.open(pdf_file)
                    doc.insert_pdf(src_doc)
                doc.save(pdf_out)
        except Exception as fitz_err:
            pass

        return current_df, ret_img, ret_img, True, json_zip, img_zip, pdf_out
        
    except Exception as e:
        print(f"Validation error: {e}")
        return current_df, None, None, False, None, None, None

with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as app:
    # HEADER
    with gr.Row(elem_classes="header-row"):
        gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center; width: 100%; padding: 10px 20px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="background-color: var(--text-primary); border-radius: 6px; padding: 4px; display: flex;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--bg-main)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                </div>
                <h2 style="margin: 0; font-size: 20px; font-weight: 700; letter-spacing: -0.5px; color: var(--text-primary);">QC Finish By Others</h2>
            </div>
            <a href="#" style="color: var(--text-secondary); text-decoration: none; display: flex; align-items: center; gap: 6px; font-size: 14px;">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path></svg>
                Help
            </a>
        </div>
        """)
        
    with gr.Row():
        # LEFT PANEL
        with gr.Column(scale=2, elem_classes="custom-panel"):
            gr.HTML('<h3 style="margin-top:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-primary); display: flex; align-items: center; gap: 8px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> DOCUMENT UPLOAD</h3>')
            
            gr.Markdown("#### Submittal PDF")
            sub_pdf = gr.File(label="", file_count="single", file_types=[".pdf"])
            
            gr.Markdown("#### Arch PDF")
            arch_pdf = gr.File(label="", file_count="single", file_types=[".pdf"])
            
            process_btn = gr.Button("⚙ Process Documents", elem_classes="primary-btn")
            
            status_html = gr.HTML("<div style='min-height: 40px; margin-top: 10px; color: var(--status-pass); font-weight: 500; font-size: 14px;'></div>")
            
            gr.HTML("<hr style='border-color: var(--border-color); margin: 20px 0;'>")
            
            view_dropdown = gr.Dropdown(label="SELECT VIEW", choices=[], interactive=False)
            rule_dropdown = gr.Dropdown(label="SELECT RULE", choices=[], multiselect=True, interactive=False)
            
            run_btn = gr.Button("▶ Run Validation", elem_classes="primary-btn")
            pdf_basename_state = gr.State(None)

        # MIDDLE PANEL
        with gr.Column(scale=4, elem_classes="custom-panel"):
            gr.HTML('<h3 style="margin-top:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-primary); display: flex; align-items: center; justify-content: space-between;">'
                    '<div style="display: flex; align-items: center; gap: 8px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> DRAWING VIEWER</div>'
                    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 3h6v6"></path><path d="M9 21H3v-6"></path><path d="M21 3l-7 7"></path><path d="M3 21l7-7"></path></svg></h3>')
            
            with gr.Row(elem_classes="toolbar-row"):
                btn_fit = gr.Button("Fit Width", elem_classes="icon-btn")
                btn_zi = gr.Button("Zoom In", elem_classes="icon-btn")
                btn_zo = gr.Button("Zoom Out", elem_classes="icon-btn")
                btn_100 = gr.Button("100%", elem_classes="icon-btn")
                btn_reset = gr.Button("Reset", elem_classes="icon-btn")
                btn_toggle = gr.Button("Toggle Markup", elem_classes="toggle-link")
                
            drawing_img = gr.Image(type="filepath", label="", show_label=False, interactive=False, height=450, elem_id="viewer_img")
            
            img_caption = gr.HTML("<div style='text-align: center; margin-top: 10px;'><span style='border-radius:50%; border: 1px solid var(--text-secondary); width: 24px; height: 24px; display: inline-block; line-height: 24px; margin-right: 8px;'>-</span><strong style='font-size: 16px;'>NO VIEW SELECTED</strong><br><span style='font-size: 12px; color: var(--text-secondary);'>Select a view to display details</span></div>")
            
            # States for toggling markup
            orig_img_state = gr.State(None)
            markup_img_state = gr.State(None)
            showing_markup_state = gr.State(False)
            
            gallery = gr.Gallery(label="Thumbnails", show_label=False, elem_id="gallery", columns=6, rows=1, height=120)

        # RIGHT PANEL
        with gr.Column(scale=4, elem_classes="custom-panel"):
            gr.HTML('<h3 style="margin-top:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-primary); display: flex; align-items: center; gap: 8px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg> VALIDATION RESULTS</h3>')
            
            results_table = gr.Dataframe(
                value=dummy_data,
                headers=["Rule ID", "Rule Name", "View ID", "View Name", "View Type", "Status", "Confidence", "Reasoning"],
                datatype=["markdown", "str", "markdown", "markdown", "str", "markdown", "str", "str"],
                interactive=False,
                wrap=True
            )
            
    # BOTTOM PANEL
    with gr.Row(elem_classes="custom-panel"):
        gr.HTML('<div style="width: 100%;"><h3 style="margin-top:0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: var(--text-primary); display: flex; align-items: center; gap: 8px;"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg> EXPORT RESULTS</h3></div>')
        
        with gr.Row():
            btn_dl_json = gr.DownloadButton("Download Result JSON (ZIP)", elem_classes="export-card-dl")
            btn_dl_img = gr.DownloadButton("Download Markup Images (ZIP)", elem_classes="export-card-dl")
            btn_dl_pdf = gr.DownloadButton("Download Markup PDF", elem_classes="export-card-dl")

    # Events
    process_btn.click(
        process_documents,
        inputs=[sub_pdf, arch_pdf],
        outputs=[status_html, view_dropdown, rule_dropdown, pdf_basename_state, gallery, results_table]
    )
    
    # Update view, caption, and enable rule dropdown when a view is selected
    def select_view_handler(view_val, pdf_basename):
        if not view_val or not pdf_basename:
            return gr.update(interactive=False), None, None, "<div style='text-align: center; margin-top: 10px;'>NO VIEW SELECTED</div>"
            
        crop_name = view_val.split("|")[0].strip()
        vname = view_val.split("|")[1].strip() if "|" in view_val else "Unknown View"
        
        meta_file = f"submittal_metadata_{pdf_basename}.json"
        src_path = None
        scale_val = "N/A"
        
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            target_meta = next((m for m in metadata if os.path.basename(m.get('source_file', '')) == crop_name), None)
            if target_meta:
                src_path = target_meta.get("source_file")
                scale_val = target_meta.get("scale", "N/A")
                
        # Format dynamic caption
        idx = view_dropdown.choices.index(view_val) + 1 if hasattr(view_dropdown, 'choices') and view_val in view_dropdown.choices else 1
        caption = f"<div style='text-align: center; margin-top: 10px;'><span style='border-radius:50%; border: 1px solid var(--text-secondary); width: 24px; height: 24px; display: inline-block; line-height: 24px; margin-right: 8px;'>{idx}</span><strong style='font-size: 16px; text-transform: uppercase;'>{vname}</strong><br><span style='font-size: 12px; color: var(--text-secondary);'>SCALE: {scale_val}</span></div>"
        
        # Filter rules based on view applicability
        try:
            import sys
            import importlib
            if "submittal" not in sys.modules:
                import submittal
            else:
                importlib.reload(submittal)
            get_matched_rule_view = submittal.get_matched_rule_view
        except Exception as e:
            get_matched_rule_view = None
            
        rules_files = glob.glob(os.path.join("rules", "*.json"))
        filtered_rules = []
        for rf in rules_files:
            try:
                with open(rf, 'r') as f:
                    rdata = json.load(f)
                    rlist = rdata if isinstance(rdata, list) else [rdata]
                    for r in rlist:
                        matched = False
                        if get_matched_rule_view:
                            # Use backend logic
                            matched = get_matched_rule_view(vname, r)
                        else:
                            # Fallback logic
                            req_views = [v.lower() for v in r.get("views_need_to_check", [])]
                            vtype = target_meta.get("view_type", "Unknown") if target_meta else "Unknown"
                            matched = any(rv in vtype.lower() or rv in vname.lower() for rv in req_views)
                        
                        if matched:
                            filtered_rules.append(f"{r.get('rule_id', 'Unknown')} | {r.get('rule_name', os.path.basename(rf))}")
            except Exception:
                pass
        
        return gr.update(interactive=True, choices=filtered_rules, value=None), src_path, src_path, caption
        
    view_dropdown.change(
        select_view_handler,
        inputs=[view_dropdown, pdf_basename_state],
        outputs=[rule_dropdown, drawing_img, orig_img_state, img_caption]
    )
    
    run_btn.click(
        run_validation,
        inputs=[view_dropdown, rule_dropdown, results_table, pdf_basename_state],
        outputs=[results_table, drawing_img, markup_img_state, showing_markup_state, btn_dl_json, btn_dl_img, btn_dl_pdf]
    )
    
    # Toggle Markup Logic
    def toggle_markup(orig, markup, currently_showing):
        if not markup or not orig: 
            return orig, currently_showing
        
        new_showing = not currently_showing
        img_to_show = markup if new_showing else orig
        return img_to_show, new_showing
        
    btn_toggle.click(
        toggle_markup,
        inputs=[orig_img_state, markup_img_state, showing_markup_state],
        outputs=[drawing_img, showing_markup_state]
    )
    
    # Zoom & Pan JS injection
    btn_zi.click(None, None, None, js="""() => {
        let img = document.querySelector('#viewer_img img');
        if(img) {
            let s = parseFloat(img.dataset.scale || 1.0) + 0.2;
            img.dataset.scale = s;
            img.style.transform = `scale(${s})`;
        }
    }""")
    
    btn_zo.click(None, None, None, js="""() => {
        let img = document.querySelector('#viewer_img img');
        if(img) {
            let s = Math.max(0.2, parseFloat(img.dataset.scale || 1.0) - 0.2);
            img.dataset.scale = s;
            img.style.transform = `scale(${s})`;
        }
    }""")
    
    btn_100.click(None, None, None, js="""() => {
        let img = document.querySelector('#viewer_img img');
        if(img) {
            img.dataset.scale = 1.0;
            img.style.transform = `scale(1.0)`;
            img.style.width = "auto";
            img.style.height = "auto";
            img.style.objectFit = "none";
        }
    }""")
    
    btn_fit.click(None, None, None, js="""() => {
        let img = document.querySelector('#viewer_img img');
        if(img) {
            img.dataset.scale = 1.0;
            img.style.transform = `scale(1.0)`;
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
        }
    }""")
    
    btn_reset.click(None, None, None, js="""() => {
        let img = document.querySelector('#viewer_img img');
        if(img) {
            img.dataset.scale = 1.0;
            img.style.transform = `scale(1.0)`;
            img.style.width = "100%";
            img.style.height = "100%";
            img.style.objectFit = "contain";
        }
    }""")

    # Set default theme to dark
    app.load(None, None, None, js="""() => {
        if (!localStorage.getItem("dark_mode_enforced")) {
            document.querySelector("html").classList.add("dark");
            localStorage.setItem("theme", "dark");
            localStorage.setItem("dark_mode_enforced", "true");
        }
    }""")

if __name__ == "__main__":
    app.queue().launch(max_file_size="500mb")
