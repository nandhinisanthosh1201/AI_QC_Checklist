import streamlit as st
import os
import uuid
from pathlib import Path
import subprocess
import json
import glob
from PIL import Image

# Streamlit page config
st.set_page_config(page_title="AI Arch Check Validation", layout="wide", page_icon="🏢")

# ---------------------------------------------------------
# STYLING & CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Add a bit of custom polish */
    .stButton>button {
        width: 100%;
        font-weight: bold;
        background-color: #0ea5e9;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0284c7;
        color: white;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.2);
    }
    .metric-card {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #334155;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        border: 1px solid #334155;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("🏢 AI-Based Arch Check Validation for Submittals Against Architectural Drawings")
st.markdown("Automated Room Name & Number Verification via Advanced Vision AI")

# ---------------------------------------------------------
# SIDEBAR: UPLOADS & ACTIONS
# ---------------------------------------------------------
with st.sidebar:
    st.header("New Analysis")
    
    submittal_upload = st.file_uploader("1. Upload Submittal Image", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
    arch_uploads = st.file_uploader("2. Upload Architectural Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    run_btn = st.button("Run Analysis", disabled=not (submittal_upload and arch_uploads))

# ---------------------------------------------------------
# EXECUTION LOGIC
# ---------------------------------------------------------
if run_btn:
    # Create live uploads folder
    upload_dir = Path("inputs/live_uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = uuid.uuid4().hex[:8]
    
    # Save submittal
    sub_ext = os.path.splitext(submittal_upload.name)[1]
    submittal_path = upload_dir / f"{run_id}_submittal{sub_ext}"
    with open(submittal_path, "wb") as f:
        f.write(submittal_upload.getbuffer())
        
    # Save arch images
    arch_paths = []
    for arch in arch_uploads:
        # retain original filename for matching, just prefix it
        arch_path = upload_dir / f"{run_id}_{arch.name}"
        with open(arch_path, "wb") as f:
            f.write(arch.getbuffer())
        arch_paths.append(str(arch_path))
        
    run_output_dir = f"outputs/run_{run_id}"
    
    # Build command
    cmd = [
        "python", "Roomnn-1.py",
        "--submittal", str(submittal_path),
        "--output-dir", run_output_dir
    ]
    if arch_paths:
        cmd.extend(["--arch"])
        cmd.extend(arch_paths)
        
    # Run subprocess with LIVE streaming output
    with st.status("Initializing AI Verification Pipeline...", expanded=True) as status:
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log_placeholder = st.empty()
            logs = []
            
            for line in iter(process.stdout.readline, ''):
                clean_line = line.strip()
                logs.append(clean_line)
                
                # Update status if it looks like a major stage change
                if "Step" in clean_line or "[Qwen]" in clean_line or "Analysing" in clean_line:
                    status.update(label=clean_line, state="running")
                
                # Keep logs readable (last 15 lines)
                log_placeholder.code("\\n".join(logs[-15:]), language="text")
                
            process.stdout.close()
            return_code = process.wait()
            
            if return_code != 0:
                status.update(label="Analysis Failed!", state="error", expanded=True)
                st.error("An error occurred during processing. See logs above.")
            else:
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # Find the newest JSON in the specific run output directory
                json_files = glob.glob(f"{run_output_dir}/**/arch_room_rule_result*.json", recursive=True)
                if json_files:
                    latest_json = max(json_files, key=os.path.getmtime)
                    st.session_state["latest_result"] = latest_json
                    st.session_state["uploaded_archs"] = arch_paths
                    st.session_state["submittal_path"] = str(submittal_path)
                else:
                    st.error("Script completed but no result JSON was found.")
        except Exception as e:
            status.update(label="Execution Error", state="error")
            st.error(f"Failed to run script: {e}")

# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------
if "latest_result" in st.session_state:
    try:
        with open(st.session_state["latest_result"], "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # --- Metrics ---
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Views", data.get("total_views", 0))
        col2.metric("Pass", data.get("pass_count", 0))
        col3.metric("Review Required", data.get("review_required_count", 0))
        col4.metric("Fail", data.get("fail_count", 0))
        
        st.divider()
        
        # --- Data Grid ---
        st.subheader("Extracted Views")
        views = data.get("view_validation_results", [])
        
        if views:
            # Convert to a flat dictionary for the dataframe
            df_data = []
            for v in views:
                rule = v.get("rule_result", {})
                
                sub_name = v.get('submittal_room_name') or '-'
                sub_num = v.get('submittal_room_number') or '-'
                arch_name = rule.get('architectural_room_name') or '-'
                arch_num = rule.get('architectural_room_number') or '-'
                
                df_data.append({
                    "View ID": v.get("view_id"),
                    "Subm Title": v.get("submittal_view_title") or '-',
                    "Subm Type": v.get("submittal_view_type") or '-',
                    "Subm Name": sub_name,
                    "Subm Number": sub_num,
                    "Arch Title": v.get("arch_view_title") or '-',
                    "Arch Name": arch_name,
                    "Arch Number": arch_num,
                    "Status": rule.get("status", "UNKNOWN"),
                    "Reason": rule.get("reason", "")
                })
            
            st.dataframe(df_data, use_container_width=True, hide_index=True)
        else:
            if data.get("error"):
                st.error(f"Model Error: {data.get('error')}")
            else:
                st.info("No views were extracted.")
            
        st.divider()
        
        # --- Visualizer ---
        st.subheader("Image Visualizer")
        
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            st.markdown("#### Submittal Markups")
            markup_path = data.get("markup_image")
            if markup_path and os.path.exists(markup_path):
                st.image(markup_path, use_container_width=True)
            else:
                st.info("All views passed. No markups generated.")
                if "submittal_path" in st.session_state and os.path.exists(st.session_state["submittal_path"]):
                    st.image(st.session_state["submittal_path"], caption="Original Submittal (No Failures)", use_container_width=True)

        with vis_col2:
            st.markdown("#### Architectural References")
            archs = st.session_state.get("uploaded_archs", [])
            if not archs:
                st.info("No architectural references were uploaded.")
            elif len(archs) == 1:
                st.image(archs[0], use_container_width=True)
            else:
                tabs = st.tabs([f"Arch File {i+1}" for i in range(len(archs))])
                for i, tab in enumerate(tabs):
                    with tab:
                        st.image(archs[i], caption=os.path.basename(archs[i]), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading results: {e}")
else:
    st.info("Upload files and click 'Run Analysis' to see results here.")
