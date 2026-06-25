import os
import glob
import json
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for the frontend

BASE_DIR = os.path.dirname(os.path.abspath(__name__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
INPUTS_DIR = os.path.join(BASE_DIR, "inputs")

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

@app.route('/')
def serve_frontend_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filepath>')
def serve_frontend_assets(filepath):
    return send_from_directory(FRONTEND_DIR, filepath)

@app.route('/api/results', methods=['GET'])
def get_results():
    results = []
    # Find all arch_room_rule_result_*.json files in outputs directory and its subdirectories
    search_pattern = os.path.join(OUTPUTS_DIR, '**', 'arch_room_rule_result_*.json')
    for filepath in glob.glob(search_pattern, recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # To easily map paths, inject a view_id if missing or structure it
                results.append(data)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    
    # Sort results by run_timestamp descending
    results.sort(key=lambda x: x.get('run_timestamp', ''), reverse=True)
    return jsonify(results)

@app.route('/api/images/outputs/<path:filepath>', methods=['GET'])
def serve_output_image(filepath):
    # Ensure secure path joining
    return send_from_directory(OUTPUTS_DIR, filepath)

@app.route('/api/images/inputs/<path:filepath>', methods=['GET'])
def serve_input_image(filepath):
    # Ensure secure path joining
    return send_from_directory(INPUTS_DIR, filepath)

import subprocess
import uuid

UPLOADS_DIR = os.path.join(INPUTS_DIR, "live_uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

@app.route('/api/run', methods=['POST'])
def run_live():
    if 'submittal' not in request.files:
        return jsonify({"error": "No submittal file provided"}), 400
        
    submittal_file = request.files['submittal']
    arch_files = request.files.getlist('arch_images')
    
    # Save files with unique prefixes to avoid collisions
    run_id = str(uuid.uuid4())[:8]
    
    submittal_path = os.path.join(UPLOADS_DIR, f"{run_id}_{submittal_file.filename}")
    submittal_file.save(submittal_path)
    
    arch_paths = []
    for af in arch_files:
        if af.filename:
            apath = os.path.join(UPLOADS_DIR, f"{run_id}_{af.filename}")
            af.save(apath)
            arch_paths.append(apath)
            
    # Build command
    cmd = ["python", "Roomnn-1.py", "--submittal", submittal_path, "--output-dir", OUTPUTS_DIR]
    if arch_paths:
        cmd.extend(["--arch"] + arch_paths)
        
    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE_DIR)
        
        if result.returncode != 0:
            return jsonify({"error": "Script execution failed", "logs": result.stderr or result.stdout}), 500
            
        # Find the latest JSON generated for this submittal
        sub_stem = os.path.splitext(os.path.basename(submittal_path))[0]
        json_pattern = os.path.join(OUTPUTS_DIR, sub_stem, 'arch_room_rule_result_*.json')
        json_files = glob.glob(json_pattern)
        if not json_files:
            return jsonify({"error": "Script finished but no JSON output was found.", "logs": result.stdout}), 500
            
        latest_json = max(json_files, key=os.path.getctime)
        with open(latest_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return jsonify(data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
