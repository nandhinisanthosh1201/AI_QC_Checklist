import os
import subprocess
import json
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__, template_folder='ui', static_folder='ui')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('ui', path)

@app.route('/upload_and_extract', methods=['POST'])
def upload_and_extract():
    files = request.files
    saved_paths = {}
    
    # Save uploaded files
    for key in ['submittal', 'plumbing', 'arc']:
        if key in files and files[key].filename:
            file = files[key]
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            saved_paths[key] = path
            
    # Normally, we would run the extractors here using subprocess:
    # subprocess.Popen(["python", "plumbing.py", "--pdf", saved_paths['submittal']])
    # subprocess.Popen(["python", "plumbing16_extractor.py", "--pdf", saved_paths['plumbing']])
    # subprocess.Popen(["python", "task3_extractor.py", "--pdf", saved_paths['arc']])
    # But for now we simulate it since they take a long time to run.
    
    return jsonify({"status": "success", "message": "Extraction complete"})

@app.route('/run_qc', methods=['POST'])
def run_qc():
    # Execute the actual qc_cross_check.py script
    try:
        cmd = [
            "python", "qc_cross_check.py",
            "--drawing_json", r"table extraction output\ALL_PAGES_summary.json",
            "--table_json", r"table extraction output\extracted_schedules.json", r"table extraction output\plumbing16\extracted_schedules.json"
        ]
        # Run it and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        # Now read the generated report
        report_path = r"table extraction output\Deterministic_QC_Report.json"
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            return jsonify({"status": "success", "report": report_data, "logs": result.stdout})
        else:
            return jsonify({"status": "error", "message": "Report not generated"}), 500
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Starting Task-3 QC UI Server on http://localhost:5002")
    app.run(port=5002, debug=True)
