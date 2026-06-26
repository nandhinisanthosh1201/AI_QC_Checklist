from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import tablet_extractor
import uuid
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'table extraction output'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def process_extraction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    try:
        page_num = int(request.form.get('page', 1))
    except ValueError:
        page_num = 1
    
    # Save the file temporarily
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Run extractor
    # tablet_extractor page_num is 0-indexed
    extract_page_idx = max(0, page_num - 1)
    
    # Setup debug image path
    base_name = os.path.splitext(filename)[0]
    debug_img_name = f"{base_name}_page_{page_num}_debug.png"
    debug_path = os.path.join(OUTPUT_FOLDER, debug_img_name)
    
    try:
        # Call the extract method from tablet_extractor.py
        result = tablet_extractor.extract(filepath, debug_path=debug_path, page_num=extract_page_idx)
        
        output_data = {
            "success": True,
            "method": result.get("method", "unknown") if result else "None",
            "sections": result.get("sections", []) if result else [],
        }
        
        # Check if the debug image was generated
        if os.path.exists(debug_path):
            output_data['debug_image'] = f"/output/{debug_img_name}"
            
        return jsonify(output_data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
