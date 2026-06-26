import os
import fitz
import json
import base64
import requests
import io
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

# Import from master_pipeline
from master_pipeline import (
    load_dino_model, extract_vector_text, group_title_blocks, get_dino_clusters,
    API_KEY, EXTRA_PADDING
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("Loading Grounding DINO Model... This might take a moment.")
model, processor, device = load_dino_model()
print("Grounding DINO Model Loaded!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF file uploaded'}), 400
        
    file = request.files['pdf']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    page_num = int(request.form.get('page_num', 1))
    target_drawing = request.form.get('target_drawing', '').strip()
    prompt_text = request.form.get('prompt', '').strip()
    ai_model = request.form.get('ai_model', 'qwen/qwen3-vl-32b-instruct')
    action = request.form.get('action', 'annotate')
    
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(pdf_path)
    
    try:
        doc = fitz.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return jsonify({'error': f'Invalid page number. Document has {len(doc)} pages.'}), 400
            
        page = doc[page_num - 1] # 0-indexed
        
        # 1. Text & Titles
        items = extract_vector_text(page)
        title_blocks = group_title_blocks(items)
        
        if not title_blocks:
            return jsonify({'error': 'No title blocks found on this page.'}), 400
            
        # 2. DINO Clusters
        drawing_clusters = get_dino_clusters(page, model, processor, device)
        
        # 3. Match Titles
        final_drawings_pdf_bboxes = []
        
        if len(title_blocks) == 1:
            t = title_blocks[0]
            num_bbox = t["number"]["bbox"]
            tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
            sca_bbox = t["scale"]["bbox"]
            
            t_all_x0 = [x[0] for x in [num_bbox, tit_bbox, sca_bbox]]
            t_all_y0 = [x[1] for x in [num_bbox, tit_bbox, sca_bbox]]
            t_all_x1 = [x[2] for x in [num_bbox, tit_bbox, sca_bbox]]
            t_all_y1 = [x[3] for x in [num_bbox, tit_bbox, sca_bbox]]
            title_bbox_final = [min(t_all_x0), min(t_all_y0), max(t_all_x1), max(t_all_y1)]
            
            dino_bbox_final = None
            if drawing_clusters:
                d_x0 = min([c[0] for c in drawing_clusters])
                d_y0 = min([c[1] for c in drawing_clusters])
                d_x1 = max([c[2] for c in drawing_clusters])
                d_y1 = max([c[3] for c in drawing_clusters])
                dino_bbox_final = [d_x0, d_y0, d_x1, d_y1]

            all_x0, all_y0, all_x1, all_y1 = [], [], [], []
            for drawing in page.get_drawings():
                r = drawing["rect"]
                if r.width < 1 or r.height < 1: continue
                if (r.x1 - r.x0) > page.rect.width * 0.3 or (r.y1 - r.y0) > page.rect.height * 0.5: continue
                all_x0.append(r.x0); all_y0.append(r.y0); all_x1.append(r.x1); all_y1.append(r.y1)
                
            for i in items:
                if i["bbox"][0] > page.rect.width * 0.85: continue
                all_x0.append(i["bbox"][0]); all_y0.append(i["bbox"][1]); all_x1.append(i["bbox"][2]); all_y1.append(i["bbox"][3])
                
            if all_x0:
                fx0, fy0, fx1, fy1 = min(all_x0) - EXTRA_PADDING, min(all_y0) - EXTRA_PADDING, max(all_x1) + EXTRA_PADDING, max(all_y1) + EXTRA_PADDING
                final_drawings_pdf_bboxes.append({
                    "number_text": title_blocks[0]["number"]["text"],
                    "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)],
                    "title_bbox": title_bbox_final,
                    "dino_bbox": dino_bbox_final
                })
        else:
            for t in title_blocks:
                num_bbox = t["number"]["bbox"]
                tit_bbox = t["title"]["bbox"] if t["title"] else t["scale"]["bbox"]
                sca_bbox = t["scale"]["bbox"]
                
                all_x0 = [x[0] for x in [num_bbox, tit_bbox, sca_bbox]]
                all_y0 = [x[1] for x in [num_bbox, tit_bbox, sca_bbox]]
                all_x1 = [x[2] for x in [num_bbox, tit_bbox, sca_bbox]]
                all_y1 = [x[3] for x in [num_bbox, tit_bbox, sca_bbox]]
                
                tx0, ty0, tx1, ty1 = min(all_x0), min(all_y0), max(all_x1), max(all_y1)
                tx_c = (tx0 + tx1) / 2.0
                
                best_cluster = None
                min_score = float('inf')
                
                for cx0, cy0, cx1, cy1 in drawing_clusters:
                    ccx = (cx0 + cx1) / 2.0
                    ccy = (cy0 + cy1) / 2.0
                    if ccy > ty0: continue  
                    dx = abs(ccx - tx_c)
                    dy = abs(ty0 - cy1)
                    if dx > page.rect.width * 0.15: continue
                    score = (dx * 10.0) + dy
                    if score < min_score:
                        min_score = score
                        best_cluster = [cx0, cy0, cx1, cy1]
                
                if best_cluster:
                    fx0 = min(best_cluster[0], tx0) - EXTRA_PADDING
                    fy0 = min(best_cluster[1], ty0) - EXTRA_PADDING
                    fx1 = max(best_cluster[2], tx1) + EXTRA_PADDING
                    fy1 = max(best_cluster[3], ty1) + EXTRA_PADDING
                else:
                    fx0, fy0, fx1, fy1 = tx0 - EXTRA_PADDING, ty0 - EXTRA_PADDING, tx1 + EXTRA_PADDING, ty1 + EXTRA_PADDING
                    
                final_drawings_pdf_bboxes.append({
                    "number_text": t["number"]["text"],
                    "pdf_bbox": [max(0, fx0), max(0, fy0), min(page.rect.width, fx1), min(page.rect.height, fy1)],
                    "title_bbox": [tx0, ty0, tx1, ty1],
                    "dino_bbox": best_cluster
                })
                
        # Filter
        if target_drawing:
            final_drawings_pdf_bboxes = [d for d in final_drawings_pdf_bboxes if d["number_text"] == target_drawing]
            
        if not final_drawings_pdf_bboxes:
            return jsonify({'error': f"Target drawing '{target_drawing}' not found on page {page_num}."}), 404

        # Render Images
        scale_factor = 100.0 / 72.0
        mat = fitz.Matrix(scale_factor, scale_factor)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        output_images = []
        extraction_results = []
        
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        for i, data in enumerate(final_drawings_pdf_bboxes):
            pdf_bbox = data["pdf_bbox"]
            title_bbox = data.get("title_bbox")
            dino_bbox = data.get("dino_bbox")
            
            def scale_box(b):
                if b is None: return None
                return [int(b[0] * scale_factor), int(b[1] * scale_factor), int(b[2] * scale_factor), int(b[3] * scale_factor)]
                
            px_pdf = scale_box(pdf_bbox)
            px_title = scale_box(title_bbox)
            px_dino = scale_box(dino_bbox)
            
            full_img_copy = img.copy()
            draw_copy = ImageDraw.Draw(full_img_copy)
            
            if px_title: draw_copy.rectangle(px_title, outline="yellow", width=8)
            if px_dino: draw_copy.rectangle(px_dino, outline="blue", width=8)
            if px_pdf: draw_copy.rectangle(px_pdf, outline="green", width=15)
            
            safe_number_text = "".join(c for c in data['number_text'] if c.isalnum() or c in " -_").strip()
            if not safe_number_text: safe_number_text = f"unknown_{i}"
            out_filename = f"page_{page_num}_drawing_{safe_number_text}.jpg"
            out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)
            full_img_copy.save(out_path, format="JPEG", quality=85)
            
            output_images.append(f"/{app.config['OUTPUT_FOLDER']}/{out_filename}")
            
            if action == 'extract':
                # Full sheet at 100 DPI
                buf = io.BytesIO()
                full_img_copy.save(buf, format="JPEG", quality=85)
                b64_full = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                # High-res crop at 300 DPI
                zoom = 300.0 / 72.0
                mat_high = fitz.Matrix(zoom, zoom)
                rect = fitz.Rect(pdf_bbox)
                pix_crop = page.get_pixmap(matrix=mat_high, clip=rect)
                img_crop = Image.frombytes("RGB", [pix_crop.width, pix_crop.height], pix_crop.samples)
                
                buf_crop = io.BytesIO()
                img_crop.save(buf_crop, format="JPEG", quality=90)
                b64_crop = base64.b64encode(buf_crop.getvalue()).decode("utf-8")

                directive = "\n\nCRITICAL INSTRUCTION: I am providing TWO images. Image 1 is the full sheet for context (with a green box around the target). Image 2 is a HIGH-RESOLUTION CROP of ONLY the target drawing. You MUST use Image 2 to read the text, dimensions, and codes clearly. Do not guess or make up text; read it directly from the high-res crop. You may use Image 1 only if you need to reference a global sheet legend."
                
                default_prompt = "Extract details for the target drawing."
                qwen_prompt = prompt_text + directive if prompt_text else default_prompt + directive
                
                payload = {
                    "model": ai_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": qwen_prompt},
                                {"type": "text", "text": "Image 1 (Full Sheet Context):"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_full}"}},
                                {"type": "text", "text": "Image 2 (High-Res Crop of Target Drawing):"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}}
                            ]
                        }
                    ],
                    "max_tokens": 1024
                }
                
                try:
                    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
                    if res.status_code == 200:
                        content = res.json()["choices"][0]["message"]["content"]
                        if "```json" in content: content = content.split("```json")[1].split("```")[0].strip()
                        elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
                        try:
                            parsed = json.loads(content)
                            extraction_results.append({"drawing": data['number_text'], "data": parsed})
                        except:
                            extraction_results.append({"drawing": data['number_text'], "raw": content})
                    else:
                        extraction_results.append({"drawing": data['number_text'], "error": res.text})
                except Exception as e:
                    extraction_results.append({"drawing": data['number_text'], "error": str(e)})

        return jsonify({
            "images": output_images,
            "extraction": extraction_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)
