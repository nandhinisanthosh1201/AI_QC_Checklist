# Grain Direction & Architectural Casework QC System

An automated Computer Vision and Rules Engine pipeline to audit architectural submittal drawings for material finish tags and grain direction compliance.

## Project Structure

```
.
├── step1_web_scrapping/                # Step 1: Web scraping & multimodal material extraction
│   ├── main_pipeline.py
│   ├── extractor.py
│   ├── web_search_engine.py
│   ├── grain_verifier.py
│   └── analyze_grain.py
│
├── step2_grain_symbols/                # Step 2: Vector CAD grain symbol scanner
│   └── step2_grain_detector.py
│
├── step3_elevation_crops/              # Step 3: Elevation title-block detector & cropper
│   └── step3_elevation_cropper.py
│
├── step4_component_detection/          # Step 4: YOLO casework component detector
│   └── detect_components.py
│
├── step5_material_tags/                # Step 5: Material tag OCR & spatial association
│   └── step5_material_tags.py
│
├── step6_rule_engine/                  # Step 6: Material tag matching & bank inheritance
│   └── step6_rule_engine.py
│
├── step7_direction_engine/             # Step 7: Grain direction compliance validator
│   └── step7_direction_engine.py
│
├── run_pipeline.py                     # Master CLI runner for Steps 1–7
├── requirements.txt                    # Python dependencies
└── .gitignore                          # Git ignore rules
```

## Setup & Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all 7 steps end-to-end:

```bash
python run_pipeline.py --pdf <path_to_submittal.pdf> --mat-tab <path_to_material_tab.png> --project <project_name>
```

Run specific steps (e.g., steps 6 and 7):

```bash
python run_pipeline.py --pdf <path_to_submittal.pdf> --mat-tab <path_to_material_tab.png> --project <project_name> --steps 6,7
```
