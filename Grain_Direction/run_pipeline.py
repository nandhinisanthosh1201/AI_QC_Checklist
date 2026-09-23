#!/usr/bin/env python3
"""
run_pipeline.py
===============
Master runner for the 7-Step Grain Direction & Material QC Pipeline.

Folder Structure:
    ├── step1_material_scraping/
    │   └── main_pipeline.py
    ├── step2_grain_detector.py
    ├── step3_elevation_cropper.py
    ├── step4_component_detector.py
    ├── step5_material_tags.py
    ├── step6_rule_engine.py
    ├── step7_direction_engine.py
    └── run_pipeline.py

Usage:
    python run_pipeline.py --pdf <submittal.pdf> --mat-tab <mat_tab.png> --project <project_name>
"""

import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    else:
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
import subprocess
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
OUTPUTS    = ROOT_DIR / "outputs"
MODELS_DIR = ROOT_DIR / "models"


def proj_dir(project_name: str) -> Path:
    return OUTPUTS / project_name


def banner(step_num, title):
    print("\n" + "=" * 70)
    print(f"  STEP {step_num}: {title}")
    print("=" * 70)


def run_step1(mat_tab_img: Path, project_name: str, force: bool = False) -> bool:
    banner(1, "MATERIAL SCRAPING (Vision AI + Web Search)")
    out_dir = proj_dir(project_name) / "Step1_MaterialScraping"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = out_dir / "materials_cache.json"
    if cache_path.exists() and not force:
        print(f"[SKIP] Step 1 already done: materials_cache.json found.")
        return True

    script = ROOT_DIR / "step1_web_scrapping" / "main_pipeline.py"
    step1_cwd = ROOT_DIR / "step1_web_scrapping"
    cmd = [sys.executable, "-u", str(script), str(mat_tab_img), str(out_dir)]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(step1_cwd))
    if result.returncode != 0:
        print("[ERROR] Step 1 failed!")
        return False
    print("[OK] Step 1 complete.")
    return True


def run_step2(pdf_path: Path, project_name: str, force: bool = False) -> bool:
    banner(2, "GRAIN SYMBOL DETECTION")
    out_dir = proj_dir(project_name) / "Step2_GrainSymbols"
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_file = out_dir / "candidates.csv"
    if cand_file.exists() and not force:
        print("[SKIP] Step 2 already done. candidates.csv found.")
        return True

    script = ROOT_DIR / "step2_grain_symbols" / "step2_grain_detector.py"
    cmd = [
        sys.executable, "-u", str(script),
        str(pdf_path),
        "--out", str(out_dir),
    ]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[WARN] Step 2 returned non-zero. Continuing...")
    else:
        print("[OK] Step 2 complete.")
    return True


def run_step3(pdf_path: Path, project_name: str, force: bool = False) -> bool:
    banner(3, "ELEVATION CROP DETECTION")
    pdf_stem = pdf_path.stem
    step3_base = proj_dir(project_name) / "Step3_ElevationCrops"
    target_out = step3_base / f"{pdf_stem}_annotated"

    if target_out.exists() and force:
        import shutil
        shutil.rmtree(target_out, ignore_errors=True)
    elif man_file.exists() and not force:
        print(f"[SKIP] Step 3 already done. manifest.csv found.")
        return True

    step3_base.mkdir(parents=True, exist_ok=True)
    annotated_pdf = proj_dir(project_name) / "Step2_GrainSymbols" / f"{pdf_stem}_annotated.pdf"
    input_pdf = annotated_pdf if annotated_pdf.exists() else pdf_path

    script = ROOT_DIR / "step3_elevation_crops" / "step3_elevation_cropper.py"
    cmd = [
        sys.executable, "-u", str(script),
        str(input_pdf),
        str(step3_base),
        "--filter", "elevation",
    ]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Step 3 failed!")
        return False
    print("[OK] Step 3 complete.")
    return True


def run_step4(project_name: str, force: bool = False) -> bool:
    banner(4, "COMPONENT DETECTION (YOLO)")
    step3_base = proj_dir(project_name) / "Step3_ElevationCrops"
    crops_dir  = None

    for folder in step3_base.rglob("crops"):
        if folder.is_dir() and list(folder.glob("*.png")):
            crops_dir = folder
            break

    if crops_dir is None:
        for folder in step3_base.rglob("*"):
            if folder.is_dir() and folder.name != "Step3_ElevationCrops":
                pngs = list(folder.glob("p*.png"))
                if pngs and any("bboxes" not in p.name for p in pngs):
                    crops_dir = folder
                    break

    if crops_dir is None:
        print(f"[ERROR] No cropped PNGs found under {step3_base}")
        return False

    out_dir = proj_dir(project_name) / "Step4_ComponentDetection"
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_file = out_dir / "components_data.json"
    if comp_file.exists() and not force:
        print("[SKIP] Step 4 already done.")
        return True

    script = ROOT_DIR / "step4_component_detection" / "detect_components.py"
    model_path = MODELS_DIR / "best_production_model.pt"
    cmd = [
        sys.executable, "-u", str(script),
        "--input",  str(crops_dir),
        "--output", str(out_dir),
    ]
    if model_path.exists():
        cmd.extend(["--model", str(model_path)])

    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Step 4 failed!")
        return False
    print("[OK] Step 4 complete.")
    return True


def run_step5(pdf_path: Path, project_name: str, force: bool = False) -> bool:
    banner(5, "MATERIAL TAG DETECTION")
    step3_base = proj_dir(project_name) / "Step3_ElevationCrops"
    manifest_path = None

    for root, dirs, files in os.walk(str(step3_base)):
        if "manifest.csv" in files:
            manifest_path = Path(root) / "manifest.csv"
            break

    if manifest_path is None:
        print(f"[ERROR] manifest.csv not found under {step3_base}.")
        return False

    step1_dir = proj_dir(project_name) / "Step1_MaterialScraping"
    step4_dir = proj_dir(project_name) / "Step4_ComponentDetection"
    out_dir   = proj_dir(project_name) / "Step5_MaterialTags"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag_file = out_dir / "tags_data.json"
    if tag_file.exists() and not force:
        print("[SKIP] Step 5 already done.")
        return True

    script = ROOT_DIR / "step5_material_tags" / "step5_material_tags.py"
    cmd = [
        sys.executable, "-u", str(script),
        "--pdf",      str(pdf_path),
        "--manifest", str(manifest_path),
        "--step1",    str(step1_dir),
        "--step4",    str(step4_dir),
        "--output",   str(out_dir),
    ]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Step 5 failed!")
        return False
    print("[OK] Step 5 complete.")
    return True


def run_step6(project_name: str, force: bool = False) -> bool:
    banner(6, "MATERIAL TAG & GRAIN PRESENCE QC")
    script = ROOT_DIR / "step6_rule_engine" / "step6_rule_engine.py"
    cmd = [
        sys.executable, "-u", str(script),
        "--outputs-dir", str(OUTPUTS),
        "--project",     project_name,
    ]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Step 6 failed!")
        return False
    print("[OK] Step 6 complete.")
    return True


def run_step7(project_name: str, force: bool = False) -> bool:
    banner(7, "GRAIN DIRECTION COMPLIANCE QC (VERTICAL VS HORIZONTAL)")
    script = ROOT_DIR / "step7_direction_engine" / "step7_direction_engine.py"
    cmd = [
        sys.executable, "-u", str(script),
        "--outputs-dir", str(OUTPUTS),
        "--project",     project_name,
    ]
    print(f"Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print("[ERROR] Step 7 failed!")
        return False
    print("[OK] Step 7 complete.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Grain Direction Pipeline — Run all 7 steps")
    ap.add_argument("--pdf",     required=True,  help="Path to the input submittal PDF")
    ap.add_argument("--mat-tab", required=True,  help="Path to the material tab image")
    ap.add_argument("--project", required=True,  help="Project folder name")
    ap.add_argument("--steps",   default="1,2,3,4,5,6,7", help="Steps to run, e.g. --steps 6,7")
    ap.add_argument("--force",   action="store_true", help="Force re-run even if outputs exist")
    args = ap.parse_args()

    pdf_path    = Path(args.pdf).resolve()
    mat_tab_img = Path(args.mat_tab).resolve()
    project     = args.project
    steps       = [int(s.strip()) for s in args.steps.split(",")]
    force       = args.force

    if not pdf_path.exists():
        print(f"[ERROR] PDF not found: {pdf_path}"); sys.exit(1)
    if not mat_tab_img.exists():
        print(f"[ERROR] Mat tab image not found: {mat_tab_img}"); sys.exit(1)

    (OUTPUTS / project).mkdir(parents=True, exist_ok=True)

    step_map = {
        1: lambda: run_step1(mat_tab_img, project, force=force),
        2: lambda: run_step2(pdf_path,    project, force=force),
        3: lambda: run_step3(pdf_path,    project, force=force),
        4: lambda: run_step4(project, force=force),
        5: lambda: run_step5(pdf_path,    project, force=force),
        6: lambda: run_step6(project, force=force),
        7: lambda: run_step7(project, force=force),
    }

    for s in steps:
        ok = step_map[s]()
        if not ok:
            print(f"\n[PIPELINE STOPPED] Step {s} failed.")
            sys.exit(1)

    print("\n" + "=" * 70)
    print(f"  ALL STEPS COMPLETE - {project}")
    print(f"  Outputs saved in: {OUTPUTS / project}")
    print("=" * 70)


if __name__ == "__main__":
    main()
