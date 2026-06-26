# Task 3: Architectural Plumbing Schedule Extraction

## 1. The Approach: Geometric Grid-Aware Extraction
Standard table extraction tools (like Pandas or standard OCR algorithms) fail completely on architectural schedules because these schedules are heavily nested, containing "sub-rows" (multiple components per fixture) and "colspan headers" (headers spanning multiple columns). 

Our approach abandoned naive text parsing in favor of **Vector Geometry Analysis**. Since the PDF was exported from Revit, the table lines exist as physical mathematical vectors in the PDF. By using `pdfplumber`, we extracted the exact spatial coordinates (X, Y bounding boxes) of every single word and every single drawn line (`page.edges`). We then reconstructed the logical grid from these physical lines and spatially mapped the text into the correct cells.

---

## 2. Difficulties Faced & Solutions
We encountered several severe edge cases specific to architectural engineering drawings. Here is how we solved them:

### The "Phantom Row" Offset Bug 
* **Difficulty:** We initially calculated row boundaries by taking the mathematical halfway Y-point between tag names (e.g., halfway between `WC-1` and `WC-2`). However, because WC-1 had 4 sub-components stacked vertically, the sub-components extended *past* the mathematical midpoint, causing WC-1's bottom components to bleed into WC-2's data. 
* **Fix:** We replaced mathematical midpoints with physical geometry. We extracted only `major_h_lines` (drawn lines wider than 600 pixels, ignoring decorative text underlines) and snapped the row boundaries exactly to the physical borders drawn in the PDF.

### The Manufacturer Misalignment Shift (The "ZURN vs JR SMITH" Bug)
* **Difficulty:** Inside the Design columns, sub-components were listed with labels (e.g., `SUPPORT CARRIER:`). Our script sliced the cell exactly at the Y-coordinate of the label. However, the engineers physically printed the manufacturer name (`JR SMITH`) on the line *immediately above* the role label. This caused `JR SMITH` to be grouped into the previous component, shifting all manufacturers off by one.
* **Fix:** We implemented an intelligent "look-back" algorithm. When the script finds a role label, it checks the line immediately above it. If that line contains an unlabeled manufacturer name, the algorithm shifts the component boundary up by one line to capture it.

### Implicit "Ditto" Rows (The Blank Specs Bug)
* **Difficulty:** For fixtures like `KS-2`, the schedule only provided the modifier `(ACCESSIBLE)` and left the entire specifications block completely blank, expecting the reader to inherit the specs from `KS-1`. Our script originally extracted these as literally empty values.
* **Fix:** We implemented a targeted forward-fill logic pass. If a row's fixture name is clearly a variant modifier (starts with a parenthesis like `(ACCESSIBLE)`) and its specs are completely blank, the script programmatically copies the specs and base name from the row directly above it.

### Private Use Area Encodings (The `\uf06c` Bug)
* **Difficulty:** Architectural drawings use symbol fonts (like Wingdings) to print bullets indicating "applicable." These were being extracted as `\uf06c`, a raw Unicode codepoint. 
* **Fix:** We implemented explicit character mapping/stripping during the text-cleaning phase.

---

## 3. Technical Flow (End-to-End Pipeline)

### Phase 1: Grid & Boundary Discovery (`plumbing16_extractor.py`)
1. **Line Harvesting:** The script scans the PDF for long vertical lines to define rigid Column `x_start`/`x_end` boundaries.
2. **Header Classification:** It identifies the top row and groups columns into logical types (`TAG`, `FIXTURE`, `SPEC`, `DESIGN`) by keyword matching.
3. **Row Partitioning:** It locates every valid Tag (`[A-Z]+-\d+`) in the `TAG` column, then scans for major horizontal grid lines to calculate the exact `y_start` and `y_end` for every single fixture row.

### Phase 2: Spatial Text Mapping
1. **Word Assignment:** Every word object in the PDF is evaluated by its `(X, Y)` centroid and dropped into its corresponding `(Row, Column)` cell.
2. **Spec Assembly:** Text inside simple `SPEC` columns (like Pipe Size, Flow Rate) is concatenated and cleaned.

### Phase 3: Nested Component Parsing
1. **Y-Band Slicing:** The complex `DESIGN` columns (Manufacturer, Model, Description) are broken down further. The script scans the text for known `ROLE_MARKERS` (e.g., `FLUSH VALVE`, `SUPPORT CARRIER`).
2. **Sub-Row Extraction:** It creates horizontal "Y-bands" for each role and slices the text horizontally, assigning the exact manufacturer, model, and description to the correct sub-component object.
3. **Validation:** It counts the number of sub-components found versus the number expected based on the raw text and issues terminal warnings for mismatches.
4. **Post-Processing:** The script runs the "forward-fill" pass for variant rows and exports a pristine, nested `extracted_schedules.json` file.

### Phase 4: Automated Quality Control (`qc_cross_check.py`)
1. **Dictionary Merging:** The script loads the master general schedule and the newly extracted plumbing schedule, merging them into a unified `master_tags` dictionary.
2. **Ground Truth Comparison:** It loads the `ALL_PAGES_summary.json` (the AI-extracted annotations found throughout the drawing set).
3. **Discrepancy Reporting:** It loops through every tag found on the drawing pages and checks the `MAKE`, `MODEL`, `FINISH`, and `COMMENTS` against the deterministic master schedules. If `JR SMITH` is listed in the schedule but `SLOAN` is written on page 54, it logs it into a final `Deterministic_QC_Report.json` detailing exact mismatches for human review.
