import os
import json
import pandas as pd
from typing import Dict, Any

from src.utils.llm_client import call_vision_llm
from src.models.data_models import VisibleTagsOutput

def convert_schedule_excel_to_json(excel_path: str, output_json_path: str) -> None:
    """
    Converts a human-filled Excel schedule into the structured schedule.json format.
    Assumes each sheet in the Excel file represents a different schedule category 
    (e.g., 'APPLIANCE', 'RESILIENT WALL BASE').
    """
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
    print(f"Converting Schedule Excel to JSON: {excel_path}")
    
    # Read all sheets
    xls = pd.ExcelFile(excel_path)
    schedule_data = {}
    
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        # Convert NaN to None for clean JSON serialization
        df = df.where(pd.notnull(df), None)
        
        # Convert dataframe to list of dicts
        items = df.to_dict(orient="records")
        
        # Filter out completely empty rows
        cleaned_items = [item for item in items if any(val is not None for val in item.values())]
        
        schedule_data[sheet_name] = cleaned_items
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(schedule_data, f, indent=4)
        
    print(f"Schedule JSON saved to {output_json_path}")


def extract_visible_tags(arch_image_path: str, output_json_path: str) -> None:
    """
    Uses the Vision LLM to extract all visible identification tags 
    from the cropped architectural drawing.
    """
    if not os.path.exists(arch_image_path):
        raise FileNotFoundError(f"Architectural image not found: {arch_image_path}")
        
    print(f"Extracting visible tags from Architectural Crop: {arch_image_path}")
    
    prompt = """
    You are an expert architectural drafter. Please scan this architectural drawing crop.
    First, look for the title of the view (usually written at the bottom with a scale, e.g., "ELEVATION NORTH" or "8 - ELEVATION NORTH").
    Then, extract ALL text present in the drawing. You MUST extract every single piece of text, including:
    - Alphanumeric item codes and tags (e.g., MW-1, E-13, 17/AE542)
    - Short labels (e.g., "MICROWAVE", "TRASH")
    - Full sentences, construction notes, and paragraphs (e.g., "DECK MOUNTED SOAP DISPENSER", "SCHEDULED DOOR ASSEMBLY", "ACCESSIBLE SINK PER SCHEDULE")
    - Any text that has a leader line (arrow) pointing to an object.
    
    CRITICAL INSTRUCTION: Do NOT skip any text just because it looks like a regular sentence or paragraph. If it is written in the drawing, treat it as a 'tag' and extract it along with its bounding box!
    
    Treat every distinct block of text or sentence as a separate item in the 'visible_tags' list.
    
    Return the output STRICTLY in JSON format matching this schema:
    {
      "view_title": "ELEVATION NORTH",
      "visible_tags": [
        {
          "tag": "MW-1",
          "bbox": [x1, y1, x2, y2],
          "nearby_text": "Microwave"
        }
      ]
    }
    
    If no tags are found, return an empty list for 'visible_tags'.
    Do not include any markdown formatting like ```json in your response, just the raw JSON object.
    """
    
    try:
        response_text = call_vision_llm(prompt=prompt, image_path=arch_image_path, response_format="json_object")
        
        # Clean response if it contains markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json").strip("```").strip()
        elif response_text.startswith("```"):
            response_text = response_text.strip("```").strip()
            
        # Parse JSON to validate
        try:
            parsed_json = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode failed (likely truncated). Attempting simple repair... Error: {e}")
            try:
                parsed_json = json.loads(response_text + '"]}]}')
            except:
                try:
                    parsed_json = json.loads(response_text + '"}]}')
                except:
                    try:
                        parsed_json = json.loads(response_text + '}]}')
                    except:
                        print("Repair failed. Proceeding with empty tags list to prevent pipeline crash.")
                        parsed_json = {"visible_tags": []}

        # Validate against Pydantic model
        validated_data = VisibleTagsOutput(**parsed_json)
        
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(validated_data.model_dump(), f, indent=4)
            
        print(f"Extracted {len(validated_data.visible_tags)} tags. Saved to {output_json_path}")
        
    except Exception as e:
        print(f"Error extracting visible tags: {str(e)}")
        raise e
