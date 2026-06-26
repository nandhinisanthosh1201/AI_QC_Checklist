import json

def extract_titles(json_path="vector_text_output.json", out_path="titles_output.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_items = []
    for page in data:
        # Some items might not have 'page' properly set if modified, let's ensure it's there
        for item in page.get("items", []):
            item["page"] = page.get("page", 1)
            all_items.append(item)

    # Sort all extracted text items by font_size in descending order
    all_items.sort(key=lambda x: x["font_size"], reverse=True)

    # Group texts by rounded font size to identify the title sizes
    size_groups = {}
    for item in all_items:
        size = round(item["font_size"], 1)
        if size not in size_groups:
            size_groups[size] = []
        size_groups[size].append(item)

    sizes = sorted(size_groups.keys(), reverse=True)
    
    print("--- Top Font Sizes and their sample texts ---")
    for size in sizes[:5]:
        texts = set(item["text"] for item in size_groups[size])
        print(f"Size {size}: {', '.join(list(texts)[:10])}")

    # Titles in architectural drawings are usually the largest text in the document.
    # We set a threshold to capture titles. You can adjust this based on the output.
    TITLE_MIN_SIZE = 14.0
    
    titles = [item for item in all_items if item["font_size"] >= TITLE_MIN_SIZE]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(titles, f, indent=2)

    print(f"\n✅ Extracted {len(titles)} potential titles (font size >= {TITLE_MIN_SIZE}) and saved to {out_path}.")
    print("\nSample titles with coordinates:")
    for t in titles[:10]:
        print(f"'{t['text']}' (Page {t['page']}) -> BBox: {t['bbox']}")

if __name__ == "__main__":
    extract_titles()
