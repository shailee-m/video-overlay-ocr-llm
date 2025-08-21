import json

file_path = "./output/en_properties-gu/ocr_results_ordered.json"

# Step 1: Read the JSON file
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Filter out duplicates based on frame_index
seen = set()
cleaned_data = []
for entry in data:
    if entry["frame_index"] not in seen:
        cleaned_data.append(entry)
        seen.add(entry["frame_index"])

# Step 3: Write the cleaned data back to the file
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Removed {len(data) - len(cleaned_data)} duplicates. File updated: {file_path}")
