import os
import json
import xml.etree.ElementTree as ET

# Paths
xml_folder = "Dataset_Dog_Cat/Annotations"   # Folder where XML files are saved
output_json = "Dataset_Dog_Cat/annotations.json"

# Label mapping
label_map = {"cat": 1, "dog": 2}

annotations = {}

# Loop through all XML files
for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    filename = root.find("filename").text
    objects = root.findall("object")

    boxes = []
    labels = []

    for obj in objects:
        label = obj.find("name").text.lower()
        if label not in label_map:
            continue

        xmlbox = obj.find("bndbox")
        xmin = int(xmlbox.find("xmin").text)
        ymin = int(xmlbox.find("ymin").text)
        xmax = int(xmlbox.find("xmax").text)
        ymax = int(xmlbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])

    annotations[filename] = {"boxes": boxes, "labels": labels}

# Save to JSON
with open(output_json, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"✅ annotations.json created at: {output_json}")
