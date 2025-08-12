# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\convert-pascal-to-yolo.py"

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
from pathlib import Path

VOC_DIR = r"D:\2024-16-SCO\imgs\20240210\raw\final-dataset\labels-voc-species"
YOLO_DIR = r"D:\2024-16-SCO\imgs\20240210\raw\final-dataset\labels-yolo-species"
CLASSES_FILE = os.path.join(YOLO_DIR, "classes.txt")

def get_classes(voc_dir):
    classes = set()
    for file in os.listdir(voc_dir):
        if not file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(voc_dir, file))
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.add(class_name)
    return list(classes)

def save_classes(classes, file_path):
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

def convert_voc_to_yolo(voc_dir, yolo_dir, classes):
    
    if not os.path.exists(yolo_dir):
        os.makedirs(yolo_dir)

    for file in tqdm(os.listdir(voc_dir)):
        if not file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(voc_dir, file))
        root = tree.getroot()

        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)

        yolo_annotations = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in classes:
                continue
            class_id = classes.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            yolo_annotations.append(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}")

        yolo_file = os.path.join(yolo_dir, file.replace(".xml", ".txt"))
        Path(os.path.dirname(yolo_file)).mkdir(parents=True, exist_ok=True)
        with open(yolo_file, 'w') as f:
            f.write("\n".join(yolo_annotations))

if __name__ == "__main__":
    print("Extracting classes...")
    files = [file for file in os.listdir(VOC_DIR) if file.endswith(".xml")]
    classes = set()
    for file in tqdm(files):
        tree = ET.parse(os.path.join(VOC_DIR, file))
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            classes.add(class_name)
    classes = list(classes)
    save_classes(classes, CLASSES_FILE)
    
    print("Converting annotations...")
    convert_voc_to_yolo(VOC_DIR, YOLO_DIR, classes)