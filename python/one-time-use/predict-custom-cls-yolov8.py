# Script to add species specific classifications to detections
# Peter van Lunteren, 4 october 2023

from ultralytics import YOLO
import json
import numpy as np
import os
from crop_detections import *
import tempfile

# set PYTHONPATH=%PYTHONPATH%;C:\Peter\classifier\MegaDetector\classification


model_path = r'C:\Users\smart\runs\classify\train4\weights\best.pt'
json_path = r'C:\Users\smart\Desktop\test-classify-dir\image_recognition_file.json'
cls_thresh = 0.4
model = YOLO(model_path)

# fetch classifications for single crop
def get_classification(img_fpath):
    results = model(img_fpath)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    classifications = []
    for idx, v in names_dict.items():
        classifications.append([v, probs[idx]])
    return classifications

# fetch label map from json
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# run through json and convert detections to classficiations
index = 0
temp_dir = tempfile.TemporaryDirectory()
with open(json_path) as image_recognition_file_content:
    data = json.load(image_recognition_file_content)
    label_map = fetch_label_map_from_json(json_path)
    if 'classification_categories' not in data:
        data['classification_categories'] = {}
    inverted_cls_label_map = {v: k for k, v in data['classification_categories'].items()}
    for image in data['images']:
        fname = image['file']
        for detection in image['detections']:
            conf = detection["conf"]
            category_id = detection['category']
            category = label_map[category_id]
            if conf >= cls_thresh and category == 'animal':
                img_fpath = os.path.join(os.path.dirname(json_path), fname)
                bbox = detection['bbox']
                crop_path = os.path.join(temp_dir.name, f'crop_{index}.jpg')
                index += 1
                pil_img = Image.open(img_fpath)
                save_crop(pil_img, bbox, square_crop = True, save = crop_path)
                name_classifications = get_classification(crop_path)

                # check if name already in classification_categories
                idx_classifications = []
                for elem in name_classifications:
                    name = elem[0]
                    if name not in inverted_cls_label_map:
                        highest_index = 0
                        for key, value in inverted_cls_label_map.items():
                            value = int(value)
                            if value > highest_index:
                                highest_index = value
                        inverted_cls_label_map[name] = str(highest_index + 1)
                    idx_classifications.append([inverted_cls_label_map[name], round(elem[1], 5)])
                
                # name_idx = inverted_cls_label_map[name]


                idx_classifications = sorted(idx_classifications, key=lambda x:x[1], reverse=True)
                detection['classifications'] = idx_classifications

                if os.path.exists(crop_path):
                    os.remove(crop_path)
                # TODO: maak hier een tempfile van die weer wordt verwijderd
                # print(f"We've got one             : {fname}")
                # print(f"      With classification : {name}, {prob}")
    

data['classification_categories'] = {v: k for k, v in inverted_cls_label_map.items()}
with open(json_path, "w") as json_file:
    json.dump(data, json_file, indent=1)

temp_dir_path = temp_dir.name

temp_dir.cleanup()

if os.path.exists(temp_dir_path):
    print("temp dir exists")
else:
    print("temp dir deleted")

                # "classification_categories" next to detection_categories

                # "classifications": [
                #         ["1000004", 1.0]
                #     ]






# print(f"prob_pred   : {prob_pred}")
# print(f"name_pred   : {name_pred}")
# name_pred = names_dict[np.argmax(probs)]
# prob_pred = np.argmax(probs)

# print(f"names_dict : {names_dict}")
# print(f"probs      : {probs}")

# print(f"prob_pred  : {prob_pred}")

# print(f"names_dict[np.argmax(probs)] : {names_dict[np.argmax(probs)]}")