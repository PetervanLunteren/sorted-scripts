# PAS OP: ik heb de output veranderd van txt naar xslx maar nog niet getest. Dit moet xlsx zijn to visualise results.




# Script to validate a yolov8 classifier on the test data is used during training
# It will test the precision, recall, and accuracy for all individual classes
# It is designed to be ran directly after the training has completed

'''
conda activate ecoassistcondaenv-yolov8 && python "/Users/peter/Documents/scripting/val-test-set.py" "<input_dir>" "<output_dir>"
'''

# import packages
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
from ultralytics import YOLO
import os
from sklearn import metrics # pip install scikit-learn
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import sys


# catch arguments
training_input_dir = os.path.normpath(sys.argv[1])
training_output_dir = os.path.normpath(sys.argv[2])

# log
print("\nRunning class-specific validation on test set...")
print(f"training_input_dir  : {training_input_dir}")
print(f"training_output_dir : {training_output_dir}")

# write results to xlsx
def dict2xlsx(output_dict, fname):
    df = pd.DataFrame(data=output_dict)
    df = (df.T)
    dst = os.path.join(os.path.dirname(os.path.dirname(model_fpath)), fname)
    df.to_excel(dst)
    print(f"Written results to {dst}")

# fetch classifications for single crop
def get_classification(img_fpath):
    results = model(img_fpath, verbose = False, imgsz = 224)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    classifications = []
    for idx, v in names_dict.items():
        classifications.append([v, probs[idx]])
    max_class = max(classifications, key=lambda x: x[1])
    return max_class

# crop detection with equal sides (Thanks Dan Morris)
def remove_background(img, bbox_norm):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    box_size = max(box_w, box_h)
    xmin = max(0, min(
        xmin - int((box_size - box_w) / 2),
        img_w - box_w))
    ymin = max(0, min(
        ymin - int((box_size - box_h) / 2),
        img_h - box_h))
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)
    if box_w == 0 or box_h == 0:
        return
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
    crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    return crop

# get subdir names which are the class names
def get_classes(test_dir):
    return [name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))]

# find highest number of train dir
def get_highest_train_subdir(main_folder):
    train_dirs = [dir for dir in os.listdir(main_folder) if dir.startswith('train')]
    if not train_dirs:
        print("No train directories found. Exiting...")
        exit()
    train_numbers = [int(dir[5:]) for dir in train_dirs if dir[5:].isdigit()]
    if not train_numbers:
        return os.path.join(main_folder, f'train')
    return os.path.join(main_folder, f'train{max(train_numbers)}')

# load model
model_fpath = os.path.join(get_highest_train_subdir(os.path.join(training_output_dir, "runs", "classify")), "weights", "best.pt")
model = YOLO(model_fpath)
print(f"\nLoaded model: {model_fpath}")

# validate on each subfolder
test_dir = os.path.join(training_input_dir, "test")

# count
total_imgs = 0
for true_class in get_classes(test_dir):
    imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
    total_imgs += len(imgs)

# predict
print("\nPredicting images on test set...\n")
y_pred = []
y_true = []
pbar = tqdm(total=total_imgs)
for true_class in get_classes(test_dir):
    imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
    for img in imgs:
        pred_class, conf = get_classification(img)
        y_pred.append(pred_class)
        y_true.append(true_class)
        pbar.update(1)

# calculate metrics
print("\nClassification report for test set:\n")
classification_report_str = metrics.classification_report(y_true, y_pred, zero_division = np.nan)

# print to console
print(classification_report_str) 

# write to file
# dst_xlsx = os.path.join(os.path.dirname(os.path.dirname(model_fpath)), "test-set-results.xlsx")
# with open(dst_txt, "w") as f:
#     f.write(classification_report_str)
# print(f"Written results to {dst_txt}")
output_dict = metrics.classification_report(y_true, y_pred, zero_division = np.nan, output_dict=True)
dict2xlsx(output_dict, "test-set-results.xlsx")