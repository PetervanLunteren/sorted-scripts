# Script to validate a yolov8 classifier on the test data is used during training, and on a human verified out-of-sample dataset. 
# This out-of-sample dataset should consist of crops of animals which are > 0.6 MD confidence.
# The true label should be in the filename before underscore: 'zebra_1457.jpg'
# The dataset on which the model trained should still be in tact. The metrics yielded from another testset will not be very helpful.
# It will test the precision, recall, and accuracy for all individual classes.

'''
conda activate ecoassistcondaenv-yolov8 && python "C:\Peter\training-utils\scripts\4-class-specific-validation.py"
'''

########## USER SETTINGS ############

model_fpath = None # If you set this to None, it will take the last trained model
training_output_dir = r"C:\Peter\desert-lion-project\model-development\training-output" # not neccesary if model fpath is provided

# do you want the model to validate on a test set? This should be the test split on which the model was trained.
validate_test_dir = False 
test_dir = r"C:\Peter\training-utils\current-train-set\test"

# if there is a human verified out-of-sample dataset / calibration dataset
# we can run the model over that to see how well it performs on those crops
validate_out_of_sample_dataset = True # this should ne a human verified out-of-sample datset annotated with PASCAL VOC xml files
out_of_sample_dir = r"C:\Users\smart\Desktop\out-of-sample-dataset\pass" 

# import packages
import warnings
warnings.filterwarnings('always')
from ultralytics import YOLO
import os
from sklearn import metrics # pip install scikit-learn
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd

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

# write results to xlsx
def dict2xlsx(output_dict, fname):
    df = pd.DataFrame(data=output_dict)
    df = (df.T)
    dst = os.path.join(os.path.dirname(os.path.dirname(model_fpath)), fname)
    df.to_excel(dst)
    print(f"Written results to {dst}")

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
model_fpath = os.path.join(get_highest_train_subdir(os.path.join(training_output_dir, "runs", "classify")), "weights", "best.pt") if model_fpath is None else model_fpath
model = YOLO(model_fpath)
print(f"\nLoaded model: {model_fpath}")

# if you specified to validate the test directory
if validate_test_dir:
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
    print(metrics.classification_report(y_true, y_pred, zero_division = np.nan))
    output_dict = metrics.classification_report(y_true, y_pred, zero_division = np.nan, output_dict=True)
    dict2xlsx(output_dict, "test-set-results.xlsx")

# here it will loop over the crops inside this folder and extract true_label from filename (lion_00236.jpg).
if validate_out_of_sample_dataset:
    y_pred = []
    y_true = []
    imgs = os.listdir(out_of_sample_dir)
    for img in tqdm(imgs):
        true_label = img.split('_')[0]
        img_fpath = os.path.join(out_of_sample_dir, img)
        crop = Image.open(img_fpath)
        pred_label, _ = get_classification(crop.resize((224, 224)))
        y_pred.append(pred_label)
        y_true.append(true_label)

    # calculate metrics
    print("\nClassification report for out-of-sample dataset:\n")
    print(metrics.classification_report(y_true, y_pred, zero_division = np.nan))
    output_dict = metrics.classification_report(y_true, y_pred, zero_division = np.nan, output_dict=True)
    dict2xlsx(output_dict, "out-of-of-sample-results.xlsx")


exit()






################ this is how i checked which crops passed the 0.6 conf MD trhesh
################ cropping was done with the code below that.
# import json
# import os
# from pathlib import Path
# import shutil

# recognition_file = r"C:\Users\smart\Desktop\out-of-sample-dataset\image_recognition_file.json"

# # open json file
# with open(recognition_file) as image_recognition_file_content:
#     data = json.load(image_recognition_file_content)

# n_below = 0
# n_above = 0

# for image in data['images']:
#     file = image['file']
#     src = os.path.join(os.path.dirname(recognition_file), file)
#     if 'detections' in image:
#         for detection in image['detections']:
#             has_conf_above_thresh = False
#             conf = detection["conf"]
#             if conf > 0.6:
#                 has_conf_above_thresh = True
        
#         if not has_conf_above_thresh:
#             n_below += 1
#             typ = "fail"
#         else:
#             n_above += 1
#             typ = "pass"
#         dst = os.path.join(os.path.dirname(recognition_file), typ, file)
#         Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
#         shutil.move(src, dst)
#         # print(dst)


# print(f"n_below : {n_below}")
# print(f"n_above : {n_above}")








# if you specified to validate a PSACAL VOC annotated out-of-sample dataset
if validate_out_of_sample_dataset:

    # count
    n_crops = 0
    xml_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(out_of_sample_dir) for f in filenames if os.path.splitext(f)[1] == '.xml']
    for xml_file in xml_files:
        ann_root = ET.parse(xml_file).getroot()
        for obj in ann_root.findall('object'):
            true_label = obj.findtext('name')
            if true_label in ['animal', 'person', 'vehicle']:
                continue
            n_crops += 1


    # read info from xml files and run model over them
    print("\nPredicting images on out-of-sample set...\n")
    y_pred = []
    y_true = []
    i = 1
    pbar = tqdm(total=n_crops)
    for xml_file in xml_files:

        # read  xml
        ann_tree = ET.parse(xml_file)
        ann_root = ann_tree.getroot()

        # get image path
        img_ext = os.path.splitext(ann_root.findtext('filename'))[1]
        img_full_path = os.path.splitext(xml_file)[0] + img_ext

        # read resolution
        size = ann_root.find('size')
        im_width = int(size.findtext('width'))
        im_height = int(size.findtext('height'))

        # loop through xml detections 
        for obj in ann_root.findall('object'):
            true_label = obj.findtext('name')

            # covert to new label if specified
            if true_label in convert_classes:
                true_label = convert_classes[true_label]

            # we're not interested in the MD classes
            if true_label in ['animal', 'person', 'vehicle']:
                continue

            # read bbox
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.findtext('xmin')))
            ymin = int(float(bndbox.findtext('ymin')))
            xmax = int(float(bndbox.findtext('xmax')))
            ymax = int(float(bndbox.findtext('ymax')))

            # convert
            w_box = round(abs(xmax - xmin) / im_width, 5)
            h_box = round(abs(ymax - ymin) / im_height, 5)
            xo = round(xmin / im_width, 5)
            yo = round(ymin / im_height, 5)
            bbox = [xo, yo, w_box, h_box]

            # crop
            crop = remove_background(Image.open(img_full_path), bbox)

            # # save
            # crop.save(os.path.join(r"C:\Users\smart\Desktop\out-of-sample-dataset", f'{true_label}_{i}.jpg'))
            # i += 1
            
            # predict
            pred_label, _ = get_classification(crop.resize((224, 224)))

            # add to lists
            y_pred.append(pred_label)
            y_true.append(true_label)

            # update prgressbar
            pbar.update(1)

    # calculate metrics
    print("\nClassification report for out-of-sample dataset:\n")
    print(metrics.classification_report(y_true, y_pred, zero_division = np.nan))