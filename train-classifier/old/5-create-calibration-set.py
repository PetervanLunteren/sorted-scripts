# Script to loop through json and find random sequences for each class to human verify and then use as calibration dataset

# 1. you first need to devide all sequences into separate PvL_seq_xxxxx subfolders, use 1-sort-dir-in-sequences.py for that
# 2. then you need to run EA (with a classifier) over the client provided images, make sure the categories are final, otherwise you'll have to repeat this step
# 3. then run this script which will then take random sequences for each class and move them to a dst_dir
# 4. then run EA again on dst_dir, and human verify all the images, save the somewhere inside the project folder.
# 5. when that is done, you can proceed with the ecosystem_calibration.py

# """
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\4-create-calibration-set.py"
# 
# conda activate ecoassistcondaenv && python /Users/peter/Documents/scripting/sorted-scripts/python/train-classifier/4-create-calibration-set.py
# """

# Peter van Lunteren, 12 dec 2023

# USER INPUT
json_fpath = r"C:\Peter\desert-lion-project\philips-footage\unsorted\IntoNature4\CT1+2\image_recognition_file.json"
dst_dir = r"C:\Users\smart\Desktop\calibration_set"

# above which threshold do we want that animals are classified? If it finds one animal, it 
# will take the entire sequence.
conf_thresh = 0.6 

# how many images per class do you want in your claibration dataset? It works with sequences, 
# so it can be a bit more. And also keep in mind that not all images might actually have the 
# class inside.
n_imgs_per_class = 150 

# perhaps there are some folders from which you dont want to take images because there are 
# not representative of the project
excluded_folders = ["Filmhouse"]


# n_imgs_per_class = {"mongoose" : 100,
#                     "leopard" : 70,
#                     "kudu" : 120,
#                     "hyrax" : 100,
#                     "honey badger" : 100,
#                     }


# packages
import os
from tqdm import tqdm
import random
import json
import uuid
from pathlib import Path
import shutil
from copy import copy
import pathlib
import pandas as pd
import time
from collections import defaultdict 
src_dir = os.path.dirname(json_fpath)

def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# get an empty detections dictionary to be filled
def initialize_class_dict(init_value):
    label_map_init = fetch_label_map_from_json(json_fpath)
    detections_init = defaultdict()
    for key, value in label_map_init.items():
        detections_init[value] = init_value
    for cat in ['animal', 'person', 'vehicle']:
        if cat in detections_init:
            del detections_init[cat]
    cat_list = detections_init.keys()
    return [detections_init, cat_list]

# get the paths and counts of images in a sequence folder
def return_imgs(seq):
    seq_fpath = os.path.join(src_dir, seq)
    img_paths = [os.path.join(seq, f) for f in os.listdir(seq_fpath) if f != ".DS_Store" and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    return img_paths

# gather all sequences with at least one detection in dictionary
label_map = fetch_label_map_from_json(json_fpath)
with open(json_fpath) as json_content:
    data = json.load(json_content)
seq_dict, cat_list = initialize_class_dict([])
for image in data['images']:
    file = image['file']
    if 'detections' in image:
        for detection in image['detections']:
            conf = detection["conf"]
            cat = label_map[detection['category']]
            if conf >= conf_thresh:
                seq = os.path.dirname(file)
                if cat in seq_dict:
                    seq_list = copy(seq_dict[cat])
                    seq_list.append(seq)
                    seq_dict[cat] = copy(seq_list)

# remove duplicates, forbidden folders, and random shuffle
for cat, v in seq_dict.items():
    seqs_incl = list(dict.fromkeys(v))
    seqs_excl = []
    for seq in seqs_incl:
        for excluded_folder in excluded_folders:
            if not excluded_folder in seq:
                seqs_excl.append(seq)
            # else:
            #     print(f"Excluded seq : {seq}")
    random.shuffle(seqs_excl)
    seq_dict[cat] = seqs_excl

# make sure that there are no duplicate seqs over different categories and move files
seen = set()
dupes = []
n_tot = 0
for cat, seqs in seq_dict.items():
    imgs_list = [] 
    for seq in seqs:
        if not seq in seen:
            seen.add(seq)
            imgs_list.extend(return_imgs(seq))
    imgs_list = imgs_list[:n_imgs_per_class]
    for img in imgs_list:
        path_components = Path(img).parts
        img_fname = path_components[-1]
        seq_dirname = path_components[-2]
        src_fpath = os.path.join(src_dir, img)
        dst_fpath = os.path.join(dst_dir, cat, seq_dirname, img_fname)

        # copy files
        Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_fpath, dst_fpath)

    n_tot += len(imgs_list)
    print(f"Moved {str(len(imgs_list)).ljust(4)} images for class {cat}")
print(f"Moved {str(n_tot).ljust(4)} images in total")

























exit()

    # seq_dict[k] = seqs_excl
                
# copy files
for cat, seqs in seq_dict.items():
    imgs_list = [] 
    for seq in seqs:
        imgs_list.extend(return_imgs(seq))
    imgs_list = imgs_list[:n_imgs_per_class]
    for img in imgs_list:
        path_components = Path(img).parts
        img_fname = path_components[-1]
        seq_dirname = path_components[-2]
        src_fpath = os.path.join(src_dir, img)
        dst_fpath = os.path.join(dst_dir, cat, seq_dirname, img_fname)

        # print("")
        # print(f"img             : {img}")
        # print(f"path_components : {path_components}")
        # print(f"img_fname       : {img_fname}")
        # print(f"seq_dirname     : {seq_dirname}")
        # print(f"src_fpath       : {src_fpath}")
        # print(f"dst_fpath       : {dst_fpath}")

        Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_fpath, dst_fpath)
    
    print(f"Found {str(len(imgs_list)).ljust(3)} images for class {cat}")









exit()


# # select images
# selection_dict, _ = initialize_class_dict([])
# n_total_filled = 0
# n_total_seqs = 0
# for cat in cat_list:
#     print("")
#     print(f"{cat}")
#     print(len(seq_dict[cat]))
#     enough = False
#     n_filled = 0
#     n_seqs = 0
#     if len(seq_dict[cat]):
#         for i, seq in enumerate(seq_dict[cat]):
#             imgs, n_imgs = return_imgs(seq)
#             for img in imgs:
#                 if not enough:
#                     path_components = Path(img).parts
#                     img_fname = path_components[-1]
#                     seq_dirname = path_components[-2]
#                     src_fpath = os.path.join(src_dir, img)
#                     dst_fpath = os.path.join(dst_dir, cat, seq_dirname, img_fname)

#                     # print("")
#                     # print(f"img             : {img}")
#                     # print(f"path_components : {path_components}")
#                     # print(f"img_fname       : {img_fname}")
#                     # print(f"seq_dirname     : {seq_dirname}")
#                     # print(f"src_fpath       : {src_fpath}")
#                     # print(f"dst_fpath       : {dst_fpath}")

#                     # Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
#                     # shutil.copy2(src_fpath, dst_fpath)

#                     n_filled += 1
#                     n_seqs = i + 1
            
#             if n_filled >= n_imgs_per_class and not enough:
#                 enough = True

#     print(f"n seqs copied : {n_seqs}")
#     print(f"n imgs copied : {n_filled}")
#     n_total_filled += n_filled
#     n_total_seqs += n_seqs
#     if not enough:
#         print(f" --> {str(n_imgs_per_class - n_filled).ljust(3)} images short for target of {n_imgs_per_class}!")

# print("")
# print(f"all classes together")
# print(f"n seqs copied : {n_total_seqs}")
# print(f"n imgs copied : {n_total_filled}")


# # "C:\Peter\desert-lion-project\philips-footage\unsorted\IntoNature4\CT1+2\Camera Trap2\Video_Clips\Filmhouse\Party\JPG\000\PvL_seq_ebfeb\0001.JPG"