# test script to find out if we can communicate between json and yolo annotations

"""

conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv;python /Users/peter/Desktop/yolo2json.py


"""

# TODO:
# - moet werken met absulte of relative paden
# - categories van MD moeten nog geconvert worden




# import packages like a christmas tree
import os
import re
import sys
import cv2
import git
import json
import math
import time
import torch
import random
import signal
import shutil
import platform
import datetime
import PIL.Image
import traceback
import subprocess
import webbrowser
import numpy as np
import PIL.ExifTags
import pandas as pd
import tkinter as tk
from tkinter import *
from pathlib import Path
from random import randint
from functools import partial
from subprocess import Popen, PIPE
import xml.etree.cElementTree as ET
from PIL import ImageTk, Image, ImageFilter
from bounding_box import bounding_box as bb
from tkinter import filedialog, ttk, messagebox as mb

# init vars
# indir = r"/Users/peter/Desktop/_source"
recognition_file = r"/Users/peter/Desktop/_destination/image_recognition_file.json"

# # function to 
# def update_json_from_anotation_file(annotation_file, recognition_file):
#     # log
#     print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

#     # open annotation file
#     with open(annotation_file, 'r') as annot_file:
#         # init vars
#         detections = []

#         # loop over all detections
#         for line in annot_file.readlines():
#             # read data from annotation file
#             label, xo, yo, w_box, h_box = line.split(' ')

#             # convert to COCO json format
#             x_left = float(xo) - (float(w_box)/2)
#             y_left = float(yo) - (float(h_box)/2)
#             w_box = float(w_box)
#             h_box = float(h_box)
#             bbox = [x_left, y_left, w_box, h_box]

#             # append mannually checked detections
#             detection = {'category': label, 'conf': 1.0, 'bbox': bbox}
#             detections.append(detection)

#     # adjust original json content
#     for image in data['images']:

#         # split extentions to check similarity
#         image_no_ext = os.path.splitext(image['file'])[0]
#         annot_no_ext = os.path.splitext(annot)[0]

#         # find the right file to adjust
#         if image_no_ext == annot_no_ext:
#             image['detections'] = detections
#             image['manually_approved'] = True

#     # write updated json file
#     with open(recognition_file, "w") as image_recognition_file_content:
#         data = json.dump(data, image_recognition_file_content, indent = 1)


# the process of creating a subfolder with annotation file, classes.txt and image copy, open labelImg, then update json with checked detections, then remove the files again and start over with nre image. 
def manualy_approve_image(image_path, original_detections, label_map, inverted_label_map):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # init vars
    img_filename_with_extention = os.path.basename(image_path)
    img_filename_without_extention = os.path.splitext(img_filename_with_extention)[0]
    subfolder = os.path.join(os.path.dirname(image_path), 'human-in-the-loop-subfolder')

    # create subfolder
    Path(subfolder).mkdir(parents=True, exist_ok=True)

    # create classes.txt
    classes_txt = os.path.join(subfolder, "classes.txt")
    with open(classes_txt, 'w') as f:
        for key in label_map:
            f.write(f"{label_map[key]}\n")

    # create annotation file
    annot_file = os.path.join(subfolder, img_filename_without_extention + ".txt")
    with open(annot_file, 'w') as f:
        for detection in original_detections:
            # convert COCO to YOLO
            category = detection["category"]
            print(f"      category : {category}")
            print(f"type(category) : {type(category)}")
            label = label_map[category]
            w_box = detection['bbox'][2]
            h_box = detection['bbox'][3]
            xo = detection['bbox'][0] + (w_box/2)
            yo = detection['bbox'][1] + (h_box/2)

            # correct for the non-0-index-starting default label map of MD
            if inverted_label_map == {'animal': '1', 'person': '2', 'vehicle': '3'}:
                class_id = int(inverted_label_map[label])-1
            else:
                class_id = int(inverted_label_map[label])
            f.write(f"{class_id} {xo} {yo} {w_box} {h_box}\n")
    
    # copy image into subfolder
    shutil.copy2(image_path, os.path.join(subfolder, img_filename_with_extention))

    # open labelImg and manually adjust image detections
    p = Popen(f"'/Applications/.EcoAssist_files/EcoAssist/label.command' '{subfolder}' '{classes_txt}'",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True)
    for line in p.stdout:
        print(line, end='')
    
    # convert YOLO back to COCO
    with open(annot_file, 'r') as annot_file:
        # init vars
        checked_detections = []

        # loop over all detections
        for line in annot_file.readlines():
            # read data from annotation file
            label, xo, yo, w_box, h_box = line.split(' ')

            # correct for the non-0-index-starting default label map of MD
            if inverted_label_map == {'animal': '1', 'person': '2', 'vehicle': '3'}:
                class_id = str(int(label)+1)
            else:
                class_id = str(label)

            # convert to COCO json format
            x_left = float(xo) - (float(w_box)/2)
            y_left = float(yo) - (float(h_box)/2)
            w_box = float(w_box)
            h_box = float(h_box)
            bbox = [x_left, y_left, w_box, h_box]

            # append manually checked detections
            checked_detection = {'category': class_id, 'conf': 1.0, 'bbox': bbox}
            checked_detections.append(checked_detection)
    
    # remove subfolder with contents
    shutil.rmtree(subfolder)
    
    # return approved detections in COCO format
    return checked_detections





# helper function
# fetch label map from json
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map










# Actually perform the human-in-the-loop action based on the json file
with open(recognition_file, "r") as image_recognition_file_content:
    data = json.load(image_recognition_file_content)
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    for image in data['images']:
        # if human-in-the-loop condition apply
        checked_detections = manualy_approve_image(image_path = image['file'], original_detections = image['detections'], label_map = label_map, inverted_label_map = inverted_label_map)
        image['detections'] = checked_detections
        image['manually_approved'] = True
    
# write updated json file
with open(recognition_file, "w") as image_recognition_file_content:
    data = json.dump(data, image_recognition_file_content, indent = 1)














# manualy_approve_image('/Users/peter/Desktop/_destination/7773.jpg')





# # Read json file
# with open(recognition_file, "r") as image_recognition_file_content:
#     data = json.load(image_recognition_file_content)
#     print(f"json file content BEFORE is: {json.dumps(data, indent=2)}\n\n")










# # loop through annotation files
# for filename in os.listdir(indir):
#     annot = os.path.join(indir, filename)
#     if os.path.isfile(annot) and annot.endswith('.txt') and filename != 'classes.txt':
#         # print(f"Found annotation file   : {annot}")
#         with open(annot, 'r') as annot_file:
#             detections = []
#             lines = annot_file.readlines()
#             for i in range(len(lines)):
#                 line = lines[i]
#                 # print(f"Line {i}")
#                 # read data in annotation file
#                 label, xo, yo, w_box, h_box = line.split(' ')

#                 # convert to json format
#                 left_x = float(xo) - (float(w_box)/2)
#                 left_y = float(yo) - (float(h_box)/2)
#                 w_box = float(w_box)
#                 h_box = float(h_box)
#                 # bbox = [10, 10, 10, 10] # DEBUG
#                 bbox = [left_x, left_y, w_box, h_box]

#                 # append mannually checked detections
#                 detection = {'category': label, 'conf': 1.0, 'bbox': bbox}
#                 detections.append(detection)

#             # create image data
#             # image_data = {'file': annot, 'detections': detections}

#         # print(f"image_data : {image_data}")

#         # adjust original json content
#         for image in data['images']:
#             image_no_ext = os.path.splitext(image['file'])[0]
#             annot_no_ext = os.path.splitext(annot)[0]
#             print(f"looping through the images : {image}")
#             print(f"image : {image_no_ext}")
#             print(f"annot : {annot_no_ext}\n\n")

#             if image_no_ext == annot_no_ext:
#                 print(f"Got the right file te pakken : {annot}")
#                 image['detections'] = detections
#                 image['manually_approved'] = True
            
#         print(f"json file content AFTER is: {json.dumps(data, indent=2)}\n\n")
#         # with open(recognition_file, "w") as image_recognition_file_content:
#         #     data = json.dump(data, image_recognition_file_content, indent = 1)
            

#         # # write
#         # with open(recognition_file, "w") as json_file:
#         #     json.dump(data, recognition_file, indent=1)


#                 # # find the associated json dump
#                 # detections_list = data['images']['detections']





# # data['images']
# # print(f"data['images'] : {data['images']}")
# # print(f"data['images']['detections'] : {data['images']['detections']}")
# # print(f" : {}")
# # print(f" : {}")
# # print(f" : {}")
# # print(f" : {}")






# # with open(json) as json:
# #     data = json.load(json)
# #     for image in data['images']:
# #         file = image['file']
# #         detections_list = image['detections']
# #         n_detections = len(detections_list)
# #         progress_postprocess_progbar['value'] += 100 / n_images
# #         print(f"Found annotation ")