# Script to create xml files (for labelling which can be used for training) based on the MegaDetector json-output.
# Specify the .json file and make sure you did not move the images after the MegaDetector has run.

# conda env = cameratraps-detectorGPU5

import json
import os
import pathlib
import xml.etree.cElementTree as ET
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image


path_to_json = r"C:\Users\GisBeheer\Desktop\out2.json"

# init empty df
df = pd.DataFrame(columns=["file", "filepath", "n_detections",
                           "n_animals", "max_conf_animal", "animal_present",
                           "n_persons", "max_conf_person", "person_present",
                           "n_vehicles", "max_conf_vehicle", "vehicle_present"])

# convert megadetector json to df
with open(path_to_json) as json_file:
    data = json.load(json_file)
for image in data['images']:
    file_path = str(image['file'])
    file_comp = os.path.normpath(file_path).split(os.sep)
    file_short = file_comp[-1]
    print(file_short)
    n_detections = len(image['detections'])
    detections_list = image['detections']
    annotation_list = []
    n_animals = 0
    max_conf_animal = 0
    n_persons = 0
    max_conf_person = 0
    n_vehicles = 0
    max_conf_vehicle = 0
    if not n_detections == 0:
        for detection in image['detections']:
            category = detection['category']
            conf = detection['conf']
            if category == '1':
                n_animals += 1
                if conf > max_conf_animal:
                    max_conf_animal = conf
            elif category == '2':
                n_persons += 1
                if conf > max_conf_person:
                    max_conf_person = conf
            else:
                n_vehicles += 1
                if conf > max_conf_vehicle:
                    max_conf_vehicle = conf
    values_to_add = {"file": file_short,
                     "filepath": file_path,
                     "n_detections": n_detections,
                     "n_animals": n_animals,
                     "max_conf_animal": max_conf_animal,
                     "animal_present": True if n_animals else False,
                     "n_persons": n_persons,
                     "max_conf_person": max_conf_person,
                     "person_present": True if n_persons else False,
                     "n_vehicles": n_vehicles,
                     "max_conf_vehicle": max_conf_vehicle,
                     "vehicle_present": True if n_vehicles else False}
    row_to_add = pd.Series(values_to_add)
    df = df.append(row_to_add, ignore_index=True)

pd.set_option("display.max_rows", None, "display.max_columns", None)
df = pd.DataFrame(df)
print(df.head())
df.to_csv(r"C:\Users\GisBeheer\Desktop\checkit.csv", index=False)




# ##### move images and xmls to subdir based on their label in the csv file (True.species)
# n_jpgs = 0
# n_xmls = 0
# csv_list = labels['photo_id'].tolist()
# csv_list = [str(i) for i in csv_list]
# dupes_in_csv = pd.Series(csv_list)[pd.Series(csv_list).duplicated()].values # these images contain multiple species
# for index, row in labels.iterrows():
#     image = os.path.join(image_dir, str(row['photo_id']) + ".jpg")
#     xml_file = os.path.join(image_dir, str(row['photo_id']) + ".xml")
#     subdir = os.path.join(image_dir, str(row['True.species']))
#     Path(subdir).mkdir(parents=True, exist_ok=True)
#     if str(row['photo_id']) in dupes_in_csv: # if in duplicate of csv, then there are multiple species present in the image
#         subdir = os.path.join(image_dir, "multiple_spp")
#         Path(subdir).mkdir(parents=True, exist_ok=True)
#     if os.path.isfile(image):
#         n_jpgs += 1
#         src = image
#         dst = os.path.join(subdir, pathlib.PurePath(image).name)
#         shutil.move(src, dst)
#         print(str(row['photo_id']) + ".jpg moved to {}".format(subdir))
#     if os.path.isfile(xml_file):
#         # os.remove(xml_file)
#         n_xmls += 1
#         src = xml_file
#         dst = os.path.join(subdir, pathlib.PurePath(xml_file).name)
#         shutil.move(src, dst)
#         print(str(row['photo_id']) + ".xml moved to {}".format(subdir))
# print("Number of jpgs moved :", n_jpgs)
# print("Number of xmls moved :", n_xmls)

