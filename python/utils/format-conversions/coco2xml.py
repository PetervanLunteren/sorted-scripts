# Script to create xml files (for labelling which can be used for training) based on the COCO Camera Traps format json file.

# conda env = cameratraps-detectorGPU5

import json
import os
import pathlib
import xml.etree.cElementTree as ET
from pathlib import Path, PurePath
import shutil
import csv
from itertools import chain

import numpy as np
import pandas as pd
from PIL import Image



def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def create_labimg_xml(image_path, annotation_list):

    # example:
    # anotation_list = ['left1,bottom1,X,X,right1,top1,X,X,label1',
    #                   'left2,bottom2,X,X,right2,top2,X,X,label2',
    #                   'left3,bottom3,X,X,right3,top3,X,X,label3']

    image_path = Path(image_path)
    img = np.array(Image.open(image_path).convert('RGB'))

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img.shape[1])
    ET.SubElement(size, 'height').text = str(img.shape[0])
    ET.SubElement(size, 'depth').text = str(img.shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for annot in annotation_list:
        tmp_annot = annot.split(',')
        cords, label = tmp_annot[0:-2], tmp_annot[-1]
        xmin, ymin, xmax, ymax = cords[0], cords[1], cords[4], cords[5]

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = label
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    indent(annotation)
    tree = ET.ElementTree(annotation)
    xml_file_name = image_path.parent / (image_path.name.split('.')[0] + '.xml')
    tree.write(xml_file_name)


###### extract data from json in COCO Camera Traps format
path_to_json = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS Camera Traps\analyse_files\wcs_camera_traps.json"
with open(path_to_json) as json_file:
    data = json.load(json_file)

# populate dict with image_id as key and the rest of the image info as value
image_dict = {}
for image in data['images']:
    image_dict[image['id']] = image

# populate dict with category_id as key and animal species as value
category_dict = {}
for category in data['categories']:
    category_dict[category['id']] = category['name']

# populate dict with image_id as key and the rest of the detection info as value
detection_dict = {}
for detection in data['annotations']:
    detection_dict[detection['image_id']] = detection

# this prints the information which is stored in the dicts
print("image_dict (key=image_id):", list(image_dict.keys())[0], "-->", image_dict[list(image_dict.keys())[0]])
print("category_dict (key=category_id):", list(category_dict.keys())[0], "-->", category_dict[list(category_dict.keys())[0]])
print("detection_dict (key=image_id):", list(detection_dict.keys())[0], "-->", detection_dict[list(detection_dict.keys())[0]])

# output a .txt with file names for a selective download via Azure
species = ["bos_taurus", 71] # [dir_name, category_id] check .json for which category_id belongs to the species
dir_path = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS Camera Traps"
path_txtfile = os.path.join(dir_path, "analyse_files", "{}.txt".format(species[0])) # will be created
website = "https://lilablobssc.blob.core.windows.net/wcs-unzipped"
txtfile = open(path_txtfile, 'w')
for detection in detection_dict.keys(): # write .txt file
    if detection_dict[detection]['category_id'] == species[1]:
        file_name = image_dict[detection_dict[detection]['image_id']]['file_name']
        txtfile.write("{}\n".format(file_name))
txtfile.close()
new_dir = os.path.join(dir_path, species[0])
Path(new_dir).mkdir(parents=True, exist_ok=True)
print("commands for azure download:")
print(r"cd C:\git\azcopy_windows_amd64_10.13.0")
print(r'azcopy copy "{}" "{}" --list-of-files="{}"'.format(website, new_dir, path_txtfile))


# azcopy copy "https://lilablobssc.blob.core.windows.net/wcs-unzipped" "V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS Camera Traps\horse" --list-of-files="V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS Camera Traps\analyse_files\horse.txt"

#### use this data and create an xml file for the ones which are provided with an bbox
# # create dict and fill it with the information we need: bbox, file_name and label
# n_with_bbox = 0
# n_without_bbox = 0
# with_bbox = []
# without_bbox = []
# all_annots = {}
# for annot in data['annotations']:
#     if "bbox" in annot:
#         image = image_dict[annot['image_id']]
#         file_name = image['file_name']
#         label = category_dict[annot['category_id']]
#         left = str(annot['bbox'][0])
#         top = str(annot['bbox'][1])
#         right = str(annot['bbox'][0] + annot['bbox'][2])
#         bottom = str(annot['bbox'][1] + annot['bbox'][3])
#         list = [left, bottom, left, left, right, top, left, left, label]
#         string = ','.join(map(str, list))
#         if file_name in all_annots.keys():
#             new_value = [all_annots[file_name], [string]]
#             new_value = sum(new_value, []) # unlist
#             all_annots[file_name] = new_value
#         else:
#             all_annots[file_name] = [string]
#         with_bbox.append(image)
#         n_with_bbox += 1
#     else:
#         image = image_dict[annot['image_id']]
#         without_bbox.append(image)
#         n_without_bbox += 1
#
# missing_path = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\missouri_camera_traps_images"
#
# for image_path in all_annots.keys():
#     if os.path.isfile(os.path.join(missing_path, image_path)):
#         print("\nfile = ", os.path.join(missing_path, image_path))
#         print("annotation_list = ", all_annots[image_path])
#         # create_labimg_xml(os.path.join(first_path_to_image_dir, image_path), all_annots[image_path])
