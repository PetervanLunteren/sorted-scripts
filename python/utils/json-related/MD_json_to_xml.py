# Script to create xml files (for labelling which can be used for training) based on the MegaDetector json-output.
# Specify the .json file and make sure you did not move the images after the MegaDetector has run.

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

    # expected input (X = doesn't matter...):
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


###### input
path_to_json = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps\output.json"

##### create XML files based on MegaDetector output json
## First we will need to extract the labels from somwhere (csv, json or dir names). This is different every time.
annotation_list = []
with open(path_to_json) as json_file:
    data = json.load(json_file)
n_images = len(data['images'])
for image in data['images']:
    file = str(image['file'])
    file_path_components = os.path.normpath(file).split(os.path.sep)
    spp_label = file_path_components[-5]
    file_name_without_ext = str(os.path.splitext(pathlib.PurePath(file).name)[0])
    n_detections = len(image['detections'])
    detections_list = image['detections']
    annotation_list = []
    if not n_detections == 0:
        for detection in image['detections']:
            category = detection['category']
            if category == '1':
                label = spp_label # Here we need to give it the label instead of 'animal'
            elif category == '2':
                label = 'person'
            else:
                label = 'vehicle'
            im = Image.open(file)
            width, height = im.size
            left = int(round(detection['bbox'][0] * width))
            top = int(round(detection['bbox'][1] * height))
            right = int(round(detection['bbox'][2] * width)) + left
            bottom = int(round(detection['bbox'][3] * height)) + top
            list = [left, bottom, left, left, right, top, left, label]
            # No clue why create_labimg_xml() expects so many values in annotation_list...
            # As far as I can see the function only uses [0, 1, 4, 5, -1].
            # Just filled the rest up with xmin and it seems to work...
            string = ','.join(map(str, list))
            annotation_list.append(string)
        create_labimg_xml(file, annotation_list)
        print("{} has been processed. .xml file was written.".format(file))
    else:
        print("{} has been processed, but there were not detections.".format(file))





# ##### move images n dirs up
# for root, dirs, files in os.walk(r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps"):
#     if root != r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps" and root != r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps\analyse_files" and files != [] and dirs == []:
#         print("\nroot", root)
#         print("dirs", dirs)
#         print("files", files)
#
#         for file in files:
#             src = os.path.join(root, file)
#             root_path_components = os.path.normpath(root).split(os.path.sep)
#             new_file_name = '-'.join([root_path_components[-1], file]) # combi of folder and file, otherwise overwriting of equally named files
#             dst = os.path.join("V:" + os.sep, *root_path_components[1:-3], new_file_name)
#             print("\nsrc :", src)
#             print("dst :", dst)
#             shutil.move(src, dst)


# src = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps\bos_taurus\wcs-unzipped\animals\0001\1941.jpg"
# dst = r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps\bos_taurus\1941.jpg"
# shutil.move(src, dst)

# r"V:\Projecten\A70_30_65\Losse_camera\Data\Trainingsdata_nog_niet_toegevoegd\nog_niet_gecheckt\WCS_cameratraps\bos_taurus\wcs-unzipped\animals\0001\1941.jpg"

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






# ##### Check how many files are in labels.csv and in the dir,check duplicates and compare to the megadetector json file
# csv_list = labels['photo_id'].tolist()
# csv_list = [str(i) for i in csv_list]
# dir_list = [i[:-4] for i in os.listdir(image_dir)]
# n_files_in_dir = 0
# in_dir_not_csv = []
# for csv_file in dir_list:
#     n_files_in_dir += 1
#     if csv_file not in csv_list:
#         in_dir_not_csv.append(csv_file)
# print("n_files_in_dir :", n_files_in_dir)
# print("in_dir_not_csv :", in_dir_not_csv)
# n_files_in_csv = 0
# in_csv_not_dir = []
# for dir_file in csv_list:
#     n_files_in_csv += 1
#     if dir_file not in dir_list:
#         in_csv_not_dir.append(dir_file)
# print("n_files_in_csv :", n_files_in_csv)
# print("in_csv_not_dir :", in_csv_not_dir)
# print("dupes in csv_list :", pd.Series(csv_list)[pd.Series(csv_list).duplicated()].values)
# print("dupes in dir_list :", pd.Series(dir_list)[pd.Series(dir_list).duplicated()].values)
# with open(path_to_json) as json_file:
#     data = json.load(json_file)
# n_images = len(data['images'])
# print("Number of images analysed by megadetector :", n_images)






####### copy the images which are in labels.csv (if you want to work with a subset)
# list_of_photos = labels.photo_id.tolist()
# dir = r"/Users/Shared/Niet in de backup/Mammal_web_fotos/gold_standard_photos"
# for filename in os.listdir(dir):
#     if int(filename[:-4]) in list_of_photos:
#         src = os.path.join(dir, filename)
#         dst = os.path.join('/Users/Shared/Niet in de backup/part of pictures/images', filename)
#         copyfile(src, dst)
#         print("moved {}".format(filename))
