# script to convert cameratrap dataset https://lila.science/datasets/ena24detection to YOLOv5 format and select classes
# Peter van Lunteren, 25 April

# command to run script
# conda activate coco2yolo;python /Users/peter/Desktop/new_annot_set/convert_COCO_to_YOLOv5.py

# import packages
import os
import json
import shutil
from pathlib import Path

# set vars
annotations_json = "/Users/peter/Desktop/new_annot_set/images/annotations.json"
path_to_images = "/Users/peter/Desktop/new_annot_set/images"

choices = {}

# choices = {'American Black Bear' : 0,
#            'American Crow' : 1}

# choices = {"Bird" : 0,
#             "Eastern Gray Squirrel" : 1,
#             "Eastern Chipmunk" : 2,
#             "Woodchuck" : 3,
#             "Wild Turkey" : 4,
#             "White_Tailed_Deer" : 5,
#             "Virginia Opossum" : 6,
#             "Eastern Cottontail" : 7,
#             "Human" : 8,
#             "Vehicle" : 9,
#             "Striped Skunk" : 10,
#             "Red Fox" : 11,
#             "Eastern Fox Squirrel" : 12,
#             "Northern Raccoon" : 13,
#             "Grey Fox" : 14,
#             "Horse" : 15,
#             "Dog" : 16,
#             "American Crow" : 17,
#             "Chicken" : 18,
#             "Domestic Cat" : 19,
#             "Coyote" : 20,
#             "Bobcat" : 21,
#             "American Black Bear" : 22}

# read annotations.json
with open(annotations_json) as annotations_json_content:
    data = json.load(annotations_json_content)

# get image data in dict
image_data = {}
for image in data['images']:
    image_data[image['id']] = [image['width'], image['height']]

# get category data in dict
category_data = {}
for category in data['categories']:
    category_data[category['id']] = category['name']

# write classes.txt
classes_txt = os.path.join(path_to_images, "processed", "classes.txt")
Path(os.path.dirname(classes_txt)).mkdir(parents=True, exist_ok=True)
with open(classes_txt, 'w') as f:
    for key in choices:
        f.write(f"{key}\n")

# remove old annotations
for annot in data['annotations']:
    image_id = annot['image_id']
    image = os.path.join(path_to_images, f"{image_id}.jpg")
    if os.path.isfile(image):
        annotation_txt = os.path.join(path_to_images, f"{image_id}.txt")
        if os.path.isfile(annotation_txt):
            os.remove(annotation_txt)

# loop through annotations
n = 1
n_tot = len(data['annotations'])
images_with_detections = []
for annot in data['annotations']:
    print(f"Iteration {n} of {n_tot}")
    n += 1
    image_id = annot['image_id']
    image = os.path.join(path_to_images, f"{image_id}.jpg")
    images_with_detections.append(image)
    if os.path.isfile(image):

        # collect information
        annotation_txt = os.path.join(path_to_images, f"{image_id}.txt")
        w_image = image_data[image_id][0]
        h_image = image_data[image_id][1]
        w_box = round(annot['bbox'][2]/w_image, 6)
        h_box = round(annot['bbox'][3]/h_image, 6)
        xo = round(annot['bbox'][0]/w_image + (w_box/2), 6)
        yo = round(annot['bbox'][1]/h_image + (h_box/2), 6)
        category_id = annot['category_id']
        category_name = category_data[category_id]

        # if class is chosen
        if category_name in choices:
            # copy image into subfolder
            src_jpg = os.path.join(path_to_images, f"{image_id}.jpg")
            dst_jpg = os.path.join(path_to_images, "processed", category_name, f"{image_id}.jpg")
            Path(os.path.dirname(dst_jpg)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_jpg, dst_jpg)

            # create annotation in subfolder
            dst_txt = os.path.join(path_to_images, "processed", category_name, f"{image_id}.txt")
            with open(dst_txt, 'a+') as f:
                f.write(f"{choices[category_name]} {xo} {yo} {w_box} {h_box}\n")