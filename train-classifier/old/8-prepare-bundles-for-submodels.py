# Script to prepare bundles of images that can be sent to the client for submodel-annotation 

# The idea is that this script takes the crops per sequences randomly and places them in fixed size
# folders. I can then send them to the client, which will annotate them using EcoAssist. It will read
# the XLSX file that was used to make the species identification model, and will prepare the folders
# with local and non-local images ittermitently. This is done so that we create a bit more heterogenity.
# Furthermore, it will prepare all the JSON files so the user can start with the EcoAssist annotation 
# process so the right away.

# For example, for 2024-14-CAN I used this script to prepare a few 1000-img-folders. The client then used 
# those folders to annotate them to sex and age.

# You'll need the XSLX that was used to make the spp-id-model, so it knows where to find the local and
# non-local images. Then set where you want the bundles to be placed, how many images per folder, and
# how many folders per class. See USER INPUT below.

# PREPARATION:
# Sometimes I take a visually similar species to substitute for a class in the species identification 
# model. Those images should not be in the submodel bundles. To avoid this, you must add a column to the
# XLSX file with the spp-id-model that says "Add_to_submodel" and set it to "TRUE" for the folders that 
# should be added to the bundles. Set it to FALSE for the ones you want to avoid. The script expects this
# column and all rows to have a value. It will crash if not.

# """
# conda activate ecoassistcondaenv-pytorch && pip install openpyxl
# conda activate ecoassistcondaenv-pytorch && python "C:\Users\smart\Desktop\8-prepare-bundles-for-submodels.py"
# """

# Peter van Lunteren, 29 nov 2023

# USER INPUT
######### have you created an extra column? See PREPARATION above #########

spp_map_xlsx = r"C:\Users\smart\Desktop\2024-14-CAN-spp-plan.xlsx"                          # same XLSX as used for the species identification model, plus a column called 'Add_to_submodel' (see PREPARATION)

dst_folder = r"C:\Users\smart\Desktop\temp"                                                 # folder where the budles will be placed in

max_n_imgs_per_subdir = 1000                                                                # size of the bundles

max_n_dirs_per_class = 30                                                                   # maximum number of subdirectories per class. It doesn't really make sense to prepare all images if we have multiple 100000s of images per class.

species_to_be_bundled = ["caribou",
                         "elk",
                         "moose",
                         "mule deer",
                         "white-tailed deer"]                                               # these are the classes for which bundles will be prepared (must be identical to the spp_map_xlsx)  

annotation_classes = {
    "1": "unk",  # unknown
    "2": "ad m", # adult male
    "3": "ad f", # adult female
    "4": "juv"   # juvenile  
    }                                                                                       # these are the classes that the used will see during annotation. make sure 1 is always unknown, as this will be the default. The rest can be in any order. This will also be the detection_categories in the JSON files

use_spp_submodels = True                                                                    # if you already have (temporary) submodels for the species and want to use them to predict the JSON classes, set to True. If False, all images will be written as 'unknown'.

spp_submodels = {
    "caribou": r"C:\Users\smart\Desktop\models\caribou_v1.pt",
    "elk": r"C:\Users\smart\Desktop\models\elk_v1.pt",
    "moose": r"C:\Users\smart\Desktop\models\moose_v1.pt",
    "mule-deer": r"C:\Users\smart\Desktop\models\mule_deer_v1.pt",
    "white-tailed-deer": r"C:\Users\smart\Desktop\models\white_tailed_deer_v1.pt"
    }                                                                                       # if using (temporary) submodels, provide the paths to the models here. The keys must be identical to the spp_map_xlsx, but with dashes instead of spaces

conf_thresh = 0.6                                                                           # if using (temporary) submodels, provide the confidence threshold here. If the confidence is lower, the class will be set to 'unknown'


# DEBUG - Grant wanted shorter labels, so here are abbreviations for the spp
spp_abbreviations = {"caribou" : "ca",
                     "elk": "el",
                     "moose": "mo",
                     "mule-deer": "md",
                     "white-tailed-deer": "wd"}

# packages
import os
from tqdm import tqdm
import random
import json
from pathlib import Path
import shutil
import pandas as pd
from PIL import Image
import xml.etree.cElementTree as ET
import numpy as np
from ultralytics import YOLO

# make sure windows trained models work on unix too
if use_spp_submodels:
    import pathlib
    import platform
    plt = platform.system()
    if plt != 'Windows': pathlib.WindowsPath = pathlib.PosixPath

# set seed so it can be reproduced to create more dirs if needed
random.seed(420)
if random.randint(1, 100) != 4:
    print(f"Random seed check failed: random int should be 4, but is {random.randint(1, 100)}.")
    print("Exiting script...")
    exit()
else:
    print(f"Random seed check passed.")

# create a nested dictionary with classes, sources and folderspaths, either import from xlsx file, or manually
rows = pd.read_excel(spp_map_xlsx)
columns_to_select = ['Class', 'Source', 'Path', 'Add_to_submodel']
try:
    rows = rows[columns_to_select]
except KeyError:
    print(f"The XLSX file must contain the following columns: {columns_to_select}")
    print(f"Did you forget to add the 'Add_to_submodel' column? See PREPARATION in script.")
    print(f"Exiting script...")
    exit()

train_input = {}
for index, row in rows.iterrows():
    add_to_submodel = row['Add_to_submodel']
    if add_to_submodel != True and add_to_submodel != False:
        print(f"Error reading value '{add_to_submodel}' for 'Add_to_submodel' for row index {index} in XLSX file. Must be 'TRUE' or 'FALSE'.")
        print(f"Did you forget to add values to all rows for the 'Add_to_submodel' column? See PREPARATION in script.")
        print(f"Exiting script...")
        exit()
    if add_to_submodel:
        spp_class = row['Class'].strip().lower()
        path = row['Path'].replace('"', '').replace("'", '')
        source = row['Source'].strip().lower()
        if source != 'local' and source != 'non-local':
            print(f"ERROR READING LABEL MAP: Source for row index {index} is invalid: '{source}'")
            exit()
        if spp_class in train_input:
            old_source_dict = train_input[spp_class]
            if source in old_source_dict:
                train_input[spp_class][source].append(path)
            else:
                train_input[spp_class][source] = [path]
        else:
            train_input[spp_class] = {}
            train_input[spp_class][source] = [path]

# create pascal voc annotation files from a list of detections
def create_pascal_voc_annotation(image_path, annotation_list, human_verified, img_dir):

    # init vars
    image_path = Path(image_path)
    img = np.array(Image.open(image_path).convert('RGB'))
    annotation = ET.Element('annotation')

    # set verified flag if been verified in a previous session
    if human_verified:
        annotation.set('verified', 'yes')

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
        xmin, ymin, xmax, ymax = cords[0], cords[1], cords[4], cords[5] # left, top, right, bottom

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
    xml_file_name = return_xml_path(image_path, img_dir)
    Path(os.path.dirname(xml_file_name)).mkdir(parents=True, exist_ok=True)
    tree.write(xml_file_name)

# helper function to correctly indent pascal voc annotation files
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

# return xml path with temp-folder squeezed in
def return_xml_path(img_path, head_path):
    tail_path = os.path.splitext(os.path.relpath(img_path, head_path))
    temp_xml_path = os.path.join(head_path, "temp-folder", tail_path[0] + ".xml")
    return os.path.normpath(temp_xml_path)

# function to check the number of images in a sequence folder
def check_n_imgs_in_seq(seq_path):
    n_imgs = len([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f != ".DS_Store" and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
    return n_imgs

# function to return the paths of images in a sequence folder
def get_img_paths_from_seq(seq_path):
    img_paths = [os.path.join(seq_path, f) for f in os.listdir(seq_path) if f != ".DS_Store" and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    return img_paths

# fetch label map from json
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# predict from cropped image
def get_classification(PIL_crop, submodel):
    results = submodel(PIL_crop, verbose=False)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    classifications = []
    for idx, v in names_dict.items():
        classifications.append([v, probs[idx]])
    highest_prediction = max(classifications, key=lambda x: x[1])
    class_name = highest_prediction[0]
    confidence = highest_prediction[1]
    return [class_name, confidence]

# interleave two lists and add remaining part at the end
def interleave_lists(list1, list2):
    interleaved_list = [item for pair in zip(list1, list2) for item in pair]
    interleaved_list.extend(list1[len(interleaved_list) // 2:] or list2[len(interleaved_list) // 2:])
    return interleaved_list

# create list of sequences and the total number of images per class 
img_counts = {}
for key, value in train_input.items():
    
    # skip classes that are not in the species_to_be_bundled list
    if key not in species_to_be_bundled:
        continue

    # get local images
    src_dirs_local = value['local'] if 'local' in value else []
    img_paths_local = []
    for src_dir in src_dirs_local:
        print(f"Scanning '{src_dir}'")
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                img_paths_local.extend(get_img_paths_from_seq(subdir))
    random.shuffle(img_paths_local)

    # get non-local images
    src_dirs_non_local = value['non-local'] if 'non-local' in value else []
    img_paths_non_local = []
    for src_dir in src_dirs_non_local:
        print(f"Scanning '{src_dir}'")
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                img_paths_non_local.extend(get_img_paths_from_seq(subdir))
    random.shuffle(img_paths_non_local)
    
    # interleave lists so that we have locals and non-locals alternately, until one list is empty and then add the remaining part of the other list
    img_list = interleave_lists(img_paths_local, img_paths_non_local)

    # check the number of images per class and decrease if needed
    max_n_img_copied = max_n_imgs_per_subdir * max_n_dirs_per_class
    n_imgs = min(len(img_list), max_n_img_copied)
    print(f"Class '{key}' has a total of {len(img_list)} images, of which {n_imgs} will be bundled")
    
    # create bundles
    imgs_copied_to_subdir = 0
    total_imgs_copied = 0
    dir_index = 1
    break_outer_loop = False
    pbar = tqdm(total=n_imgs)
    
    # loop through src paths
    for fpath_src in img_list:
            
        # update progress bar
        pbar.update(1)
        
        # remove all path before "PvL_seq_" and compile dst path
        index = fpath_src.find("PvL_seq_")
        class_name = key.replace(' ', '-') # waarom doe ik dit? Nooit de labels veranderen!
        fpath_dst = os.path.join(dst_folder, class_name, f"{dir_index}-{class_name}", fpath_src[index:])
        
        # copy
        Path(os.path.dirname(fpath_dst)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(fpath_src, fpath_dst)
        imgs_copied_to_subdir += 1
        total_imgs_copied += 1
        
        # new folder if filled sufficiently
        if imgs_copied_to_subdir >= max_n_imgs_per_subdir:
            imgs_copied_to_subdir = 0
            dir_index += 1
        
        # break if max number of images is reached
        if total_imgs_copied >= max_n_img_copied:
            break_outer_loop = True
            break
        
        # break outer loop too
        if break_outer_loop:
            break

    # close progress bar
    pbar.close()
    
    # separate print statements
    print("\n")

# get a list with all bundles
all_bundles_together = []
Path(dst_folder).mkdir(parents=True, exist_ok=True)
class_dirs = [os.path.join(dst_folder, f) for f in os.listdir(dst_folder) if os.path.isdir(os.path.join(dst_folder, f))]
for class_dir in class_dirs:
    bundles = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
    all_bundles_together.extend(bundles)
all_bundles_together = sorted(all_bundles_together)

# print the list of all bundles
print(f"Total number of bundles: {len(all_bundles_together)}")
print(json.dumps(all_bundles_together, indent=4))

# perform some actions per bundle to imitate the EcoAssist workflow
print("Preparing the JSON files for the human-in-the-loop annotation process...")
pbar = tqdm(total=len(all_bundles_together))
for image_dir in all_bundles_together:
    
    # change pbar description to spp name
    bundle_name = os.path.basename(image_dir)
    pbar.set_description(f"{bundle_name}")
    
    class_name = os.path.basename(os.path.dirname(image_dir))
    # print(f"class_name: {class_name}")
    submodel = YOLO(spp_submodels[class_name])
    
    # init vars
    output_json_file = os.path.join(image_dir, "image_recognition_file.json")

    # for the prediction take the original annotation labels
    detection_categories = annotation_classes.copy() # TODO: waarom doe ik hier zo moeilijk. De detection categories worden toch helemaal neit aangepast? Gewoon annotation_classes doen ipv copieren etc. Aah. ik pas ze wel aan, namelijk met de sppname erachter aan....
    inverted_detection_categories = {v: k for k, v in detection_categories.items()}
    
    # Prepare the structure of the JSON
    json_data = {
        "images": [],
        "detection_categories": detection_categories,
        "info": {
            "ecoassist_metadata": {
                "version": "5.14",
                "custom_model": False,
                "custom_model_info": {}
            }
        }
    }

    # Loop over all images in the directory
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                
                # get prediction
                if use_spp_submodels:
                    full_path = os.path.join(root, filename)
                    # print(f"full_path : {full_path}")
                    pred_class, pred_conf = get_classification(Image.open(full_path), submodel)
                    # print(f"pred_class: {pred_class}")
                    # print(f"pred_conf : {pred_conf}")
                    if pred_conf > conf_thresh:
                        detection_category = inverted_detection_categories[pred_class]
                    else:
                        detection_category = "1" # the unknown class
                    # print(f"detection_category : {detection_category}")
                    # print("\n")
                else:
                    detection_category = "1" # the unknown class
                    pred_conf = 1
                
                # create json data
                relative_path = os.path.join(os.path.basename(root), filename)
                image_data = {
                    "file": relative_path,
                    "detections": [
                        {
                            "category": detection_category,
                            "conf": pred_conf,
                            "bbox": [
                                0.05,  # x-center
                                0.05,  # y-center
                                0.1,   # width
                                0.1    # height
                            ]
                        }
                    ]
                }
                json_data["images"].append(image_data)

    # after inference, change the labels to make them unique
    spp = os.path.basename(os.path.dirname(image_dir))
    spp = spp_abbreviations[spp] # DEBUG if you want to abbreviate spp names, supply an abbreviation dict
    for key, value in detection_categories.items():
        detection_categories[key] = f"{value} - {spp}"
    json_data['detection_categories'] = detection_categories

    # write the JSON data to a file
    with open(output_json_file, 'w') as f:
        json.dump(json_data, f, indent=4)
        
    # update progress bar
    pbar.update(1)

# close progress bar
pbar.close()