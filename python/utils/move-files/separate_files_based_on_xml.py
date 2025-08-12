# Script to divide the images and xml files to their own subdir (based on the name tag)
# it will devide it based on all categories, e.g., "cat", "cow", "cow_cat", and "background"


# WINDOWS: first time with new ecoassistcondaenv
# """
# conda activate ecoassistcondaenv
# """

# WINDOWS: execute script in miniforge prompt
# """
# conda activate ecoassistcondaenv && python 'C:\Users\smart\Desktop\separate_files_based_on_xml.py'
# """

# MACOS: first time with new ecoassistcondaenv
# """
# conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv
# """

# MACOS: execute script in terminal
# """
# pmset noidle &;conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv;python '/Users/peter/Documents/scripting/sorted-scripts/python/utils/file-path-or-name-related/separate_files_based_on_xml.py'
# """

import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as et
from tqdm import tqdm

chosen_dir = r"C:\Peter\desert-lion-project\training-dataset-general-lion-model"

for file in tqdm(os.listdir(chosen_dir)):
    if file.endswith(".xml"):
        xml_file = os.path.join(chosen_dir, file)
        tree = et.parse(xml_file)

        # get categories
        detection_categories = []
        for obj in tree.findall("./object"):
            cat = obj.find('.//name').text.strip().replace(' ', '_')
            detection_categories.append(cat)
        detection_categories = sorted(list(set(detection_categories)))
        cat_string = '_'.join([str(elem) for elem in detection_categories])

        # check if the xml has any objects
        if cat_string == '':
            dir_full = os.path.join(chosen_dir, 'background')
        else:
            dir_full = os.path.join(chosen_dir, cat_string)
        
        # move
        Path(dir_full).mkdir(parents=True, exist_ok=True)
        shutil.move(xml_file, os.path.join(chosen_dir, dir_full))
        for ext in ['.jpg', '.jpeg', '.png']:
            img = os.path.splitext(os.path.join(chosen_dir, file))[0] + ext
            if os.path.isfile(img):
                shutil.move(img, os.path.join(chosen_dir, dir_full))
                break