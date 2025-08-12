# Script to separate annotated bundles into separate dirs

# This script assumes that you have prepared the bundles with 8-prepare-bundles-for-submodels.py
# Then these bundles are annotated by the client and with EcoAssist, so there are accompanying xml files with the annotations.
# This script will separate the annotated images into separate directories based on the annotations.
# It will read the label in the xml file, and copy the image to the corresponding directory.
# The images are already cropped when bundled, so we don't need to crop them here.
# It will not do anything with the bbox, as that doesn't represent the animal in this case. It's just a placeholder for the label.

# TODO: er kunnen een aantal classes tussen zitten die we eigenlijk niet in de training dataset willen hebben. Zorg ervoor dat die er niet inkomen: "person", "vehicle", "unidentified animal"

# """ 
# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\8-separate-annotated-bundles.py"
# """

# packages
import os
import xml.etree.ElementTree as et
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import subprocess

# user input
src_dir = r"C:\Users\smart\Downloads\Trained Bundles"
dst_dir = r"C:\Users\smart\Downloads\dst"

# count how many files
n_files = 0
for root, dirs, files in os.walk(src_dir):
    for fn in files:
        n_files += 1

# main loop
pbar = tqdm(total=n_files)
idx = 0
for root, dirs, files in os.walk(src_dir):
    for fn in files:
        idx += 1
        pbar.update(1)
        
        # if idx < 22050: # DEBUG
        #     continue
        
        fp_src = os.path.join(root, fn)
        if fp_src.endswith(".xml"):

            # check image extention
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                img = os.path.splitext(fp_src)[0] + ext
                if os.path.isfile(os.path.splitext(fp_src)[0] + ext):
                    break # found image
            else:
                continue

            # init vars
            xml_fpath = fp_src
            img_fpath = os.path.splitext(xml_fpath)[0] + ext
            fp_rel = os.path.dirname(os.path.relpath(fp_src, src_dir))

            # read xml
            try:
                tree = et.parse(xml_fpath)
            except et.ParseError as e:
                print(f"Error parsing XML file {xml_fpath}: {e}")
                # parent_folder = os.path.dirname(xml_fpath)
                # subprocess.Popen(f'explorer "{parent_folder}"')

                
            img_height = int(tree.find('.//size//height').text)
            img_width  = int(tree.find('.//size//width').text)

            # convert objects to crops
            # it = 1
            for obj in tree.iter("object"):
                name = obj.find('.//name').text
                # dst_fpath = os.path.join(dst_dir, name, fp_rel, f"{os.path.splitext(fn)[0]}_{it}{ext}")
                dst_fpath = os.path.join(dst_dir, name, fp_rel, f"{os.path.splitext(fn)[0]}{ext}")
                Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
                Image.open(img_fpath).save(dst_fpath)
                # it += 1