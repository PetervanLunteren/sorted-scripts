# Script to find xml objects and square crop them to be used as training data for a classifier

# TODO: er kunnen een aantal classes tussen zitten die we eigenlijk niet in de training dataset willen hebben. Zorg ervoor dat die er niet inkomen: "person", "vehicle", "unidentified animal"

# This script is designed to convert client annotated images into crops than can be used as training data for a classifier.
# The images must be labelled with PASCAL VOC, if they are sorted in species subfolders see 0-crop-images-from-client-images-dir-labelled.py.

# Make sure you sorted all images in sequences with (sort-dir-in-sequences.py) before running this script. All images should be inside subfolders called 'PvL_seq_XXXXXXXXXXXX'

# There may be as many subfolder as you want (original folder structure preferred).

# in short, the steps are:
# 1. sort all images in sequence 'PvL_seq_xxxxxxx' subfolders (using sort-dir-in-sequences.py) (outdated, I don't use sequences anymore)
# 2. Run this script, which:
#       - loops thourgh all files and crops out all detections
#       - places them in the same folder structre in a user defined folder

# """ 
# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\0-crop-images-from-client-images-xml-labelled.py"
# """

# packages
import os
import sys
import xml.etree.ElementTree as et
from PIL import Image, ImageOps
from tqdm import tqdm
from pathlib import Path

# user input
# if you have only one dir you want to process
src_dirs = [r"C:\Users\smart\Downloads\Local Images 20250226\Moose\Adult Female"]
dst_dirs = [r"C:\Users\smart\Downloads\Local Images 20250226_fmt\Moose\Adult Female"]

# # if you have more than one dir you want to process
# src_dirs = [r"C:\Users\smart\Downloads\Local Images 20250208\Caribou\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Caribou\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Caribou\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Elk\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Elk\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Elk\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Moose\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Moose\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208\Moose\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208\mule deer\female",
#             r"C:\Users\smart\Downloads\Local Images 20250208\mule deer\juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208\mule deer\male",
#             r"C:\Users\smart\Downloads\Local Images 20250208\white-tailed deer\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208\white-tailed deer\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208\white-tailed deer\Male"]
# dst_dirs = [r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Caribou\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Caribou\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Caribou\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Elk\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Elk\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Elk\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Moose\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Moose\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\Moose\Male",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\mule deer\female",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\mule deer\juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\mule deer\male",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\white-tailed deer\Female",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\white-tailed deer\Juvenile",
#             r"C:\Users\smart\Downloads\Local Images 20250208_fmt\white-tailed deer\Male"]

# FUCTIONS
def pad_crop(box_size):
    input_size_network = 224
    default_padding = 30
    diff_size = input_size_network - box_size
    if box_size >= input_size_network:
        box_size = box_size + default_padding
    else:
        if diff_size < default_padding:
            box_size = box_size + default_padding
        else:
            box_size = input_size_network    
    return box_size

# function to crop detection with equal sides
def remove_background(img, bbox_norm):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    box_size = max(box_w, box_h)
    box_size = pad_crop(box_size)
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

# function to process each src_dir and dst_dir
def process_src_dir(src_dir, dst_dir):
    
    print(f"\n\nProcessing...")
    print(f"src_dir: {src_dir}")
    print(f"dst_dir: {dst_dir}")

    # count how many xml files
    n_files = 0
    for root, dirs, files in os.walk(src_dir):
        for fn in files:
            n_files += 1

    # main loop
    pbar = tqdm(total=n_files)

    for root, dirs, files in os.walk(src_dir):
        for fn in files:
            pbar.update(1)
            fp_src = os.path.join(root, fn)
            
            # fp_src_exists = os.path.exists(fp_src)

            # print("\n")
            # print(f"fn : {fn}")
            # print(f"fp_src : {fp_src}")
            # print(f"fp_dst : {fp_dst}")
            # print(f"fp_exists : {fp_src_exists}")

            # if not fp_src_exists:
            #     print("\n")
            #     print(f"fn : {fn}")
            #     print(f"fp : {fp}")
            #     print(f"fp_exists : {fp_src_exists}")

            if fp_src.endswith(".xml"):

                # print(f"XML : {fp_src}")

                # check image extention
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    # img = os.path.splitext(fp_src)[0] + ext
                    if os.path.isfile(os.path.splitext(fp_src)[0] + ext):
                        # print(f"IMG : {img}")
                        break
                else:
                    continue

                # exit()
                # continue

                # init vars
                xml_fpath = fp_src
                img_fpath = os.path.splitext(xml_fpath)[0] + ext
                fp_rel = os.path.dirname(os.path.relpath(fp_src, src_dir))
                # img_fpath = os.path.join(src_dir, img_fname)
                # fp_dst = os.path.normpath(os.path.join(dst_dir, os.path.relpath(fp_src, src_dir)))

                # read xml
                tree = et.parse(xml_fpath)
                img_height = int(tree.find('.//size//height').text)
                img_width  = int(tree.find('.//size//width').text)

                # convert objects to crops
                it = 1
                for obj in tree.iter("object"):
                    name = obj.find('.//name').text
                    dst_fpath = os.path.join(dst_dir, name, fp_rel, f"{os.path.splitext(fn)[0]}_{it}{ext}")
                    if os.path.exists(dst_fpath):
                        it += 1
                        continue
                    
                    xmin = round(float(obj.find('.//bndbox//xmin').text) / img_width, 5)
                    ymin = round(float(obj.find('.//bndbox//ymin').text) / img_height, 5)
                    bbox_width = round((float(obj.find('.//bndbox//xmax').text) / img_width) - xmin, 5)
                    bbox_height = round((float(obj.find('.//bndbox//ymax').text) / img_height) - ymin, 5)
                    crop = remove_background(Image.open(img_fpath), [xmin, ymin, bbox_width, bbox_height])
                    # dst_fpath = os.path.join(dst_dir, name, f"{os.path.splitext(fn)[0]}_{it}{ext}")
                    # print(f"dst_fpath : {dst_fpath}")
                    
                    if crop is None:
                        print("THIS CROP IS NONE")
                        print(f"name : {name}")
                        print(f"fp_src : {fp_src}")
                        print(f"img_fpath : {img_fpath}")
                        print(f"fp_rel : {fp_rel}")
                        print(f"img_height : {img_height}")
                        print(f"img_width : {img_width}")
                        print(f"name : {name}")
                        print(f"xmin : {xmin}")
                        print(f"ymin : {ymin}")
                        print(f"bbox_width : {bbox_width}")
                        print(f"bbox_height : {bbox_height}")
                        print(f"crop : {crop}")
                        continue

                    
                    # print(f"dst_fpath : {dst_fpath}\n")
                    Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
                    
                    crop.save(dst_fpath)
                    it += 1

# check if src_dir exists
for src_dir in src_dirs:
    if not os.path.exists(src_dir):
        print(f"src_dir '{src_dir}' does not exist!")
        exit()
    if not os.path.isdir(src_dir):
        print(f"src_dir '{src_dir}' is not a directory!")
        exit()
    print(f"src_dir '{src_dir}' exists!")
print("\n\nall src_dirs exist!\n\n")
        
# check if dst_dir exists, if not create it
for dst_dir in dst_dirs:
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

# process all dirs in loop
for src_dir, dst_dir in zip(src_dirs, dst_dirs):
    process_src_dir(src_dir, dst_dir)


# exit()

# for xml_fname in tqdm(os.listdir(src_dir)):
#     if xml_fname.endswith(".xml"):

#         # check image extention
#         for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
#             if os.path.isfile(os.path.splitext(os.path.join(src_dir, xml_fname))[0] + ext):
#                 break
#         else:
#             continue

#         # init vars
#         xml_fpath = os.path.join(src_dir, xml_fname)
#         img_fname = os.path.splitext(xml_fname)[0] + ext
#         img_fpath = os.path.join(src_dir, img_fname)
#         tree = et.parse(xml_fpath)
#         img_height = int(tree.find('.//size//height').text)
#         img_width  = int(tree.find('.//size//width').text)

#         # convert objects to crops
#         it = 1
#         for obj in tree.iter("object"):
#             name = obj.find('.//name').text
#             xmin = round(float(obj.find('.//bndbox//xmin').text) / img_width, 5)
#             ymin = round(float(obj.find('.//bndbox//ymin').text) / img_height, 5)
#             bbox_width = round((float(obj.find('.//bndbox//xmax').text) / img_width) - xmin, 5)
#             bbox_height = round((float(obj.find('.//bndbox//ymax').text) / img_height) - ymin, 5)
#             crop = remove_background(Image.open(img_fpath), [xmin, ymin, bbox_width, bbox_height])
#             dst_fpath = os.path.join(dst_dir, name, f"{os.path.splitext(xml_fname)[0]}_{it}{ext}")
#             Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
#             crop.save(dst_fpath)
#             it += 1