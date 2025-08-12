# This script is designed to quickly review random selections of 100 crops to get a feeling of the dataset. 
# Is everything mostely taken at night? Or at Day? What are the camera setups? Is there anything off? Does it need verification?? 
# Press any key to continue... The files are not moved in any way.

# simply supply the label_map xlsx and it will take all paths and classes automatically. 

# Peter van Lunteren, 24 oct 2023

# WINDOWS: execute script in miniforge prompt

# conda activate "C:\Users\smart\AddaxAI_files\envs\env-pytorch" && python "C:\Users\smart\Desktop\inspect_random_selections.py"


# user input
label_map_xlsx = r"C:\Users\smart\Desktop\2024-25-ARI-spp-plan.xlsx"

import openpyxl # pip install openpyxl
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd



def read_excel_file(filename):
    workbook = openpyxl.load_workbook(filename)
    sheet = workbook["label_map"]
    nested_list = []
    headers = [cell.value for cell in sheet[1]]
    class_col = headers.index('Class') + 1
    path_col = headers.index('Path') + 1
    source_col = headers.index('Source') + 1
    for row in sheet.iter_rows(min_row=2, values_only=True):
        class_value = row[class_col - 1]
        source_value = row[source_col - 1] 
        path_value = row[path_col - 1] 
        nested_list.append([class_value, source_value, path_value])
    return nested_list

def up_or_downsample(list, n):
    return pd.DataFrame(list).sample(n=n, replace = len(list) < n)[0].tolist()

def show_images(list_of_files, cat, source, spp_name):
    print(" Press 'q' to quit.\n Press 'spacebar' to display the next batch of images.\n Press 'enter' to go to the next folder in the list.")
    random.shuffle(list_of_files)
    chunks = [list_of_files[x:x+100] for x in range(0, len(list_of_files), 100)]

    for chunk in tqdm(chunks):
        if len(chunk) < 100:
            chunk = up_or_downsample(chunk, 100)
        cols = []
        cv_chunk = []
        for im in chunk:
            img = cv2.imread(im)
            if img is None:
                continue
            cv_chunk.append(cv2.resize(img, (75, 75)))

        if len(cv_chunk) == 0:
            print(f"⚠️ No valid images found in chunk.")
            continue

        squared_list = [cv_chunk[x:x+10] for x in range(0, len(cv_chunk), 10)]
        for img_row in squared_list:
            cols.append(np.vstack(img_row))

        try:
            collage = np.hstack(cols)
        except ValueError as e:
            print(f"\n\n ERROR : {e} \n\n")
            print(f"\n\n NOT ABLE TO SHOW IMAGES FOR {spp_name} \n\n")
            break

        title = f"CLASS {cat} - SOURCE {source} - SPECIES {spp_name}"

        # Show image using matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

        key = input("Press Enter to continue, [q] to quit, [n] for next category: ").strip().lower()
        if key == 'q':
            return False
        elif key == 'n':
            break

    return True

# def show_images(list_of_files, cat, source, spp_name):
#     print(" Press 'q' to quit.\n Press 'spacebar' to display the next batch of images.\n Press 'enter' to go to the next folder in the list.")
#     random.shuffle(list_of_files)
#     chunks = [list_of_files[x:x+100] for x in range(0, len(list_of_files), 100)]
    
#     for chunk in tqdm(chunks):
#         if len(chunk) < 100:
#             chunk = up_or_downsample(chunk, 100)
#         cols = []
#         cv_chunk = []
#         for im in chunk:
#             cv_chunk.append(cv2.resize(cv2.imread(im), (75, 75)))

#         squared_list = [cv_chunk[x:x+10] for x in range(0, len(cv_chunk), 10)]
#         for img_row in squared_list:
#             cols.append(np.vstack(img_row))
#         try:
#             collage = np.hstack(cols)
#         except ValueError as e:
#             print(f"\n\n ERROR : {e} \n\n")
#             print(f"\n\n NOT ABLE TO SHOW IMAGES FOR {spp_name} \n\n")
#             break
#         window_name = "window"
#         title = f"CLASS {cat} - SOURCE {source} - SPECIES {spp_name}"
#         cv2.setWindowTitle(window_name, title)
#         cv2.imshow(window_name, collage)
        
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('q'):  # Press 'q' to quit
#             cv2.destroyAllWindows()
#             return False
#         elif key == 32:  # Press spacebar (32 is ASCII code for space)
#             continue  # Continue displaying the next batch of images
#         elif key == 13 or key == 10:  # Press enter/return to move to next directory
#             break  # Exit the loop to move to the next directory
    
#     cv2.destroyAllWindows()
#     return True




for category, source, dir_to_be_checked in read_excel_file(label_map_xlsx):
    list_of_files = []
    print(f"Looking for files in {dir_to_be_checked}")
    scientific_name = os.path.basename(os.path.normpath(dir_to_be_checked))
    for subdir, dirs, fnames in os.walk(dir_to_be_checked):
        for fname in fnames:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                fpath_src = os.path.join(subdir, fname)
                list_of_files.append(fpath_src)
    
    if not show_images(list_of_files, category, source, scientific_name):
        break  # Exit if 'q' is pressed





exit()



import os

# user input
dir_to_be_checked = r'C:\Peter\projects\2024-14-CAN\training-set\lila-extra-crops\arctonyx collaris'

# packages
from PIL import Image
import cv2 
import cv2 
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
from pathlib import Path
import math
import numpy as np
import random

def show_images(list_of_files):
    random.shuffle(list_of_files)
    chunks = [list_of_files[x:x+100] for x in range(0, len(list_of_files), 100)]
    for chunk in tqdm(chunks):

        cols = []
        cv_chunk = []
        for im in chunk:
            cv_chunk.append(cv2.resize(cv2.imread(im), (75, 75)))

        squared_list = [cv_chunk[x:x+10] for x in range(0, len(cv_chunk), 10)]
        for img_row in squared_list:
            cols.append(np.vstack(img_row))
        collage = np.hstack(cols)
        cv2.imshow("", collage)
        cv2.waitKey(0)

# create list of images to be verified (src and dst)
list_of_files = []
print(f"Looking for files in {dir_to_be_checked}")
for subdir, dirs, fnames in os.walk(dir_to_be_checked):
    for fname in fnames:
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            fpath_src = os.path.join(subdir, fname)
            list_of_files.append(fpath_src)

# run
show_images(list_of_files)