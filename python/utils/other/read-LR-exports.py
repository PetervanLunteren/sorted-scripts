# script to read the LR exported paths and extract the images from its 'IntoNature4' folder 
# and separate them into species folders.


# conda activate ecoassistcondaenv & python "C:\Users\smart\Desktop\read-LR-exports.py"

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def unpack_txt_file(txt_file):
    with open(txt_file) as f:
        paths = f.read()
    paths = paths.split("\t")
    sorted(paths, key=str.lower)

    return paths


debug_n = 0

def extract_spp_files(txt_file):
    global debug_n

    head, tail = os.path.split(txt_file)
    spp_name = os.path.splitext(tail)[0].lower()
    # spp_group = spp_group.lower()
    # if spp_group == "bird species":
    #     spp_group = "birds"

    with open(txt_file) as f:
        paths = f.read()

    paths = paths.split("\t")
    sorted(paths, key=str.lower)

    n_present = 0
    absent = []
    n_images = 0
    n_videos = 0

    for original_path in tqdm(paths):
        if original_path != "":
            original_path = os.path.normpath(original_path)
            rel_path = original_path.replace('\\Users\\smartparks\\Documents\\IntoNature4\\', '')
            src = os.path.join('C:\\Peter\\desert-lion-project\\philips-footage\\unsorted\\IntoNature4', rel_path)

            if src.lower().endswith(('.png', '.jpg', '.jpeg')):
                n_images += 1
                typ = "images"
            else:
                n_videos += 1
                typ = "videos"

            dst = os.path.join('C:\\Peter\\desert-lion-project\\philips-footage\\sorted', spp_name, rel_path)

            # print("")
            # print(f"spp_name      : {spp_name}")
            # print(f"original_path : {original_path}")
            # print(f"rel_path      : {rel_path}")
            # print(f"src           : {src}")
            # print(f"dst           : {dst}")

            if os.path.isfile(src):
                if not image_dict[src]:
                    n_present += 1
                    debug_n += 1
                    # print(f"debug_n : {debug_n}")
                    # print(f"src     : {src}")
                    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
                    shutil.move(src, dst)
            else:
                absent.append(src)
            
    # print("")
    print(f"Absent  : {len(absent)}")
    # print(f"absent_paths       : {absent}")
    print(f"Present : {n_present}")
    # print(f"n_absent           : {len(absent)}")
    # print(f"n_images           : {n_images}")
    # print(f"n_videos           : {n_videos}")
 

# image_fp = []

# for folder in ["Carnivores", "Mammals", "Bird Species"]:
#     print(f"folder   : {folder}")
#     folder_path_head = r"C:\Peter\desert-lion-project\philips-footage\species-annotated-paths"
#     folder_path_full = os.path.join(folder_path_head, folder)
#     txt_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path_full) for f in filenames if os.path.splitext(f)[1] == '.txt']
#     for txt_file in txt_files:
#         print(f"txt_file : {txt_file}")
#         image_fp.extend(unpack_txt_file(txt_file))
#         print(f"n_images : {len(image_fp)}")

# print(f"total image paths  : {len(image_fp)}")
# single_class_images = list(dict.fromkeys(image_fp))
# print(f"unique image paths : {len(single_class_images)}")

# # total image paths  : 74430
# # unique image paths : 68730



root = r"C:\Peter\desert-lion-project\philips-footage\species-annotated-paths"
txt_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root) for f in filenames if os.path.splitext(f)[1] == '.txt']

image_dict = {}

for txt_file in txt_files:
    print(txt_file)
    with open(txt_file) as f:
        paths = f.read()
    paths = paths.split("\t")
    for path in paths:
        original_path = os.path.normpath(path)
        rel_path = original_path.replace('\\Users\\smartparks\\Documents\\IntoNature4\\', '')
        src = os.path.join('C:\\Peter\\desert-lion-project\\philips-footage\\unsorted\\IntoNature4', rel_path)
        # print(f"src : {src}")
        if src in image_dict:
            image_dict[src] = True
        else:
            image_dict[src] = False
# src : C:\Peter\desert-lion-project\philips-footage\unsorted\IntoNature4\Camera Trap\North\Okongwe_Water\2009\11\SUNP0727.JPG
# src : C:\Peter\desert-lion-project\philips-footage\unsorted\IntoNature4\Camera Trap\North\Okongwe_Water\2011\01\20110115-PICT2377.JPG

n_dupl = 0
n_single = 0
for path, dupl_bool in image_dict.items():
    if dupl_bool:
        n_dupl += 1
    else:
        n_single += 1

print(f"n_dupl   : {n_dupl}")
print(f"n_single : {n_single}")

# n_dupl   : 5254
# n_single : 63476



for folder in ["Carnivores", "Mammals", "Bird Species"]:
    folder_path_head = r"C:\Peter\desert-lion-project\philips-footage\species-annotated-paths"
    folder_path_full = os.path.join(folder_path_head, folder)
    txt_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path_full) for f in filenames if os.path.splitext(f)[1] == '.txt']
    for txt_file in txt_files:
        extract_spp_files(txt_file)

