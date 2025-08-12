# Script to remove one dir in the paths of each file. This script was used to get rid of 
# incorrectly made sequence directories, but keep the rest of the folder structure in place.
# You can select the folder to be removed with regex.
# For example:
# Users/peter/Desktop/elephant/PvL_seq_12345/serengeti/S1/img1.jpg -> Users/peter/Desktop/elephant/serengeti/S1/img1.jpg
# Users/peter/Desktop/elephant/PvL_seq_12345/serengeti/S2/img2.jpg -> Users/peter/Desktop/elephant/serengeti/S2/img2.jpg
# Users/peter/Desktop/elephant/PvL_seq_12345/serengeti/S3/img3.jpg -> Users/peter/Desktop/elephant/serengeti/S3/img3.jpg
# Users/peter/Desktop/elephant/PvL_seq_12345/serengeti/S4/img4.jpg -> Users/peter/Desktop/elephant/serengeti/S4/img4.jpg
# Users/peter/Desktop/elephant/PvL_seq_12345/karoo/S1/img1.jpg     -> Users/peter/Desktop/elephant/karoo/S1/img1.jpg



# python "C:\Users\smart\Desktop\remove_one_dir_from_path.py"


import os
import shutil
from pathlib import Path

root = r"C:\Peter\fetch-lila-bc-data\downloads\loxodonta africana"

# loop recursively through all files
for subdir, dirs, files in os.walk(root):
    for file in files:
        file_path = os.path.join(subdir, file)
        if not file_path.endswith(".DS_Store"):

            # remove folder and create src and dst
            src_list = list(Path(file_path).parts)
            dst_list = [i for i in src_list if not i.startswith('PvL_seq_')]
            src = Path(*src_list)
            dst = Path(*dst_list)

            # debug purposes
            print("")
            print(f"src_list : {src_list}")
            print(f"dst_list : {dst_list}")
            print(f"src      : {src}")
            print(f"dst      : {dst}")

            # move file
            Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
