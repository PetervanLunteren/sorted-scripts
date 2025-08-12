# Script to move all files a number of dirs up. Handy if you have a dir with many subdirs 
# and want to get rid of the subdirs. Just make sure all the files have unique names. 
# It will not overwrite, but will error.

# Example command: python move_files_one_dir_up_recursive.py "/Users/m1/Desktop/PHOTOS/Great_Tinamou" 2

import os
import sys
import shutil
from pathlib import Path

chosen_dir = str(sys.argv[1])
n_dirs_up = int(sys.argv[2])

def up_one_dir(path):
    try:
        parent_dir = Path(path).parents[n_dirs_up]
        shutil.move(path, parent_dir)
    except IndexError:
        # no upper directory
        pass

for subdir, dirs, files in os.walk(chosen_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if not file_path.endswith(".DS_Store"):
            print(file_path)
            up_one_dir(file_path)
