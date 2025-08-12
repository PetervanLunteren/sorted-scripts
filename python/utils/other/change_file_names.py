# Script to change all filenames in a dir

import os
import xml.etree.ElementTree as et

chosen_dir = r"/Users/petervanlunteren/Desktop/labelled_imgs"

for old_filename in os.listdir(chosen_dir):
    new_filename = old_filename.replace("_images", "")
    old_path = os.path.join(chosen_dir, old_filename)
    new_path = os.path.join(chosen_dir, new_filename)
    os.rename(old_path, new_path)
    print("Renamed '" + old_filename + "' to '" + new_filename + "'")