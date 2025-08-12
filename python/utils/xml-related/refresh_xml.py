# Script to update the folder, filename and path tags of the xml files to the current folder, filename and path

import os
import xml.etree.ElementTree as et

chosen_dir = r"/Users/petervanlunteren/Desktop/labelled_imgs"

for file in os.listdir(chosen_dir):
    if file.endswith(".xml"):
        xml_file = os.path.join(chosen_dir, file)
        tree = et.parse(xml_file)
        new_folder = str(os.path.basename(os.path.dirname(xml_file)))
        tree.find('.//folder').text = new_folder
        old_file_extension = os.path.splitext(tree.find('.//filename').text)[1]
        new_filename = os.path.basename(os.path.splitext(xml_file)[0] + old_file_extension)
        tree.find('.//filename').text = new_filename
        new_path = os.path.splitext(xml_file)[0] + old_file_extension
        tree.find('.//path').text = new_path
        tree.write(xml_file)
        print(xml_file, "updated!")
