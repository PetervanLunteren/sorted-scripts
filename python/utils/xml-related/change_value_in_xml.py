# Script to find specific values in xml files and change it. It also refreshes the xml file with the new filename and paths etc.
# This particular script chnages all name values to a user input. You can use it from the command line:

# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\change_value_in_xml.py"


import os
import sys
from collections import defaultdict
import xml.etree.ElementTree as et
import json
import tqdm

chosen_dir = r"D:\2024-16-SCO\imgs\20240210\raw\annotated-full"

species_count = defaultdict(int)

xml_files = [file for file in os.listdir(chosen_dir) if file.endswith(".xml")]

for file in tqdm.tqdm(xml_files, desc="Processing XML files"):
    xml_file = os.path.join(chosen_dir, file)
    tree = et.parse(xml_file)

    # # update the attributes to trick EcoAssist into thinking everything was manually verified
    # root = tree.getroot()
    # root.set('json_updated', 'yes')
    # root.set('verified', 'yes')

    # # refreshes the xml file with the new filename and paths etc
    # new_folder = str(os.path.basename(os.path.dirname(xml_file)))
    # tree.find('.//folder').text = new_folder
    # old_file_extension = os.path.splitext(tree.find('.//filename').text)[1]
    # new_filename = os.path.basename(os.path.splitext(xml_file)[0] + old_file_extension)
    # tree.find('.//filename').text = new_filename
    # new_path = os.path.splitext(xml_file)[0] + old_file_extension
    # tree.find('.//path').text = new_path

    # logic to extract species from filename
    species = file.split("_-_")[0] 
    if species == "Thermal Images":
        species = "Animal (thermal)"
    elif species == "Feral goats":
        species = "Feral goat"
    
    # loop over all objects
    for i in tree.iter("object"):
        
        # adjust the object name
        i.find('.//name').text = species
        
        # count the number of objects
        species_count[i.find('.//name').text] += 1

    # write new xml file
    tree.write(xml_file)
    
    # # print statement
    # print(xml_file, "updated!")
    # print(f"{file.ljust(80)} updated! -> {species}")
        
print(json.dumps(species_count, indent = 2))
