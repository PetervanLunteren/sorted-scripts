# Script to loop through json and find unique values
# Peter van Lunteren 16 sept 2023

'''
pmset noidle &;conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv;python "/Users/peter/Documents/scripting/sorted-scripts/python/utils/loop_json_find_values.py"
'''

import json
import re




recognition_file = "/Users/peter/Documents/scripting/lila-bc-summary/json/wcs_20220205_bboxes_with_classes.json"


# open json file
locations = []
with open(recognition_file) as image_recognition_file_content:
    data = json.load(image_recognition_file_content)
# n_images = len(data['images'])
for image in data['images']:
    location = image['country_code']

    # match = re.search('.+?(?=_)', image['location'])
    # if match:
    #     location = match.group()
    if location not in locations:
        locations.append(location)

print(locations)
