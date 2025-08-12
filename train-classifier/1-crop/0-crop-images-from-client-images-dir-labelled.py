# This script is designed to convert client annotated images into crops that can be used as training data for a classifier.
# The images must be sorted in species subfolders. If they are labelled with PASCAL VOC, see 0-crop-images-from-client-images-xml-labelled.py.

# The root must be structured as follows and have an MD ecoassist output file which has located all the animals in the images.

# root/
#   /image_regicnition_file.json
#   /species_A_sub_folder
#   /species_B_sub_folder
#   /species_C_sub_folder
#   /etc

# In the species subfolders there may be as many subfolder as you want (original folder structure preferred), 
# but the first folder must contain the species name. This will be used for the category.

# you need to add some custom logic to the script to extract the location information from the path and write it to a json file.
# it needs the location information (from path? from csv? from json?) to be able to split the images based on location.
# search for "CUSTOM CODE" in the script to see where you need to add your custom logic.

# If you used claude to extract the location information using "/Users/peter/Documents/scripting/claude-prompts/path-idependence-analysis.yaml", you can import the CSV file search CLAUDE CSV FORMAT

# in short, the steps are:
# 1. Use AddaxAI to run MD on root without any classifier. We'll use this output file to crop out the animal detections.
# 2. Run this script, which:
#       - loops thourgh MD output and crops out the detections
#       - places them in the same folder structre in a user defined folder
#       - saves the crops as images
#       - writes a json file with the location information for each crop, so that split-based-on-locations.json can read it and split based on location

# EXECUTION

# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\0-crop-images-from-client-images-dir-labelled.py"


# USER INPUT
src_dir = r"C:\Peter\projects\2024-25-ARI\data\raw\imgs+frames"
dst_dir = r"C:\Peter\projects\2024-25-ARI\data\fmt\local"
database_name = "2024-25-ARI"


# IMPORT PACKAGES
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import json
import csv
from PIL import Image, ImageOps, ImageFile

# let's don't freak out when the image is incomplete
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    box_size = pad_crop(box_size) # hier worden de boxen dus gepad - kleine dieren worden niet extreem vergroot
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

# load a csv file with file_path and site_id columns to lookup
# CLAUDE CSV FORMAT
def load_site_lookup(csv_path):
    """Loads a CSV and returns a dict mapping file_path → dict of identifiers."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {
            row['filepath']: {
                'organisation_id': row['organisationID'],
                'site_id': row['siteID'], 
                'deployment_id': row['deploymentID'],
                'date': row['date']
            }
            for row in reader
        }
    
csv_lookup_fpath = os.path.join(src_dir, "path-independence-analysis.csv")

if os.path.exists(csv_lookup_fpath):
    csv_lookup_present = True
else:
    csv_lookup_present = False

if csv_lookup_present:
    csv_lookup = load_site_lookup(csv_lookup_fpath)
    
    
    
    
    

# LOOP
counter = 0
fnames = []
recognition_file = os.path.join(src_dir, "image_recognition_file.json")

# open json file
with open(recognition_file) as image_recognition_file_content:
    data = json.load(image_recognition_file_content)

for image in tqdm(data['images']):

    # get image info
    file = image['file']
    n_crop = 0

    # loop through detections
    if 'detections' in image:
                
        # initialize n_crop
        n_crop = 0
        
        # EDIT NOV 2024: NEW
        # the following code will only take the highest confidence animal detection in the image if there are no vehicles or persons present
        # methods copied from https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12318
        
        #  "detection_categories": {
        #   "1": "animal",
        #   "2": "person",
        #   "3": "vehicle"
        #  },
        
        # get all detections persons and vehicles above 0.7 confidence
        person_detections = [detection for detection in image['detections'] if detection['conf'] <= 0.7 and detection['category'] == '2']   # "2": "person",
        vehicle_detections = [detection for detection in image['detections'] if detection['conf'] <= 0.7 and detection['category'] == '3']  # "3": "vehicle"
        
        # if there are persons or vehicles in the image above 0.7, skip image
        if len(person_detections) > 0 or len(vehicle_detections) > 0:
            continue

        # if there are only animals in the image, sort them by confidence
        sorted_animal_detections = sorted(
                        [detection for detection in image['detections'] if detection['category'] == '1'], # "1": "animal",
                        key=lambda x: (-x['conf'])
                    )
        
        # and take only the highest confidence animal, if there is one
        if len(sorted_animal_detections) == 0:
            continue       
        detection = sorted_animal_detections[0]
        
        # save crop or full image depending on user input
        conf = detection['conf']
        if conf >= 0.4: # skip all animals with a confidence below 0.4
            
            
            
            # crop and save
            bbox = detection['bbox']
            src_img = os.path.join(src_dir, file)
            crop = remove_background(Image.open(src_img), bbox)
            path_elems = os.path.normpath(file).split(os.path.sep)
            spp_class = path_elems[0]
            fname = path_elems[-1]
            n_crop += 1
            subdirs = path_elems[1:-1]
            if subdirs != []:
                folder_structure = os.path.join(*subdirs)
                dst_fpath = os.path.join(dst_dir, spp_class, database_name, folder_structure, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
            else:
                dst_fpath = os.path.join(dst_dir, spp_class, database_name, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
            Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
            crop.save(dst_fpath)
            
            
            
            
            
            
            
            
            #### EXPLENATION #####
            
            
            
            # write an individual json file with the location information so that split-based-on-locations.json can read it and split based on location
            
            # each image should have an associated json file with the same name as the image + ".json", e.g.:
            # puertorico_1a_20141207_075345_img_0058_1.jpg.json
            # puertorico_1a_20141207_075345_img_0058_1.jpg
            
            # the json file should contain the location information, e.g.:
            
            # {
            #     "location": "island-conservation-camera-traps_puertorico_1a"
            # }
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            ######### CUSTOM CODE SITEID in NAME #########
            
        
            # this is some custom code....
            # it depends on the project how the location is stored.
            
            # # in this particular case, the location is stored in the path as folder name that starts with "location_"
            # location = [elem for elem in path_elems if elem.startswith("location_")]
            # if len(location) != 1:
            #     print(f"WARNING: {file} has no unique location folder. Found {len(location)} locations.")
            #     exit()
            # location = location[0].replace("location_", "")
            # location_json_path = dst_fpath + ".json"
            # location_json_content = {"location": database_name + '_' + location}
            # with open(location_json_path, "w") as json_file:
            #     json.dump(location_json_content, json_file, indent=4)
        
        
        
        
        
        
        
        
        
        
        
        
        
            ######### CUSTOM CODE SITEID in CSV CLAUDE CSV FORMAT #########

            if csv_lookup_present:
                
                key = os.path.normpath(src_img).replace(os.path.normpath(src_dir), ".")
                
                # this is for when you have a csv file with file_path and site_id columns to lookup
                identifiers = csv_lookup[key]
                
                organisation_id = identifiers['organisation_id'].strip().replace(" ", "_")
                site_id = identifiers['site_id'].strip().replace(" ", "_")
                deployment_id = identifiers['deployment_id'].strip().replace(" ", "_")
                date = identifiers['date'].strip().replace(" ", "_")
                
                # # independence analysis
                # print(f"organisation_id: {organisation_id}")
                # print(f"site_id: {site_id} {type(site_id)}")
                # print(f"deployment_id: {deployment_id} {type(deployment_id)}")
                # print(f"date: {date} {type(date)}")
                
                independence_statistic = None

                # we'll take the site_id if available
                if site_id is not None and site_id != "":
                    independence_statistic = f"{organisation_id}_{site_id}"

                # we'll take the deployment id if that is available
                if independence_statistic is None:
                    if deployment_id is not None and deployment_id != "":
                        independence_statistic = f"{organisation_id}_{deployment_id}"

                # we'll take the date if that is available
                if independence_statistic is None:
                    if date is not None and date != "":
                        independence_statistic = f"{organisation_id}_{date}"
                        
                # print(f"independence_statistic: {independence_statistic}\n\n")
             
                # if none of that is available, we'll skip the json file creation
                if independence_statistic:
                    location_json_path = dst_fpath + ".json"
                    location_json_content = {"location": database_name + '_' + independence_statistic}
                    with open(location_json_path, "w") as json_file:
                        json.dump(location_json_content, json_file, indent=4)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # # EDIT NOV 2024: NEW
        # # sort detections by category and confidence
        # # vehicles and people are listed first, then per category sorted by confidence
        # # and all detections below the *thresh* are removed
        # sorted_filtered_detections = sorted(
        #     [detection for detection in image['detections'] if detection['conf'] >= thresh],
        #     key=lambda x: (-int(x['category']), -x['conf'])
        # )
        
        # # if the frist item is not an animal, skip image
        # # that means there is an vehicle or person above the *thresh*
        # if sorted_filtered_detections[0]['category'] != '1':
        #     continue

        # # get the first detection, which is the highest confidence animal
        # detection = sorted_filtered_detections[0]
        
        # # crop and save
        # bbox = detection['bbox']
        # crop = remove_background(Image.open(src_img), bbox)
        # path_elems = os.path.normpath(file).split(os.path.sep)
        # spp_class = path_elems[0]
        # fname = path_elems[-1]
        # n_crop += 1
        # subdirs = path_elems[1:-1]
        # if subdirs != []:
        #     folder_structure = os.path.join(*subdirs)
        #     dst_fpath = os.path.join(dst_dir, spp_class, database_name, folder_structure, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
        # else:
        #     dst_fpath = os.path.join(dst_dir, spp_class, database_name, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
        # Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
        # crop.save(dst_fpath)
        
        

        # # EDIT NOV 2024: OLD        
        # for detection in image['detections']:

        #     # get confidence
        #     conf = detection["conf"]
        #     cat = detection['category']

        #     # if animal and above user specified thresh
        #     if conf >= thresh and cat == '1':

        #         fnames.append(src_img)

        #         bbox = detection['bbox']
        #         crop = remove_background(Image.open(src_img), bbox)
        #         path_elems = os.path.normpath(file).split(os.path.sep)
        #         spp_class = path_elems[0]
        #         fname = path_elems[-1]
        #         n_crop += 1
        #         subdirs = path_elems[1:-1]
        #         if subdirs != []:
        #             folder_structure = os.path.join(*subdirs)
        #             dst_fpath = os.path.join(dst_dir, spp_class, database_name, folder_structure, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
        #         else:
        #             dst_fpath = os.path.join(dst_dir, spp_class, database_name, f"{os.path.splitext(fname)[0]}_{n_crop}{os.path.splitext(fname)[1]}")
        #         Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
        #         crop.save(dst_fpath)

