
# this is a helper script that perpares LILA BC dataset with individual json files next to the images
# the individual json files contain the location ID and other metadata
# this information will then be gathered in the split-based-on-locations.py script

# Make sure the dataset is raw and untouched. The filepaths should not be altered.
# donwload the dataset metadata json from https://lila.science/datasets

# you'll want to have the json in the same folder as the images
# make sure the paths match. Open thje json and manually check. 

# for the ohio-small-animals dataset, the json should be put next to the "Images/" folder

# {
#  "images": [
#   {
#    "id": "Images/Blanks/BIWA4S2020-06-25_16-19-56.JPG",
#    "file_name": "Images/Blanks/BIWA4S2020-06-25_16-19-56.JPG",
#    "datetime": "2020-06-25 16:19:56",
#    "location": "BIWA4S",
#    "seq_id": "location_BIWA4S_sequence_index_00000",
#    "seq_num_frames": 3,
#    "frame_num": 0
#   },

# the script will error if it cant find 10000 images in a row. Sometimes some images are 
# missing due to privacy etc., so thats why we use 10000 in a row.




# do this with all datasets you are processing with the split-based-on-locations.py script
# that will make sure it will find the proper locations.

import json
import os
from tqdm import tqdm

## COMMAND

# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\extract_location_id.py"



main_metadata_file = r"G:\lila-bc-africa\raw\snapshot-safari-2024-expansion\snapshot_safari_2024_metadata.json"

print(f"Processing {main_metadata_file}...")

# this will read the lila bc dataset json and write the location ID to indiviudal jsons next to the images
# it assumes that the images are in the same folder as the json file
def write_individual_json_files_with_image_information(json_file):
    image_parent_dir = os.path.dirname(json_file)
    dataset_name = os.path.splitext(os.path.basename(json_file))[0]

    print(f"Image parent directory: {image_parent_dir}")

    # Load the COCO JSON file
    with open(json_file, "r") as f:
        coco_data = json.load(f)

    print(f"Dataset name: {dataset_name}")
    print(f"Number of images: {len(coco_data['images'])}")

    # keep track of how many images in a row are missing
    n_consecutive_files_not_found = 0

    # Iterate over each image entry
    for image in tqdm(coco_data["images"]):
        file_name = os.path.normpath(image["file_name"])
        metadata = {
            "location": dataset_name + "_" + image["location"],
            "seq_id": image.get("seq_id", "unknown"),
            "seq_num_frames": image.get("seq_num_frames", "unknown"),
            "frame_num": image.get("frame_num", "unknown"),
            "datetime": image.get("datetime", "unknown")
        }
        
        # Generate JSON file path
        img_fpath = os.path.join(image_parent_dir, file_name)
        if os.path.exists(img_fpath):
            individual_json_path = os.path.join(image_parent_dir, file_name + ".json")
            with open(individual_json_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)
            n_consecutive_files_not_found = 0
        else:
            n_consecutive_files_not_found += 1
        
        # error if more than 10000 images in a row are missing
        if n_consecutive_files_not_found > 10000:
            print("\n\n")
            print(f"Image not found!")
            print(f"img_fpath        : {img_fpath}")
            print(f"json_file        : {json_file}")
            print(f"image_parent_dir : {image_parent_dir}")
            print(f"dataset_name     : {dataset_name}")
            print(f"image_metadata   :\n {json.dumps(image, indent=4)}")
            
            print("Tt assumes that the images are in the same folder as the json file. Open the JSON file and check the paths.")
            exit()



# run
write_individual_json_files_with_image_information(main_metadata_file)