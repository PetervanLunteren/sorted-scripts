# Script to find locations in LILA BC metadata or, if not LILA ds, estimate locations based on filename patterns and copy files from 
# the main database to a training  folder train/classA, train/classB, test/classA, test/classB, val/classA, val/classB, etc.

# IMPORTANT: This script is quite new (march 202) and hasnt been thoughroughly tested. Make sure you do some checks if all goes well.

# if you want to retrieve location IDs from LILA BC, you'll need to process the dataset with the extract_location_id.py script first.

# it will probabaly prompt the user to check the locations and adjust the parameters if needed. You'll see.

# This script organizes files into locations based on their folder structure. It first checks if the location ID could be retrieved from
# LIla BC (see comment abvove). If not, it calculates the similarity between the file paths of images and groups them into locations based on 
# the similarity scores. You'll get the chance to visually check the locations and adjust the parameters if needed.

# After grouping the files, it splits them into training, validation, and test sets. Then, it copies the files 
# into the right folders for each set.

# Next, it creates a CSV files and logs, listing the file paths and their corresponding class labels. It 
# also generates a class file that maps class names to IDs.


# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\split-based-on-locations.py"


# Peter van Lunteren, 12 mar 2025

# TODO: Add a dedupe step that loops through the images in order of dir layer, and image with the next. If more than 
#           x% similarity, do not add to the train set. This makes sure we have unsimilar images.  

# packages
import numpy as np
from collections import defaultdict 
import os
import numpy as np
import difflib
import pandas as pd
from pathlib import Path
import json
import csv
from tqdm import tqdm
import os
import shutil
import pathlib
import random
import sys

# user iputs
spp_map_xlsx = r"C:\Users\smart\Desktop\2024-25-ARI-spp-plan.xlsx"                      # will read paths form here
dst_folder = r"F:\temp-train-sets\2024-25-ARI"                                    # path to destination folder
duplicate_without_test = True                                                           # if True, a duplicate dataset will be created witwith the test set added to train. The idea is that you can retrain on this set once the hyper parameter tuning has been done and the model is ready for publication.
duplicate_subset = True                                                                 # if True, a duplicate dataset will be created with only 20% of the data. The idea is that you can use this set for hyper parameter tuning. 
max_images_per_row = 25000                                                           # the maximum number of images per ROW in the train set. This is done by randomly deleting images above the *max_images_per_row* threshold. Disable this by setting it to a high number. If you have lines for local and non-local images, just take a number higher than the locals, and it will only cap the non-locals. 

# for the time estimation, we need some baselines of previous runs
eta_baselines = [
    {'img_size': 480, 'n_images': 238000, 'architecture': 'efficientnet_v2_m', 'time_min': 33}, # time_min is the number of minutes it took a full epoch, so train + val
    # You can add more baselines here for robustness
    # TODO: the architecture is not used yet calculate per architechture
]

# constants
AddaxAI_files = r"C:\Users\smart\AddaxAI_files"  # path to the AddaxAI_files to find MD utils

# Set seed for reproducibility
random_seed = 420
np.random.seed(random_seed)
random.seed(random_seed)

# load in the MD utils
sys.path.append(os.path.join(AddaxAI_files, "cameratraps"))
from megadetector.utils.split_locations_into_train_val import split_locations_into_train_val

# create dst folder if needed
Path(dst_folder).mkdir(parents=True, exist_ok=True)

# exit script if the training dir is not empty
if len(os.listdir(dst_folder)) != 0: 
    print(f"\n\nWARNING: dst dir {dst_folder} is not empty. Do you want to remove the content and continue?")
    continue_pompt = input("Press [Enter] to remove and continue, or type 'exit' to exit script.").strip().lower() 
    if continue_pompt == "exit":
        print("Exiting script....")
        exit()
    elif continue_pompt == "": # Enter
        print("Removing content from dst directory....")
        shutil.rmtree(dst_folder)
        Path(dst_folder).mkdir(parents=True, exist_ok=True)
    else:
        print("Invalid input. Exiting script....")
        exit()

# create log file
def log(msg):
    log_file = os.path.join(dst_folder, "log.txt")
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
    with open(log_file, "a") as f:
        f.write(str(msg) + "\n")
    print(msg)

# read Excel file
rows = pd.read_excel(spp_map_xlsx, sheet_name='label_map')[['Class', 'Source', 'Path']]
rows = rows.reset_index()
train_input = {}
for index, row in tqdm(rows.iterrows(), total=len(rows), desc="Processing label map"):
    spp_class = row['Class'].strip().lower()
    path = row['Path'].replace('"', '')
    source = row['Source'].strip().lower()

    if source not in {'local', 'non-local'}:
        log(f"ERROR READING LABEL MAP: Source for row index {index} is invalid: '{source}'")
        exit()

    if spp_class in train_input:
        if source in train_input[spp_class]:
            train_input[spp_class][source].append(path)
        else:
            train_input[spp_class][source] = [path]
    else:
        train_input[spp_class] = {source: [path]}
all_classes = list(train_input.keys())

# return a list of all image paths in a directory
def get_img_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_paths.append(os.path.join(root, file))
    return file_paths

# count imgs
log("\n\nCounting the number of images in the dataset...")
total_images = 0
img_counts_per_class = defaultdict(int)
for spp_class, source_dict in train_input.items():
    # log(spp_class)
    for source, paths in source_dict.items():
        for path in paths:
            if not os.path.exists(path):
                log(f"ERROR: Path {path} does not exist, while listed in the spp plan. Please check the spp plan and the paths.")
                exit()
            img_count = len(get_img_paths(path))
            total_images += img_count
            img_counts_per_class[spp_class] += img_count
log("\n\nTotal number of images found: " + str(total_images))
log("\n\nNumber of images per class:")
log(json.dumps(img_counts_per_class, indent=4))
average_images_per_class = round(total_images / len(all_classes))
log(f"\n\nAverage number of images per class: {average_images_per_class}")

# estimate the time it takes for one epoch based on the number of images
def estimate_training_time(eta_baselines, target_img_size, target_n_images):
    estimates = []
    for base in eta_baselines:
        base_area = base['img_size'] ** 2
        target_area = target_img_size ** 2
        area_ratio = target_area / base_area
        image_ratio = target_n_images / base['n_images']
        est_time = base['time_min'] * area_ratio * image_ratio
        estimates.append(est_time)
    avg_estimate = sum(estimates) / len(estimates)
    return avg_estimate

# get the image paths (rel, abs, class) in a nested list so they stay together
nested_list_fpath = os.path.join(r'C:\Users\smart\Desktop', "nested_list.json")
if not os.path.exists(nested_list_fpath):
    
    # create the nested list
    nested_list = []
    log("\n\nPopulating the nested list with all information...")
    for spp_class, source_dict in tqdm(train_input.items()):
        for source, paths in source_dict.items():
            for path in paths:
                img_paths = get_img_paths(path)
                if max_images_per_row:
                    img_paths = random.sample(img_paths, min(len(img_paths), max_images_per_row))
                for abs_path in img_paths:
                    rel_path = abs_path.replace(path, "")
                    rel_path = rel_path.lstrip(os.sep)
                    individual_json = abs_path + ".json"
                    if os.path.exists(individual_json):
                        with open(individual_json, "r") as f:
                            metadata = json.load(f)
                        location_id = metadata.get("location")
                    else:
                        location_id = None
                    nested_list.append([rel_path, abs_path, spp_class, location_id])

    # write the nested list to json
    log("saving to file...")
    with open(nested_list_fpath, 'w') as f:
        json.dump(nested_list, f)
    log(f"\n\nSaved rel_path_location_id_dict to {nested_list_fpath} so that you don't have to do this again.")
    
else:
    log(f"Skipping the creation of nested list. Loading from {nested_list_fpath}.")
    input(f"Press enter to proceed.")
    with open(nested_list_fpath, 'r') as f:
        nested_list = json.load(f)

# Optional: count images per class after row-based capping
from collections import Counter
class_counts = Counter([x[2] for x in nested_list])
log("\n\nFinal image count per class (after row-based capping):")
log(json.dumps(class_counts, indent=4))

# calculate the estimated time per epoch based on the actual nested_list length
n_images_with_cutoff = len(nested_list)
log(f"\n\nCalculating estimated time per epoch based on the current dataset...")
log(f"Total number of images after row-based capping: {n_images_with_cutoff}")

for img_size in [182, 224, 256, 320, 384, 448, 480, 512]:
    estimated_time = estimate_training_time(eta_baselines, img_size, n_images_with_cutoff)
    log(f"Estimated time per epoch with image size {img_size}: {estimated_time:.2f} minutes (~{estimated_time / 60:.2f} hours)")
log("\n")

# sort filepaths by directory layer, so first all the files in the first dir, then in the second, etc.
def sort_nested_paths_per_dir_layer(nested_list):
    # Split each path into components with a progress bar
    split_paths = []
    for sublist in tqdm(nested_list, desc="Splitting paths"):
        split_paths.append((os.path.normpath(sublist[0]).split(os.sep), sublist))

    # Sort paths alphabetically by each directory layer
    sorted_paths = sorted(tqdm(split_paths, desc="Sorting paths"), key=lambda x: tuple(x[0]))

    # Reconstruct the sorted list
    sorted_nested_list = [sublist for _, sublist in tqdm(sorted_paths, desc="Reconstructing list")]

    return sorted_nested_list

# Now we have the nested list sorted by the directory layers of the absolute paths
sorted_nested_list = sort_nested_paths_per_dir_layer(nested_list)

# function to compute similarity scores of the relative paths
# it calculates the similarity of the current relative path with the previous relative path
# relative because that is the part that the data provider has structured
# it then compares the similarity score with a rolling average of the previous similarity scores
# if the current similarity score is significantly lower than the rolling average, it is considered a new location
# it then returns two dicts: one with species counts per location, and one with the absolute paths per location
def split_on_location(nested_list):
    
    # Default parameter values
    rolling_average_window = 1000
    thresh = 0.15
    
    log("\n\n")
    log("You're about to start the location splitting process. This process will group images based on their location in the file path.")
    log("If you have retrieved the location IDs from LILA BC with extract_locatation_id.py, the process will use those to group the images.")
    log("The process will calculate the similarity between the file paths of images and group them into locations based on the similarity scores.")
    log("The process will then prompt you to visually check the locations and adjust the parameters if needed.")
    log("\n\n")
    
    while True:  # Loop to allow parameter changes
        # init vars
        relpaths = [sublist[0] for sublist in nested_list]
        abspaths = [sublist[1] for sublist in nested_list]
        spp_classes = [sublist[2] for sublist in nested_list]
        location_ids = [sublist[3] for sublist in nested_list]
        location_idx = 0 # for naming new locations
        current_location = []
        similarity_list = []
        
        # list with locations to visually check if the locations look alright
            # location 1:
            #    first  : \A1E__2020-03-25__05-02-35(1)__peromyscus_leucopus.JPG
            #    middle : \A1E__2020-04-07__17-05-09(3)__thamnophis_sirtalis.JPG
            #    last   : \A1E__2020-04-12__02-55-56(4)__peromyscus_leucopus.JPG
        locations_that_were_similarity_checked = {}
        locations_that_were_lila_bc_retrieved = defaultdict(list)
        
        # dict with class counts per location 
            # {'addax-location-000': {'bear':4,'wolf':10},
            #  'addax-location-001': {'bear':12,'elk':20}}
        class_counts_per_location = defaultdict(lambda: defaultdict(int))

        # dict with absolute paths and classes per location 
            # "addax-location-00172": [
            #     [
            #     "C:\\Peter\\projects\\2024-28-OHI\\Sorted_by_species\\Invertebrate\\fcp1_apis_mellifera (2).JPG",
            #     "invertebrate"
            #     ]
            # ]
        paths_and_classes_per_location = defaultdict(list)
        
        # do the first image separately as it can't compare to the previous one
        curr_fname_rel = relpaths[0]
        curr_fname_abs = abspaths[0]
        curr_spp_class = spp_classes[0]
        curr_location_id = location_ids[0]

        # Check if location_id is already assigned in the dictionary
        # if there is an unknown location id, we don't want to use it except if it also contains "nz-trailcams_"
        # as the nz-trailcams dataset has a lot of unknowns for sublocations in their location IDs, hence still useful
        if curr_location_id and ("unknown" not in curr_location_id or "nz-trailcams_" in curr_location_id):

            location_id = curr_location_id
            
            # Add to the existing location in the dictionary
            paths_and_classes_per_location[location_id].append([curr_fname_abs, curr_spp_class])
            class_counts_per_location[location_id][curr_spp_class] += 1
            
            # Append the location to locations_that_were_lila_bc_retrieved
            locations_that_were_lila_bc_retrieved[location_id].append(curr_fname_rel)
        else:
            
            # if not already a location id, start a new location
            current_location = [curr_fname_rel]
            similarity_list = []
            location_idx += 1
            class_counts_per_location[f"addax-location-{location_idx:05d}"][curr_spp_class] = 1
            paths_and_classes_per_location[f"addax-location-{location_idx:05d}"] = [[curr_fname_abs, curr_spp_class]] 

        # loop
        for i in tqdm(range(1, len(relpaths)), desc="Processing locations"):
                
            # init vars 
            curr_fname_rel = relpaths[i]
            prev_fname_rel = relpaths[i - 1]
            curr_fname_abs = abspaths[i]
            curr_spp_class = spp_classes[i]
            curr_location_id = location_ids[i]
            
            # Check if location_id is already assigned in the dictionary
            # if there is an unknown location id, we don't want to use it except if it also contains "nz-trailcams_"
            # as the nz-trailcams dataset has a lot of unknowns for sublocations in their location IDs, hence still useful
            if curr_location_id and ("unknown" not in curr_location_id or "nz-trailcams_" in curr_location_id):
                location_id = curr_location_id
                
                # Add to the existing location in the dictionary
                paths_and_classes_per_location[location_id].append([curr_fname_abs, curr_spp_class])
                class_counts_per_location[location_id][curr_spp_class] += 1
                
                # Append the location to locations_that_were_lila_bc_retrieved
                locations_that_were_lila_bc_retrieved[location_id].append(curr_fname_rel)
                
                # Skip similarity check for this image
                continue  
            
            # here we compute the similarity scores. 
            similarity_score = difflib.SequenceMatcher(None, curr_fname_rel, prev_fname_rel).ratio() 
            
            # append similarity score to the list
            try:
                similarity_list.append(similarity_score)
            except Exception as e:
                log("\n\n\n")
                log(f"Error appending similarity score: {e}")
                log(f"curr_location_id: {curr_location_id}")
                log(f"curr_fname_rel: {curr_fname_rel}")
                log(f"prev_fname_rel: {prev_fname_rel}")
                log(f"curr_fname_abs: {curr_fname_abs}")
                log(f"curr_spp_class: {curr_spp_class}")
                log("\n\n\n")
                exit()
            
            # make sure that it is based on a rolling average window
            if len(similarity_list) > rolling_average_window:
                similarity_list.pop(0)
            average_similarity = np.mean(similarity_list)

            # check if there is a significant change in similarity
            if similarity_score < average_similarity * (1 - thresh):
                
                # significantly different, so start a new location
                locations_that_were_similarity_checked[f"addax-location-{location_idx:05d}"] = current_location # save current location
                current_location = [curr_fname_rel]     # start new location with 
                similarity_list = []                    # reset the similarity scores
                
                # increment the location index and start a new key in the dicts
                location_idx += 1
                class_counts_per_location[f"addax-location-{location_idx:05d}"][spp_classes[i]] = 1
                paths_and_classes_per_location[f"addax-location-{location_idx:05d}"] = [[curr_fname_abs, curr_spp_class]] 
                
            else:
                
                # same location, so add the information to the current location idx
                current_location.append(curr_fname_rel)
                class_counts_per_location[f"addax-location-{location_idx:05d}"][spp_classes[i]] += 1 
                paths_and_classes_per_location[f"addax-location-{location_idx:05d}"].append([curr_fname_abs, curr_spp_class]) 

        # save location if there is still information in the current location
        if len(current_location) > 0:
            locations_that_were_similarity_checked[f"addax-location-{location_idx:05d}"] = current_location

        # log
        log(f"\n\nFound {len(locations_that_were_lila_bc_retrieved.keys())} locations that were pre compiled with jsons.")
        log(f"\nFound {len(locations_that_were_similarity_checked.keys())} locations that were retrieved with similarity checking (thresh = {thresh}).")
        
        # Prompt user for location review
        if len(locations_that_were_similarity_checked.keys()) > 0:
            
            total_locations = len(locations_that_were_similarity_checked)
            total_images = sum(len(images) for images in locations_that_were_similarity_checked.values())
            average_images_per_location = total_images / total_locations if total_locations > 0 else 0

            log(f"Average number of images per location: {average_images_per_location:.2f}")
            log(f"\n\nFound {len(locations_that_were_similarity_checked.keys())} locations that were retrieved with similarity checking (thresh = {thresh}). Average number of images per location: {average_images_per_location:.2f}. Do you want to visually check?")
            show_locations = input("Press [Enter] to check, or any other key to accept it and continue with the script.").strip().lower() 
            if show_locations == "":
                for location_id, location_images in locations_that_were_similarity_checked.items():
                    log(f"{location_id} - ({len(location_images)} images):")
                    log(f"     first  : {location_images[0]}")
                    log(f"     middle : {location_images[len(location_images) // 2]}")
                    log(f"     last   : {location_images[-1]}")

                # Prompt for parameter adjustment
                log("\n\nCurrent parameters were\n\n"
                     f"   thresh : {thresh}\n\n"
                     "Do you want to change the parameters of this function and try again?")
                change_params = input("Press [Enter] to try again, or any other key to accept it and continue with the script.").strip().lower() 
                if change_params == "":
                    # rolling_average_window = int(input("\nEnter new value for rolling_average_window : "))
                    thresh = float(input("Enter new value for thresh                   : "))
                    log("\n\nRestarting with new parameter values...\n")
                    continue  # Restart the loop with new parameters
                else:
                    log("\n\nContinuing with current parameter values...")
                    break  # Exit the loop and return results
            else:
                log("\n\nNot displaying locations, proceeding...")
                break  # Exit loop and return results
        else:
            log("\n\nAll image locations were retrieved via LILA BC, so no need to do similarity parameter tweaking.")
            input(f"Press enter to proceed.")
            break

    # combine the dicts and return
    locations_dict = {**locations_that_were_similarity_checked, **locations_that_were_lila_bc_retrieved}

    # return all locations
    return paths_and_classes_per_location, class_counts_per_location, locations_dict, thresh

# get the location information
paths_and_classes_per_location_all, class_counts_per_location_all, locations_dict_all, selected_thresh = split_on_location(sorted_nested_list)

# Compute total detections across all locations
total_detections_all = sum(
    sum(counts.values()) for counts in class_counts_per_location_all.values()
)

# Find locations with all classes
full_class_locations = [
    loc for loc, counts in class_counts_per_location_all.items() if set(counts.keys()) == all_classes
]

# Sort locations by number of unique classes
sorted_by_class_count = sorted(
    class_counts_per_location_all.keys(),
    key=lambda loc: (len(class_counts_per_location_all[loc]), sum(class_counts_per_location_all[loc].values())),
    reverse=True
)

# use MD utils to split the locations into train and val
# not interactive anymore, as it will always return the best option of 10.000 trials.
def interactive_split_train_val(location_to_category_counts, split="val"):
    
    # Default parameter values
    fraction = 0.10 if split == "val" else 0.05
    
    # max_allowable_error = 0.10 if split == "val" else 0.05 # old code
    # max_allowable_error of 1.0, effectively allowing any error
    # this because often we deal with many classes of few images, and in order to 
    # split classes of 1 image, we need 100% error. The function returns the best 
    # option where weighted average absolute error across all categories is closest 
    # to the *target_val_fraction*. 
    max_allowable_error = 1.0
    
    # Loop to allow parameter changes
    while True:  

        # Call the function
        try:
            location_ids, faction_dict = split_locations_into_train_val(
                location_to_category_counts,
                target_val_fraction=fraction,
                default_max_allowable_error=max_allowable_error
            )
            
            # when using a max_allowable_error of 1.0, the function will return the best option where weighted average absolute error across all categories is closest to the *target_val_fraction*.
            # so no need to check the results, as it will be the best option available.
            log(f"\n\nSuccessfully split the {split} locations with the above settings.")
            break # Exit loop and return results
            
        except Exception as e:
            log(f"\n\nError: {e}")
            log("   Parameters to strict! Please adjust the parameters and try again.")
            pass
        
        # Prompt user for review
        log(f"\nYou just used the MD utils function to split the {split} locations. The output is listed above.")
        log("\nCurrent parameter values were\n\n"
             f"   fraction            : {fraction} (this is the goal percentage you want in your {split} set for each class)\n"
             f"   max_allowable_error : {max_allowable_error} (this is the allowed wiggle room)\n\n"
             "Do you want to change the parameters of this function and try again?")
        change_params = input("Press [Enter] to try again, or any other key to accept it and continue with the script.").strip().lower() 
        if change_params == "":
            fraction = float(input("Enter new value for fraction            : "))
            max_allowable_error = float(input("Enter new value for max_allowable_error : "))
            log("\nRestarting with new parameter values...\n")
            continue  # Restart loop with new values
        else:
            log("\nProceeding with current results...")
            break  # Exit loop and return results

    return location_ids, faction_dict, max_allowable_error, fraction

# remove the test locations from the class counts and use that as input for the val splitting
log("\n\n You're about to start the VAL location splitting process. \n\n")
locations_list_val, faction_dict_val_no_n, chosen_max_allowable_error_val, chosen_fraction_val = interactive_split_train_val(class_counts_per_location_all, split = "val")
class_counts_per_location_all

# now convert the fractions returned by MD to counts
total_counts = {}
for location, species_data in class_counts_per_location_all.items():
    for species, count in species_data.items():
        total_counts[species] = total_counts.get(species, 0) + count
calculated_val_counts = {}
for species, count in total_counts.items():
    frac = faction_dict_val_no_n.get(species, 0)
    calculated_val_counts[species] = round(count * frac)

# get the location information for validation set
class_counts_per_location_train_and_test = {loc: counts for loc, counts in class_counts_per_location_all.items() if loc not in locations_list_val}
log("\n\n You're about to start the TEST location splitting process. \n\n")
locations_list_test, faction_dict_test_no_n, chosen_max_allowable_error_test, chosen_fraction_test = interactive_split_train_val(class_counts_per_location_train_and_test, split = "test")

# now convert the fractions returned by MD to counts
train_test_counts = {}
for location, species_data in class_counts_per_location_all.items():
    for species, count in species_data.items():
        train_test_counts[species] = train_test_counts.get(species, 0) + count
calculated_test_counts = {}
for species, count in train_test_counts.items():
    frac = faction_dict_test_no_n.get(species, 0)
    calculated_test_counts[species] = round(count * frac)

# gather all counts in a single dict
split_count_dict = {}
for class_name in all_classes:
    val_n = calculated_val_counts.get(class_name, 0)
    test_n = calculated_test_counts.get(class_name, 0)
    train_n = total_counts[class_name] - val_n - test_n
    total_n = total_counts[class_name]
    split_count_dict[class_name] = {
        "val_n": val_n,
        "test_n": test_n,
        "train_n": train_n,
        "total_n": total_n
    }

# get a list of locations that are in the train set
locations_list_train = [loc for loc in class_counts_per_location_all.keys() if loc not in locations_list_test and loc not in locations_list_val]

# get the number of locations in each set
num_train_locations = len(locations_list_train)
num_val_locations = len(locations_list_val)
num_test_locations = len(locations_list_test)
num_total_locations = num_train_locations + num_val_locations + num_test_locations


# Prepare the data for the table
data = {}
column_totals = {"total": 0, "train": 0, "val": 0, "test": 0}
for class_name, counts in sorted(split_count_dict.items()):
    total = counts["total_n"]
    train = counts["train_n"]
    val = counts["val_n"]
    test = counts["test_n"]

    column_totals["total"] += total
    column_totals["train"] += train
    column_totals["val"] += val
    column_totals["test"] += test
    
    data[class_name.capitalize()] = [
        f"{total} ({100 * total / total:.1f}%)",
        f"{train} ({100 * train / total:.1f}%)",
        f"{val} ({100 * val / total:.1f}%)",
        f"{test} ({100 * test / total:.1f}%)"
    ]

# Add the last row with total counts and percentages
data["Total"] = [
    f"{column_totals['total']} ({100:.1f}%)",
    f"{column_totals['train']} ({100 * column_totals['train'] / column_totals['total']:.1f}%)",
    f"{column_totals['val']} ({100 * column_totals['val'] / column_totals['total']:.1f}%)",
    f"{column_totals['test']} ({100 * column_totals['test'] / column_totals['total']:.1f}%)"
]

# Create a DataFrame with first and last rows added
split_count_df = pd.DataFrame(data, index=["Total", "Train", "Val", "Test"]).T

# double check and prompt user to confirm
log("\n\n")
log(f"number of total detections : {sum(sum(species.values()) for species in class_counts_per_location_all.values())}")
log(f"number of total locations  : {len(locations_dict_all.keys())}")
log(f"number of train locations  : {len(locations_list_train)}")
log(f"number of val locations    : {len(locations_list_val)}")
log(f"number of test locations   : {len(locations_list_test)}")
log("\n\n")
log(split_count_df)

# Prompt user for confirmation
log("\n\nDo you want to proceed with the above split?")
confirm_split = input("Press [Enter] to proceed, or any other key to exit the script.").strip().lower() 
if not confirm_split == "":
    log("Exiting script....")
    exit()

# save variables to json
with open(os.path.join(dst_folder, 'variables.json'), 'w') as f:
    json.dump({
        "chosen_max_allowable_error_val": chosen_max_allowable_error_val,
        "chosen_fraction_val": chosen_fraction_val,
        "chosen_max_allowable_error_test": chosen_max_allowable_error_test,
        "chosen_fraction_test": chosen_fraction_test,
        "number of total detections" : sum(sum(species.values()) for species in class_counts_per_location_all.values()),
        "number of total locations": len(locations_dict_all.keys()),
        "number of train locations": len(locations_list_train),
        "number of val locations": len(locations_list_val),
        "number of test locations": len(locations_list_test),
        "locations_list_train": locations_list_train,
        "locations_list_val": locations_list_val,
        "locations_list_test": locations_list_test,
        "class_counts_per_location_all": class_counts_per_location_all,
        "class_counts_per_location_train_and_test": class_counts_per_location_train_and_test,
        "split_count_dict": split_count_dict
    }, f, indent=4)

# write location list to file
with open(os.path.join(dst_folder, 'locations_list_all_summary.txt'), 'w') as f:
    f.write("All locations summary:\n")     
    for location_id, location_images in locations_dict_all.items():
        f.write(f"{location_id}:\n")
        f.write(f"     first  : {location_images[0]}\n")
        f.write(f"     middle : {location_images[len(location_images) // 2]}\n")
        f.write(f"     last   : {location_images[-1]}\n")

# copy the excel file to the dst folder
shutil.copy2(spp_map_xlsx, os.path.join(dst_folder, "used_spp_plan.xlsx"))

# save locations_dict_all to json file
with open(os.path.join(dst_folder, 'locations_dict_all.json'), 'w') as f:
    json.dump(locations_dict_all, f, indent=4)

# save df to csv
split_count_df.to_csv(os.path.join(dst_folder, 'split_count_df.csv'))

# create a taxon csv file so the model knows the taxonomy during inference
# this includes all taxon information like class, order, family, genus, species,
# but can also include custom levels like "sex-age-group", "colouring", etc.
# and a few aggregated columns like "only_above_1000", "only_above_10000", etc.
# this is used to minimise the number of possible predcitions and group the 
# classes together if they fall under the threshold 
def create_taxon_csv(dst_dir):
    # read the excel file
    df = pd.read_excel(spp_map_xlsx, sheet_name='label_map')

    # check if there are any custom level columns added to the normal taxon columns
    custom_levels = [col for col in df.columns if col.startswith("Custom_level_")]
    if len(custom_levels) > 0:
        print(f"Found custom levels: {custom_levels}")

    # remove columns that are not needed
    columns_to_keep = ["GBIF_usageKey", "Class", "N_images", "GBIF_class", "GBIF_order", "GBIF_family", "GBIF_genus", "GBIF_species"] + custom_levels
    df = df[columns_to_keep]

    # define aggregation mapping
    aggregate_mapping = {
        "GBIF_usageKey": "first",
        "Class": "first",
        "N_images": "sum",
        "GBIF_class": "first",
        "GBIF_order": "first",
        "GBIF_family": "first",
        "GBIF_genus": "first",
        "GBIF_species": "first"
    }

    # add custom levels to the mapping
    for col in custom_levels:
        aggregate_mapping[col] = "first" 

    # aggregate
    df = df.groupby("Class", as_index=False).agg(aggregate_mapping)

    # define rename mapping
    rename_mapping = {
        "GBIF_usageKey": "GBIF_usageKey",
        "Class": "model_class",
        "N_images": "n_training_images",
        "GBIF_class": "level_class",
        "GBIF_order": "level_order",
        "GBIF_family": "level_family",
        "GBIF_genus": "level_genus",
        "GBIF_species": "level_species"
    }

    # add custom levels to the mapping
    for col in custom_levels:
        rename_mapping[col] = col.replace("Custom_level_", "level_")

    # rename columns
    df = df.rename(columns=rename_mapping)

    # get a list of all the level columns
    level_cols = [col for col in df.columns if col.startswith("level_")]

    # sort them by taxonomy
    df = df.sort_values(by=level_cols, ascending=False)

    # prefix the level columns with the level name ("Aves" -> "class Aves")
    for col in level_cols:
        prefix = col.replace("level_", "").replace("Custom_level_", "")
        df[col] = df[col].apply(lambda x: f"{prefix} {x}" if pd.notnull(x) else None)

    # fill in None values with the most specific level possible ("class Aves", None, None, None -> "class Aves", "class Aves", "class Aves", "class Aves")
    df[level_cols] = df[level_cols].apply(lambda row: row.ffill(axis=0), axis=1)

    # group model classes together if they fall under the trehsold
    # This function will add a new column to the DataFrame with the name "only_above_{threshold}"
    # and fill it with the class name if the number of training images is above the threshold
    # or the class name of the most specific level possible if the number of training images is below the threshold
    def add_only_above_column(threshold):

        # init the column
        aggregated_dict = {}
        df[f'only_above_{threshold}'] = None

        # iterate through the level columns in reverse order (most specific to least specific)
        for level_col in level_cols[::-1]:
            
            # group by the level column and aggregate the number of training images
            aggregated_temp_df = df.groupby(level_col, as_index=False).agg({"n_training_images": "sum"})
            
            # convert the DataFrame to a dictionary with level column as the key and n_training_images as the value
            aggregated_dict = {
                row[level_col]: row['n_training_images'] for _, row in aggregated_temp_df.iterrows()
            }
            
            # loop through each row in the original DataFrame
            for index, row in df.iterrows():

                # continue if the column already has a value
                if df.at[index, f'only_above_{threshold}'] is not None:
                    continue

                proposed_model_class = row[level_col]
                summed_training_images = aggregated_dict[proposed_model_class]
                if summed_training_images >= threshold:
                    df.at[index, f'only_above_{threshold}'] = proposed_model_class
        
        # if there are still None values in the column, set them to the level_class (highest level)
        for index, row in df.iterrows():
            if df.at[index, f'only_above_{threshold}'] is None:
                df.at[index, f'only_above_{threshold}'] = row['level_class']

    # add columns with aggregated classes based on several thresholds
    add_only_above_column(1000)
    add_only_above_column(10000)
    add_only_above_column(100000)

    # the true model classes should all be lowercase, as the pytorch pipeline down the road expoects that 
    df["model_class"] = df["model_class"].str.lower()

    # save as csv
    csv_file = os.path.join(dst_dir, "taxon-mapping.csv")
    df.to_csv(csv_file, index=False)

# function to prepare all the files and folders and copy images
def prepare_files_and_folders(add_test_to_train=False, subset=False):
    
    # create the dst folders
    dst_folder_local = os.path.join(dst_folder, "without_test" if add_test_to_train else "with_test")
    dst_folder_local = os.path.join(dst_folder_local, "subset" if subset else "full")
    Path(dst_folder_local).mkdir(parents=True, exist_ok=True)

    # create class_file.txt
    class_file = os.path.join(dst_folder_local, 'class_file.txt')
    if os.path.exists(class_file):
        os.remove(class_file)
    classes_alphabetical = sorted(all_classes)
    with open(class_file, 'w') as f:
        idx = 1
        f.write(f"id,class\n")
        for spp_class in classes_alphabetical:
            f.write(f"{idx},{spp_class}\n")
            idx += 1
    
    # place the split count information also in the dst folder
    split_count_df.to_csv(os.path.join(dst_folder_local, 'split_count_df.csv'))
    
    # also place the used_spp_plan.xlsx in the local dst folder
    shutil.copy2(spp_map_xlsx, os.path.join(dst_folder_local, "used_spp_plan.xlsx"))

    # save variables to json
    with open(os.path.join(dst_folder_local, 'location_counts.json'), 'w') as f:
        json.dump({
            "number of total locations": len(locations_dict_all.keys()),
            "number of train locations": len(locations_list_train),
            "number of val locations": len(locations_list_val),
            "number of test locations": len(locations_list_test)
        }, f, indent=4)

    # init animl csv files
    train_csv = os.path.join(dst_folder_local, 'train_data.csv')
    val_csv = os.path.join(dst_folder_local, 'val_data.csv')
    test_csv = os.path.join(dst_folder_local, 'test_data.csv')
    for csv_fpath in [train_csv, val_csv, test_csv]:
        if os.path.exists(csv_fpath):
            os.remove(csv_fpath)
        df = pd.DataFrame(columns=['FilePath', 'FileName', 'species'])
        df.to_csv(csv_fpath, index=False)
        
    # write taxon csv
    create_taxon_csv(dst_folder)
    create_taxon_csv(dst_folder_local)
        
    # loop through the locations
    log("\n\n")
    image_idx = 1
    location_idx = 0
    
    # pbar = tqdm(total=total_images) # DEBUG the tqdm total is wrong, because it doesnt take into account the capped off images. does this work now?
    pbar = tqdm(total=n_images_with_cutoff)
    
    for location_id, detection_list in paths_and_classes_per_location_all.items():
        pbar.set_description(location_id)
        location_idx += 1
        
        # here we add the test split to train if specified 
        if add_test_to_train:
            typ = "train" if location_id in locations_list_train else "val" if location_id in locations_list_val else "train"
        else:
            typ = "train" if location_id in locations_list_train else "val" if location_id in locations_list_val else "test"
        for idx, detection in enumerate(detection_list):
            
            # only take 20% of the data for hyper parameter tuning
            if subset:
                if idx % 5 != 0:  
                    pbar.update(1)
                    continue  

            # loop through images
            img, class_name = detection[0], detection[1]
            
            # open typ-specific csv
            csv_fpath = os.path.join(dst_folder_local, f"{typ}_data.csv")
            with open(csv_fpath, 'a', newline='') as f:
                writer = csv.writer(f)

                # paths
                src_file = pathlib.PurePath(img)
                src_file_ext = os.path.splitext(src_file)[1]
                unique_fname = f"{location_idx:06}-{image_idx:06}" + src_file_ext.lower() 
                dst_file = os.path.join(dst_folder_local, typ, class_name, unique_fname)
                src_file = os.path.normpath(src_file)
                dst_file = os.path.normpath(dst_file)
                
                # copy files
                Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                image_idx += 1
                
                # write to typ-specific csv
                writer.writerow([dst_file, unique_fname, class_name])
                
                pbar.update(1)

# create the files and folders with the test set
log("\n\nCopying files and creating folders with test set...")
prepare_files_and_folders(add_test_to_train=False, subset=False)

# create the files and folders without the test set
if duplicate_without_test:
    log("\n\nCreating duplicate dataset without test set...")
    prepare_files_and_folders(add_test_to_train=True, subset=False)

# create the files and folders without the test set
if duplicate_subset:
    log("\n\nCreating duplicate dataset subset...")
    prepare_files_and_folders(add_test_to_train=True, subset=True)