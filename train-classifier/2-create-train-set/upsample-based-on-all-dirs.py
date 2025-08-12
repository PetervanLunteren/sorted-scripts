# Script to copy files from the main database to a training folder train/classA, train/classB, test/classA, test/classB, etc.

# You can choose to balance the dataset or not. If you choose to balance the dataset, it will up or downsample the classes.
# If not balancing, it will just fill the datasets with all available images. The training phase will then need to take care of class weights.

# It will up or downsample the images in order to get a fixed amount of images in the training set (*total_n_imgs_per_class*)
# It will always first fill test and train with local images, and only if needed supplement with non-local images

# it is different from the method upsample-based-on-seq-dirs-only.py in the way that it sorts the images by dir all layers,
# instead of just by the sequence dir. This way it keeps the images from different folders from mixing up. It therefore keeps 
# track of all the dir-stored information like dataset, project, season, etc.

# """
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\upsample-based-on-all-dirs.py"
# """

# Peter van Lunteren, 18 nov 2024

# USER INPUT
prop_to_val = 0.2                                                       # (required) the proportion of the total number of images that will be used for validation
prop_to_test = 0                                                        # (optional) the proportion of the total number of images that will be used for testing. Will consist exclusively of local images. Classes without local images will not be tested.
spp_map_xlsx = r"C:\Users\smart\Desktop\2024-16-SCO-spp-plan.xlsx"      # it imports classes and paths from xlsx file. It should have at least three columns: 'Class', 'Path', and 'Source'.
dst_folder = r"F:\datasets\2024-16-SCO-unbalanced-non-padded"           # the destination folder where the training and test data will be copied to (can be a external SSD too)
balance_dataset = False                                                 # if True, it will up or down/up-sample the classes to balance the dataset. It will keep track of local and non-lcoal images, where it will first fill up with local images, and only if needed supplement with non-local images.
                                                                        # if False, it will fill the datasets all available images, and will not balance the dataset. It will of course keep track of local and non-local images. When training on this dataset, it will be important to use class weights or something similar to balance the classes.

# settings for balance_dataset = False
max_n_imgs_per_class = 9999999                                          # If not balancing the dataset, this is the maximum number of images you want to use for each class. If there are more images available, it will downsample, giving the local images priority. If there are less images available, it will take all available images.

# settings for balance_dataset = True
total_n_imgs_per_class = 100000                                         # If balancing the dataset, this is the total number of images you want to use for each class. Train, test, and val will be subsets of this number. If not balancing, the number will be ignored.



# packages
import os
from tqdm import tqdm
import random
from pathlib import Path
import shutil
import pathlib
import csv
import openpyxl                  # pip install openpyxl
import pandas as pd
from tabulate import tabulate         # pip install tabulate

# Set seed for reproducibility
random.seed(420)

# create dst folder if needed
Path(dst_folder).mkdir(parents=True, exist_ok=True)

# create a nested dictionary with classes, sources and folderspaths, either import from xlsx file
rows = pd.read_excel(spp_map_xlsx, sheet_name='label_map')[['Class', 'Source', 'Path']]
rows = rows.reset_index()
train_input = {}
for index, row in rows.iterrows():
    spp_class = row['Class'].strip().lower()
    path = row['Path'].replace('"', '') 
    source = row['Source'].strip().lower()
    if source != 'local' and source != 'non-local':
        print(f"ERROR READING LABEL MAP: Source for row index {index} is invalid: '{source}'")
        exit()
    if spp_class in train_input:
        old_source_dict = train_input[spp_class]
        if source in old_source_dict:
            train_input[spp_class][source].append(path)
        else:
            train_input[spp_class][source] = [path]
    else:
        train_input[spp_class] = {}
        train_input[spp_class][source] = [path]

# function to up or downsample a list to a certain number of elements
# upsampling: it repeats the list until it reaches the required number of images, hereby making sure all images are included evenly
# downsampling: it takes the first n elements of the list
def up_or_downsample(lst, n):
    random.shuffle(lst)
    if len(lst) == 0:
        return []
    else:
        return (lst * (n // len(lst) + 1))[:n]

# this function takes an image list (local or non-local) and puts it partly in test, partly in val, and partly in train, depending on the set *prop_to_test*, *prop_to_val*
def slice_list_to_test_val_train(img_list, prop_to_test, prop_to_val):
    n_test = int(prop_to_test * len(img_list))
    n_val = int(prop_to_val * len(img_list))
    test_imgs = img_list[:n_test]                 # fill test up to its cutoff
    val_imgs = img_list[n_test:n_test+n_val]      # fill val up to its cutoff from the last cutoff
    train_imgs = img_list[n_test+n_val:]          # the rest will go into train
    return [test_imgs, val_imgs, train_imgs]


# exit script if the training dir is not empty
if len(os.listdir(dst_folder)) != 0: 
    print("WARNING: Training directory is not empty. Are you sure you want to continue?") 
    exit()

# there always needs to be some test data
if prop_to_val == 0:
    print("WARNING: The proportion of validation data is set to 0. There should always be some test data.") 
    exit() 

# This function preserves the hierarchy of parent and child folders while sorting all file paths. It ensures that:
# Files and subfolders within the same parent folder are sorted together.
# Files and folders from different parent folders are kept separate and sorted relative to their full path.
def sort_paths_per_dir_layer(file_paths):
    # Split each path into components
    split_paths = [os.path.normpath(path).split(os.sep) for path in file_paths]
    
    # Sort paths alphabetically by each directory layer
    sorted_paths = sorted(split_paths, key=lambda x: tuple(x))
    
    # Reconstruct the file paths
    sorted_file_paths = []
    for path in sorted_paths:
        drive, tail = os.path.splitdrive(os.path.join(*path))
        if drive:
            sorted_file_paths.append(drive + os.sep + tail)
        else:
            sorted_file_paths.append(tail)
            
    # return
    return sorted_file_paths

# return a list of all image paths in a directory
def get_img_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_paths.append(os.path.join(root, file))
    return file_paths

# create list of sequences and the total number of images per class 
class_dict = {}
session_img_total = 0
img_counts = {}
table = []
for key, value in train_input.items():

    # print new line for readability
    print("\nCalculating for class", key)

    # get local imgs
    src_dirs_local = value['local'] if 'local' in value else []
    local_imgs = []
    for src_dir in src_dirs_local:
        print(f"Scanning '{src_dir}'")
        if not os.path.exists(src_dir):
            print(f"ERROR: Source directory '{src_dir}' does not exist. Check the path in the xlsx file.")
            exit()
        
        local_imgs.extend(get_img_paths(src_dir))
    
    # get non-local imgs
    src_dirs_non_local = value['non-local'] if 'non-local' in value else []
    non_local_imgs = []
    for src_dir in src_dirs_non_local:
        print(f"Scanning '{src_dir}'")
        if not os.path.exists(src_dir):
            print(f"ERROR: Source directory '{src_dir}' does not exist. Check the path in the xlsx file.")
            exit()
        non_local_imgs.extend(get_img_paths(src_dir))

    # now that we know which images are available, we can decide how to fill the test and train sets
    
    # if the dataset doesn't need to be balanced, we can just use all available images - except when the total number of images is more than *max_n_imgs_per_class*
    if not balance_dataset:
        
        # is the number of local images more than *max_n_imgs_per_class*?
        if len(local_imgs) > max_n_imgs_per_class:
            print(f"For class {key}, there are {len(local_imgs)} local images available, which is more than the maximum number of images allowed ({max_n_imgs_per_class})")
            print(f"   ... so we'll downsample to {max_n_imgs_per_class} images.")
            selected_local_imgs = up_or_downsample(local_imgs, max_n_imgs_per_class) # downsample to max_n_imgs_per_class
            selected_non_local_imgs = [] # no non-local images needed
            
        # if not: is the total number of images more than *max_n_imgs_per_class*?
        elif len(local_imgs) + len(non_local_imgs) > max_n_imgs_per_class:
            
            # yes, we have more than the maximum number of images allowed, we need to downsample
            print(f"For class {key}, there are {len(local_imgs) + len(non_local_imgs)} images available, which is more than the maximum number of images allowed ({max_n_imgs_per_class})")
            print(f"   ... so we'll downsample to {max_n_imgs_per_class} images.")
               
            # if so, take all the local images...
            selected_local_imgs = local_imgs
            print(f"   ... we have {len(local_imgs)} local images, so we'll take all of them.")
            
            # ... and downsample the non-local images for the remaining number of images
            selected_non_local_imgs = up_or_downsample(non_local_imgs, max_n_imgs_per_class - len(local_imgs))
            print(f"   ... we have {len(non_local_imgs)} non-local images, so we'll downsample to {max_n_imgs_per_class - len(local_imgs)} non-local images.")

        else:
            
            # no, we have less than the maximum number of images allowed. Thats easy. Just take all available images.
            print(f"For class {key}, there are {len(local_imgs) + len(non_local_imgs)} images available, which is less than the maximum number of images allowed ({max_n_imgs_per_class})")
            selected_local_imgs = local_imgs
            selected_non_local_imgs = non_local_imgs

    # if the dataset needs to be balanced, we need to decide how to fill the test and train sets and how to up and downsample the images
    else:
        
        # define values based on image availablilty
        req_val_size = int(total_n_imgs_per_class * prop_to_val)
        req_train_size = total_n_imgs_per_class - req_val_size
        ava_total_n_imgs = len(local_imgs) + len(non_local_imgs)

        print(f"For class {key}, there are {ava_total_n_imgs} images available ...")
        print(f"   ... of which {len(local_imgs)} are local ...")
        print(f"   ... and {len(non_local_imgs)} are non-local.")
        print(f"We need to create a dataset of {total_n_imgs_per_class} images, so ...")

        # now we need to decide how to fill the test and train sets... there are two situations...
        # 1. we can fill everything with local images only
        if len(local_imgs) >= total_n_imgs_per_class:
            print("   ... we can fill everything with local images.")
            
            # in that case we'll just sample the local images to required number of images ...
            selected_local_imgs = up_or_downsample(local_imgs, total_n_imgs_per_class)
            
            # TODO: it feels weird to not use the non_local images at all. In this case, it would be good to add a subsample of non-local images still, and downsample the local images a bit more....
            
            # ... and we don't need any non-local images
            selected_non_local_imgs = []

        # 2. we can't fill everything with local images only, so we need to supplement with non-local images
        else: 
            # how many do we need to supplement?
            required_non_local_supplement = total_n_imgs_per_class - len(local_imgs)
            print(f"   ... we need to supplement {total_n_imgs_per_class} - {len(local_imgs)} = {required_non_local_supplement} non-local images to reach the required number ...")
            print(f"   ... and we have {len(non_local_imgs)} non-local images available.")
        
            # now there are again two situations
            # 2A. we have enough non-local images to fill to the required number
            if len(non_local_imgs) >= required_non_local_supplement:
                print("   ... which means we have enough non-local images to fill to the required number ...")
                print("   ... and we'll use all local images and a sample the non-local images.")
                
                # in that case we'll just take all local images...
                selected_local_imgs = local_imgs
                
                # ... and sample the rest of the non-local images
                selected_non_local_imgs = up_or_downsample(non_local_imgs, required_non_local_supplement)
                
            # 2B. we don't have enough non-local images to fill to the required number
            # nine times out of ten this will be the case
            else:
                print("   ... which means we don't have enough non-local images to fill to the required number ...")
                print("   ... and we'll upsample both the local and the non-local images.")
                
                # in this case, we need to know the distribution of local/non-local images in the available images
                ratio_ava_local_imgs = len(local_imgs) / ava_total_n_imgs
                ratio_ava_non_local_imgs = 1 - ratio_ava_local_imgs
        
                # now we need to know how many local and non-local images we need to reach the required number while keeping the original distribution
                required_local_imgs = int(total_n_imgs_per_class * ratio_ava_local_imgs)
                required_non_local_imgs = total_n_imgs_per_class - required_local_imgs
                
                # now we know which number to upsample to, to keep the original distribution
                selected_local_imgs = up_or_downsample(local_imgs, required_local_imgs)
                selected_non_local_imgs = up_or_downsample(non_local_imgs, required_non_local_imgs)

    # now we're going to sort them by dir layer
    # this is the crucial difference with sorting only on sequence dir, because now
    # you also use the dir-stored information like dataset, project, season, etc.
    
    # first we'll sort the images by dir layer
    sorted_local_imgs = sort_paths_per_dir_layer(selected_local_imgs)
    sorted_non_local_imgs = sort_paths_per_dir_layer(selected_non_local_imgs)
    
    # Tag images with their origin
    sorted_local_imgs = [(img, "local") for img in sorted_local_imgs]
    sorted_non_local_imgs = [(img, "non-local") for img in sorted_non_local_imgs]
    
    # then we'll slice them into test and train in the sorted order
    # this way we can keep the images from different folders from mixing up
    test_local, val_local, train_local = slice_list_to_test_val_train(sorted_local_imgs, prop_to_test, prop_to_val) # if val, it will be taken exclusevly from the local images
    _, val_non_local, train_non_local = slice_list_to_test_val_train(sorted_non_local_imgs, 0, prop_to_val)         # prop_to_val is always 0 for non-local images
    
    # now we need to add the local and non-local images together
    test_imgs = test_local
    val_imgs = val_local + val_non_local
    train_imgs = train_local + train_non_local

    # shuffle the train once more, so that the batches consist of different scenarios
    # no need to shuffle the test, because it's only used for evaluation
    random.shuffle(train_imgs)

    # it can happen that the test images are upsampled so there are duplicates in the test set, we'll remove any duplicates here
    seen = set()
    test_imgs = [img for img in test_imgs if img not in seen and not seen.add(img)]
    
    # smooth out rounding errors
    if balance_dataset:
        print(f"validation set size before smoothing: {len(val_imgs)}")
        print(f"training set size before smoothing:   {len(train_imgs)}")
        val_imgs = up_or_downsample(val_imgs, req_val_size)
        train_imgs = up_or_downsample(train_imgs, req_train_size)
        print(f"validation set size after smoothing:  {len(val_imgs)}")
        print(f"training set size after smoothing:    {len(train_imgs)}")

    # store the paths in a dictionary
    class_dict[key] = {"test" : test_imgs,
                       "val" : val_imgs,
                       "train" : train_imgs}
    
    # keep track of total number of images over all classes
    session_img_total += len(test_imgs) + len(val_imgs) + len(train_imgs)
    table.append([key, len(test_imgs), len(val_imgs), len(train_imgs)])

# print overview of counts
print(tabulate(table, headers=["class", "n_test", "n_val", "n_train"], tablefmt="grid"))

# create class_file.txt
class_file = os.path.join(dst_folder, 'class_file.txt')
if os.path.exists(class_file):
    os.remove(class_file)
classes_alphabetical = sorted(class_dict.keys())
with open(class_file, 'w') as f:
    idx = 1
    f.write(f"id,class\n")
    for spp_class in classes_alphabetical:
        f.write(f"{idx},{spp_class}\n")
        idx += 1

# init animl csv files
train_csv = os.path.join(dst_folder, 'train_data.csv')
val_csv = os.path.join(dst_folder, 'val_data.csv')
test_csv = os.path.join(dst_folder, 'test_data.csv')
for csv_fpath in [train_csv, val_csv, test_csv]:
    if os.path.exists(csv_fpath):
        os.remove(csv_fpath)
    df = pd.DataFrame(columns=['FilePath', 'FileName', 'species'])
    df.to_csv(csv_fpath, index=False)

# open the global csv for all src and dst
image_idx = 1
all_csv_fpath = os.path.join(dst_folder, "src_dst.csv")
with open(all_csv_fpath, 'a', newline='') as all_csv:
    all_writer = csv.writer(all_csv)
    all_writer.writerow(["class_name", "typ", "origin", "unique_fname", "dst_file", "src_file"])
    
    pbar = tqdm(total=session_img_total)
    for class_name, values in class_dict.items():
        pbar.set_description(class_name)
        for typ in ['test', 'val', 'train']:
            
            # open typ-specific csv
            csv_fpath = os.path.join(dst_folder, f"{typ}_data.csv")
            with open(csv_fpath, 'a', newline='') as f:
                writer = csv.writer(f)
                
                # loop through images
                for (img, origin) in values[typ]:
                    
                    # paths
                    src_file = pathlib.PurePath(img)
                    src_file_ext = os.path.splitext(src_file)[1]
                    unique_fname = f"{origin}{-image_idx:09}" + src_file_ext.lower() 
                    dst_file = os.path.join(dst_folder, typ, class_name, unique_fname)
                    src_file = os.path.normpath(src_file)
                    dst_file = os.path.normpath(dst_file)
                    
                    # copy files
                    Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    image_idx += 1
                    
                    # write to typ-specific csv
                    writer.writerow([dst_file, unique_fname, class_name])
                    
                    # write to global csv
                    all_writer.writerow([class_name, typ, origin, unique_fname, dst_file, src_file])
                    
                    pbar.update(1)