# Script to copy files from the main database to a training folder for training a yolov8 classifier



############################################################################
# Er zit ergens een klein foutje in volgens mij
# voor class zebra en class african wild cat zijn meer dan 3500 test images gepakt...
# even kijken wat daar aan de hand is

# UPDATE: Dit heeft volgens mij te maken met het feit dat je de test set van classes 
# die net onder de test threshold vallen, niet downsampled. Lijkt me niet een enrom 
# probleem, maar - REPREX: pak class cat van NZF project met 
# prop_to_test = 0.1 en n_samples_train = 150000. 
############################################################################







# This script is different from 2B-create-train-test-val-based-on-seqs.py in the way that it will up
# or downsample the images in order to get a fixed amount of images in the training set (*n_samples_train*)
# It will always first fill test and train with local images, and only if needed supplement with non-local images
# The *n_samples_train* corresponds to the N_SAMPLES parameter of Mega Efficient Wildlife Classifiers. 

# """
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\upsample-based-on-seq-dirs-only.py"
# """

# Peter van Lunteren, 29 nov 2023

# USER INPUT
prop_to_test = 0.2                  # the total number of images that will be used for testing
total_n_imgs_per_class = 10000      # the total number of images you want to use for each class
                                    # the train and test will be created from this number
                                    # the trainingset will be up- or downsampled if required
                                    # while the test set will only be downsampled if too large
                                    # as it doesnt really make sense to upsample the testset

# it imports classes and paths from xlsx file. It should have at least three columns: 'Class', 'Path', and 'Source'.
spp_map_xlsx = r"C:\Users\smart\Desktop\2024-19-HWI-spp-plan.xlsx"

# With this script you can create a dataset to train a classifier. It will not spearate sequences, 
# so that the entire sequence will either be in test, train, or val. It will copy the original images
# and place them in a single dir per class per train, test, val. Hence it will give the images a unique name. 
# The resulting folder structure will be like this:

# dataset
#    |- train
#    |     |- class A
#    |     |     |- img-hash-1.jpg
#    |     |     |- img-hash-2.jpg
#    |     |      \ img-hash-3.jpg
#    |     |
#    |      \ class B
#    |           |- img-hash-1.jpg
#    |           |- img-hash-2.jpg
#    |            \ img-hash-3.jpg
#    |- test
#    |     |- class A
#    |     |     |- img-hash-4.jpg
#    |     |     |- img-hash-5.jpg
#    |     |      \ img-hash-6.jpg
#    |     |
#    |      \ class B
#    |           |- img-hash-4.jpg
#    |           |- img-hash-5.jpg
#    |            \ img-hash-6.jpg
#     \ val
#          |- class A
#          |     |- img-hash-7.jpg
#          |     |- img-hash-8.jpg
#          |      \ img-hash-9.jpg
#          |
#           \ class B
#                |- img-hash-7.jpg
#                |- img-hash-8.jpg
#                 \ img-hash-9.jpg

# packages
import os
from tqdm import tqdm
import random
import json
from pathlib import Path
import shutil
import pathlib
import pandas as pd
import time
from tabulate import tabulate         # pip install tabulate

# set destination
dst_folder = r"F:\datasets\2024-19-HWI-10K-seqdirs"

# create a nested dictionary with classes, sources and folderspaths, either import from xlsx file
rows = pd.read_excel(spp_map_xlsx)[['Class', 'Source', 'Path']]
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

# this function takes a list, converts it to df, samples to N (can be up or down), and converts it back to list
def up_or_downsample(list, n):
    return pd.DataFrame(list).sample(n=n, replace = len(list) < n)[0].tolist()

# this function takes an image list (local or non-local) and puts it partly in test and partly in train, depending on the set *prop_to_test*
def slice_list_to_test_and_train(img_list, frac):
    cutoff = int(frac * len(img_list))
    test_imgs = img_list[:cutoff] # fill test up until cutoff
    train_imgs = img_list[cutoff:] # the rest will go into train
    return [test_imgs, train_imgs]

# export train_input to keep track of training set later on
def export_paths_to_json():
    with open(f"C:\\Users\\smart\\Desktop\\src-paths-{time.strftime('%H%M%S')}.json", 'w') as fp:
        formatted_train_input = json.dumps(train_input, indent = 2)
        print(formatted_train_input, file=fp)
    df = pd.DataFrame.from_dict(img_counts, orient='index', columns = ['n'])
    df.to_excel(f"C:\\Users\\smart\\Desktop\\image-counts-{time.strftime('%H%M%S')}.xlsx")

# exit script if the training dir is not empty
if len(os.listdir(dst_folder)) != 0: 
    print("WARNING: Training directory is not empty. Are you sure you want to conintue?") 
    exit()

# there always needs to be some test data
if prop_to_test == 0:
    print("WARNING: The proportion of test data is set to 0. There should always be some test data.") 
    exit() 

# function to check the number of images in a sequence folder
def check_n_imgs_in_seq(seq_path):
    n_imgs = len([os.path.join(seq_path, f) for f in os.listdir(seq_path) if f != ".DS_Store" and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
    return n_imgs

# function to return the paths of images in a sequence folder
def get_img_paths_from_seq(seq_path):
    img_paths = [os.path.join(seq_path, f) for f in os.listdir(seq_path) if f != ".DS_Store" and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    return img_paths

# create list of sequences and the total number of images per class 
class_dict = {}
session_img_total = 0
img_counts = {}
table = []
for key, value in train_input.items():

    # get local sequences
    src_dirs_local = value['local'] if 'local' in value else []
    seq_paths_local = []
    for src_dir in src_dirs_local:
        print(f"Scanning '{src_dir}'")
        if not os.path.exists(src_dir):
            print(f"ERROR: Source directory '{src_dir}' does not exist. Check the path in the xlsx file.")
            exit()
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                seq_paths_local.append(subdir)
                if src_dir in img_counts:
                    img_counts[src_dir] += check_n_imgs_in_seq(subdir)
                else:
                    img_counts[src_dir] = check_n_imgs_in_seq(subdir)
    random.shuffle(seq_paths_local)

    # get non-local sequences
    src_dirs_non_local = value['non-local'] if 'non-local' in value else []
    seq_paths_non_local = []
    for src_dir in src_dirs_non_local:
        print(f"Scanning '{src_dir}'")
        if not os.path.exists(src_dir):
            print(f"ERROR: Source directory '{src_dir}' does not exist. Check the path in the xlsx file.")
            exit()
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                seq_paths_non_local.append(subdir)
                if src_dir in img_counts:
                    img_counts[src_dir] += check_n_imgs_in_seq(subdir)
                else:
                    img_counts[src_dir] = check_n_imgs_in_seq(subdir)
    random.shuffle(seq_paths_non_local)

    # now that the sequences are shuffled, we can unpack the images and keep them in this order
    local_imgs = []
    non_local_imgs = []
    for seq in seq_paths_local:
        local_imgs.extend(get_img_paths_from_seq(seq))
    for seq in seq_paths_non_local:
        non_local_imgs.extend(get_img_paths_from_seq(seq))

    # define values based on image availablilty
    n_samples_train = int(total_n_imgs_per_class * (1 - prop_to_test))
    prop_to_train = 1 - prop_to_test
    n_samples_test = int((n_samples_train * prop_to_test) / prop_to_train) # calculate based on proportions - set max in case there are many images
    img_tresh = n_samples_train + n_samples_test
    n_tot_imgs = len(local_imgs) + len(non_local_imgs)
    local_ratio = len(local_imgs) / n_tot_imgs
    non_local_ratio = 1 - local_ratio

    # now comes the tricky part, how to slice and sample the test and train images
    # LESS THEN N_SAMPLES > UP SAMPLING
    # if there aren't enough images to reach the sample size, we will need to upsample anyway and use all images that we have, local and non-local.
    # that makes it easy, because we will just cut the local and non-local images into two pieces, for test and train, based on the prop_to_test.
    if n_tot_imgs < img_tresh:
        # slice images
        test_local, train_local = slice_list_to_test_and_train(local_imgs, prop_to_test)
        test_non_local, train_non_local = slice_list_to_test_and_train(non_local_imgs, prop_to_test)

        # add them together
        test_imgs = test_local + test_non_local
        train_imgs = train_local + train_non_local

        # upsample train, no need to upsample test
        train_imgs = up_or_downsample(train_imgs, n_samples_train)

    # MORE THAN N_SAMPLES > DOWN SAMPLING
    # if there are more images than the sample size, we will need to downsample. In that case we will give priority to the local images.
    # that means that the amount of local images is important.
    else:
        # let's see if we can fill everything with only local images
        if len(local_imgs) >= img_tresh:
            # if so, no need to bother with non-local images. We can fill our dataset with local images entirely
            test_imgs, train_imgs = slice_list_to_test_and_train(local_imgs, prop_to_test)

            # we'll have too much in train and test, so we'll have to downsample both 
            test_imgs = up_or_downsample(test_imgs, n_samples_test)
            train_imgs = up_or_downsample(train_imgs, n_samples_train)

        # let's see if there are no local images at all
        elif len(local_imgs) == 0:
            # if there are no local images at all, it is easy because we just fill test and train with non-local images entirely
            test_imgs, train_imgs = slice_list_to_test_and_train(non_local_imgs, prop_to_test)

            # we'll have to much in train and test, so we'll have to downsample both
            test_imgs = up_or_downsample(test_imgs, n_samples_test)
            train_imgs = up_or_downsample(train_imgs, n_samples_train)

        # the last option is a combination of local and non-local images, as we don't have enough local images to fill train and test entirely,
        # so we need to supplement with non-local images
        else:
            # slice all images
            test_local, train_local = slice_list_to_test_and_train(local_imgs, prop_to_test)
            test_non_local, train_non_local = slice_list_to_test_and_train(non_local_imgs, prop_to_test)

            # check how many non-local images we need to supplement 
            required_train_supplement = n_samples_train - len(train_local)
            required_test_supplement = n_samples_test - len(test_local)

            # downsample to the required number of supplements
            train_non_local = up_or_downsample(train_non_local, required_train_supplement)
            test_non_local = up_or_downsample(test_non_local, required_test_supplement)

            # add them together, then you'll have all the local images, and a subsection of the non-local images
            test_imgs = test_local + test_non_local
            train_imgs = train_local + train_non_local

    # store the paths in a dictionary
    class_dict[key] = {"test" : test_imgs,
                       "train" : train_imgs}
    
    # keep track of total number of images over all classes
    session_img_total += len(test_imgs) + len(train_imgs)
    table.append([key, len(test_imgs), len(train_imgs)])

# print overview of counts
print(tabulate(table, headers=["class", "n_test", "n_train"], tablefmt="grid"))

# copy files
export_paths_to_json()
pbar = tqdm(total=session_img_total)
for class_name, values in class_dict.items():
    pbar.set_description(f"{class_name.ljust(20)}")
    image_idx = 1
    for typ in ['test', 'train']:
        for img in values[typ]:
            src_file = pathlib.PurePath(img)
            src_file_ext = os.path.splitext(src_file)[1]
            unique_fname = f"{class_name.replace(' ', '')}-{typ}-{image_idx:06}".lower() + src_file_ext.lower()
            dst_file = os.path.join(dst_folder, typ, class_name, unique_fname)
            src_file = os.path.normpath(src_file)
            dst_file = os.path.normpath(dst_file)
            Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            image_idx += 1
            pbar.update(1)