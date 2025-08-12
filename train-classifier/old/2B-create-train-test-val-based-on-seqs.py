# Script to copy files from the main database to a training folder for training a yolov8 classifier

# This script will create the folderstructure nessesary, and keeps track of sequences and whether 
# the image was taken from the local project, or from other projects (e.g., lila bc). It will make
# sure that sequences will stay together, and will not end up in different folders. Furthermore,
# it will make sure that half of the local images will be added to val, the remaining to test 
# and the remaining to train. Then, the second half of the local images will be added at the 
# end to train. The actual number of local images ending up in either dir is of course subject
# to availability. For example, if there are 1000 local images, 1000 non local images, 0.1 
# prop_to_val (so 200 images), and 0.2 prop_to_test (so 400 images), half of the local images 
# (500) will be added to val (200), the remaining (300) to test. So val will consist of 100% 
# local images, and test will consist of 75% (300) local and 25% (100) non-local images. Train 
# will then consist of the other half of local images, which is 500 local images, and the 
# remaining 900 non local images. So, if there are proportionally many local images, val and 
# test will consist entirely of local images.

"""
conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\2B-create-train-test-val-based-on-seqs.py"
"""

# Peter van Lunteren, 29 nov 2023

# USER INPUT
prop_to_test = 0.2      # the total number of images that will be used for testing
prop_to_val  = 0.0      # the total number of images that will be used for validation

# you can also import classes and paths from xlsx file. It should have at least two columns: 'Class', 'Path', and 'Source'.
# if False, set manually below
import_paths_from_spp_map = True
spp_map_xlsx = r"C:\Users\smart\Desktop\2024-24-TAN-spp-plan.xlsx"

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
import uuid
from pathlib import Path
import shutil
import pathlib
import pandas as pd
import time
from tabulate import tabulate # pip install tabulate

# set destination
dst_folder = r'C:\Peter\training-utils\current-train-set'

# create a nested dictionary with classes, sources and folderspaths, either import from xlsx file, or manually
# option 1: import from spp_map_xlsx and convert to dict
if import_paths_from_spp_map:
    rows = pd.read_excel(spp_map_xlsx)[['Class', 'Source', 'Path']]
    rows = rows.reset_index()
    train_input = {}
    for index, row in rows.iterrows():
        spp_class = row['Class'].strip().lower()
        path = row['Path'].replace('"', '').replace("'", '')
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

# option 2: manually define dictionary for which data belongs to which class
# Format is -> {class_name : {"local" : ["path_1", "path_2"], "non-local" : ["path_3"]}}
else:
    train_input = {
        "rhinoceros": {
            "local": [
                "C:\\Peter\\training-utils\\verified-datasets\\diceros bicornis\\desert-lion-conservation-project"
            ],
            "non-local": [
                "C:\\Peter\\training-utils\\verified-datasets\\ceratotherium simum\\lila-bc",
                "C:\\Peter\\training-utils\\verified-datasets\\diceros bicornis\\lila-bc",
                "C:\\Peter\\training-utils\\verified-datasets\\rhinocerotidae\\lila-bc"
            ]
        },
        "caracal": {
            "non-local": [
                "C:\\Peter\\training-utils\\verified-datasets\\caracal caracal\\lila-bc"
            ]
        },
        "dummy": {
            "local": [
                "C:\\Peter\\training-utils\\verified-datasets\\cn-francolins\\desert-lion-conservation-project"
            ]
        },
        "klipspringer": {
            "local": [
                "C:\\Peter\\training-utils\\verified-datasets\\oreotragus oreotragus\\desert-lion-conservation-project"
            ],
            "non-local": [
                "C:\\Peter\\training-utils\\verified-datasets\\oreotragus oreotragus\\lila-bc"
            ]
    }}

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
prop_to_train = round(1 - prop_to_test - prop_to_val, 3)
session_img_total = 0
img_counts = {}
for key, value in train_input.items():

    # get local sequences
    src_dirs_local = value['local'] if 'local' in value else []
    seq_paths_local = []
    for src_dir in src_dirs_local:
        print(f"Scanning '{src_dir}'")
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                seq_paths_local.append([subdir, 'local'])
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
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                seq_paths_non_local.append([subdir, 'non-local'])
                if src_dir in img_counts:
                    img_counts[src_dir] += check_n_imgs_in_seq(subdir)
                else:
                    img_counts[src_dir] = check_n_imgs_in_seq(subdir)
    random.shuffle(seq_paths_non_local)

    # Add half of the shuffled local seqs to the beginning and the other half to the end of the shuffled non-local seqs.
    # This will make sure that half of the local images will end up in either val or test (depending on prop_to_val) and 
    # half of the local images will end up in training. If there are no local images, all will be non-local. If there are
    # more local images than non-local images, the entire test and/or val will consist of local, and the train of more than
    # half local images. This way there is a higher proportion of local images in the test/val sets, if there are any local
    # images. 
    seq_paths_local_fst_half = seq_paths_local[:len(seq_paths_local)//2]
    seq_paths_local_sec_half = seq_paths_local[len(seq_paths_local)//2:]
    seq_paths_tot = [*seq_paths_local_fst_half, *seq_paths_non_local, *seq_paths_local_sec_half]

    # calculate the total number of images for each class and add it to the total for this training session
    class_img_total = 0
    for src_dir, _ in seq_paths_tot:
        for subdir, dirs, fnames in os.walk(src_dir):
            if os.path.basename(os.path.normpath(subdir)).startswith('PvL_seq_'):
                class_img_total += check_n_imgs_in_seq(subdir)
    session_img_total += class_img_total

    # calculate train, test, val counts
    n_imgs_val_req = round(class_img_total * prop_to_val)
    n_imgs_test_req = round(class_img_total * prop_to_test)
    n_imgs_train_req = class_img_total - n_imgs_test_req - n_imgs_val_req

    # place the seq paths into train, val, test lists
    img_paths_val = []
    n_imgs_val_real = 0

    img_paths_test = []
    n_imgs_test_real = 0

    img_paths_train = []
    n_imgs_train_real = 0
    
    for i, [seq, source] in enumerate(seq_paths_tot):

        # set sequence index
        seq_idx = i + 1

        # check what needs to be filled
        if prop_to_val > 0:
            fill_val = True if n_imgs_val_real <= n_imgs_val_req else False
        else:
            fill_val = False
        fill_test = True if n_imgs_test_real <= n_imgs_test_req and not fill_val else False
        fill_train = True if not fill_val and not fill_test else False

        # move seq from train to val or test
        n_imgs_in_seq = check_n_imgs_in_seq(seq)
        fp_imgs_in_seq = get_img_paths_from_seq(seq)
        fp_imgs_in_seq_with_metadata = [[x, seq_idx, source] for x in fp_imgs_in_seq]
        if fill_val:
            img_paths_val.extend(fp_imgs_in_seq_with_metadata)
            n_imgs_val_real += n_imgs_in_seq

        elif fill_test:
            img_paths_test.extend(fp_imgs_in_seq_with_metadata)
            n_imgs_test_real += n_imgs_in_seq

        elif fill_train:
            img_paths_train.extend(fp_imgs_in_seq_with_metadata)
            n_imgs_train_real += n_imgs_in_seq

    # calculate differences between required and realised numbers
    real_val_prop = round(len(img_paths_val)/class_img_total, 3)
    diff_val = round(abs(real_val_prop - prop_to_val), 2)
    real_test_prop = round(len(img_paths_test)/class_img_total, 3)
    diff_test = round(abs(real_test_prop - prop_to_test), 2)
    real_train_prop = round(len(img_paths_train)/class_img_total, 3)
    diff_train = round(abs(real_train_prop - prop_to_train), 2)
    diff_total = round(diff_val + diff_test + diff_train, 2)

    # calculate train, test, val counts
    class_dict[key] = {
                     'class_img_total' : class_img_total,

                     'n_imgs_val_req' : n_imgs_val_req,
                     'n_imgs_val_real' : n_imgs_val_real,
                     'prop_to_val' : prop_to_val,
                     'real_val_prop' : real_val_prop,
                     'diff_val' : diff_val,
                     'img_paths_val' : img_paths_val,

                     'n_imgs_test_req' : n_imgs_test_req,
                     'n_imgs_test_real' : n_imgs_test_real,
                     'prop_to_test' : prop_to_test,
                     'real_test_prop' : real_test_prop,
                     'diff_test' : diff_test,
                     'img_paths_test' : img_paths_test,

                     'n_imgs_train_req' : n_imgs_train_req,
                     'n_imgs_train_real' : n_imgs_train_real,
                     'prop_to_train' : prop_to_train,
                     'real_train_prop' : real_train_prop,
                     'diff_train' : diff_train,
                     'img_paths_train' : img_paths_train,

                     'total_diff' : diff_total}

# now we have constructed a dict with all information, we need to check how it looks
# print extensive information per class
print("")
summaries = []
for key, value in class_dict.items():
    summaries.append([key, value['diff_train'], value['class_img_total']])
    stats = [['val', value['n_imgs_val_req'], value['n_imgs_val_real'], value['prop_to_val'], value['real_val_prop'], value['diff_val']],
             ['test', value['n_imgs_test_req'], value['n_imgs_test_real'], value['prop_to_test'], value['real_test_prop'], value['diff_test']],
             ['train', value['n_imgs_train_req'], value['n_imgs_train_real'], value['prop_to_train'], value['real_train_prop'], value['diff_train']],
             ['total', '-', '-', '-', '-', value['total_diff']]]
    print(f"{key}:")
    print(tabulate(stats, headers=['Type', 'Required imgs', 'Realised imgs', 'Required prop', 'Realised prop', 'Difference in prop']))
    print("")

# print a summary per class
for summary in summaries:
    print(f"Percentage off for class {summary[0].ljust(25)} {str(round(summary[1], 2)).ljust(5)} (n images: {str(summary[2])})")
print("")

# ask if the differences are acceptable
inp = input('Do you want to continue with these values? Answer [Y]es or [N]o. ')
if inp.lower() == 'y':
    export_paths_to_json()
    print("Continuing...")
    pbar = tqdm(total=session_img_total)
    for class_name, values in class_dict.items():
        pbar.set_description(f"{class_name.ljust(20)}")
        image_idx = 1
        for typ in ['val', 'test', 'train']:
            if values[f"img_paths_{typ}"] != []:
                image_infos = values[f"img_paths_{typ}"]
                for image_info in image_infos:
                    src_file = pathlib.PurePath(image_info[0])
                    seq_idx = image_info[1]
                    local_non_local = image_info[2]
                    src_file_ext = os.path.splitext(src_file)[1]
                    unique_fname = f"seq{seq_idx:06}-img{image_idx:06}-{local_non_local}" + src_file_ext
                    dst_file = os.path.join(dst_folder, typ, class_name, unique_fname)
                    src_file = os.path.normpath(src_file)
                    dst_file = os.path.normpath(dst_file)
                    Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    image_idx += 1
                    pbar.update(1)