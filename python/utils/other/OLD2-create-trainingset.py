# Script to copy files from the main database to a training folder for training a yolov8 classifier
# we can't work with a yaml file, so we need to make the folder structure every time
# Peter van Lunteren 3 oct 2023

# """ MACOS
# conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv;python "/Users/peter/Documents/scripting/sorted-scripts/python/utils/xml-related/crop_detects_from_xml.py"
# """

# """ WINDOWS
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\2-create-trainingset.py"
# """


import os
import random
from pathlib import Path
import shutil


# define proportions
prop_to_test = 0.2
prop_to_val = 0.1


# define dictionary for which data belongs to which class. Format is -> class_name : [path_to_dir1, path_to_dir2]
class_dict = {'Panthera_leo' : [r'C:\Peter\verified-datasets\square-crops-for-classification\panthera-leo-LILABC'],
              'Other_animal' : [r'C:\Peter\verified-datasets\square-crops-for-classification\non-lion-NAMIBIA']}

# define the destination folder where you want to collect all the copies of training images.
training_folder = r'C:\Users\smart\Desktop\CURRENT_TRAIN_SET'

# move files 
for class_name, data_folders in class_dict.items():
    test_files = []
    val_files = []
    train_files = []
    for data_folder in data_folders:
        print(f"Fetching images for class {class_name} from dir {data_folder}")
        files = []
        for f in os.listdir(data_folder):
            if os.path.isfile(os.path.join(data_folder, f)) and not f.endswith(".DS_Store") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                files.append(os.path.join(data_folder, f))
        total_n = len(files)
        n_test = int(total_n * prop_to_test)
        n_val = int(total_n * prop_to_val)
        random.shuffle(files)
        test_files.extend(files[:n_test])
        val_files.extend(files[n_test:n_test+n_val])
        train_files.extend(files[n_test+n_val:])

        for elem in [[test_files, 'test'], [val_files, 'val'], [train_files, 'train']]:
            for src_file in elem[0]:
                fname = os.path.basename(src_file)
                dst_file = os.path.join(training_folder, elem[1], class_name, fname)
                Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                print(f"src_file : {src_file}")
                print(f"dst_file : {dst_file}")
        
        print(f"test_files : {len(test_files)}")
        print(f"val_files : {len(val_files)}")
        print(f"train_files : {len(train_files)}")