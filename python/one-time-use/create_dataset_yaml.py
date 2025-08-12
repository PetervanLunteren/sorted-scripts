# Script to create .yaml files for training data when training a yolov8 classifier
# Peter van Lunteren 3 oct 2023

import sys
import os
import random

# check data and prepare for training
def create_yaml_for_yolov8_classification(data_folder, prop_to_test, prop_to_val):
    # # log
    # print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # # convert pascal voc to yolo
    # pascal_voc_to_yolo(data_folder)

    # get list of all classes 
    data_folder = os.path.normpath(data_folder)
    class_names = []
    class_folders = []
    for class_name in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, class_name)):
            class_names.append(class_name)
            class_folders.append(os.path.join(data_folder, class_name))
    # class_names = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    # class_folder = [folder for os.path.join(data_folder, name) in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    print(f"class_names   : {class_names}")
    print(f"class_folders : {class_folders}")

    # get info per class
    test_files = []
    val_files = []
    train_files = []
    for class_folder in class_folders:
        print(f"getting info for class : {class_folder}")

        # get list of all images per class
        files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f)) and not f.endswith(".DS_Store") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
        # for f in os.listdir(class_folder):
        #     if os.path.isfile(os.path.join(data_folder, f)):
        #         print(f)
        # exit()
        
        print(f"total n files : {len(files)}")

        # calculate amounts
        prop_to_test = float(prop_to_test)
        prop_to_val = float(prop_to_val)
        total_n = len(files)
        n_test = int(total_n * prop_to_test)
        n_val = int(total_n * prop_to_val)
        print(f"n_test : {n_test}")
        print(f"n_val : {n_val}")
        print(f"n_train : {total_n - n_val - n_test}")

        # select random files
        random.shuffle(files)
        # files[:n_test].append(test_files)
        # files[n_test:n_test+n_val].append(val_files)
        # files[n_test+n_val:].append(train_files)
        test_files.extend(files[:n_test])
        val_files.extend(files[n_test:n_test+n_val])
        train_files.extend(files[n_test+n_val:])
        print(f"After appending:")
        print(f"test_files : {len(test_files)}")
        print(f"val_files : {len(val_files)}")
        print(f"train_files : {len(train_files)}")

    # remove files for previous training
    old_files = ["dataset.yaml", "train_selection.txt", "train_selection.cache", "train_selection.cache.npy", "val_selection.txt", "val_selection.cache",
                 "val_selection.cache.npy", "test_selection.txt", "test_selection.cache", "test_selection.cache.npy"]
    for filename in old_files:
        old_file = os.path.join(data_folder, filename)
        if os.path.isfile(old_file):
            os.remove(old_file)

    # write text files with images
    for elem in [[train_files, "train"], [val_files, "val"], [test_files, "test"]]:
        counter = 0
        with open(os.path.join(data_folder, elem[1] + "_selection.txt"), 'w') as f:
            for file in elem[0]:
                f.write("./" + file + "\n")
                counter += 1
        print(f"\nWill use {counter} images as {elem[1]}")

    # create dataset.yaml
    if prop_to_test == 0:
        yaml_content = f"# set paths\npath: '{data_folder}'\ntrain: ./train_selection.txt\nval: ./val_selection.txt\n\n# n classes\nnc: {len(class_names)}\n\n# class names\nnames: {class_names}\n"
    else:
        yaml_content = f"# set paths\npath: '{data_folder}'\ntrain: ./train_selection.txt\nval: ./val_selection.txt\ntest: ./test_selection.txt\n\n# n classes\nnc: {len(class_names)}\n\n# class names\nnames: {class_names}\n"
    yaml_file = os.path.join(data_folder, "dataset.yaml")
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
        print(f"\nWritten {yaml_file} with content:\n\n{yaml_content}\n")

data_folder = sys.argv[1]
prop_to_test = sys.argv[2]
prop_to_val = sys.argv[3]

create_yaml_for_yolov8_classification(data_folder, prop_to_test, prop_to_val)