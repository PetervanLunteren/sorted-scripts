# Move files based on presence in csv

import pandas as pd
import os
import shutil

df = pd.read_csv('/Users/petervanlunteren/Downloads/TUT_OBJ_DETECT_PLUGIN/uploads/Object_labels/lion_tiger_train_prepared.csv', delimiter=',')

# get required images
req_images = []
for index, row in df.iterrows():
    req_images.append(row['path'])

# loop thourgh all the images
src_dir = '/Users/petervanlunteren/Desktop/all_photos'
dst_dir = '/Users/petervanlunteren/Desktop/selection'
for filename in os.listdir(src_dir):
    f = os.path.join(src_dir, filename)
    if os.path.isfile(f) and filename in req_images:
        print(f"copying {filename}...")
        shutil.copyfile(f, os.path.join(dst_dir, filename))
        req_images.remove(filename)
        