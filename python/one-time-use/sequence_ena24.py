# Script to semi-automatically devide the ENA24 dataset into sequences based on image similarity
# It's still a automated way of doing stuff, so don't expect perfect results.
# It's always a tradeoff, and I chose to use cutoff = 7.
# where it will sometimes lump sequences together, but is quite good in not separating sequences so you'll get data independece issues in your train / test / val splits resulting in overconfident models.
# Peter van Lunteren, 26 Oct 2023

# # execute script in miniforge prompt
# """
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\sequence_ena24.py"
# """

# image folder
ena24_dir = r"C:\Users\smart\Desktop\ENA24-copy"

# set *manually_investigate* to True if you want to manually investigate *investigate_n_images* images
manually_investigate = False
investigate_n_images = 1000

from PIL import Image
import imagehash
import os
import re
import json
import numpy as np
import cv2
from tqdm import tqdm

# get a list of all files
files = []
for fname in os.listdir(ena24_dir):
    if fname.endswith('.jpg'):
        files.append(fname)

# make sure it is in the right human-sorted order
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]
files.sort(key=natural_keys)

# compare the images with the previous one, if too different, new sequence.
# the *cutoff* defines the maximum bits that could be different between the images.
# cutoff = 10 created 34 seqs in the first 100 images. I could clearly see that it put multiple sequences together.
# cutoff = 5 created 38 seqs in the first 100 images. Here I could see that it separated sequences where the animal was quite large and moving around, but TBH you're not really going to avoid that with this automated method.
# cutoff = 7 created 37 seqs in the first 100 images. This is probably the best tradeoff. Given the fact that a ML person would likely want more sequences together, than separated sequences. This would result in the train / test / val split continaing images of the same sequence and give an overconfident model. 
cutoff = 7
seq_id = 1
seqs = {"ENA24 : 1" : []}
prev_fname = "1.jpg"
first_iteration = True

if manually_investigate:
    pbar = tqdm(files[:investigate_n_images])
else:
    pbar = tqdm(files)

for fname in pbar:
    curr_fname = fname
    hash0 = imagehash.average_hash(Image.open(os.path.join(ena24_dir, prev_fname)))
    hash1 = imagehash.average_hash(Image.open(os.path.join(ena24_dir, curr_fname)))
    if hash0 - hash1 < cutoff:
        if manually_investigate:
            print(f'Comparing "{prev_fname}" - "{curr_fname}" : "{curr_fname}" belongs to the same sequence and will be added to "ENA24 : {seq_id}"')
        img_in_seq = seqs[f"ENA24 : {seq_id}"]
        img_in_seq = img_in_seq.append(curr_fname)
    else:
        if manually_investigate:
            print(f'Comparing "{prev_fname}" - "{curr_fname}" : "{curr_fname}" does not belong to the same sequence and a new sequence will be created')
        seq_id += 1
        seqs[f"ENA24 : {seq_id}"] = [curr_fname]
    prev_fname = curr_fname
    pbar.update(1)

# let's see how that looks
if manually_investigate:
    print(json.dumps(seqs, indent = 2))

# visually check
if manually_investigate:
    prev_cv2_objs = []
    for seq, fnames in seqs.items():
        curr_cv2_objs = []
        for fname in fnames:
            fpath = os.path.join(ena24_dir, fname)
            curr_cv2_obj = cv2.imread(fpath)
            curr_cv2_obj = cv2.resize(curr_cv2_obj, (200, 100))
            curr_cv2_objs.append(curr_cv2_obj)
        
        curr_seq = np.concatenate(curr_cv2_objs, axis=0)
        curr_name = f"CURRENT SEQ"
        cv2.namedWindow(curr_name)
        cv2.moveWindow(curr_name, 500,30)
        cv2.imshow(curr_name, curr_seq)

        if prev_cv2_objs != []:
            prev_seq = np.concatenate(prev_cv2_objs, axis=0)
            prev_name = f"PREVIOUS SEQ"
            cv2.namedWindow(prev_name)
            cv2.moveWindow(prev_name, 50,30)
            cv2.imshow(prev_name, prev_seq)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        prev_cv2_objs = curr_cv2_objs

# convert dict to image : seq
imgs = {}
for seq, fnames in seqs.items():
    for fname in fnames:
        imgs[fname] = seq

# let's see how that looks
if manually_investigate:
    print(json.dumps(imgs, indent = 2))

# export to json
export_path = os.path.join(ena24_dir, "ENA24_img2seq_dict.json")
with open(export_path, "w") as f:
    json.dump(imgs, f, indent=2)