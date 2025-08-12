# Script to loop lila bc csv file and try to identify dequences in the ENA24 and ICCT datasets
# Peter van Lunteren, 16 Sept 2023

# """
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\create_sequences.py"
# """

# # import packages
import requests
import os
import sys
import subprocess
from subprocess import Popen, PIPE
from pathlib import Path
import shutil
import linecache
import time
import pandas as pd
import zipfile
from tqdm import tqdm
import urllib.request
import json
import xml.etree.cElementTree as ET
from pycocotools.coco import COCO
from pascal_voc_writer import Writer
import pickle
import psutil
import tempfile
import datetime 
from datetime import datetime 
from tabulate import tabulate
import collections
import re

# # first collect all the ENA rows so that we can sort them properly
# df = pd.read_csv(r"C:\Peter\fetch-lila-bc-data\csv\lila_image_urls_and_labels.csv", dtype = 'string')
# ENA_rows = {}
# for i, row in tqdm(df.iterrows()):
#     dataset_name = row['dataset_name']
#     if dataset_name == "ENA24":
#         fname = row['url'].replace('https://lilablobssc.blob.core.windows.net/ena24/images/', '')
#         ENA_rows[fname] = [row['original_label']]

# # write to json so that we don't have to read that whole csv again
# export_path = r"C:\Users\smart\Desktop\ena_rows.json"
# with open(export_path, "w") as f:
#     json.dump(ENA_rows, f, indent=2)

# exit()

# read from json so that we don't have to read that whole csv again
export_path = r"C:\Users\smart\Desktop\ena_rows.json"
with open(export_path, "r") as f:
    ENA_rows = json.load(f)

# get a list of keys in human-sorted order
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]
keylist = list(ENA_rows.keys())
keylist.sort(key=natural_keys)


# now add sequence information
ENA_rows_with_sequence = {}
prev_label = ""
seq_id = 1
for key in keylist[:100]:
    print(f"key : {key}, original_label : {ENA_rows[key]}")
    curr_label = ENA_rows[key][0]
    ENA_rows_with_sequence[key] = ENA_rows[key]
    if curr_label != prev_label: # new sequence
        seq_id += 1
    ENA_rows_with_sequence[key].append(f"ENA24 : {seq_id}")
    prev_label = curr_label

# it looks like the filenames are renamed according to the label given. In other words, all labels are lumped together.
# If I would create sequences based on this, I would only have 162 sequences. That means an average of 8789 / 162 = 54 images per sequence.
print(json.dumps(ENA_rows_with_sequence, indent = 2))
