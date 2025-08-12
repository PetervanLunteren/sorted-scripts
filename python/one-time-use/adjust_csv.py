# Script to loop lila bc csv file and fill in the sequence_ID for ENA24 and the datetime for ICCT datasets
# Peter van Lunteren, 26 Oct 2023

# """
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\adjust_csv.py"
# """

"""
conda activate ecoassistcondaenv; python "/Users/peter/Desktop/adjust_csv.py"
"""

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
# from tabulate import tabulate
import collections
import re

df = pd.read_csv(r"C:\Peter\fetch-lila-bc-data\csv\lila_image_urls_and_labels.csv", dtype = 'string')

with open(r"C:\Users\smart\Desktop\ENA24-copy\ENA24_img2seq_dict.json", "r") as f:
    ENA_seqs = json.load(f)

n_different_format = 0
n_changed = 0
n_value_error = 0
for i, row in tqdm(df.iterrows()):
    dataset_name = row['dataset_name']

    if dataset_name == "Island Conservation Camera Traps":
        fpath_rel = row['url'].replace('https://lilablobssc.blob.core.windows.net/islandconservationcameratraps/public', '')
        fname = os.path.basename(os.path.normpath(fpath_rel))
        try:
            timestamp_raw = re.search("_\d{8}_\d{6}_", fname).group(0)[1:16].split('_')
        except:
            try:
                timestamp_raw = re.search("_\d{14}_", fname).group(0)[1:16].split('_')
            except:
                n_different_format += 1
                continue

        year = timestamp_raw[0][:4]
        month = timestamp_raw[0][4:6]
        day = timestamp_raw[0][6:]
        hour = timestamp_raw[1][:2]
        min = timestamp_raw[1][2:4]
        sec = timestamp_raw[1][4:]
        datetime_str = f'{month}-{day}-{year} {hour}:{min}:{sec}'
        
        # check format by trying to convert, will raise error if not well formatted
        try:
            formatted_datetime = datetime.strptime(datetime_str, '%m-%d-%Y %H:%M:%S').strftime("%Y:%m:%d %H:%M:%S") 
        except ValueError:
            n_value_error += 1
            continue
        
        # adjust
        n_changed += 1
        # df.loc[i, 'datetime'] = datetime_str # didn't dare to adjust the datetime fields with so much uncertainty

    elif dataset_name == "ENA24":
        fname = row['url'].replace('https://lilablobssc.blob.core.windows.net/ena24/images/', '')
        if fname in ENA_seqs: # some images were removed from the dataset (probabaly containing humans)
            df.loc[i, 'sequence_id'] = ENA_seqs[fname]
        else: 
            df.loc[i, 'sequence_id'] = "ENA24 : unknown"

print(f"n_different_format : {n_different_format}")
print(f"n_value_error      : {n_value_error}")
print(f"n_changed          : {n_changed}")

# write adjusted csv
df.to_csv(r"C:\Peter\fetch-lila-bc-data\csv\lila_image_urls_and_labels_ADJUSTED_PVL.csv")
print("Done")