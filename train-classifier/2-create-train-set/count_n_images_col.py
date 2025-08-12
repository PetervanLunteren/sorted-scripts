# Script to fetch taxon info from GBIF API and update an Excel file
# Peter van Lunteren, 14 April 2025

# this script is created to double check the n_images column in the excel file. 
# sometimes during a project you mvoe around some dirs and then its hard to keep the n_images column up to date
# it will add a new column to the existing excel file

# make sure to run this script on the PC as it needs to check the path

# import packages
import pandas as pd
from tqdm import tqdm
import os

# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\count_n_images_col.py"

import os
import pandas as pd
from tqdm import tqdm

# user input
excel_file = r"C:\Users\smart\Desktop\2024-25-ARI-spp-plan.xlsx"

# read all sheets first
all_sheets = pd.read_excel(excel_file, sheet_name=None)

# work on label_map
df = all_sheets["label_map"]

# return a list of all image paths in a directory
def get_img_paths(dir):
    file_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                file_paths.append(os.path.join(root, file))
    return file_paths

# calculate and assign n_double_check
n_double_check = []
something_is_wrong = False
for index, row in tqdm(df.iterrows(), total=len(df)):
    path = row['Path']
    n_images_from_excel = row['N_images']
    n_images_calculated = len(get_img_paths(path))
    n_double_check.append(n_images_calculated)

    if n_images_from_excel != n_images_calculated:
        something_is_wrong = True
        print(f"\n\nFor path '{path}' the n_images from excel and calculated are not the same!")
        print(f"\nn_images_from_excel : {n_images_from_excel}")
        print(f"n_images_calculated : {n_images_calculated}")
        print(f"row number          : {index + 2}")
        if not os.path.exists(path):
            print(f"Dir does not exist!")
        print("\n\n")

df["n_double_check"] = n_double_check
all_sheets["label_map"] = df

# write back to Excel, preserving all other sheets
with pd.ExcelWriter(excel_file, engine="openpyxl", mode="w") as writer:
    for sheet_name, sheet_df in all_sheets.items():
        sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
