# Script to validate a yolov8 or aniML classifier on the test and val data is used during training
# It will test the precision, recall, and accuracy for all individual classes and generate plots and graphs
# it will also create error reports PDF

# it can be executed in two ways:

    # 1. run this script directly after yolov8 training has completed. The it needs the following arguments:
    ## python val-test-set.py dataset_folder output_folder

    # 2. run this script on the aniML test results. Then you'll need to run the aniML test script first to create 
    # test_results.csv, and then run this script pointing to the config file
    ## conda activate ecoassistcondaenv-pytorch && python val-test-set.py "animl" "path\to\used-config.yml"

# TODO: it would be great to have a bar chart with the number of errors per location. That will tell us something about the quality of the data per location. 

# import packages
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
from ultralytics import YOLO
import os
from sklearn import metrics # pip install scikit-learn
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import pandas as pd
import re
import seaborn as sns
import json
import torch
from PyPDF2 import PdfMerger # python -m pip install --upgrade pip & pip install PyPDF2
import sys
import openpyxl # python -m pip install --upgrade pip & pip install openpyxl
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import subprocess

# let's not freak out on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# find script location
script_dir = os.path.dirname(os.path.realpath(__file__))

# import module from parent directory
sys.path.append(os.path.join(script_dir, ".."))
from testutils import *

# fetch classifications for single crop
def get_classification(model, img_fpath):
    results = model(img_fpath, verbose = False, imgsz = 224)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    classifications = []
    for idx, v in names_dict.items():
        classifications.append([v, probs[idx]])
    max_class = max(classifications, key=lambda x: x[1])
    return max_class

# find highest number of train dir
def get_highest_train_subdir(main_folder):
    train_dirs = [dir for dir in os.listdir(main_folder) if dir.startswith('train')]
    if not train_dirs:
        print("\nNo train directories found. Exiting...")
        exit()
    train_numbers = [int(dir[5:]) for dir in train_dirs if dir[5:].isdigit()]
    if not train_numbers:
        return os.path.join(main_folder, f'train')
    return os.path.join(main_folder, f'train{max(train_numbers)}')

# get subdir names which are the class names
def get_classes(test_dir):
    return [name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))]

# get the same csv files as aniML has
def predict_yolov8(training_input_dir, training_output_dir):

    print(f"\ntraining_input_dir  : {training_input_dir}")
    print(f"training_output_dir : {training_output_dir}\n")

    for split in ['val', 'test']:

        # log
        print(f"\nRunning class-specific testing on {split} set...")

        # check if split set exists
        split_dir = os.path.join(training_input_dir, split)
        if not os.path.exists(split_dir):
            print(f"\nSplit set directory not found: {split_dir}")
            print(f"Skipping {split}...")
            continue

        # load model
        model_fpath = os.path.join(get_highest_train_subdir(os.path.join(training_output_dir, "runs", "classify")), "weights", "best.pt")
        model = YOLO(model_fpath)
        print(f"\nLoaded model: {model_fpath}")

        # validate on each subfolder
        test_dir = os.path.join(training_input_dir, split)

        # count
        total_imgs = 0
        for true_class in get_classes(test_dir):
            imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
            total_imgs += len(imgs)

        # predict 
        print(f"\nPredicting images on {split} set...\n")
        y_pred = []
        y_true = []
        confs = []
        paths = []
        pbar = tqdm(total=total_imgs)
        for true_class in get_classes(test_dir):
            imgs = [os.path.join(test_dir, true_class, img) for img in os.listdir(os.path.join(test_dir, true_class)) if os.path.isfile(os.path.join(test_dir, true_class, img))]
            for img in imgs:
                try:
                    pred_class, conf = get_classification(model, img)
                    y_pred.append(pred_class)
                    y_true.append(true_class)
                    confs.append(conf)
                    paths.append(img)
                except FileNotFoundError:
                    print(f"\nSkipping corrupted or truncated image: {img}\n")
                    continue
                pbar.update(1)
        corr = [pred == true for pred, true in zip(y_pred, y_true)]

        # save test results as csv 
        csv_fpath = os.path.join(os.path.dirname(os.path.dirname(model_fpath)), f"{split}_results_adjusted.csv")
        df = pd.DataFrame({"path": paths, "true": y_true, "pred": y_pred, "conf": confs, "corr": corr})
        df.to_csv(csv_fpath)
        print(f"Created: .\\{split}_results_adjusted.csv")


########################################
####### PREPARE ANIML OR YOLOV8 ########
########################################

def prepare_csv_files_and_generate_metrics(column_name_to_mapping = None):

    # check if command comes from yolov8 training or from aniML
    if sys.argv[1] == "yolov8":
        
        # if yolov8, we need to gegenerate the test and val CSVs
        print("\nWe are running this script after yolov8 training")
        print("will first generate test and val CSVs...")
        src_dir = os.path.normpath(sys.argv[2])
        val_src_dir = os.path.join(src_dir, 'val')
        test_src_dir = os.path.join(src_dir, 'test')
        train_src_dir = os.path.join(src_dir, 'train')
        dst_dir = os.path.normpath(sys.argv[3])
        exp_dir = get_highest_train_subdir(os.path.join(dst_dir, "runs", "classify"))
        print(f"\nval_src_dir   : {val_src_dir}")
        print(f"test_src_dir  : {test_src_dir}")
        print(f"train_src_dir : {train_src_dir}")
        print(f"dst_dir       : {dst_dir}")
        predict_yolov8(src_dir, dst_dir)
        
    elif sys.argv[1] == "animl":
        
        # if aniML, we need to get the config file and set vars
        print("\nWe are running this script after aniML training")
        cfg = yaml.safe_load(open(sys.argv[2], 'r'))
        exp_dir = os.path.dirname(sys.argv[2])
        print(f"\nexp_dir: {exp_dir}")
        src_dir = os.path.normpath(os.path.dirname(cfg['training_set']))
        val_src_dir = os.path.join(src_dir, 'val')
        test_src_dir = os.path.join(src_dir, 'test')
        train_src_dir = os.path.join(os.path.dirname(cfg['training_set']), 'train')

        # convert the ANIML CSVs to the right format
        for split in ['val', 'test']:
            
            # init vars
            csv_fpath_orginial = os.path.join(exp_dir, f'{split}_results.csv')
            csv_fpath_adjusted = os.path.join(exp_dir, f'{split}_results_adjusted.csv')
            
            # skip if never tested on this split
            if not os.path.exists(csv_fpath_orginial):
                print(f"\nResults file not found: {csv_fpath_orginial}")
                print(f"Skipping {split}...")
                continue
            
            # prepare vars
            dst_dir = os.path.join(exp_dir, 'test-results', f'on-{split}')
            Path(dst_dir).mkdir(parents=True, exist_ok=True)
            if split == 'val':
                split_src_dir = os.path.join(src_dir, 'val')
            else:
                split_src_dir = os.path.join(src_dir, 'test')
            train_src_dir = os.path.join(src_dir, 'train')
            tested_classes = get_classes(split_src_dir)
            all_classes = get_classes(train_src_dir)

            # get classes from model
            active_model = os.path.join(exp_dir, 'best.pt')
            checkpoint = torch.load(active_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            categories = checkpoint['categories']
            categories_inverted = {v-1: k for k, v in categories.items()}

            # make sure it is in the right format
            column_mapping = {
                "FilePath": "path",
                "Ground Truth": "true",
                "Predicted": "pred",
                "Confidence": "conf",
                "Correct": "corr"}
            df = pd.read_csv(csv_fpath_orginial)
            df = df.rename(columns={col: column_mapping[col] for col in df.columns if col in column_mapping})
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df['true'] = df['true'].replace(categories_inverted)
            df['pred'] = df['pred'].replace(categories_inverted)

            # remap to higher level taxons 
            if column_name_to_mapping:
                taxon_mapping_csv = os.path.join(src_dir, 'taxon-mapping.csv')
                taxon_mapping_df = pd.read_csv(taxon_mapping_csv)

                # re map the predictions
                mapping_dict = {}
                for _, row in taxon_mapping_df.iterrows():
                    model_class = row["model_class"]
                    mapping_dict[model_class] = row[column_name_to_mapping]
                df['true'] = df['true'].replace(mapping_dict)
                df['pred'] = df['pred'].replace(mapping_dict)
                df['corr'] = df['pred'] == df['true']

            # write csv
            df.to_csv(csv_fpath_adjusted, index=False)
            csv_fpath = csv_fpath_adjusted

    #################################
    ####### GENERATE METRICS ########
    #################################

    # create plots and graphs
    pdf_pages = []
    split_idx = 1
    main_front_page_generated = False
    for split in ['test', 'val']:
        
        # init vars
        csv_fpath = os.path.join(exp_dir, f'{split}_results_adjusted.csv')
        
        # skip if never tested on this split
        if not os.path.exists(csv_fpath):
            print(f"\nResults file not found: {csv_fpath}")
            print(f"\nSkipping {split}...")
            continue
        
        # if the split exists, generate paragragh idx
        split_idx += 1
        
        # prepare vars
        dst_dir = os.path.join(exp_dir, 'test-results', f'on-{split}')
        Path(dst_dir).mkdir(parents=True, exist_ok=True)
        split_src_dir = val_src_dir if split == 'val' else test_src_dir
        tested_classes = get_classes(split_src_dir)
        all_classes = get_classes(train_src_dir)

        # open csv predictions
        print(f"\n\nReading: {csv_fpath}\n\n")
        df = pd.read_csv(csv_fpath)
        paths = df["path"]
        true = df["true"]
        pred = df["pred"]
        conf = df["conf"]
        corr = df["corr"]

        # clean up class names
        true = [cls.rstrip('.').strip() for cls in true]
        pred = [cls.rstrip('.').strip() for cls in pred]

        # make sure all_classes consists only of classes that are actually present in the test set
        all_classes = np.unique(np.concatenate([true, pred]))

        # plot training metrics
        cm = confusion_matrix(true, pred)
        confuse = pd.DataFrame(cm, columns=all_classes, index=all_classes)
        confuse.to_csv(os.path.join(dst_dir, "confusion_matrix.csv"))
        print(f"Created: .\\confusion_matrix.csv")

        # save classification report
        classification_report_txt = classification_report(true, pred, target_names=all_classes, zero_division=np.nan)
        classification_report_dict = classification_report(true, pred, target_names=all_classes, output_dict=True, zero_division=np.nan)
        report_path = os.path.join(dst_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report_txt)
        print(f"Created: .\\classification_report.txt")

        # read in spp_plan.xlsx
        spp_plan_xlsx = pd.read_excel(os.path.join(src_dir, 'used_spp_plan.xlsx'))

        # remap to higher level taxons 
        if column_name_to_mapping:
            taxon_mapping_csv = os.path.join(src_dir, 'taxon-mapping.csv')
            taxon_mapping_df = pd.read_csv(taxon_mapping_csv)

            # re map the predictions
            mapping_dict = {}
            for _, row in taxon_mapping_df.iterrows():
                model_class = row["model_class"]
                mapping_dict[model_class] = row[column_name_to_mapping]

            spp_plan_xlsx['Class'] = spp_plan_xlsx['Class'].str.lower()
            spp_plan_xlsx['Class'] = spp_plan_xlsx['Class'].replace(mapping_dict)

        # group xlsx
        spp_plan_grouped = spp_plan_xlsx.groupby(['Class', 'Source'])['N_images'].sum().unstack(fill_value=0)
        spp_plan_grouped.columns = spp_plan_grouped.columns.str.lower() # Convert column names to lowercase

        spp_plan_dict = {
            class_name: {
                "total_images": int(row.sum()),  # Convert to Python int
                "local_images": int(row.get("local", 0)),  # Convert to Python int
                "non_local_images": int(row.get("non-local", 0))  # Convert to Python int
            }
            for class_name, row in spp_plan_grouped.iterrows()
        }

        # get the split count information
        split_count_df = pd.read_csv(
            os.path.join(src_dir, 'split_count_df.csv'),
            index_col=0
        )

        # rebuild the split_count_df.csv as it is not based on the taxonomic levels, only on the true model classes
        if column_name_to_mapping:
            split_count_df.index = split_count_df.index.str.lower()
            split_count_df.index = split_count_df.index.to_series().replace(mapping_dict)
            numeric_df = split_count_df.applymap(lambda x: int(str(x).split(' ')[0]))
            grouped = numeric_df.groupby(split_count_df.index).sum()

            def format_with_percentage(df):
                result = df.copy()
                total = df["Total"]
                for col in df.columns:
                    percent = (df[col] / total * 100).round(1)
                    result[col] = df[col].astype(str) + " (" + percent.astype(str) + "%)"
                return result

            split_count_df = format_with_percentage(grouped)
        
        # get the location counts
        with open(os.path.join(src_dir, 'location_counts.json'), 'r') as f:
            location_counts = json.load(f)

        # get project name
        def extract_matching_element(lst):
            pattern = r'^\d{4}-\d{2}-[A-Za-z]{3}$'  # regex for 4 digits-2 digits-3 letters
            for element in lst:
                if re.match(pattern, element):
                    return element
            return None
        project_name = extract_matching_element(src_dir.split(os.sep)) # somewhere in the path to the dataset there must be the project name xxxx-xx-xxx

        # error reports 
        if not main_front_page_generated:
            front_page = create_front_page(df = df, 
                                dst_dir = os.path.dirname(dst_dir), # this is for both val and test
                                classification_report_dict = classification_report_dict,
                                spp_plan_dict = spp_plan_dict,
                                split = split,
                                split_count_df = split_count_df,
                                location_counts = location_counts,
                                confusion_matrix = cm,
                                all_classes = all_classes,
                                project_name = project_name,
                                column_name_to_mapping = column_name_to_mapping)
            pdf_pages.append(front_page)
            main_front_page_generated = True # make sure we only generate the main front page once
        
        split_front_page = create_split_front_page(df = df,
                            dst_dir = dst_dir,
                            classification_report_dict = classification_report_dict,
                            spp_plan_dict = spp_plan_dict,
                            split = split,
                            split_count_df = split_count_df,
                            location_counts = location_counts,
                            confusion_matrix = cm,
                            all_classes = all_classes,
                            project_name = project_name,
                            paragraph_idx = split_idx)
        pdf_pages.append(split_front_page)
        
        # get class specific error reports # OPTIONAL
        print(f"\nGenerating class-specific error reports for {split} set...")
        for idx, chosen_class in enumerate(tested_classes):
            pdf = create_class_specific_error_report_PDF(df = df,
                                                        chosen_class = chosen_class,
                                                        dst_dir = dst_dir,
                                                        metrics = classification_report_dict[chosen_class],
                                                        paragraph_idx = split_idx,
                                                        idx = idx + 6 )
            pdf_pages.append(pdf)

    # merge all PDFS together
    print("Merging PDFs...")
    merger = PdfMerger()
    for pdf in pdf_pages:
        print(f"Appending: {pdf}")
        merger.append(pdf)
    output_pdf_path = os.path.join(os.path.dirname(dst_dir), f"evaluation_report_{column_name_to_mapping}_full.pdf")
    merger.write(output_pdf_path)
    merger.close()

    # Remove single PDFs
    for pdf in pdf_pages:
        print(f"Removing: {pdf}")
        os.remove(pdf)
        
    # compress the PDF
    compressed_pdf = os.path.join(os.path.dirname(output_pdf_path), f"evaluation_report_{column_name_to_mapping}.pdf")
    gs_command = [
        r"C:\Program Files\gs\gs10.05.0\bin\gswin64c.exe",  # Full path to Ghostscript
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/ebook",  # Options: /screen, /ebook, /printer, /prepress
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={compressed_pdf}",
        output_pdf_path
    ]

    subprocess.run(gs_command)

    # remove non compressed
    if os.path.exists(output_pdf_path):
        os.remove(output_pdf_path)

    print("Compression complete:", output_pdf_path)



# MAIN
# if there is a taxon-mapping csv, I want to repeat the evaluation for each column
if sys.argv[1] == "yolov8":
    src_dir = os.path.normpath(sys.argv[2])
elif sys.argv[1] == "animl":
    cfg = yaml.safe_load(open(sys.argv[2], 'r'))
    src_dir = os.path.normpath(os.path.dirname(cfg['training_set']))
taxon_mapping_csv = os.path.join(src_dir, 'taxon-mapping.csv')

if os.path.exists(taxon_mapping_csv):
    # read in info
    taxon_mapping_df = pd.read_csv(taxon_mapping_csv)

    # read in which levels we have
    level_cols = [col for col in taxon_mapping_df.columns if col.startswith('level_')]
    only_above_cols = [col for col in taxon_mapping_df.columns if col.startswith('only_above_')]

    # generate one with the true model classes
    prepare_csv_files_and_generate_metrics(column_name_to_mapping = "model_class")

    # generate a report per level ion the taxonomic tree
    for col in level_cols + only_above_cols:
        prepare_csv_files_and_generate_metrics(column_name_to_mapping = col)

else:
    # this means we do not have any taxobnomic information, so only do it on the true model classes
    prepare_csv_files_and_generate_metrics(column_name_to_mapping = "model_class")
