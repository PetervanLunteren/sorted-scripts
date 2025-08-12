# Script to test a classifier on a set of previously unseen images, and calculate the best possible
# combination of EcoAssist parameters. I call this 'ecosystem calibration'. You'll need a set of human
# verified images with xml annotations. Preferably these verified set should contain a representative 
# set of images. So run the model over some example data, export 50 random images for each class and
# find the accosiated sequences. You should more or less have 100-200 images of each class.

# How do do get everything ready for this script?
# step 1: create a calibration set with create-calibration-set.py
# step 2: run MD over this calibration set and human verify 
# step X: when everything is human verified, export as training data (thus with xml annotation files). This will be your calibration directory.
                # volgens mij hoeft dit niet:                # step 3: take all EA output TXT and JSON files and store them somwhere 
                # volgens mij hoeft dit niet:                # step 4: take the image_recognition_file.json and rename it to ground-truth.json, leave it in the claibration set root and remove the rest of the files
                # volgens mij hoeft dit niet:                # step 5: run EA again, but without smoothing selected and minimum classification threshold. Then rename the image_recognition_file.json to "no-smooth-001.json"
                # volgens mij hoeft dit niet:                # step 6: repeat step 5 but now with a classification threshold of 0.6, rename to "no-smooth-060.json"
# step 6: now run EA over the directory from the previous step (so with the xml annotations), with smoothing selected and minimum classification threshold, and leave all the output files where they are.
# step 7: check the input variables and run this script

# TODO: maak een overzichtje van de hoeveelheid test images in the ground truth
# TODO: maak een stap dat je eerst een json maakt van de xml files daaro, dat is dan stap 1, daarna ga je pas die jsons vergelijken. Op die manier kun je heel makkelijk nog wat extra data toevoegen en weghalen.
# TODO: arg1 - 7 zijn voor classification smoothing, terwijl de andere voor sequence smoothing zijn. Dat betekent dat ze niet allemaal van alkaar af hangen enje niet alles door de heavy loop hoeft te gooien. Je kunt er twee heavies van maken! Dat is al een stuk beter. 
# TODO: kijk even hoe en wat met die conf thresholds. Hoe werkt dat precies???? Misschien even goed als ik dan wat echte data heb
# TODO: doe een test run op threshold 0 en op ave_rc en pr met nullen mee gerekend en kijk of er dan meer verschil zit tussen de parameters 

# make sure tqdm's from modules are not printed
import tqdm
def tqdm_replacement(iterable_object,*args,**kwargs):
    return iterable_object
tqdm_copy = tqdm.tqdm 
tqdm.tqdm = tqdm_replacement

# import packages
from itertools import product
from decimal import Decimal
from tqdm import tqdm
from math import cos, sin, sqrt
import json
import itertools
import os
import xml.etree.ElementTree as ET
from copy import copy
import difflib
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import make_interp_spline 
import sys
import pandas as pd
from pathlib import Path

# """
# conda activate ecoassistcondaenv && python "C:\Users\smart\Desktop\6-ecosystem_calibration.py"
# conda activate ecoassistcondaenv && python /Users/peter/Documents/scripting/sorted-scripts/python/train-classifier/6-ecosystem_calibration.py
# """

#############################################
################### INPUT ###################
#############################################

calibration_jsons_dir = r"C:\Peter\desert-lion-project\calibration-set"
# calibration_jsons_dir = "/Users/peter/Desktop/calibration-set"
EcoAssist_files = r"C:\Users\smart\EcoAssist_files"
# EcoAssist_files = "/Applications/.EcoAssist_files"
export_dir = os.path.join(calibration_jsons_dir, "calibration-output-folder")

# if the classes changed since the human verification {old_class : new_class}
convert_classes = {"aardwolf" : "other",
                   "donkey" : "other",
                   "genet" : "other",
                   "honey badger" : "other"}

# these are the argument values which the script takes as starting values
defaults = {"arg1" : 4,
            "arg2" : 3,
            "arg3" : 2,
            "arg4" : 0.6,
            "arg5" : 0.3,
            "arg6" : 0.2,
            "arg7" : 0.05,
            "arg8" : 5,
            "arg9" : 5,
            "arg10" : {None : 3},
            "arg11" : 3,
            "arg12" : 3,
            "arg13" : 0.6,
            "arg14" : 0.6,
            "arg15" : 0.6,
            "arg16" : 0.15}

# make it easy to define range below 1
def decimal_range(min, max, decimal_precision):
    inv_int = int(round(1 / decimal_precision))
    return [round(x * decimal_precision, 10) for x in range(min * inv_int, max * inv_int)]

# set the ranges through which the arguments have to loop
## CLASSIFICATION SMOOTHING
arg1_values = list(range(0, 20, 5)) # min_detections_above_threshold - int 4 - can be anything
arg2_values = list(range(0, 20, 5)) # max_detections_secondary_class - int 3 - can be anything
arg3_values = list(range(0, 20, 5)) # min_detections_to_overwrite_other - int 2 - can be anything
arg4_values = decimal_range(0, 1, 0.2) # classification_confidence_threshold - float 0.6 - between 0 and 1
arg5_values = decimal_range(0, 1, 0.2) # classification_overwrite_threshold - float 0.3 - between 0 and 1
arg6_values = decimal_range(0, 1, 0.2) # detection_confidence_threshold - float 0.2 - between 0 and 1
arg7_values = decimal_range(0, 1, 0.2) # detection_overwrite_threshold - float 0.05 - between 0 and 1

# SEQUENCE SMOOTHING
arg8_values = list(range(0, 20, 5)) # min_dominant_class_classifications_above_threshold_for_class_smoothing - int 5 - can be anything
arg9_values = list(range(0, 20, 5)) # max_secondary_class_classifications_above_threshold_for_class_smoothing - int 5 - can be anything
arg10_values = list([{None : 3}]) # min_dominant_class_ratio_for_secondary_override_table - just leave {None : 3} for now
arg11_values = list(range(0, 20, 5)) # min_dominant_class_classifications_above_threshold_for_other_smoothing - int 3 - can be anything
arg12_values = list(range(0, 20, 5)) # min_dominant_class_classifications_above_threshold_for_unclassified_smoothing - int 3 - can be anything
arg13_values = decimal_range(0, 1, 0.2) # flipped_other_confidence_value - float 0.6 - between 0 and 1
arg14_values = decimal_range(0, 1, 0.2) # flipped_class_confidence_value - float 0.6 - between 0 and 1
arg15_values = decimal_range(0, 1, 0.2) # flipped_unclassified_confidence_value - float 0.6 - between 0 and 1
arg16_values = decimal_range(0, 1, 0.2) # min_detection_confidence_for_unclassified_flipping - float 0.15 - between 0 and 1

#############################################
################# FUNCTIONS #################
#############################################

# get Dan's code in here too
sys.path.insert(0, EcoAssist_files)
sys.path.insert(0, os.path.join(EcoAssist_files, "cameratraps"))
from detection.video_utils import frame_results_to_video_results
from md_utils.ct_utils import is_list_sorted
from collections import defaultdict 

# this function smooths json with input args and compares the output to the ground truth
def test_args(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16, print_class_stats = False):
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    data = smooth_json(json_to_smooth, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16)
    sys.stdout = save_stdout
    stat_values  = verify_classification_threshold(data_input = data)
    return stat_values

# this function takes a unprocessed json ("image_recognition_file_original.json")
# and smoothes it based on the input arguments
# output is json data that can be compared to the ground truth
# Dan morris wrote this function
def smooth_json(json_input_fpath,
                min_detections_above_threshold,
                max_detections_secondary_class,
                min_detections_to_overwrite_other,
                classification_confidence_threshold,
                classification_overwrite_threshold,
                detection_confidence_threshold,
                detection_overwrite_threshold,
                min_dominant_class_classifications_above_threshold_for_class_smoothing,
                max_secondary_class_classifications_above_threshold_for_class_smoothing,
                min_dominant_class_ratio_for_secondary_override_table,
                min_dominant_class_classifications_above_threshold_for_other_smoothing,
                min_dominant_class_classifications_above_threshold_for_unclassified_smoothing,
                flipped_other_confidence_value,
                flipped_class_confidence_value,
                flipped_unclassified_confidence_value,
                min_detection_confidence_for_unclassified_flipping):
                   
    # init vars
    filename_base = os.path.normpath(os.path.dirname(json_input_fpath))
    classification_detection_files = [json_input_fpath]
    overflow_folder_handling_enabled = False

    # check if user assigned other and non-other categories
    global other_category_names
    global non_other_category_names
    other_category_names_assigned = False
    non_other_category_names_assigned = False
    if 'other_category_names' in vars() or 'other_category_names' in globals():
        other_category_names_assigned = True
    if 'non_other_category_names' in vars() or 'non_other_category_names' in globals():
        non_other_category_names_assigned = True

    # if user has not assigned values to other_category_names and non_other_category_names themselves, we'll try to
    # automatically distille the other category
    if other_category_names_assigned == False or non_other_category_names_assigned == False:
        with open(json_input_fpath,'r') as f:
            d = json.load(f)
            categories = list(d['classification_categories'].values())
            if 'other' not in categories:
                other_category_names = []
                non_other_category_names = categories
                print(f"<EA>Warning: category 'other' not present in json file. The variables other_category_names"
                    " and non_other_category_names also not assigned in EcoAssist\smooth_params.py. Will not"
                    " perform 'other'-smoothing, but will proceed with classification and sequence smoothing"
                    " as usual.<EA>")
            else:
                other_category_names = ['other']
                categories.remove('other')
                non_other_category_names = categories

    smoothed_classification_files = []
    for final_output_path in classification_detection_files:
        classifier_output_path = final_output_path
        classifier_output_path_within_image_smoothing = classifier_output_path.replace(
            '.json','_within_image_smoothing.json')
        with open(classifier_output_path,'r') as f:
            d = json.load(f)
        category_name_to_id = {d['classification_categories'][k]:k for k in d['classification_categories']}
        other_category_ids = []
        for s in other_category_names:
            if s in category_name_to_id:
                other_category_ids.append(category_name_to_id[s])
            else:
                print('<EA>Warning: "other" category {} not present in file {}<EA>'.format(
                    s,classifier_output_path))
        n_other_classifications_changed = 0
        n_other_images_changed = 0
        n_detections_flipped = 0
        n_images_changed = 0
        
        # Before we do anything else, get rid of everything but the top classification for each detection.
        for im in d['images']:
            if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
                continue
            detections = im['detections']
            for det in detections:
                if 'classifications' not in det or len(det['classifications']) == 0:
                    continue
                classification_confidence_values = [c[1] for c in det['classifications']]
                assert is_list_sorted(classification_confidence_values,reverse=True)
                det['classifications'] = [det['classifications'][0]]
            # ...for each detection in this image
        # ...for each image
        
        for im in tqdm(d['images']):
            if 'detections' not in im or im['detections'] is None or len(im['detections']) == 0:
                continue
            detections = im['detections']
            category_to_count = defaultdict(int)
            for det in detections:
                if ('classifications' in det) and (det['conf'] >= detection_confidence_threshold):
                    for c in det['classifications']:
                        if c[1] >= classification_confidence_threshold:
                            category_to_count[c[0]] += 1
                    # ...for each classification
                # ...if there are classifications for this detection
            # ...for each detection
                            
            if len(category_to_count) <= 1:
                continue
            category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                        key=lambda item: item[1], 
                                                        reverse=True)}
            keys = list(category_to_count.keys())
            
            # Handle a quirky special case: if the most common category is "other" and 
            # it's "tied" with the second-most-common category, swap them
            if (len(keys) > 1) and \
                (keys[0] in other_category_ids) and \
                (keys[1] not in other_category_ids) and \
                (category_to_count[keys[0]] == category_to_count[keys[1]]):
                    keys[1], keys[0] = keys[0], keys[1]
            
            max_count = category_to_count[keys[0]]
            # secondary_count = category_to_count[keys[1]]
            # The 'secondary count' is the most common non-other class
            secondary_count = 0
            for i_key in range(1,len(keys)):
                if keys[i_key] not in other_category_ids:
                    secondary_count = category_to_count[keys[i_key]]
                    break
            most_common_category = keys[0]
            assert max_count >= secondary_count
            
            # If we have at least *min_detections_to_overwrite_other* in a category that isn't
            # "other", change all "other" classifications to that category
            if max_count >= min_detections_to_overwrite_other and \
                most_common_category not in other_category_ids:
                other_change_made = False
                for det in detections:
                    if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold): 
                        for c in det['classifications']:                
                            if c[1] >= classification_overwrite_threshold and \
                                c[0] in other_category_ids:
                                n_other_classifications_changed += 1
                                other_change_made = True
                                c[0] = most_common_category
                        # ...for each classification
                    # ...if there are classifications for this detection
                # ...for each detection
                
                if other_change_made:
                    n_other_images_changed += 1
            # ...if we should overwrite all "other" classifications
        
            if max_count < min_detections_above_threshold:
                continue
            if secondary_count >= max_detections_secondary_class:
                continue
            
            # At this point, we know we have a dominant category; change all other above-threshold
            # classifications to that category.  That category may have been "other", in which
            # case we may have already made the relevant changes.
            n_detections_flipped_this_image = 0
            for det in detections:
                if ('classifications' in det) and (det['conf'] >= detection_overwrite_threshold):
                    for c in det['classifications']:
                        if c[1] >= classification_overwrite_threshold and \
                            c[0] != most_common_category:
                            c[0] = most_common_category
                            n_detections_flipped += 1
                            n_detections_flipped_this_image += 1
                    # ...for each classification
                # ...if there are classifications for this detection
            # ...for each detection
            
            if n_detections_flipped_this_image > 0:
                n_images_changed += 1
        # ...for each image    
        
        print('<EA>Classification smoothing: changed {} detections on {} images<EA>'.format(
            n_detections_flipped,n_images_changed))
        print('<EA>"Other" smoothing: changed {} detections on {} images<EA>'.format(
            n_other_classifications_changed,n_other_images_changed))
        with open(classifier_output_path_within_image_smoothing,'w') as f:
            json.dump(d,f,indent=1)
        print('Wrote results to:\n{}'.format(classifier_output_path_within_image_smoothing))
        smoothed_classification_files.append(classifier_output_path_within_image_smoothing)
    # ...for each file we want to smooth

    #% Read EXIF data from all images
    from data_management import read_exif
    exif_options = read_exif.ReadExifOptions()
    exif_options.verbose = False
    # exif_options.n_workers = default_workers_for_parallel_tasks
    # exif_options.use_threads = parallelization_defaults_to_threads
    exif_options.processing_library = 'pil'
    exif_options.byte_handling = 'delete'
    exif_results_file = os.path.join(filename_base,'exif_data.json')
    if os.path.isfile(exif_results_file):
        print('Reading EXIF results from {}'.format(exif_results_file))
        with open(exif_results_file,'r') as f:
            exif_results = json.load(f)
    else:        
        exif_results = read_exif.read_exif_from_folder(filename_base,
                                                    output_file=exif_results_file,
                                                    options=exif_options)

    #% Prepare COCO-camera-traps-compatible image objects for EXIF results
    import datetime    
    from data_management.read_exif import parse_exif_datetime_string
    min_valid_timestamp_year = 2000
    now = datetime.datetime.now()
    image_info = []
    images_without_datetime = []
    images_with_invalid_datetime = []
    exif_datetime_tag = 'DateTimeOriginal'
    for exif_result in tqdm(exif_results):
        im = {}

        # By default we assume that each leaf-node folder is a location
        if overflow_folder_handling_enabled:
            im['location'] = relative_path_to_location(os.path.dirname(exif_result['file_name']))
        else:
            im['location'] = os.path.dirname(exif_result['file_name'])

        im['file_name'] = exif_result['file_name']
        im['id'] = im['file_name']
        if ('exif_tags' not in exif_result) or (exif_result['exif_tags'] is None) or \
            (exif_datetime_tag not in exif_result['exif_tags']): 
            exif_dt = None
        else:
            exif_dt = exif_result['exif_tags'][exif_datetime_tag]
            exif_dt = parse_exif_datetime_string(exif_dt)
        if exif_dt is None:
            im['datetime'] = None
            images_without_datetime.append(im['file_name'])
        else:
            dt = exif_dt
            
            # An image from the future (or within the last hour) is invalid
            if (now - dt).total_seconds() <= 1*60*60:
                print('<EA>Warning: datetime for {} is {}<EA>'.format(
                    im['file_name'],dt))
                im['datetime'] = None            
                images_with_invalid_datetime.append(im['file_name'])
            
            # An image from before the dawn of time is also invalid
            elif dt.year < min_valid_timestamp_year:
                print('<EA>Warning: datetime for {} is {}<EA>'.format(
                    im['file_name'],dt))
                im['datetime'] = None
                images_with_invalid_datetime.append(im['file_name'])
            
            else:
                im['datetime'] = dt
        image_info.append(im)
    # ...for each exif image result

    print('<EA>Parsed EXIF datetime information, unable to parse EXIF data from {} of {} images<EA>'.format(
        len(images_without_datetime),len(exif_results)))

    #% Assemble into sequences
    from data_management import cct_json_utils
    print('Assembling images into sequences')
    save_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    cct_json_utils.create_sequences(image_info)
    sys.stdout = save_stdout

    # Make a list of images appearing at each location
    sequence_to_images = defaultdict(list)
    for im in tqdm(image_info):
        sequence_to_images[im['seq_id']].append(im)
    all_sequences = list(sorted(sequence_to_images.keys()))

    #% Load classification results
    sequence_level_smoothing_input_file = smoothed_classification_files[0]
    with open(sequence_level_smoothing_input_file,'r') as f:
        d = json.load(f)

    # Map each filename to classification results for that file
    filename_to_results = {}
    for im in tqdm(d['images']):
        filename_to_results[im['file'].replace('\\','/')] = im

    #% Smooth classification results over sequences (prep)
    classification_category_id_to_name = d['classification_categories']
    classification_category_name_to_id = {v: k for k, v in classification_category_id_to_name.items()}
    class_names = list(classification_category_id_to_name.values())
    animal_detection_category = '1'
    assert(d['detection_categories'][animal_detection_category] == 'animal')
    other_category_ids = set([classification_category_name_to_id[s] for s in other_category_names])

    # These are the only classes to which we're going to switch other classifications
    category_names_to_smooth_to = set(non_other_category_names)
    category_ids_to_smooth_to = set([classification_category_name_to_id[s] for s in category_names_to_smooth_to])
    assert all([s in class_names for s in category_names_to_smooth_to])    

    #% Smooth classification results over sequences (supporting functions)
    def results_for_sequence(images_this_sequence):
        """
        Fetch MD results for every image in this sequence, based on the 'file_name' field
        """
        results_this_sequence = []
        for im in images_this_sequence:
            fn = im['file_name']
            results_this_image = filename_to_results[fn]
            assert isinstance(results_this_image,dict)
            results_this_sequence.append(results_this_image)
        return results_this_sequence

    def top_classifications_for_sequence(images_this_sequence):
        """
        Return all top-1 animal classifications for every detection in this 
        sequence, regardless of  confidence

        May modify [images_this_sequence] (removing non-top-1 classifications)
        """
        classifications_this_sequence = []
        for im in images_this_sequence:
            fn = im['file_name']
            results_this_image = filename_to_results[fn]
            if results_this_image['detections'] is None:
                continue
            for det in results_this_image['detections']:
                
                # Only process animal detections
                if det['category'] != animal_detection_category:
                    continue
                
                # Only process detections with classification information
                if 'classifications' not in det:
                    continue
                
                # We only care about top-1 classifications, remove everything else
                if len(det['classifications']) > 1:
                    
                    # Make sure the list of classifications is already sorted by confidence
                    classification_confidence_values = [c[1] for c in det['classifications']]
                    assert is_list_sorted(classification_confidence_values,reverse=True)
                    
                    # ...and just keep the first one
                    det['classifications'] = [det['classifications'][0]]
                    
                # Confidence values should be sorted within a detection; verify this, and ignore 
                top_classification = det['classifications'][0]
                classifications_this_sequence.append(top_classification)
            # ...for each detection in this image
        # ...for each image in this sequence
        return classifications_this_sequence
    # ...top_classifications_for_sequence()


    def count_above_threshold_classifications(classifications_this_sequence):    
        """
        Given a list of classification objects (tuples), return a dict mapping
        category IDs to the count of above-threshold classifications.
        
        This dict's keys will be sorted in descending order by frequency.
        """
        
        # Count above-threshold classifications in this sequence
        category_to_count = defaultdict(int)
        for c in classifications_this_sequence:
            if c[1] >= classification_confidence_threshold:
                category_to_count[c[0]] += 1
        
        # Sort the dictionary in descending order by count
        category_to_count = {k: v for k, v in sorted(category_to_count.items(),
                                                    key=lambda item: item[1], 
                                                    reverse=True)}
        
        keys_sorted_by_frequency = list(category_to_count.keys())
            
        # Handle a quirky special case: if the most common category is "other" and 
        # it's "tied" with the second-most-common category, swap them.
        if len(other_category_names) > 0:
            if (len(keys_sorted_by_frequency) > 1) and \
                (keys_sorted_by_frequency[0] in other_category_names) and \
                (keys_sorted_by_frequency[1] not in other_category_names) and \
                (category_to_count[keys_sorted_by_frequency[0]] == \
                category_to_count[keys_sorted_by_frequency[1]]):
                    keys_sorted_by_frequency[1], keys_sorted_by_frequency[0] = \
                        keys_sorted_by_frequency[0], keys_sorted_by_frequency[1]
        sorted_category_to_count = {}    
        for k in keys_sorted_by_frequency:
            sorted_category_to_count[k] = category_to_count[k]
        return sorted_category_to_count
    # ...def count_above_threshold_classifications()
        
    def sort_images_by_time(images):
        """
        Returns a copy of [images], sorted by the 'datetime' field (ascending).
        """
        return sorted(images, key = lambda im: im['datetime'])        

    def get_first_key_from_sorted_dictionary(di):
        if len(di) == 0:
            return None
        return next(iter(di.items()))[0]

    def get_first_value_from_sorted_dictionary(di):
        if len(di) == 0:
            return None
        return next(iter(di.items()))[1]

    #% Smooth classifications at the sequence level (main loop)
    n_other_flips = 0
    n_classification_flips = 0
    n_unclassified_flips = 0

    # Break if this token is contained in a filename (set to None for normal operation)
    debug_fn = None
    for i_sequence,seq_id in tqdm(enumerate(all_sequences),total=len(all_sequences)):
        images_this_sequence = sequence_to_images[seq_id]
        
        # Count top-1 classifications in this sequence (regardless of confidence)
        classifications_this_sequence = top_classifications_for_sequence(images_this_sequence)
        
        # Handy debugging code for looking at the numbers for a particular sequence
        for im in images_this_sequence:
            if debug_fn is not None and debug_fn in im['file_name']:
                raise ValueError('')
        if len(classifications_this_sequence) == 0:
            continue
        
        # Count above-threshold classifications for each category
        sorted_category_to_count = count_above_threshold_classifications(classifications_this_sequence)
        if len(sorted_category_to_count) == 0:
            continue
        
        max_count = get_first_value_from_sorted_dictionary(sorted_category_to_count)    
        dominant_category_id = get_first_key_from_sorted_dictionary(sorted_category_to_count)
        
        # If our dominant category ID isn't something we want to smooth to, don't mess around with this sequence
        if dominant_category_id not in category_ids_to_smooth_to:
            continue
        
        ## Smooth "other" classifications ##
        if max_count >= min_dominant_class_classifications_above_threshold_for_other_smoothing:        
            for c in classifications_this_sequence:           
                if c[0] in other_category_ids:
                    n_other_flips += 1
                    c[0] = dominant_category_id
                    c[1] = flipped_other_confidence_value

        # By not re-computing "max_count" here, we are making a decision that the count used
        # to decide whether a class should overwrite another class does not include any "other"
        # classifications we changed to be the dominant class.  If we wanted to include those...
        # 
        # sorted_category_to_count = count_above_threshold_classifications(classifications_this_sequence)
        # max_count = get_first_value_from_sorted_dictionary(sorted_category_to_count)    
        # assert dominant_category_id == get_first_key_from_sorted_dictionary(sorted_category_to_count)
        
        ## Smooth non-dominant classes ##
        if max_count >= min_dominant_class_classifications_above_threshold_for_class_smoothing:
            
            # Don't flip classes to the dominant class if they have a large number of classifications
            category_ids_not_to_flip = set()
            for category_id in sorted_category_to_count.keys():
                secondary_class_count = sorted_category_to_count[category_id]
                dominant_to_secondary_ratio = max_count / secondary_class_count
                
                # Don't smooth over this class if there are a bunch of them, and the ratio
                # if primary to secondary class count isn't too large
                
                # Default ratio
                ratio_for_override = min_dominant_class_ratio_for_secondary_override_table[None]
                
                # Does this dominant class have a custom ratio?
                if dominant_category_id in min_dominant_class_ratio_for_secondary_override_table:
                    ratio_for_override = \
                        min_dominant_class_ratio_for_secondary_override_table[dominant_category_id]
                if (dominant_to_secondary_ratio < ratio_for_override) and \
                    (secondary_class_count > \
                    max_secondary_class_classifications_above_threshold_for_class_smoothing):
                    category_ids_not_to_flip.add(category_id)
            for c in classifications_this_sequence:
                if c[0] not in category_ids_not_to_flip and c[0] != dominant_category_id:
                    c[0] = dominant_category_id
                    c[1] = flipped_class_confidence_value
                    n_classification_flips += 1
            
        ## Smooth unclassified detections ##
        if max_count >= min_dominant_class_classifications_above_threshold_for_unclassified_smoothing:
            results_this_sequence = results_for_sequence(images_this_sequence)
            detections_this_sequence = []
            for r in results_this_sequence:
                if r['detections'] is not None:
                    detections_this_sequence.extend(r['detections'])
            for det in detections_this_sequence:
                if 'classifications' in det and len(det['classifications']) > 0:
                    continue
                if det['category'] != animal_detection_category:
                    continue
                if det['conf'] < min_detection_confidence_for_unclassified_flipping:
                    continue

                det['classifications'] = [[dominant_category_id,flipped_unclassified_confidence_value]]
                n_unclassified_flips += 1
                                
    # ...for each sequence    
    print('\nFinished sequence smoothing\n')
    print('<EA>Flipped {} "other" classifications<EA>'.format(n_other_flips))
    print('<EA>Flipped {} species classifications<EA>'.format(n_classification_flips))
    print('<EA>Flipped {} unclassified detections<EA>'.format(n_unclassified_flips))
    
    #% convert data to adjusted label map
    # fetch label maps
    cls_label_map = d['classification_categories']
    det_label_map = d['detection_categories']
    inverted_cls_label_map = {v: k for k, v in cls_label_map.items()}
    inverted_det_label_map = {v: k for k, v in det_label_map.items()}

    # add cls classes to det label map
    for k, v in inverted_cls_label_map.items():
        inverted_det_label_map[k] = str(len(inverted_det_label_map) + 1)

    # loop and adjust
    for image in d['images']:
        if 'detections' in image:
            for detection in image['detections']:
                category_id = detection['category']
                category = det_label_map[category_id]
                if 'classifications' in detection:
                    highest_classification = detection['classifications'][0]
                    class_idx = highest_classification[0]
                    class_name = cls_label_map[class_idx]
                    detec_idx = inverted_det_label_map[class_name]
                    detection['prev_conf'] = detection["conf"]
                    detection['prev_category'] = detection['category']
                    detection["conf"] = highest_classification[1]
                    detection['category'] = str(detec_idx)

    # return smoothened data
    d['detection_categories'] = {v: k for k, v in inverted_det_label_map.items()}
    return d

# get label map from json data
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# # function to compare data with ground truth and calculate stats per class
# def verify_json(data_input, conf_thresh, print_class_stats = False):
#     #  you can input either a path, or a dataset
#     if isinstance(data_input, str):
#         label_map = fetch_label_map_from_json(data_input)
#         with open(data_input) as json_content:
#             data = json.load(json_content)
#     else:
#         data = data_input
#         label_map = data['detection_categories']
    
#     # init 
#     tps = initialize_class_dict()
#     fps = initialize_class_dict()
#     fns = initialize_class_dict()
#     prs = initialize_class_dict()
#     rcs = initialize_class_dict()

#     # loop
#     for image in data['images']:
#         counts = {}
#         file = image['file']
#         # print("debug 1")
#         if 'detections' in image:
#             # print("debug 2")
#             for detection in image['detections']:
#                 # print("debug 3")
#                 conf = detection["conf"]
#                 cat = label_map[detection['category']]

#                 # print(f"cat         : {cat}")
#                 # print(f"conf        : {conf}")
#                 # print(f"conf_thresh : {conf_thresh}")

#                 # don't do unclassified classes
#                 if cat in ['animal', 'person', 'vehicle']: 
#                     continue
                


#                 if conf >= conf_thresh:
#                     if cat in counts:
#                         counts[cat] += 1
#                     else:
#                         counts[cat] = 1
            
#             # determine the classes to compare
#             real_dict = ground_truth[file]
#             # print("real_dict : ", real_dict)
#             pred_dict = counts
#             # print("pred_dict : ", pred_dict)
#             cats = list(real_dict.keys()) + list(pred_dict.keys())
#             cats = list(set(cats))

#             # calculate true and false positives and negatives
#             for cat in cats:
#                 real_n = real_dict[cat] if cat in real_dict else 0
#                 pred_n = pred_dict[cat] if cat in pred_dict else 0

#                 # all well predicted for this species
#                 if real_n == pred_n:
#                     tp = real_n
#                     fp = 0
#                     fn = 0
                
#                 # more predicted than actually real
#                 elif real_n < pred_n:
#                     tp = real_n
#                     fp = pred_n - real_n
#                     fn = 0
                
#                 # some animals were not identified
#                 elif real_n > pred_n:
#                     tp = pred_n
#                     fp = 0
#                     fn = real_n - pred_n

#                 # add to totals
#                 tps[cat] += tp
#                 fps[cat] += fp
#                 fns[cat] += fn

#     # calculate precision and recall for each class
#     prs_list = []
#     rcs_list = []
#     for cat in classes_list:
#         try:
#             pr = tps[cat] / (tps[cat] + fps[cat])
#         except ZeroDivisionError:
#             pr = None

#         try:
#             rc = tps[cat] / (tps[cat] + fns[cat])
#         except ZeroDivisionError:
#             rc = None

#         prs_list.append(pr)
#         rcs_list.append(rc)
    
#     # print(prs_list) # DEBUG
#     # print(rcs_list)

#     try: 
#         ave_pr = sum(filter(None, prs_list)) / len(list(filter(None, prs_list)))
#         ave_rc = sum(filter(None, rcs_list)) / len(list(filter(None, rcs_list)))
#     except ZeroDivisionError:
#         return ["unknown"] * 2

#     return [ave_pr, ave_rc, *prs_list, *rcs_list]

# # function to compare data with ground truth and calculate stats per class
# def verify_classification_threshold(data_input, conf_thresh, print_class_stats = False):
#     #  you can input either a path, or a dataset
#     if isinstance(data_input, str):
#         label_map = fetch_label_map_from_json(data_input)
#         with open(data_input) as json_content:
#             data = json.load(json_content)
#     else:
#         data = data_input
#         label_map = data['detection_categories']
    
#     # init 
#     tps = initialize_class_dict()
#     fps = initialize_class_dict()
#     fns = initialize_class_dict()
#     prs = initialize_class_dict()
#     rcs = initialize_class_dict()

#     # loop
#     # n_classified = 0
#     # n_unclassified = 0
#     for image in data['images']:
#         counts = {}
#         file = image['file']
#         if 'detections' in image:
#             for detection in image['detections']:
#                 cat = label_map[detection['category']]

#                 # only interested in classified animals
#                 if cat in ['animal', 'person', 'vehicle']: 
#                     continue

#                 detection_conf = detection["prev_conf"]
#                 classification_conf = detection["conf"]


#                 # print(f"cat                 : {cat}")
#                 # print(f"detection_conf      : {detection_conf}")
#                 # print(f"classification_conf : {classification_conf}")
#                 # print("")


#                 # only process detection if it meets the threshold
#                 if detection_conf < conf_thresh:
#                     continue
#                 else:
#                     conf = classification_conf

#                 # if conf >= conf_thresh:
#                 if cat in counts:
#                     counts[cat] += 1
#                 else:
#                     counts[cat] = 1
            
#             # determine the classes to compare
#             if file in ground_truth:
#                 real_dict = ground_truth[file]
#             else:
#                 real_dict = {}
#             print("")
#             print("real_dict : ", real_dict)
#             pred_dict = counts
#             print("pred_dict : ", pred_dict)
#             cats = list(real_dict.keys()) + list(pred_dict.keys())
#             cats = list(set(cats))
#             print(cats)
#             # exit()

#             # calculate true and false positives and negatives
#             for cat in cats:
#                 real_n = real_dict[cat] if cat in real_dict else 0
#                 pred_n = pred_dict[cat] if cat in pred_dict else 0

#                 # all well predicted for this species
#                 if real_n == pred_n:
#                     tp = real_n
#                     fp = 0
#                     fn = 0
                
#                 # more predicted than actually real
#                 elif real_n < pred_n:
#                     tp = real_n
#                     fp = pred_n - real_n
#                     fn = 0
                
#                 # some animals were not identified
#                 elif real_n > pred_n:
#                     tp = pred_n
#                     fp = 0
#                     fn = real_n - pred_n

#                 # add to totals
#                 print("")
#                 print(cat)
#                 print(f"tp : {tp}")
#                 print(f"fp : {fp}")
#                 print(f"fn : {fn}")
#                 tps[cat] += tp
#                 fps[cat] += fp
#                 fns[cat] += fn

#     # calculate precision and recall for each class
#     prs_list = []
#     rcs_list = []
#     for cat in classes_list:
#         try:
#             pr = tps[cat] / (tps[cat] + fps[cat])
#         except ZeroDivisionError:
#             pr = None

#         try:
#             rc = tps[cat] / (tps[cat] + fns[cat])
#         except ZeroDivisionError:
#             rc = None

#         prs_list.append(pr)
#         rcs_list.append(rc)
    
#     # print(prs_list) # DEBUG
#     # print(rcs_list)

#     # return [n_unclassified, n_classified]


#     try: 
#         ave_pr = sum(filter(None, prs_list)) / len(list(filter(None, prs_list)))
#         ave_rc = sum(filter(None, rcs_list)) / len(list(filter(None, rcs_list)))
#     except ZeroDivisionError:
#         return

#     return [ave_pr, ave_rc, *prs_list, *rcs_list]

# function to compare data with ground truth and calculate stats per class
def verify_classification_threshold(data_input, conf_thresh = 0.5, print_class_stats = False, ground_truth_data = False):
    #  you can input either a path, or a dataset
    if isinstance(data_input, str):
        label_map = fetch_label_map_from_json(data_input)
        with open(data_input) as json_content:
            data = json.load(json_content)
    else:
        data = data_input
        label_map = data['detection_categories']
    
    # init 
    tps = initialize_class_dict()
    fps = initialize_class_dict()
    fns = initialize_class_dict()
    prs = initialize_class_dict()
    rcs = initialize_class_dict()

    # loop
    for image in data['images']:
        counts = {}
        file = image['file']
        if 'detections' in image:
            for detection in image['detections']:
                cat = label_map[detection['category']]

                # only interested in classified animals
                if cat in ['animal', 'person', 'vehicle']: 
                    continue

                # TODO: volgens mij is er helemaal geen ground_truth json meer. Dit is een lijst met observaties. Geen json. Moet dit dan nog wel?
                # the json of the ground truth data look different than the processed data
                # ground truth detections don't have "prev_conf", "prev_detection", or "classifications"
                # hence they need to be handled differently
                # if it is not ground truth data, it will check if it meets the threshold
                # for ground truth, all passes the trehshold because everything is human verified
                if not ground_truth_data:
                    detection_conf = detection["prev_conf"]
                    classification_conf = detection["conf"]
                    if detection_conf < conf_thresh:
                        continue
                    else:
                        conf = classification_conf
                else: conf = detection["conf"]

                # keep track of the number of predictions per animal
                if cat in counts:
                    counts[cat] += 1
                else:
                    counts[cat] = 1
            

            # # determine the classes to compare
            # if file in ground_truth:
            #     real_dict = ground_truth[file]
            # else:
            #     real_dict = {}
            # pred_dict = counts
            # cats = list(set(list(real_dict.keys()) + list(pred_dict.keys())))

            # # TODO: het gaat toch niet goed hier.... er blijven false positives over... Op het moment dat er bijvoorbeeld 10 cattle word voorspeld, maar er is maar 1 zebra, dan krijgt alle class zebra minpunten. Cattle krijgt helemaal niets... Je moet dus op het moment dat je iets een fp of fn maakt, dat van de dict afhalen, en de rest wordt dan fp voor de overgebleven class. 

            # # calculate true and negative positives etc for each class
            # for cat in cats:
            #     tp = min(real_dict.get(cat, 0), pred_dict.get(cat, 0))
            #     fp = max(0, pred_dict.get(cat, 0) - tp)
            #     fn = max(0, real_dict.get(cat, 0) - tp)
                
            #     tps[cat] += tp
            #     fps[cat] += fp
            #     fns[cat] += fn
            #     print("")
            #     print(f"cat       : {cat}")
            #     print(f"real_dict : {real_dict}")
            #     print(f"pred_dict : {pred_dict}")
            #     print(f"cats      : {cats}")
            #     print(f"tp        : {tp}")
            #     print(f"fp        : {fp}")
            #     print(f"fn        : {fn}")

            # determine the classes to compare
            pred_dict = counts
            if file in ground_truth:
                real_dict = ground_truth[file]
            else:
                real_dict = {}
                pred_dict = {}
            animals = list(set(list(real_dict.keys()) + list(pred_dict.keys())))

            # predicted_categories = list(set(list(pred_dict.keys())))
            # real_categories = list(set(list(real_dict.keys())))

            # cats = list(set(list(real_dict.keys()) + list(pred_dict.keys())))

            # TODO: het gaat toch niet goed hier.... er blijven false positives over... Op het moment dat er bijvoorbeeld 10 cattle word voorspeld, maar er is maar 1 zebra, dan krijgt alle class zebra minpunten. Cattle krijgt helemaal niets... Je moet dus op het moment dat je iets een fp of fn maakt, dat van de dict afhalen, en de rest wordt dan fp voor de overgebleven class. 

            # calculate true and negative positives etc for each class
            # for cat in cats:
            # n_unaccounted = 0
            # for cat in real_dict.keys():
            for animal in animals:
                # print("")
                # print(animal)
                real_count = real_dict.get(animal, 0)
                pred_count = pred_dict.get(animal, 0)
                tp = min(real_count, pred_count)
                fp = pred_count - tp
                fn = real_count - tp



                # tp = min(real_dict.get(cat, 0), pred_dict.get(cat, 0))
                # fp = max(0, pred_dict.get(cat, 0) - tp)
                # fn = max(0, real_dict.get(cat, 0) - tp)
                # if tp > 0:
                #     pred_dict[cat] -= tp
                # if fn > 0:

                tps[animal] += tp
                fps[animal] += fp
                fns[animal] += fn
                # print(f"tp        : {tp}")
                # print(f"fp        : {fp}")
                # print(f"fn        : {fn}")
                # print(f"real_dict : {real_dict}")
                # print(f"pred_dict : {pred_dict}")

    # calculate precision and recall for each class
    prs_list = []
    rcs_list = []
    tps_tot = 0
    fps_tot = 0
    fns_tot = 0
    for cat in classes_list:
        tps_tot += tps[cat]
        fps_tot += fps[cat]
        fns_tot += fns[cat]
        pr = tps[cat] / (tps[cat] + fps[cat]) if (tps[cat] + fps[cat]) > 0 else 0
        rc = tps[cat] / (tps[cat] + fns[cat]) if (tps[cat] + fns[cat]) > 0 else 0
        prs_list.append(pr)
        rcs_list.append(rc)
        # print("")
        # print(f"cat                 : {cat}")
        # print(f"pr = tp / (tp + fp) : {pr}")
        # print(f"tp                  : {tps[cat]}")
        # print(f"fp                  : {fps[cat]}")
        # print(f"pr                  : {pr}")
    
    # # caluclate average precision and recall values for this calibration
    # try: 
    #     ave_pr = sum(filter(None, prs_list)) / len(list(filter(None, prs_list)))
    #     ave_rc = sum(filter(None, rcs_list)) / len(list(filter(None, rcs_list)))
    # except ZeroDivisionError:
    #     return

    # If you want to include the O, you'll have to uncomment these lines
    ave_pr = sum(prs_list) / len(prs_list)
    ave_rc = sum(rcs_list) / len(rcs_list)

    # TODO: calculate the avarages with the total tps, fps, fns. That way you don't give equal weight to every class, but if there are more gemsbokken, they have more weight.
    ave_pr = tps_tot / (tps_tot + fps_tot) if (tps_tot + fps_tot) > 0 else 0
    ave_rc = tps_tot / (tps_tot + fns_tot) if (tps_tot + fns_tot) > 0 else 0

    # return
    return [ave_pr, ave_rc, *prs_list, *rcs_list]

# plot mean, precision and recall values and annotate optimum
plot_list = []
def plot_multiple_optima(x, ys, xlabel, ylabels):
    x = np.array(x)
    for i, y in enumerate(ys):
        y = np.array(y)
        xmax, ymax = fetch_optimum_or_default(x = x, y = y, arg_name = xlabel)
        if ylabels[i] == 'average':
            text= "max (x={:.2f}, y={:.2f})".format(xmax, ymax)
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
            kw = dict(xycoords='data',textcoords="axes fraction",
                arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
            plt.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
        xnew = np.linspace(x.min(), x.max(), 300) 
        gfg = make_interp_spline(x, y, k=3) 
        y_new = gfg(xnew) 
        plt.plot(xnew, y_new, label = ylabels[i])
    plt.legend(loc="lower left")
    plt.xlabel(xlabel)
    plt.savefig(os.path.join(export_dir, xlabel + '.png'))
    plt.close()
    plot_list.append(plt)
    return [xmax, ymax]

# get an empty detections dictionary to be filled
def initialize_class_dict():
    # global classes_list

    label_map = fetch_label_map_from_json(json_no_smooth)
    detections_init = {}
    for cat in label_map.values():
    # for cat in classes_list: DEBUG
        detections_init[cat] = 0
    # for key, value in label_map_init.items(): DEBUG
    #     detections_init[value] = 0

    # we're not interested in the MD classes
    for cat in ['animal', 'person', 'vehicle']:
        if cat in detections_init:
            del detections_init[cat]

    return detections_init

# # DEBUG create a dict with a list of detections per image
# def set_ground_truth():
#     json_fpath = ground_truth_json
#     label_map = fetch_label_map_from_json(json_fpath) # DEBUG
#     # label_map = fetch_label_map_from_json(r"C:\Peter\desert-lion-project\calibration_set")
#     ground_truth = {}
#     with open(json_fpath) as json_content:
#         data = json.load(json_content)
#     for image in data['images']:
#         counts = {}
#         if 'detections' in image:
#             for detection in image['detections']:
#                 conf = detection["conf"]
#                 cat = label_map[detection['category']]

#                 # we're not interested in the MD classes
#                 if cat in ['animal', 'person', 'vehicle']:
#                     continue

#                 if cat in counts:
#                     counts[cat] += 1
#                 else:
#                     counts[cat] = 1
#             ground_truth[image['file']] = counts
#     return ground_truth

# input two lists and return the optima
def fetch_optimum_or_default(x, y, arg_name):
    x = np.array(x)
    y = np.array(y)

    # if there is a default value
    if arg_name in defaults:
        default_value = defaults[arg_name]

        # get all x values where y is max
        y_max = y.max()
        x_max_list = x[np.where(y == y_max)]

        # if the default x is amongst the best values, take that
        if any(np.isin(x_max_list, default_value)):
            x_max = default_value

        # otherwise just take the first of all the optima
        else:
            x_max = x_max_list[0]
    
    # perhaps we're trying to get the maximum for the arg10 calculations
    # in that case the arg_name is a class, like "lion"
    elif arg_name in classes_list:

        default_value = default_value_arg10

        # get all x values where y is max
        y_max = y.max()
        x_max_list = x[np.where(y == y_max)]

        # if the default x is amongst the best values, take that
        if any(np.isin(x_max_list, default_value)):
            x_max = default_value

        # otherwise just take the first of all the optima
        else:
            x_max = x_max_list[0]
    
    # if there is no default value available
    else:
        x_max = x[np.argmax(y)]
        y_max = y.max()

    # return
    return [float(x_max), float(y_max)]

# this function will loop though values per class for argument 10
def alter_argument10(range_list, pbar):

    # loop though values and calulate statistics
    for k in optimal_values['arg10'].keys():

        # don't try different options for "None"
        if k == None:
            continue

        # init vars
        x = []
        pr_values = []
        rc_values = []
        ave_values = []

        # loop
        for value in range_list:
            optimal_values['arg10'][k] = value
            result = test_args(*optimal_values.values())
            if result:
                pr, rc = result[:2]
                ave = (pr + rc) / 2
                x.append(value)
                pr_values.append(pr)
                rc_values.append(rc)
                ave_values.append(ave)
            pbar.update(1)

        # plot results and save as png
        xmax, ymax = plot_multiple_optima(x, [ave_values],
                                            id2cat[k], ["average"])

        # save new optimum
        optimal_values['arg10'][k] = xmax
        
        # update pbar
        pbar.set_description(f"{id2cat[k]} (x = {xmax:.2f}, y = {ymax:.2f})")

# function to loop though values of a single argument
def alter_single_argument(arg_values, arg_name, pbar):
    # init vars
    x = []
    pr_values = []
    rc_values = []
    ave_values = []

    # loop though values and calulate statistics
    for arg in arg_values:
        optimal_values[arg_name] = arg
        result = test_args(*optimal_values.values())
        if result:
            pr, rc = result[:2]
            ave = (pr + rc) / 2
            x.append(arg)
            pr_values.append(pr)
            rc_values.append(rc)
            ave_values.append(ave)
        pbar.update(1)

    # plot results and save as png
    xmax, ymax = plot_multiple_optima(x, [ave_values],
                                        arg_name,
                                        ["average"])
    # save new optimum
    optimal_values[arg_name] = xmax
    
    # update pbar
    pbar.set_description(f"{arg_name} (x = {xmax:.2f}, y = {ymax:.2f})")

# loop through thresholds to find optimum threshold
def find_optimal_threshold(data, range_list, name):
    x = []
    pr_values = []
    rc_values = []
    ave_values = []
    for conf_thresh in tqdm_copy(range_list):
        result = verify_classification_threshold(data, conf_thresh = conf_thresh)
        if result:
            pr, rc = result[:2]
            ave = (pr + rc) / 2
            x.append(conf_thresh)
            pr_values.append(pr)
            rc_values.append(rc)
            ave_values.append(ave)
    plot_multiple_optima(x, [pr_values, rc_values, ave_values],
                         name, ["precision", "recall", "average"])

# # DEBUG read info from xml files and create ground truth dataset that can be compared to
# def create_ground_truth_data_from_xmls():

#     # init vars
#     label2id = {}
#     label_idx = 1
#     xml_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(calibration_jsons_dir) for f in filenames if os.path.splitext(f)[1] == '.xml']
#     ground_truth = {"images": [], "detection_categories": {}}

#     for xml_file in xml_files:

#         # read  xml
#         ann_tree = ET.parse(xml_file)
#         ann_root = ann_tree.getroot()

#         # get image path
#         img_ext = os.path.splitext(ann_root.findtext('filename'))[1]
#         img_full_path = os.path.splitext(xml_file)[0] + img_ext
#         img_rela_path = img_full_path.replace(os.path.normpath(calibration_jsons_dir) + os.sep, "")

#         # init image info
#         image_dict = {"file": img_rela_path,
#                     "detections" : []}

#         # loop through xml detections 
#         for obj in ann_root.findall('object'):
#             label = obj.findtext('name')
#             if label not in label2id:
#                 label2id[label] = str(label_idx)
#                 label_idx += 1
#             category_id = str(label2id[label])
#             detection_dict = {"category" : category_id, "conf" : 1.0}
#             image_dict["detections"].append(detection_dict)

#         # append imag info
#         ground_truth["images"].append(image_dict)
    
#     # add label map
#     ground_truth["detection_categories"] = {v: k for k, v in label2id.items()}

#     # return result
#     return ground_truth

# read info from xml files and create ground truth dataset that can be compared to
def set_ground_truth_and_classes_list():

    # init vars
    xml_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(calibration_jsons_dir) for f in filenames if os.path.splitext(f)[1] == '.xml']
    ground_truth = {}
    classes_list = []

    for xml_file in xml_files:

        # read  xml
        ann_tree = ET.parse(xml_file)
        ann_root = ann_tree.getroot()

        # get image path
        img_ext = os.path.splitext(ann_root.findtext('filename'))[1]
        img_full_path = os.path.splitext(xml_file)[0] + img_ext
        img_rela_path = img_full_path.replace(os.path.normpath(calibration_jsons_dir) + os.sep, "")

        # init image info
        counts = {}

        # loop through xml detections 
        for obj in ann_root.findall('object'):
            label = obj.findtext('name')

            # covert to new label if specified
            if label in convert_classes:
                label = convert_classes[label]

            # we're not interested in the MD classes
            if label in ['animal', 'person', 'vehicle']:
                continue

            # fill classes list
            if label not in classes_list:
                classes_list.append(label)

            # fill counts dict
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

            ground_truth[img_rela_path] = counts

    return [ground_truth, classes_list]

# export the counts of ground truth
def export_bar_plot(data):
    total_counts = {}
    for _, counts in data.items():
        for cat, count in counts.items():
            if cat not in total_counts:
                total_counts[cat] = count
            else:
                total_counts[cat] += count
    print(json.dumps(total_counts, indent = 2))
    classes = list(total_counts.keys())
    counts = list(total_counts.values())
    print(f"There are a total of {sum(counts)} annotated instances in the calibration set")
    plt.figure(figsize = (10, 5))
    plt.bar(classes, counts, width = 0.4)
    plt.ylabel("Number of instances in calibration set")
    plt.xticks(rotation = 90)
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(os.path.join(export_dir, 'instances_bar_chart.png'))
    plt.close()

#############################################
################ CONSTANTS ##################
#############################################

# the jsons that it needs to do the computations
json_no_smooth = os.path.join(calibration_jsons_dir, "image_recognition_file.json")
json_to_smooth = os.path.join(calibration_jsons_dir, "image_recognition_file_original.json")
# ground_truth_json = os.path.join(calibration_jsons_dir, "ground-truth.json") 

# the number of options for the heavy looping
n_options_multiple = len(arg1_values) * len(arg2_values) * \
                    len(arg3_values) * len(arg4_values) * \
                    len(arg5_values) * len(arg6_values) * \
                    len(arg7_values) * len(arg8_values) * \
                    len(arg9_values) * len(arg10_values) * \
                    len(arg11_values) * len(arg12_values) * \
                    len(arg13_values) * len(arg14_values) * \
                    len(arg15_values) * len(arg16_values)

# the number of options to do the light looping
n_options_single = len(arg1_values) + len(arg2_values) + \
                    len(arg3_values) + len(arg4_values) + \
                    len(arg5_values) + len(arg6_values) + \
                    len(arg7_values) + len(arg8_values) + \
                    len(arg9_values) + \
                    len(arg11_values) + len(arg12_values) + \
                    len(arg13_values) + len(arg14_values) + \
                    len(arg15_values) + len(arg16_values)

# we'll need a list with values and argument names to loop though
argument_list = [[arg1_values, "arg1"],
                 [arg2_values, "arg2"],
                 [arg3_values, "arg3"],
                 [arg4_values, "arg4"],
                 [arg5_values, "arg5"],
                 [arg6_values, "arg6"],
                 [arg7_values, "arg7"],
                 [arg8_values, "arg8"],
                 [arg9_values, "arg9"],
                 [arg11_values, "arg11"],
                 [arg12_values, "arg12"],
                 [arg13_values, "arg13"],
                 [arg14_values, "arg14"],
                 [arg15_values, "arg15"],
                 [arg16_values, "arg16"]]

#############################################
################### MAIN ####################
#############################################

# create export dir if not present yet
Path(export_dir).mkdir(parents=True, exist_ok=True)

# read xml data into ground truth data
ground_truth, classes_list = set_ground_truth_and_classes_list()

# check how many instances the claibration is based on
export_bar_plot(ground_truth)

# # define the classes were interested in 
# classes_list = list(initialize_class_dict().keys())

# we'll start by altering the defaults and go from there
optimal_values = copy(defaults)

# calibrate the optimal classification threshold without smoothing
find_optimal_threshold(data = json_no_smooth,
                       range_list = decimal_range(0, 1, 0.01),
                       name = "classification thresh without smoothing")

###################### THESE ARE SOME TESTS ############################# START

save_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
default_smooth = smooth_json(json_to_smooth, *defaults.values())
adjusted_smooth = smooth_json(json_to_smooth, *defaults.values())
defaults["arg10"]["aardwolf"] = 4
aardwolf4_smooth = smooth_json(json_to_smooth, *defaults.values())
defaults["arg10"]["aardwolf"] = 150
aardwolf15_smooth = smooth_json(json_to_smooth, *defaults.values())
sys.stdout = save_stdout

print(aardwolf4_smooth == aardwolf15_smooth)
# exit()

import random
random.seed(420)
for image in adjusted_smooth['images']:
    if 'detections' in image:
        for detection in image['detections']:
            if "prev_conf" in detection:
                if random.uniform(0, 1) > 0.1:
                    detection["category"] = "11"

# print("")
# pr, rc  = verify_classification_threshold(ground_truth_json, conf_thresh = 0.20, ground_truth_data = True)[:2]
# # print(f"ground_truth_json  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
# pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.60)[:2]
# print(f"json_no_smooth     - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
# pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.60)[:2]
# print(f"default_smooth     - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
# pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.60)[:2]
# print(f"adjusted_smooth    - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

# exit()

print("")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.01)[:2]
print(f"classification threshold of 0.01 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.1)[:2]
print(f"classification threshold of 0.10 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.3)[:2]
print(f"classification threshold of 0.30 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.6)[:2]
print(f"classification threshold of 0.60 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.95)[:2]
print(f"classification threshold of 0.95 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(json_no_smooth, conf_thresh = 0.98)[:2]
print(f"classification threshold of 0.98 without smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

print("")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.01)[:2]
print(f"classification threshold of 0.01 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.1)[:2]
print(f"classification threshold of 0.10 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.3)[:2]
print(f"classification threshold of 0.30 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.6)[:2]
print(f"classification threshold of 0.60 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.95)[:2]
print(f"classification threshold of 0.95 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(default_smooth, conf_thresh = 0.98)[:2]
print(f"classification threshold of 0.98 with default smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

print("")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.01)[:2]
print(f"classification threshold of 0.01 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.1)[:2]
print(f"classification threshold of 0.10 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.3)[:2]
print(f"classification threshold of 0.30 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.6)[:2]
print(f"classification threshold of 0.60 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.95)[:2]
print(f"classification threshold of 0.95 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf4_smooth, conf_thresh = 0.98)[:2]
print(f"classification threshold of 0.98 with aardwolf4_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

print("")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.01)[:2]
print(f"classification threshold of 0.01 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.1)[:2]
print(f"classification threshold of 0.10 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.3)[:2]
print(f"classification threshold of 0.30 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.6)[:2]
print(f"classification threshold of 0.60 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.95)[:2]
print(f"classification threshold of 0.95 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(aardwolf15_smooth, conf_thresh = 0.98)[:2]
print(f"classification threshold of 0.98 with aardwolf15_smooth  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

print("")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.01)[:2]
print(f"classification threshold of 0.01 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.1)[:2]
print(f"classification threshold of 0.10 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.3)[:2]
print(f"classification threshold of 0.30 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.6)[:2]
print(f"classification threshold of 0.60 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.95)[:2]
print(f"classification threshold of 0.95 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")
pr, rc  = verify_classification_threshold(adjusted_smooth, conf_thresh = 0.98)[:2]
print(f"classification threshold of 0.98 with adjusted smoothing  - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

###################### THESE ARE SOME TESTS ############################# END

# exit()

# check the accuracy of the model with default smoothing values
save_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
default_smooth = smooth_json(json_to_smooth, *defaults.values())
sys.stdout = save_stdout
pr, rc  = verify_classification_threshold(data_input = default_smooth)[:2] # TODO: ik doe dit nu zonder thresh. Doe ik dat wel goed? , conf_thresh = optimal_classification_threshold)[:2]
print(f"Results with default smoothing params are     - {((pr + rc) / 2):.2f} (pr = {pr:.2f}, rc = {rc:.2f})")

# loop each argument separately and keep the rest constant
pbar_single_args = tqdm_copy(total = n_options_single)
for arg_values, arg_name in argument_list:
    alter_single_argument(arg_values, arg_name, pbar_single_args)
print("\nUPDATED ARGUMENTS:")
print(json.dumps(optimal_values, indent = 2))

# # again, but not in reversed order
# pbar_single_args = tqdm_copy(total = n_options_single - 1)
# for arg_values, arg_name in argument_list[::-1]:
#     alter_single_argument(arg_values, arg_name, pbar_single_args)
# print("\nUPDATED ARGUMENTS:")
# print(json.dumps(optimal_values, indent = 2))

# calibrate the optimal classification threshold with the current optimal smoothing parameters
save_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
find_optimal_threshold(data = smooth_json(json_to_smooth, *optimal_values.values()),
                        range_list = decimal_range(0, 1, 0.01),
                        name = "classification thresh with smoothing")
sys.stdout = save_stdout

# argument 10 is a special case, where it takes a dictionary with class specific values
# hence we need a special approach to test this argument
# it will first create a dictionary with all default values, and then loop for each class separatly
# id2cat = {cat: 0 for cat in classes_list} # fetch_label_map_from_json(ground_truth_json)
id2cat = fetch_label_map_from_json(json_no_smooth)
print(id2cat)
# exit()
# cat2id = {v: k for k, v in id2cat.items()}
default_value_arg10 = 3
arg10 = {None : default_value_arg10}
for k in id2cat.keys():
    arg10[k] = default_value_arg10
optimal_values["arg10"] = arg10
range_list = range(0, 20, 1)
pbar = tqdm_copy(total = len(range_list) * len(list(id2cat.keys())))
alter_argument10(range_list, pbar)
print("\nUPDATED ARGUMENTS:")
print(json.dumps(optimal_values, indent = 2))

exit()

#############################################
######## HEAVY FORCE CALC EVERYTHING ########
#############################################

# initialize dict to fill with stats
stats = {}
for key in ["arg1", "arg2", "arg3", "arg4", "arg5", "arg6",
            "arg7", "arg8", "arg9", "arg10", "arg11", "arg12",
            "arg13", "arg14", "arg15", "arg16", "pr_ave", "rc_ave"]:
    stats[key] = []
for typ in ["pr_", "rc_"]:
    for cat in classes_list:
        stats[typ + cat] = []

# Iterate through all combinations of arguments
with tqdm_copy(total=n_options_multiple) as pbar:
    for arg1 in arg1_values:
        for arg2 in arg2_values:
            for arg3 in arg3_values:
                for arg4 in arg4_values:
                    for arg5 in arg5_values:
                        for arg6 in arg6_values:
                            for arg7 in arg7_values:
                                for arg8 in arg8_values:
                                    for arg9 in arg9_values:
                                        for arg10 in arg10_values:
                                            for arg11 in arg11_values:
                                                for arg12 in arg12_values:
                                                    for arg13 in arg13_values:
                                                        for arg14 in arg14_values:
                                                            for arg15 in arg15_values:
                                                                for arg16 in arg16_values:
                                                                    arg_values = [arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15, arg16]
                                                                    stat_values = test_args(*arg_values)
                                                                    stat_values[:0] = arg_values # add arg values to the beginning of list
                                                                    for i, [k, v] in enumerate(stats.items()):
                                                                        values = copy(stats[k])
                                                                        values.append(stat_values[i])
                                                                        stats[k] = copy(values)
                                                                    pbar.update(1)

stats = pd.DataFrame.from_dict(stats)

# # plot results for each arg to get a feeling of the data
# for arg in ["arg1", "arg2", "arg3", "arg4", "arg5", "arg6"]:
#     plot_optimum(x = stats[arg], y = stats["score"], label = arg)

# # plot the change of the score to get a feeling of the distribution
# plot_optimum(x = stats.index, y = stats["precision"], label = 'epoch')
# plot_optimum(x = stats.index, y = stats["recall"], label = 'epoch')

# stats = {"arg1"  :     [11, 12, 13, 14, 15, 16, 17, 18, 19],
#          "arg2"  :     [21, 22, 23, 24, 25, 26, 27, 28, 29],
#          "arg3"  :     [31, 32, 33, 34, 35, 36, 37, 38, 39],
#          "arg4"  :     [41, 42, 43, 44, 45, 46, 47, 48, 49],
#          "arg5"  :     [51, 52, 53, 54, 55, 56, 57, 58, 59],
#          "arg6"  :     [61, 62, 63, 64, 65, 66, 67, 68, 69],
#          "precision" : [31, 31, 31, 30, 3 , 3 , 3 , 30, 3 ],
#          "recall" :    [2 , 21, 2 , 20, 25, 25, 25, 25, 25]}
# stats = pd.DataFrame.from_dict(stats)

# stats['pr_rc_combi'] = (stats['pr_ave'] + stats['rc_ave']) / 2
stats.insert(0, 'pr_rc_combi', (stats['pr_ave'] + stats['rc_ave']) / 2)

xlsx_export_path = os.path.join(export_dir, 'stats.xlsx')
stats.to_excel(xlsx_export_path, index=True)
print(f"\nResults written to {xlsx_export_path}")

max_pr_idxs = stats.index[stats['pr_ave'] == stats['pr_ave'].max()].tolist()
max_rc_idxs = stats.index[stats['rc_ave'] == stats['rc_ave'].max()].tolist()
overlapping_max_rows = [i for i in max_pr_idxs if i in max_rc_idxs]
if len(overlapping_max_rows) > 0:
    print(f"\nPrecision and recall are both optimal with the same arguments: \n{stats.iloc[overlapping_max_rows]}")
else:
    print(f"\nPrecision and recall are not optimal with the same arguments...")

    max_df = stats[stats['pr_rc_combi'] == stats['pr_rc_combi'].max()]
    print(f"\nBest of both worlds: \n{max_df}")

    max_df = stats[stats['pr_ave'] == stats['pr_ave'].max()]                        # first filter out the pr maxima
    optima = max_df.index[max_df['rc_ave'] == max_df['rc_ave'].max()].tolist()      # then check the maximum for rc
    print(f"\nBest precision: \n{stats.iloc[optima]}")

    max_df = stats[stats['rc_ave'] == stats['rc_ave'].max()]                         # first filter out the rc maxima
    optima = max_df.index[max_df['pr_ave'] == max_df['pr_ave'].max()].tolist()       # then check the maximum for pr
    print(f"\nBest recall: \n{stats.iloc[optima]}")


# # print class specific values 
# ui = input("For which row index do you want to calculate class specific statistics? ")
# arg1, arg2, arg3, arg4, arg5, arg6, _, _, _ = stats.iloc[int(ui)].values.flatten().tolist()
# test_args(arg1, arg2, arg3, arg4, arg5, arg6, print_class_stats = True)

