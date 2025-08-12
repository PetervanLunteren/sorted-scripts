# script to sort a folder of camtrap images into subdirs

# it will look recusively for all images and sort them into subfolders 'PvL_seq_xxxxxxxxxxx'
# it will retain its original folder structure and place the 'PvL_seq_' folder at the last layer.
# if it can find xml file that belongs to the image (PASCAL VOC annotation), it will copy that too


# """
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\1-sort-dir-in-sequences.py"
# """






img_dir = r"C:\Users\smart\Downloads\2024-16-SCO"






import os
import sys
sys.path.insert(0, '/Applications/.EcoAssist_files/cameratraps')
sys.path.insert(0, r'C:\Users\smart\EcoAssist_files\cameratraps')
from megadetector.data_management import read_exif
import json
from tqdm import tqdm 
from collections import defaultdict 
from pathlib import Path
import shutil
import uuid





def separate_into_sequences(img_dir):
    input_path = img_dir
    overflow_folder_handling_enabled = False
    exif_options = read_exif.ReadExifOptions()
    exif_options.verbose = False
    # exif_options.n_workers = default_workers_for_parallel_tasks
    # exif_options.use_threads = parallelization_defaults_to_threads
    exif_options.processing_library = 'pil'
    exif_options.byte_handling = 'delete'

    exif_results_file = os.path.join(img_dir,'exif_data.json')

    if os.path.isfile(exif_results_file):
        print('Reading EXIF results from {}'.format(exif_results_file))
        with open(exif_results_file,'r') as f:
            exif_results = json.load(f)
    else:        
        exif_results = read_exif.read_exif_from_folder(input_path,
                                                    output_file=exif_results_file,
                                                    options=exif_options)


    #% Prepare COCO-camera-traps-compatible image objects for EXIF results
    import datetime    
    from megadetector.data_management.read_exif import parse_exif_datetime_string
    min_valid_timestamp_year = 1900
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
            im['datetime'] = dt
            
            # # An image from the future (or within the last hour) is invalid
            # if (now - dt).total_seconds() <= 1*60*60:
            #     print('<EA>Warning: datetime for {} is {}<EA>'.format(
            #         im['file_name'],dt))
            #     im['datetime'] = None            
            #     images_with_invalid_datetime.append(im['file_name'])
            
            # # An image from before the dawn of time is also invalid
            # elif dt.year < min_valid_timestamp_year:
            #     print('<EA>Warning: datetime for {} is {}<EA>'.format(
            #         im['file_name'],dt))
            #     im['datetime'] = None
            #     images_with_invalid_datetime.append(im['file_name'])
            
            # else:
            #     im['datetime'] = dt
        image_info.append(im)
    # ...for each exif image result

    print('<EA>Parsed EXIF datetime information, unable to parse EXIF data from {} of {} images<EA>'.format(
        len(images_without_datetime),len(exif_results)))

    #% Assemble into sequences
    from megadetector.data_management import cct_json_utils
    print('Assembling images into sequences')
    cct_json_utils.create_sequences(image_info)

    # Make a list of images appearing at each location
    sequence_to_images = defaultdict(list)
    for im in tqdm(image_info):
        sequence_to_images[im['seq_id']].append(im)
    all_sequences = list(sorted(sequence_to_images.keys()))
    # print(f"\n\nall_sequences     : {all_sequences}")
    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    # pp.pprint(dict(sequence_to_images))

    n_imgs = 0
    for seq, imgs in sequence_to_images.items():
        for img in imgs:
            n_imgs += 1

    pbar = tqdm(total=n_imgs)
    for seq, imgs in sequence_to_images.items():
        seq_unique_id = f"PvL_seq_{str(uuid.uuid4())[:5]}" # there were some pathnames that were too long... so I had to cut some string length
        img_counter = 1
        for img in imgs:
            location = img['location']
            img_rel_fpath_list = list(Path(img['file_name']).parts)
            id = img['id']
            datetime = img['datetime']
            seq_id = im['seq_id']
            seq_num_frames = img['seq_num_frames']
            frame_num = img['frame_num']
            src_fpath = Path(os.path.normpath(os.path.join(img_dir, Path(*img_rel_fpath_list))))
            fname_ext = os.path.splitext(img_rel_fpath_list[-1])[1]
            fname = f"{img_counter:04}{fname_ext}" # there were some pathnames that were too long... so I had to cut some string length
            img_counter += 1
            dst_fpath = Path(os.path.normpath(os.path.join(img_dir, Path(*img_rel_fpath_list[:-1]), seq_unique_id, fname)))

            # double check if the file exists
            if os.path.isfile(src_fpath):
                try:
                    Path(os.path.dirname(dst_fpath)).mkdir(parents=True, exist_ok=True)
                    shutil.move(src_fpath, dst_fpath)
                except Exception as error:
                    print(f"ERROR - filepath is probabaly too long! '{src_fpath}'")
                    print(f"ERROR       : {error}\n\n")
            else:
                print(f"ERROR - Could not find file : {src_fpath}")
                print(f"ERROR       : {error}\n\n")

            # check if there is an accompanying xml file (PASCAL VOC annotation)
            # if so, move that one too
            xml_src_fpath = os.path.splitext(src_fpath)[0] + ".xml"
            xml_dst_fpath = os.path.splitext(dst_fpath)[0] + ".xml"

            if os.path.isfile(xml_src_fpath):
                try:
                    Path(os.path.dirname(xml_dst_fpath)).mkdir(parents=True, exist_ok=True)
                    shutil.move(xml_src_fpath, xml_dst_fpath)
                except Exception as error:
                    print(f"ERROR - filepath is probabaly too long! '{xml_src_fpath}'")
                    print(f"ERROR       : {error}\n\n")

            # update pbar
            pbar.update(1)


    if os.path.isfile(exif_results_file):
        os.remove(exif_results_file)



separate_into_sequences(img_dir)

# print(f"\n\nsequence_to_images : {pprint(dict(defaultdict))}")


# #% Load classification results
# sequence_level_smoothing_input_file = smoothed_classification_files[0]
# with open(sequence_level_smoothing_input_file,'r') as f:
#     d = json.load(f)

# # Map each filename to classification results for that file
# filename_to_results = {}
# for im in tqdm(d['images']):
#     filename_to_results[im['file'].replace('\\','/')] = im