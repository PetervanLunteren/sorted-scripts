# script to call system command in loop to process all video's in a certain folder

# cameratraps/detection/video_utils.py normally processes every frame, which takes a really really long time.
# Adjust the two lines of range(0,n_frames) to range(0,n_frames,int(round(get_video_fs(input_video_file)/1)) in
# the for loops in cameratraps/detection/video_utils.py to have fewer frames processed. '/1' doet ie 1 frame per
# seconde, '/3' doet ie 3 frames per seconde, etc.

import os
import json
from pathlib import Path
import time

# function to process all the video's in a dir (not in its subdirs)
n_videos = 0
def process_videos(dir):
    global n_videos
    for file in os.listdir(dir):
        ext = os.path.splitext(file)[-1].lower()
        if ext in movie_extensions:
            n_videos += 1
            cmd = "python {} --confidence_threshold 0.95 {} {}".format(r"C:\git\CameraTraps\detection\process_video.py",
                                                                       r"C:\git\md_v4.1.0.pb",
                                                                       os.path.join(dir, file))
            print(cmd)
            os.system(cmd)



# function to place the json file and movie in created 'animals' and 'empties' subfolders based on json output
def place_in_subdir(dir):
    for file in os.listdir(dir):
        if file.endswith(".json") and not file.endswith('.DS_Store'):
            n_animals = 0
            path_to_json = os.path.join(dir, file)
            with open(path_to_json) as json_file:
                data = json.load(json_file)
            for image in data['images']:
                detections_list = image['detections']
                n_detections = len(image['detections'])
                for i in range(n_detections):
                    if detections_list[i]["category"] == "1":
                        n_animals += 1
            if n_animals > 0:
                video_file = os.path.splitext(path_to_json)[0]
                Path(os.path.join(dir, 'animals')).mkdir(parents=True, exist_ok=True)
                dst = os.path.splitext(path_to_json)[0]
                src = os.path.join(os.path.dirname(video_file), 'animals', os.path.basename(video_file))
                print("\ndst :", dst, "\nsrc :", src)
                os.replace(dst, src)
                Path(os.path.join(dir, 'animals', 'json_files')).mkdir(parents=True, exist_ok=True)
                dst_json = path_to_json
                src_json = os.path.join(os.path.dirname(dst_json), 'animals', 'json_files', os.path.basename(dst_json))
                print("\ndst :", dst_json, "\nsrc :", src_json)
                os.replace(dst_json, src_json)
            elif n_animals == 0:
                video_file = os.path.splitext(path_to_json)[0]
                Path(os.path.join(dir, 'empties')).mkdir(parents=True, exist_ok=True)
                dst = os.path.splitext(path_to_json)[0]
                src = os.path.join(os.path.dirname(video_file), 'empties', os.path.basename(video_file))
                print("\ndst :", dst, "\nsrc :", src)
                os.replace(dst, src)
                Path(os.path.join(dir, 'empties', 'json_files')).mkdir(parents=True, exist_ok=True)
                dst_json = path_to_json
                src_json = os.path.join(os.path.dirname(dst_json), 'empties', 'json_files', os.path.basename(dst_json))
                print("\ndst :", dst_json, "\nsrc :", src_json)
                os.replace(dst_json, src_json)

dirs = [r'V:\Projecten\A70_30_65\Losse_camera\Data\Input\21-0965\100EK113',
        r'V:\Projecten\A70_30_65\Losse_camera\Data\Input\21-0965\101EK113']

movie_extensions = ('.mp4', '.avi', '.mpeg', '.mpg', '.MP4', '.AVI', '.MPEG', '.MPG')
t0 = time.time()

for dir in dirs:
    print(dir)
    process_videos(dir)
    place_in_subdir(dir)

t1 = time.time()
print("\n\nTime elapsed to process {} video's: {}h and {}min".format(n_videos,
                                                                     round((t1-t0))//3600,
                                                                     round((round((t1-t0))-((round((t1-t0))//3600)*3600))/60)))
