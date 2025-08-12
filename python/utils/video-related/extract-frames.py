# this script extracts frames (1 per second) from every video file in a dir (recurvise) and saves them as images in a folder with the same name
# you'll end up with the normal folder structure and an additional folder with the extracted frames

# If you then run MD on images, it will find the images in the extracted frames folder and not the original video


# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\extract-frames.py"

import cv2
import os
from tqdm import tqdm
from PIL import Image
import piexif

dir_to_search = r"C:\Peter\projects\2024-25-ARI\data\raw\imgs+videos+frames"


def extract_frames(video_path):
    # Get the video file name without extension
    video_name = os.path.splitext(video_path)[0]
    
    # Create output folder with the same name as the video
    output_folder = video_name
    
    # # DEBUG remove the previous folder with contents
    # if os.path.exists(output_folder):
    #     import shutil
    #     shutil.rmtree(output_folder)
    # return
    
    # create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval to extract one frame per second
    frame_interval = int(fps)  # One frame per second

    # Frame counter
    frame_count = 0

    while True:
        ret, frame = video.read()

        # If no frame is returned, the video has ended
        if not ret:
            break

        # Save the frame if it's the desired interval
        if frame_count % frame_interval == 0:
            # Construct the output filename
            output_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            
            # Save the frame as an image
            cv2.imwrite(output_filename, frame)
            
            # make sure they all have the same datetime
            # Define the datetime string
            date_time_original = "2024:01:01 00:00:00"

            # # Prepare the EXIF data
            # exif_dict = {
            #     "0th": {},
            #     "Exif": {
            #         piexif.ExifIFDName.DateTimeOriginal: date_time_original,
            #         piexif.ExifIFDName.DateTimeDigitized: date_time_original,
            #     },
            #     "1st": {},
            #     "GPS": {}
            # }

            DATE_TIME_ORIGINAL = 36867  # Tag for DateTimeOriginal
            DATE_TIME_DIGITIZED = 36868  # Tag for DateTimeDigitized

            # Prepare the EXIF data
            exif_dict = {
                "0th": {},
                "Exif": {
                    DATE_TIME_ORIGINAL: date_time_original,
                    DATE_TIME_DIGITIZED: date_time_original,
                },
                "1st": {},
                "GPS": {}
            }

            # Convert the EXIF data to binary
            exif_bytes = piexif.dump(exif_dict)

            # Save the image with EXIF data
            # Use the `cv2.imwrite` method to save the image with EXIF
            # You can read the image using PIL to add the EXIF data
            

            # Open the saved image with PIL
            pil_image = Image.open(output_filename)

            # Save the image with EXIF data
            pil_image.save(output_filename, exif=exif_bytes)

        # Increase frame count
        frame_count += 1

    # Release the video
    video.release()

# Example usage

video_list = []

for root, dirs, files in os.walk(dir_to_search):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_file_path = os.path.join(root, file)
            video_list.append(video_file_path)

for video_file in tqdm(video_list):
    extract_frames(video_file)

print("Done")