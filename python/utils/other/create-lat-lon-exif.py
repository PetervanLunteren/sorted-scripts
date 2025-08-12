# Script to add random location data to images for illustrative purposes

# '''
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\create-lat-lon-exif.py"
# '''


import os
import random
from PIL import Image
import piexif

def add_location_exif(image_path):
    # Generate random coordinates within the specified range
    latitude = round(random.uniform(-21, -19), 6)
    longitude = round(random.uniform(14, 17), 6)

    # Convert coordinates to Exif format
    lat_deg = int(abs(latitude))
    lat_min = int((abs(latitude) - lat_deg) * 60)
    lat_sec = int(((abs(latitude) - lat_deg) * 60 - lat_min) * 60)
    lat_ref = 'S' if latitude < 0 else 'N'

    lon_deg = int(abs(longitude))
    lon_min = int((abs(longitude) - lon_deg) * 60)
    lon_sec = int(((abs(longitude) - lon_deg) * 60 - lon_min) * 60)
    lon_ref = 'W' if longitude < 0 else 'E'

    # Convert coordinates to Exif format (numerator, denominator)
    lat_components = [(lat_deg, 1), (lat_min, 1), (lat_sec, 1)]
    lon_components = [(lon_deg, 1), (lon_min, 1), (lon_sec, 1)]

    # Create Exif data dictionary
    exif_data = {
        piexif.GPSIFD.GPSLatitude: lat_components,
        piexif.GPSIFD.GPSLatitudeRef: lat_ref,
        piexif.GPSIFD.GPSLongitude: lon_components,
        piexif.GPSIFD.GPSLongitudeRef: lon_ref,
    }

    # Read existing Exif data
    try:
        exif_dict = piexif.load(image_path)
    except piexif.InvalidImageDataError:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}}

    # Handle existing Exif value at tag 37380
    if 'Exif' in exif_dict and 37380 in exif_dict['Exif']:
        if isinstance(exif_dict['Exif'][37380], tuple):
            if len(exif_dict['Exif'][37380]) == 2:
                pass  # Already in the correct format
            else:
                exif_dict['Exif'][37380] = (exif_dict['Exif'][37380][0], 1)

    # Update Exif data with GPS information
    exif_dict.update({"GPS": exif_data})

    # Encode Exif data and save it back to the image
    exif_bytes = piexif.dump(exif_dict)
    Image.open(image_path).save(image_path, exif=exif_bytes)

def main():
    # Specify the directory containing JPG images
    image_directory = r'C:\Users\smart\Desktop\Example_folder'

    # Loop over all JPG images in the directory
    for filename in os.listdir(image_directory):
        if filename.lower().endswith('.jpg'):
            image_path = os.path.join(image_directory, filename)
            add_location_exif(image_path)
            print(f"Added GPS location to {filename}")

if __name__ == "__main__":
    main()

