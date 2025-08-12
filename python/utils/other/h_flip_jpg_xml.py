# Script to copy existing images and their xml, horizontally flip them and ensure that the boundingbox is still
# applicable to the new images. Then save them with a new name. This way we can make more training data.
import cv2
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import time

# bosmuis
# egel
# rat
# spitsmuis
# vogel
# wezel
# woelmuis

dir_to_be_flipped = r"V:\Projecten\A70_30_65\Marterkist\Data\Trainingsdata_compleet\wezel"
output_dir = r'V:\Projecten\A70_30_65\Marterkist\Data\Trainingsdata_compleet\wezel_hflip' # dir will be created if it does not exist

Path(output_dir).mkdir(parents=True, exist_ok=True)
n_files = 0
t0 = time.time()
for file in os.listdir(dir_to_be_flipped):
    if file.endswith(".xml") and (os.path.exists(os.path.join(dir_to_be_flipped, os.path.splitext(file)[0]+'.jpg')) or
                                  os.path.exists(os.path.join(dir_to_be_flipped, os.path.splitext(file)[0]+'.jpeg'))):
        # set vars
        xml_file_name = file
        xml_full_path = os.path.join(dir_to_be_flipped, xml_file_name)
        if os.path.exists(os.path.join(dir_to_be_flipped, os.path.splitext(file)[0] + '.jpg')):
            img_file_name = os.path.splitext(file)[0]+'.jpg'
        else:
            img_file_name = os.path.splitext(file)[0] + '.jpeg'
        img_full_path = os.path.join(dir_to_be_flipped, img_file_name)

        # modify the image
        orginial_image = cv2.imread(img_full_path)
        flipped_image = cv2.flip(orginial_image, 1)
        output_path_img = os.path.join(output_dir, "hflip_" + img_file_name)
        cv2.imwrite(output_path_img, flipped_image)
        print("\nimage horizontally flipped:\t", img_file_name)
        n_files += 1

        # modify the xml file
        tree = ET.parse(xml_full_path)
        root = tree.getroot()
        y_pixels, x_pixels, n_channels = orginial_image.shape

            # modify x values
        xmin = tree.find(".//xmin")
        xmax = tree.find(".//xmax")
        xmin_new = x_pixels - int(xmax.text)
        xmax_new = x_pixels - int(xmin.text)
        xmin.text = str(xmin_new)
        xmax.text = str(xmax_new)

            # modify corresponding filename
        filename = tree.find(".//filename")
        filename.text = str("hflip_" + filename.text)

        output_path_xml = os.path.join(output_dir, "hflip_" + xml_file_name)
        tree.write(output_path_xml)
        print("xml file adjusted: \t\t\t", xml_file_name)
        n_files += 1

t1 = time.time()

print("\nTime elapsed to process {} files: {}".format(n_files, str(time.strftime("%Hh%Mm%Ss", time.gmtime(t1-t0)))))