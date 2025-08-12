# Script to copy exif data from certain image to the others

# '''
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\copy-exif.py"
# '''

# import
from PIL import Image
import os
import piexif

# define
def copy_exif_data(src_img_path, dst_img_path):
    try:
        src_img = Image.open(src_img_path)
        dst_img = Image.open(dst_img_path)
        exif_dict = piexif.load(src_img.info["exif"])
        dst_img = dst_img.convert("RGB") 
        exif_bytes = piexif.dump(exif_dict)
        dst_img.save(dst_img_path, "jpeg", exif=exif_bytes)
        print(f"Exif data copied successfully from {src_img_path} to {dst_img_path}")
    except Exception as e:
        print(f"Error: {e}")

# run
common_dir = "C:/Users/smart/Desktop/Example_folder"
src_imgs = ["02.JPG", "20.JPG", "23.JPG", "02.JPG", "24.JPG"]
dst_imgs = ["04.JPG", "04.JPG", "06.JPG", "07.JPG", "12.JPG"]
for i in range(len(src_imgs)):
    src_image_path = os.path.join(common_dir, src_imgs[i])
    dst_image_path = os.path.join(common_dir, dst_imgs[i])
    copy_exif_data(src_image_path, dst_image_path)