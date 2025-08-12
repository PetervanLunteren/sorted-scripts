# script to loop through YOLO label txt files and change class index
# python "/Users/peter/Documents/repos/sorted-scripts/python/utils/change-class-index-yolo-txts.py"

import os

txt_dir = "/Users/peter/Documents/repos/EcoAssist_other_files/data/annotated-datasets/small"
old_index = 4
new_index = 3

for filename in os.listdir(txt_dir):
    txt_file = os.path.join(txt_dir, filename)
    # open all txt files
    if txt_file.endswith(".txt") and not filename == "classes.txt":
        # print original txt content
        print(f"txt_file : {txt_file}\n")
        f = open(txt_file, "r")
        content_str = f.read()
        f.close()
        print(f"Original content:\n{content_str}\n")

        # modify content per line
        content_list = content_str.split()
        elem = 0
        new_content_str = ""
        for line in range(int(len(content_list)/5)):
            class_id = content_list[elem]
            xo = content_list[elem + 1]
            yo = content_list[elem + 2]
            w_box = content_list[elem + 3]
            h_box = content_list[elem + 4]
            if class_id == str(old_index):
                string = f'{new_index} {xo} {yo} {w_box} {h_box}\n'
            else:
                string = f'{class_id} {xo} {yo} {w_box} {h_box}\n'
            new_content_str += string
            elem += 5
        
        # print modified txt content
        print(f"Modified content:\n{new_content_str}\n")
        
        f = open(txt_file, "w")
        f.write(new_content_str)
        f.close()