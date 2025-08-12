# Script to loop through folder and find and count certain files


# conda activate ecoassistcondaenv & python "C:\Users\smart\Desktop\check-contents-of-folder.py"


# import required module
import os
import json
 
# assign directory
directory = r"C:\Peter\desert-lion-project\philips-footage\unsorted\IntoNature4\CT_NEW_11Aug2023"
 
# iterate over files in
ext_count_dict = {}
ext_fpath_dict = {}
for root, dirs, files in os.walk(directory):
    for filename in files:
        fname_no_ext, fname_ext = os.path.splitext(filename)
        fname_ext = fname_ext.lower()
        if fname_ext in ext_count_dict:
            ext_count_dict[fname_ext] += 1
            ext_fpath_dict[fname_ext].append(os.path.join(root, filename))
        else:
            ext_count_dict[fname_ext] = 1
            ext_fpath_dict[fname_ext] = [os.path.join(root, filename)]

# print results
print(json.dumps(ext_count_dict, indent = 2))

# # optionally print list of files
# for file in ext_fpath_dict[".avi"]:
#     print(file)


