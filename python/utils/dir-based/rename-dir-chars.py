# script to rename characters in subdirectories
# this is quite difficult because if you change the name of a dir, you can't conintue your os.walk() and have to start over
# first it fills a list with subdirs, and then runs the function over each subdir. You don't want to run the function over a dir that is too large, 
# because it will reloop everything again every time....

# python "C:\Users\smart\Desktop\rename-hastags.py"


import os

def get_subdirectories(directory_path, depth=1):
    subdirectories = []
    len_root = len(os.path.normpath(directory_path).split(os.path.sep))
    for root, dirs, files in os.walk(directory_path):
        if dirs:
            for dir in dirs:
                subdir = os.path.join(root, dir)
                dirs_list = os.path.normpath(subdir).split(os.path.sep)
                if len(dirs_list) == depth + len_root:
                    subdirectories.append(subdir)
                    print(f"Adding {subdir}")



        # if root == directory_path:
        #     subdirectories.extend([os.path.join(root, d) for d in dirs])
        # elif root.startswith(directory_path):
        #     current_depth = root[len(directory_path):].count(os.sep)
        #     if current_depth == depth:
        #         subdirectories.extend([os.path.join(root, d) for d in dirs])
        #     elif current_depth > depth:
        #         break
    return subdirectories

directory_path = r'C:\Peter\fetch-lila-bc-data\downloads\loxodonta africana'
# first_layer_subdirs = get_subdirectories(directory_path, depth=1)
# second_layer_subdirs = get_subdirectories(directory_path, depth=2)
third_layer_subdirs = get_subdirectories(directory_path, depth=3)

# print("\n\nFirst layer subdirectories:")
# for dir in first_layer_subdirs:
#     print(dir)

# print("\n\nSecond layer subdirectories:")
# for dir in second_layer_subdirs:
#     print(dir)

# print("\n\nThird layer subdirectories:")
# for dir in third_layer_subdirs:
#     print(dir)

# exit()

def rename_files(directory):
    global hastags_present
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            old_path = os.path.normpath(os.path.join(root, file_name))
            if '#' in old_path:
                new_path = os.path.normpath(os.path.join(root.replace('#', 'h'), file_name.replace('#', 'h')))
                # print(f"Renaming: {old_path} -> {new_path}")
                os.renames(old_path, new_path)
                return
    print("DEBUG - got to the part where it is done")
    hastags_present = False


def remove_hastags(dir):
    global hastags_present
    hastags_present = True

    while hastags_present:
        rename_files(dir)
    print(f"CLEANED dir : {dir}")

print("Proceeding with the hastag removal")
for dir in third_layer_subdirs:
    print(f"init remove_hastags({dir})")
    remove_hastags(dir)


# (base) C:\Users\smart>python "C:\Users\smart\Desktop\rename-hastags.py"
# Renaming: C:\Peter\fetch-lila-bc-data\downloads\loxodonta africana\snapshot-safari_ENO\ENO_S1\B05\B05_R1\PvL_loc_B05\seq_ENO_S1#B05#1#265\PvL_seq_7a7c3\0001.JPG -> C:\Peter\fetch-lila-bc-data\downloads\loxodonta africana\snapshot-safari_ENO\ENO_S1\B05\B05_R1\PvL_loc_B05\seq_ENO_S1hB05h1h265\PvL_seq_7a7c3\0001.JPG
# Traceback (most recent call last):
#   File "C:\Users\smart\Desktop\rename-hastags.py", line 19, in <module>
#     rename_files(directory_path)
#   File "C:\Users\smart\Desktop\rename-hastags.py", line 15, in rename_files
#     os.rename(old_path, new_path)
# FileNotFoundError: [WinError 3] The system cannot find the path specified: 'C:\\Peter\\fetch-lila-bc-data\\downloads\\loxodonta africana\\snapshot-safari_ENO\\ENO_S1\\B05\\B05_R1\\PvL_loc_B05\\seq_ENO_S1#B05#1#265\\PvL_seq_7a7c3\\0001.JPG' -> 'C:\\Peter\\fetch-lila-bc-data\\downloads\\loxodonta africana\\snapshot-safari_ENO\\ENO_S1\\B05\\B05_R1\\PvL_loc_B05\\seq_ENO_S1hB05h1h265\\PvL_seq_7a7c3\\0001.JPG'