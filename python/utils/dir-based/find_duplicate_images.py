# Script to find duplicate images in a folder
# Peter van Lunteren 16 Sept 2023


# MACOS: first time with new ecoassistcondaenv
"""
conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv
pip install difPy
"""

# MACOS: execute script in terminal
"""
pmset noidle &;conda activate /Applications/.EcoAssist_files/miniforge/envs/ecoassistcondaenv;python "/Users/peter/Documents/scripting/sorted-scripts/python/utils/find_duplicate_images.py"
"""

import difPy
import json

# user input
folder = "/Users/peter/Desktop/test_images"


def print_result(json_object):
    if json_object != {}:
        print(json.dumps(json_object, indent=2))
    else:
        print("Found no duplicates!")

if __name__ == '__main__':
    dif = difPy.build(folder, show_progress=True, logs=True)
    search = difPy.search(dif, similarity='duplicates', show_progress=True, logs=True)
    print_result(search.result)
    print_result(search.lower_quality)


