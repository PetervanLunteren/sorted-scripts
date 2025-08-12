
# give me a python script that deletes everything from a dir recursively.
# I want it to delete all files permanently, no need to go to trash. I
# want it to output a tqmd progressbar. I want it to be callable from
# CLI like this:

# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\delete_dir.py" "F:\2024-06-NZF"

import os
import shutil
import sys
from tqdm import tqdm

def count_items(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files) + len(dirs)
        # if count % 100000 == 0:  # Check if count is divisible by 100000
        print(f"Count: {count}")
    return count

def delete_directory_contents(directory):
    total_items = count_items(directory)
    progress_bar = tqdm(total=total_items, desc="Deleting", unit="item")

    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            try:
                os.unlink(os.path.join(root, name))
                progress_bar.update(1)
            except Exception as e:
                print(f"Failed to delete {os.path.join(root, name)}. Reason: {e}")

        for name in dirs:
            try:
                shutil.rmtree(os.path.join(root, name))
                progress_bar.update(1)
            except Exception as e:
                print(f"Failed to delete {os.path.join(root, name)}. Reason: {e}")

    progress_bar.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_dir.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        sys.exit(1)

    delete_directory_contents(directory_path)