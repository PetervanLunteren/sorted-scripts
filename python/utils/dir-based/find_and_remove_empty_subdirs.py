# loops through all subfolders of *root* and deletes the empties

# https://stackoverflow.com/questions/47093561/remove-empty-folders-python

# If you use os.walk(topdown=False), it goes in reverse order, 
# so you encounter the child directories before the parents.
# Then if you track which directories you've deleted, you can 
# delete parent directories recursively.

# python "C:\Users\smart\Desktop\find_and_remove_empty_subdirs.py"

import os
from tqdm import tqdm

def delete_empty_folders(root):

    deleted = set()
    
    for current_dir, subdirs, files in tqdm(os.walk(root, topdown=False)):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

    return deleted

dirs_deleted = delete_empty_folders(r"C:\Peter\fetch-lila-bc-data\downloads\aves")

print(f"dirs : {dirs_deleted}")
print(f"n dirs deleted : {len(dirs_deleted)}")