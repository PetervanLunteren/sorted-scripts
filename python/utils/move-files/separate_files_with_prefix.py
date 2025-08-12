# Script to extract files starting or ending with a string

import os
from pathlib import Path
import shutil

# bosmuis
# egel
# rat
# spitsmuis
# vogel
# wezel
# woelmuis

dir_to_be_separated = r"V:\Projecten\A70_30_65\Marterkist\Data\Trainingsdata_compleet\woelmuis"
output_dir = r"V:\Projecten\A70_30_65\Marterkist\Data\Trainingsdata_compleet\hflip_woelmuis"
Path(output_dir).mkdir(parents=True, exist_ok=True)

for path in os.listdir(dir_to_be_separated):
    if path.startswith("hflip_"):
        src = os.path.join(dir_to_be_separated, path)
        dst = os.path.join(output_dir, path)
        print(src)
        print(dst)
        shutil.move(src, dst)