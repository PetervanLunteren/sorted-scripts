from os import listdir
from os.path import isfile, join
import shutil
import os, random

# Open a file
mypath = r"V:\Projecten\A70_30_65\Marterkist\Data\Data voor Jelte - nog niet gecheckt\Output_ronde2_threshold070\bosmuis"
output_folder = r"V:\Projecten\A70_30_65\Marterkist\Data\Train\test"

files =[]

for file in os.listdir(mypath):
    if file.endswith(".jpg"):
        files.append(file)

random_files = random.sample(files, 27)

print (random_files)

for file in random_files:
    nametxt = file.split(".")[0]
    print (nametxt)
    inputfileXLM = mypath + os.sep + nametxt + ".xml"
    inputfileJPG = os.path.join(mypath, file)
    shutil.move(inputfileJPG, output_folder)
    shutil.move(inputfileXLM, output_folder)