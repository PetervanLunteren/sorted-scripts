# Script to check the contents of a training dir with pascal voc annotations
# Peter van Lunteren, 17 september 2023

# # WINDOWS: execute script in miniforge prompt
# """
# conda activate ecoassistcondaenv-base && python "C:\Users\smart\Desktop\scan_train_dir.py"
# """

from tabulate import tabulate # pip install tabulate
import os
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt

# user input
folder = r"C:\Users\smart\Downloads\20240711_grant_verified_imgs"

def scan_train_dir(folder_path):
    classes_list = []
    classes_counts = {}
    counts = {'background' : 0, 'images': 0, 'lonesome_xmls' : 0}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            
            # init vars
            is_background = None
            file_name, file_ext = os.path.splitext(file)

            # for all images
            if file_ext.lower() in ['.jpg', '.jpeg', '.gif', '.png']:
                counts['images'] += 1

                # an image without an xml is a background
                if not os.path.isfile(os.path.join(root, f"{file_name}.xml")):
                    is_background = True

            # loop through xmls
            elif file_ext.lower() == ".xml": 

                # check if it has an accompanying image
                lonesome_xml_bools = []
                for ext in ['.jpg', '.jpeg', '.gif', '.png', '.JPG', '.JPEG', '.GIF', '.PNG']:
                    if os.path.isfile(os.path.join(root, f"{file_name}{ext}")):
                        lonesome_xml_bools.append(False)
                    else:
                        lonesome_xml_bools.append(True)
                if all(lonesome_xml_bools):
                    counts['lonesome_xmls'] += 1
                    continue
                
                # read xml
                xml_path = os.path.join(root, file)
                tree = ET.parse(xml_path)
                root_xml = tree.getroot()

                # a xml without any objects is also a background
                is_background = True
                for obj in root_xml.findall('object'):
                    is_background = False
                    class_name = obj.find('name').text

                    # check if it is a known class
                    if class_name not in classes_list:
                        classes_list.append(class_name)

                    # keep count
                    if class_name in classes_counts:
                        classes_counts[class_name] += 1
                    else:
                        classes_counts[class_name] = 1
                
                # count background
                if is_background == True:
                    counts['background'] += 1
    
    # count instances
    total_instances = 0
    for key, value in classes_counts.items():
        total_instances += value

    # print
    print("\n The dataset constists of:\n")
    class_stats = []
    perc_stats = []
    for key, value in classes_counts.items():
        class_stats.append([key, value, round(value / total_instances * 100)])
        perc_stats.append(round(value / total_instances * 100))
    print(tabulate(class_stats, headers=["class", "n", "%"]))
    perc_backgrounds = round(counts['background'] / counts['images'] * 100) if counts['background'] != 0 else "0"
    print(f"\n ... and {counts['background']} background images ({perc_backgrounds}% of total n images)\n")

    # function to add value labels
    def addlabels(x,y):
        for i in range(len(x)):
            plt.text(i,y[i],str(y[i]) + "%")

    # show 
    classes = list(classes_counts.keys())
    counts = list(classes_counts.values())
    percentages = list(perc_stats)
    fig, ax = plt.subplots()
    fig = plt.figure(figsize = (10, 5))
    plt.bar(classes, counts, width = 0.4)
    plt.ylabel("No. of instances")
    plt.xticks(rotation=90)
    addlabels(classes, percentages)
    plt.show(block=True)

# run
scan_train_dir(folder)