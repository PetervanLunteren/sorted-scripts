# This script is for tesing a model. You need a folder with subfolder per class with manually checked validation images.
# The script lets the model run over the images once, then uses the confidence values to see what happens with
# different thresholds. It outputs tables and a plot.
#
# Use python interpreter 3.7 from conda env 'tensorflow'

import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from detector import DetectorTF2

# input
validation_dir = r"V:\Projecten\A70_30_65\Marterkist\Data\Test_data" # path to the folder containing subfolders with validation images
model_path = r"V:\Projecten\A70_30_65\Marterkist\ExportModel\saved_model_fixed_20k0003_35k0001"  #  model you want to test
path_to_labelmap = r"V:\Projecten\A70_30_65\Marterkist\Model\labelmap.pbtxt"  # associated labelmap.pbtxt

###### def to run the model over the frames
def DetectImagesFromFolder(detector, images_dir):
    global stats
    for file in os.scandir(images_dir):
        if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            image_path = file.path
            img = cv2.imread(image_path)
            det_boxes = detector.DetectFromImage(img)
            Actual_label = os.path.normpath(file).split(os.sep)[-2]
            image_name = os.path.normpath(file).split(os.sep)[-1]
            print(det_boxes)
            if len(det_boxes) > 0:
                Predicted_label = str(det_boxes[0][4])
                Confidence = round(det_boxes[0][5], 4)
            else:
                Predicted_label = "_anders"
                Confidence = np.nan
            print(f"Image           = {image_name}\n"
                  f"Actual label    = {Actual_label}\n"
                  f"Predicted label = {Predicted_label}\n"
                  f"Confidence      = {Confidence}\n")
            values_to_add = {"Actual_label": Actual_label,
                             "Predicted_label": Predicted_label,
                             "Confidence": Confidence,
                             "Threshold": 0}
            row_to_add = pd.Series(values_to_add)
            stats = stats.append(row_to_add, ignore_index=True)


# fill list of dirs to be tested, amount of files in total and the files per class
class_dirs = []
n_test_images = 0
for root, dirs, files in os.walk(validation_dir):
    if dirs == []:
        class_dirs.append(root)
        n_test_images += len(files)

# # create df with the predicted labels and their confidence intervals
# stats = pd.DataFrame(columns=["Actual_label", "Predicted_label", "Confidence", "Threshold"])
# for class_dir in class_dirs:
#     detector = DetectorTF2(model_path, path_to_labelmap, class_id=None)
#     DetectImagesFromFolder(detector, class_dir)
# stats.to_csv(r"C:\Users\GisBeheer\Desktop\dataframe_stats.csv", index=False)

# use this df to calc multiple thresholds
stats = pd.read_csv(r"C:\Users\GisBeheer\Desktop\dataframe_stats.csv")
stats["Confidence"] = pd.to_numeric(stats["Confidence"], downcast="float")
stats_clean = stats.copy()
for threshold in np.arange(0.1, 1, 0.05):
    threshold = round(threshold, 2)
    if threshold != 0:
        df = stats_clean.copy()
        df['Predicted_label'] = np.where(stats_clean['Confidence'] >= threshold, stats_clean['Predicted_label'], '_anders')
        df['Threshold'] = threshold
        stats = stats.append(df)

# compare the predicted labels with the actual labels and produce tables with statistics
pd.set_option("display.max_rows", None, "display.max_columns", None)
stats["Pred_correct"] = np.where(stats["Actual_label"] == stats["Predicted_label"], True, False)
df_cl_th = stats.groupby(['Threshold', "Actual_label"], as_index=False)\
       .agg(n_correct=pd.NamedAgg(column='Pred_correct', aggfunc=lambda x: (x == True).sum()),
            n_incorrect=pd.NamedAgg(column='Pred_correct', aggfunc=lambda x: (x == False).sum()))\
       .reset_index()
df_cl_th['based_on_n'] = df_cl_th['n_correct'] + df_cl_th['n_incorrect']
df_cl_th['total_n_test_images'] = n_test_images
df_cl_th['perc_correct'] = round((df_cl_th['n_correct']/(df_cl_th['n_correct'] + df_cl_th['n_incorrect'])), 2)
stats.to_csv(r"C:\Users\GisBeheer\Desktop\predicted_labels.csv", index=False)

print("\nPercentage correct prediction per threshold:")
df_th = df_cl_th.groupby('Threshold', as_index=False).mean()\
    .drop(columns=['index', 'n_correct', 'n_incorrect', 'based_on_n',
                   'total_n_test_images'])
df_th = df_th.round(2)
print(df_th)

print("\nPercentage correct prediction per class with best threshold:")
print(df_cl_th
      # [df_cl_th.Actual_label == 'bosmuis']                                          # show the values for one class in all thresholds
      [(df_cl_th.Threshold == df_th['Threshold'][df_th['perc_correct'].idxmax()])]  # show the values for all classes in the highest overall threshold
      .drop(columns=['index', 'based_on_n', 'total_n_test_images']))

print("\nMinimum confidence is: ", stats["Confidence"].min(), "\n") # the reason that threshold of 0 until 0.5 is overall best, is because the lowest detection threshold is 0.5. So nothing happens with thresholds below 0.5. Above that we are skipping correct detections as '_anders'.

# plot
x = df_th['Threshold']
y = df_th['perc_correct']
xnew = np.linspace(df_th['Threshold'].min(), df_th['Threshold'].max(), num=50, endpoint=True)
f_cubic = interp1d(x, y, kind='cubic')
plt.plot(x, y, 'o')
plt.plot(xnew, f_cubic(xnew), '--')
plt.show()

