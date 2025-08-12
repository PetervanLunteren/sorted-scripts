# This script combines several scripts and analyses videos to output an xlsx file with the detections.
# Use python interpreter 3.7 from conda env 'tensorflow'

import datetime
import math
import os
import shutil
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path, PurePath
from xml.etree import ElementTree as et

import cv2
import pandas as pd
from detector import DetectorTF2

t0 = time.time()  # start the timer

# input
dir_to_analyse = r"C:\Users\GisBeheer\Desktop\test_data\VERKEERD"  # This dir and all dirs in its tree will be analysed (if they contain images)
analyse_videos = False  # True for videos and False for images

model_path = r"V:\Projecten\A70_30_65\Marterkist\ExportModel\saved_model_fixed_20k0003_35k0001"  # which model do you want to use?
path_to_labelmap = r"V:\Projecten\A70_30_65\Marterkist\Model\labelmap.pbtxt"  # choose the associated labelmap.pbtxt
need_trainings_data = True  # trainings data will produce xml files and can be used for further training
need_visual_data = True  # the detection will be drawn on the image so that it is easily visible
threshold = 0.7  # everything with a confidence below the threshold will be classified as 'anders'


###### Step 1: cut the videos into frames
def getTimestamp(File):
    modTimesinceEpoc = os.path.getmtime(File)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
    modtime = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    timestampstr = timestamp.replace(":", "")
    timestampstr = timestampstr.replace(" ", "_")
    timestampstr = timestampstr.replace("-", "_")
    return timestampstr, modtime


def cut_video2frames(inPath):
    global dir_with_photos
    dir_with_photos = os.path.join(os.path.dirname(inPath), 'analysis_files', 'frames',
                                   os.path.basename(os.path.normpath(inPath)))
    Path(dir_with_photos).mkdir(parents=True, exist_ok=True)
    onlyfiles = [f for f in listdir(inPath) if
                 isfile(join(inPath, f)) and f.endswith(('.avi', '.AVI', '.mp4', '.MP4', '.JPEG', '.PNG'))]

    for videoFile in onlyfiles:
        naam = videoFile.split(".")[0]
        print("Cutting {} into frames...".format(videoFile))
        videoFile = inPath + os.sep + videoFile
        cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path
        timestamp = getTimestamp(videoFile)
        modtime = timestamp[1]
        frameRate = cap.get(5)  # frame rate
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
        duration = int(frame_count / frameRate)  # duration is total number of frames / frames per second
        # x = 1
        Y = 1000
        count = 0
        while (cap.isOpened()):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                fileName = naam + "_" + timestamp[0] + "%d.jpg" % Y
                Y += 1
                outfile = dir_with_photos + os.sep + fileName
                cv2.imwrite(outfile, frame)
                deltaSec = count - duration
                date = datetime.datetime(year=modtime[0], month=modtime[1], day=modtime[2], hour=modtime[3],
                                         minute=modtime[4], second=modtime[5])
                newDate = date + datetime.timedelta(seconds=deltaSec)
                modTime = time.mktime(newDate.timetuple())
                os.utime(outfile, (modTime, modTime))
                count = count + 1
        cap.release()


def cut_multiple_videos2frames(dir_with_subdirs):
    dirpaths = []
    for path in os.listdir(dir_with_subdirs):
        full_path = os.path.join(dir_with_subdirs, path)
        if os.path.isdir(full_path):
            dirpaths.append(full_path)
    for dirpath in dirpaths:
        cut_video2frames(dirpath)


def count_n_files(dir):
    n_files = 0
    for root, subdirectories, files in os.walk(dir):
        for file in files:
            if file.endswith(('.avi', '.AVI', '.mp4', '.MP4', '.JPEG', '.PNG', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG',
                              '.PNG')):
                n_files += 1
    return n_files


# if this script is ran multiple times, make sure to delete the files of last time. So remove the output dirs.
analysis_files = os.path.join(dir_to_analyse, 'analysis_files')
final_output_xlsx = os.path.join(dir_to_analyse, 'final_output_xlsx')
if os.path.exists(analysis_files) and os.path.isdir(analysis_files):
    print(
        "This is not the first time this script is ran on {}. Removing dir 'analysis_files' before continuing...".format(
            dir_to_analyse))
    shutil.rmtree(analysis_files)
if os.path.exists(final_output_xlsx) and os.path.isdir(final_output_xlsx):
    print("Removing dir 'final_output_xlsx' before continuing...\n")
    shutil.rmtree(final_output_xlsx)

n_files = count_n_files(
    dir_to_analyse)  # count the number of videos for the time calculation at the end

if analyse_videos:
    cut_multiple_videos2frames(dir_to_analyse)


###### Step 2: Run the model over the frames
def getDate(file):
    modTimesinceEpoc = os.path.getmtime(
        file)  # Get file's Last modification time stamp only in terms of seconds since epoch
    modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(
        modTimesinceEpoc))  # Convert seconds since epoch to readable timestamp
    return modificationTime


def writeXML(XMLfile, NewPath, name, xmin, ymin, xmax, ymax, klasse):
    tree = et.parse(XMLfile)
    NewPath = NewPath + os.sep + name
    tree.find('.//path').text = NewPath
    tree.find('.//filename').text = name
    for elem in tree.findall('.//object//name'):
        elem.text = klasse
    tree.find('.//object//bndbox//xmin').text = str(xmin)
    tree.find('.//object//bndbox//ymin').text = str(ymin)
    tree.find('.//object//bndbox//xmax').text = str(xmax)
    tree.find('.//object//bndbox//ymax').text = str(ymax)
    xmlName = os.path.splitext(NewPath)[0]
    outputfile = xmlName + ".xml"
    tree.write(outputfile)


def DetectFromVideo(detector, Video_path, save_output=False, output_dir='output/'):
    cap = cv2.VideoCapture(Video_path)
    if save_output:
        output_path = os.path.join(output_dir, 'detection_' + Video_path.split("/")[-1])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (frame_width, frame_height))

    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret: break
        timestamp1 = time.time()
        det_boxes = detector.DetectFromImage(img)
        elapsed_time = round((time.time() - timestamp1) * 1000)  # ms
        img = detector.DisplayDetections(img, det_boxes, det_time=elapsed_time)
        cv2.imshow('TF2 Detection', img)
        if cv2.waitKey(1) == 27:
            break
        if save_output:
            out.write(img)
    cap.release()
    if save_output:
        out.release()


def DetectImagesFromFolder(detector, images_dir, output_dir_train, output_dir_visual):
    global threshold
    global need_trainings_data
    global need_visual_data

    for file in os.scandir(images_dir):
        if file.is_file() and file.name.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            image_path = file.path
            name = file.name
            img = cv2.imread(image_path)
            det_boxes = detector.DetectFromImage(img)
            if need_visual_data:
                img_box = detector.DisplayDetections(img, det_boxes)
            dateMod = getDate(image_path).replace(" ", ",")
            if len(det_boxes) > 0:
                numb = det_boxes[0][5]
                if numb >= threshold:
                    print("In image {} a {} has been detected with a certainty of {}".format(image_path, det_boxes[0][4],
                                                                                             round(det_boxes[0][5], 2)))
                    logFile.write(name + "," + str(det_boxes[0][5]) + "," + det_boxes[0][4] + "," + dateMod + "\n")
                    if need_trainings_data:
                        output_dirRes = output_dir_train + os.sep + det_boxes[0][4]
                        img_out = os.path.join(output_dirRes, name)
                        cv2.imwrite(img_out, img)
                        writeXML(XMLfile, output_dirRes, name, det_boxes[0][0], det_boxes[0][1], det_boxes[0][2],
                                 det_boxes[0][3], det_boxes[0][4])
                    if need_visual_data:
                        output_dirRes = output_dir_visual + os.sep + det_boxes[0][4]
                        img_out = os.path.join(output_dirRes, name)
                        cv2.imwrite(img_out, img_box)
                else:
                    print(
                        "In image {} something is detected, but not with a certainty above the threshold of {}".format(image_path,
                                                                                                                   threshold))
                    logFile.write(name + "," + "-999" + "," + "anders" + "," + dateMod + "\n")
                    if need_trainings_data:
                        output_dirRes = output_dir_train + os.sep + "_anders"
                        img_out = os.path.join(output_dirRes, name)
                        cv2.imwrite(img_out, img)
                    if need_visual_data:
                        output_dirRes = output_dir_visual + os.sep + "_anders"
                        img_out = os.path.join(output_dirRes, name)
                        cv2.imwrite(img_out, img_box)
            else:
                print("In image {} nothing has been detected".format(image_path))
                logFile.write(name + "," + "-999" + "," + "anders" + "," + dateMod + "\n")
                if need_trainings_data:
                    output_dirRes = output_dir_train + os.sep + "_anders"
                    img_out = os.path.join(output_dirRes, name)
                    cv2.imwrite(img_out, img)
                if need_visual_data:
                    output_dirRes = output_dir_visual + os.sep + "_anders"
                    img_out = os.path.join(output_dirRes, name)
                    cv2.imwrite(img_out, img_box)


def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()
            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None
    return items


# other variables which do not need to be adjusted regularly
id_list = None
XMLfile = r"V:\Projecten\A70_30_65\Marterkist\Data\vidIMG_MK.xml"
save_output = True

images_dirs = []
if analyse_videos:
    dir_containing_subdirs_with_photos = os.path.join(dir_to_analyse, 'analysis_files', 'frames')
else:
    dir_containing_subdirs_with_photos = dir_to_analyse
for dir in [x[0] for x in os.walk(dir_containing_subdirs_with_photos)]:  # check the entire tree for dirs which contain at least one image
    for fname in os.listdir(dir):
        if fname.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            images_dirs.append(dir)
images_dirs = list(dict.fromkeys(images_dirs))
print(f"images_dirs: {images_dirs}")

LogFolder = os.path.join(dir_to_analyse, 'analysis_files',
                         "logfiles")  # all log files will be placed here
Path(LogFolder).mkdir(parents=True, exist_ok=True)

logFileNames = []
for path in images_dirs:
    path = os.path.normpath(path)
    path_components = path.split(os.sep)
    supplementary_path = path_components[len(os.path.normpath(dir_to_analyse).split(os.sep)):] # take the path components inside the tree (the path inside dir_to_analyse)
    logFileName = '-'.join(supplementary_path) # make one string of it to ensure unique names and no overwriting of logfiles if dirs contain equally named subdirs
    if logFileName == '':
        logFileName = path_components[-1]
    logFileNames.append(logFileName)
print(f"logFileNames: {logFileNames}")

if need_trainings_data:
    out_dir_traindata = os.path.join(dir_to_analyse, 'analysis_files',
                                     "output_traindata")  # individual output dirs will be created automatically
    output_dirs_traindata = []
    for path in images_dirs:
        path = os.path.normpath(path)
        path_components = path.split(os.sep)
        supplementary_path = path_components[len(os.path.normpath(dir_to_analyse).split(os.sep)):]  # take the path components inside the tree (the path inside dir_to_analyse)
        dir_name = '-'.join(supplementary_path)  # make one string of it to ensure unique names and no overwriting of logfiles if dirs contain equally named subdirs
        output_path = os.path.join(out_dir_traindata, dir_name)
        output_dirs_traindata.append(output_path)
        Path(os.path.join(output_path, "_anders")).mkdir(parents=True, exist_ok=True)  # create "anders" dir
        for item in read_label_map(path_to_labelmap):  # create the rest of the items in the labelmap as dirs
            subdir = os.path.join(output_path, item)
            Path(subdir).mkdir(parents=True, exist_ok=True)

if need_visual_data:
    out_dir_visualdata = os.path.join(dir_to_analyse, 'analysis_files',
                                      "output_visualdata")  # individual output dirs will be created automatically
    output_dirs_visualdata = []
    for path in images_dirs:
        path = os.path.normpath(path)
        path_components = path.split(os.sep)
        supplementary_path = path_components[len(os.path.normpath(dir_to_analyse).split(os.sep)):]  # take the path components inside the tree (the path inside dir_to_analyse)
        dir_name = '-'.join(supplementary_path)  # make one string of it to ensure unique names and no overwriting of logfiles if dirs contain equally named subdirs
        output_path = os.path.join(out_dir_visualdata, dir_name)
        output_dirs_visualdata.append(output_path)
        Path(os.path.join(output_path, "_anders")).mkdir(parents=True, exist_ok=True)  # create "anders" dir
        for item in read_label_map(path_to_labelmap):  # create the rest of the items in the labelmap as dirs
            subdir = os.path.join(output_path, item)
            Path(subdir).mkdir(parents=True, exist_ok=True)

for i in range(0, len(images_dirs)):
    print("\nProcessing {}...".format(images_dirs[i]))
    logFile = open(LogFolder + os.sep + logFileNames[i] + ".csv", "w+")
    detector = DetectorTF2(model_path, path_to_labelmap, class_id=id_list)
    if need_trainings_data and need_visual_data:
        DetectImagesFromFolder(detector, images_dirs[i], output_dirs_traindata[i], output_dirs_visualdata[i])
    elif need_trainings_data and not need_visual_data:
        DetectImagesFromFolder(detector, images_dirs[i], output_dirs_traindata[i], "")
    elif not need_trainings_data and need_visual_data:
        DetectImagesFromFolder(detector, images_dirs[i], "", output_dirs_visualdata[i])
    logFile.close()

cv2.destroyAllWindows()


###### Step 3: produce xlsx output
def create_summary_from_csv_files(csvfile_dir):
    xlsxdir = os.path.join(csvfile_dir, 'xslx_files')
    Path(xlsxdir).mkdir(parents=True, exist_ok=True)
    csv_paths = []
    for file in os.listdir(csvfile_dir):
        if file.endswith(".csv"):
            s = os.path.join(csvfile_dir, file)
            csv_paths.append(s)

    for csv_file in csv_paths:
        fileNaam = os.path.splitext(PurePath(csv_file).name)[0].replace("Logfile_", "")
        dfx = pd.read_csv(csv_file, delimiter=',', names=["Name", "Score", "Klasse", "Date", "Time", "Aantal"],
                          skip_blank_lines=True, skipinitialspace=True, engine='python', header=None)
        idnr = 0
        # datumtijd = ""
        Oldtijd = ""
        dfx["TimeID"] = ""

        for index, row in dfx.iterrows():
            # naam = row["Name"]
            date_time_str = row["Date"] + " " + row["Time"]
            datumtijd = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
            if Oldtijd != "":
                tijdVerschil = datumtijd - Oldtijd
                if tijdVerschil.seconds > 30:
                    idnr = idnr + 1
            Oldtijd = datumtijd
            TimeIDval = "D" + str(idnr)
            dfx.at[index, 'TimeID'] = TimeIDval

        ext = str(os.path.splitext(dfx['Name'][0])[-1])
        if analyse_videos:  # add column for the file name
            dfx['File'] = dfx['Name'].str[:-len(ext)].str[:-22]  # remove the timestamp (22 charachters) and extention
        else:
            dfx['File'] = dfx['Name'].str[:-len(ext)]  # remove extention

        dfxSum = dfx.groupby(['TimeID', 'Klasse']).agg({'Score': 'sum', 'Aantal': 'max'})[
            ['Score', 'Aantal']].reset_index()
        dfxHigh = dfxSum[~dfxSum['Klasse'].isin(['overig', 'muistunnel', 'vleermuis'])].groupby('TimeID').apply(
            lambda x: x.loc[x.Score == x.Score.max(), ['Score', 'Klasse', 'Aantal']]).reset_index()
        dfxHigh['finKlasse'] = dfxHigh['Klasse']
        dfxHigh['maxAantal'] = dfxHigh['Aantal']
        dfx = pd.merge(dfx, dfxHigh[['TimeID', 'finKlasse', 'maxAantal']], on='TimeID', how='left')
        dfx.finKlasse.fillna(dfx.Klasse, inplace=True)
        dfx.to_excel(xlsxdir + os.sep + fileNaam + ".xlsx")

    # make summaries of these xlsx files so humans can interpret them
    final_output_dir = os.path.join(dir_to_analyse, 'final_output_xlsx')
    Path(final_output_dir).mkdir(parents=True, exist_ok=True)
    xlsx_paths = []
    for file in os.listdir(xlsxdir):
        if file.endswith(".xlsx"):
            s = os.path.join(xlsxdir, file)
            xlsx_paths.append(s)

    print("")
    for file in xlsx_paths:
        file_name = os.path.splitext(PurePath(file).name)[0]
        naam = file_name + "_detections.xlsx"
        outputFile = os.path.join(final_output_dir, naam)
        df = pd.read_excel(file)
        files = df.groupby(['TimeID']).agg(
            files=('File', 'unique'))  # create extra df with TimeID and its associated files
        summary = df.groupby(['TimeID']).first()
        dfnew = summary[['finKlasse', 'maxAantal', 'Date', 'Time']]
        dfnew = dfnew.sort_values(by=['Date', 'Time'])
        dfnew = pd.merge(dfnew, files, on='TimeID', how='left')
        dfnew.to_excel(outputFile)
        print("{} summarised and saved as xlsx at '{}'.".format(file_name, outputFile))


create_summary_from_csv_files(LogFolder)

# Step 4: How long did it take?
t1 = time.time()
print("\nTime elapsed to analyse {} {}: {}".format(n_files, 'videos' if analyse_videos else 'images',
                                                   str(time.strftime("%Hh%Mm%Ss", time.gmtime(t1 - t0)))))
print("On average that is {} second per {}".format(str(round((t1 - t0) / n_files,3)),
                                            'video' if analyse_videos else 'image'))
