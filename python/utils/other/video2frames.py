import cv2     # for capturing videos
import math   # for mathematical operations
import os
import time
import datetime
from os import listdir
from os.path import isfile, join
from pathlib import Path

def getTimestamp(File):

    modTimesinceEpoc = os.path.getmtime(File)
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
    modtime = time.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    timestampstr = timestamp.replace(":", "")
    timestampstr = timestampstr.replace(" ", "_")
    timestampstr = timestampstr.replace("-", "_")


    return timestampstr, modtime

def cut_video2frames(inPath):
    outPath = os.path.join(os.path.dirname(inPath), 'fotos', os.path.basename(os.path.normpath(inPath)))
    print(outPath)
    Path(outPath).mkdir(parents=True, exist_ok=True)

    onlyfiles = [f for f in listdir(inPath) if isfile(join(inPath, f))]

    for videoFile in onlyfiles:

        naam = videoFile.split(".")[0]
        print(videoFile)

        videoFile = inPath + os.sep + videoFile

        cap = cv2.VideoCapture(videoFile)  # capturing the video from the given path

        timestamp = getTimestamp(videoFile)
        modtime = timestamp[1]

        frameRate = cap.get(5)  # frame rate
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
        duration = int(frame_count / frameRate)  # duration is total number of frames / frames per second

        x = 1
        Y = 1000
        count = 0
        while (cap.isOpened()):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frameId % math.floor(frameRate) == 0):
                fileName = naam + "_" + timestamp[0] + "%d.jpg" % Y;
                Y += 1
                outfile = outPath + os.sep + fileName
                cv2.imwrite(outfile, frame)

                deltaSec = count - duration
                date = datetime.datetime(year=modtime[0], month=modtime[1], day=modtime[2], hour=modtime[3],
                                         minute=modtime[4], second=modtime[5])
                newDate = date + datetime.timedelta(seconds=deltaSec)
                modTime = time.mktime(newDate.timetuple())
                os.utime(outfile, (modTime, modTime))

                count = count + 1

        cap.release()

dirpaths = []



# alle folders in deze dir worden gedaan en er wordt een nieuwe folder 'fotos' aan gemaakt waar de output naar toe gaat
d = r"V:\Projecten\A70_30_65\Marterkist\Data\Input\21-0303\Marterboxen\2e poging"
for path in os.listdir(d):
    full_path = os.path.join(d, path)
    if os.path.isdir(full_path):
        dirpaths.append(full_path)

for dirpath in dirpaths:
    cut_video2frames(dirpath)

print ("All done!")