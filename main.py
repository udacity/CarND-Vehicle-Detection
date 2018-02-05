from heat_map import *
from hog_subsample import *
from lesson_functions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name

#  project video
# cap = cv2.VideoCapture('project_video.mp4')
# test video
cap = cv2.VideoCapture('test_video.mp4')

# create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('project_result.mp4',fourcc, 25.0, (1280,720))

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

finalList = []
numFrame = 0
heatmap = np.zeros((720, 1280), dtype=float)
threshold = 5


while cap.isOpened():
    # Capture frame-by-frame
    # cap.read() generate a RGB frame 0-255
    ret, frame = cap.read()

    if ret == True:
        numFrame += 1
        # print(frame)
        # get box list for current frame
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = find_cars(frame)

        imgResult, heatmap, finalList = gen_frame_result(frame, numFrame, finalList, bbox, heatmap, threshold)
        # save current frame
        out.write(imgResult)

        # Display the resulting frame
        cv2.imshow('Frame', imgResult)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()