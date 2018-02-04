from lesson_functions import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
from heat_map import *
from hog_subsample import *

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('project_video.mp4')
# create output video
out = cv2.VideoWriter('project_result.mp4',fourcc, 25.0, (640,480))

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

finalList = [[[]]]
numFrame = 0
heatmap = np.zeros(720, 1280).astype(np.float)
threshold = 3

# ystart = 400
# ystop = 656
# scale = 1.5
# hog_channel = 2

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        numFrame += 1

        # get box list for current frame
        out_img, bbox = find_cars(frame, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                                  spatial_size, hist_bins, hog_channel)

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