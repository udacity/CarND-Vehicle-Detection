import cv2
import numpy as np

secStart = 4
secEnd = 7
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('project_video.mp4')

# create output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('project_video_cut.mp4',fourcc, 25.0, (1280,720))

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

# Read until video is completed
numFrameStart = np.int(secStart * 25)
numFrameEnd = np.int(secEnd * 25)

numFrame = 0
while cap.isOpened() and numFrame <= numFrameEnd:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        if numFrame >= numFrameStart:
            out.write(frame)
        numFrame += 1
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
# Closes all the frames
cv2.destroyAllWindows()