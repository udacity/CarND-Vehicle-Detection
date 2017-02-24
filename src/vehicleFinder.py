# Imports
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from Line import Line
from utils import Utils


class VehicleFinder:
    def __init__(self):
        self.utils = Utils()

    def findVehicles(self):


        return None

    def runDebugPipeline(self, max_lost_frames, left_line, right_line):
        # for debugging
        # read images in
        images = glob.glob('../video_output/frame*.jpg')
        img_from_camera = cv2.imread('../test_images/test2.jpg')
        lost_frame = 0

        for image in images:
            print(image)
            image = cv2.imread(image)
            left_line, right_line, result = self.findVehicles(image, left_line, right_line)

            if left_line.detected == False | right_line.detected == False:
                lost_frame += 1
            else:
                lost_frame = 0

            print('lost_frame:', lost_frame, 'offset:', left_line.line_base_pos, 'left radius:', left_line.radius_of_curvature, 'right radius:', right_line.radius_of_curvature)

            if lost_frame < max_lost_frames:
                left_line.detected = True
                right_line.detected = True
            else:
                left_line.detected = False
                right_line.detected = False

            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            plt.imshow(result)
            plt.show()

    def runVideoPipeline(self, max_lost_frames, left_line, right_line):
        lost_frame = 0
        # read video in
        video = cv2.VideoCapture('../project_video.mp4')
        # to output video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('../output_project_video.mp4', fourcc, 20.0, (1280, 720))

        # while the video is open, process images
        while video.isOpened():
            # read each frame
            success, image = video.read()

            # run the pipeline on the frame
            left_line, right_line, result = self.findVehicles(image, left_line, right_line)

            # see if the line was detected
            if left_line.detected == False | right_line.detected == False:
                # if not add to lost (unusable) frame count
                lost_frame += 1
            else:
                # reset the count, good frame
                lost_frame = 0

            # print line information, offset, and lost frame count
            print('lost_frame:', lost_frame,
                  'offset:', left_line.line_base_pos,
                  'left radius:', left_line.radius_of_curvature,
                  'right radius:', right_line.radius_of_curvature
                  )

            # if lost_frame is under the threshold, just pretend like we don't notice it
            if lost_frame < max_lost_frames:
                left_line.detected = True
                right_line.detected = True
            else:
                left_line.detected = False
                right_line.detected = False

            # Write the output
            out.write(result)
            # show the frames with the lane marked
            cv2.imshow('frame', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        out.release()
        cv2.destroyAllWindows()
