import cv2
import numpy as np
import time

from car_classifier import CarClassifier
from car_detector import CarDetector
from lane_detector import LaneDetectionPipeline
from camera import Camera

DEBUG_VISUALIZE = False
DEBUG_WINDOW = 'output'

def process_video(filename, car_detector, lane_detector, camera, start_frame=0):
    delay = 0
    frame_idx = 0

    video_src = cv2.VideoCapture(filename)
    video_out = cv2.VideoWriter('output_images/output.avi', cv2.VideoWriter_fourcc(*'DIB '), 25.0, (1280,720))

    while video_src.isOpened():
        ret_ok, frame = video_src.read()
        frame_idx = frame_idx + 1

        if not ret_ok:
            break

        if frame_idx < start_frame:
            continue

        t1 = time.time()

        # apply camera distortion correction
        corrected = camera.undistort(frame)

        # run car detection pipeline
        car_detector.run(corrected)

        # run lane detection pipeline
        corrected = lane_detector.run(corrected)

        # draw rectangles around cars
        result = car_detector.draw_car_rects(corrected)

        # print duration of frame
        t_delta = int((time.time() - t1) * 1000)
        cv2.putText(result, "{} ms - {:.3f} fps".format(t_delta, 1000/t_delta), (900, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,255), 2)
        print('Frame {} : {} ms'.format(frame_idx, t_delta))

        # integrate the heatmap
        if DEBUG_VISUALIZE:
            frame   = cv2.resize(result, (640,360))

            heatmap = cv2.resize(car_detector.heatmap.astype(np.uint8), (640, 360)) 
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)

            heatraw = cv2.resize(car_detector.heatmap_raw.astype(np.uint8), (640,360))
            heatraw = cv2.applyColorMap(heatraw, cv2.COLORMAP_HOT)

            info = np.zeros((360,640,3))

            for idx, r in enumerate(car_detector.car_rects):
                car_heat = car_detector.heatmap_raw[r[1]:r[3],r[0]:r[2]]
                cv2.putText(info, "car {}: min heat = {}, max heat = {}".format(idx, np.min(car_heat), np.max(car_heat)), (20, 30 + (idx * 22)), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2)

            result[:360,:640] = frame
            result[:360:,640:] = heatmap
            result[360:,:640] = info 
            result[360:,640:] = heatraw

        # debug output
        if DEBUG_VISUALIZE:
            cv2.imshow(DEBUG_WINDOW, result)

        # video output 
        video_out.write(result)

        # input when debug-mode is activated
        if DEBUG_VISUALIZE:
            key = cv2.waitKey(delay)
            if key == 27:           # esc
                break
            elif key == ord(' '):   # spacebar
                delay = 0 if delay == 1 else 1
            elif key == ord('s'):
                cv2.imwrite('output_images/captured.jpg', corrected)

    video_out.release()

def debug_output_search_windows(filename, car_detector, camera):
    src_img = camera.undistort(cv2.imread(filename))
    video_out = cv2.VideoWriter('output_images/windows.avi', cv2.VideoWriter_fourcc(*'DIB '), 5.0, (1280,720))

    iterations = [
        {'size' :  64, 'y':(400,496), 'x':(416, src_img.shape[1]), 'overlap':0.50},
        {'size' :  96, 'y':(384,576), 'x':(224, src_img.shape[1]), 'overlap':0.50},
        {'size' : 128, 'y':(400,656), 'x':( 0, src_img.shape[1]), 'overlap':0.75}
    ]

    for settings in iterations:
        frame = np.copy(src_img)

        title = "size = {}, y_range = {} - {}, x_range = {} - {}".format(settings["size"], settings["y"][0], settings["y"][1], settings["x"][0], settings["x"][1])
        cv2.putText(frame, title, (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
        cv2.line(frame, (0, settings["y"][0]), (frame.shape[1], settings["y"][0]), (0, 0, 255), 2)
        cv2.line(frame, (0, settings["y"][1]), (frame.shape[1], settings["y"][1]), (0, 0, 255), 2)

        for w in car_detector.sliding_windows(settings['x'], settings['y'], settings['size'], settings['overlap']):
            cv2.rectangle(frame, w[0], w[1], (0, 255, 0), 2)
            video_out.write(frame)
            cv2.rectangle(frame, w[0], w[1], (255, 0, 0), 2)

        for idx in range(5):
            video_out.write(frame)

    video_out.release()


def main():
    # initialize camera (for distortion correction)
    camera = Camera('camera_udacity')
    
    if not camera.load():
        print("ERROR LOADING CAMERA CALIBRATION")
    
    # initialize car classifier and detector
    clf = CarClassifier.restore('classifier.h5')
    car_detector = CarDetector(clf, heat_threshold=75, num_heat_frames=5)

    # initialize lane detector
    lane_detector = LaneDetectionPipeline()

    # debug visualization
    if DEBUG_VISUALIZE:
        cv2.namedWindow(DEBUG_WINDOW)

    # process video
    process_video("project_video.mp4", car_detector, lane_detector, camera)
    #debug_output_search_windows("test_images/test1.jpg", detector, camera)

    #process_video("../CarND-Advanced-Lane-Lines/challenge_video.mp4", detector, camera)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()