# Imports
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

from Line import Line
from utils import Utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from search_classify import *
from hog_subsample import *
from sklearn.externals import joblib
from sklearn.utils import shuffle
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


class VehicleFinder:
    def __init__(self):
        # self.utils = Utils()
        self.x_scaler_filename = 'x_scaler.pkl'
        self.svc_filename = 'svc.pkl'
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins

    def train(self):

        # Read in cars and notcars
        print('Training...')
        cars = glob.glob('../train_images/vehicles/KITTI_extracted/*.png')
        notcars = glob.glob('../train_images/non-vehicles/GTI/*.png')

        test_cars = glob.glob('../train_images/vehicles/GTI*/*.png')
        test_notcars = glob.glob('../train_images/non-vehicles/Extras/*.png')

        ### TODO: Tweak these parameters and see how the results change.
        color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off

        car_features = extract_features(cars, color_space=color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, color_space=color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)
        car_test_features = extract_features(test_cars, color_space=color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block,
                                        hog_channel=self.hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_test_features = extract_features(test_notcars, color_space=color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        X_t = np.vstack((car_test_features, notcar_test_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        X_scaler_t = StandardScaler().fit(X_t)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        scaled_X_t = X_scaler_t.transform(X_t)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        y_t = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))

        # Split up data into randomized training and test sets
        # rand_state = np.random.randint(0, 100)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     scaled_X, y, test_size=0.2, random_state=rand_state)

        ### Generate data additional data (OPTIONAL!)
        ### and split the data into training/validation/testing sets here.
        ### Feel free to use as many code cells as needed.
        X_train, y_train = shuffle(scaled_X, y)

        # X_test = X_train[:4000].copy()
        # y_test = y_train[:4000].copy()
        #
        # X_train = X_train[4001:]
        # y_train = y_train[4001:]

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(scaled_X_t, y_t), 4))
        print(svc)
        joblib.dump(svc, self.svc_filename)
        joblib.dump(X_scaler, self.x_scaler_filename)

    def validate(self):
        print("Validation...")

        # Load in trained svc and x_scaler
        svc = joblib.load(self.svc_filename)
        X_scaler = joblib.load(self.x_scaler_filename)
        # Check the prediction time for a single sample
        t = time.time()
        ### TODO: Tweak these parameters and see how the results change.
        color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off
        y_start_stop = [375, 650]  # Min and max in y to search in slide_window()
        # Not using this below because i read the png's in using cv2 so they are 0 to 255
        # Uncomment the following line if you extracted training
        # data from .png images (scaled 0 to 1 by mpimg) and the
        # image you are searching is a .jpg (scaled 0 to 255)
        # image = image.astype(np.float32)/255

        images = glob.glob('../test_images/test*.jpg')

        for img in images:
            image = mpimg.imread(img)
            draw_image = np.copy(image)

            # windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
            #                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
            #
            # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
            #                              spatial_size=spatial_size, hist_bins=hist_bins,
            #                              orient=orient, pix_per_cell=pix_per_cell,
            #                              cell_per_block=cell_per_block,
            #                              hog_channel=hog_channel, spatial_feat=spatial_feat,
            #                              hist_feat=hist_feat, hog_feat=hog_feat)
            #
            # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
            # ystart = 400
            # ystop = 656
            # scale = 1.5
            # window_img = find_cars(draw_image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
            #                        cell_per_block, spatial_size,
            #                        hist_bins)
            #
            # plt.imshow(window_img)
            # plt.show()
            scales = [1.25, 1.5, 1.75, 2, 2.25]
            # find_cars_multi_scale(scales)
            ystart = 400
            ystop = 656
            # scale = 1.25
            window_img = find_cars_multi_scale(draw_image, ystart, ystop, scales, svc, X_scaler, self.orient, self.pix_per_cell,
                                   self.cell_per_block, self.spatial_size,
                                   self.hist_bins)

            plt.imshow(window_img)
            plt.show()

    def run_video_pipeline(self):
        lost_frame = 0
        # read video in
        video = cv2.VideoCapture('../test_video.mp4')
        # to output video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('../output_project_video.mp4', fourcc, 20.0, (1280, 720))
        # Load in trained svc and x_scaler
        svc = joblib.load(self.svc_filename)
        X_scaler = joblib.load(self.x_scaler_filename)
        ystart = 400
        ystop = 656
        scales = [1.25, 1.5, 1.75, 2, 2.25]

        # while the video is open, process images
        while video.isOpened():
            # read each frame
            success, image = video.read()

            # run the pipeline on the frame
            result = find_cars_multi_scale(image, ystart, ystop, scales, svc, X_scaler, self.orient, self.pix_per_cell,
                               self.cell_per_block, self.spatial_size,
                               self.hist_bins)

            # Write the output
            out.write(result)
            # show the frames with the lane marked
            # cv2.imshow('frame', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # video.release()
        out.release()
        cv2.destroyAllWindows()
