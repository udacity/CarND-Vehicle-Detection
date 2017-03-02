# Imports
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        # self.scales = [0.70, 0.80, 0.90, 1, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45]
        self.scales = [1, 1.25, 1.5, 2]

    def train(self):

        # Read in cars and notcars
        print('Training...')
        cars = glob.glob('../train_images/vehicles/GTI/*.png')
        notcars = glob.glob('../train_images/non-vehicles/Extras/*.png')

        # cars, notcars = shuffle(cars, notcars)

        test_cars = glob.glob('../train_images/vehicles/KITTI_extracted/*.png')
        test_notcars = glob.glob('../train_images/non-vehicles/GTI/*.png')

        # test_cars, test_notcars = shuffle(test_cars, test_notcars)

        color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off

        # Extract features for cars and not cars in both training and testing data sets
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
        scaled_X_t = X_scaler.transform(X_t)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        y_t = np.hstack((np.ones(len(car_test_features)), np.zeros(len(notcar_test_features))))

        # Split up data into randomized training and test sets
        # rand_state = np.random.randint(0, 100)
        # X_train, X_test, y_train, y_test = train_test_split(
        #     scaled_X, y, test_size=0.2, random_state=rand_state)

        X_train, y_train = shuffle(scaled_X, y)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC or linear with probability
        svc = LinearSVC()
        # svc = RandomForestClassifier()
        # svc = SVC(kernel='linear', probability=True)

        # Check the training time for the SVC
        t = time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(scaled_X_t, y_t), 4))
        print(svc)

        # Save the svc and scaler values so I don't have to retrain everytime I run
        joblib.dump(svc, self.svc_filename)
        joblib.dump(X_scaler, self.x_scaler_filename)

    # Run on the testing image set for quick validation that classifier works correctly
    def validate(self):
        print("Validation...")

        # Load in trained svc and x_scaler
        svc = joblib.load(self.svc_filename)
        X_scaler = joblib.load(self.x_scaler_filename)

        # Create heatmaps array and sum
        heatmaps = []
        heatmap_sum = np.zeros((720,1280)).astype(np.float64)

        #Load test images
        images = glob.glob('../test_images/test*.jpg')

        for img in images:
            image = mpimg.imread(img)
            draw_image = np.copy(image)

            # Define where to start and stop the window
            ystart = 400
            ystop = 656

            # Find cars using multiple scales
            all_boxes = find_cars_multi_scale(draw_image, ystart, ystop, self.scales, svc, X_scaler, self.orient,
                                               self.pix_per_cell,
                                               self.cell_per_block, self.spatial_size,
                                               self.hist_bins)
            if len(all_boxes) >= 1:
                heat = np.zeros_like(image[:, :, 0]).astype(np.float)
                # Add heat to each box in box list
                heat = heatMapUtils.add_heat(heat, all_boxes)
                print('after adding heat')
                # plt.imshow(heat)
                # plt.show()
                # Apply threshold to help remove false positives
                heat = heatMapUtils.apply_threshold(heat, 4)
                print('after applying threshold')
                # plt.imshow(heat)
                # plt.show()
                # Visualize the heatmap when displaying
                heatmap = np.clip(heat, 0, 255)
                print('visualize heatmap')
                # plt.imshow(heatmap)
                # plt.show()
                # heatmap_sum += heatmap
                # heatmaps.append(heat)
                # print('append heatmaps')
                # # plt.imshow(heatmap)
                # # plt.show()
                # # subtract off old heat map to keep running sum of last n heatmaps
                # if len(heatmaps) > 5:
                #     old_heatmap = heatmaps.pop(0)
                #     heatmap_sum -= old_heatmap
                #     heatmap_sum = np.clip(heatmap_sum, 0.0, 1000000.0)

                # Find final boxes from heatmap using label function
                labels = label(heatmap)
                print('labels')
                # plt.imshow(labels)
                # plt.imshow()
                draw_img = heatMapUtils.draw_labeled_bboxes(np.copy(image), labels)

                plt.imshow(draw_img)
                plt.show()

    def run_video_pipeline(self):
        print('Running through video...')

        # read video in
        # video = cv2.VideoCapture('../test_video.mp4')
        video = cv2.VideoCapture('../project_video.mp4')

        # to output video
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('../output_project_video.mp4', fourcc, 20.0, (1280, 720))

        # Load in trained svc and x_scaler
        svc = joblib.load(self.svc_filename)
        X_scaler = joblib.load(self.x_scaler_filename)

        # Define where to start and stop the window
        ystart = 400
        ystop = 656

        # Heatmaps
        heatmaps = []
        heatmap_sum = np.zeros((720, 1280)).astype(np.float64)

        # while the video is open, process images
        while video.isOpened():
            # read each frame
            success, image = video.read()

            # run the pipeline on the frame
            all_boxes = find_cars_multi_scale(image, ystart, ystop, self.scales, svc, X_scaler, self.orient, self.pix_per_cell,
                                           self.cell_per_block, self.spatial_size,
                                           self.hist_bins)

            if len(all_boxes) >= 1:
                heat = np.zeros_like(image[:, :, 0]).astype(np.float)
                # Add heat to each box in box list
                heat = heatMapUtils.add_heat(heat, all_boxes)

                # Apply threshold to help remove false positives
                heat = heatMapUtils.apply_threshold(heat, 8)

                # Visualize the heatmap when displaying
                heatmap = np.clip(heat, 0, 255)

                heatmap_sum += heatmap
                heatmaps.append(heat)

                # subtract off old heat map to keep running sum of last n heatmaps
                if len(heatmaps) > 2:
                    old_heatmap = heatmaps.pop(0)
                    heatmap_sum -= old_heatmap
                    heatmap_sum = np.clip(heatmap_sum, 0.0, 1000000.0)

                # Find final boxes from heatmap using label function
                labels = label(heatmap_sum)
                draw_img = heatMapUtils.draw_labeled_bboxes(np.copy(image), labels)

                # Write the output
                out.write(draw_img)
            else:
                # Write the output
                out.write(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # video.release()
        out.release()
        cv2.destroyAllWindows()
