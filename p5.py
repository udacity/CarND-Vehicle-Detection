"""
   Project 5 - Car detection on roads
   Tariq Rafique
"""
import glob
import os
import pickle
import time

import numpy as np
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle

import cv2
import featurelib as lib
import searchlib as searchlib


"""
Steps to completion
0. Read in the test set of images. make sure set is randomized and split into test/validation set
1. Train a classifier using features that you've selected
2. Validate on the test set
3. Implement sliding window / heatmap approach on a single image
4. Add video implementation
"""


def generate_data():
    vehicle_images = glob.glob("data/vehicles/**/*.png", recursive=True)
    non_vehicle_images = glob.glob(
        "data/non-vehicles/**/*.png", recursive=True)
    Y = np.hstack((np.ones(len(vehicle_images)),
                   np.zeros(len(non_vehicle_images))))
    X = np.hstack((vehicle_images, non_vehicle_images))
    X, Y = shuffle(X, Y)
    return train_test_split(X, Y, test_size=0.33)


def get_images():
    cars = glob.glob("data/vehicles/**/*.png", recursive=True)
    not_cars = glob.glob("data/non-vehicles/**/*.png", recursive=True)

    # cars = glob.glob("data/vehicles/KITTI_extracted/**/*.png", recursive=True)
    # not_cars = glob.glob("data/non-vehicles/Extras/**/*.png", recursive=True)

    # cars = glob.glob(
    #     "data/smaller/onlyvehicles_smallset/**/*.jpeg", recursive=True)
    # not_cars = glob.glob(
    #     "data/smaller/non-vehicles_smallset/**/*.jpeg", recursive=True)
    # return shuffle(cars)[0:1000], shuffle(not_cars)[0:1000]
    return shuffle(cars), shuffle(not_cars)


def train_classifier():
    pickle_file = 'data/svc.p'
    if os.path.exists(pickle_file):
        print("loading pickle file for svc. skipping training", pickle_file)
        obj = pickle.load(open(pickle_file, "rb"))
        return obj['svc'], obj['scaler']

    t1 = time.time()
    print("finding images")
    cars, not_cars = get_images()
    print("extracting features for this many inputs", len(cars) + len(not_cars))
    car_features = lib.extract_features(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = lib.extract_features(not_cars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    t2 = time.time()
    print("Time to extract features", round(t2 - t1, 2))
    # Use a linear SVC
    svc = LinearSVC(C=svc_C)
    svc.fit(X_train, y_train)
    t3 = time.time()
    print(round(t3 - t2, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    print("Saving svc to pickle file")
    pickle.dump({'svc': svc, 'scaler': X_scaler}, open(pickle_file, 'wb'))

    return svc, X_scaler


def process_frame(image, svc, X_scaler, scale, y_start_stop):


    out_img = searchlib.find_cars(image, y_start_stop[0], y_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    # draw_image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    # windows = searchlib.slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
    #                                  xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    # hot_windows = searchlib.search_windows(image, windows, svc, X_scaler, color_space=color_space,
    #                                        spatial_size=spatial_size, hist_bins=hist_bins,
    #                                        orient=orient, pix_per_cell=pix_per_cell,
    #                                        cell_per_block=cell_per_block,
    #                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                                        hist_feat=hist_feat, hog_feat=hog_feat)
    # out_img = searchlib.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    return out_img

def main():
    svc, X_scaler = train_classifier()
    y_start_stop = [400, 700]
    scale = 1.5
    # test_files = glob.glob("test_images/*.jpg")
    # for file in test_files:
    #     image = lib.read_image_in_colorspace(file, color_space="YCrCb")

    #     t4 = time.time()
    #     out_img = process_frame(image, svc, X_scaler, scale, y_start_stop)
    #     t5 = time.time()
    #     print("Displaying image.... Press space to exit ", round(t5-t4, 2))
    #     cv2.imshow('image', out_img)
    #     cv2.waitKey(0)

    test_videos = ['test_video.mp4']
    #test_videos = ['project.mp4']

    for vid_file in test_videos:
        clip = VideoFileClip(vid_file)
        output_clip = clip.fl_image(
            lambda img: process_frame(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb), svc, X_scaler, scale, y_start_stop))
        output_clip.write_videofile(
            'output_' + vid_file, audio=False, threads=4)


color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
svc_C = 1.0

main()
