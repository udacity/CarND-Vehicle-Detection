import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Read in cars and notcars
cars = []
notcars = []
images = glob.glob('./vehicles/*/*.png')
for image in images:
    cars.append(image)
images = glob.glob('./non-vehicles/*/*.png')
for image in images:
    notcars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
print("number of car images", len(cars))
print("number of noncar images", len(notcars))
# sample_size = 500
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]

# TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 60  # HOG orientations
pix_per_cell = 6  # HOG pixels per cell
cell_per_block = 3  # HOG cells per block
hog_channel = 1  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
# y_start_stop = [450, 700]  # Min and max in y to search in slide_window()

car_features = extract_features(cars, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
print("car_features created")
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)
print("notcar features created")
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)


# dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
# X_scaler = dist_pickle["scaler"]
dist_pickle = {"color_space": color_space,
               "orient": orient,
               "pix_per_cell": pix_per_cell,
               "cell_per_block": cell_per_block,
               "hog_channel": hog_channel,
               "spatial_size": spatial_size,
               "hist_bins": hist_bins,
               "scaler": X_scaler,
               }
print(dist_pickle)
pickle.dump(dist_pickle, open("svc_pickle.p", "wb"))

