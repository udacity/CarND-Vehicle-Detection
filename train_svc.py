import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

# Read in cars and notcars in RGB
cars = []
notcars = []
# images = glob.glob('./train_images/vehicles/*/*.png') # windows
images = glob.glob('./train_images/vehicles/*/*.png') # mac
for image in images:
    cars.append(image)

# images = glob.glob('./train_images/non-vehicles/*/*.png') # windows
images = glob.glob('./train_images/non-vehicles/*/*.png') # mac
for image in images:
    notcars.append(image)

# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time

# sample_size = 2000
# cars = cars[0:sample_size]
# notcars = notcars[0:sample_size]
print("number of car images", len(cars))
print("number of noncar images", len(notcars))

### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 60  # HOG orientations
pix_per_cell = 16  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
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

print("car feature extracted")
notcar_features = extract_features(notcars, color_space=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

print("not car feature extracted")
X = np.vstack((car_features, notcar_features)).astype(np.float64)
print("X shape is", X.shape)
# Fit a per-column scaler
print("X shape is ", X.shape)
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.3, random_state=rand_state)

print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

# joblib.dump(svc, 'svc_model.pkl')

dist_pickle = {"color_space": color_space,
               "orient": orient,
               "pix_per_cell": pix_per_cell,
               "cell_per_block": cell_per_block,
               "hog_channel": hog_channel,
               "spatial_size": spatial_size,
               "hist_bins": hist_bins,
               "scaler": X_scaler,
               "svc": svc
               }
print("print out dist_pickle", dist_pickle)

with open('svc_pickle.p', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dist_pickle, f, pickle.HIGHEST_PROTOCOL)
print("model saved to svc_pickle.p")
