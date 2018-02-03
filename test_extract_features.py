import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import extract_features

from sklearn.model_selection import train_test_split


# Read in cars and notcars
#images = glob.glob('./*.jpeg')
cars = []
notcars = []

images = glob.glob('./vehicles/*/*.png')
for image in images:
    cars.append(image)

# print(cars[-10:-1])
images = glob.glob('./non-vehicles/*/*.png')
for image in images:
    notcars.append(image)

print(notcars[0:10])

test_images = []
images = glob.glob('./test_images/*.jpg')
for image in images:
    test_images.append(image)
print(test_images[:10])

# TODO: Tweak these parameters and see how the results change.
color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 60  # HOG orientations
pix_per_cell = 6  # HOG pixels per cell
cell_per_block = 3  # HOG cells per block
hog_channel = 1  # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [450, 700]  # Min and max in y to search in slide_window()


test_feature = extract_features(test_images, color_space=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
