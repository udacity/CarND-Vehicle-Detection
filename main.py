import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

import helper
import vehicle

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from moviepy.editor import VideoFileClip


vehicle_files_dir = './data/vehicles/'
non_vehicle_files_dir = './data/non-vehicles/'

vehicle_files = helper.extract_files(vehicle_files_dir)
vehicle_images = [mpimg.imread(file) for file in vehicle_files]

non_vehicle_files = helper.extract_files(non_vehicle_files_dir)
non_vehicle_images = [mpimg.imread(file) for file in non_vehicle_files]

print('Number of vehicle files: {}'.format(len(vehicle_files)))
print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))


color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
x_start_stop=[None, None]
y_start_stop = [400,600]
xy_window=(96, 85)
xy_overlap=(0.75, 0.75)

C_values = [0.08, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4, 1.8, 2.6]
penalties = ['l2']
losses = ['hinge', 'squared_hinge']

training_accuracies = []
validation_accuracies = []
best_c = 1.0
best_penalty = 'l2'
best_loss = 'hinge'

best_accuracy = 0.0

vehicle_features = vehicle.extract_features(vehicle_images, color_space, orient, spatial_size, hist_bins,
                                            pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat,
                                            hog_channel)
print('Shape of the vehicle features: {}'.format(vehicle_features.shape))

non_vehicle_features = vehicle.extract_features(non_vehicle_images, color_space, orient, spatial_size,
                                                hist_bins, pix_per_cell, cell_per_block, spatial_feat,
                                                hist_feat, hog_feat, hog_channel)
print('Shape of the non-vehicle features: {}'.format(non_vehicle_features.shape))

X_features = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
print('Shape of the entire dataset: {}'.format(vehicle_features.shape))

y_features = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))

for c in C_values:
    for penalty in penalties:
        for loss in losses:
            X_train, X_test, y_train, y_test = train_test_split(X_features, y_features,
                                                                test_size=0.3, random_state=2048)

            scaler = StandardScaler().fit(X_train)

            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            svc = LinearSVC(C=c, penalty=penalty, loss=loss).fit(X_train, y_train)
            accuracy = svc.score(X_test, y_test)
            print('Validation accuracy: {:.4f} with C: {}, panelty: {}, loss: {}'.format(
                accuracy, c, penalty, loss))

            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_c = c
                best_loss = loss
                best_penalty = penalty

            validation_accuracies.append(accuracy)
            training_accuracies.append(svc.score(X_train, y_train))

print('Best validation accuracy: {:.4f}'.format(best_accuracy))
print('Best parameters: C: {}, penalty: {}, loss: {}'.format(best_c, best_penalty, best_loss))

print('')
print('Retaining with best hyper-parameters')

scaler = StandardScaler().fit(X_features)
X_features = scaler.transform(X_features)
svc = LinearSVC(C=best_c, penalty=best_penalty, loss=best_loss).fit(X_features, y_features)

vehicle_detector = vehicle.VehicleDetector(color_space=color_space,
                                  orient=orient,
                                  pix_per_cell=pix_per_cell,
                                  cell_per_block=cell_per_block,
                                  hog_channel=hog_channel,
                                  spatial_size=spatial_size,
                                  hist_bins=hist_bins,
                                  spatial_feat=spatial_feat,
                                  hist_feat=hist_feat,
                                  hog_feat=hog_feat,
                                  y_start_stop=y_start_stop,
                                  x_start_stop=x_start_stop,
                                  xy_window=xy_window,
                                  xy_overlap=xy_overlap,
                                  heat_threshold = 15,
                                  scaler=scaler,
                                  classifier=svc)

output_file = './processed_project_video.mp4'
input_file = './project_video.mp4'

clip = VideoFileClip(input_file)
out_clip = clip.fl_image(vehicle_detector.detect)
out_clip.write_videofile(output_file, audio=False)