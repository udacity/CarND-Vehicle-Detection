import cv2
import numpy as np
import pickle
from util import *
from helpers import *
from scipy.ndimage.measurements import label
from collections import deque
import skvideo.io
import time


video_file = 'project_video.mp4'
model_file = 'model/raw/LinearSVC.p'
scaler_file = 'model/raw/X_scaler.p'

# load video into memory
video = skvideo.io.vread(video_file)

# create video generator
# g_video = skvideo.io.vreader(video_file)

out_path = './output_images/frames/'

with open(model_file, 'rb') as file:
    clf = pickle.load(file)

with open(scaler_file, 'rb') as file:
    X_scaler = pickle.load(file)

heatmap_cache = deque(maxlen=8)
scale = 1.0

for n_fr, fr in enumerate(video):
    s_time = time.time()
    frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    img = annotate_img(frame, heatmap_cache, clf, X_scaler)
    f_name = out_path + str(n_fr) + '.jpg'
    cv2.imwrite(f_name, img)
    end_time = time.time()
    print('{:.2f} s/frame'.format(end_time - s_time))
    print('{} to 1260'.format(n_fr))
