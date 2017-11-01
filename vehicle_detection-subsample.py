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
model_file = './model/raw/LinearSVC.p'
scaler_file = './model/raw/X_scaler.p'

# load video into memory
video = skvideo.io.vread(video_file)

# create video generator
# g_video = skvideo.io.vreader(video_file)

out_path = './testing/'
out_video_file = out_path + 'processed.mp4'

with open(model_file, 'rb') as file:
    clf = pickle.load(file)

with open(scaler_file, 'rb') as file:
    X_scaler = pickle.load(file)

heatmap_cache = deque(maxlen=10)

color = {0: (0,0,255), 1: (0,255,0)}
for n_fr, fr in enumerate(video):
    if 150 < n_fr:
        s_time = time.time()
        frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        onwindows = find_cars_multi(frame, clf, X_scaler)

        print('num of on_windows:', len(onwindows))
        windows = draw_boxes(frame, onwindows)
        cv2.imshow('windows', windows)
        cv2.waitKey(2000)

        heatmap = np.zeros_like(frame[:, :, 0]).astype(np.uint8)
        heatmap = add_heat(heatmap, onwindows)
        heatmap = apply_threshold(heatmap, 2)

        heatmap_cache.append(heatmap)
        heatmap_graph = apply_threshold(sum(heatmap_cache), 8)
        heatmap_graph = np.clip(heatmap_graph, 0, 255)
        labels = label(heatmap_graph)
        img_out = draw_labeled_bboxes(frame, labels)

        f_name = out_path + str(n_fr) + '.jpg'
        cv2.imwrite(f_name, img_out)
        end_time = time.time()
        print('frame: {}, {:.2f} s/frame'.format(n_fr, end_time - s_time))
