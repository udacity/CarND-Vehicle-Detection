import cv2
import numpy as np
import pickle
from util import *
from helpers import *
from scipy.ndimage.measurements import label
from collections import deque


with open('LinearSVC_trained.p', 'rb') as file:
    clf = pickle.load(file)

with open('X_scaler.p', 'rb') as file:
    X_scaler = pickle.load(file)

heatmap_cache = deque(maxlen=10)

def annotate_img(img):
    global heatmap_cache, clf, X_scaler
    win_128 = slide_window(img, xy_window=(128, 128), xy_overlap=(0.8, 0.5))
    win_64 = slide_window(img, xy_window=(64, 64), xy_overlap=(0.8, 0.5))

    img_out = np.copy(img)

    on_win_128 = search_windows(img, win_128, clf, X_scaler, orient=9, pix_per_cell=8, color_space='HLS', hist_bins=32,hog_channel='ALL',spatial_feat=False, spatial_size=(16,16))
    on_win_64 = search_windows(img, win_64, clf, X_scaler, orient=9, pix_per_cell=8, color_space='HLS', hist_bins=32,hog_channel='ALL',spatial_feat=False, spatial_size=(16,16))

    all_on_windows = on_win_128 + on_win_64

    print('num on windows: ', len(all_on_windows))
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.uint8)
    heatmap = add_heat(heatmap, all_on_windows)

    heatmap_cache.append(heatmap)
    heatmap_graph = apply_threshold(sum(heatmap_cache), 16)
    heatmap_graph = np.clip(heatmap_graph, 0, 255)
    labels = label(heatmap_graph)
    img_out = draw_labeled_bboxes(img_out, labels)

    return img_out
