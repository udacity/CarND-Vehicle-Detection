import cv2
import numpy as np
import pickle
from util import *
from helpers import *
from scipy.ndimage.measurements import label
from collections import deque
import skvideo.io
import time


def annotate_img(img):
    global heatmap_cache, clf, X_scaler, scale

    draw_img = cv2.resize(img, (np.int(img.shape[1]/scale), np.int(img.shape[0]/scale)))

    hog_full = hog_feature_unravel(draw_img, space='HLS', chan='ALL', orient=9, pix_per_cell=8, cell_per_block=2)

    win_128 = slide_window(draw_img, xy_window=(128, 128), xy_overlap=(0.75, 0.75))
    win_64 = slide_window(draw_img, xy_window=(64, 64), xy_overlap=(0.75, 0.75))

    on_win_128 = search_windows_subsample(img, win_128, clf, X_scaler, hog_full, overlap=0.75, pix_per_cell=8, color_space='HLS', hist_bins=32, hist_chan='S', hog_channel='ALL', spatial_feat=False, spatial_size=(16, 16))
    on_win_64 = search_windows_subsample(img, win_64, clf, X_scaler,  hog_full, overlap=0.75, pix_per_cell=8, color_space='HLS', hist_bins=32, hist_chan='S', hog_channel='ALL', spatial_feat=False, spatial_size=(16, 16))

    all_on_windows = on_win_128 + on_win_64

    print('num on windows: ', len(all_on_windows))
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.uint8)
    heatmap = add_heat(heatmap, all_on_windows)

    heatmap_cache.append(heatmap)
    heatmap_graph = apply_threshold(sum(heatmap_cache), 16)
    heatmap_graph = np.clip(heatmap_graph, 0, 255)
    labels = label(heatmap_graph)
    img_out = draw_labeled_bboxes(draw_img, labels)

    return img_out


video_file = 'test_video.mp4'
model_file = 'model/LinearSVC_histS_HOG_trained.p'
scaler_file = 'model/X_scaler.p'

# load video into memory
video = skvideo.io.vread(video_file)

# create video generator
# g_video = skvideo.io.vreader(video_file)

out_path = './testing/'

with open(model_file, 'rb') as file:
    clf = pickle.load(file)

with open(scaler_file, 'rb') as file:
    X_scaler = pickle.load(file)

heatmap_cache = deque(maxlen=10)
scale = 1.0

for n_fr, fr in enumerate(video):
    s_time = time.time()
    frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    img = annotate_img(frame)
    f_name = out_path + str(n_fr) + '.jpg'
    cv2.imwrite(f_name, img)
    end_time = time.time()
    print('{:.2f} s/frame'.format(end_time - s_time))
