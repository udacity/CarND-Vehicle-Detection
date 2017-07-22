import numpy as np
import cv2

from collections import deque
from scipy.ndimage.measurements import label

from car_classifier import CarClassifier

class DetectedRect:
    def __init__(self, min_c, max_c, weight):
        self.min_c = min_c
        self.max_c = max_c
        self.weight = weight

class CarDetector:
    def __init__(self, classifier, num_heat_frames=5, heat_threshold=10):
        self.classifier = classifier

        self.detected_rects = []
        self.heatmaps = deque(maxlen=num_heat_frames)
        self.heatmap = None
        self.heatmap_raw = None
        self.heat_threshold = heat_threshold
        self.car_rects = []

    def clear_detections(self):
        self.detected_rects.clear()

    def clear_heatmaps(self):
        self.heatmaps.clear()

    def find_car_rects_sw(self, img, x_bounds, y_bounds, window_size, overlap=0.75):

        sub_imgs = []
        windows  = list(self.sliding_windows(x_bounds, y_bounds, window_size, overlap))

        # extract images
        for w in windows:
            sub_imgs.append(cv2.resize(img[w[0][1]:w[1][1],w[0][0]:w[1][0]], (64,64)))

        # classify in batch
        pred = self.classifier.predict(sub_imgs)

        # save the rectangle if we think it's a car
        for w in np.array(windows)[pred==0]:
            self.detected_rects.append(DetectedRect(w[0], w[1], 1))

    def generate_heatmap(self):
        heatmap = np.zeros((720, 1280), dtype=np.float32)

        for rect in self.detected_rects:
            heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]] += 1

        # increase heat for rectangles with overlap (to make boundingbox big enough for entire car)
        for rect in self.detected_rects:
            if np.max(heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]]) > 2:
                heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]] += 3
        
        self.heatmaps.append(heatmap)

    def process_heatmaps(self):
        # sum the heatmaps
        self.heatmap_raw = np.sum(np.array(self.heatmaps), axis=0, dtype=np.uint32)

        # apply threshold
        self.heatmap = np.copy(self.heatmap_raw)
        self.heatmap[self.heatmap <= self.heat_threshold] = 0 

        # extract labels
        labels = label(self.heatmap)

        # extract boundingboxes of the labels
        self.car_rects.clear()

        for car_number in range(1, labels[1]+1):
            # get the indices of the pixels belonging to the current car
            non_zero = (labels[0] == car_number).nonzero()
            x_coords = np.array(non_zero[1])
            y_coords = np.array(non_zero[0])

            # save the bounding box
            self.car_rects.append((np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)))

    def run(self, img):
        img_h, img_w, _ = img.shape
        self.clear_detections()

        self.find_car_rects_sw(img, (416, img_w), (400, 496), 64, 0.50)
        self.find_car_rects_sw(img, (224, img_w), (384, 576), 96, 0.75)
        self.find_car_rects_sw(img, (  0, img_w), (400, 656), 128, 0.75)

        self.generate_heatmap()
        self.process_heatmaps()

    def draw_detected_rects(self, img):
        draw_img = np.copy(img)

        for rect in self.detected_rects:
            cv2.rectangle(draw_img, tuple(rect.min_c), tuple(rect.max_c), (0, 0, 255), 2)

        return draw_img

    def draw_car_rects(self, img):
        draw_img = np.copy(img)

        for rect in self.car_rects:
            cv2.rectangle(draw_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 4)

        return draw_img

    def sliding_windows(self, x_bounds, y_bounds, window_size, overlap):
        region_w = int(x_bounds[1] - x_bounds[0])
        region_h = int(y_bounds[1] - y_bounds[0])

        pixels_per_step = int(window_size * (1.0 - overlap))

        num_x_windows = 1 + ((region_w - window_size) // pixels_per_step)
        num_y_windows = 1 + ((region_h - window_size) // pixels_per_step)

        for x_window in range(num_x_windows):
            for y_window in range(num_y_windows):
                x = (x_window * pixels_per_step) + x_bounds[0]
                y = (y_window * pixels_per_step) + y_bounds[0]
                yield (x, y), (x+window_size, y+window_size)
