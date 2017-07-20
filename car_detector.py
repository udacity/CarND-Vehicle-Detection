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

        feature_img = self.classifier.prepare_img(img)

        sub_imgs = []
        windows  = list(self.sliding_windows(x_bounds, y_bounds, window_size, overlap))

        # extract images
        for w in windows:
            sub_imgs.append(cv2.resize(feature_img[w[0][1]:w[1][1],w[0][0]:w[1][0]], (64,64)))

        # classify in batch
        pred = self.classifier.predict(sub_imgs)

        # save the rectangle if we think it's a car
        for w in np.array(windows)[pred==0]:
            self.detected_rects.append(DetectedRect(w[0], w[1], 1))


    def generate_heatmap(self):
        heatmap = np.zeros((720, 1280), dtype=np.float32)

        for rect in self.detected_rects:
            heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]] += 1

        for rect in self.detected_rects:
            if np.max(heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]]) > 2:
                heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]] += 2
        
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

        #self.find_car_rects(img, (32, img_w), (384, 672), 3)
        #self.find_car_rects(img, ( 0, img_w), (400, 656), 2)
        #self.find_car_rects(img, (32, img_w), (384, 576), 1.5)
        #self.find_car_rects(img, (400, img_w), (400, 496), 1)

        self.find_car_rects_sw(img, (400, img_w), (400, 496), 64, 0.5)
        self.find_car_rects_sw(img, ( 32, img_w), (384, 576), 96, 0.75)
        self.find_car_rects_sw(img, (  0, img_w), (400, 656), 128, 0.75)
        #self.find_car_rects_sw(img, ( 32, img_w), (384, 672), 196)
        #self.find_car_rects_sw(img, (  0, img_w), (400, 656), 256)

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
            cv2.rectangle(draw_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

        return draw_img

    def output_detection_windows(self, img, x_bounds, y_bounds, scale):
        region_w = int((x_bounds[1] - x_bounds[0]) / scale)
        region_h = int((y_bounds[1] - y_bounds[0]) / scale)

        num_x_blocks = (region_w // self.classifier.hog_pixels_per_cell) + 1
        num_y_blocks = (region_h // self.classifier.hog_pixels_per_cell) + 1
        window_size  = 64           # XXX move to classifier
        num_blocks_per_window = (window_size // self.classifier.hog_pixels_per_cell) - self.classifier.hog_cells_per_block + 1
        cells_per_step = 2          
        num_x_steps  = (num_x_blocks - num_blocks_per_window) // cells_per_step
        num_y_steps  = (num_y_blocks - num_blocks_per_window) // cells_per_step

        for x_block in range(num_x_steps):
            for y_block in range(num_y_steps):
                x = np.int(x_block * cells_per_step * self.classifier.hog_pixels_per_cell * scale) + x_bounds[0]
                y = np.int(y_block * cells_per_step * self.classifier.hog_pixels_per_cell * scale) + y_bounds[0]
                s = np.int(window_size * scale)

                yield (x, y), (x+s, y+s)

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



