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

    def find_car_rects(self, img, x_bounds, y_bounds, scale):
        # copy the region of interest
        img_region = img[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]
        region_h, region_w, channels = img_region.shape

        # optionally resize the img
        if scale != 1:
            img_region = cv2.resize(img_region, (0,0), fx=1/scale, fy=1/scale)
            region_h, region_w, channels = img_region.shape

        # define blocks and steps based on the settings of the classifier
        num_x_blocks = (region_w // self.classifier.hog_pixels_per_cell) + 1 # - self.classifier.hog_cells_per_block + 1
        num_y_blocks = (region_h // self.classifier.hog_pixels_per_cell) + 1 # - self.classifier.hog_cells_per_block + 1
        window_size  = 64           # XXX move to classifier
        num_blocks_per_window = (window_size // self.classifier.hog_pixels_per_cell) - self.classifier.hog_cells_per_block + 1
        cells_per_step = 2          
        num_x_steps  = (num_x_blocks - num_blocks_per_window) // cells_per_step
        num_y_steps  = (num_y_blocks - num_blocks_per_window) // cells_per_step

        # compute HOG features for the entire image
        feature_img = self.classifier.prepare_img(img_region)
        hog_features = self.classifier.extract_hog_features(feature_img, False)

        # slide the window of the image
        for x_block in range(num_x_steps):
            for y_block in range(num_y_steps):
                hog_x = x_block * cells_per_step
                hog_y = y_block * cells_per_step

                features = []

                # extract hog features for this step
                for hog_feature in hog_features:
                    hog = np.ravel(hog_feature[hog_y:hog_y+num_blocks_per_window, hog_x:hog_x+num_blocks_per_window])
                    features.append(hog)

                # extract sub images and color features 
                img_x = hog_x * self.classifier.hog_pixels_per_cell
                img_y = hog_y * self.classifier.hog_pixels_per_cell

                color_img = cv2.resize(feature_img[img_y:img_y+window_size, img_x:img_x+window_size], (64,64))
                features.extend(self.classifier.extract_color_features(color_img))

                # scale features
                X_norm = self.classifier.scaler.transform(np.concatenate(features).reshape(1, -1))

                # make a prediction
                pred = self.classifier.classifier_decision_function(X_norm)

                # save the rectange if we think it's a car
                if pred > 0:
                    rect_x = np.int(img_x * scale) + x_bounds[0]
                    rect_y = np.int(img_y * scale) + y_bounds[0]
                    rect_s = np.int(window_size * scale)
                    self.detected_rects.append(DetectedRect((rect_x, rect_y), (rect_x + rect_s, rect_y + rect_s), pred))

    def find_car_rects_sw(self, img, x_bounds, y_bounds, window_size):

        feature_img = self.classifier.prepare_img(img)

        for w in self.sliding_windows(x_bounds, y_bounds, 64, 0.5):
            # extract image
            sub_img = cv2.resize(feature_img[w[0][1]:w[1][1],w[0][0]:w[1][0]], (64,64))

            # extract features
            features = []

            # -- hog
            features.extend(self.classifier.extract_hog_features(sub_img))

            # -- image
            features.extend(self.classifier.extract_color_features(sub_img))

            # scale features
            X_norm = self.classifier.scaler.transform(np.concatenate(features).reshape(1, -1))

            # make a prediction
            pred = self.classifier.classifier_decision_function(X_norm)

            # save the rectange if we think it's a car
            if pred > 0:
                self.detected_rects.append(DetectedRect(w[0], w[1], pred))


    def generate_heatmap(self):
        heatmap = np.zeros((720, 1280), dtype=np.float32)

        for rect in self.detected_rects:
            heatmap[rect.min_c[1]:rect.max_c[1], rect.min_c[0]:rect.max_c[0]] += rect.weight
        
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

        self.find_car_rects(img, (32, img_w), (384, 672), 3)
        self.find_car_rects(img, ( 0, img_w), (400, 656), 2)
        self.find_car_rects(img, (32, img_w), (384, 576), 1.5)
        self.find_car_rects(img, (400, img_w), (400, 496), 1)

        #self.find_car_rects(img, (400, img_w), (400, 496), 64)
        #self.find_car_rects(img, ( 32, img_w), (384, 576), 96)
        #self.find_car_rects(img, (  0, img_w), (400, 656), 128)
        #self.find_car_rects(img, ( 32, img_w), (384, 672), 196)
        #self.find_car_rects(img, (  0, img_w), (400, 656), 256)

        self.generate_heatmap()
        self.process_heatmaps()

    def draw_detected_rects(self, img):
        draw_img = np.copy(img)

        for rect in self.detected_rects:
            cv2.rectangle(draw_img, rect.min_c, rect.max_c, (0, 0, 255), 2)

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



