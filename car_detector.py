import numpy as np
import cv2

from collections import deque
from scipy.ndimage.measurements import label

from car_classifier import CarClassifier

class CarDetector:
    def __init__(self, classifier, num_heat_frames=5, heat_threshold=10):
        self.classifier = classifier

        self.y_min = 400
        self.y_max = 640

        self.detected_rects = []
        self.heatmaps = deque(maxlen=num_heat_frames)
        self.heatmap = None
        self.heat_threshold = heat_threshold
        self.car_rects = []

    def clear_detections(self):
        self.detected_rects.clear()

    def clear_heatmaps(self):
        self.heatmaps.clear()

    def find_car_rects(self, img, scale):
        # copy the region of interest
        img_region = img[self.y_min:self.y_max,:]
        region_h, region_w, channels = img_region.shape

        # optionally resize the img
        if scale != 1:
            img_region = cv2.resize(img_region, (0,0), fx=1/scale, fy=1/scale)
            region_h, region_w, channels = img_region.shape

        # define blocks and steps based on the settings of the classifier
        num_x_blocks = (region_w // self.classifier.hog_pixels_per_cell) - self.classifier.hog_cells_per_block + 1
        num_y_blocks = (region_h // self.classifier.hog_pixels_per_cell) - self.classifier.hog_cells_per_block + 1
        num_feat_per_block = self.classifier.hog_orientations * self.classifier.hog_cells_per_block ** 2
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
                pred = self.classifier.classifier_predict(X_norm)

                # save the rectange if we think it's a car
                if pred == 1:
                    rect_x = np.int(img_x * scale)
                    rect_y = np.int(img_y * scale)
                    rect_s = np.int(window_size * scale)
                    self.detected_rects.append((rect_x, rect_y + self.y_min, 
                                                rect_x + rect_s, rect_y + self.y_min + rect_s))

    def generate_heatmap(self):
        heatmap = np.zeros((720, 1280), dtype=np.uint32)

        for rect in self.detected_rects:
            heatmap[rect[1]:rect[3], rect[0]:rect[2]] += 1
        
        self.heatmaps.append(heatmap)

    def process_heatmaps(self):
        # sum the heatmaps
        self.heatmap = np.sum(np.array(self.heatmaps), axis=0, dtype=np.uint32)

        # apply threshold
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
        self.clear_detections()
        self.find_car_rects(img, 1)
        self.find_car_rects(img, 2)
        self.find_car_rects(img, 3)
        self.find_car_rects(img, 4)
        self.find_car_rects(img, 5)

        self.generate_heatmap()
        self.process_heatmaps()

    def draw_detected_rects(self, img):
        draw_img = np.copy(img)

        for rect in self.detected_rects:
            cv2.rectangle(draw_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

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
