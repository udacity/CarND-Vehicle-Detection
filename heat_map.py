import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def gen_frame_result(img, numFrame, finalList, bbox, heatmap, threshold):
    if numFrame <= 10:
        # add box_list to list
        finalList.append(bbox)
        if len(bbox) > 0:
            for box in bbox:
                heatmap[box[1]:box[3], box[0]:box[2]] += 1
        imgResult = img
    else:
        # delete old list, add new list, apply threshold
        oldList = finalList.pop(0)
        finalList.append(bbox)
        if len(oldList) > 0:
            for box in oldList:
                heatmap[box[1]:box[3], box[0]:box[2]] -= 1
        if len(bbox) > 0:
            for box in bbox:
                heatmap[box[1]:box[3], box[0]:box[2]] += 1
        heatmapSh = apply_threshold(heatmap, threshold)

        # draw_labeled_bboxes
        labels = label(heatmapSh)
        imgResult = draw_labeled_bboxes(img, labels)

    return imgResult, heatmap, finalList