import pandas as pd
import numpy as np
import cv2

import os.path
import os
import random

from tqdm import tqdm

def range_overlap(a_min, a_max, b_min, b_max):
    return (a_min <= b_max) and (b_min <= a_max)

def rect_overlap(r1, r2):
    return range_overlap(r1[0][0], r1[1][0], r2[0][0], r2[1][0]) and \
           range_overlap(r1[0][1], r1[1][1], r2[0][1], r2[1][1])

def extract_image(img, dst, rect):
    sub = img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
    if sub.shape[0] > 0 and sub.shape[1] > 0:
        cv2.imwrite(dst, cv2.resize(sub, (64,64)))

def export_images(img_dict, base_dir, car_dir, notcar_dir):
    car_seq = 0
    notcar_seq = 0

    # create destination directories
    if not os.path.exists(car_dir):
        os.mkdir(car_dir)

    if not os.path.exists(notcar_dir):
        os.mkdir(notcar_dir)

    # process the images
    print("Exporting images")
    for filename, record in tqdm(img_dict.items()):

        # open image
        img = cv2.imread(os.path.join(base_dir, filename))
        
        num_non_cars = 0

        # extract car images
        for r in record['car']:
            if (r[1][1]-r[0][1] > 0) and ((r[1][0]-r[0][0])/(r[1][1] - r[0][1])) > 0.95:          # mostly square
                car_seq = car_seq + 1
                num_non_cars = num_non_cars + 1
                extract_image(img, os.path.join(car_dir, 'car_{:0>4}.png'.format(car_seq)), r)

        # extract as many non-car images to keep the dataset balanced
        while num_non_cars > 0: 
            # random rectangle
            x = random.randrange(0, img.shape[1] - 65)
            y = random.randrange(0, img.shape[0] - 65)
            nr = ((x, y), (x+64, y+64))

            # should overlap any detected object
            overlap = False
            for r in record['car'] + record['other']:
                overlap = rect_overlap(nr, r)
                if overlap:
                    break

            if not overlap:
                notcar_seq = notcar_seq + 1
                num_non_cars = num_non_cars - 1
                extract_image(img, os.path.join(notcar_dir, 'noncar_{:0>4}.png'.format(notcar_seq)), nr)

def crowdai_index(filename):
    # load csv index
    index = pd.read_csv(filename)

    # build a dictionary of all rectangles per image
    print("Building index")
    img_dict = {}

    for _, row in tqdm(index.iterrows()):
        filename = row['Frame']
        xmin, xmax = sorted((row['xmin'], row['xmax']))
        ymin, ymax = sorted((row['ymin'], row['ymax']))
        r = ((xmin, ymin), (xmax, ymax))
        otype = 'car' if row['Label'] == 'Car' else 'other'

        if filename in img_dict:
            record = img_dict[filename]
        else:
            record = {'car': [], 'other': []}
            img_dict[filename] = record
        record[otype].append(r)

    return img_dict

def autti_index(filename):
    # load csv index
    index = pd.read_csv(filename)

    # build a dictionary of all rectangles per image
    print("Building index")
    img_dict = {}

    for _, row in tqdm(index.iterrows()):
        filename = row['frame']
        xmin, xmax = sorted((row['xmin'], row['xmax']))
        ymin, ymax = sorted((row['ymin'], row['ymax']))
        r = ((xmin, ymin), (xmax, ymax))
        otype = 'car' if row['label'] == 'car' and row['occluded'] == 0 else 'other'

        if filename in img_dict:
            record = img_dict[filename]
        else:
            record = {'car': [], 'other': []}
            img_dict[filename] = record
        record[otype].append(r)

    return img_dict

def process_crowdai():
    BASE_DIR = os.path.join("data", "extra", "object-detection-crowdai")
    INDEX_FILE = os.path.join(BASE_DIR, "labels.csv")
    CAR_DIR  = os.path.join("data", "vehicles", "crowdai")
    NOTCAR_DIR  = os.path.join("data", "non-vehicles", "crowdai")

    img_dict = crowdai_index(INDEX_FILE)

    export_images(img_dict, BASE_DIR, CAR_DIR, NOTCAR_DIR)

def process_autti():
    BASE_DIR = os.path.join("data", "extra", "object-dataset")
    INDEX_FILE = os.path.join(BASE_DIR, "labels.csv")
    CAR_DIR  = os.path.join("data", "vehicles", "autti")
    NOTCAR_DIR  = os.path.join("data", "non-vehicles", "autti")

    img_dict = autti_index(INDEX_FILE)

    export_images(img_dict, BASE_DIR, CAR_DIR, NOTCAR_DIR)


def main():
    random.seed()
    process_crowdai()
    process_autti()

if __name__ == '__main__':
    main()