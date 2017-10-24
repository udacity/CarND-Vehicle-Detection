import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
import numpy as np


def get_img_array_from_path(path, type='jpeg', r=True):

    imgs = []
    imgs_path = glob.glob(path + '*.' + type, recursive=r)
    print('loading images from: ', imgs_path)

    for file in imgs_path:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(64, 64))
        imgs.append(img)

    return np.array(imgs)


# Define a function to return some characteristics of the dataset
def peek_data(car_list, notcar_list):
    """
    Get some characteristics about dataset.
    :param car_list: img list of cars
    :param notcar_list: img list of not_cars
    :return: randomly-selected example imgs dict, data characteristics dict
    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar

    idx = random.randint(0, len(car_list))
    example_imgs ={'cars': car_list[idx], 'noncars': notcar_list[idx]}
    # Define a key "image_shape" and store the test image shape 3-tuple
    data_dict["image_shape"] = example_imgs['cars'].shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = example_imgs['cars'].dtype
    # Return data_dict
    return example_imgs, data_dict


# helper to display images in a path
def peek_img_path(path, imgs_per_line=4, w=16, h=20):
    imgs = []
    for file in os.listdir(path):
        imgs.append(Image.open(path + file))

    num = len(imgs)

    plt.figure(figsize=(w,h))
    if num <= imgs_per_line:
        for i in range(num):
            plt.subplot(1, num, i+1)
            plt.imshow(imgs[i])
    else:
        i = 0
        for j in range(imgs_per_line):
            while i < num:
                plt.subplot(num, imgs_per_line, i+1)
                plt.imshow(imgs[i])
                i+=1


def peek_img_list(imgs_list, imgs_per_line=4, w=16, h=20, title='title'):

    num = len(imgs_list)

    plt.figure(figsize=(w,h))
    if num <= imgs_per_line:
        for i in range(num):
            plt.subplot(1, num, i+1)
            plt.imshow(imgs_list[i])
            plt.title(title)
    else:
        i = 0
        for j in range(imgs_per_line):
            while i < num:
                plt.subplot(num, imgs_per_line, i+1)
                plt.imshow(imgs_list[i])
                plt.title(title)
                i+=1


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)