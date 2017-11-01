import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
import glob
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def list_list_mean(list1, list2):
    meanx = (list1[0]+list2[0]) //2
    meany = (list1[1]+list2[1]) //2
    result = [meanx, meany]
    return result


def get_img_array_from_path(path, type='jpeg', r=True, resize=True):

    imgs = []
    imgs_path = glob.glob(path + '*.' + type, recursive=r)
    print('loading images from: ', imgs_path)

    for file in imgs_path:
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, dsize=(64, 64))
        imgs.append(img)

    return np.array(imgs)


def load_img_sequence_from_path(path, type='jpg', start=0, end=1260):

    imgs = []
    print('loading images from: ', path)

    for i in range(start, end):
        file = path + str(i) + '.' + type
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def plot3d(pixels, colors,
           axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255))):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors.reshape((-1, 3)), edgecolors='none')

    return ax  # return Axes3D object for further manipulation


def split_data(scaled_X, y, test_size=0.2):
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=test_size, random_state=rand_state)

    return X_train, X_test, y_train, y_test


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    img_out = img.copy()

    for box in bboxes:
        corner_1 = box[0]
        corner_2 = box[1]

        cv2.rectangle(img_out, corner_1, corner_2, color, thick)

    return img_out


def get_features_norm(feature1, feature2):
    X = np.vstack([feature1, feature2]).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    return X_scaler, scaled_X


# Search for template matches
# and return a list of bounding boxes
def find_matches(img, template_list, method=cv2.TM_CCOEFF_NORMED):
    # Define an empty list to take bbox coords
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    # Iterate through template list
    for temp in template_list:
        # Read in templates one by one
        tmp = mpimg.imread(temp)
        # Use cv2.matchTemplate() to search the image
        result = cv2.matchTemplate(img, tmp, method)
        # Use cv2.minMaxLoc() to extract the location of the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # Append bbox position to list
        bbox_list.append((top_left, bottom_right))
        # Return the list of bounding boxes

    return bbox_list


def image_random_transform(image, prob=0.5, trans_range=15, blur_range=10, rotate_range=15.0, brighten_range=50, TEST=False):
    '''
    Perform random transformations to a source image,
    Returns a list of modified images.

    Random transformations include: bluring, rotating and brightness change

    '''

    imgs = []
    # random translation
    if random.random() < prob:
        r, col, _ = image.shape
        tx = trans_range*np.random.uniform() - trans_range/2.
        ty = trans_range*np.random.uniform() - trans_range/2.
        tM = np.float32([[1, 0, tx], [0, 1, ty]])
        trans = cv2.warpAffine(image, tM, (col, r))
        imgs.append(trans)

    # # random guassian blur
    # if random.random() < prob:
    #
    #     blur_range = int(blur_range / 2) * 2
    #     blur_px = random.choice(range(1, blur_range, 2))
    #
    #     blur = cv2.GaussianBlur(image, (blur_px, blur_px), 0)
    #     imgs.append(blur)
    #
    #     if TEST:
    #         print("bluring")
    #         print("blur:", blur_px)
    # else:
    #     # print("skip bluring")
    #     if TEST:
    #         print('')

    if random.random() < prob:

        # random rotate
        angle = random.uniform(-rotate_range, rotate_range)

        pivot_w = random.uniform(0, image.shape[0])
        pivot_h = random.uniform(0, image.shape[1])

        M = cv2.getRotationMatrix2D((pivot_w, pivot_h), angle, 1)
        rotate = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        imgs.append(rotate)
        if TEST:
            print("rotating")
            print("angle:", angle, "pivot:", pivot_w, pivot_h)
    else:
        # print("skip rotating")
        if TEST:
            print('')

    if random.random() < prob:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        value = random.randint(-brighten_range, brighten_range)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if value > 0:
                    if int(hsv[i, j, 2]) + value > 255:
                        hsv[i, j, 2] = 255
                    else:
                        hsv[i, j, 2] += value
                else:
                    if int(hsv[i, j, 2]) + value < 0:
                        hsv[i, j, 2] = 0
                    else:
                        hsv[i, j, 2] -= -value

        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        imgs.append(hsv)

        if TEST:
            print("brightness")
            print("brighten: ", value)
    else:
        # print("skip brightening")
        if TEST:
            print('')

    if TEST:
        num_imgs = len(imgs)
        plt.figure(figsize=(20, 5))
        plt.subplot(1, num_imgs + 1, 1)
        plt.imshow(image)

        for i in range(num_imgs):
            plt.subplot(1, num_imgs + 1, i + 2)
            plt.imshow(imgs[i])

    return len(imgs), imgs
