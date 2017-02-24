import os

import matplotlib.gridspec as gridspec
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import cv2


def extract_files(parent, extension='.png'):
    """

    :param parent:
    :param extension:
    :return:
    """
    file_container = []
    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith(extension):
                file_container.append(os.path.join(root, file))
    return file_container


def display_random_images(image_files, num_of_images=12, images_per_row=6, main_title=None):
    """

    :param image_files:
    :param num_of_images:
    :param images_per_row:
    :param main_title:
    :return:
    """
    random_files = np.random.choice(image_files, num_of_images)
    images = []
    for random_file in random_files:
        images.append(img.imread(random_file))

    grid_space = gridspec.GridSpec(num_of_images // images_per_row + 1, images_per_row)
    grid_space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(images_per_row, num_of_images // images_per_row + 1))

    for index in range(0, num_of_images):
        axis_1 = plt.subplot(grid_space[index])
        axis_1.axis('off')
        axis_1.imshow(images[index])

    if main_title is not None:
        plt.suptitle(main_title)
    plt.show()


def visualize_hog_features(hog_features, images, color_map=None, suptitle=None):
    """

    :param hog_features:
    :param images:
    :param color_map:
    :param suptitle:
    :return:
    """
    num_images = len(images)
    space = gridspec.GridSpec(num_images, 2)
    space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(4, 2 * (num_images // 2 + 1)))

    for index in range(0, num_images*2):
        if index % 2 == 0:
            axis_1 = plt.subplot(space[index])
            axis_1.axis('off')
            axis_1.imshow(images[index // 2], cmap=color_map)
        else:
            axis_2 = plt.subplot(space[index])
            axis_2.axis('off')
            axis_2.imshow(hog_features[index // 2], cmap=color_map)

    if suptitle is not None:
        plt.suptitle(suptitle)
    plt.show()


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def draw_sliding_windows(image, windows, color=(197, 27, 138), thick=3):
    """
    Draw app possible sliding windows on top of the given image.

    :param image:
    :param windows:
    :param color:
    :param thick:
    :return:
    """
    for window in windows:
        cv2.rectangle(image, window[0], window[1], color, thick)
    return image

def apply_threshold(heatmap, threshold):
    """
    Simple unitliy function which encapsulates heap-map thresholding algorithm

    :param heatmap:
    :param threshold:
    :return:
    """
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    """
    Draw boxes on top of the given image

    :param img:
    :param labels:
    :return:
    """
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (197, 27, 138), 3)
    # Return the image
    return img


if __name__ == '__main__':
    import vehicle
    import cv2

    vehicle_files_dir = './data/vehicles/'
    non_vehicle_files_dir = './data/non-vehicles/'

    vehicle_files = extract_files(vehicle_files_dir)
    non_vehicle_files = extract_files(non_vehicle_files_dir)

    print('Number of vehicle files: {}'.format(len(vehicle_files)))
    print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))
    image = img.imread(vehicle_files[0])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = vehicle.get_hog_features(gray, orient,
                                                   pix_per_cell, cell_per_block,
                                                   vis=True, feature_vec=False)

    # Plot the examples
    a = []
    b = []
    a.append(hog_image)
    b.append(gray)
    visualize_hog_features(a, b)
