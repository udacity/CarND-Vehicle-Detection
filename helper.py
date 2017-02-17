import os

import matplotlib.gridspec as gridspec
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np


def extract_files(parent, extension = '.png'):
    file_container = []
    for root, dirs, files in os.walk(parent):
        for file in files:
            if file.endswith(extension):
                file_container.append(os.path.join(root, file))
    return file_container


def display_random_images(image_files, num_of_images=12, images_per_row=6, main_title=None):
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


def visualize_hog_features(hog_features, images, color_map = None):
    num_images = len(images)
    space = gridspec.GridSpec(num_images // 2 + 1, 2)
    space.update(wspace=0.1, hspace=0.1)
    plt.figure(figsize=(6, 6*(num_images // 2 + 1)))

    for index in range(0, num_images):
        axis_1 = plt.subplot(space[index])
        axis_1.axis('off')
        axis_1.imshow(images[index], cmap=color_map)

        axis_1 = plt.subplot(space[index + 1])
        axis_1.axis('off')
        axis_1.imshow(hog_features[index], cmap=color_map)

    plt.show()

if __name__ == '__main__':
    import  vehicle
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
