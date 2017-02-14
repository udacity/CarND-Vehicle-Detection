import os

import matplotlib.gridspec as gridspec
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np


def extract_files(parent):
    file_container = []
    for root, dirs, files in os.walk(parent):
        for file in files:
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


if __name__ == '__main__':
    files = extract_files('./data/vehicles')
    display_random_images(files, 7)
