import cv2
import numpy as np
from skimage.feature import hog


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # if vis is True, returns both features and a visualization
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
                                  visualise=True, feature_vector=False)
        return features.ravel(), hog_image
    # otherwise returns features only
    else:

        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell), \
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, \
                       visualise=False, feature_vector=feature_vec)
        return features.ravel()


def bin_spatial(image, size=(32, 32)):
    color1 = cv2.resize(image[:, :, 0], size).ravel()
    color2 = cv2.resize(image[:, :, 1], size).ravel()
    color3 = cv2.resize(image[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    color_1_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_2_hist = np.histogram(img[:, :, 0], bins=nbins)
    color_3_hist = np.histogram(img[:, :, 0], bins=nbins)
    return np.concatenate((color_1_hist[0], color_2_hist[0], color_3_hist[0]))


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(images, cspace='RGB', orient=9, spatial_size=(32, 32), hist_bins=32,
                     pix_per_cell=8, cell_per_block=2,
                     spatial_feat=True, hist_feat=True, hog_feat=True, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for image in images:
        file_features = []
        # Read in each one by one
        # image = mpimg.imread(image)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            # 4) Append features to list
            file_features.append(spatial_features)
        # 5) Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            # 6) Append features to list
            file_features.append(hist_features)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
            file_features.append(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return np.array(features)


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan / nx_pix_per_step) - 1
    ny_windows = np.int(yspan / ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, vis=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            if vis:
                hog_features, hog_image = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                           pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # 8) Append features to list
            img_features.append(hog_features)

    # 9) Return concatenated array of features
    if hog_feat and vis:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        # features = single_img_features(test_img, color_space=color_space,
        #                                spatial_size=spatial_size, hist_bins=hist_bins,
        #                                orient=orient, pix_per_cell=pix_per_cell,
        #                                cell_per_block=cell_per_block,
        #                                hog_channel=hog_channel, spatial_feat=spatial_feat,
        #                                hist_feat=hist_feat, hog_feat=hog_feat, vis=False)
        # 5) Scale extracted features to be fed to classifier
        #plt.imshow(test_img)

        features = extract_features([test_img], cspace=color_space, orient=orient, spatial_size=spatial_size, hist_bins=hist_bins,
                         pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                         spatial_feat=spatial_feat, hist_feat=hog_feat, hog_feat=hist_feat, hog_channel=hog_channel)

        #print(features.shape)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


if __name__ == '__main__':
    vehicle_files_dir = './data/vehicles/'
    non_vehicle_files_dir = './data/non-vehicles/'
    import helper
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split


    vehicle_files = helper.extract_files(vehicle_files_dir)
    vehicle_images = [mpimg.imread(file) for file in vehicle_files]
    #vehicle_images = vehicle_images[1000:5000]

    non_vehicle_files = helper.extract_files(non_vehicle_files_dir)
    non_vehicle_images = [mpimg.imread(file) for file in non_vehicle_files]
    #non_vehicle_images = non_vehicle_images[1000:5000]

    print('Number of vehicle files: {}'.format(len(vehicle_files)))
    print('Number of non-vehicle files: {}'.format(len(non_vehicle_files)))

    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    vehical_features = extract_features(vehicle_images, color_space, orient, spatial_size, hist_bins,
                                                pix_per_cell, cell_per_block, spatial_feat, hist_feat, hog_feat,
                                                hog_channel)
    print(vehical_features.shape)

    non_vehical_features = extract_features(non_vehicle_images, color_space, orient, spatial_size,
                                                    hist_bins, pix_per_cell, cell_per_block, spatial_feat,
                                                    hist_feat, hog_feat, hog_channel)
    print(non_vehical_features.shape)

    features = np.vstack((vehical_features, non_vehical_features)).astype(np.float64)
    print(features.shape)

    scaler = StandardScaler().fit(features)

    X_features = scaler.transform(features)

    y_features = np.hstack((np.ones(len(vehicle_images)), np.zeros(len(non_vehicle_images))))

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_features,
                                                        test_size=0.05, random_state=1024)
    svc = LinearSVC().fit(X_train, y_train)
    accuracy = svc.score(X_test, y_test)

    print('testing :{}'.format(accuracy))
    print('training :{}'.format(svc.score(X_train, y_train)))

    img_path = './test_images/test4.jpg'
    image = mpimg.imread(img_path)
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    y_start_stop = [300, None]  # Min and max in y to search in slide_window()

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                                   xy_window=(100, 100), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, svc, scaler, color_space=color_space,
                                         spatial_size=spatial_size, hist_bins=hist_bins,
                                         orient=orient, pix_per_cell=pix_per_cell,
                                         cell_per_block=cell_per_block,
                                         hog_channel=hog_channel, spatial_feat=spatial_feat,
                                         hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = helper.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    plt.imshow(window_img)
    plt.show()
