import cv2
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist
from skimage.feature import hog
from helpers import *
from parameters import param
from vehicle import Vehicle


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
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


def get_labeled_bboxes(labels):
    # Iterate through all detected car
    bbox_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox_list.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

    return bbox_list


def slide_window(img,
                 x_start_stop=[param['xstart'], param['xstop']],
                 y_start_stop=[param['ystart'], param['ystop']],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Returns a list of windows64, covering the input img."""
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
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows64 in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows64 one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows64
    return window_list


def spatial_bin_feature(img, space=param['spatial_space'], chan=param['spatial_chan'], size=param['spatial_size']):

    if space != 'RGB':
        if space == 'HSV':
            chan_map = {'H':0, 'S':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif space == 'LUV':
            chan_map = {'L':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif space == 'HLS':
            chan_map = {'H':0, 'L':1, 'S':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif space == 'YUV':
            chan_map = {'Y':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif space == 'YCrCb':
            chan_map = {'Y':0, 'Cr':1, 'Cb':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        chan_map = {'R':0, 'G':1, 'B':2}
        feature_image = np.copy(img)

    if chan =='ALL':
        # Use cv2.resize().ravel() to create the feature vector
        features = cv2.resize(feature_image, size).ravel()
        # Return the feature vector
    else:
        features = cv2.resize(feature_image[:,:,chan_map[chan]], size).ravel()

    return features


def color_hist_feature(img, bins=param['color_bins'], bins_range=param['color_bins_range'],
                       space=param['color_space'], chan=param['color_chan']):

    if space != 'RGB':
        if space == 'HSV':
            chan_map = {'H':0, 'S':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif space == 'LUV':
            chan_map = {'L':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif space == 'HLS':
            chan_map = {'H':0, 'L':1, 'S':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif space == 'YUV':
            chan_map = {'Y':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif space == 'YCrCb':
            chan_map = {'Y':0, 'Cr':1, 'Cb':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        chan_map = {'R':0, 'G':1, 'B':2}
        feature_image = np.copy(img)

    if chan == 'ALL':
        hist_features_list = []
        for ch in range(feature_image.shape[2]):
            hist, bin_edges = np.histogram(feature_image[:,:,ch], bins=bins, range=bins_range)
            hist_features_list.append(hist)
        hist_features = np.ravel(hist_features_list)
    else:
        hist_features, bin_edges = np.histogram(feature_image[:,:,chan_map[chan]], bins=bins, range=bins_range)

    return hist_features


def hog_feature(img, space=param['hog_space'], chan=param['hog_chan']):

    if space != 'RGB':
        if space == 'HSV':
            chan_map = {'H':0, 'S':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif space == 'LUV':
            chan_map = {'L':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif space == 'HLS':
            chan_map = {'H':0, 'L':1, 'S':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif space == 'YUV':
            chan_map = {'Y':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif space == 'YCrCb':
            chan_map = {'Y':0, 'Cr':1, 'Cb':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        chan_map = {'R':0, 'G':1, 'B':2}
        feature_image = np.copy(img)

    if chan == 'ALL':
        hog_features_list = []
        for ch in range(feature_image.shape[2]):
            features = hog(feature_image[:,:,ch], orientations=param['orient'],
                           pixels_per_cell=(param['pix_per_cell'], param['pix_per_cell']),
                           cells_per_block=(param['cell_per_block'], param['cell_per_block']),
                           transform_sqrt=False, visualise=False, feature_vector=False)
            hog_features_list.append(np.ravel(features))
        hog_features = np.ravel(hog_features_list)
        # print('hog, ALL shape:',hog_features.shape)
    else:
        hog_features = hog(feature_image[:,:,chan_map[chan]], orientations=param['orient'],
                           pixels_per_cell=(param['pix_per_cell'], param['pix_per_cell']),
                           cells_per_block=(param['cell_per_block'], param['cell_per_block']),
                           transform_sqrt=False, visualise=False, feature_vector=False)
        # print('hog, single shape:',hog_features.shape)
    return hog_features


def hog_feature_unravel(img, space=param['hog_space'], chan=param['hog_chan']):

    if space != 'RGB':
        if space == 'HSV':
            chan_map = {'H':0, 'S':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif space == 'LUV':
            chan_map = {'L':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif space == 'HLS':
            chan_map = {'H':0, 'L':1, 'S':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif space == 'YUV':
            chan_map = {'Y':0, 'U':1, 'V':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif space == 'YCrCb':
            chan_map = {'Y':0, 'Cr':1, 'Cb':2}
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        chan_map = {'R':0, 'G':1, 'B':2}
        feature_image = np.copy(img)

    if chan == 'ALL':
        hog_f = []
        for ch in range(img.shape[2]):
            hog_f_ch = hog(feature_image[:,:,ch], orientations=param['orient'],
                           pixels_per_cell=(param['pix_per_cell'], param['pix_per_cell']),
                           cells_per_block=(param['cell_per_block'], param['cell_per_block']),
                           transform_sqrt=False, visualise=False, feature_vector=False)
            # print("hog_f_ch shape: ", hog_f_ch.shape)
            hog_f.append(hog_f_ch)

        return hog_f[0], hog_f[1], hog_f[2]
    else:
        hog_f = hog(feature_image[:,:,chan_map[chan]], orientations=param['orient'],
                           pixels_per_cell=(param['pix_per_cell'], param['pix_per_cell']),
                           cells_per_block=(param['cell_per_block'], param['cell_per_block']),
                           transform_sqrt=False, visualise=False, feature_vector=False)

        return hog_f


def single_img_features(img,
                        spatial_feat=param['spatial_feat'],
                        color_hist_feat=param['color_hist_feat'],
                        hog_feat=param['hog_feat']):

    img_features = []

    if spatial_feat:
        spatial_f = spatial_bin_feature(img)
        img_features.append(spatial_f)
        # print('spatial_feat len: ', len(spatial_f))

    if color_hist_feat:
        hist_f = color_hist_feature(img)
        img_features.append(hist_f)
        # print('color_feat len: ', len(hist_f))

    if hog_feat:
        hog_f = hog_feature(img)
        img_features.append(hog_f)
        # print('hog_f len: ', len(hog_f))

    return np.concatenate(img_features)


def extract_features_from_img_list(img_list,
                                   color_hist_feat=param['color_hist_feat'],
                                   spatial_feat=param['spatial_feat'],
                                   hog_feat=param['hog_feat']):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for img in img_list:
        img_feature = []
        if spatial_feat:
            spatial_f = spatial_bin_feature(img)
            img_feature.append(spatial_f)
        if color_hist_feat:
            hist_f = color_hist_feature(img)
            img_feature.append(hist_f)
        if hog_feat:
            hog_f = hog_feature(img)
            img_feature.append(hog_f)
        features.append(np.concatenate(img_feature))

    return np.array(features)


def search_windows_naive(img, windows, clf, scaler, ref_window_size=param['ref_window_size']):
    # 1) Create an empty list to receive positive detection windows64
    on_windows = []
    # 2) Iterate over all windows64 in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]],
                              (ref_window_size, ref_window_size))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img)
        # print('total feature len: ', len(features))
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows64 for positive detections
    return on_windows


def annotate_img(img, heatmap_cache, clf, X_scaler, thresh=param['thresh'], smooth=param['smooth'],
                 l1_size=param['l1_sliding_window_size'], l2_size=param['l2_sliding_window_size']):

    win_l1 = slide_window(img, xy_window=(l1_size, l1_size), xy_overlap=(param['overlap'], param['overlap']))
    win_l2 = slide_window(img, xy_window=(l2_size, l2_size), xy_overlap=(param['overlap'], param['overlap']))

    on_win_l1 = search_windows_naive(img, win_l1, clf, X_scaler)
    on_win_l2 = search_windows_naive(img, win_l2, clf, X_scaler)

    all_on_windows = on_win_l1 + on_win_l2

    # cv2.imshow('on_windows',draw_boxes(img,all_on_windows))
    # cv2.waitKey(2000)

    print('num on_windows: ', len(all_on_windows))
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.uint8)
    heatmap = add_heat(heatmap, all_on_windows)
    heatmap = apply_threshold(heatmap, 1)

    heatmap_cache.append(heatmap)
    heatmap_graph = apply_threshold(sum(heatmap_cache), thresh)
    heatmap_graph = np.clip(heatmap_graph, 0, 255)
    heatmap_graph = cv2.GaussianBlur(heatmap_graph, (smooth,smooth), 0)
    labels = label(heatmap_graph)

    img_out = draw_labeled_bboxes(img, labels)

    return img_out

# this is used by annotate_img_vehicle to store a list of cars found
vehicle_list = []


def annotate_img_vehicle(img, clf, X_scaler, vehicles=vehicle_list,
                 l1_size=param['l1_sliding_window_size'], l2_size=param['l2_sliding_window_size'], smooth=param['smooth']):
    win_l1 = slide_window(img, xy_window=(l1_size, l1_size), xy_overlap=(param['overlap'], param['overlap']))
    win_l2 = slide_window(img, xy_window=(l2_size, l2_size), xy_overlap=(param['overlap'], param['overlap']))

    on_win_l1 = search_windows_naive(img, win_l1, clf, X_scaler)
    on_win_l2 = search_windows_naive(img, win_l2, clf, X_scaler)

    all_on_windows = on_win_l1 + on_win_l2

    print('num on_windows: ', len(all_on_windows))
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.uint8)
    heatmap = add_heat(heatmap, all_on_windows)

    # checking new cars using a higher threshold to be more sure about a car
    heatmap_high_confidence = apply_threshold(heatmap, 4)
    heatmap_high_confidence = cv2.GaussianBlur(heatmap_high_confidence, (smooth, smooth), 0)
    bbox_list_high_confidence = get_labeled_bboxes(label(heatmap_high_confidence))

    print('alive vehicles: ', len(vehicles))
    for bbox in bbox_list_high_confidence:
        bbox_center = [(bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2]

        # first car creation
        if len(vehicles) == 0:
            vehicles.append(Vehicle(bbox))
            print('create first car {}'.format(vehicles[-1].id))
        # there are cars already, check if this bbox belongs to any of them
        else:
            exist_flag = 0
            for car in vehicles:
                # print('bbox - car {} distance: {}'.format(car.id, pdist(np.vstack((car.center, bbox_center)))[0]))
                if pdist(np.vstack((car.center, bbox_center)))[0] < max(car.size):
                    exist_flag = 1
                    break
                    # print('this bbox belongs to existing vehicles')
            if exist_flag == 0:
                vehicles.append(Vehicle(bbox))
                print('Found new car {}'.format(vehicles[-1].id))

    # updating cars
    heatmap_low_confidence = apply_threshold(heatmap, 2)
    bbox_list_low_confidence = get_labeled_bboxes(label(heatmap_low_confidence))

    for car in vehicles:
        car.update(bbox_list_low_confidence)

    # first, check if car is alive, if not drop it from vehicle list
    n_cars_prev = len(vehicles)
    vehicles = [car for car in vehicles if car.alive is True]
    n_cars_current = len(vehicles)
    print('dropping {} non-alive cars'.format(n_cars_prev - n_cars_current))

    # second, merge cars which are too close to each other
    remove_idxes = []
    for i in range(len(vehicles)):
        ref_car = vehicles[i]
        for j in range(i+1, len(vehicles)):
            current_car = vehicles[j]
            if pdist(np.vstack((ref_car.center, current_car.center)))[0] < 16:
                remove_idxes.append(i)
                current_car.center = list_list_mean(current_car.center, ref_car.center)
                current_car.vel = list_list_mean(current_car.vel, ref_car.vel)

    print('merging cars nearby.')
    n_cars_current = len(vehicles)
    print('remove idx: ', remove_idxes)
    vehicles = [car for idx, car in enumerate(vehicles) if idx not in remove_idxes]
    n_cars_merged = len(vehicles)
    print('dropping {} nearby cars'.format(n_cars_current - n_cars_merged))

    bbox_draw = []

    # draw bbox for each car
    for car in vehicles:
        bbox_draw.append(car.get_bbox())

    img_out = draw_boxes(img, bbox_draw)

    return img_out


# needs testing:
def patch_features(img, p_size=64,
                   overlap=param['overlap'],
                   xstart=param['xstart'], xstop=param['xstop'], ystart=param['ystart'], ystop=param['ystop'],
                   spatial_feat=param['spatial_feat'],
                   color_hist_feat=param['color_hist_feat'],
                   hog_feat=param['hog_feat'], hog_full=None, hog_chan=param['hog_chan'],
                   pix_per_cell=param['pix_per_cell'], cell_per_block=param['cell_per_block']):

    imgx = img.shape[1]
    imgy = img.shape[0]

    if xstart == None:
        xstart = 0
    if xstop == None:
        xstop = imgx
    if ystart == None:
        ystart = 0
    if ystop == None:
        ystop = imgy

    img_features = []

    sub_img = cv2.resize(img[xstart:xstart + p_size, ystart:ystart + p_size], (p_size, p_size))

    if spatial_feat:
        spatial_f = spatial_bin_feature(sub_img)
        img_features.append(spatial_f)

    if color_hist_feat:
        hist_f = color_hist_feature(sub_img)
        img_features.append(hist_f)

    if hog_feat:
        # Warning: this implementation needs to be tested.
        cells_per_step = int(pix_per_cell * (1 - overlap))
        nblocks_per_patch = (p_size // pix_per_cell) - cell_per_block + 1

        nxblocks = (imgx // pix_per_cell) - cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_patch) // cells_per_step
        xpix_per_step = imgx // nxsteps
        xpos = (xstart // xpix_per_step) * cells_per_step

        nyblocks = (imgy // pix_per_cell) - cell_per_block + 1
        nysteps = (nyblocks - nblocks_per_patch) // cells_per_step
        ypix_per_step = imgy//nysteps
        ypos = (ystart // ypix_per_step) * cells_per_step

        if hog_chan == 'ALL':
            hog_feature_list = []
            for ch in range(img.shape[2]):
                features = hog_full[ch][ypos:ypos+nblocks_per_patch, xpos:xpos+nblocks_per_patch]
                hog_feature_list.append(np.ravel(features))
            hog_features = np.ravel(hog_feature_list)
        else:
            hog_features = hog_full[ypos:ypos+nblocks_per_patch, xpos:xpos+nblocks_per_patch]

        img_features.append(hog_features)

    return np.concatenate(img_features)


# needs testing:
def search_windows_subsample(img, windows, clf, scaler, hog_full, patch_size=64):
    # 1) Create an empty list to receive positive detection windows64
    on_windows = []
    # 2) Iterate over all windows64 in the list
    for window in windows:
        # 3) Extract the test window from original image
        # test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (patch_size, patch_size))
        # 4) Extract features for that window using single_img_features()
        features = patch_features(img, p_size=patch_size,
                                  xstart=window[0][1], xstop=window[1][1], ystart=window[0][0], ystop=window[1][0])
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows64 for positive detections
    return on_windows


# needs testing: the result does not agree with the naive method
def find_cars_multi(img, clf, X_scaler, ref_window_size=param['ref_window_size'],
                    ystart=param['ystart'], ystop=param['ystop'], xstart=param['xstart'], xstop=param['xstop'],
                    scale=param['scale'], overlap=param['overlap'],
                    spatial_feat=param['spatial_feat'], color_hist_feat=param['color_hist_feat'],
                    hog_feat=param['hog_feat'],
                    pix_per_cell=param['pix_per_cell'], cell_per_block=param['cell_per_block']):

    imgx = img.shape[1]
    imgy = img.shape[0]

    if xstart == None:
        xstart = 0
    if xstop == None:
        xstop = imgx
    if ystart == None:
        ystart = 0
    if ystop == None:
        ystop = imgy

    on_windows = []

    img = img.astype(np.float32) / 255
    img_tosearch = img[ystart:ystop, xstart:xstop]

    for current_scale in range(1, scale+1):
        # print('current scale: ', current_scale)
        if current_scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / current_scale), np.int(imshape[0] / current_scale)))
        single_channel = img_tosearch[:, :, 0]

        # number of blocks in current image
        nxblocks = (single_channel.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (single_channel.shape[0] // pix_per_cell) - cell_per_block + 1
        # nfeat_per_block = orient * cell_per_block ** 2
        # print('n_blocks on current image:', nxblocks)

        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        # no matter which scale, the number of blocks per each window has to be the same as the original
        nblocks_per_window = (ref_window_size // pix_per_cell) - cell_per_block + 1
        cells_per_step = int(pix_per_cell * (1 - overlap))

        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1, hog2, hog3 = hog_feature_unravel(img_tosearch)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # due to scaling, the effective window_size changes
                subimg = cv2.resize(img_tosearch[ytop:ytop + ref_window_size, xleft:xleft + ref_window_size], (ref_window_size, ref_window_size))

                features = []
                if spatial_feat:
                    spatial_f = spatial_bin_feature(img)
                    features.append(spatial_f)
                    # print('spatial_feat len: ', len(spatial_f))

                if color_hist_feat:
                    hist_f = color_hist_feature(subimg)
                    features.append(hist_f)
                    # print('color_feat len: ', len(hist_f))

                if hog_feat:
                    # Extract HOG for this patch
                    hog_feature_list = []
                    # due to scaling, this effectively spans a larger region but keeps the length of HOG vector the same
                    hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feature_list.append(hog_feat1)
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feature_list.append(hog_feat2)
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feature_list.append(hog_feat3)
                    hog_f = np.ravel(hog_feature_list)
                    # print('hog_f len: ', len(hog_f))

                    features.append(hog_f)

                feature = np.concatenate(features)
                # print('total feature len: ', len(feature))

                # Scale features and make a prediction
                test_features = X_scaler.transform(np.array(feature).reshape(1, -1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * current_scale)
                    ytop_draw = np.int(ytop * current_scale)
                    win_draw = np.int(ref_window_size * current_scale)

                    startx = xbox_left - xstart
                    starty = ytop_draw + ystart
                    endx = startx + win_draw
                    endy = starty + win_draw

                    on_windows.append(((startx, starty), (endx, endy)))

    return on_windows


def find_cars(img, clf, X_scaler, ref_window_size=param['ref_window_size'],
              ystart=param['ystart'], ystop=param['ystop'], xstart=param['xstart'], xstop=param['xstop'],
              scale=1.0, overlap=param['overlap'],
              orient=param['orient'], pix_per_cell=param['pix_per_cell'], cell_per_block=param['cell_per_block']):

    on_windows = []
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, xstart:xstop]

    if scale != 1:
        imshape = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    single_channel = img_tosearch[:, :, 0]

    # Define blocks and steps as above
    nxblocks = (single_channel.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (single_channel.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    nblocks_per_window = (ref_window_size // pix_per_cell) - cell_per_block + 1
    cells_per_step = int(pix_per_cell * (1 - overlap))

    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1, hog2, hog3 = hog_feature_unravel(img_tosearch)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            subimg = cv2.resize(img_tosearch[ytop:ytop + ref_window_size, xleft:xleft + ref_window_size], (64, 64))

            features = []

            hist_f = color_hist_feature(subimg)
            features.append(hist_f)

            # Extract HOG for this patch
            hog_feature_list = []
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feature_list.append(hog_feat1)
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feature_list.append(hog_feat2)
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feature_list.append(hog_feat3)
            hog_f = np.ravel(hog_feature_list)

            features.append(hog_f)

            feature = np.concatenate(features)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.array(feature).reshape(1, -1))
            test_prediction = clf.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(ref_window_size * scale)

                startx = xbox_left - xstart
                starty = ytop_draw + ystart
                endx = startx + win_draw
                endy = starty + win_draw

                on_windows.append(((startx, starty), (endx, endy)))

    return on_windows


