param = {
    # for search
    'ref_window_size': 64,
    'l1_sliding_window_size': 96,
    'l2_sliding_window_size': 64,
    'xstart': None,
    'xstop': None,
    'ystart': 400, # just above horizon
    'ystop': 675, # just above the car hood
    'overlap': 0.75,
    'scale': 2,
    # for spatial feature
    'spatial_feat': False,
    'spatial_space': 'RGB',
    'spatial_chan': 'ALL',
    'spatial_size': (32, 32),
    # for color histogram feature
    'color_hist_feat': True,
    'color_space': 'HLS',
    'color_bins': 16,
    'color_bins_range': (0,256),
    'color_chan': 'S',
    # for HOG feature
    'hog_feat': True,
    'hog_space': 'YCrCb',
    'hog_chan': 'ALL',
    'orient': 9,
    'pix_per_cell': 8,
    'cell_per_block': 2,
    # for smoothing
    'thresh': 5,
    'smooth': 41
}

