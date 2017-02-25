import numpy as np
import collections


# Define a class to receive the characteristics of each line detection
class Frame:
    def __init__(self, n_iterations):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line, using a deque for rolling storage
        self.previous_box = collections.deque(maxlen=n_iterations)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial values of the last n fits of the line, using a deque for rolling storage
        self.recent_poly = collections.deque(maxlen=n_iterations)
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
