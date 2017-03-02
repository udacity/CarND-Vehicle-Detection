import numpy as np
import collections


# Define a class to receive the characteristics of each line detection
class Frame:
    def __init__(self, n_iterations):
        # x values of the last n fits of the line, using a deque for rolling storage
        self.previous_heatmaps = collections.deque(maxlen=n_iterations)
        # average x values of the fitted line over the last n iterations
        self.averaged_heatmap = None
