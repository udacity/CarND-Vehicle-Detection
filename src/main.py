# Imports
import cv2
from pipeline import Pipeline
from Line import Line

# Instantiate utils (calibrates camera)
pipeline = Pipeline()
right_line = Line(10)
left_line = Line(10)
lines = [left_line, right_line]
counter = 0
max_lost_frames = 2
lost_frame = 0

# pipeline.runDebugPipeline(max_lost_frames, left_line, right_line)
#
# pipeline.runVideoPipeline(max_lost_frames, left_line, right_line)

pipeline.findVehicles()
