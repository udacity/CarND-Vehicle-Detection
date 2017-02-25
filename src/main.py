# Imports
import cv2
from pipeline import Pipeline
from Line import Line
from vehicleFinder import VehicleFinder
import sys

# Instantiate utils (calibrates camera)
# TODO: if using this pipeline, bring over additional videos and images
# pipeline = Pipeline()
vehicle_finder = VehicleFinder()
right_line = Line(10)
left_line = Line(10)
lines = [left_line, right_line]
counter = 0
max_lost_frames = 2
lost_frame = 0

# pipeline.runDebugPipeline(max_lost_frames, left_line, right_line)
#
# pipeline.runVideoPipeline(max_lost_frames, left_line, right_line)

if len(sys.argv) > 1 and str(sys.argv[1]) == 'validate':
    vehicle_finder.validate()
elif len(sys.argv) > 1 and str(sys.argv[1]) == 'train':
    vehicle_finder.train()
elif len(sys.argv) > 1 and str(sys.argv[1]) == 'run':
    vehicle_finder.run_video_pipeline()
else:
    vehicle_finder.train()
    vehicle_finder.validate()
