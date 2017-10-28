from helpers import *
import skvideo.io
import numpy as np

imgs = load_img_sequence_from_path('./output_images/frames/')
outputdata = np.array(imgs).astype(np.uint8)

skvideo.io.vwrite('output_images/output.mp4', outputdata)