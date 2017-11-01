from multiprocessing import Process
from util import *
from helpers import *
import time
import pickle
import skvideo.io
from collections import deque


# python HOG function does not support threading well,
# so we use multiprocessing to fully benefit from multicore machines.
class DetectWorker(Process):
    counter = 0

    def __init__(self, model_file, scaler_file, video_file, frames=(0, 300), out_path='./'):
        super(DetectWorker, self).__init__()

        with open(model_file, 'rb') as file:
            self.model = pickle.load(file)

        with open(scaler_file, 'rb') as file:
            self.scaler = pickle.load(file)

        self.video = skvideo.io.vread(video_file)
        self.frames = frames
        self.out_path = out_path
        self.imgs = []
        self.heatmap_cache = deque(maxlen=6)
        self.id = type(self).counter
        type(self).counter += 1

    def run(self):
        print('process {} running ...'.format(self.id))
        for n_fr, fr in enumerate(self.video):
            if self.frames[0] <= n_fr < self.frames[1]:
                s_time = time.time()
                frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                img_out = annotate_img_vehicle(img=frame, clf=self.model, X_scaler=self.scaler)
                cv2.imwrite(self.out_path + str(n_fr) + '.jpg', img_out)
                self.imgs.append(img_out)
                end_time = time.time()
                print('process {}: frame: {}/{}, {:.2f} s/frame'.format(self.id, n_fr, self.frames[1], end_time - s_time))
