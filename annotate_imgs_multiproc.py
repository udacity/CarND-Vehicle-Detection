import pickle
import skvideo.io
import DetectWorker
import numpy as np
import cv2


# input
video_file = 'project_video.mp4'
model_file = 'model/augmented/LinearSVC.p'
scaler_file = 'model/augmented/X_scaler.p'

# ouput
out_path = './out-frames/'

# set this for multiprocessing
n_workers = 5

# load video into memory
video = skvideo.io.vread(video_file)

n_fr = len(video)
step_size = n_fr//n_workers

workers = []
for i in range(0, n_fr, step_size):
    process = DetectWorker.DetectWorker(model_file, scaler_file, video_file, (i, i + step_size), out_path)
    workers.append(process)

n_workers = len(workers)
print('{} workers'.format(n_workers))

print('Workers starting ... ')
for worker in workers:
    worker.start()

for worker in workers:
    worker.join()
print('Workers joined.')

# print('writing images to disk')
# count = 0
# for worker in workers:
#     for img in worker.imgs:
#         f_name = out_path + str(count) + '.jpg'
#         cv2.imwrite(f_name, img)
#         count += 1

print('exiting main process')


