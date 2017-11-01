import numpy as np
from parameters import param
from collections import deque
from scipy.spatial.distance import pdist


class Vehicle:

    counter = 0

    def __init__(self, bbox):
        self.center = [(bbox[0][0] + bbox[1][0])//2, (bbox[0][1] + bbox[1][1])//2]
        self.vel = [0, 0]
        self.size = [bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
        # at creation, the bbox for this vehicle is at least 96x96
        self.size[0] = max(96, self.size[0])
        self.size[1] = max(96, self.size[1])

        # use a ring buffer to keep a history and get average
        self.centers_cache = deque(maxlen=10)
        self.vels_cache = deque(maxlen=10)
        self.sizes_cache = deque(maxlen=10)

        # to prevent ring buffers from empty, at creation push current properties
        self.centers_cache.append(self.center)
        self.vels_cache.append(self.vel)
        self.sizes_cache.append(self.size)

        Vehicle.counter += 1
        self.id = Vehicle.counter

        # This is used to indicate if this car is no longer found on image
        self.blood = 10
        self.alive = True

    def update(self, bbox_list):

        if self.blood < 0:
            print('>{}: is no long with us, blood: {}'.format(self.id, self.blood))
            self.alive = False
            return

        # constructing a list of bboxes that are close to this vehicle
        bbox_nearby = []
        for bbox in bbox_list:
            bbox_center = [(bbox[0][0] + bbox[1][0])//2, (bbox[0][1] + bbox[1][1])//2]
            if pdist(np.vstack((self.center, bbox_center)))[0] < 1.0 * max(self.size):
                bbox_nearby.append(bbox)
        # print('>{}: n_bbox {}'.format(self.id, len(bbox_nearby)))

        # when nearby bboxes are not found, extrapolate bbox center, keep other properties.
        if len(bbox_nearby) == 0:
            print('No bbox candidate for vehicle {}'.format(self.id))

            # when bood is below 0, we know this vehicle has been lost
            # calling functions will deal with this, e.g. stopping showing this vehicle
            self.blood -= 1
            print('>{}: blood level: {}'.format(self.id, self.blood))

            # extrapolate bbox center, bleed bbox velocity, keep other properties.
            # print('>{}: old center: {}'.format(self.id, self.center))
            self.center[0] = self.center[0] + self.vel[0]
            self.center[1] = self.center[1] + self.vel[1]
            # print('>{}: new center: {}'.format(self.id, self.center))
            self.centers_cache.append(self.center)
            self.center = [int(sum(col) / len(col)) for col in zip(*self.centers_cache)]
            self.vels_cache.append((int(self.vel[0]*0.5), int(self.vel[0]*0.5)))
            # print('>{}: old vel: {}'.format(self.id, self.vel))
            self.vel = [int(sum(col) / len(col)) for col in zip(*self.vels_cache)]
            # print('>{}: new vel: {}'.format(self.id, self.vel))
            # choose not to update sizes
            # self.sizes_cache.append((int(self.size[0]*1.2), int(self.size[1]*1.2)))
            # print('>{}: old size: {}'.format(self.id, self.size))
            # self.size = [int(sum(col) / len(col)) for col in zip(*self.sizes_cache)]
            # print('>{}: new size: {}'.format(self.id, self.size))
        else:
            # restore this car's life to full because new bboxes confirm it as alive
            self.blood = 10
            print('updating vehicle with new bboxes {}'.format(bbox_nearby))

            # use bbox cluster to update center and size
            new_centers = []
            new_sizes = []
            for bbox in bbox_nearby:
                bbox_center = [(bbox[0][0] + bbox[1][0])//2, (bbox[0][1] + bbox[1][1])//2]
                new_centers.append(bbox_center)
                bbox_size = [bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]]
                new_sizes.append(bbox_size)

            # update center
            # print('>{}: old center: {}'.format(self.id, self.center))
            new_center = [int(sum(col) / len(col)) for col in zip(*new_centers)]
            # print('>{}: appending new center {}'.format(self.id, new_center))
            self.centers_cache.append(new_center)
            avg = [int(sum(col) / len(col)) for col in zip(*self.centers_cache)]
            self.center = avg
            # print('>{}: avg center: {}'.format(self.id, avg))
            # print('>{}: new center: {}'.format(self.id, self.center))

            # update velocity in pixel space
            # print('>{}: old vel: {}'.format(self.id, self.vel))
            new_vel = [new_center[0] - self.center[0], new_center[1] - self.center[1]]
            self.vels_cache.append(new_vel)
            avg = [int(sum(col) /len(col)) for col in zip(*self.vels_cache)]
            self.vel = avg
            # print('>{}: new vel: {}'.format(self.id, self.vel))

            # update size
            # print('>{}: old size: {}'.format(self.id, self.size))
            max_x = max(x for x, y in new_sizes)
            max_y = max(y for x, y in new_sizes)

            new_size = [max_x, max_y]

            # when the size change is not drastic, accept it. Otherwise do not take it
            if np.abs(new_size[0] - self.size[0]) < 0.5 * self.size[0] or np.abs(new_size[1] - self.size[1]) < 0.5 * self.size[1]:
                # print('>{}: appending new size {}'.format(self.id, new_size))
                # print('>{}: new sizes: {}'.format(self.id, new_sizes))
                self.sizes_cache.append(new_size)
                # print(self.sizes_cache)
                self.size = [int(sum(col) /len(col)) for col in zip(*self.sizes_cache)]
            else:
                self.size = self.size
            # print('>{}: new size: {}'.format(self.id, self.size))

    def get_bbox(self):
        if self.alive:
            # ((startx, starty), (endx, endy))
            return ((self.center[0] - self.size[0]//2, self.center[1] - self.size[1]//2),
                    (self.center[0] + self.size[0]//2, self.center[1] + self.size[1]//2))
        else:
            return (0,0),(0,0)


