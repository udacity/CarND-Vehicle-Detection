import numpy as np
import cv2

from collections import deque

class LaneDetectionPipeline:
    def __init__(self, smooth_factor=3):
        self.compute_perspective_transform()

        self.debug = True
        self.debug_out = None

        self.threshold_min = {
            'g' : 200,
            'r' : 200,
            's' : 100,
            'l_n' : 180,
            'l_l' : 220,
            'b_n' : 160,
            'b_l' : 200,
        }

        self.lanes = [deque(maxlen=smooth_factor), deque(maxlen=smooth_factor)]
        self.cur_lane = [None, None]
        self.lane_ok = [False, False]

        self.curve_rad = None
        self.center_offset = None

    def set_threshold_min(self, channel, value):
        self.threshold_min[channel] = value

    def get_threshold_min(self, channel):
        return self.threshold_min[channel]

    def compute_perspective_transform(self):
        src_points = np.float32([[565,475], [720,475], [1030,675], [280,675]])
        dst_points = np.float32([[384,200], [896,200], [896,720], [384,720]])

        self.persp_transform = cv2.getPerspectiveTransform(src_points, dst_points)
        self.persp_transform_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    def transform_topdown(self, img):
        return cv2.warpPerspective(img, self.persp_transform,
                                   (img.shape[1], img.shape[0]),
                                   flags=cv2.INTER_LINEAR)

    def gradient_threshold(self, img, dir='x', t_min=20, t_max=100):
        if dir == 'x':
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
        else:
            sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

        scaled_sobel = np.uint8(255*sobel/np.max(sobel))

        # threshold gradient
        mask = np.zeros_like(scaled_sobel)
        mask[(scaled_sobel >= t_min) & (scaled_sobel <= t_max)] = 1

        return mask

    def color_threshold(self, img, c_min=175, c_max=255):
        mask = np.zeros_like(img)
        mask[(img >= c_min) & (img < c_max)] = 1
        return mask
    
    def threshold(self, img):

        img_h, img_w, _ = img.shape

        # convert image to HLS colorspace and scale the luminance channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        l_channel = hls[:,:,1]  * (255 / np.max(hls[:,:,1]))

        # convert image to LAB colorspace and take the 'b' (blue-yellow) channel
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        b_channel = lab[:,:,2] 

        # determine which thresholds to use
        brightness = np.mean(b_channel)
        l_thresh = 'l_n' if np.mean(l_channel) < 100 else 'l_l'
        b_thresh = 'b_n' if np.mean(b_channel) < 150 else 'b_l'

        # apply thresholds
        l_mask = self.color_threshold(l_channel, self.threshold_min[l_thresh], 255)
        b_mask = self.color_threshold(b_channel, self.threshold_min[b_thresh], 255)
        
        # combine the masks
        mask = np.zeros_like(b_mask)
        mask[(l_mask == 1) | (b_mask == 1)] = 1

        return mask
    
    def sliding_window_search(self, img_mask):
        # fill some shorthand variables
        img_w, img_h = (img_mask.shape[1], img_mask.shape[0])
        
        window_n = 9
        window_w = 200
        window_d = window_w // 2
        window_h = img_h // window_n
        recenter_min = 25
        
        # save the indices of the active pixels in the mask
        active_y, active_x = np.nonzero(img_mask)
                
        # create a RGB output image to draw on and  visualize the result
        if self.debug:
            self.debug_out = np.dstack((img_mask, img_mask, img_mask)) * 255
               
        # compute the histogram of the lower half of the image (sum each column)
        hist = np.sum(img_mask[img_h//2:,:], axis=0)
        
        # get the position of the peak in the left and right half of the histogram
        center_x = np.int(img_w / 2)
        
        c_left   = np.argmax(hist[:center_x])
        c_right  = np.argmax(hist[center_x:]) + center_x
        y_max    = img_h
        y_min    = y_max - window_h
        
        # indices into the active_x/y arrays of points that fall with the sliding windows
        l_lane_indices = []
        r_lane_indices = []
        
        for window in range(window_n):
            # compute horizontal edges of the window
            l_x_min = c_left - window_d
            l_x_max = c_left + window_d
            r_x_min = c_right - window_d
            r_x_max = c_right + window_d
                     
            # debug-visualisation
            if self.debug:
                cv2.rectangle(self.debug_out, (l_x_min, y_min), (l_x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(self.debug_out, (r_x_min, y_min), (r_x_max, y_max), (0, 255, 0), 2)
            
            # save coordinates of the detected pixels in the windows            
            l_inds = ((active_y >= y_min) & (active_y < y_max) & (active_x >= l_x_min) & (active_x < l_x_max)).nonzero()[0]
            r_inds = ((active_y >= y_min) & (active_y < y_max) & (active_x >= r_x_min) & (active_x < r_x_max)).nonzero()[0]
                    
            l_lane_indices.append(l_inds)
            r_lane_indices.append(r_inds)
        
            # recenter the windows ?
            if len(l_inds) > recenter_min:
                c_left = np.int(np.mean(active_x[l_inds]))
            if len(r_inds) > recenter_min:
                c_right = np.int(np.mean(active_x[r_inds]))
            
            # move to the next vertical slice
            y_max = y_min
            y_min = y_max - window_h
            
        # lane indices are arrays of arrays: concatenate into 1 array
        l_lane_indices = np.concatenate(l_lane_indices)
        r_lane_indices = np.concatenate(r_lane_indices)
        
        # color the detected pixels
        if self.debug:
            self.debug_out[active_y[l_lane_indices], active_x[l_lane_indices]] = [255, 0, 0]
            self.debug_out[active_y[r_lane_indices], active_x[r_lane_indices]] = [0, 0, 255]
        
        # fit second order polynomial functions to the points
        self.lane_from_points(0, active_x[l_lane_indices], active_y[l_lane_indices])
        self.lane_from_points(1, active_x[r_lane_indices], active_y[r_lane_indices])

    def draw_filled_poly(self, img, poly, margin, color):
        plot_y  = np.linspace(0, img.shape[1]-1, img.shape[0])
        plot_lx = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2] - margin
        plot_rx = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2] + margin

        l_border = np.transpose(np.vstack([plot_lx, plot_y]))
        r_border = np.transpose(np.vstack([plot_rx, plot_y]))

        points = np.concatenate((np.int32(l_border), np.int32(np.flipud(r_border))))
        cv2.fillPoly(img, [points], color)
    
    def margin_search(self, img_mask):
        # fill some shorthand variables
        img_w, img_h = (img_mask.shape[1], img_mask.shape[0])
        margin       = 50
        
        # save the indices of the active pixels in the mask
        active_y, active_x = np.nonzero(img_mask)
                
        # create a RGB output image to draw on and  visualize the result
        if self.debug:
            self.debug_out = np.dstack((img_mask, img_mask, img_mask)) * 255

        # find indices of the pixels within the margins of the previously detected lanes
        l_inds = ((active_x > (self.cur_lane[0][0]*(active_y**2) + self.cur_lane[0][1]*active_y + self.cur_lane[0][2] - margin)) & 
                  (active_x < (self.cur_lane[0][0]*(active_y**2) + self.cur_lane[0][1]*active_y + self.cur_lane[0][2] + margin))) 
        r_inds = ((active_x > (self.cur_lane[1][0]*(active_y**2) + self.cur_lane[1][1]*active_y + self.cur_lane[1][2] - margin)) & 
                  (active_x < (self.cur_lane[1][0]*(active_y**2) + self.cur_lane[1][1]*active_y + self.cur_lane[1][2] + margin))) 

        # color the detected pixels
        if self.debug:
            self.draw_filled_poly(self.debug_out, self.cur_lane[0], margin, [0,255,0])
            self.draw_filled_poly(self.debug_out, self.cur_lane[1], margin, [0,255,0])
            self.debug_out[active_y[l_inds], active_x[l_inds]] = [255, 0, 0]
            self.debug_out[active_y[r_inds], active_x[r_inds]] = [0, 0, 255]

        # fit second order polynomial functions to the points
        self.lane_from_points(0, active_x[l_inds], active_y[l_inds])
        self.lane_from_points(1, active_x[r_inds], active_y[r_inds])

    def lane_from_points(self, side, points_x, points_y):

        # only if the point-arrays contain valid data
        if len(points_x) == 0 or len(points_x) != len(points_y):
            self.lane_ok[side] = False
            return

        # fit the polynomial
        lane = np.polyfit(points_y, points_x, 2)

        # reject outliers
        if self.cur_lane[side] is not None and self.lane_ok[side] == True:
            delta = np.absolute(lane - self.cur_lane[side])

            if delta[0] > 0.001 or delta[1] > 1 or delta[2] > 100:
                self.lane_ok[side] = False
                return

        self.lane_ok[side] = True
        self.lanes[side].append(lane)

    def average_lanes(self):
        if len(self.lanes[0]) > 0:
            self.cur_lane[0] = np.mean(np.array(list(self.lanes[0])), axis=0)
        if len(self.lanes[1]) > 0:
            self.cur_lane[1] = np.mean(np.array(list(self.lanes[1])), axis=0)

    def lane_image(self, img_w, img_h):

        # convert the polynomial functions to a list of points
        plot_y  = np.linspace(0, img_h-1, img_h)
        plot_lx = self.cur_lane[0][0]*plot_y**2 + self.cur_lane[0][1]*plot_y + self.cur_lane[0][2]
        plot_rx = self.cur_lane[1][0]*plot_y**2 + self.cur_lane[1][1]*plot_y + self.cur_lane[1][2]

        # combine the separate x/y arrays in to arrays with (x,y)-pairs
        l_lane = np.transpose(np.vstack([plot_lx, plot_y]))
        r_lane = np.transpose(np.vstack([plot_rx, plot_y]))

        # output to the debug surface
        if self.debug:
            cv2.polylines(self.debug_out, [np.int32(l_lane)], False, [0,255,255], 4)
            cv2.polylines(self.debug_out, [np.int32(r_lane)], False, [0,255,255], 4)

        # create a new image to draw the lane on
        layer_zero  = np.zeros((img_h, img_w), dtype=np.uint8)
        lane_output = np.dstack((layer_zero, layer_zero, layer_zero))
        
        points = np.concatenate((np.int32(l_lane), np.int32(np.flipud(r_lane))))
        cv2.fillPoly(lane_output, [points], (0,255,0))
        
        # reproject to the original space
        lane_unwarp = cv2.warpPerspective(lane_output, self.persp_transform_inv, (img_w, img_h))
        return lane_unwarp

    def lane_measurements(self, img_w, img_h):
        # conversion factors using US standard regulations and manual measurements taken from the binary image
        xm_per_pixel = 3.7 / 530
        ym_per_pixel = 3 / 126

        # convert the polynomial functions to a list of points
        plot_y = np.linspace(0, img_h-1, img_h)
        plot_lx = self.cur_lane[0][0]*plot_y**2 + self.cur_lane[0][1]*plot_y + self.cur_lane[0][2]
        plot_rx = self.cur_lane[1][0]*plot_y**2 + self.cur_lane[1][1]*plot_y + self.cur_lane[1][2]
        
        # refit the polynomial functions in world coordinates
        l_wc = np.polyfit(plot_y*ym_per_pixel, plot_lx*xm_per_pixel, 2)
        r_wc = np.polyfit(plot_y*ym_per_pixel, plot_rx*xm_per_pixel, 2)

        # compute curvature radius at the point closest to the car
        y = img_h - 1
        l_curverad = ((1 + (2*l_wc[0]*y*ym_per_pixel + l_wc[1])**2)**1.5) / np.absolute(2*l_wc[0])
        r_curverad = ((1 + (2*l_wc[0]*y*ym_per_pixel + l_wc[1])**2)**1.5) / np.absolute(2*l_wc[0])
        self.curve_rad = (l_curverad + r_curverad) / 2

        # compute offset from center
        lane_center = (plot_lx[-1] + plot_rx[-1]) / 2
        self.center_offset = ((img_w / 2) - lane_center) * xm_per_pixel

    def run(self, img):
        top_down = self.transform_topdown(img)
        binary = self.threshold(top_down)
        
        if self.lane_ok[0] and self.lane_ok[1]:
            self.margin_search(binary)

        if not (self.lane_ok[0] and self.lane_ok[1]):
            self.sliding_window_search(binary)

        self.average_lanes()

        if not (self.cur_lane[0] is None or self.cur_lane[1] is None):
            self.lane_measurements(binary.shape[1], binary.shape[0])
            cv2.putText(img, "Curvature radius = {:.0f}m".format(self.curve_rad), (20, 30), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,255,255), 2)
            cv2.putText(img, "Offset from center = {:.3f}m".format(self.center_offset), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 2)
            lane_img = self.lane_image(binary.shape[1], binary.shape[0])
            return cv2.addWeighted(img, 1, lane_img, 0.3, 0)
        else:
            return img