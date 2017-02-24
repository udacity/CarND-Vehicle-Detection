# Imports
import numpy as np
import cv2
import glob


class Utils:
    def __init__(self):
        self.xm_per_pixel = 3.7/700
        self.ym_per_pixel = 30/720
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrate_camera()

    # Calibrating the camera using chessboard images
    @staticmethod
    def calibrate_camera():
        # The number of x corners in our calibration checkerboard
        nx = 9
        # The number of y corners in our calibration checkerboard
        ny = 6

        # Prepare object points, for the location of each corner (i.e. (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0))
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

        gray_shape = []

        # Step through the list and search for chessboard corners
        for file_name in images:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners, used for debugging
                # img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

        # DEBUGGING: for saving the undistorted images
        # Saving the undistorted images, used in write up
        # counter = 1
        # for file in images:
        #    img = cv2.imread(file)
        #    img = cv2.undistort(img, mtx, dist, None, mtx)
        #    cv2.imwrite('undistorted-images/calibration' + str(counter) + '.jpg', img)
        #    counter += 1

        # Calibrate the camera with all of the images and return these variables that will be used later on
        return cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    # Create a thresholded binary image from the warped on.
    @staticmethod
    def create_threshold_binary(img, kernel_size=5):
        # Apply Gaussian Noise, to help line recognition
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # convert img to HLS colorspace
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # take the S channel of HLS
        s_channel = hls[:, :, 2]

        # convert img to gray colorspace
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sobel x
        sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))

        # Threshold x gradient
        thresh_min = 20
        thresh_max = 100
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

        # Threshold color channel
        s_thresh_min = 150
        s_thresh_max = 255
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        #combining the two in black and white
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # DEBUGGING: to verify binary images are correct
        # plt.imshow(sxbinary, cmap='gray')
        # plt.show()
        #
        # plt.imshow(s_binary, cmap='gray')
        # plt.show()
        #
        # plt.imshow(combined_binary, cmap='gray')
        # plt.show()

        return combined_binary

    # Create a method to undistort
    # not including src and dst points since they are going to remain fixed based on where the camera is positioned and the image
    def warp(self, img):
        # flip the image shape for warpPerspective
        img_size = (img.shape[1], img.shape[0])
        # keeping these as tight to the lines as possible because it pulls in a good amount in the surrounding part outside the lines
        src = np.float32(
            [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
             [((img_size[0] / 6) - 10), img_size[1]],
             [(img_size[0] * 5 / 6) + 60, img_size[1]],
             [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
        # make sure this pulls the perspective "apart", it creates the paralell-ness and is really the key here
        dst = np.float32(
            [[(img_size[0] / 4), 0],
             [(img_size[0] / 4), img_size[1]],
             [(img_size[0] * 3 / 4), img_size[1]],
             [(img_size[0] * 3 / 4), 0]])

        Minv = cv2.getPerspectiveTransform(dst, src)
        M = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        # DEBUGGING: to verify warped image
        # plt.imshow(warped, cmap='gray')
        # plt.show()

        # Save all the warped images, for the writeup
        # test_images = glob.glob('../test_images/test*.jpg')

        # counter = 1
        # for test_img in test_images:
        #    test_img = cv2.imread(test_img)
        #    top_down, perspective_M = undistort_and_warp(test_img, mtx, dist)
        #    cv2.imwrite('warped-images/test' + str(counter) + '.jpg', top_down)
        #    counter += 1

        # returning the M inverse because we're using it outside of here, and not the M
        return warped, Minv

    # Extract the polynomial coefficients from an image given line indices, flag to return in meters or not
    def extract_polynomial(self, img, left_lane_indices, right_lane_indices, in_meters=True):
        # Nonzero pixels from image
        nonzero = img.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        # extract left and right pixel positions
        leftx = nonzerox[left_lane_indices]
        lefty = nonzeroy[left_lane_indices]
        rightx = nonzerox[right_lane_indices]
        righty = nonzeroy[right_lane_indices]

        # put polynomial in meters, if not pixels
        if in_meters:
            left_fit = np.polyfit(lefty * self.ym_per_pixel, leftx * self.xm_per_pixel, 2)
            right_fit = np.polyfit(righty * self.ym_per_pixel, rightx * self.xm_per_pixel, 2)
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    # Find the lane lines with an already existing polynomial
    def find_lanes_with_fit(self, img, left_fit, right_fit, margin):
        # Nonzero pixels from the image
        nonzero = img.nonzero()
        nonzeroy = nonzero[0]
        nonzerox = nonzero[1]

        # Calculate the left polynomial
        left_polynomial = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
        right_polynomial = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]

        # Section off which region
        left_lane_indices = ((nonzerox > (left_polynomial - margin)) & (nonzerox < (left_polynomial + margin)))
        right_lane_indices = ((nonzerox > (right_polynomial - margin)) & (nonzerox < (right_polynomial + margin)))

        # DEBUGGING: to verify the lines have been formed correctly
        # new_left_fit, new_right_fit = self.extract_polynomial(img, left_lane_indices, right_lane_indices, False)

        # # Generate x and y values for plotting
        # ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        # left_fitx = new_left_fit[0] * ploty ** 2 + new_left_fit[1] * ploty + new_left_fit[2]
        # right_fitx = new_right_fit[0] * ploty ** 2 + new_right_fit[1] * ploty + new_right_fit[2]
        #
        # # Create an image to draw on and an image to show the selection window
        # out_img = np.dstack((img, img, img)) * 255
        # window_img = np.zeros_like(out_img)
        # # Color in left and right line pixels
        # out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        #
        # # Generate a polygon to illustrate the search window area
        # # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))
        #
        # # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        # result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        # plt.imshow(result)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        return left_lane_indices, right_lane_indices

    # Find the lanes by using rectangles to search and then fit with polynomial
    def find_lanes_with_rectangle(self, img, margin):
        # histogram to figure out where the peaks are and hence the lane lines
        histogram = np.sum(img[img.shape[0]/2:, :], axis=0)

        #to visualize, for debuggins
        # out_img = np.dstack((img, img, img))*255

        #figure out the middle of the lane (aka where the camera is)
        midpoint = np.int(histogram.shape[0]/2)
        # find the max of the histogram to the left of the midpoint
        leftx_base = np.argmax(histogram[:midpoint])
        # find the mas of the histogram to the right of the midpoint
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # DEBUGGING: to visualize the histogram
        # print(leftx_base)
        # print(rightx_base)
        # plt.plot(histogram)
        # plt.show()

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # number of sliding windows
        nwindows = 9
        # minmum number of pixels to recenter the window
        min_pixels = 50

        # height of the windows
        window_height = np.int(img.shape[0]/nwindows)

        # get all nonzero pixels location (x, y)
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # lists for left and right lanes locations (pixel indices)
        left_lane_indices = []
        right_lane_indices = []

        # iterate through the windows
        for window in range(nwindows):
            # find the window boundaries
            window_y_low = img.shape[0] - (window+1)*window_height # "low" corresponds to the bottom on the image, which is actually the higher pixel value, so it's window+1
            window_y_high = img.shape[0] - window*window_height
            window_x_left_low = leftx_current - margin # leftx_current is in the middle so we need to pull to each side, hence +/- margin
            window_x_left_high = leftx_current + margin
            window_x_right_low = rightx_current - margin
            window_x_right_high = rightx_current + margin

            # draw the rectangle to visualize, for debugging
            # cv2.rectangle(out_img, (window_x_left_low, window_y_low), (window_x_left_high, window_y_high), (0,255,0), 2)
            # cv2.rectangle(out_img, (window_x_right_low, window_y_low), (window_x_right_high, window_y_high), (0,255,0), 2)

            # get all the non zero pixel in our rectangle for left and right sides
            good_left_indices = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_x_left_low) & (nonzerox < window_x_left_high)).nonzero()[0]
            good_right_indices = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & (nonzerox >= window_x_right_low) & (nonzerox < window_x_right_high)).nonzero()[0]

            # add good pixels to the list
            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            # if you found more than the minimum number of pixels, recenter around the mean
            # this would be in the case of a turning line, you would get more pixels on the corner because it's bent?
            if len(good_left_indices) > min_pixels:
                leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
            if len(good_right_indices) > min_pixels:
                rightx_current = np.int(np.mean(nonzerox[good_right_indices]))

        # Concatenate the arrays of indices
        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        # DEBUGGING: to visualize the rectangular windows and the line that was found
        # left_fit, right_fit = self.extract_polynomial(img, left_lane_indices, right_lane_indices, False)
        #
        # # Generate x and y values for plotting
        # ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        #
        # out_img[nonzeroy[left_lane_indices], nonzerox[left_lane_indices]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_indices], nonzerox[right_lane_indices]] = [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        return left_lane_indices, right_lane_indices

    # Find the radius of the curve
    def radius_of_curve(self, img, left_indices, right_indices):
        # Take the bottom of the image
        ymax = img.shape[0]

        # get the polynomial
        left_fit, right_fit = self.extract_polynomial(img, left_indices, right_indices)

        # Calculate the new radii of curvature, in meters
        left_curverad = ((1 + (2*left_fit[0]*ymax*self.ym_per_pixel + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*ymax*self.ym_per_pixel + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

        return left_curverad, right_curverad

    # Find the Offset of the vehicle from the center of the lane
    def calculate_distance_to_center(self, img, left_indices, right_indices):
        # Take the bottome of the image
        ymax = img.shape[0]
        # Take the middle of the image
        xmidpoint = img.shape[1]/2

        # get the polynomial
        left_fit, right_fit = self.extract_polynomial(img, left_indices, right_indices, False)

        # Calculate the polynomial
        left_polynomial = left_fit[0] * (ymax ** 2) + left_fit[1] * ymax + left_fit[2]
        right_polynomial = right_fit[0] * (ymax ** 2) + right_fit[1] * ymax + right_fit[2]

        # Find the midpoint from the two lines
        midpoint = (left_polynomial + right_polynomial) / 2

        # Calculate the difference between the middle of the camera (car) and the lane center, convert to meters
        offset = abs(xmidpoint - midpoint) * self.xm_per_pixel

        return offset

    def 