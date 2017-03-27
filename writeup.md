# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car-not-car]: ./writeup/car-not-car.png
[sliding_window]: ./writeup/sliding-window-multiscale.png
[sliding_window2]: ./writeup/sliding-window2.png
[heatmap]: ./writeup/heatmap.png
[labeledheatmap]: ./writeup/labeledheatmap.png
[final_frame]: ./writeup/final_frame.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I chose to utilize a combination of HOG, spatial binning, and color histogram features.
The code for capturing these features is contained in the file `featurelib.py` in the method `get_hog_features`, `bin_spatial`, and `color_hist` starting at lines `12`, `32`, and `39` respectively

I started by reading in all the `vehicle` and `non-vehicle` images in the `get_images` method on line `34` in `p5.py`. The read image file names for each class (car vs. non-car) were shuffled before being returned. The data set had about 16,000+ images for the categories combined. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][car-not-car]

The images were converted to `YCrCb` space before being fed to the training classifier. Images were all read using `cv2.imread` to ensure that they were in the `[0,255]` range.

The parameters for HOG features was as follows
* `orientations = 9`
* `pixels_per_cell = (8, 8)`
* `cells_per_block = 2`
* `transform_sqrt = True` # for normalization to reduce effects of shadowing & illumination variance

The parameters for spatial binning were
* `spatial_size = (16, 16)`

The parameters for color histogram for each channel were
* `bins = 16`
* `range = (0, 256)`

#### 2. Explain how you settled on your final choice of HOG parameters.
The hog parameters and color space was selected by looking tweaking parameters that gave the best SVM classifier test score. The lecture notes mentioned that cars had a bias towards saturation, so I tried the `HLS`, `HSV` colorspace. Additional colorspaces like `LUV`, `YUV`, and `YCrCb` were tried.

The best result came from `YCrCb` color space with these HOG parameters
* `orientations = 9`
* `pixels_per_cell = (8, 8)`
* `cells_per_block = 2`
* `transform_sqrt = True` # for normalization to reduce effects of shadowing & illumination variance

![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier training code can be found in the `train_classifier` method on line `40` in `p5.py`. The classifier first reads in all the image files for each class. It generates a y-label vector (1 for car, 0 for non-car). For each image, the HOG, spatial binning, and color histogram features are extracted and appended a feature set array. This data is normalized using `StandardScaler` from `sklearn.preprocessing`. The parameters for feature extraction are defined above in section 1.

The data set is then further randomized and split into 80% training and 20% validation set.

I used the `LinearSVC` classifier from `sklearn.svm` with the default parameters. (I did try to change the `C` parameter from `0.01` to `100`, but it didn't yield better results compared to the default of `1`. The network is trained using `svc.fit` and the data is saved to the `data/svc.p` pickle file for quick reuse when predicting labels for images / video.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `find_cars` method on line `7` in `searchlib.py`. The window overlap chosen was 75 percent. Scales of 1.33, 1, 2 were searched. The values were selected by trial and error to show good matching of cars in the test images.

The y-range of searching was narrowed from 400-700 pixels. This skips searching the sky and trees. The additional optimization was from the course notes that collects the hog features once on the image and then selects different regions from the hog features per window without recomputing the hog features every time.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I search on 3 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a good overall result

Here is a test image showing the raw bounding boxes when running my pipline
There were some false positives in the images, but they were eliminated by using a heatmap approach over multiple frames

![alt text][sliding_window]
![alt text][sliding_window2]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

In addition to using a heatmap threshold, I added effectively a low pass filter by averaging the heatmap over the last 5 frames. This was to help filter out outliers that intermittently appear in a frame, but don't consistently track through

### Here are four frames and their corresponding heatmaps:

![alt text][heatmap]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all four frames:
![alt text][labeledheatmap]

### Here the resulting bounding boxes are drawn onto the 3rd frame in the series:
![alt text][final_frame]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

One of the initial problems I ran into was that training had high accuracy on the test data set, but detection was failing on the input images and window search. Turns out that in copy pasting code from the quizes, I had two methods that were extracting hog features and they were doing things slightly differently. This required cleaninup the code and ensuring that there was a single method to extract features and scale them.

There were also issues in dealing with png images, so I decided to use cv.imread for all image input and would normalize the images to RGB for all further processing so that I could re-use the same code for testing on single images vs video frames

Using a low pass filter to average the result from the past few frames does introduce a bit of a lag. If a vehicle were to appear suddenly in front of the car, there may be a few frames of wait before it's recognized

There could also be issues if a big vehicle suddenly entered the frame wright in front of the car. It may be detected by the sliding window approach since we're not searching a window that's a significant size of the frame.

The pipeline could be sped up by reducing the area that we search for smaller scale sizes that's further down the lane.

The filtering method could be tuned to have a lag of fewer frames so that we can detect sudden vehicles apparing in the frame quicker

There is also an issue of false positives still showing up near the bridge. I didn't have the time to fix those (since even filtering didn't remove them. they were consistent vehicle detections), but I would capture frames from the video that are problematic and tune the classifier so that it worked well with those frames. Maybe even train with the larger udacity data set to get more training samples
