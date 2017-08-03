# Vehicle Detection Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Evaluate Neural Network approach to object detection

## Final Result Video

[![IMAGE ALT TEXT](https://img.youtube.com/vi/FdZvMoP0dRU/0.jpg)](https://www.youtube.com/watch?v=FdZvMoP0dRU "Veh Detection Video.")


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image3b]: ./examples/sample_region_matches.png
[image3c]: ./examples/hog_subsample.png
[image4b]: ./examples/results.png
[image4]: ./examples/box_matches.png
[image5]: ./examples/heatmap.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/dog.jpg
[image9]: ./examples/yolo_network.png
[image10]: ./examples/persp_transform.png
[video1]: ./project_video_out.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Tracking objects using HOG features from the training images.

The code for this step is contained in the IPython notebook (here)[https://github.com/tawnkramer/CarND-Vehicle-Detection/blob/master/VehicleDetection.ipynb].

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Final choice of HOG parameters.

I tried various combinations of parameters and arrived at using the YCrCb colorspace using 8 pixels per cell with 18 orientation directions for the gradients. I used a 16x16 spatial binning size and 2 cells per block.

#### 3. Training a Classifier using HOG, Color, and Spatial Features

I trained a linear SVM using a combined feature vector of HOG features, spatial features, and a histogram of color features across all three channels of YCrCb. This can be seen in cell 2 of [my python notebook](https://github.com/tawnkramer/CarND-Vehicle-Detection/blob/master/VehicleDetection.ipynb).
The spacial features resized the image to 16x16 pixels and used the resulting color values for each pixel. All three feature vectors were combined and then normalized for each training image.

Training images were categorized as containing or a car or not. And then a Linear SVM was trained with 80% of samples. The resulting 20% were used to validate the results. The accuracy agains the validation set was 100%.

### Sliding Window Search

#### 1. Choosing a region of image to search

I first use a sliding window approach, where the features for each region are calculated and then evaluated against the trained model. This technique creates a window of subset of the image, then moves it by some standard offset, often overlapping the previous window by some amount. There's a tradeoff between accuracy and time, as many windows will be expensive to evaluate.

![alt text][image3]

I moved to a faster approach that extracted features once from a subregion of the image below the horizon. Then it subsampled the region by overlaying windows. Each window was scaled to different factors, so that multuple box sizes can be tested efficiently.

![alt text][image3c]

#### 2. Initial Results

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here is an example image:

![alt text][image4]

And occasionally some false positives occured, as this shadowed area of the guard rail.

![alt text][image3b]

#### 3. Filtering boxes

From the list of candiate boxes, I created a heat map. I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Each bounding box had a single vote, and combined with other boxes to increase the likleyhood of a car detection. Then a thresholding operation was performed to cull low condfidense boxes. This sometimes resulted in a non-detection of a car when only one box was found.

![alt text][image4b]

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are three frames and their corresponding heatmaps:

![alt text][image5]


### video result:

[link to my hog video result](./project_video_out.mp4)

## Neural Network Approach

Some research indicated that modern neural networks have some increased capacity for locating objects of many different classes at once in different subregions of an image, even when overlapping or partially obscurred. I chose to research [YOLO](https://arxiv.org/abs/1506.02640) and investigate how it worked. YOLO is short for You Only Look Once, and is an approach that uses a single pass through a deep fully convolutional network to generate bounding box candidates, and confidense scores. A post processing step takes the final output tensor, which may be of dimensions like 7x7x30, and analyzes it for proposals.

The 7x7 represents the number of regions in height and width evenly divided into the image. Each cell block contains the class probablity that, if a bounding box is found, it will contain an object of a certain class. The 30 values of the tensor for each block contain two bounding box proposals, each with a confidense value and dimensions - 5 values each. The remaining 20 values are one hot encoded class scores indicating confidense for each class. Typically a softmax will turn this one hot encoding in a probablity and the most likely is chosen. 

Then a thresholding operation occurs to cull bounding boxes with lower confidense and the remaing presented as results.
![alt text][image8]

#### 1. Initial setup

I used the code from allanzelener on github [here](https://github.com/allanzelener/YAD2K) as a starting point. I download pre-trained network weights and converted them to Keras/Tensorflow format using the provided scripts. 

```bash
wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
```

This network uses 24 convolutional layers, with batch normalization and leaky-relu activation. 
![alt text][image9]

#### 2. Initial scan

I then created a python script to run this scan over multiple frames of a video and output a final video. This script is [here](https://github.com/tawnkramer/CarND-Vehicle-Detection/blob/master/process_vid_yolo.py).

This created much more consistent results, outlining most all cars and very few failures. But the regions were not very stable from frame to frame.

#### 3. Stablization

For continuity, I created a running list of bounding boxes over multiple frames. For each I tracked the average color and dimension of the box. When I get a new candidate box on each frame, I would attempt to match it with a previous box by position and dominant image color. Then I would interpolate towards the new box with some slower rate. I also determine a velocity in X and Y that updates the center of the box each frame. The combination smooths the position and dimensions of the car bounding boxes.

#### 4. Metrics

The box center is reverse projected onto into a more linear space using the same method used in advanced lane finding. 
![alt text][image10]

In this space it was simple to assign a lane position by reverse projecting the center of the bounding box with cv2.perspectiveTransform. As the operation required unusual levels of encapsulation in lists and then dereferencing, it's included here:

```code

def tm(pt_xy, M):
    '''
    perform perspective transform on a single point, given x, y pixel
    and persp matrix M
    return the x, y pixel pair in transformed space
    '''
    pt = np.array([pt_xy])
    pt = np.array([pt])
    res = cv2.perspectiveTransform(pt, M)
    return res[0][0]
    
```

An simple relative speed estimate is done using the bounding box velocity relative to the current vehicle speed.

---

### Video Result

[![IMAGE ALT TEXT](https://img.youtube.com/vi/FdZvMoP0dRU/0.jpg)](https://www.youtube.com/watch?v=FdZvMoP0dRU "Veh Detection Video.")

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I enjoyed working with more traditional image feature based methods and classifiers. They were comprehensible. And when they failed, their failings were consistent with how they operated. However, the number of tunable hyper parameters, and the tendancy to create outliers of both false positives and weak positives, created a real challenge to constructing a robust solution. 

I found the YOLO neural network approach to be immediately powerful. The pre-trained network allowed me to focus on stablizing post-processing and image metrics. The result was robust and overall ran at a faster frame rate.

My lane detection metrics assume a straight road and do not account for curvature. In the later parts of the video you can see where a car along the curve crosses the lane threshold without changing lanes.

The car velocity estimate is a weak approximation and included mainly for fun.

The lane assignment would fail when the main car changes lanes, and needs work to determine our current lane. It also assumes all cars to the left are in an oncoming lane and would need work to assign more accurately.

I spent some time trying to track cars through overlaps, but that fails at the moment. I tried using the cars last momentum and detect when it was obscurred and continue moving the box until it was discoverred again. This didn't work as well as I hoped and is disabled in the final implementation.

The color approximation is a weak indicator identity in the bounding box, and could be replaced by some online SVM continually fitting against more traditional features like hog, spacial, or histogram of colors. That might allow it to maintain more continuity through obscurations.







