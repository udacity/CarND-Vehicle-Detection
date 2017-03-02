**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/figure_1.png
[image5]: ./examples/figure_1-heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/figure_1-1.png
[video1]: ./output_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I extracted the HOG features in lines 20 through 37 of the python file `lesson_functions.py`.  

I started by reading all the `vehicle` and `non-vehicle` images, along with separate testing `vehicle` and `non-vehicle` images. I did this because if I read in all images from those two directories, the time-series data would end up in both the training and testing sets, which would result in a untrustworthy testing score.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found a lot of interesting things happening. I ran through a lot of different configurations (which can be seen below) and then realized that forming my training data selection is very important. My final choices were: 
        `
        self.orient = 9  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (32, 32)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.scales = [1, 1.25, 1.5, 2]
        `
Most of these choices are reflected in the youtube walkthrough, and were definitely show to reflect the best selection when running through all of the following configurations:
        
.7174 -> SVC linear, prob, spatial=(16,16) color_space=HSV 
.7175 -> SVC linear, prob, C=1.25, spatial=(16,16) bins=16 color_space=HSV
.6947 -> SVC linear, prob, C=1.25, spatial=(32,32) bins=32 color_space=HSV
.6949 -> SVC linear, prob, C=1.5, spatial=(32,32) bins=32 color_space=HSV
.6947 -> SVC linear, prob, C=1.0, spatial=(32,32) bins=32 color_space=HSV
.7141 -> LinearSVC, C=1.0 spatial=(32,32) bins=32 color_space=HSV
.6947 -> SVC linear, C=1.0 spatial=(32,32) bins=32 color_space=HSV
.7175 -> SVC linear, C=1.0  spatial=(16,16) bins=16 color_space=HSV
.7342 -> LinearSVC, C=1.0 spatial(16, 16) bins=16 color_space=HSV
.7342 -> LinearSVC, C=1.5, spatial(16, 16) bins=16 color_space=HSV
.4802 -> RandomForest (all defaults) color_space=HSV
.8371 -> SVC linear, prob, C=1.0, spatial=(32,32) bins=32 color_space=HSV *
.8531 -> SVC linear, prob, C=1.0, spatial=(16,16) bins=16 color_space=HSV *
.7171 -> SVC linear, prob, C=1.0, spatial=(16,16) bins=16 color_space=YCrCb *
.9855 -> SVC linear, prob, C=1.0, spatial=(16,16) bins=16 color_space=HSV **
.9858 -> LinearSVC, C=1.0, spatial=(16,16) bins=16 color_space=HSV **
.9821 -> LinearSVC, C=1.0, spatial=(32,32) bins=32 color_space=YCrCb **
.9883 -> LinearSVC, C=1.0, spatial=(32,32) bins=32 color_space=HSV **

* changed my training data to use the Extra folder for non-vehicles, instead of the GTI folder. I added a small amount of GTI data to even out my vehicle/non-vehicle ratio
** changed my training data set again to use all the time-series data as training data and none in the testing data

I ultimately selected the last configuration due to it being better than all the other classifiers in testing, but also due to the increased speed over the linear classifier with probability. I also chose several scales because I found that it can reinforce the heatmap and the locations of the good data.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, bin, and histogram features on lines 45 to 139 of `vehicleFinder.py`. My thought behind this was the more information the better.

I found that which training data set I selected was very important to training this data. I wanted to be careful about not have untrustworthy testing data, so I kept all of the time-series data to itself in the testing set, meaning the classifier has never seen it. Unfortunately, this mean I still had a large testing data set. Then I realized I could also do the exact opposite, use all the time-series data to train, and then only test on the KITTI data, this way I could balance my testing data to 80% training and 20% testing. Incorporating the time-series data into training provided to be the most useful of anything in training my classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed my sliding window search on lines 46 to 85 of the `hog_subsample.py` python file. I actually created another method to perform multiple scales (lines 91 through 112 of `hog_subsample.py`). I thought this was a good idea because cars would change in scale as they move closer or further.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Using a linear SVC with probability helped me a lot while my training data set wasn't working very well. I think that knowing whether the classifier is confident about it's guess is very important. Having it guess the trees are a car with only 20% confidence is a complete waste, and should definitely be ignored. But if those trees are 51% should it potentially round up? I think not, and I think in most cases, it's probably best to check that probability. It's a bit more costly in terms of compute time, but if you really wanted make it a 0 or 1 answer, you could do the same with probability by setting your threshold very high like 99%.
 
 That being said, incorporating the time-series data was extremely important. Clearing having lots of similar images that are slightly distorted/changed from each other cause the classifier to because much more robust as predicting cars. All of my other tricks, like using multiple-scales, and probability definitely helped, but training the classifier has the greatest impact.
 
 The sliding window and pipleine in action can be seen in this photo:

![alt text][image3]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I started by returning all of the potential boxes from my `find_cars_multi_scale()` method on line 235, then I added heat to a blank image, and thresholded that image with `applyThreshold()` on line 245. After that I am combining it with the heat values from the previous frame. Then on line 260 I apply the `scipy.ndimage.measurements.label()` function so any overlapping boxes could become one label (blob), and then on line 261 I drew said boxes.

### Here is the heat map corresponding to the above image:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had all sorts of problems getting my classifier trained well. I changed a lot of parameters that didn't alter my classifier's testing score much, if at all. This helped me recognize how important training data is to get a classifer to generalize correctly. I had a few bugs that caused me a decent amount of heartache. The first was that I hadn't flattened my multi-scale array, and hence wasn't getting all of my bboxes put into my heatmap. That didn't take much time to fix, but it took a little while for me to recognize that several of them were missing. The second was the `mpimg.imread()` vs. `cv2.imread()` issue. Since they both change the values for jpg files vs. png files, it was slightly messy/confusing. I'm thinking next time I do this I should go back and stick to just a single way of importing images. It would be nice if there was some sort of standard for this, but oh well.

I think using several scales has helped make the pipeline a bit more robust. Most cars are going to be changing in distance so having different scales running at the same time definitely seem like a good idea. If we had a TON of compute power I think it would be great to see scales in increments of 1/10th being run through the pipeline.

To make my pipeline more robust, I would be interested in looking at the speed of another object. A tree on the side of the road that is stationary could potentially be moving out of frame way faster than a car in your lane. I can guess that most vehicle are only going +/- 10mph than our car. I think that would be a good clue that an object is probably a car and not something else.

I did a crude smashing together of heatmaps, and I think a better way of averaging the labels over frames would have definitely helped smooth things out and make my video more robust as well. 

Ultimately, I think using some sort of radar to figure out the objects that are closest to you in combination with this classifier would be the best solution. (close objects) + (object that looks like a car) = a good indication that it's a car and potentially hazardous/needs to be tracked.

