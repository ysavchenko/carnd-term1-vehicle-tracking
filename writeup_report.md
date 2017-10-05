##Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[training_examples]: ./images/training_examples.png
[hog_example]: ./images/hog_exampe.png
[feature_normalization]: ./images/feature_normalization.png
[sliding_window_scales]: ./images/sliding_window_scales.jpg
[pipeline_test]: ./images/pipeline_test.png
[heatmap]: ./images/heatmap.png
[video1]: ./project_video.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

To extract HOG features we rely on `hog()` function from `skimage.feature`. There is a short utility function called `get_hog_features()` in the second cell of IPython notebook, which is basically a wrapper around call to `hog()` with limiting some of the possible parameters (for example, we only use square blocks and cells and do power law correction).

Then I've tried extracting HOG features from training images (code in IPython notebook expects them to be located in `/data/vehicle` and `/data/non-vehicle` folders). Here are examples of an image from both data sets:

![training_examples]

Total number of images in the data set is 8792 for `Vehicle` class and 8968 for `Non-Vehicle`.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![hog_example]

####2. Explain how you settled on your final choice of HOG parameters.

##### Color space

Before tuning HOG parameters I've decided to test which color space is the best for HOG classification and if it is better to use just one or all components for the final feature vector. For testing I took all images from the training set (first tried first 1000 images, but results were too similar with different color spaces), trained a simple Linear SVM classifier on them and compared test results. Train and test split was the same for all color space combinations, 20% of all images were reserved for testing.

You can see the results below:

| Color Space | Component #0 | Component #1 | Component #3 | All Components |
|:-----------:|-------------:|-------------:|-------------:|---------------:|
| RGB | 0.9048 | 0.9181 | 0.9167 | 0.9245 |
| HSV | 0.8986 | 0.9029 | 0.9198 | 0.9589 |
| HLS | 0.8998 | 0.9108 | 0.8899 | 0.9521 |
| YUV | 0.9105 | 0.9009 | 0.9220 | **0.9645** |
| YCrCb | 0.9102 | 0.9231 | 0.9017 | **0.9634** |

*LUV was not used because it has negative values and `transform_sqrt` enabled in `hog()` function does not allow negative values.*

As you can see, YUV and YCrCb showed the best results. Difference in test scores considering the size of test set is mis-classification of just 4 images. Since the code in learning materials used YCrCb this would also be the color space we'll be using from now on.

##### Orientations

Now let's see how different number of orientations affect testing accuracy (the same test configuration is used there). Learning materials mentioned that usually used numbers are between 6 and 12, so these are the numbers tried.

| Orientations | Number of features | Test Accuracy |
|:------------:|-------------------:|--------------:|
| 6 | 3528 | **0.9662** |
| 7 | 4116 | 0.9651 |
| 8 | 4704 | **0.9665** |
| 9 | 5292 | 0.9637 |
| 10 | 5880 | 0.9642 |
| 11 | 6468 | 0.9620 |
| 12 | 7056 | 0.9609 |

As you can see, the difference is not very significant, best and worst number of orientations differ just by around 10 images in the test set. Since we also try to make training/testing faster we should pick as low number of orientations as possible. So, according to our test result I picked 6 orientations.

##### Pixels per cell and cells per block

And finally I've tried several combinations of pixels per cell and cells per block.

| Pixels per cell | 2x2 blocks | 3x3 blocks | 4x4 blocks |
|:---------------:|-----------:|-----------:|-----------:|
| 4x4 | `16200` 0.9490 | `31752` 0.9507 | *Too many features* |
| 8x8 | `3528` 0.9659 | `5832` 0.9707 | `7200` 0.9713 |
| 16x16 | `648` 0.9758 | `648` **0.9786** | `288` 0.9747 |

In each cell you can see the size of feature vector and test accuracy (4x4 pixels and 4x4 blocks had too many features and was not evaluated). 

Results of this test surprised me a little because the best results were showed by the configurations with much less features. I think it happens because less features lead to better generalization in this case.

So at the end I've picked 16x16 cells and 3x3 blocks.

##### (Bonus) Spatial and histogram evaluation

Since I've also decided to use spatial binning and color histograms as a part of the final feature vector it might be useful to find the optimal parameters for them as well. We will not try different color spaces though because HOG already uses YCrCb and we will use the same color space for all features.

Here are the test results for different image sizes for spatial binning:

| Size | Number of features | Test Accuracy |
|:------------:|-------------------:|--------------:|
| 12x12 | 432 | 0.9169 |
| 16x16 | 768 | **0.9234** |
| 24x24 | 1728 | 0.8992 |
| 32x32 | 3072 | 0.8986 |
| 48x48 | 6912 | 0.8998 |

And here are the color histogram results:

| Size | Number of features | Test Accuracy |
|:------------:|-------------------:|--------------:|
| 12 | 36 | 0.8699 |
| 16 | 48 | 0.8874 |
| 24 | 72 | 0.9108 |
| 32 | 96 | 0.9231 |
| 48 | 144 | 0.9276 |
| 64 | 192 | 0.9296 |
| 96 | 288 | **0.9437** |
| 128 | 384 | 0.9398 |
| 192 | 576 | 0.9423 |
| 256 | 768 | 0.9403 |

Based on these test results the size for spatial binning will be 16x16 and number of bins for the histogram will be 96. 

Based on these values (plus HOG feature extraction settings) the final size of feature vector for each image will be 1704.

One important note: I can understand that evaluating each parameter separately from the others might not find an optimum combination, but I think it is a good place to start.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After identifying settings for feature extraction earlier now is the time to configure a classifier.

But before we do this let's make sure our feature vectors are normalized. Here is an image illustrating this:

![feature_normalization]

As for the classifier I've decided to use SVM as suggested in the learning materials. Then the choice would be: either to use configurable kernel or stick with linear SVM. So I've used `GridSearchCV` to try different combinations of `C` and `gamma` parameters (just `C` for linear classifier) and get the best configuration for simple linear classifier and the one with RBF kernel. Then I've used these parameters to train two classifiers and compare their performance on the test set. Results of this test are summarized in the table below:

| SVM Kernel | Optimal parameters | Test score | Time to test |
|:----------:|:------------------:|-----------:|-------------:|
| Linear | C = 0.01 | 0.9910 | 0.01s |
| RBF | C = 100, gamma = 0.00001 | 0.9910 | 6.32s |

Results are overwhelmingly in favor of a simple linear classifier. It showed the same test results as the one with RBF kernel, but did it much faster.

You can see the code used to run these tests by searching for `GridSearchCV` in IPython Notebook.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

To do a window search I took the method from 'Hog Sub-sampling Window Search' lecture. It is very effective because HOG features are calculated only once for the whole image. You can find the method in submitted IPython Notebook, function name is `find_cars()`. I only did some minor changes: added `cells_per_step` and `color_space` as parameters and now method returns rectangles it found on the image instead of painting them directly on the image itself.

Before we can apply this function to images here are some things we should decide on several parameters:

- Which scales to use
- Define Y ranges for each scale
- How much the window should overlap

I've used empirical method to select Y ranges and scales (just looking at test images and choosing what made sense). I ended up with 3 scales: 1, 1.5 and 2 and 3 corresponding ranges: (400, 480), (400, 560) and (400, 640) illustrated in the image below:

![sliding_window_scales]

As for window overlapping I decided to use 75% overlap. Our cell size (16x16) for 64x64 image only gives us options for 25%, 50% and 75% overlap and I selected 75% to make sure we cover more possibilities for vehicle location than 50% overlap does. In terms of `cells_per_step` it translates into `1`.

Later when we apply our pipeline to video processing we can try different overlap parameters ans see how it performs.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some examples of how the classifier works: 3 scales (1, 1.5 and 2), linear classifier trained on HOG features plus spatial binning and color histogram.

![pipeline_test]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

By running recognition pipeline with parameters (for HOG, spatial and color histogram binning) I've got a pipeline that on actual video lost the cars 50% of the time (although there were almost none false pasitives). I've tweaked the model a little by trying different parameter variation and finally ended up by increasing spatial binning resolution to 48x48. 

Below you can see the video using these parameters without joining overlapping rectangles.

[![Project Video](http://img.youtube.com/vi/ljfXsTmoCMI/0.jpg)](http://www.youtube.com/watch?v=ljfXsTmoCMI)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the combined heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![heatmap]

All this logic of applying heatmap and joining rectangles in implemented in `join_rectangles()` function in the code.

Additionally to joining rectangles I've also added exponential smoothing to them (done in `smooth_rectangles()` function). What it does is looking for rectangles with similar coordinates in previous frames and calculating average between current and past rectangles.

You can see the final result (with joining rectangles and smoothing in the video below):

[![Joined Video](http://img.youtube.com/vi/btgYlCkczVk/0.jpg)](http://www.youtube.com/watch?v=btgYlCkczVk)

#### 3. (Bonus) Vehicles + lanes

I've also took image processing pipeline from the previous project and used it together with vehicle recognition pipeline from this project (see function `combined_process_image()` in the code).

Resulting video with both lane and vehicle tracking is below:

[![Combined Video](http://img.youtube.com/vi/v-KW6GHUaLk/0.jpg)](http://www.youtube.com/watch?v=v-KW6GHUaLk)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most important problem is not enough training data for our classifier. I think more different vehicles/non-vehicles would make classifier much better. We might also need more examples of different car types on the traning data.

Another issue is the classifier itself. Although linear SVM classifier is very fast I feel that it does not have enough complexity to split classes appropriately. More complex classifier might be needed there (I could be wrong though).

There also could be more intelligent false-positives detector algorithm: for example, new car should appear on the horizon and from the side, not accidentally on the middle of the road. This logic would filter out false-positive like the ones you can see in the project viewo above.

I know this project is intended to show usage of SVM for classification purposes, but I think neural networks would be a good choice for the classifier as well. Convolutional layers would extract features and later layers would act as a classifier.