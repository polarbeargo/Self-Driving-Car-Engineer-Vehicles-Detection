
##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carOrNcar.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/draw_boxes.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./output_images/labled.png
[image7]: ./output_images/project2.png
[image8]: ./output_images/project1.png
[image9]: ./output_images/ncar3dColorSpaceHLS.png
[image10]: ./output_images/ncar3dColorSpaceHSV.png
[image11]: ./output_images/car3dColorSpaceHSV.png
[image12]: ./output_images/car3dColorSpaceHLS.png
[image13]: ./output_images/carDetect.png
[image14]: ./output_images/heatMap.png
[image15]: ./output_images/apply_threshold.png
[video1]: ./project_video_orin.mp4
[video2]: ./project_video_out.mp4

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fourth code cell of the IPython notebook.  
I started by reading in all the `vehicle` and `non-vehicle` images, then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.
Here is an example using HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(8, 8)`:


![][image1]  

Visualized color space in 3D plot using `plot3d()`, the code for this step is contained in the fifth code cell of the IPython notebook. After explored different color spaces like `RGB, HSV, HLS, YCrCb`, found out the `YCrCb` color space in the car images, cluster the objects well and stands out against the background. Here are some of explore color space plot:  

|              vehicle                    |                     non-vehicle                        | 
| :-------------------------------------: |:------------------------------------------------------:|
|![][image11]                      |                ![][image10]                     |
|![][image12]                      |                ![][image9]                     |


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)`, `hog_channel = "ALL"` and `cells_per_block=(2, 2)` have the highest test accuracy = 0.9868 the HOG parameters tuning code is contained in the 6th code cell of the IPython notebook, the Results are as following:
  
```
98.27 Seconds to extract HOG features...
Using: 11 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6468
```  

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a balanced dataset provided by udacity, i.e., have as many positive as negative examples. The Data Summary code is contained in the third code cell of the IPython notebook. The Dataset contain 8792 cars_images and 8968 notcars_images with image shape (64, 64, 3).
Then create an array stack of feature vectors and define the labels vector, then split up data into randomized training and test sets the code is contained in the sixth code cell of the IPython notebook. Then use a `LinearSVC()` to train our classifier, the code is contained in the seventh code cell of the IPython notebook. The Results as following:  
    
```
9.25 Seconds to train SVC...
Test Accuracy of SVC =  0.9868
My SVC predicts:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  1.]
For these 10 labels:  [ 1.  1.  1.  0.  1.  0.  0.  1.  1.  1.]
0.0057 Seconds to predict 10 labels with SVC
``` 
###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Like I learned from Behavior cloning project, I restricting search area on the image with `ystart` and `ystop` in the `find_cars()` function to filter out the unwanted noise as start. The find_cars only extract hog features once and then can be sub-sampled to get all of its overlaying windows. Each window is defined by a scaling factor `scale` where a scale of 1 would result in a window that's 8 x 8 cells then the overlap of each window is in terms of the cell distance. This means that a cells_per_step = 2 would result in a search window overlap of 75%. Then run this same function multiple times for different scale values to generate multiple-scaled search windows. The search area, scale and HOG parameters are defined in `process_img()` and `video_pipeline()` in the 10thand 15th code cell of the the IPython notebook as follow:  

```
    colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

    search_spaces = [
    (400, 464, 1.0),
    (416, 480, 1.0),
    (400, 496, 1.5),
    (432, 528, 1.5),
    (400, 528, 2.0),
    (432, 560, 2.0),
    (400, 596, 3.5),
    (464, 660, 3.5)
    ]
```

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features in the feature vector, which provided a nice result.  Here are some example images:

![][image4]


### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=ElI4HmAOJfY) and another [link to my video result combined with Advanced Lane Lines project results](https://www.youtube.com/watch?v=mixOeI5b8ZY)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. The code is contained in the 9th code cell of the IPython notebook.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a heatmaps as you see that overlapping detections exist for each of the two vehicles:

![][image14]
  
A threshold is applied to the heatmap (in this test Image, with a value of 1), setting all pixels that don't exceed the threshold to zero. The result as following:  

![][image15]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap and the output of `draw_labeled_bboxes()` in test_images:
![alt text][image6]
![alt text][image13]  

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![][image7]  

### Combined with the result from Advanced Lane Lines Project:
![][image8]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Even False Positives methods was applied, sometimes cars are not detected and still lots of false positive are present. I think the classifier is not perfect in this implementation. Detection from previous frames reduce the effect of being misclassified, My pipeline will fail when facing different lighting conditions (as you see the video, those false positive occured in trees and road side shade) and vehicles HOG features aren't in the training dataset. Extra balanced datasets (like Udacity labelled dataset), a very high accuracy classifier and maximizing window overlap might improve the classifier accuracy but it would also reduce performance from real-time.

Lot of work can be done to increase the performances (grid search, augment dataset) and design the bounding boxes in order to better fit the corrisponding detected object. Speed also is a problem but with a more precise classifier the overlap between windows can be increased hence resulting in less windows to search and more speed. At the last, I would like to thanks Udacity for providing this high level challenge and valuable guidance in this project.
 

