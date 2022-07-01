# Computer-Vision-and-Image-Processing

<h3>Task1: Panorama Creation</h3>

* Found keypoints (i.e. points of interest) in the given images, using SIFT point detector.
* Used SIFT to extract features for these keypoints.
* Matched the keypoints between two images by comparing their feature distance using KNN, k=2. I have used “ratio testing” to filter good matches (n0 = 0.75).
* Computed the homography matrix using RANSAC algorithm.
* Used the homography matrix to stitch the two given images into a single panorama.

<h3>Task2: Edge Detection</h3>

* Performed Image denoising and Edge detection
* Implemented a 3x3 median filter to denoise image with salt-and-pepper noise
* Manually created different filters to detect horizontal, vertical, 45° direction and 135° direction (i.e. diagonal) edges 

<h3>Task3: Morphology Image Processing</h3>

* Implemented four morphology operations
  * Erode
  * Dilate
  * Open
  * Close
* Implemented boundary extraction using erode operation

