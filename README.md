# Computer-Vision-and-Image-Processing

<h3>Task1: Panorama Creation</h3>

* Found keypoints (i.e. points of interest) in the given images, using SIFT point detector.
* Used SIFT to extract features for these keypoints.
* Matched the keypoints between two images by comparing their feature distance using KNN, k=2. I have used “ratio testing” to filter good matches (n0 = 0.75).
* Computed the homography matrix using RANSAC algorithm.
* Used the homography matrix to stitch the two given images into a single panorama.

<h4>Left and Right Images</h4>
<div align ='center'>
<p>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/left.jpg" width = '350' height = '300'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/right.jpg" width = '350' height = '300'> </p>
</div>

<h4>Panorama:</h4>
<div align ='center'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task1_result.jpg" width = '700' height = '400'>
</div>

<h3>Task2: Denoising and Edge Detection</h3>

* Performed Image denoising and Edge detection
* Implemented a 3x3 median filter to denoise image with salt-and-pepper noise
* Manually created different filters to detect horizontal, vertical, 45° direction and 135° direction (i.e. diagonal) edges 

<h4>Noisy image and denoised image:</h4>
<div align = 'center'>
<p>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/task2.png" width = '350' height = '280'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task2_denoise.jpg" width = '350' height = '280'>
</p>
</div>


<h4>Edge detection:</h4>
<div align = 'center'>
<p>

<h4>Edges along x and y</h4>

<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task2_edge_x.jpg" width = '350' height = '280'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task2_edge_y.jpg" width = '350' height = '280'>

<h4>Edges along 45° and 135°</h4>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task2_edge_diag1.jpg" width = '350' height = '280'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task2_edge_diag2.jpg" width = '350' height = '280'>
</p>
</div>


<h3>Task3: Morphology Image Processing</h4>

* Implemented four morphology operations
  * Erode
  * Dilate
  * Open
  * Close
* Implemented denoising using open and close operations
* Implemented boundary extraction using erode operation.

<h4>Noisy image, Denoised image and Boundary extracted image</h3>
<div align = 'center'>
<p>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/task3.png" width = '250' height = '250'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task3_denoise.jpg" width = '250' height = '250'>
<img src="https://github.com/nkrgit/Computer-Vision-and-Image-Processing/blob/main/results/task3_boundary.jpg" width = '250' height = '250'>
</p>
</div>
