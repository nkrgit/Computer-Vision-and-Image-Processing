"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""

import cv2
import numpy as np
from numpy import linalg as la
import math
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random
#used pip install opencv-python==3.4.2.17 opencv-contrib-python==3.4.2.17


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
    img1 = right_img
    img2 = left_img


    # #SIFT descriptors to extract features

    sift = cv2.xfeatures2d.SIFT_create();

    # Keypoints extraction
    kpr, desr = sift.detectAndCompute(img1, None)
    kpl, desl = sift.detectAndCompute(img2, None);


    rnorm = np.zeros((desr.shape[0], 1)) #rnorm to store norm value
    oright = np.zeros((desr.shape[0], 1)) #oright to store sorted points indices
    twobestr = np.zeros((desr.shape[0],4)) #array to store two min distances and their indices



    # KNN

    for i in range(0, desr.shape[0]):
        rnorm = la.norm(desr[i]-desl, axis=1)
        sortedrnorm = np.sort(rnorm)
        oright = np.argsort(rnorm)
        twobestr[i][0] = sortedrnorm[0]
        twobestr[i][1] = sortedrnorm[1]
        twobestr[i][2] = oright[0]
        twobestr[i][3] = oright[1]



    #Match keypoint using test ratio

    n0 = 0.75
    matchedright = []
    matchedleft = []

    for i in range(0, desr.shape[0]):
        if twobestr[i][0] < n0 * twobestr[i][1]: #test ratio condition to filter out best keypoints
            matchedright.append(i)
            matchedleft.append(int(twobestr[i][2])) #appending left best index

    print("Filtered keypoints after ratio testing: ", len(matchedright))
    #Extracting x,y coordinates from the filtered keypoints
    src_pts = np.float32([kpr[m].pt for m in matchedright])
    dst_pts = np.float32([kpl[m].pt for m in matchedleft])

    # Homography matrix using RANSAC

    src_ptsnew = np.zeros((len(src_pts), 3))
    dst_ptsnew = np.zeros((len(dst_pts), 3))
    for i in range(len(src_pts)):
        src_ptsnew[i] = np.append(src_pts[i], 1)
        dst_ptsnew[i] = np.append(dst_pts[i], 1)


    # RANSAC loop starts from here

    n = 4
    t = 5
    k = 5000
    bestcount = 0

    for j in range(k):

        random_indicesr = np.random.choice(src_ptsnew.shape[0], size=n, replace=False) #Picking 4 random points
        leftv = []
        rightv = []
        for i in random_indicesr:
             rightv.append(src_ptsnew[i])
             leftv.append(dst_ptsnew[i])


        #Intermediate H Matrix uing SVD
        k = [0, 0, 0]
        u = 0
        A = np.zeros((2*n, 9))
        for i in range(0, 2*n, 2):
            j = i+1
            k1 = rightv[u]
            k3 = -leftv[u][0]*rightv[u]
            k4 = -leftv[u][1]*rightv[u]

            kl = np.append(k1,k)
            A[i] = np.append(kl, k3)

            kr = np.append(k, k1)
            A[j] = np.append(kr, k4)
            u = u + 1


        [u, e, vt] = np.linalg.svd(A) #SVD Operation
        H = vt[8, :] #Homography matrix is the last row of vt
        H = np.reshape(H, (3, 3))
        primeh = []
        lfp = []
        inliers = []

        for i in range(0, len(dst_ptsnew)):

            if i not in random_indicesr:
                c = np.dot(H, src_ptsnew[i])
                dw = [j / c[2] for j in c]
                primeh.append(dw)
                lfp.append(dst_ptsnew[i])
                if la.norm(lfp[-1] - primeh[-1]) < t:#condition for testing if norm is less than residual distance
                    inliers.append(i)


        if len(inliers) > bestcount:
            bestcount = len(inliers) #Storing the maxmimum count of inliers
            bestinliers = inliers

    print("Inlier Count: ", bestcount)


    #H Matrix uing SVD

    leftv = []
    rightv = []
    for i in bestinliers:
        rightv.append(src_ptsnew[i])
        leftv.append(dst_ptsnew[i])

    k = [0, 0, 0]
    u = 0
    A = np.zeros((2 * bestcount, 9))
    for i in range(0, 2 * bestcount, 2):
        j = i + 1
        k1 = rightv[u]
        k3 = -leftv[u][0] * rightv[u]
        k4 = -leftv[u][1] * rightv[u]

        kl = np.append(k1, k)
        A[i] = np.append(kl, k3)

        kr = np.append(k, k1)
        A[j] = np.append(kr, k4)
        u = u + 1

    [u, e, vt] = np.linalg.svd(A)
    H = vt[8, :]
    H = np.reshape(H, (3, 3)) #Final Homography matrix H
    H = H/H[2][2]

    print("H matrix: ", H)


    # Using H to stitch 2 images

    #Left and right image corners
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    rightpts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    leftpts = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    #Finding new corners after transformation
    newpts = cv2.perspectiveTransform(rightpts, H)
    pts = np.concatenate((leftpts, newpts), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    tranM = [-xmin, -ymin]

    #Translation Matrix
    H1 = np.array([[1, 0, tranM[0]], [0, 1, tranM[1]], [0, 0, 1]])

    #Warping right image
    result_img = cv2.warpPerspective(img1, H1.dot(H), (xmax - xmin, ymax - ymin))

    #Stitching left image to right image
    result_img[tranM[1]:h2 + tranM[1], tranM[0]:w2 + tranM[0]] = img2

    #raise NotImplementedError
    return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)
