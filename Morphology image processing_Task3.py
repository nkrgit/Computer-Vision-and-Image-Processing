"""
Morphology Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with commonly used morphology
binary image processing techniques. Use the proper combination of the four commonly used morphology operations, 
i.e. erosion, dilation, open and close, to remove noises and extract boundary of a binary image. 
Specifically, you are given a binary image with noises for your testing, which is named 'task3.png'.  
Note that different binary image might be used when grading your code. 

You are required to write programs to: 
(i) implement four commonly used morphology operations: erosion, dilation, open and close. 
    The stucturing element (SE) should be a 3x3 square of all 1's for all the operations.
(ii) remove noises in task3.png using proper combination of the above morphology operations. 
(iii) extract the boundaries of the objects in denoised binary image 
      using proper combination of the above morphology operations. 
Hint: 
• Zero-padding is needed before morphology operations. 

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy libraries, HOWEVER, 
you are NOT allowed to use any functions or APIs directly related to morphology operations.
Please implement erosion, dilation, open and close operations ON YOUR OWN.
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np


def morph_erode(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return erode_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology erosion on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    # TO DO: implement your solution here

    #Zero padding
    z = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    z1 = np.zeros((img.shape[0], img.shape[1]))

    b = [[1, 1, 1], [1, 1, 1], [1, 1, 1]] #structuring element
    z[1:img.shape[0]+1, 1:img.shape[1]+1] = img
    a = z
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            c = [a[i - 1][j - 1], a[i - 1][j], a[i - 1][j + 1], a[i][j - 1], a[i][j], a[i][j + 1], a[i + 1][j - 1],
                 a[i + 1][j], a[i + 1][j + 1]]
            c = np.reshape(c, (3, 3))
            l = np.prod(b * c)
            if l != 0: #using product of neighbours to "b", if it is not zero then 255 is placed at center: if atleast one neighbour element is zero the center element is zero
                z1[i - 1][j - 1] = 255
            else:
                z1[i - 1][j - 1] = 0

    erode_img = z1.astype(np.uint8)
    return erode_img


def morph_dilate(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return dilate_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology dilation on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    """
    # TO DO: implement your solution here

    # Zero padding
    z = np.zeros((img.shape[0]+2, img.shape[1]+2))
    z1 = np.zeros((img.shape[0], img.shape[1]))

    b = [[1,1,1], [1,1,1], [1,1,1]] #structuring element
    z[1:img.shape[0]+1, 1:img.shape[1]+1] = img
    a = z
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            c = [a[i - 1][j - 1], a[i - 1][j], a[i - 1][j + 1], a[i][j - 1], a[i][j], a[i][j + 1], a[i + 1][j - 1],
                 a[i + 1][j], a[i + 1][j + 1]]
            c = np.reshape(c, (3, 3))
            l = np.sum(b*c)
            if l != 0: #using sum of neighbours to "b", if it is not zero then 255 is placed at center: if atleast one neighbourhood element is non-zero, the center element is 255
                z1[i-1][j-1] = 255
            else:
                z1[i - 1][j - 1] = 0

    dilate_img = z1.astype(np.uint8)

    return dilate_img


def morph_open(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return open_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology opening on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
    # TO DO: implement your solution here

    # open: erode and then dilate
    erode_img = morph_erode(img)
    open_img = morph_dilate(erode_img)
    open_img = open_img.astype(np.uint8)
    return open_img

def morph_close(img):
    """
    :param img: numpy.ndarray(int or bool), image
    :return close_img: numpy.ndarray(int or bool), image, same size as the input image

    Apply mophology closing on input binary image. 
    Use 3x3 squared structuring element of all 1's. 
    You can use the combination of above morph_erode/dilate functions for this. 
    """
   # TO DO: implement your solution here

   # close: dilate and then erode
    dilate_img = morph_dilate(img)
    close_img = morph_erode(dilate_img)
    close_img = close_img.astype(np.uint8)
    return close_img


def denoise(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Remove noises from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    # TO DO: implement your solution here

    #for denoising open and then close
    open_img = morph_open(img)
    denoise_img = morph_close(open_img)
    denoise_img = denoise_img.astype(np.uint8)

    return denoise_img


def boundary(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Extract boundaries from binary image using morphology operations. 
    If you convert the dtype of input binary image from int to bool,
    make sure to convert the dtype of returned image back to int.
    """
    # TO DO: implement your solution here

    #boundary: erode and substract eroded image from denoised image
    erode_img = morph_erode(img)
    bound_img = img - erode_img

    bound_img = bound_img.astype(np.uint8)

    return bound_img


if __name__ == "__main__":
    img = imread('task3.png', IMREAD_GRAYSCALE)
    denoise_img = denoise(img)
    imwrite('results/task3_denoise.jpg', denoise_img)
    bound_img = boundary(denoise_img)
    imwrite('results/task3_boundary.jpg', bound_img)





