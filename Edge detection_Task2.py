"""
 Grayscale Image Processing
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to experiment with two commonly used 
image processing techniques: image denoising and edge detection. 
Specifically, you are given a grayscale image with salt-and-pepper noise, 
which is named 'task2.png' for your code testing. 
Note that different image might be used when grading your code. 

You are required to write programs to: 
(i) denoise the image using 3x3 median filter;
(ii) detect edges in the denoised image along both x and y directions using Sobel operators (provided in line 30-32).
(iii) design two 3x3 kernels and detect edges in the denoised image along both 45° and 135° diagonal directions.
Hint: 
• Zero-padding is needed before filtering or convolution. 
• Normalization is needed before saving edge images. You can normalize image using the following equation:
    normalized_img = 255 * frac{img - min(img)}{max(img) - min(img)}

Do NOT modify the code provided to you.
You are NOT allowed to use OpenCV library except the functions we already been imported from cv2. 
You are allowed to use Numpy for basic matrix calculations EXCEPT any function/operation related to convolution or correlation. 
You should NOT use any other libraries, which provide APIs for convolution/correlation ormedian filtering. 
Please write the convolution code ON YOUR OWN. 
"""

from cv2 import imread, imwrite, imshow, IMREAD_GRAYSCALE, namedWindow, waitKey, destroyAllWindows
import numpy as np

# Sobel operators are given here, do NOT modify them.
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(int)
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(int)


def filter(img):
    """
    :param img: numpy.ndarray(int), image
    :return denoise_img: numpy.ndarray(int), image, same size as the input image

    Apply 3x3 Median Filter and reduce salt-and-pepper noises in the input noise image
    """

    # TO DO: implement your solution here

    #Zero padding
    newimg = np.zeros((img.shape[0]+2, img.shape[1]+2))
    newimg[1:img.shape[0]+1, 1:img.shape[1]+1] = img

    # median filter
    a = newimg
    for i in range(1,img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            b = [a[i-1][j-1], a[i-1][j], a[i-1][j+1], a[i][j-1], a[i][j], a[i][j+1], a[i+1][j-1], a[i+1][j], a[i+1][j+1]]
            img[i-1][j-1] = np.median(b) #center element is replaced with median of itself and its neighbourhood

    denoise_img = img
    #raise NotImplementedError
    return denoise_img


def convolve2d(img, kernel):
    """
    :param img: numpy.ndarray, image
    :param kernel: numpy.ndarray, kernel
    :return conv_img: numpy.ndarray, image, same size as the input image

    Convolves a given image (or matrix) and a given kernel.
    """

    # TO DO: implement your solution here

    flippedkernel = np.flip(kernel)
    z = np.zeros((img.shape[0], img.shape[1]))
    newimg = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    newimg[1:img.shape[0] + 1, 1:img.shape[1] + 1] = img
    a = newimg
    for i in range(1,img.shape[0]+1):
        for j in range(1, img.shape[1]+1):
            b = [a[i-1][j-1], a[i-1][j], a[i-1][j+1], a[i][j-1], a[i][j], a[i][j+1], a[i+1][j-1], a[i+1][j], a[i+1][j+1]]
            b = np.reshape(b, (3,3))
            l = flippedkernel * b  #flippedkernel is multuplied with itself and pixel neighbourhood
            z[i-1][j-1] = np.sum(l)

    conv_img = z
    #raise NotImplementedError
    return conv_img


def edge_detect(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_x: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_y: numpy.ndarray(int), image, same size as the input image, edges along y direction
    :return edge_mag: numpy.ndarray(int), image, same size as the input image,
                      magnitude of edges by combining edges along two orthogonal directions.

    Detect edges using Sobel kernel along x and y directions.
    Please use the Sobel operators provided in line 30-32.
    Calculate magnitude of edges by combining edges along two orthogonal directions.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here

    #edge_x and edge_y

    img_x = convolve2d(img, sobel_x)
    img_y = convolve2d(img, sobel_y)
    img_mag = np.sqrt((img_x ** 2) + (img_y ** 2))

    #Normalized edges
    edge_x = 255*((img_x - np.min(img_x))/(np.max(img_x) - np.min(img_x)))
    edge_y = 255*((img_y - np.min(img_y))/(np.max(img_y) - np.min(img_y)))
    edge_mag = 255*((img_mag - np.min(img_mag))/(np.max(img_mag) - np.min(img_mag)))

    edge_x = edge_x.astype(np.uint8)
    edge_y = edge_y.astype(np.uint8)
    edge_mag = edge_mag.astype(np.uint8)

    #raise NotImplementedError
    return edge_x, edge_y, edge_mag


def edge_diag(img):
    """
    :param img: numpy.ndarray(int), image
    :return edge_45: numpy.ndarray(int), image, same size as the input image, edges along x direction
    :return edge_135: numpy.ndarray(int), image, same size as the input image, edges along y direction

    Design two 3x3 kernels to detect the diagonal edges of input image. Please print out the kernels you designed.
    Detect diagonal edges along 45° and 135° diagonal directions using the kernels you designed.
    All returned images should be normalized to [0, 255].
    """

    # TO DO: implement your solution here
    #Kernels for 45 and 135
    kernel_45 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]]).astype(int)
    kernel_135 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]).astype(int)

    img_45 = convolve2d(img, kernel_45)
    img_135 = convolve2d(img, kernel_135)

    # Normalized edges
    edge_45 = 255 * ((img_45 - np.min(img_45)) / (np.max(img_45) - np.min(img_45)))
    edge_135 = 255 * ((img_135 - np.min(img_135)) / (np.max(img_135) - np.min(img_135)))

    edge_45 = edge_45.astype(np.uint8)
    edge_135 = edge_135.astype(np.uint8)

    #Designed Kernels
    print("Kernel 45: ", kernel_45)
    print("Kernel 135: ", kernel_135)



    #raise NotImplementedError
    return edge_45, edge_135


if __name__ == "__main__":
    noise_img = imread('task2.png', IMREAD_GRAYSCALE)
    denoise_img = filter(noise_img)
    imwrite('results/task2_denoise.jpg', denoise_img)
    edge_x_img, edge_y_img, edge_mag_img = edge_detect(denoise_img)
    imwrite('results/task2_edge_x.jpg', edge_x_img)
    imwrite('results/task2_edge_y.jpg', edge_y_img)
    imwrite('results/task2_edge_mag.jpg', edge_mag_img)
    edge_45_img, edge_135_img = edge_diag(denoise_img)
    imwrite('results/task2_edge_diag1.jpg', edge_45_img)
    imwrite('results/task2_edge_diag2.jpg', edge_135_img)





