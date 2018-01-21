import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.ndimage import gaussian_filter1d


def gaussian_blur(image, sigma):

    img = gaussian_filter1d(image, sigma, axis = 0, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)
    img = gaussian_filter1d(img, sigma, axis = 1, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

    return img;

def pixelate(image, n):
    #input images have size 112x112

    #n should be a divisor of 112
    
    '''
    maybe needed if n is no divisor of the image size
    square_w = n
    square_h = n
    num_cols = int(round(img.shape[1]) / square_h)
    num_rows = int(round(img.shape[0] / square_w))
    square_h = float(img.shape[0]) / num_rows
    '''
    
    result = np.copy(image)
    
    H , W, C = image.shape
    num_squares = H//n
    
    #extract the squares and their respective colour values
    for i in range(num_squares):
        for j in range(num_squares):
            
            r = image[i*n:(i+1)*n, j*n:(j+1)*n, 0 ]
            g = image[i*n:(i+1)*n, j*n:(j+1)*n, 1 ]
            b = image[i*n:(i+1)*n, j*n:(j+1)*n, 2 ]
            
            av_r = 0
            av_g = 0
            av_b = 0
            
            av_r= np.mean(r)
            av_g= np.mean(g)
            av_b= np.mean(b)
            
            result[i*n:(i+1)*n, j*n:(j+1)*n, 0] = av_r
            result[i*n:(i+1)*n, j*n:(j+1)*n, 1] = av_g
            result[i*n:(i+1)*n, j*n:(j+1)*n, 2] = av_b
            
    
   
    return result;

#construct a test case to test the previously defined functions

#img1 = plt.imread('Al_Gore_0004.jpg')
#
#
#img1 = gaussian_blur(img1,4)
#
#plt.figure(2)
#plt.imshow(img1)
#plt.title('applying gaussian blur')
#plt.show()

