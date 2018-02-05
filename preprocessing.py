import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import glob
import pickle
import scipy.misc as mi
from PIL import Image
from obfuscation_methods import (gaussian_blur, pixelate)
import os,csv

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
dir_path = os.path.dirname(os.path.realpath(__file__))
savepath = dir_path + '/datasets/complete/'
inputpath = dir_path + '/datasets/complete/originals'
#savepath = dir_path + '/datasets/test/'
#inputpath = dir_path + '/datasets/test/originals'

degrees = [2, 4, 6, 8, 10, 12]

for deg in degrees:
    out_path = savepath + 'blurred_' + str(deg)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path = savepath + 'pixelated_' + str(deg)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

# Run over all images in the specified foder
#csv_file = open(dir_path + "/Datasets/FaceReconstructionData.csv", 'r+')
#writer = csv.writer(csv_file)
for path, dirs, files in os.walk(inputpath):
    for fname in files:
        print(fname)
        img = np.array(Image.open(inputpath + '/' + fname))
        H, W, C = img.shape
        # Crop the image to 112x112
        #mrgn = int((H - 112) / 2)
        #img_crp = img[mrgn:(H-mrgn), mrgn:(W-mrgn), :]
        for deg in degrees:
            # Blur the image
            img_blur = gaussian_blur(img, deg)
            # Pixelate the image
            img_pix = pixelate(img, deg)
            # Save the processed images
            #mi.imsave(savepath + 'originals/' + fname, img)
            mi.imsave(savepath + 'blurred_' + str(deg) + '/' + fname, img_blur)
            mi.imsave(savepath + 'pixelated_' + str(deg) + '/' + fname, img_pix)

#num_samples = 10
#
## Load all images from folder
## To load the 10 test images inter use Datasets/test/*.jpg
## To load the 10 test images inter use Datasets/complete/*.jpg
#filelist = glob.glob('Datasets/test/*.jpg')
#X = np.array([np.array(Image.open(fname)) for fname in filelist])
#N, H, W, C = X.shape
#
## Crop all images to 112x112
#mrgn = int((H - 112) / 2)
#X_crp = X[:, mrgn:(H-mrgn), mrgn:(W-mrgn), : ]
#print(X_crp.shape)
## select random indices
#idxs = np.random.choice(N, num_samples, replace=False)
#
## Blur all images with a Gaussian filter of varying sigma values
#sigma = np.array([4,8,16])
#X_gaus_4 = np.zeros(X_crp.shape)
#X_gaus_8 = np.zeros(X_crp.shape)
#X_gaus_16 = np.zeros(X_crp.shape)
#
#for i in range(N):
#    X_gaus_4[i] = gaussian_blur(X_crp[i], sigma[0])
#    X_gaus_8[i] = gaussian_blur(X_crp[i], sigma[1])
#    X_gaus_16[i] = gaussian_blur(X_crp[i], sigma[2])
#
## Pixelate images
#X_pix_4 = np.zeros(X_crp.shape)
#
#for i in range(N):
#    X_pix_4[i] = pixelate(X_crp[i], 4)
#
##        img_name = ''
##        img_name = fname.replace('.jpg', '')
##        end = filename.rfind('_')
##        img_name = img_name[:end]
##        writer.writerow([fname, img_name])
