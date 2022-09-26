#!/usr/bin/env python3
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# -------------------------------------------------------------
#Image Blurring (Image Smoothing) 

img = cv.imread('opencv-logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)

plt.figure(1)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

# -------------------------------------------------------------
#Image Blurring (Image Smoothing) 
#img = cv.imread('opencv-logo.png')
img = cv.imread('filter.png')

#Avering
blurAv = cv.blur(img,(5,5))

#Gaussian Blurring
blurGau = cv.GaussianBlur(img,(5,5),0)

#Median Blurring
median = cv.medianBlur(img,5)

#Bilateral Filtering
blurBil = cv.bilateralFilter(img,9,75,75)

plt.figure(2)
plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(232),plt.imshow(blurAv),plt.title('Avering')
plt.xticks([]), plt.yticks([])

plt.subplot(233),plt.imshow(blurGau),plt.title('Gaussian Blurring')
plt.xticks([]), plt.yticks([])

plt.subplot(234),plt.imshow(median),plt.title('Median Blurring')
plt.xticks([]), plt.yticks([])

plt.subplot(235),plt.imshow(blurBil),plt.title('Bilateral Filtering')
plt.xticks([]), plt.yticks([])

plt.show()