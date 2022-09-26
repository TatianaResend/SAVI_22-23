#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Initialization:
imgGrad = cv.imread('gradient.png',0)

#Execution:
# -------------------------------------------------------------
#Simple Thresholding 
ret,thresh1 = cv.threshold(imgGrad,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(imgGrad,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(imgGrad,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(imgGrad,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(imgGrad,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [imgGrad, thresh1, thresh2, thresh3, thresh4, thresh5]

plt.figure(1)
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# -------------------------------------------------------------
#Adaptive Thresholding 
imgSod = cv.imread('sudoku.png',0)
imgSod = cv.medianBlur(imgSod,3)

ret,th1 = cv.threshold(imgSod,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(imgSod,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(imgSod,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [imgSod, th1, th2, th3]

plt.figure(2)
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()