#!/usr/bin/env python3
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

#Initialization:
img = cv.imread('messi5.jpg')

#Execution:

# -------------------------------------------------------------
#Scaling -> 2x
res = cv.resize(img,None,fx=2, fy=2, interpolation = cv.INTER_CUBIC)
#OR
height, width = img.shape[:2]
res2 = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)

# -------------------------------------------------------------
#Translation -> (100,50)
rows,cols,channels = img.shape
M_t = np.float32([[1,0,100],[0,1,50]])  
dst_t = cv.warpAffine(img,M_t,(cols,rows))

# -------------------------------------------------------------
#Rotation -> 90º
# cols-1 and rows-1 are the coordinate limits.
M_r = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),90,1)
dst_r = cv.warpAffine(img,M_r,(cols,rows))

# -------------------------------------------------------------
#Affine Transformation
imgDraw = cv.imread('drawing.png')
rowsDraw,colsDraw,chDraw = imgDraw.shape
pts1Draw = np.float32([[50,50],[200,50],[50,200]])
pts2Draw = np.float32([[10,100],[200,50],[100,250]])
M_at = cv.getAffineTransform(pts1Draw,pts2Draw)
dst_at = cv.warpAffine(imgDraw,M_at,(colsDraw,rowsDraw))

# -------------------------------------------------------------
#Perspective Transformation
imgSod = cv.imread('sudoku.png')
rowsSod,colsSod,chSod = imgSod.shape
pts1Sod = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2Sod = np.float32([[0,0],[300,0],[0,300],[300,300]])
M_pt = cv.getPerspectiveTransform(pts1Sod,pts2Sod)
dst_pt = cv.warpPerspective(imgSod,M_pt,(300,300))

plt.subplot(221),plt.imshow(img),plt.title('img Orignal')
plt.subplot(222),plt.imshow(res),plt.title('Scaling')
plt.subplot(223),plt.imshow(dst_t),plt.title('Translation')
plt.subplot(224),plt.imshow(dst_r),plt.title('Rotation')
plt.show()

plt.subplot(121),plt.imshow(imgDraw),plt.title('Input')
plt.subplot(122),plt.imshow(dst_at),plt.title('Output')
plt.show()

plt.subplot(121),plt.imshow(imgSod),plt.title('Input')
plt.subplot(122),plt.imshow(dst_pt),plt.title('Output')
plt.show()
#Termination:
