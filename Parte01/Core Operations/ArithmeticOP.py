#!/usr/bin/env python3

import numpy as np
import cv2 as cv

#Execution:
# -------------------------------------------------------------
#Image Addition

x = np.uint8([250])
y = np.uint8([10])
#OpenCV addition is a saturated operation while Numpy addition is a modulo 
# operation. 
print( cv.add(x,y) ) # 250+10 = 260 => 255

print( x+y )          # 250+10 = 260 % 256 = 4

# -------------------------------------------------------------
#Image Blending

img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logoWhite.png')
img2 = img2[20:270, 100:302]

#Obects dimension
print( img1.shape )
print( img2.shape )

dst = cv.addWeighted(img1,0.7,img2,0.3,0)

cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()

# -------------------------------------------------------------
#Bitwise Operations

# Load two images
img3 = cv.imread('messi5.jpg')              #(342,548,3)
img4 = cv.imread('opencv-logoWhite.png')    #(378,428,3)
img4 = img4[0:342, 0:428]

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img4.shape
roi = img3[0:rows, 0:cols]
print('img4',img4.shape)
print('roi',roi.shape)

# Now create a mask of logo and create its inverse mask also
img4gray = cv.cvtColor(img4,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img4gray, 150, 255, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)

# Now black-out the area of logo in ROI
img3_bg = cv.bitwise_and(roi,roi,mask = mask)

# Take only region of logo from logo image.
img4_fg = cv.bitwise_and(img4,img4,mask = mask_inv)

# Put logo in ROI and modify the main image
dst = cv.add(img3_bg,img4_fg)
img3[0:rows, 0:cols ] = dst

cv.imshow('res',img3)
cv.waitKey(0)
cv.destroyAllWindows()