#!/usr/bin/env python3

import numpy as np
import cv2 as cv

#Execution:

e1 = cv.getTickCount()

# your code execution
# -------------------------------------------------------------
# Load two images
img3 = cv.imread('messi5.jpg')              #(342,548,3)
img4 = cv.imread('opencv-logoWhite.png')    #(378,428,3)
img4 = img4[0:342, 0:428]

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img4.shape
roi = img3[0:rows, 0:cols]

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
# -------------------------------------------------------------

e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()

print(time)

#Example OpenCV
img1 = cv.imread('messi5.jpg')
e1 = cv.getTickCount()
for i in range(5,49,2):
    img1 = cv.medianBlur(img1,i)
e2 = cv.getTickCount()
t = (e2 - e1)/cv.getTickFrequency()
print( t )
