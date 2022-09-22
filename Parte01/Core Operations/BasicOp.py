#!/usr/bin/env python3

import numpy as np
import cv2 as cv

#Initialization:
img = cv.imread('messi5.jpg')

px = img[100,100]
print( px )

#Execution:
# -------------------------------------------------------------
#Accesing and Modifying pixel values

# accessing only blue pixel (0-B,1-G,2-R)
blue = img[100,100,0]
print( blue )
img[100,100] = [255,255,255]
print( img[100,100] )

# accessing RED value
print( img.item(10,10,2) )

# modifying RED value
img.itemset((10,10,2),100)
print( img.item(10,10,2) )

# -------------------------------------------------------------
#Accessing Image Properties 
print( img.shape )

print( img.size )

print( img.dtype )

# -------------------------------------------------------------
#Image ROI
ball = img[280:340, 330:390]
img[273:333, 100:160] = ball

# -------------------------------------------------------------
#Splitting and Merging Image Channels 
b,g,r = cv.split(img)
img = cv.merge((b,g,r))
#OR
#b = img[:,:,0]
img[:,:,2] = 0      #Pixels red -> 0

#Termination
cv.imshow("Display window", img)
k = cv.waitKey(0)