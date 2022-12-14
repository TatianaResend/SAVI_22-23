#!/usr/bin/env python3

#The OpenCV python library is imported
import cv2 as cv
import sys

#Initialization:
img = cv.imread(cv.samples.findFile("starry_night.jpg"))

#Execution:
#Check if the image was load correctly
if img is None: 
    sys.exit("Could not read the image.")

cv.imshow("Display window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("starry_night.png", img)

#Note - Give permission in the terminal:
# chmod ugo+rwx