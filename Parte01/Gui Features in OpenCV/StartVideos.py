#!/usr/bin/env python3

#Capture Video from Camera or Playing Video from File

import numpy as np
import cv2 as cv

#Initialization:
cap = cv.VideoCapture(0)               #From Camera
#cap = cv.VideoCapture('vtest.avi')    #From File

#Execution:
if not cap.isOpened():                 #From Camera
    print("Cannot open camera") 
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)

    if cv.waitKey(1) == ord('q'):
        break

#Termination:
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()