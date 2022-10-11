#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd
import csv

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    file = open("./docs/OxfordTownCentre/TownCentre-groundtruth.top")

    cap = cv2.VideoCapture("./docs/OxfordTownCentre/TownCentreXVID.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    frame_contour = 0
    while(cap.isOpened()): # this is an infinite loop
        
        # Step 1: get frame
        ret, frame_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        #frame_rgb_height, frame_rgb_width = frame_rgb.shape[:2]
        # Resize the video for a more convinient view
        #frame_rgb = cv2.resize(frame_rgb, [frame_rgb_width//2, frame_rgb_height//2])

        #Read CSV
        csvreader = csv.reader(open("./docs/OxfordTownCentre/TownCentre-groundtruth.top"))
        for row in csvreader:
            
            if len(row) != 12: 
                continue
            
            personNumber, frameNumber, _, _, _, _, _, _, bodyLeft, bodyTop, bodyRight, bodyBottom = row
            personNumber = int(personNumber)
            frameNumber = int(frameNumber)
            bodyLeft = int(float(bodyLeft))
            bodyTop = int(float(bodyTop))
            bodyRight = int(float(bodyRight))
            bodyBottom = int(float(bodyBottom))

            
            if frame_contour != frameNumber:
                continue

            x1=bodyLeft
            y1=bodyTop
            x2=bodyRight
            y2=bodyBottom
            #x1=100
            #y1=100
            #x2=200
            #y2=200
            print(x1)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0,255,0), 3)

        cv2.imshow("Tracking", frame_rgb)
        
        if cv2.waitKey(10) == ord('q'):
            break
    # ------------------------------------------
    # Termination
    # ------------------------------------------

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


#Notes:
# https://academictorrents.com/details/35e83806d9362a57be736f370c821960eb2f2a01
# https://earthly.dev/blog/csv-python/
# https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/
