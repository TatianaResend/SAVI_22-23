#!/usr/bin/env python3

import csv
from copy import deepcopy
from turtle import color

import cv2
import numpy as np
from functions import Detection, Tracker

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    cap = cv2.VideoCapture("./docs/OxfordTownCentre/TownCentreXVID.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    window_name = 'image_rgb'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 500)


    # Count number of persons in the dataset
    number_or_persons = 0
    csv_reader = csv.reader(open('./docs/OxfordTownCentre/TownCentre-groundtruth.top'))
    for row in csv_reader:
        if len(row) != 12: # skip badly formatted rows
            continue

        person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row
        person_number = int(person_number) # convert to number format (integer)
        if person_number >= number_or_persons:
            number_or_persons = person_number + 1

    # Create the colors for each person
    colors = np.random.randint(0, high=255, size=(number_or_persons, 3), dtype=int)

    # Object Detection in Real-time
    person_detector = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    detection_counter = 0
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8
    # ------------------------------------------
    # Execution
    # ------------------------------------------
    frame_counter = 0
    while(cap.isOpened()): # this is an infinite loop

        # Step 1: get frame
        ret, image_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        image_gui = deepcopy(image_rgb)
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

        # ------------------------------------------
        # Detection of persons 
        # ------------------------------------------
        bboxes = person_detector.detectMultiScale(image_gray,scaleFactor=1.2,minNeighbors=4,minSize=(20,40))
            #scaleFactor – This tells how much the object’s size is reduced in each image.
            #minNeighbors – This parameter tells how many neighbours each rectangle candidate should consider.
            #minSize — This signifies the minimum possible size of an object to be detected. An object smaller than minSize would be ignored.
        
        # ------------------------------------------
        # Create Detections per haar cascade bbox
        # ------------------------------------------
        detections = []
        for bbox in bboxes:
            x1, y1, w, h = bbox 
            detection = Detection(x1,y1,w,h,image_gray,id=detection_counter)
            detection_counter += 1
            detection.draw(image_gui)
            detections.append(detection)
            #img = detection.extractSmallImage(image_gray)  #NOK
            #cv2.imshow('detection ' + str(detection.id), detection.image )   <- see the detections!

            
        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection)


        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        if frame_counter == 0:
            for detection in detections:
                tracker = Tracker(detection, id=tracker_counter)
                tracker_counter += 1
                trackers.append(tracker)
    
        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------

        # Draw trackers
        for tracker in trackers:
            tracker.draw(image_gui)


        for tracker in trackers:
            print(tracker)

        cv2.imshow(window_name,image_gui) # show the image

        if cv2.waitKey(0) == ord('q'):
            break

        frame_counter += 1

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#Notes:
# https://www.analyticsvidhya.com/blog/2022/04/object-detection-using-haar-cascade-opencv/
