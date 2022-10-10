#!/usr/bin/env python3

import cv2
import numpy as np

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Create a Jupyter-notebook and declare our trackers.
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create() 
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create() 
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    #if tracker_type == 'GOTURN':
    #    tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    cap = cv2.VideoCapture("./docs/OxfordTownCentre/TownCentreXVID.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")


    # ------------------------------------------
    # Execution
    # ------------------------------------------

    while(cap.isOpened()): # this is an infinite loop
        
        # Step 1: get frame
        ret, frame_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        if ret == False:
            break

        frame_rgb_height, frame_rgb_width = frame_rgb.shape[:2]
        # Resize the video for a more convinient view
        frame_rgb = cv2.resize(frame_rgb, [frame_rgb_width//2, frame_rgb_height//2])

        # Initialize video writer to save the results
        output = cv2.VideoWriter(f'{tracker_type}.avi', 
                                cv2.VideoWriter_fourcc(*'XVID'), 60.0, 
                                (frame_rgb_width//2, frame_rgb_height//2), True)
        if not ret:
            print('cannot read the video')
        # Select the bounding box in the first frame
        bbox = cv2.selectROI(frame_rgb, False)
        ret = tracker.init(frame_rgb, bbox)
        # Start tracking
        while True:
            ret, frame_rgb = cap.read()
            frame_rgb = cv2.resize(frame_rgb, [frame_rgb_width//2, frame_rgb_height//2])
            if not ret:
                print('something went wrong')
                break
            timer = cv2.getTickCount()
            ret, bbox = tracker.update(frame_rgb)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            if ret:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame_rgb, p1, p2, (255,0,0), 2, 1)
            else:
                cv2.putText(frame_rgb, "Tracking failure detected", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.putText(frame_rgb, tracker_type + " Tracker", (100,20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            cv2.putText(frame_rgb, "FPS : " + str(int(fps)), (100,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            cv2.imshow("Tracking", frame_rgb)
            output.write(frame_rgb)
            
        
            if cv2.waitKey(10) == ord('q'):
                break
    # ------------------------------------------
    # Termination
    # ------------------------------------------

    cap.release()
    output.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()