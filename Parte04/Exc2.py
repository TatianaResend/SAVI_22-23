#!/usr/bin/env python3

import cv2
import numpy as np

import csv
import copy

def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    cap = cv2.VideoCapture("./docs/OxfordTownCentre/TownCentreXVID.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

#   # Resize image
#   window_name = 'image_rgb'
#   cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#   cv2.resizeWindow(window_name, 800, 500)  

    # Count number of persons in the dataset
    number_or_persons = 0

    # Read CSV
    file = open("./docs/OxfordTownCentre/TownCentre-groundtruth.top")
    csv_reader = csv.reader(file)

    for row in csv_reader:
        if len(row) != 12: # skip badly formatted rows
            continue
        person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row
       
        # Create the colors for each person
        person_number = int(person_number) # convert to number format (integer)
        if person_number >= number_or_persons:
                number_or_persons = person_number + 1 

    colors = np.random.randint(0, high=255, size=(number_or_persons, 3), dtype=int)  
                                            #size (rows,columns)    
                                                  
    # ------------------------------------------
    # Execution
    # ------------------------------------------
    frame_contour = 0
    while(cap.isOpened()): # this is an infinite loop
        
        # Step 1: get frame
        ret, frame_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        image_gui = copy.deepcopy(frame_rgb)    # Shallow copy
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        #image_gui_height, image_gui_width = image_gui.shape[:2]
        # Resize the video for a more convinient view
        #image_gui = cv2.resize(image_gui, [image_gui_width//2, image_gui_height//2])

        # Draw ground truth bboxes
        csv_reader = csv.reader(open("./docs/OxfordTownCentre/TownCentre-groundtruth.top"))  
        for row in csv_reader:
            
            if len(row) != 12: 
                continue
            
            person_number, frame_number, _, _, _, _, _, _, body_left, body_top, body_right, body_bottom = row
            person_number = int(person_number)    # convert to number format (integer)
            frame_number = int(frame_number)
            body_left = int(float(body_left))
            body_top = int(float(body_top))
            body_right = int(float(body_right))
            body_bottom = int(float(body_bottom))

            
            if frame_contour != frame_number:   # do not draw bbox of other frames
                continue

            x1=body_left
            y1=body_top
            x2=body_right
            y2=body_bottom
            color = colors[person_number,:]

            cv2.rectangle(image_gui, (x1, y1), (x2, y2), (int(color[0]),int(color[1]),int(color[2])), 3)

            print('person ' + str(person_number) + ' frame ' + str(frame_number))

        cv2.imshow("Tracking", image_gui)
        
        if cv2.waitKey(25) == ord('q'):
            break

        frame_contour += 1

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
