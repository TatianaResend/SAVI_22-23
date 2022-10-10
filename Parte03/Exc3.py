#!/usr/bin/env python3
#Solução da aula + casa

import cv2
import numpy as np


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    blackout_time = 0.5# secs
    threshold_difference = 20
    #average = [0,0,0]

    # Define rectangles (only once)
    rects = [{'name': 'r1', 'x1': 200, 'y1': 500, 'x2': 390, 'y2': 600, 'ncars': 0, 'tic_since_car_count': -500, 'average': [0,0,0]}, 
         {'name': 'r2', 'x1': 400, 'y1': 500, 'x2': 590, 'y2': 600, 'ncars': 0, 'tic_since_car_count': -500, 'average': [0,0,0]},
         {'name': 'r3', 'x1': 650, 'y1': 500, 'x2': 890, 'y2': 600, 'ncars': 0, 'tic_since_car_count': -500, 'average': [0,0,0]},
         {'name': 'r4', 'x1': 900, 'y1': 500, 'x2': 1200, 'y2': 600, 'ncars': 0, 'tic_since_car_count': -500, 'average': [0,0,0]}]

    cap = cv2.VideoCapture("./docs/traffic.mp4")
    if (cap.isOpened()== False):
        print("Error opening video stream or file")

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    is_first_time = True
    while(cap.isOpened()): # this is an infinite loop

        # Step 1: get frame
        ret, image_rgb = cap.read() # get a frame, ret will be true or false if getting succeeds
        if ret == False:
            break
        stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000
        
        # Step 2: convert to gray
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        # Step 3: get average color in rectangle
        for rect in rects:

            total = 0
            number_of_pixels = 0
            for row in range(rect['y1'],rect['y2']): # iterate all image pixels inside the rectangle
                for col in range(rect['x1'],rect['x2']):
                    number_of_pixels += 1
                    total += image_gray[row, col] # add pixel color to the total count
                
            # after computing the total we should divide to get the average
            rect['avg_color'] = int(total / number_of_pixels)

            # How to get the model average? We know that in the first frame there are no cars in the rectangles. The first measurement is the model average
            if is_first_time:
                rect['model_avg_color'] = rect['avg_color']

            # Compute the different in color and make a decision
            diff = abs(rect['avg_color'] - rect['model_avg_color'])

            if diff > 20 and (stamp - rect['tic_since_car_count']) > blackout_time:
                rect['ncars'] = rect['ncars'] + 1
                rect['tic_since_car_count'] = stamp
                
                #rows,cols,channels = image_rgb.shape
                
                # Average color (B G R)
                average_row=np.median(image_rgb[rect['y1']-10:rect['y1'],rect['x1']+50:rect['x2']-100],axis=0)
                rect['average']=np.median(average_row, axis=0)
                #print(average)      

        is_first_time = False

        # Drawing --------------------------

        for rect in rects:
            # draw rectangles
            cv2.rectangle(image_rgb, (rect['x1'],rect['y1']), (rect['x2'],rect['y2']), (0,255,0),2)

            # Add text with avg color
            text = 'avg=' + str(rect['avg_color']) + ' m=' + str(rect['model_avg_color'])
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

            # Add text with avg color
            text = 'ncars=' + str(rect['ncars']) 
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # Add text time since last car count
            text = 'Time since lcc=' + str(round(stamp - rect['tic_since_car_count'],1))  + ' secs'
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
            
            # Add text color
            text = 'color' + str(np.around(rect['average']))
            image_rgb = cv2.putText(image_rgb, text, (rect['x1'], rect['y1']-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect['average'], 1, cv2.LINE_AA)    
        
        # Add text total cars
        totalcars = rects[0]['ncars']+rects[1]['ncars']+rects[2]['ncars']+rects[3]['ncars']
        text = 'total cars=' + str(totalcars) 
        image_rgb = cv2.putText(image_rgb, text, (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

        cv2.imshow('image_rgb',image_rgb) # show the image

        if cv2.waitKey(10) == ord('q'):
            break


    # ------------------------------------------
    # Termination
    # ------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#RGB, ver qual mais a cor mais próxima do carro