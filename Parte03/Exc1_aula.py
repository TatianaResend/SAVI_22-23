#!/usr/bin/env python3

import cv2
import numpy as np
from time import sleep

y1_rect=600
y2_rect=700
def main():
    cap = cv2.VideoCapture('./docs/traffic.mp4')
    
    while True:
        ret, frame1 = cap.read()  # Grab every frame of the video

        tempo = float(1 / 120)
        sleep(tempo)  # Delay between each processing

        if ret == False:
            break

       
        gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # RGB to GRAY

        cv2.rectangle(frame1, (150, y1_rect), (350, y2_rect), (0, 255, 0), 2)
        cv2.rectangle(frame1, (400, y1_rect), (600, y2_rect), (0, 255, 0), 2)
        cv2.rectangle(frame1, (700, y1_rect), (900, y2_rect), (0, 255, 0), 2)
        cv2.rectangle(frame1, (1000, y1_rect), (1200, y2_rect), (0, 255, 0), 2)    

        img1=frame1[y1_rect:y2_rect,150:350,:]
        img2=frame1[y1_rect:y2_rect,400:600,:]
        img3=frame1[y1_rect:y2_rect,700:900,:]
        img4=frame1[y1_rect:y2_rect,1000:1200,:]

        avg_color_per_row = np.average(img1, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        
        cv2.imshow('window',frame1) 
        cv2.imshow('window1',img1) 
        cv2.imshow('window2',img2) 
        cv2.imshow('window3',img3) 
        cv2.imshow('window4',img4) 

        if cv2.waitKey(1) == 27:
            break

            

if __name__ == "__main__":
    main()