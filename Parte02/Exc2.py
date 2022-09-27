#!/usr/bin/env python3

#Exercício 2 - Onde está o Wally?
import cv2
import numpy as np

def main():
    img = cv2.imread('./images/scene.jpg')
    # ------------------------------------------------------
    #Exercício 3 - Ainda o Wally?
    #img = cv2.imread('./images/school.jpg')
    #img = cv2.imread('./images/beach.jpg.')
    # ------------------------------------------------------
    template = cv2.imread('./images/wally.png')

    h,w,_ = template.shape
    
    method = cv2.TM_CCOEFF

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    color = (255,0,0) #BGR format
    cv2.rectangle(img,top_left, bottom_right, color, 2)
    
    cv2.imshow('window',img)  
    cv2.waitKey(0)

        

if __name__ == "__main__":
    main()